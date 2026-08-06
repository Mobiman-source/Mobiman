/*
 *
 * Copyright 2021-2025 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#include "ue_monitor.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <queue>
#include <sqlite3.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace srsran {
namespace ue_monitor {

namespace {

// ---------------------------------------------------------------------------
// Event types queued by CU threads, consumed by the writer thread
// ---------------------------------------------------------------------------

enum class event_kind : uint8_t { ue_ids, measurement, neighbor };

struct event_t {
  event_kind kind;

  // ue_ids event (init / ngap / f1ap): snapshot of current identity state
  uint64_t ue_index      = 0;
  bool     has_ran_ue_id = false; uint64_t ran_ue_id  = 0;
  bool     has_amf_ue_id = false; uint64_t amf_ue_id  = 0;
  bool     has_cu_f1ap   = false; uint64_t cu_f1ap_id = 0;

  // measurement event
  int64_t  virtual_id  = -1; // synthetic ID returned to caller for neighbor linking
  uint64_t du_id       = 0;
  uint16_t c_rnti      = 0;
  bool     rsrp_valid  = false; uint8_t rsrp = 0;
  bool     rsrq_valid  = false; uint8_t rsrq = 0;
  bool     sinr_valid  = false; uint8_t sinr = 0;

  // neighbor event
  // virtual_id used as serving_meas_id key
  bool     pci_valid   = false; uint16_t pci = 0;
};

// ---------------------------------------------------------------------------
// In-memory UE identity state (updated synchronously, very fast)
// ---------------------------------------------------------------------------

struct ue_id_state {
  bool     has_ran_ue_id  = false; uint64_t ran_ue_id  = 0;
  bool     has_amf_ue_id  = false; uint64_t amf_ue_id  = 0;
  bool     has_cu_f1ap_id = false; uint64_t cu_f1ap_id = 0;
  uint64_t last_du_id     = 0;
};

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

struct db_state {
  // Writer thread resources
  sqlite3*      db          = nullptr;
  sqlite3_stmt* stmt_ue_ids = nullptr;
  sqlite3_stmt* stmt_insert = nullptr;
  sqlite3_stmt* stmt_neigh1 = nullptr;
  sqlite3_stmt* stmt_neigh2 = nullptr;
  FILE*         log_fp      = nullptr;

  // virtual_id → sqlite row_id mapping (writer thread only)
  std::unordered_map<int64_t, int64_t> virtual_to_sqlite;
  // virtual_id → neighbor count (writer thread only, max 2)
  std::unordered_map<int64_t, int>     neigh_count;

  // Async queue (shared between CU threads and writer thread)
  std::mutex              queue_mtx;
  std::condition_variable queue_cv;
  std::queue<event_t>     queue;
  bool                    stopping    = false;
  std::thread             writer_thr;

  // In-memory UE identity (protected by id_mtx, updated by CU threads)
  std::mutex                                id_mtx;
  std::unordered_map<uint64_t, ue_id_state> ue_ids;

  // Synthetic row ID counter returned to callers for neighbor linking
  std::atomic<int64_t> next_virtual_id{1};

  // UDP sender
  int                udp_fd   = -1;
  struct sockaddr_in udp_addr = {};

  bool initialized = false;
};

db_state& get_state()
{
  static db_state s;
  return s;
}

// ---------------------------------------------------------------------------
// SQLite helpers (writer thread only)
// ---------------------------------------------------------------------------

void exec_sql(sqlite3* db, const char* sql)
{
  char* err = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    fprintf(stderr, "[ue_monitor] SQL error: %s\n", err);
    sqlite3_free(err);
  }
}

const char* now_str(char* buf, size_t n)
{
  time_t    t  = time(nullptr);
  struct tm tm = {};
  localtime_r(&t, &tm);
  strftime(buf, n, "%Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

void bind_opt_u8(sqlite3_stmt* stmt, int col, bool valid, uint8_t val)
{
  if (valid) sqlite3_bind_int(stmt, col, static_cast<int>(val));
  else        sqlite3_bind_null(stmt, col);
}

void bind_opt_u16(sqlite3_stmt* stmt, int col, bool valid, uint16_t val)
{
  if (valid) sqlite3_bind_int(stmt, col, static_cast<int>(val));
  else        sqlite3_bind_null(stmt, col);
}

void bind_opt_u64(sqlite3_stmt* stmt, int col, bool valid, uint64_t val)
{
  if (valid) sqlite3_bind_int64(stmt, col, static_cast<int64_t>(val));
  else        sqlite3_bind_null(stmt, col);
}

constexpr int sqlite_busy_retry_count = 8;
constexpr int sqlite_busy_retry_ms    = 20;

bool step_with_retry(db_state& s, sqlite3_stmt* stmt, const char* what)
{
  int rc = SQLITE_OK;
  for (int attempt = 0; attempt <= sqlite_busy_retry_count; ++attempt) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE || rc == SQLITE_ROW) {
      return true;
    }
    if (rc != SQLITE_BUSY && rc != SQLITE_LOCKED) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sqlite_busy_retry_ms));
    sqlite3_reset(stmt);
  }

  fprintf(stderr, "[ue_monitor] %s failed: %s\n", what, sqlite3_errmsg(s.db));
  return false;
}

// ---------------------------------------------------------------------------
// Event processors (writer thread only)
// ---------------------------------------------------------------------------

void process_ue_ids(db_state& s, const event_t& e)
{
  if (!s.stmt_ue_ids) return;
  sqlite3_reset(s.stmt_ue_ids);
  sqlite3_clear_bindings(s.stmt_ue_ids);
  sqlite3_bind_int64(s.stmt_ue_ids, 1, static_cast<int64_t>(e.ue_index));
  bind_opt_u64(s.stmt_ue_ids, 2, e.has_ran_ue_id, e.ran_ue_id);
  bind_opt_u64(s.stmt_ue_ids, 3, e.has_amf_ue_id, e.amf_ue_id);
  bind_opt_u64(s.stmt_ue_ids, 4, e.has_cu_f1ap,   e.cu_f1ap_id);
  step_with_retry(s, s.stmt_ue_ids, "ue_ids write");
}

void send_udp_measurement(db_state& s, const event_t& e, int64_t row_id)
{
  if (s.udp_fd < 0) return;

  // Fetch suci_suffix for this ue_index from ue_ids
  char suci_buf[32] = "null";
  sqlite3_stmt* stmt_suci = nullptr;
  if (sqlite3_prepare_v2(s.db,
        "SELECT suci_suffix FROM ue_ids WHERE ue_index = ?",
        -1, &stmt_suci, nullptr) == SQLITE_OK) {
    sqlite3_bind_int64(stmt_suci, 1, static_cast<int64_t>(e.ue_index));
    if (sqlite3_step(stmt_suci) == SQLITE_ROW &&
        sqlite3_column_type(stmt_suci, 0) != SQLITE_NULL) {
      const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt_suci, 0));
      if (v) snprintf(suci_buf, sizeof(suci_buf), "\"%s\"", v);
    }
    sqlite3_finalize(stmt_suci);
  }

  // Format optional integer fields
  char ran_str[24], amf_str[24], f1ap_str[24];
  char rsrp_str[8], rsrq_str[8], sinr_str[8];
  if (e.has_ran_ue_id) snprintf(ran_str,  sizeof(ran_str),  "%llu", (unsigned long long)e.ran_ue_id);
  else                  snprintf(ran_str,  sizeof(ran_str),  "null");
  if (e.has_amf_ue_id) snprintf(amf_str,  sizeof(amf_str),  "%llu", (unsigned long long)e.amf_ue_id);
  else                  snprintf(amf_str,  sizeof(amf_str),  "null");
  if (e.has_cu_f1ap)   snprintf(f1ap_str, sizeof(f1ap_str), "%llu", (unsigned long long)e.cu_f1ap_id);
  else                  snprintf(f1ap_str, sizeof(f1ap_str), "null");
  if (e.rsrp_valid)    snprintf(rsrp_str, sizeof(rsrp_str), "%u",   (unsigned)e.rsrp);
  else                  snprintf(rsrp_str, sizeof(rsrp_str), "null");
  if (e.rsrq_valid)    snprintf(rsrq_str, sizeof(rsrq_str), "%u",   (unsigned)e.rsrq);
  else                  snprintf(rsrq_str, sizeof(rsrq_str), "null");
  if (e.sinr_valid)    snprintf(sinr_str, sizeof(sinr_str), "%u",   (unsigned)e.sinr);
  else                  snprintf(sinr_str, sizeof(sinr_str), "null");

  char ts[32];
  now_str(ts, sizeof(ts));

  char buf[512];
  int n = snprintf(buf, sizeof(buf),
    "{\"timestamp\":\"%s\","
    "\"id\":%lld,"
    "\"ue_index\":%llu,"
    "\"ran_ue_id\":%s,"
    "\"amf_ue_id\":%s,"
    "\"cu_ue_f1ap_id\":%s,"
    "\"suci\":%s,"
    "\"du_id\":%llu,"
    "\"c_rnti\":%u,"
    "\"serv_rsrp\":%s,"
    "\"serv_rsrq\":%s,"
    "\"serv_sinr\":%s,"
    "\"neigh1_pci\":0,\"neigh1_rsrp\":0,\"neigh1_rsrq\":0,\"neigh1_sinr\":0,"
    "\"neigh2_pci\":0,\"neigh2_rsrp\":0,\"neigh2_rsrq\":0,\"neigh2_sinr\":0}",
    ts,
    (long long)row_id,
    (unsigned long long)e.ue_index,
    ran_str, amf_str, f1ap_str,
    suci_buf,
    (unsigned long long)e.du_id,
    (unsigned)e.c_rnti,
    rsrp_str, rsrq_str, sinr_str);

  if (n > 0 && n < (int)sizeof(buf)) {
    ::sendto(s.udp_fd, buf, static_cast<size_t>(n), 0,
             reinterpret_cast<const struct sockaddr*>(&s.udp_addr),
             sizeof(s.udp_addr));
  }
}

void process_measurement(db_state& s, const event_t& e)
{
  if (!s.stmt_insert) return;
  sqlite3_reset(s.stmt_insert);
  sqlite3_clear_bindings(s.stmt_insert);
  sqlite3_bind_int64(s.stmt_insert, 1, static_cast<int64_t>(e.ue_index));
  bind_opt_u64(s.stmt_insert, 2, e.has_ran_ue_id, e.ran_ue_id);
  bind_opt_u64(s.stmt_insert, 3, e.has_amf_ue_id, e.amf_ue_id);
  bind_opt_u64(s.stmt_insert, 4, e.has_cu_f1ap,   e.cu_f1ap_id);
  sqlite3_bind_int64(s.stmt_insert, 5, static_cast<int64_t>(e.du_id));
  sqlite3_bind_int(s.stmt_insert,   6, static_cast<int>(e.c_rnti));
  bind_opt_u8(s.stmt_insert, 7, e.rsrp_valid, e.rsrp);
  bind_opt_u8(s.stmt_insert, 8, e.rsrq_valid, e.rsrq);
  bind_opt_u8(s.stmt_insert, 9, e.sinr_valid, e.sinr);
  if (!step_with_retry(s, s.stmt_insert, "measurement write")) {
    return;
  }
  int64_t real_id               = sqlite3_last_insert_rowid(s.db);
  s.virtual_to_sqlite[e.virtual_id] = real_id;
  s.neigh_count[e.virtual_id]       = 0;

  send_udp_measurement(s, e, real_id);
}

void send_udp_full_row(db_state& s, int64_t real_id)
{
  if (s.udp_fd < 0) return;

  static const char* sql =
    "SELECT r.timestamp, r.id, r.ue_index, r.ran_ue_id, r.amf_ue_id, r.cu_ue_f1ap_id,"
    " r.suci_suffix, r.du_id, r.c_rnti,"
    " r.rsrp, r.rsrq, r.sinr,"
    " r.neiborcell_1_pci, r.n1_rsrp, r.n1_rsrq, r.n1_sinr,"
    " r.neiborcell_2_pci, r.n2_rsrp, r.n2_rsrq, r.n2_sinr"
    " FROM ue_records r WHERE r.id = ?";

  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(s.db, sql, -1, &st, nullptr) != SQLITE_OK) return;
  sqlite3_bind_int64(st, 1, real_id);
  if (sqlite3_step(st) != SQLITE_ROW) { sqlite3_finalize(st); return; }

  // Helper lambda: returns int value or 0 if NULL
  auto col_int = [&](int col) -> long long {
    return sqlite3_column_type(st, col) == SQLITE_NULL ? 0LL : sqlite3_column_int64(st, col);
  };
  auto col_str = [&](int col, char* out, int sz) {
    if (sqlite3_column_type(st, col) == SQLITE_NULL) {
      snprintf(out, sz, "null");
    } else {
      snprintf(out, sz, "\"%s\"", sqlite3_column_text(st, col));
    }
  };

  const char* ts    = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
  char suci_buf[32]; col_str(6, suci_buf, sizeof(suci_buf));

  char buf[512];
  int n = snprintf(buf, sizeof(buf),
    "{\"timestamp\":\"%s\","
    "\"id\":%lld,"
    "\"ue_index\":%lld,"
    "\"ran_ue_id\":%lld,"
    "\"amf_ue_id\":%lld,"
    "\"cu_ue_f1ap_id\":%lld,"
    "\"suci\":%s,"
    "\"du_id\":%lld,"
    "\"c_rnti\":%lld,"
    "\"serv_rsrp\":%lld,\"serv_rsrq\":%lld,\"serv_sinr\":%lld,"
    "\"neigh1_pci\":%lld,\"neigh1_rsrp\":%lld,\"neigh1_rsrq\":%lld,\"neigh1_sinr\":%lld,"
    "\"neigh2_pci\":%lld,\"neigh2_rsrp\":%lld,\"neigh2_rsrq\":%lld,\"neigh2_sinr\":%lld}",
    ts ? ts : "",
    col_int(1), col_int(2), col_int(3), col_int(4), col_int(5),
    suci_buf,
    col_int(7), col_int(8),
    col_int(9), col_int(10), col_int(11),
    col_int(12), col_int(13), col_int(14), col_int(15),
    col_int(16), col_int(17), col_int(18), col_int(19));

  sqlite3_finalize(st);

  if (n > 0 && n < (int)sizeof(buf)) {
    ::sendto(s.udp_fd, buf, static_cast<size_t>(n), 0,
             reinterpret_cast<const struct sockaddr*>(&s.udp_addr),
             sizeof(s.udp_addr));
  }
}

void process_neighbor(db_state& s, const event_t& e)
{
  auto cnt_it = s.neigh_count.find(e.virtual_id);
  if (cnt_it == s.neigh_count.end()) return;
  int n = cnt_it->second;
  if (n >= 2) return;

  auto id_it = s.virtual_to_sqlite.find(e.virtual_id);
  if (id_it == s.virtual_to_sqlite.end()) return;
  int64_t real_id = id_it->second;

  sqlite3_stmt* stmt = (n == 0) ? s.stmt_neigh1 : s.stmt_neigh2;
  if (!stmt) return;

  sqlite3_reset(stmt);
  sqlite3_clear_bindings(stmt);
  bind_opt_u16(stmt, 1, e.pci_valid,  e.pci);
  bind_opt_u8(stmt,  2, e.rsrp_valid, e.rsrp);
  bind_opt_u8(stmt,  3, e.rsrq_valid, e.rsrq);
  bind_opt_u8(stmt,  4, e.sinr_valid, e.sinr);
  sqlite3_bind_int64(stmt, 5, real_id);
  if (!step_with_retry(s, stmt, "neighbor write")) {
    return;
  }
  cnt_it->second++;
  send_udp_full_row(s, real_id);
  // Clean up map entries after second neighbor to avoid unbounded growth.
  if (cnt_it->second >= 2) {
    s.neigh_count.erase(cnt_it);
    s.virtual_to_sqlite.erase(id_it);
  }
}

// ---------------------------------------------------------------------------
// Writer thread loop
// ---------------------------------------------------------------------------

void writer_loop(db_state* sp)
{
  db_state& s = *sp;
  while (true) {
    std::unique_lock<std::mutex> lk(s.queue_mtx);
    s.queue_cv.wait(lk, [&s] { return !s.queue.empty() || s.stopping; });

    // Drain the queue.
    while (!s.queue.empty()) {
      event_t e = std::move(s.queue.front());
      s.queue.pop();
      lk.unlock();

      switch (e.kind) {
        case event_kind::ue_ids:      process_ue_ids(s, e);      break;
        case event_kind::measurement: process_measurement(s, e); break;
        case event_kind::neighbor:    process_neighbor(s, e);    break;
      }

      lk.lock();
    }

    if (s.stopping) break;
  }
}

// ---------------------------------------------------------------------------
// Enqueue helpers (called from CU threads, lock-free hot path)
// ---------------------------------------------------------------------------

void enqueue(db_state& s, event_t e)
{
  {
    std::lock_guard<std::mutex> lk(s.queue_mtx);
    s.queue.push(std::move(e));
  }
  s.queue_cv.notify_one();
}

event_t make_ue_ids_event(uint64_t ue_index, const ue_id_state& st)
{
  event_t e;
  e.kind          = event_kind::ue_ids;
  e.ue_index      = ue_index;
  e.has_ran_ue_id = st.has_ran_ue_id; e.ran_ue_id  = st.ran_ue_id;
  e.has_amf_ue_id = st.has_amf_ue_id; e.amf_ue_id  = st.amf_ue_id;
  e.has_cu_f1ap   = st.has_cu_f1ap_id; e.cu_f1ap_id = st.cu_f1ap_id;
  return e;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void init(const char* db_path)
{
  db_state& s = get_state();
  // Use a scoped lock only for the initialization guard check.
  {
    std::lock_guard<std::mutex> lk(s.queue_mtx);
    if (s.initialized) return;
    s.initialized = true; // set early to prevent double-init races
  }

  remove(db_path);
  remove((std::string(db_path) + "-wal").c_str());
  remove((std::string(db_path) + "-shm").c_str());

  if (sqlite3_open(db_path, &s.db) != SQLITE_OK) {
    fprintf(stderr, "[ue_monitor] Cannot open database '%s': %s\n", db_path, sqlite3_errmsg(s.db));
    sqlite3_close(s.db);
    s.db = nullptr;
    return;
  }

  exec_sql(s.db, "PRAGMA journal_mode=WAL;");
  exec_sql(s.db, "PRAGMA synchronous=NORMAL;");
  exec_sql(s.db, "PRAGMA busy_timeout=3000;");

  exec_sql(s.db,
           "CREATE TABLE IF NOT EXISTS ue_ids ("
           "  ue_index       INTEGER PRIMARY KEY,"
           "  ran_ue_id      INTEGER,"
           "  amf_ue_id      INTEGER,"
           "  cu_ue_f1ap_id  INTEGER,"
           "  suci_suffix    TEXT,"
           "  first_seen     TEXT NOT NULL,"
           "  last_updated   TEXT NOT NULL"
           ");");

  sqlite3_prepare_v2(s.db,
                     "INSERT INTO ue_ids (ue_index, ran_ue_id, amf_ue_id, cu_ue_f1ap_id, suci_suffix, first_seen, last_updated)"
                     " VALUES (?,?,?,?,"
                     "  (SELECT h.suci_suffix FROM ue_ids h"
                     "   WHERE h.amf_ue_id = ?3 AND h.suci_suffix IS NOT NULL LIMIT 1),"
                     "  datetime('now','localtime'),datetime('now','localtime'))"
                     " ON CONFLICT(ue_index) DO UPDATE SET"
                     "   ran_ue_id     = COALESCE(excluded.ran_ue_id,     ran_ue_id),"
                     "   amf_ue_id     = COALESCE(excluded.amf_ue_id,     amf_ue_id),"
                     "   cu_ue_f1ap_id = COALESCE(excluded.cu_ue_f1ap_id, cu_ue_f1ap_id),"
                     "   suci_suffix   = COALESCE(suci_suffix,"
                     "     (SELECT h.suci_suffix FROM ue_ids h"
                     "      WHERE h.amf_ue_id = COALESCE(excluded.amf_ue_id, amf_ue_id)"
                     "        AND h.suci_suffix IS NOT NULL LIMIT 1)),"
                     "   last_updated  = excluded.last_updated;",
                     -1, &s.stmt_ue_ids, nullptr);

  exec_sql(s.db,
           "CREATE TABLE IF NOT EXISTS ue_records ("
           "  id                INTEGER PRIMARY KEY AUTOINCREMENT,"
           "  ue_index          INTEGER NOT NULL,"
           "  ran_ue_id         INTEGER,"
           "  amf_ue_id         INTEGER,"
           "  cu_ue_f1ap_id     INTEGER,"
           "  suci_suffix       TEXT,"
           "  du_id             INTEGER NOT NULL,"
           "  c_rnti            INTEGER NOT NULL,"
           "  rsrp              INTEGER,"
           "  rsrq              INTEGER,"
           "  sinr              INTEGER,"
           "  neiborcell_1_pci  INTEGER,"
           "  n1_rsrp           INTEGER,"
           "  n1_rsrq           INTEGER,"
           "  n1_sinr           INTEGER,"
           "  neiborcell_2_pci  INTEGER,"
           "  n2_rsrp           INTEGER,"
           "  n2_rsrq           INTEGER,"
           "  n2_sinr           INTEGER,"
           "  timestamp         TEXT NOT NULL"
           ");");

  sqlite3_prepare_v2(s.db,
                     "INSERT INTO ue_records"
                     " (ue_index, ran_ue_id, amf_ue_id, cu_ue_f1ap_id, suci_suffix, du_id, c_rnti,"
                     "  rsrp, rsrq, sinr,"
                     "  neiborcell_1_pci, n1_rsrp, n1_rsrq, n1_sinr,"
                     "  neiborcell_2_pci, n2_rsrp, n2_rsrq, n2_sinr,"
                     "  timestamp)"
                     " VALUES (?1,?2,?3,?4,"
                     "  (SELECT suci_suffix FROM ue_ids"
                     "   WHERE suci_suffix IS NOT NULL"
                     "     AND (ue_index = ?1 OR (amf_ue_id IS NOT NULL AND amf_ue_id = ?3))"
                     "   LIMIT 1),"
                     "  ?5,?6, ?7,?8,?9,"
                     "  NULL,NULL,NULL,NULL, NULL,NULL,NULL,NULL,"
                     "  datetime('now','localtime'));",
                     -1, &s.stmt_insert, nullptr);

  sqlite3_prepare_v2(s.db,
                     "UPDATE ue_records SET neiborcell_1_pci=?, n1_rsrp=?, n1_rsrq=?, n1_sinr=? WHERE id=?;",
                     -1, &s.stmt_neigh1, nullptr);

  sqlite3_prepare_v2(s.db,
                     "UPDATE ue_records SET neiborcell_2_pci=?, n2_rsrp=?, n2_rsrq=?, n2_sinr=? WHERE id=?;",
                     -1, &s.stmt_neigh2, nullptr);

  std::string log_path = db_path;
  auto        pos      = log_path.rfind(".db");
  if (pos != std::string::npos && pos == log_path.size() - 3) {
    log_path.replace(pos, 3, ".log");
  } else {
    log_path += ".log";
  }
  s.log_fp = fopen(log_path.c_str(), "w");
  if (!s.log_fp) {
    fprintf(stderr, "[ue_monitor] Cannot open log file '%s'\n", log_path.c_str());
  }

  // Setup UDP sender.
  s.udp_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (s.udp_fd >= 0) {
    memset(&s.udp_addr, 0, sizeof(s.udp_addr));
    s.udp_addr.sin_family = AF_INET;
    s.udp_addr.sin_port   = htons(5005);
    ::inet_pton(AF_INET, "35.9.28.119", &s.udp_addr.sin_addr);
    fprintf(stdout, "[ue_monitor] UDP → 35.9.28.119:5005\n");
  } else {
    fprintf(stderr, "[ue_monitor] Failed to create UDP socket\n");
  }

  // Start async writer thread.
  s.stopping    = false;
  s.writer_thr  = std::thread(writer_loop, &s);

  fprintf(stdout, "[ue_monitor] DB='%s'  log='%s'\n", db_path, log_path.c_str());
}

void close()
{
  db_state& s = get_state();
  {
    std::lock_guard<std::mutex> lk(s.queue_mtx);
    if (!s.initialized) return;
    s.stopping = true;
  }
  s.queue_cv.notify_one();
  if (s.writer_thr.joinable()) {
    s.writer_thr.join();
  }

  if (s.stmt_ue_ids) { sqlite3_finalize(s.stmt_ue_ids); s.stmt_ue_ids = nullptr; }
  if (s.stmt_insert) { sqlite3_finalize(s.stmt_insert); s.stmt_insert = nullptr; }
  if (s.stmt_neigh1) { sqlite3_finalize(s.stmt_neigh1); s.stmt_neigh1 = nullptr; }
  if (s.stmt_neigh2) { sqlite3_finalize(s.stmt_neigh2); s.stmt_neigh2 = nullptr; }
  if (s.db)          { sqlite3_close(s.db);              s.db          = nullptr; }
  if (s.log_fp)      { fclose(s.log_fp);                 s.log_fp      = nullptr; }
  if (s.udp_fd >= 0) { ::close(s.udp_fd);               s.udp_fd      = -1;      }

  s.initialized = false;
}

void record_initial_ue(uint64_t ue_index, uint64_t ran_ue_id)
{
  db_state& s = get_state();
  if (!s.initialized) init();

  ue_id_state st;
  {
    std::lock_guard<std::mutex> lk(s.id_mtx);
    auto& entry          = s.ue_ids[ue_index];
    entry.has_ran_ue_id  = true;
    entry.ran_ue_id      = ran_ue_id;
    entry.has_amf_ue_id  = false;
    // do not reset has_cu_f1ap_id — F1AP sets it before this is called
    st = entry;
  }

  enqueue(s, make_ue_ids_event(ue_index, st));

  if (s.log_fp) {
    char ts[32];
    fprintf(s.log_fp, "[%s] INIT  ue=%lu ran_ue_id=%lu\n",
            now_str(ts, sizeof(ts)), (unsigned long)ue_index, (unsigned long)ran_ue_id);
    fflush(s.log_fp);
  }
}

void record_ngap_ids(uint64_t ue_index, uint64_t ran_ue_id, uint64_t amf_ue_id)
{
  db_state& s = get_state();
  if (!s.initialized) init();

  ue_id_state st;
  {
    std::lock_guard<std::mutex> lk(s.id_mtx);
    auto& entry         = s.ue_ids[ue_index];
    entry.has_ran_ue_id = true;  entry.ran_ue_id = ran_ue_id;
    entry.has_amf_ue_id = true;  entry.amf_ue_id = amf_ue_id;
    st = entry;
  }

  enqueue(s, make_ue_ids_event(ue_index, st));

  if (s.log_fp) {
    char ts[32];
    fprintf(s.log_fp, "[%s] NGAP  ue=%lu ran_ue_id=%lu amf_ue_id=%lu\n",
            now_str(ts, sizeof(ts)), (unsigned long)ue_index,
            (unsigned long)ran_ue_id, (unsigned long)amf_ue_id);
    fflush(s.log_fp);
  }
}

void record_cu_f1ap_id(uint64_t ue_index, uint64_t cu_ue_f1ap_id)
{
  db_state& s = get_state();
  if (!s.initialized) init();

  ue_id_state st;
  {
    std::lock_guard<std::mutex> lk(s.id_mtx);
    auto& entry           = s.ue_ids[ue_index];
    entry.has_cu_f1ap_id  = true;
    entry.cu_f1ap_id      = cu_ue_f1ap_id;
    st = entry;
  }

  enqueue(s, make_ue_ids_event(ue_index, st));

  if (s.log_fp) {
    char ts[32];
    fprintf(s.log_fp, "[%s] F1AP  ue=%lu cu_ue_f1ap_id=%lu\n",
            now_str(ts, sizeof(ts)), (unsigned long)ue_index, (unsigned long)cu_ue_f1ap_id);
    fflush(s.log_fp);
  }
}

void record_ue_index_update(uint64_t old_ue_index, uint64_t new_ue_index)
{
  if (old_ue_index == new_ue_index) return;
  db_state& s = get_state();
  if (!s.initialized) init();

  ue_id_state st;
  {
    std::lock_guard<std::mutex> lk(s.id_mtx);
    auto it = s.ue_ids.find(old_ue_index);
    if (it != s.ue_ids.end()) {
      st = it->second;
    }
    // Copy identity state to new ue_index (ran_ue_id/amf_ue_id stay the same, ue_index changes).
    // Preserve cu_f1ap_id already set for new_ue_index by find_or_create_f1ap_ue_context().
    auto new_it = s.ue_ids.find(new_ue_index);
    if (new_it != s.ue_ids.end() && new_it->second.has_cu_f1ap_id) {
      st.has_cu_f1ap_id = true;
      st.cu_f1ap_id     = new_it->second.cu_f1ap_id;
    }
    s.ue_ids[new_ue_index] = st;
  }

  enqueue(s, make_ue_ids_event(new_ue_index, st));

  if (s.log_fp) {
    char ts[32];
    fprintf(s.log_fp, "[%s] HO    ue=%lu -> ue=%lu (ran_ue_id=%lu amf_ue_id=%lu)\n",
            now_str(ts, sizeof(ts)),
            (unsigned long)old_ue_index, (unsigned long)new_ue_index,
            (unsigned long)st.ran_ue_id, (unsigned long)st.amf_ue_id);
    fflush(s.log_fp);
  }
}

int64_t record_measurement(uint64_t ue_index,
                           uint64_t du_id,
                           uint16_t c_rnti,
                           bool     rsrp_valid,
                           uint8_t  rsrp,
                           bool     rsrq_valid,
                           uint8_t  rsrq,
                           bool     sinr_valid,
                           uint8_t  sinr)
{
  db_state& s = get_state();
  if (!s.initialized) init();

  // Snapshot current identity state. If du_id changed, suppress stale cu_f1ap_id
  // for this event only — do not modify in-memory state so record_cu_f1ap_id()
  // can update it independently.
  ue_id_state st;
  bool du_changed = false;
  {
    std::lock_guard<std::mutex> lk(s.id_mtx);
    auto it = s.ue_ids.find(ue_index);
    if (it != s.ue_ids.end()) {
      if (it->second.last_du_id != 0 && it->second.last_du_id != du_id) {
        du_changed = true;
      }
      it->second.last_du_id = du_id;
      st = it->second;
    }
  }

  int64_t vid = s.next_virtual_id.fetch_add(1, std::memory_order_relaxed);

  event_t e;
  e.kind          = event_kind::measurement;
  e.virtual_id    = vid;
  e.ue_index      = ue_index;
  e.has_ran_ue_id = st.has_ran_ue_id; e.ran_ue_id  = st.ran_ue_id;
  e.has_amf_ue_id = st.has_amf_ue_id; e.amf_ue_id  = st.amf_ue_id;
  e.has_cu_f1ap   = du_changed ? false : st.has_cu_f1ap_id;
  e.cu_f1ap_id    = du_changed ? 0     : st.cu_f1ap_id;
  e.du_id         = du_id;
  e.c_rnti        = c_rnti;
  e.rsrp_valid    = rsrp_valid; e.rsrp = rsrp;
  e.rsrq_valid    = rsrq_valid; e.rsrq = rsrq;
  e.sinr_valid    = sinr_valid; e.sinr = sinr;
  enqueue(s, e);

  if (s.log_fp) {
    char ts[32];
    fprintf(s.log_fp, "[%s] SERV  ue=%lu du_id=%lu c_rnti=%u rsrp=%s rsrq=%s sinr=%s vid=%lld\n",
            now_str(ts, sizeof(ts)),
            (unsigned long)ue_index, (unsigned long)du_id, (unsigned)c_rnti,
            rsrp_valid ? std::to_string(rsrp).c_str() : "-",
            rsrq_valid ? std::to_string(rsrq).c_str() : "-",
            sinr_valid ? std::to_string(sinr).c_str() : "-",
            (long long)vid);
    fflush(s.log_fp);
  }
  return vid;
}

void record_neighbor_measurement(int64_t  serving_meas_id,
                                 bool     pci_valid,
                                 uint16_t pci,
                                 bool     rsrp_valid,
                                 uint8_t  rsrp,
                                 bool     rsrq_valid,
                                 uint8_t  rsrq,
                                 bool     sinr_valid,
                                 uint8_t  sinr)
{
  if (serving_meas_id < 0) return;
  db_state& s = get_state();
  if (!s.initialized) init();

  event_t e;
  e.kind       = event_kind::neighbor;
  e.virtual_id = serving_meas_id;
  e.pci_valid  = pci_valid;  e.pci  = pci;
  e.rsrp_valid = rsrp_valid; e.rsrp = rsrp;
  e.rsrq_valid = rsrq_valid; e.rsrq = rsrq;
  e.sinr_valid = sinr_valid; e.sinr = sinr;
  enqueue(s, e);

  if (s.log_fp) {
    char ts[32];
    fprintf(s.log_fp, "[%s] NEIGH vid=%lld pci=%s rsrp=%s rsrq=%s sinr=%s\n",
            now_str(ts, sizeof(ts)),
            (long long)serving_meas_id,
            pci_valid  ? std::to_string(pci).c_str()  : "-",
            rsrp_valid ? std::to_string(rsrp).c_str() : "-",
            rsrq_valid ? std::to_string(rsrq).c_str() : "-",
            sinr_valid ? std::to_string(sinr).c_str() : "-");
    fflush(s.log_fp);
  }
}

} // namespace ue_monitor
} // namespace srsran
