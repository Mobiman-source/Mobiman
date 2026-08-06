#!/usr/bin/env python3
"""
Real-time monitoring dashboard for the Mobiman xApp.
Reads from <repo_root>/data/ran.db and serves a web dashboard at http://localhost:8050

Usage:
    python3 monitor.py
    # On local machine: ssh -L 8050:localhost:8050 user@remote
    # Then open browser: http://localhost:8050
"""

import sqlite3
import os
from collections import defaultdict

import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "ran.db")

DB = os.environ.get("RAN_DB", DEFAULT_DB_PATH)
REFRESH_MS = 2000   # dashboard refresh interval in ms
HISTORY_ROWS = 300  # how many recent rows to fetch per query

app = dash.Dash(__name__, title="RAN Monitor")

app.layout = html.Div(
    style={"fontFamily": "monospace", "backgroundColor": "#111", "color": "#eee", "padding": "16px"},
    children=[
        html.H2("RAN Real-time Monitor", style={"marginBottom": "4px"}),
        html.Div(id="db-status", style={"fontSize": "12px", "color": "#888", "marginBottom": "8px"}),
        dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),

        # SUCI filter
        html.Div(style={"marginBottom": "16px", "display": "flex", "alignItems": "center", "gap": "12px"}, children=[
            html.Span("Filter SUCI:", style={"whiteSpace": "nowrap", "fontSize": "13px"}),
            dcc.Dropdown(
                id="suci-filter",
                multi=True,
                placeholder="All UEs (select to filter...)",
                style={"flex": "1", "backgroundColor": "#222", "color": "#111", "minWidth": "300px"},
            ),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "8px"}, children=[
            dcc.Graph(id="rsrp-graph",      style={"height": "320px"}),
            dcc.Graph(id="sinr-graph",      style={"height": "320px"}),
            dcc.Graph(id="rsrq-graph",      style={"height": "320px"}),
            dcc.Graph(id="prb-dl-graph",    style={"height": "320px"}),
            dcc.Graph(id="thp-graph",       style={"height": "320px"}),
            dcc.Graph(id="rlc-delay-graph", style={"height": "320px"}),
            dcc.Graph(id="ho-count-graph",  style={"height": "320px"}),
            dcc.Graph(id="ts-delta-graph",  style={"height": "320px"}),
        ]),
    ]
)

DARK_LAYOUT = dict(
    paper_bgcolor="#1e1e1e",
    plot_bgcolor="#1e1e1e",
    font=dict(color="#ccc"),
    xaxis=dict(gridcolor="#333", title="Time (s)"),
    yaxis=dict(gridcolor="#333"),
    margin=dict(l=40, r=10, t=30, b=30),
    legend=dict(bgcolor="#2a2a2a"),
)

COLORS = ["#00b4d8", "#f4a261", "#2ec4b6", "#e63946", "#a8dadc", "#ffb703",
          "#90e0ef", "#ffd166", "#06d6a0", "#ef476f", "#caf0f8", "#ffc8a2"]


def query(sql, params=()):
    try:
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def empty_fig(title, msg="No data"):
    fig = go.Figure()
    fig.update_layout(title=dict(text=title), annotations=[dict(text=msg, showarrow=False, font=dict(size=14, color="#888"))], **DARK_LAYOUT)
    return fig


def suci_where(selected):
    """Return (extra_sql, params) for optional SUCI filtering."""
    if not selected:
        return "", ()
    placeholders = ",".join("?" * len(selected))
    return f" AND suci IN ({placeholders})", tuple(selected)


def suci_label(suci):
    return str(suci)[-6:] if suci else "unknown"


def du_color(du_id):
    if du_id is None:
        return "#888888"
    try:
        return COLORS[int(du_id) % len(COLORS)]
    except (TypeError, ValueError):
        return COLORS[hash(str(du_id)) % len(COLORS)]


# ── SUCI dropdown options ─────────────────────────────────────────────────────
@app.callback(Output("suci-filter", "options"), Input("interval", "n_intervals"))
def update_suci_options(_):
    rows = query("SELECT DISTINCT suci FROM cu_ue_meas WHERE suci IS NOT NULL ORDER BY suci")
    return [{"label": f"{suci_label(r[0])}  ({r[0]})", "value": r[0]} for r in rows]


# ── DB status ────────────────────────────────────────────────────────────────
@app.callback(Output("db-status", "children"), Input("interval", "n_intervals"))
def update_status(_):
    rows = query("SELECT COUNT(*) FROM cu_ue_meas")
    cu_cnt = rows[0][0] if rows else "?"
    rows = query("SELECT COUNT(*) FROM du_cell_kpm")
    cell_cnt = rows[0][0] if rows else "?"
    rows = query("SELECT COUNT(*) FROM du_ue_kpm")
    ue_cnt = rows[0][0] if rows else "?"
    return f"DB: {DB}  |  cu_ue_meas: {cu_cnt} rows  |  du_cell_kpm: {cell_cnt} rows  |  du_ue_kpm: {ue_cnt} rows"


# ── Serving RSRP per UE (dBm)  raw - 156 ────────────────────────────────────
@app.callback(Output("rsrp-graph", "figure"), Input("interval", "n_intervals"), Input("suci-filter", "value"))
def update_rsrp(_, selected):
    extra, params = suci_where(selected)
    rows = query(
        f"SELECT ts, suci, du_id, serv_rsrp FROM cu_ue_meas WHERE serv_rsrp IS NOT NULL{extra} ORDER BY ts DESC LIMIT ?",
        params + (HISTORY_ROWS,),
    )
    if not rows:
        return empty_fig("Serving RSRP (dBm)")
    data = defaultdict(lambda: {"x": [], "y": [], "du_id": None})
    for ts, suci, du_id, rsrp in reversed(rows):
        key = f"DU{du_id}-{suci_label(suci)}"
        data[key]["x"].append(ts)
        data[key]["y"].append(float(rsrp) - 156)
        data[key]["du_id"] = du_id
    traces = [
        go.Scatter(x=v["x"], y=v["y"], name=k, mode="lines", line=dict(color=du_color(v["du_id"])))
        for k, v in data.items()
    ]
    fig = go.Figure(traces)
    fig.update_layout(title="Serving RSRP (dBm)", yaxis_title="dBm", **DARK_LAYOUT)
    return fig


# ── Serving SINR per UE (dB)  raw × 0.5 - 23 ────────────────────────────────
@app.callback(Output("sinr-graph", "figure"), Input("interval", "n_intervals"), Input("suci-filter", "value"))
def update_sinr(_, selected):
    extra, params = suci_where(selected)
    rows = query(
        f"SELECT ts, suci, du_id, serv_sinr FROM cu_ue_meas WHERE serv_sinr IS NOT NULL{extra} ORDER BY ts DESC LIMIT ?",
        params + (HISTORY_ROWS,),
    )
    if not rows:
        return empty_fig("Serving SINR (dB)")
    data = defaultdict(lambda: {"x": [], "y": [], "du_id": None})
    for ts, suci, du_id, sinr in reversed(rows):
        key = f"DU{du_id}-{suci_label(suci)}"
        data[key]["x"].append(ts)
        data[key]["y"].append(float(sinr) * 0.5 - 23)
        data[key]["du_id"] = du_id
    traces = [
        go.Scatter(x=v["x"], y=v["y"], name=k, mode="lines", line=dict(color=du_color(v["du_id"])))
        for k, v in data.items()
    ]
    fig = go.Figure(traces)
    fig.update_layout(title="Serving SINR (dB)", yaxis_title="dB", **DARK_LAYOUT)
    return fig


# ── Serving RSRQ per UE (dB)  raw × 0.5 - 43.5 ──────────────────────────────
@app.callback(Output("rsrq-graph", "figure"), Input("interval", "n_intervals"), Input("suci-filter", "value"))
def update_rsrq(_, selected):
    extra, params = suci_where(selected)
    rows = query(
        f"SELECT ts, suci, du_id, serv_rsrq FROM cu_ue_meas WHERE serv_rsrq IS NOT NULL{extra} ORDER BY ts DESC LIMIT ?",
        params + (HISTORY_ROWS,),
    )
    if not rows:
        return empty_fig("Serving RSRQ (dB)")
    data = defaultdict(lambda: {"x": [], "y": [], "du_id": None})
    for ts, suci, du_id, rsrq in reversed(rows):
        key = f"DU{du_id}-{suci_label(suci)}"
        data[key]["x"].append(ts)
        data[key]["y"].append(float(rsrq) * 0.5 - 43.5)
        data[key]["du_id"] = du_id
    traces = [
        go.Scatter(x=v["x"], y=v["y"], name=k, mode="lines", line=dict(color=du_color(v["du_id"])))
        for k, v in data.items()
    ]
    fig = go.Figure(traces)
    fig.update_layout(title="Serving RSRQ (dB)", yaxis_title="dB", **DARK_LAYOUT)
    return fig


# ── Cell PRB usage DL (PRBs, max 273) per DU  (no SUCI filter) ───────────────
@app.callback(Output("prb-dl-graph", "figure"), Input("interval", "n_intervals"))
def update_prb(_):
    rows = query(
        "SELECT ts, du_id, prb_use_perc_dl FROM du_cell_kpm WHERE prb_use_perc_dl IS NOT NULL ORDER BY ts DESC LIMIT ?",
        (HISTORY_ROWS,),
    )
    if not rows:
        return empty_fig("Cell PRB Usage DL (PRBs)")
    data = defaultdict(lambda: {"x": [], "y": []})
    for ts, du_id, prb in reversed(rows):
        data[f"DU{du_id}"]["x"].append(ts)
        data[f"DU{du_id}"]["y"].append(float(prb))
    traces = [
        go.Scatter(x=v["x"], y=v["y"], name=k, mode="lines", line=dict(color=COLORS[i % len(COLORS)]))
        for i, (k, v) in enumerate(sorted(data.items()))
    ]
    fig = go.Figure(traces)
    fig.update_layout(title="Cell PRB Usage DL (PRBs)", yaxis_title="PRBs", yaxis_range=[-5, 278], **DARK_LAYOUT)
    return fig


# ── UE Throughput DL + UL (Mbps)  raw kbps / 1e3 ────────────────────────────
@app.callback(Output("thp-graph", "figure"), Input("interval", "n_intervals"), Input("suci-filter", "value"))
def update_thp(_, selected):
    extra, params = suci_where(selected)
    rows = query(
        f"SELECT ts, du_id, suci, thp_dl, thp_ul FROM du_ue_kpm WHERE 1=1{extra} ORDER BY ts DESC LIMIT ?",
        params + (HISTORY_ROWS,),
    )
    if not rows:
        return empty_fig("UE Throughput (Mbps)")
    data = defaultdict(lambda: {"x": [], "y": []})
    for ts, du_id, suci, thp_dl, thp_ul in reversed(rows):
        base = f"DU{du_id}-{suci_label(suci)}"
        if thp_dl is not None:
            data[f"{base}-DL"]["x"].append(ts)
            data[f"{base}-DL"]["y"].append(float(thp_dl) / 1e3)
        if thp_ul is not None:
            data[f"{base}-UL"]["x"].append(ts)
            data[f"{base}-UL"]["y"].append(float(thp_ul) / 1e3)
    traces = [
        go.Scatter(
            x=v["x"], y=v["y"], name=k, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], dash="dash" if k.endswith("-UL") else "solid"),
        )
        for i, (k, v) in enumerate(data.items())
    ]
    fig = go.Figure(traces)
    fig.update_layout(title="UE Throughput DL (solid) / UL (dash) (Mbps)", yaxis_title="Mbps", **DARK_LAYOUT)
    return fig


# ── RLC Delay DL + UL (ms)  raw μs / 1e4 ────────────────────────────────────
@app.callback(Output("rlc-delay-graph", "figure"), Input("interval", "n_intervals"), Input("suci-filter", "value"))
def update_rlc_delay(_, selected):
    extra, params = suci_where(selected)
    rows = query(
        f"SELECT ts, du_id, suci, rlc_delay_dl, rlc_delay_ul FROM du_ue_kpm WHERE 1=1{extra} ORDER BY ts DESC LIMIT ?",
        params + (HISTORY_ROWS,),
    )
    if not rows:
        return empty_fig("RLC Delay (ms)")
    data = defaultdict(lambda: {"x": [], "y": []})
    for ts, du_id, suci, delay_dl, delay_ul in reversed(rows):
        base = f"DU{du_id}-{suci_label(suci)}"
        if delay_dl is not None:
            data[f"{base}-DL"]["x"].append(ts)
            data[f"{base}-DL"]["y"].append(float(delay_dl) / 10)
        if delay_ul is not None:
            data[f"{base}-UL"]["x"].append(ts)
            data[f"{base}-UL"]["y"].append(float(delay_ul) / 10)
    traces = [
        go.Scatter(
            x=v["x"], y=v["y"], name=k, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], dash="dash" if k.endswith("-UL") else "solid"),
        )
        for i, (k, v) in enumerate(data.items())
    ]
    fig = go.Figure(traces)
    fig.update_layout(title="RLC Delay DL (solid) / UL (dash) (ms)", yaxis_title="ms", **DARK_LAYOUT)
    return fig


# ── Handover events (cumulative count per UE) ────────────────────────────────
@app.callback(Output("ho-count-graph", "figure"), Input("interval", "n_intervals"), Input("suci-filter", "value"))
def update_ho_count(_, selected):
    extra, params = suci_where(selected)
    rows = query(
        f"SELECT ts, suci, du_id FROM cu_ue_meas WHERE suci IS NOT NULL{extra} ORDER BY ts DESC LIMIT ?",
        params + (HISTORY_ROWS * 3,),
    )
    if not rows:
        return empty_fig("Handover Events (cumulative)")

    ue_timeline = defaultdict(list)
    for ts, suci, du_id in reversed(rows):
        ue_timeline[suci].append((ts, du_id))

    traces = []
    for i, (suci, timeline) in enumerate(ue_timeline.items()):
        xs, ys = [], []
        count = 0
        prev_du = None
        for ts, du_id in timeline:
            if prev_du is not None and du_id != prev_du:
                count += 1
            prev_du = du_id
            xs.append(ts)
            ys.append(count)
        traces.append(go.Scatter(x=xs, y=ys, name=suci_label(suci), mode="lines+markers",
                                 line=dict(color=COLORS[i % len(COLORS)])))

    fig = go.Figure(traces)
    fig.update_layout(title="Handover Events (cumulative count)", yaxis_title="Count", **DARK_LAYOUT)
    return fig


# ── Timestamp delta (s) per UE ───────────────────────────────────────────────
@app.callback(Output("ts-delta-graph", "figure"), Input("interval", "n_intervals"), Input("suci-filter", "value"))
def update_ts_delta(_, selected):
    extra, params = suci_where(selected)
    rows = query(
        f"SELECT ts, suci FROM cu_ue_meas WHERE suci IS NOT NULL{extra} ORDER BY ts DESC LIMIT ?",
        params + (HISTORY_ROWS,),
    )
    if not rows:
        return empty_fig("Timestamp Delta (s)")

    ue_ts = defaultdict(list)
    for ts, suci in reversed(rows):
        ue_ts[suci].append(ts)

    traces = []
    for i, (suci, tss) in enumerate(ue_ts.items()):
        if len(tss) < 2:
            continue
        xs = tss[1:]
        ys = [tss[j] - tss[j - 1] - 0.12 for j in range(1, len(tss))]
        traces.append(go.Scatter(x=xs, y=ys, name=suci_label(suci), mode="lines",
                                 line=dict(color=COLORS[i % len(COLORS)])))

    if not traces:
        return empty_fig("Timestamp Delta - 120ms (s)")
    fig = go.Figure(traces)
    fig.update_layout(title="Timestamp Delta - 120ms (jitter) per UE", yaxis_title="Δt - 0.12 (s)", **DARK_LAYOUT)
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAN real-time monitor dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Web server port (default 8050)")
    parser.add_argument("--db", type=str, default=None, help="Path to ran.db (overrides RAN_DB env var)")
    args = parser.parse_args()
    if args.db:
        DB = args.db
    print(f"[Monitor] DB: {DB}")
    print(f"[Monitor] Dashboard at http://localhost:{args.port}")
    print(f"[Monitor] SSH tunnel: ssh -L {args.port}:localhost:{args.port} <user>@<remote>")
    app.run(host="0.0.0.0", port=args.port, debug=False)
