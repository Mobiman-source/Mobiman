#!/usr/bin/env python3
"""
Ping monitor: records RTT per ping + cumulative packet loss count in real-time.
Usage: python3 ping_monitor.py [--hosts 10.45.0.3 10.45.0.10] [--interval 1] [--output ping_log.csv]
"""

import subprocess
import re
import csv
import time
import argparse
import os
from datetime import datetime
from collections import defaultdict

# Always resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "..", "data", f"ping_mobility_no_HO{_ts}.csv")


def ping_once(host: str) -> dict:
    """Ping host once, return rtt_ms (None if timeout)."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", host],
            capture_output=True, text=True, timeout=5
        )
        rtt_match = re.search(r"time=(\d+\.?\d*)\s*ms", result.stdout)
        if not rtt_match and result.returncode != 0:
            # Uncomment below to debug:
            # print(f"[DBG] stdout={result.stdout!r} stderr={result.stderr!r}")
            pass
        rtt_ms = float(rtt_match.group(1)) if rtt_match else None
        return rtt_ms
    except subprocess.TimeoutExpired:
        return None


def start_iperf(servers: list) -> list:
    """Start UL and DL iperf traffic for each server in background, return process list."""
    # Per-server bandwidth config: (ul_bw, dl_bw)
    bw_config = {
        "10.45.0.3": ("10M", "60M"),
        "10.45.0.5": ("10M", "60M"),
    }
    default_bw = ("3M", "20M")

    procs = []
    for server in servers:
        ul_bw, dl_bw = bw_config.get(server, default_bw)

        ul = subprocess.Popen(
            ["iperf3", "-c", server, "-t", "86400", "-b", ul_bw, "-u", "-R"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append(("UL", ul))
        print(f"[iperf] UL started (iperf3 -c {server} -t 86400 -b {ul_bw} -u -R, pid={ul.pid})")

        dl = subprocess.Popen(
            ["iperf", "-u", "-c", server, "-t", "86400", "-i", "1", "-b", dl_bw],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append(("DL", dl))
        print(f"[iperf] DL started (iperf -u -c {server} -t 86400 -b {dl_bw}, pid={dl.pid})")
    return procs


def main():
    parser = argparse.ArgumentParser(description="Ping monitor with real-time RTT and cumulative loss")
    # parser.add_argument("--hosts", nargs="+", default=["10.45.0.4", "10.45.0.5"])
    parser.add_argument("--hosts", nargs="+", default=["10.45.0.26"])
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--iperf-server", nargs="+", default=["10.45.0.3","10.45.0.4", "10.45.0.5"],
                        help="iperf server IP(s) for traffic generation (default: 10.45.0.3 10.45.0.4 10.45.0.5)")
    parser.add_argument("--no-iperf", action="store_true",
                        help="Disable iperf traffic generation")
    args = parser.parse_args()

    iperf_procs = []
    if not args.no_iperf:
        iperf_procs = start_iperf(args.iperf_server)

    # Cumulative counters per host
    sent     = defaultdict(int)
    lost     = defaultdict(int)
    rtt_sum  = defaultdict(float)
    rtt_min  = defaultdict(lambda: float("inf"))
    rtt_max  = defaultdict(float)

    print(f"Logging to: {args.output}")
    print(f"Hosts: {args.hosts}  |  Interval: {args.interval}s  |  Ctrl+C to stop\n")

    header = f"{'Timestamp':<25} {'Host':<16} {'RTT(ms)':>9} {'Loss':>6} {'Sent':>6} {'Avg RTT':>9} {'Min':>7} {'Max':>7}"
    print(header)
    print("-" * len(header))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "host", "rtt_ms", "loss_total", "sent_total",
                          "avg_rtt_ms", "min_rtt_ms", "max_rtt_ms"])

        try:
            while True:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                for host in args.hosts:
                    rtt = ping_once(host)
                    sent[host] += 1

                    if rtt is None:
                        lost[host] += 1
                        rtt_str = "timeout"
                    else:
                        rtt_sum[host] += rtt
                        rtt_min[host] = min(rtt_min[host], rtt)
                        rtt_max[host] = max(rtt_max[host], rtt)
                        rtt_str = f"{rtt:.1f}"

                    received = sent[host] - lost[host]
                    avg = rtt_sum[host] / received if received > 0 else 0
                    mn  = rtt_min[host] if received > 0 else 0
                    mx  = rtt_max[host] if received > 0 else 0

                    print(f"{ts:<25} {host:<16} {rtt_str:>9} {lost[host]:>6} {sent[host]:>6}"
                          f" {avg:>9.1f} {mn:>7.1f} {mx:>7.1f}")

                    writer.writerow([ts, host,
                                     rtt if rtt is not None else "",
                                     lost[host], sent[host],
                                     round(avg, 2), round(mn, 2), round(mx, 2)])
                f.flush()
                time.sleep(args.interval)

        except KeyboardInterrupt:
            for label, p in iperf_procs:
                p.terminate()
                print(f"[iperf] {label} stopped (pid={p.pid})")
            print("\n--- Summary ---")
            for host in args.hosts:
                received = sent[host] - lost[host]
                avg = rtt_sum[host] / received if received > 0 else 0
                print(f"{host}: sent={sent[host]} lost={lost[host]} "
                      f"loss%={100*lost[host]//sent[host] if sent[host] else 0}% "
                      f"avg={avg:.1f}ms min={rtt_min[host]:.1f}ms max={rtt_max[host]:.1f}ms")


if __name__ == "__main__":
    main()
