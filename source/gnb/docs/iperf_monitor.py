#!/usr/bin/env python3
"""
iperf_monitor.py: generates UL/DL iperf traffic and measures RTT to the same target.

- Background iperf traffic to cell servers (10.45.0.3/4/5)
- UL + DL iperf traffic to target host (default: 10.45.0.26)
- RTT measured via ping to the same target → reflects loaded latency

Usage:
  python3 iperf_monitor.py
  python3 iperf_monitor.py --target 10.45.0.26 --target-ul-bw 5M --target-dl-bw 20M
  python3 iperf_monitor.py --no-cell-iperf   # only target traffic + RTT
  python3 iperf_monitor.py --no-target-iperf # unloaded RTT (baseline)

Requires iperf3 server running on all target hosts.
"""

import subprocess
import re
import csv
import time
import argparse
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "..", "data", f"iperf_monitor_{_ts}.csv")


def ping_once(host: str) -> float | None:
    """Ping host once, return RTT in ms or None on timeout."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", host],
            capture_output=True, text=True, timeout=5
        )
        m = re.search(r"time=(\d+\.?\d*)\s*ms", result.stdout)
        return float(m.group(1)) if m else None
    except subprocess.TimeoutExpired:
        return None


def start_iperf_cell(servers: list) -> list:
    """Start background UL/DL iperf traffic to cell servers.
    UL: iperf3 -R (UE→network), DL: iperf (network→UE)
    """
    bw_config = {
        "10.45.0.16": ("10M", "60M"),
        "10.45.0.4": ("1M", "5M"),
        "10.45.0.5": ("10M", "60M"),
    }
    default_bw = ("3M", "20M")

    procs = []
    for server in servers:
        ul_bw, dl_bw = bw_config.get(server, default_bw)

        # UL: UE → network, iperf3 -R (server sends to client)
        ul = subprocess.Popen(
            ["iperf3", "-c", server, "-t", "86400", "-b", ul_bw, "-u", "-R"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append(("UL", server, ul))
        print(f"[iperf3] UL from {server} started  bw={ul_bw}  pid={ul.pid}")

        # DL: network → UE, iperf (client sends to server)
        dl = subprocess.Popen(
            ["iperf", "-u", "-c", server, "-t", "86400", "-i", "1", "-b", dl_bw],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append(("DL", server, dl))
        print(f"[iperf ] DL to   {server} started  bw={dl_bw}  pid={dl.pid}")

    return procs


def start_iperf_target(target: str, ul_bw: str, dl_bw: str) -> list:
    """Start UL/DL iperf traffic to the RTT measurement target.
    UL: iperf3 -R (UE→network), DL: iperf (network→UE)
    """
    procs = []

    # UL: UE → network, iperf3 -R (server sends to client)
    ul = subprocess.Popen(
        ["iperf3", "-c", target, "-t", "86400", "-b", ul_bw, "-u", "-R"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    procs.append(("UL", target, ul))
    print(f"[iperf3] UL from {target} started  bw={ul_bw}  pid={ul.pid}")

    # DL: network → UE, iperf (client sends to server)
    dl = subprocess.Popen(
        ["iperf", "-u", "-c", target, "-t", "86400", "-i", "1", "-b", dl_bw],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    procs.append(("DL", target, dl))
    print(f"[iperf ] DL to   {target} started  bw={dl_bw}  pid={dl.pid}")

    return procs


def main():
    parser = argparse.ArgumentParser(
        description="iperf traffic generator + RTT monitor (loaded latency)"
    )
    parser.add_argument("--target", default="10.45.0.26",
                        help="Target host for iperf traffic AND RTT measurement (default: 10.45.0.26)")
    parser.add_argument("--target-ul-bw", default="1M",
                        help="UL bandwidth to target (default: 1M)")
    parser.add_argument("--target-dl-bw", default="10M",
                        help="DL bandwidth from target (default: 10M)")

    parser.add_argument("--cell-servers", nargs="+",
                        default=["10.45.0.16", "10.45.0.4", "10.45.0.5"],
                        help="Cell servers for background traffic")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="Ping interval in seconds (default: 0.1)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-cell-iperf", action="store_true",
                        help="Disable background iperf to cell servers")
    parser.add_argument("--no-target-iperf", action="store_true",
                        help="Disable iperf to target (measures unloaded/baseline RTT)")
    args = parser.parse_args()

    procs = []
    if not args.no_cell_iperf:
        procs += start_iperf_cell(args.cell_servers)
    if not args.no_target_iperf:
        procs += start_iperf_target(args.target, args.target_ul_bw, args.target_dl_bw)

    # RTT state
    sent    = 0
    lost    = 0
    rtt_sum = 0.0
    rtt_min = float("inf")
    rtt_max = 0.0

    mode = "loaded" if not args.no_target_iperf else "unloaded (baseline)"
    print(f"\nLogging to : {args.output}")
    print(f"Target     : {args.target}  [{mode}]")
    print(f"Interval   : {args.interval}s  |  Ctrl+C to stop\n")

    header = (f"{'Timestamp':<25} {'RTT(ms)':>9} {'Loss':>6} {'Sent':>6}"
              f" {'Avg RTT':>9} {'Min':>7} {'Max':>7}")
    print(header)
    print("-" * len(header))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "host", "rtt_ms", "loss_total", "sent_total",
                         "avg_rtt_ms", "min_rtt_ms", "max_rtt_ms"])

        try:
            while True:
                ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                rtt = ping_once(args.target)
                sent += 1

                if rtt is None:
                    lost    += 1
                    rtt_str  = "timeout"
                else:
                    rtt_sum += rtt
                    rtt_min  = min(rtt_min, rtt)
                    rtt_max  = max(rtt_max, rtt)
                    rtt_str  = f"{rtt:.1f}"

                received = sent - lost
                avg = rtt_sum / received if received > 0 else 0.0
                mn  = rtt_min  if received > 0 else 0.0
                mx  = rtt_max  if received > 0 else 0.0

                print(f"{ts:<25} {rtt_str:>9} {lost:>6} {sent:>6}"
                      f" {avg:>9.1f} {mn:>7.1f} {mx:>7.1f}")

                writer.writerow([ts, args.target,
                                  rtt if rtt is not None else "",
                                  lost, sent,
                                  round(avg, 2), round(mn, 2), round(mx, 2)])
                f.flush()
                time.sleep(args.interval)

        except KeyboardInterrupt:
            for label, server, p in procs:
                p.terminate()
                print(f"[iperf] {label} to {server} stopped (pid={p.pid})")

            received = sent - lost
            avg = rtt_sum / received if received > 0 else 0.0
            print(f"\n--- Summary ---")
            print(f"{args.target}: sent={sent} lost={lost} "
                  f"loss%={100 * lost // sent if sent else 0}% "
                  f"avg={avg:.1f}ms "
                  f"min={rtt_min if received > 0 else 0:.1f}ms "
                  f"max={rtt_max if received > 0 else 0:.1f}ms")


if __name__ == "__main__":
    main()
