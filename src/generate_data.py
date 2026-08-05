"""
Generate a synthetic metrics dataset with *labelled* injected anomalies for the
Week 5 anomaly-detection lab. The ground-truth `is_anomaly` column lets you
score a detector with precision/recall (see evaluation.py / detect.py).

Columns: timestamp, cpu_pct, mem_pct, req_per_sec, error_rate, is_anomaly

Usage:
    python generate_data.py            # 7 days @ 1-min step, ~2% anomalies
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta


def generate(days: int = 7, step_minutes: int = 1, seed: int = 7):
    rng = random.Random(seed)
    start = datetime(2025, 10, 1, 0, 0, 0)
    n = days * 24 * (60 // step_minutes)

    rows = []
    for i in range(n):
        ts = start + timedelta(minutes=i * step_minutes)
        # normal behaviour: daily cycle + noise
        daily = math.sin((ts.hour - 6) / 24.0 * 2 * math.pi)
        cpu = 40 + 15 * daily + rng.gauss(0, 3)
        mem = 55 + 8 * daily + rng.gauss(0, 2)
        req = 300 + 120 * daily + rng.gauss(0, 20)
        err = max(0.0, 0.2 + rng.gauss(0, 0.05))   # ~0.2% baseline error rate
        is_anom = 0

        # inject anomalies ~2% of the time as correlated spikes
        if rng.random() < 0.02:
            is_anom = 1
            cpu += rng.uniform(25, 45)
            err += rng.uniform(3, 8)               # error-rate blowout
            req += rng.uniform(-150, 200)

        rows.append((
            ts.isoformat(),
            round(max(1, min(100, cpu)), 2),
            round(max(1, min(100, mem)), 2),
            round(max(1, req), 1),
            round(err, 3),
            is_anom,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate labelled synthetic metrics.")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--step-minutes", type=int, default=1)
    ap.add_argument("--out", default="metrics_sample.csv")
    args = ap.parse_args()

    rows = generate(days=args.days, step_minutes=args.step_minutes)
    anomalies = sum(r[-1] for r in rows)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "cpu_pct", "mem_pct", "req_per_sec",
                    "error_rate", "is_anomaly"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows ({anomalies} labelled anomalies, "
          f"{anomalies / len(rows) * 100:.1f}%) to {args.out}.")


if __name__ == "__main__":
    main()
