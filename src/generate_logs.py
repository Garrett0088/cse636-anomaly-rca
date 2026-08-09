"""
generate_logs.py
-----------------
Generate synthetic application logs that tell the SAME story as the incident
already injected into data/metrics_with_incident.csv, from the application's
point of view instead of the metrics' point of view.

Nothing about the incident's timing or severity is hardcoded here: this
script re-discovers the incident window at runtime by finding the longest
run of consecutive is_anomaly==1 rows in the metrics file (background
anomalies elsewhere are single isolated minutes, so the longest run is
unambiguous), then reads that window's own cpu_pct/error_rate values to
drive both how many log lines get written per minute and how severe they
are. The result: payment-svc logs an escalating WARN -> ERROR sequence that
independently names the mechanism (a connection pool running out of
connections), while order-svc separately logs its own downstream timeouts
caused by payment-svc -- two services corroborating the same root cause from
two different vantage points, with throughput-collapse-not-traffic-surge
implied by payment-svc's own request volume dropping as its errors climb.
auth-svc stays on routine INFO throughout, which is itself a clue: a real
traffic surge would hit every service, not just the one with a starved
downstream dependency.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants. These are all SPEC/design constants (line-count budget, phase
# proportions, RNG seed) -- never a metric value, threshold, or timestamp
# read out of the CSV. Every number that actually describes the incident
# (when it happened, how severe it got) is computed at runtime in main().
# ---------------------------------------------------------------------------
DEFAULT_METRICS_PATH = "data/metrics_with_incident.csv"
DEFAULT_OUT_PATH = "data/logs_sample.txt"

PRE_WINDOW_MINUTES = 15   # log coverage starts this many minutes before the incident
POST_WINDOW_MINUTES = 15  # ...and ends this many minutes after it

TARGET_LINES_MIN = 260  # inclusive lower bound on total lines written
TARGET_LINES_MAX = 300  # inclusive upper bound on total lines written

BASE_WEIGHT = 1.0   # every minute's baseline share of log volume, before intensity scaling
VOLUME_SCALE = 6.0  # how much extra volume a fully-intense (intensity=1.0) minute gets

ESCALATION_THRESHOLD = 0.5   # intensity >= this -> ERROR/pool=exhausted, else WARN/pool=degraded
RECOVERY_INFO_THRESHOLD = 0.02  # once decayed-recovery intensity drops below this, logs go pure INFO

SERVICES = ("order-svc", "auth-svc", "payment-svc")
SEED = 13  # fixed seed -> reproducible log file across runs (own stream, unrelated to the other scripts')


def parse_args() -> argparse.Namespace:
    """Define and parse the script's CLI contract."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic application logs around the injected incident window."
    )
    parser.add_argument(
        "--metrics", default=DEFAULT_METRICS_PATH,
        help="Path to the metrics CSV to read -- must be the file with the injected incident, "
             "never the untouched provenance sample.",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT_PATH,
        help="Path to write the generated log file to. Refuses to overwrite an existing file.",
    )
    return parser.parse_args()


def load_metrics(path: str) -> pd.DataFrame:
    """Load the metrics CSV with the timestamp column parsed into real datetimes."""
    # parse_dates lets every later comparison/subtraction use real Timestamp
    # arithmetic instead of fragile string comparisons.
    return pd.read_csv(path, parse_dates=["timestamp"])


def find_longest_anomaly_run(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    """Locate the single longest run of consecutive is_anomaly==1 rows.

    Returns (start_ts, end_ts, window_df) for that run. This is how the
    incident's timing is discovered at runtime instead of hardcoded: the
    file's background anomalies are each isolated single minutes, so the
    longest consecutive run is unambiguously the injected incident.
    """
    is_anom = df["is_anomaly"] == 1
    if not is_anom.any():
        # Fail loud rather than silently generating a log file with no incident at all.
        raise SystemExit("no is_anomaly==1 rows found in the metrics file -- nothing to build logs around")
    # Every time is_anom flips (0->1 or 1->0) the cumulative sum increments,
    # so rows sharing both a group id AND is_anom==True are one contiguous run.
    group_id = (is_anom != is_anom.shift()).cumsum()
    run_lengths = df.loc[is_anom].groupby(group_id[is_anom]).size()
    longest_group = run_lengths.idxmax()  # group id of the longest anomalous run
    window_df = df.loc[is_anom & (group_id == longest_group)]
    start_ts = window_df["timestamp"].iloc[0]  # first row of that run, in time order
    end_ts = window_df["timestamp"].iloc[-1]   # last row of that run
    return start_ts, end_ts, window_df


def build_log_span(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Slice the metrics file down to the pre-window + incident + post-window minutes the log will cover."""
    pre_start = start_ts - pd.Timedelta(minutes=PRE_WINDOW_MINUTES)
    post_end = end_ts + pd.Timedelta(minutes=POST_WINDOW_MINUTES)
    # A timestamp boolean mask naturally clips near the start/end of the
    # file (e.g. an incident too close to row 0) with no manual index
    # arithmetic or bounds-checking needed.
    mask = (df["timestamp"] >= pre_start) & (df["timestamp"] <= post_end)
    span = df.loc[mask, ["timestamp", "cpu_pct", "error_rate"]].reset_index(drop=True).copy()
    # Tag every minute with which side of the incident it falls on, purely
    # by comparing its own timestamp to start_ts/end_ts -- never a fixed clock time.
    span["phase"] = np.select(
        [span["timestamp"] < start_ts, span["timestamp"] <= end_ts],
        ["pre", "incident"],
        default="post",
    )
    return span


def compute_intensity(span: pd.DataFrame, window_df: pd.DataFrame) -> pd.Series:
    """Build one intensity value in [0, 1] per minute across the whole log span.

    Inside the incident window this is the REAL cpu_pct/error_rate signal,
    min-max normalized against the window's OWN range -- so a minute at the
    window's worst cpu/error reading scores ~1.0 and a minute at the
    window's best reading scores ~0.0. Outside the window: pre-incident
    minutes come out near 0 automatically (their cpu/error values sit below
    the window's own minimum). Post-incident minutes get a manufactured
    linear decay from the window's own peak intensity instead, because the
    metrics themselves are already back at baseline the row after the window
    ends (the CSV's own recovery ramp is baked into the window's last rows)
    -- without a manufactured decay there would be nothing left to scale the
    "pool draining" recovery lines against.
    """
    cpu_lo, cpu_hi = window_df["cpu_pct"].min(), window_df["cpu_pct"].max()
    err_lo, err_hi = window_df["error_rate"].min(), window_df["error_rate"].max()
    cpu_range = (cpu_hi - cpu_lo) or 1.0  # guard against a degenerate single-value window
    err_range = (err_hi - err_lo) or 1.0
    # Normalize each column separately (they live on very different scales),
    # then clip to [0,1] since pre/post rows can fall outside the window's own range.
    cpu_norm = ((span["cpu_pct"] - cpu_lo) / cpu_range).clip(0, 1)
    err_norm = ((span["error_rate"] - err_lo) / err_range).clip(0, 1)
    raw = ((cpu_norm + err_norm) / 2).to_numpy()  # equal-weight blend of both signals

    is_incident = (span["phase"] == "incident").to_numpy()
    peak = raw[is_incident].max()  # the window's own worst minute, used as the recovery decay's starting point

    is_post = (span["phase"] == "post").to_numpy()
    post_count = int(is_post.sum())
    intensity = raw.copy()
    if post_count:
        # recovery_tail is a runtime-computed FRACTION of however many
        # post-window minutes actually exist, not a fixed number of minutes.
        recovery_tail = max(1, post_count // 3)
        # Post rows are contiguous and 1-minute-spaced in the underlying
        # file, so their position in this slice IS their elapsed-minutes-
        # since-incident-end -- no separate timestamp subtraction needed.
        offsets = np.arange(post_count)
        decay = np.clip(peak * (1 - offsets / recovery_tail), 0, None)
        intensity[is_post] = decay
    return pd.Series(intensity, index=span.index)


def allocate_line_counts(intensity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn per-minute intensity into per-minute integer log-line counts summing to a random 260-300 total."""
    weight = BASE_WEIGHT + intensity * VOLUME_SCALE  # higher-intensity minutes get proportionally more lines
    n = len(weight)
    # Pick the exact total once, randomly, inside the spec's own 260-300 range.
    target_total = int(rng.integers(TARGET_LINES_MIN, TARGET_LINES_MAX + 1))
    remaining = target_total - n  # every minute is guaranteed exactly 1 line first (no silent minute)
    if remaining < 0:
        # The log span has more minutes than the line budget can cover even
        # at 1 line/minute -- fail loud instead of silently truncating.
        raise SystemExit(
            f"log span has {n} minutes, more than the {TARGET_LINES_MAX}-line budget "
            "can cover with >=1 line per minute"
        )
    shares = weight / weight.sum() * remaining  # each minute's fractional share of the remaining budget
    floor_shares = np.floor(shares).astype(int)
    leftover = remaining - int(floor_shares.sum())
    # Largest-remainder method: hand the few leftover lines to the minutes
    # with the biggest fractional part, so counts.sum() lands EXACTLY on target_total.
    fractional = shares - floor_shares
    order = np.argsort(-fractional)
    counts = np.full(n, 1) + floor_shares
    counts[order[:leftover]] += 1
    return counts


def compute_pool_max(df: pd.DataFrame, start_ts: pd.Timestamp) -> int:
    """Derive a plausible connection-pool max size from the REAL pre-incident baseline req_per_sec."""
    pre_start = start_ts - pd.Timedelta(minutes=PRE_WINDOW_MINUTES)
    baseline = df.loc[(df["timestamp"] >= pre_start) & (df["timestamp"] < start_ts), "req_per_sec"]
    # Fall back to the whole file's mean only in the edge case where the
    # incident starts too close to the very first row for a pre-window to exist.
    mean_req = baseline.mean() if len(baseline) else df["req_per_sec"].mean()
    return max(1, round(mean_req / 10))  # ties the fictional pool size to actually-measured traffic


def _rid(rng: np.random.Generator, n: int = 8) -> str:
    """Build a random lowercase-hex id string, used for order/txn/session/request ids in log lines."""
    digits = "0123456789abcdef"
    return "".join(digits[rng.integers(0, len(digits))] for _ in range(n))


ORDER_ROUTINE = [
    lambda rng: ({"order_id": _rid(rng), "latency_ms": int(rng.integers(15, 90))}, "order created"),
    lambda rng: ({"order_id": _rid(rng), "latency_ms": int(rng.integers(20, 120))}, "order shipped"),
    lambda rng: ({"order_id": _rid(rng), "sku_count": int(rng.integers(1, 6))}, "inventory reserved"),
]
AUTH_ROUTINE = [
    lambda rng: ({"user_id": _rid(rng, 6), "latency_ms": int(rng.integers(10, 60))}, "user login succeeded"),
    lambda rng: ({"session_id": _rid(rng)}, "session token refreshed"),
    lambda rng: ({"user_id": _rid(rng, 6)}, "user logged out"),
]
PAYMENT_ROUTINE = [
    lambda rng: ({"txn_id": _rid(rng), "latency_ms": int(rng.integers(25, 110))}, "payment authorized"),
    lambda rng: ({"txn_id": _rid(rng), "amount_cents": int(rng.integers(500, 20000))}, "payment captured"),
]
ROUTINE_BY_SERVICE = {"order-svc": ORDER_ROUTINE, "auth-svc": AUTH_ROUTINE, "payment-svc": PAYMENT_ROUTINE}


def format_line(ts: pd.Timestamp, level: str, service: str, fields: dict, msg: str) -> str:
    """Render one log line in the required 'TIMESTAMP LEVEL service key=value ... msg="..."' format."""
    kv = " ".join(f"{k}={v}" for k, v in fields.items())  # dict insertion order -> stable, readable field order
    return f'{ts:%Y-%m-%dT%H:%M:%S} {level} {service} {kv} msg="{msg}"'


def make_routine_line(ts: pd.Timestamp, service: str, rng: np.random.Generator) -> str:
    """Build one routine INFO line for the given service, used outside the incident and during full recovery."""
    templates = ROUTINE_BY_SERVICE[service]
    fields, msg = templates[rng.integers(0, len(templates))](rng)  # pick a random template for this service
    return format_line(ts, "INFO", service, fields, msg)


def make_payment_incident_line(ts: pd.Timestamp, intensity: float, pool_max: int, rng: np.random.Generator) -> str:
    """Build one payment-svc WARN/ERROR line naming connection-pool exhaustion as the mechanism, scaled by intensity."""
    waiters = round(pool_max * (intensity ** 2) * 2)  # squared growth: near-zero at onset, spikes at peak
    txn_id = _rid(rng)
    if intensity < ESCALATION_THRESHOLD:
        active = min(pool_max, round(pool_max * intensity))  # pool still has some headroom left
        fields = {"pool": "degraded", "active": active, "max": pool_max, "waiters": waiters, "txn_id": txn_id}
        return format_line(ts, "WARN", "payment-svc", fields, "connection pool nearing exhaustion")
    # High intensity: the pool has nothing left to give, so active == max and acquires start timing out.
    timeout_ms = round(2000 + intensity * 3000)  # acquire-timeout grows as pressure rises
    fields = {
        "pool": "exhausted", "active": pool_max, "max": pool_max,
        "waiters": waiters, "acquire_timeout_ms": timeout_ms, "txn_id": txn_id,
    }
    return format_line(ts, "ERROR", "payment-svc", fields, "failed to acquire db connection: pool exhausted")


def make_order_incident_line(ts: pd.Timestamp, intensity: float, rng: np.random.Generator) -> str:
    """Build one order-svc line showing ITS independent symptom of the same root cause: payment-svc timing out."""
    req_id = _rid(rng)
    timeout_ms = round(1000 + intensity * 4000)  # longer waits recorded as the downstream pool gets worse
    level = "WARN" if intensity < ESCALATION_THRESHOLD else "ERROR"  # mirrors payment-svc's own escalation gate
    fields = {"req_id": req_id, "downstream": "payment-svc", "timeout_ms": timeout_ms}
    return format_line(ts, level, "order-svc", fields, "downstream call to payment-svc timed out")


def make_payment_recovery_line(ts: pd.Timestamp, intensity: float, pool_max: int) -> str:
    """Build one payment-svc WARN line for the post-incident drain -- always WARN, since recovery never regresses to ERROR."""
    active = round(pool_max * intensity)
    waiters = round(pool_max * intensity * 0.5)
    fields = {"pool": "draining", "active": active, "max": pool_max, "waiters": waiters}
    return format_line(ts, "WARN", "payment-svc", fields, "connection pool draining, recovering")


def generate_lines(span: pd.DataFrame, counts: np.ndarray, pool_max: int,
                    rng: np.random.Generator) -> list[tuple[pd.Timestamp, str]]:
    """Walk every minute in the log span and emit counts[i] irregularly-timed lines for that minute."""
    lines: list[tuple[pd.Timestamp, str]] = []
    recovered_announced = False  # ensures the "pool restored to healthy" line fires exactly once
    for i, row in enumerate(span.itertuples(index=False)):
        for _ in range(int(counts[i])):
            # A random 0-59s offset inside the minute is what makes intervals irregular
            # rather than one line landing on every exact minute boundary.
            ts = row.timestamp + pd.Timedelta(seconds=int(rng.integers(0, 60)))
            if row.phase == "pre":
                service = SERVICES[rng.integers(0, len(SERVICES))]
                lines.append((ts, make_routine_line(ts, service, rng)))
            elif row.phase == "incident":
                # Weighted service pick: payment-svc and order-svc dominate as intensity
                # climbs; auth-svc keeps a flat share throughout (deliberately unaffected).
                weights = np.array([1 + row.intensity * 2, 1.0, 1 + row.intensity * 4])
                pick = int(rng.choice(3, p=weights / weights.sum()))
                if pick == 0:
                    lines.append((ts, make_order_incident_line(ts, row.intensity, rng)))
                elif pick == 1:
                    lines.append((ts, make_routine_line(ts, "auth-svc", rng)))
                else:
                    lines.append((ts, make_payment_incident_line(ts, row.intensity, pool_max, rng)))
            else:  # phase == "post"
                if row.intensity > RECOVERY_INFO_THRESHOLD:
                    # Still actively draining: payment-svc's share grows with remaining intensity.
                    weights = np.array([1.0, 1.0, 1 + row.intensity * 3])
                    pick = int(rng.choice(3, p=weights / weights.sum()))
                    if pick == 2:
                        lines.append((ts, make_payment_recovery_line(ts, row.intensity, pool_max)))
                    else:
                        lines.append((ts, make_routine_line(ts, SERVICES[pick], rng)))
                else:
                    # Fully recovered: announce it exactly once, then routine INFO from here on.
                    if not recovered_announced:
                        fields = {"pool": "healthy", "max": pool_max}
                        lines.append((ts, format_line(ts, "INFO", "payment-svc", fields,
                                                       "connection pool restored to healthy state")))
                        recovered_announced = True
                    else:
                        service = SERVICES[rng.integers(0, len(SERVICES))]
                        lines.append((ts, make_routine_line(ts, service, rng)))
    return lines


def write_logs(lines: list[tuple[pd.Timestamp, str]], out_path: str) -> None:
    """Sort every generated line chronologically and write it to out_path, refusing to overwrite an existing file."""
    out = Path(out_path)
    if out.exists():
        # Fail loud instead of silently overwriting whatever is already there.
        raise SystemExit(f"{out_path} already exists -- pass a different --out so nothing gets silently overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)  # in case --out points at a not-yet-created directory
    lines_sorted = sorted(lines, key=lambda pair: pair[0])  # one interleaved chronological stream across all services
    with out.open("w", encoding="utf-8") as fh:
        for _, text in lines_sorted:
            fh.write(text + "\n")


def main() -> None:
    """Discover the incident window, build the log span around it, and write the synthetic log file."""
    args = parse_args()
    rng = np.random.default_rng(SEED)  # one shared generator -> reproducible output across runs

    df = load_metrics(args.metrics)
    start_ts, end_ts, window_df = find_longest_anomaly_run(df)

    span = build_log_span(df, start_ts, end_ts)
    span["intensity"] = compute_intensity(span, window_df)  # per-minute severity/volume driver, all data-derived

    counts = allocate_line_counts(span["intensity"].to_numpy(), rng)
    pool_max = compute_pool_max(df, start_ts)

    lines = generate_lines(span, counts, pool_max, rng)
    write_logs(lines, args.out)

    print(f"Incident window: {start_ts} -> {end_ts} ({len(window_df)} rows)")
    print(f"Log span: {span['timestamp'].iloc[0]} -> {span['timestamp'].iloc[-1]} ({len(span)} minutes)")
    print(f"pool_max derived from pre-window baseline req_per_sec: {pool_max}")
    print(f"Wrote {len(lines)} log lines to {args.out}")


if __name__ == "__main__":
    main()
