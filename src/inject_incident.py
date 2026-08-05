"""
inject_incident.py
-------------------
Overlay a synthetic 40-minute incident (onset -> peak -> recovery) onto an
existing metrics CSV produced by generate_data.py, WITHOUT touching that
script or the source file. This script only reads one CSV and writes a new
one -- it never mutates data/metrics_sample.csv in place.

Why the incident matters: generate_data.py's 185 background anomalies are
each a single isolated minute -- great for noise, but there is no sustained,
multi-minute story a human (or an agentic root-cause tool) could point at
and say "here is what happened, and here is how it resolved." This script
adds exactly one such story: cpu/mem/errors climb (ONSET), stay unhealthy
for a while (PEAK), then recover (RECOVERY), while request throughput first
dips and then collapses under the strain before climbing back.

Why seed=42 here and not generate_data.py's seed=7: the two scripts must
never share a random stream, or debugging "why did row X get this value"
would require mentally tracing through BOTH scripts' calls in lockstep.
Picking our own fixed (but different) seed keeps the two generators fully
independent while still making THIS script's own output reproducible --
which matters because a graded assignment needs to be diffable/re-runnable
and produce identical results every time.

Note on hour boundaries: the default --start (2025-10-05T14:00:00) plus a
40-minute window ends at 14:39, so it never crosses an HH:00 boundary and
stays inside "hour 14" the whole time. That fact doesn't actually change
anything about how this script works (see build_mem_pct/build_req_per_sec
below: we read each row's OWN already-baked value as its baseline, we never
recompute generate_data.py's hourly sine wave ourselves), but it's worth
noting since it's the reason "local baseline" is such a simple concept for
this particular default window.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# Fixed phase lengths, spelled out as named constants (not magic numbers
# scattered through the code) so the 10/20/10 split from the assignment
# spec is visible and easy to double check in one place.
ONSET_LEN = 10
PEAK_LEN = 20
RECOVERY_LEN = 10
WINDOW_LEN = ONSET_LEN + PEAK_LEN + RECOVERY_LEN  # 40 total incident rows

# "A few percent" of jitter, expressed as a fraction for multiplicative use.
JITTER_FRAC = 0.03

# Our own RNG seed -- deliberately different from generate_data.py's seed=7.
SEED = 42


def parse_args() -> argparse.Namespace:
    # Centralizing the CLI contract in one function makes it obvious at a
    # glance what inputs this script accepts, and keeps main() focused on
    # the actual data-transformation logic instead of argument bookkeeping.
    parser = argparse.ArgumentParser(
        description="Inject a synthetic sustained incident into a metrics CSV."
    )
    parser.add_argument(
        "--inp", default="data/metrics_sample.csv",
        help="Path to the source CSV produced by generate_data.py (read-only).",
    )
    parser.add_argument(
        "--out", default="data/metrics_with_incident.csv",
        help="Path to write the modified CSV to (never the same as --inp).",
    )
    parser.add_argument(
        "--start", default="2025-10-05T14:00:00",
        help="Timestamp (YYYY-MM-DDTHH:MM:SS) of the first incident row.",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    # parse_dates=["timestamp"] converts the timestamp column from plain
    # strings into real pandas Timestamp objects up front. That lets us
    # compare/subtract timestamps directly later (e.g. "is this row exactly
    # one minute after that row?") instead of doing fragile string slicing.
    return pd.read_csv(path, parse_dates=["timestamp"])


def locate_incident_window(df: pd.DataFrame, start_ts: pd.Timestamp) -> pd.Index:
    # Find the row LABEL whose timestamp equals start_ts using a boolean
    # mask, rather than searching for a matching string. This is robust to
    # any formatting differences between how --start was typed and how the
    # CSV stores timestamps, since both sides are now real datetimes.
    matches = df.index[df["timestamp"] == start_ts]
    if len(matches) == 0:
        # Fail loudly and immediately rather than silently building an
        # incident window out of the wrong rows (or no rows at all).
        raise SystemExit(f"--start {start_ts} was not found in the input file")
    start_idx = matches[0]  # there should only ever be one exact match

    if start_idx + WINDOW_LEN > len(df):
        # Guard against a --start too close to the end of the file, which
        # would otherwise silently hand back a short, truncated window.
        raise SystemExit("not enough rows remain after --start to build a full 40-row window")

    # A plain positional slice of the DataFrame's index gives us the 40
    # consecutive row labels, in ascending time order, that we'll overwrite.
    window_idx = df.index[start_idx: start_idx + WINDOW_LEN]

    # Defensive check: confirm these 40 rows are truly one minute apart with
    # no gaps, duplicates, or out-of-order timestamps. If generate_data.py's
    # output were ever regenerated with a different step size, this stops
    # us from quietly overwriting the wrong set of rows.
    expected = start_ts + pd.to_timedelta(np.arange(WINDOW_LEN), unit="m")
    actual = df.loc[window_idx, "timestamp"].to_numpy()
    if not (actual == expected.to_numpy()).all():
        raise SystemExit("the 40 rows starting at --start are not contiguous one-minute timestamps")

    return window_idx


def _apply_jitter_clip_round(rng: np.random.Generator, raw: np.ndarray,
                              low: float | None, high: float | None, decimals: int) -> np.ndarray:
    # Shared finishing pipeline for every generated column: add multiplicative
    # jitter, THEN clamp, THEN round. Jitter must come before clamping so a
    # jittered value can never slip outside the column's valid range; rounding
    # comes last since rounding a value already inside range can't push it
    # back out (so no second clamp pass is ever needed).
    jitter = 1 + rng.uniform(-JITTER_FRAC, JITTER_FRAC, raw.shape[0])  # one +/-3% multiplier per row
    jittered = raw * jitter
    # Multiplicative (not additive) jitter is used because our four metrics
    # live on very different scales (cpu ~50-95 vs error_rate ~0.2-9); a
    # single fixed +/- amount would be huge for one column and invisible for
    # another, whereas a percentage naturally scales with each value's size.
    if low is not None and high is not None:
        clipped = np.clip(jittered, low, high)
    elif low is not None:
        clipped = np.maximum(jittered, low)  # floor only, no ceiling
    else:
        clipped = jittered
    return np.round(clipped, decimals)


def build_cpu_pct(rng: np.random.Generator) -> np.ndarray:
    # ONSET: cpu climbs smoothly from a mildly-elevated 45% up to 78%.
    # np.linspace gives evenly spaced values between two endpoints -- this is
    # how we build a gradual ramp instead of an instant jump. A step change
    # would be unrealistic; real resource exhaustion builds up over minutes.
    onset = np.linspace(45, 78, ONSET_LEN)
    # PEAK: held in a tight but non-flat 88-96% band. Drawing a fresh random
    # value per row (rather than repeating one number 20 times) is what
    # makes the "held high" phase still look like noisy real telemetry
    # instead of an obviously synthetic flat line.
    peak = rng.uniform(88, 96, PEAK_LEN)
    # RECOVERY: decays from 90% back down to a still-elevated 50% (not all
    # the way to a calm baseline -- recovery from an incident is gradual).
    recovery = np.linspace(90, 50, RECOVERY_LEN)
    # Stitch the three phases together in time order into one length-40
    # array. Concatenation order matters: it must match window_idx's order.
    raw = np.concatenate([onset, peak, recovery])
    return _apply_jitter_clip_round(rng, raw, low=0, high=100, decimals=2)


def build_error_rate(rng: np.random.Generator) -> np.ndarray:
    # generate_data.py centers its baseline error_rate around 0.2 (NOT a
    # 0-1 fraction like 0.002) and its own background anomalies push
    # error_rate as high as ~7-8. So our incident values must sit well
    # ABOVE 0.2, or the "incident" would actually look like errors
    # improving -- exactly backwards from the intended story.
    # ONSET: errors climb off the ~0.2 baseline up toward 2.5.
    onset = np.linspace(0.25, 2.5, ONSET_LEN)
    # PEAK: sustained elevated errors, comparable in magnitude to the worst
    # of the existing background anomaly spikes (~7), so this reads as a
    # genuinely severe incident rather than a minor blip.
    peak = rng.uniform(5.0, 9.0, PEAK_LEN)
    # RECOVERY: errors decay from 6.0 back down toward the ~0.2-0.3 baseline.
    recovery = np.linspace(6.0, 0.30, RECOVERY_LEN)
    raw = np.concatenate([onset, peak, recovery])
    # Only a LOWER bound (0) is applied here, with no upper ceiling. This
    # mirrors generate_data.py's own error_rate handling exactly -- it does
    # round(max(0.0, err), 3) with no upper clamp -- and the real data
    # already contains error_rate values above 7, so capping at 1 here
    # would silently corrupt this column relative to the rest of the file.
    return _apply_jitter_clip_round(rng, raw, low=0, high=None, decimals=3)


def build_mem_pct(rng: np.random.Generator, baseline_mem: np.ndarray) -> np.ndarray:
    # Unlike cpu_pct/error_rate, mem_pct is expressed as an OFFSET added on
    # top of each row's own pre-existing baseline value (captured by the
    # caller before anything gets overwritten). Memory pressure during an
    # incident is naturally "some amount above whatever it already was,"
    # not an absolute number disconnected from the time of day.
    onset_offset = np.linspace(0, 5, ONSET_LEN)          # excess-over-baseline ramps 0 -> 5
    peak_offset = rng.uniform(13, 17, PEAK_LEN)          # held "~15 above baseline" as a small noisy band
    recovery_offset = np.linspace(15, 0, RECOVERY_LEN)   # excess drains back down to 0 (i.e. back to baseline)
    offset = np.concatenate([onset_offset, peak_offset, recovery_offset])
    raw = baseline_mem + offset  # add our incident bump on top of the row's real underlying signal
    return _apply_jitter_clip_round(rng, raw, low=0, high=100, decimals=2)


def build_req_per_sec(rng: np.random.Generator, baseline_req: np.ndarray) -> np.ndarray:
    # req_per_sec is expressed as a FRACTION of each row's own baseline
    # traffic level, not an absolute drop. A service under incident doesn't
    # lose a fixed number of requests -- it loses a proportion of whatever
    # traffic it was already receiving at that time of day.
    onset_frac = np.linspace(1.00, 0.90, ONSET_LEN)        # slight dip to 90% of baseline
    peak_frac = rng.uniform(0.30, 0.40, PEAK_LEN)          # collapses to ~35% of baseline (noisy band around it)
    recovery_frac = np.linspace(0.35, 1.00, RECOVERY_LEN)  # climbs back up to 100% of baseline
    frac = np.concatenate([onset_frac, peak_frac, recovery_frac])
    raw = baseline_req * frac
    # Floor at 1.0 to mirror generate_data.py's own round(max(1, req), 1)
    # convention -- a request rate of zero or negative doesn't make sense.
    return _apply_jitter_clip_round(rng, raw, low=1.0, high=None, decimals=1)


def inject_incident(df: pd.DataFrame, window_idx: pd.Index, rng: np.random.Generator) -> None:
    # Capture each row's ORIGINAL mem_pct/req_per_sec value before we
    # overwrite anything, and .copy() so that later mutating df's columns
    # can't retroactively change the baseline arrays we already pulled out.
    baseline_mem = df.loc[window_idx, "mem_pct"].to_numpy().copy()
    baseline_req = df.loc[window_idx, "req_per_sec"].to_numpy().copy()

    # Build all four finished columns before writing any of them back, so
    # partial failures never leave the DataFrame half-modified.
    cpu_pct_final = build_cpu_pct(rng)
    error_rate_final = build_error_rate(rng)
    mem_pct_final = build_mem_pct(rng, baseline_mem)
    req_per_sec_final = build_req_per_sec(rng, baseline_req)

    # Assign each full 40-row array back using .loc with the actual index
    # LABELS (never .iloc, and never through a filtered/copied intermediate
    # like df[mask]). Writing straight into df this way avoids pandas'
    # SettingWithCopyWarning and makes it unambiguous that we're mutating
    # the real DataFrame, not some throwaway view of it.
    df.loc[window_idx, "cpu_pct"] = cpu_pct_final
    df.loc[window_idx, "mem_pct"] = mem_pct_final
    df.loc[window_idx, "req_per_sec"] = req_per_sec_final
    df.loc[window_idx, "error_rate"] = error_rate_final
    df.loc[window_idx, "is_anomaly"] = 1  # every injected row is ground-truth labelled anomalous


def print_summary(df: pd.DataFrame, window_idx: pd.Index,
                   anomalies_before: int, anomalies_after: int) -> None:
    # Pull every reported number straight off the final df (post-injection),
    # so the printed summary always matches exactly what got written to disk.
    incident_start = df.loc[window_idx[0], "timestamp"]
    incident_end = df.loc[window_idx[-1], "timestamp"]
    cpu_min = df.loc[window_idx, "cpu_pct"].min()
    cpu_max = df.loc[window_idx, "cpu_pct"].max()
    err_min = df.loc[window_idx, "error_rate"].min()
    err_max = df.loc[window_idx, "error_rate"].max()

    print(f"Total rows: {len(df)}")
    print(f"Anomalies before injection: {anomalies_before}")
    print(f"Anomalies after injection:  {anomalies_after}")
    print(f"Incident window: {incident_start} -> {incident_end} ({WINDOW_LEN} rows)")
    print(f"cpu_pct range in window:    {cpu_min:.2f} - {cpu_max:.2f}")
    print(f"error_rate range in window: {err_min:.3f} - {err_max:.3f}")


def main() -> None:
    args = parse_args()
    # One shared Generator instance, reused for every random draw in this
    # run, seeded once with our fixed SEED. Reusing a single instance (as
    # opposed to re-seeding per column) means the exact sequence of random
    # numbers -- and therefore the exact output file -- is identical every
    # time this script is run, which is what "reproducible" means here.
    rng = np.random.default_rng(SEED)

    df = load_data(args.inp)
    # Snapshot the anomaly count BEFORE any mutation, since we need to
    # compare "before" vs "after" in the final summary.
    anomalies_before = int(df["is_anomaly"].sum())

    start_ts = pd.to_datetime(args.start)  # turn the CLI string into a real Timestamp for exact comparison
    window_idx = locate_incident_window(df, start_ts)

    inject_incident(df, window_idx, rng)

    # Recompute (rather than naively adding 40) because one row inside our
    # window (2025-10-05T14:25:00 with the default --start) was already a
    # pre-existing background anomaly before we overwrote it -- "+40" would
    # double count that row and overstate the total by one.
    anomalies_after = int(df["is_anomaly"].sum())

    # date_format matches generate_data.py's own ts.isoformat() output
    # exactly (YYYY-MM-DDTHH:MM:SS, no space separator), so every untouched
    # row in --out reads identically to the same row in --inp.
    df.to_csv(args.out, index=False, date_format="%Y-%m-%dT%H:%M:%S")

    print_summary(df, window_idx, anomalies_before, anomalies_after)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
