"""
rca_agent.py
------------
Agentic root-cause-analysis (RCA) system. An anomaly detector has already
flagged a sustained incident in a known time window -- this module builds an
agent that is told ONLY the window and which metrics were flagged, then uses
two data-lookup tools (get_metrics, get_logs) in a native Anthropic tool-use
loop to independently work out WHY the incident happened. Nothing in this
file's prompts or tool descriptions states the cause: it must be derived
from the data, or the whole exercise is meaningless.

Run this module from the project root as:

    python -m src.rca_agent

so that `data/...` and `output/...` paths resolve correctly and the
`from src.telemetry import ...` import works.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic

from src.telemetry import dump_spans_to_json, init_telemetry, record_llm_usage

# ---------------------------------------------------------------------------
# CONSTANTS. Every tunable knob for the agent lives here so behavior (model
# choice, budget caps, file locations) can be audited or changed in one place
# instead of hunting through the functions below.
# ---------------------------------------------------------------------------
MODEL = "claude-haiku-4-5-20251001"  # cheap, fast model -- fine for a bounded tool-use loop like this
MAX_ITERATIONS = 6       # hard cap on LLM round-trips, so a confused agent can't loop forever and run up cost
MAX_TOKENS = 2000        # per-response output cap passed to the API
MAX_LOG_LINES = 60       # max log lines get_logs will ever return in one call (see truncation comment below)
MAX_METRIC_MINUTES = 120  # max window width get_metrics will compute stats over in one call

METRICS_PATH = "data/metrics_with_incident.csv"
LOGS_PATH = "data/logs_sample.txt"
REPORT_PATH = "output/rca_report.md"
SPANS_PATH = "output/spans_sample.json"

# The only facts the agent is told up front: WHEN the incident was flagged and
# WHICH columns triggered it. Never the cause -- see INITIAL_PROMPT below.
INCIDENT_WINDOW_START = "2025-10-05T14:00:00"
INCIDENT_WINDOW_END = "2025-10-05T14:39:00"

METRIC_COLUMNS = ["cpu_pct", "mem_pct", "req_per_sec", "error_rate"]


def get_metrics(start_time: str, end_time: str) -> str:
    """Summarize system metrics from METRICS_PATH over an inclusive time window.

    Parameters:
        start_time: inclusive window start, ISO 8601 (e.g. "2025-10-05T14:00:00").
        end_time: inclusive window end, ISO 8601.

    Returns:
        A plain-text block: row count, per-column min/mean/max for
        cpu_pct/mem_pct/req_per_sec/error_rate, then up to 15 evenly-sampled
        raw rows. Never raises on bad input or an oversized window -- instead
        returns a plain-text explanation, because this function is called
        with arguments an LLM generated, and a crash here would kill the
        entire investigation loop over a single malformed tool call.
    """
    # Parse defensively: the caller is a model, not our own code, so a
    # malformed timestamp string is a real possibility, not a hypothetical.
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
    except ValueError:
        return (f"Error: could not parse start_time={start_time!r} or "
                f"end_time={end_time!r} as ISO 8601 timestamps. Please retry "
                "with the format YYYY-MM-DDTHH:MM:SS.")

    # Reject an inverted or oversized window as plain text (NOT an exception)
    # so the model sees the refusal in its next turn and can self-correct by
    # asking for a narrower range, rather than the whole script crashing.
    window_minutes = (end_dt - start_dt).total_seconds() / 60
    if window_minutes < 0:
        return "Error: start_time is after end_time. Swap them and retry."
    if window_minutes > MAX_METRIC_MINUTES:
        return (f"Requested window is {window_minutes:.0f} minutes, which "
                f"exceeds the {MAX_METRIC_MINUTES}-minute cap for get_metrics. "
                "Narrow the time range (e.g. investigate in smaller chunks) "
                "and call again.")

    # Load fresh on every call rather than caching: the file is only 10,080
    # rows (one week at 1-minute resolution), so re-reading is cheap, and it
    # keeps this function stateless and independently testable.
    df = pd.read_csv(METRICS_PATH, parse_dates=["timestamp"])

    # Inclusive filter on both ends, since the caller was told the window is inclusive.
    mask = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
    window_df = df.loc[mask]

    if window_df.empty:
        return f"No metrics rows found between {start_time} and {end_time}."

    lines = [f"Metrics for {start_time} to {end_time} ({len(window_df)} rows):", ""]

    # Per-column min/mean/max gives the model a compact numeric summary
    # without needing to see every raw row to spot a trend.
    for col in METRIC_COLUMNS:
        col_min = window_df[col].min()
        col_mean = window_df[col].mean()
        col_max = window_df[col].max()
        lines.append(f"  {col}: min={col_min:.3f} mean={col_mean:.3f} max={col_max:.3f}")

    n = len(window_df)
    if n <= 15:
        # Small enough window: just show everything, no sampling needed.
        sample_df = window_df
    else:
        # Evenly-spaced sample across the whole window (not just the first 15
        # rows) so the model sees the start, middle, and end of the window --
        # a narrow uncapped read here would blow the token budget on a wide
        # window, and a naive "first 15" read would miss how the window ends.
        sample_positions = np.linspace(0, n - 1, 15).round().astype(int)
        sample_positions = sorted(set(sample_positions.tolist()))
        sample_df = window_df.iloc[sample_positions]

    lines.append("")
    lines.append(f"Sample rows ({len(sample_df)} of {n}):")
    lines.append("timestamp            cpu_pct  mem_pct  req_per_sec  error_rate")
    for row in sample_df.itertuples(index=False):
        lines.append(
            f"{row.timestamp:%Y-%m-%dT%H:%M:%S}  {row.cpu_pct:7.2f}  {row.mem_pct:7.2f}  "
            f"{row.req_per_sec:11.1f}  {row.error_rate:10.3f}"
        )

    return "\n".join(lines)


def get_logs(start_time: str, end_time: str, service: str = None) -> str:
    """Retrieve application log lines from LOGS_PATH over an inclusive time window.

    Parameters:
        start_time: inclusive window start, ISO 8601.
        end_time: inclusive window end, ISO 8601.
        service: if given, only lines containing this substring are kept
            (e.g. to isolate one service's logs). None returns all services.

    Returns:
        Up to MAX_LOG_LINES matching lines as plain text, in file order. If
        more lines match, only the first MAX_LOG_LINES are returned along
        with an explicit truncation notice, so the model can tell its
        results were cut off (as opposed to those being the only matches)
        and knows to narrow its query instead of drawing conclusions from a
        silently incomplete picture.
    """
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
    except ValueError:
        return (f"Error: could not parse start_time={start_time!r} or "
                f"end_time={end_time!r} as ISO 8601 timestamps. Please retry "
                "with the format YYYY-MM-DDTHH:MM:SS.")

    matches = []
    with open(LOGS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            # Each line starts with "YYYY-MM-DDTHH:MM:SS " -- split on the
            # first space to pull just the timestamp, leaving the rest of
            # the line untouched for the service/content filters below.
            ts_str, _, _rest = line.partition(" ")
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue  # skip any line that doesn't start with a parseable timestamp
            if not (start_dt <= ts <= end_dt):
                continue
            if service is not None and service not in line:
                continue
            matches.append(line)

    total_matches = len(matches)
    if total_matches == 0:
        return f"No log lines found between {start_time} and {end_time}" + (
            f" for service={service!r}." if service else "."
        )

    if total_matches > MAX_LOG_LINES:
        # Cap the response instead of returning every match: an uncapped
        # return on a wide window could be thousands of lines, which would
        # consume the model's entire context/token budget on one tool
        # result. Truncating -- with an explicit marker saying so -- lets
        # the model see it got a partial view and choose to narrow its
        # query, instead of silently reasoning from an incomplete sample.
        remaining = total_matches - MAX_LOG_LINES
        shown = matches[:MAX_LOG_LINES]
        shown.append(f"[TRUNCATED: {remaining} more lines. Narrow the time window or filter by service.]")
        return "\n".join(shown)

    return "\n".join(matches)


# TOOL_SCHEMAS describes get_metrics/get_logs to the model in Anthropic's tool
# format. Descriptions state only the data source and the caps -- never a
# hint at what actually caused the incident, since that would hand the
# model the answer instead of letting it investigate.
TOOL_SCHEMAS = [
    {
        "name": "get_metrics",
        "description": (
            "Retrieve aggregate system metrics (cpu_pct, mem_pct, req_per_sec, "
            "error_rate) for an inclusive UTC time window. Returns row count, "
            "per-column min/mean/max, and up to 15 evenly-sampled raw rows as "
            f"plain text. The window must not exceed {MAX_METRIC_MINUTES} "
            "minutes -- wider requests are refused, so narrow the range and "
            "call again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "Inclusive window start, ISO 8601 (e.g. 2025-10-05T14:00:00).",
                },
                "end_time": {
                    "type": "string",
                    "description": "Inclusive window end, ISO 8601.",
                },
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "get_logs",
        "description": (
            "Retrieve application log lines for an inclusive UTC time window, "
            "optionally filtered to lines mentioning a single service name. "
            f"Returns up to {MAX_LOG_LINES} matching lines as plain text; if "
            "more lines match, only the first are returned along with a "
            "truncation notice -- narrow the time window or add a service "
            "filter to see more."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "Inclusive window start, ISO 8601 (e.g. 2025-10-05T14:00:00).",
                },
                "end_time": {
                    "type": "string",
                    "description": "Inclusive window end, ISO 8601.",
                },
                "service": {
                    "type": "string",
                    "description": "Optional: only return lines containing this service name. Omit for all services.",
                },
            },
            "required": ["start_time", "end_time"],
        },
    },
]

# The initial user turn. States WHEN the incident was flagged and WHICH
# metrics triggered it -- both already known to any on-call engineer paging
# in -- and nothing else. No service name, no mechanism, no hint about which
# direction any metric moved relative to any other. The agent must discover
# all of that itself via the tools.
INITIAL_PROMPT = f"""An automated anomaly detector flagged a sustained incident in production
between {INCIDENT_WINDOW_START} and {INCIDENT_WINDOW_END}. During this window,
all four monitored metrics -- cpu_pct, mem_pct, req_per_sec, and error_rate --
showed sustained deviation from their normal baseline.

You are a root cause analysis (RCA) agent. Use the get_metrics and get_logs
tools to investigate this window -- and, if useful, the time immediately
before and after it to establish a baseline and see how things resolved --
and determine what actually happened and why.

Once you have gathered enough evidence, respond with your final answer as a
structured RCA report using exactly these sections:

## Summary
## Timeline
## Evidence
## Root Cause
## Contributing Factors
## Recommended Remediation

Do not call any more tools once you are ready to write this final report --
just respond with the report text directly."""


def run_investigation(tracer) -> tuple[str, float, int]:
    """Run the agentic tool-use loop that investigates the incident.

    Parameters:
        tracer: an OpenTelemetry Tracer (from telemetry.init_telemetry) used
            to open the root span for this investigation and a child span
            per LLM call / tool call.

    Returns:
        A (final_text, total_cost_usd, iterations_used) tuple. final_text is
        the agent's structured RCA report (or, if MAX_ITERATIONS was reached
        first, whatever text exists plus a note that the cap was hit).

    Note: this returns a 3-tuple (including iterations_used) rather than the
    2-tuple a purely text+cost contract would suggest, because the caller
    (main -> write_report) needs the iteration count and there is no other
    place to source it without reaching into telemetry internals -- which
    would make an observability side-channel load-bearing for control flow.
    """
    # No explicit api_key= argument here: the SDK reads ANTHROPIC_API_KEY
    # from the environment itself. main() already verified the variable is
    # set (and never printed it) before calling this function, so the key's
    # actual value never passes through a variable in this function at all.
    client = anthropic.Anthropic()

    conversation = [{"role": "user", "content": INITIAL_PROMPT}]
    total_cost = 0.0
    final_text = ""
    completed = False   # becomes True only when the model ends its turn with a non-tool_use stop
    iterations_used = 0

    # The root span is the top of this investigation's trace: every LLM call
    # and every tool call below becomes a child (directly or transitively)
    # of this span, so a trace viewer shows one investigation as one tree.
    with tracer.start_as_current_span("rca.investigation") as root_span:
        for iteration in range(1, MAX_ITERATIONS + 1):
            iterations_used = iteration

            # Child span per LLM round-trip -- lets a trace viewer see
            # exactly how many model calls this investigation took and how
            # long/expensive each one was.
            with tracer.start_as_current_span("rca.llm_call") as llm_span:
                llm_span.set_attribute("rca.iteration", iteration)

                # The Anthropic API is stateless -- it has no memory of prior
                # turns. Every call must resend the ENTIRE conversation so
                # far, or the model has no idea what's already happened.
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    tools=TOOL_SCHEMAS,
                    messages=conversation,
                )

                # Record token usage + cost on this call's own span, and
                # keep a running total across the whole investigation.
                cost = record_llm_usage(
                    llm_span, MODEL,
                    response.usage.input_tokens, response.usage.output_tokens,
                    response.stop_reason,
                )
                total_cost += cost

            # stop_reason tells us WHY the model stopped generating this turn:
            #   "tool_use"  -> the model wants to call one or more tools before continuing
            #   "end_turn"  -> the model is done and this is its final answer
            #   (also possible: "max_tokens", "stop_sequence", etc. -- treated
            #    the same as "not tool_use" here, since either way there are
            #    no tool calls to service and the loop should stop)
            if response.stop_reason != "tool_use":
                # Pull out just the text blocks -- a final answer should be
                # plain text, but join defensively in case of multiple blocks.
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                completed = True
                break

            # The model wants tool(s). Append its request to the conversation
            # BEFORE running any tools: the next thing the API needs to see,
            # in order, is (1) what the assistant asked for, then (2) our
            # answer to that request. Skipping this step or reordering it
            # would send the API a tool_result with no matching tool_use,
            # which the API rejects.
            conversation.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue  # a turn can mix explanatory text with tool calls; only tool_use blocks need dispatch

                # One child span per individual tool call, named after the
                # tool itself, so a trace viewer can see which specific
                # get_metrics/get_logs calls happened and how big their
                # results were, without reading the whole transcript.
                with tracer.start_as_current_span(f"rca.tool.{block.name}") as tool_span:
                    tool_span.set_attribute("rca.tool.name", block.name)
                    tool_span.set_attribute("rca.tool.start_time", str(block.input.get("start_time", "")))
                    tool_span.set_attribute("rca.tool.end_time", str(block.input.get("end_time", "")))

                    if block.name == "get_metrics":
                        result_text = get_metrics(block.input["start_time"], block.input["end_time"])
                    elif block.name == "get_logs":
                        result_text = get_logs(
                            block.input["start_time"], block.input["end_time"],
                            block.input.get("service"),
                        )
                    else:
                        # Defensive fallback: the model should never request a
                        # tool outside TOOL_SCHEMAS, but if it somehow does,
                        # report that back as a tool result instead of crashing.
                        result_text = f"Error: unknown tool {block.name!r}."

                    tool_span.set_attribute("rca.tool.result_chars", len(result_text))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # links this result back to the specific tool_use block that requested it
                    "content": result_text,
                })

            # All tool results from this turn go back as ONE user message
            # (a single assistant turn can request multiple tools at once,
            # and the API expects all their results together in the next turn).
            conversation.append({"role": "user", "content": tool_results})

        if not completed:
            # The loop ran out of iterations while the model was still
            # calling tools -- return whatever text happens to exist (likely
            # none, since a non-empty final_text would have set completed=True)
            # plus a clear note explaining why the report may be incomplete.
            note = f"\n\n[NOTE: investigation stopped after reaching the {MAX_ITERATIONS}-iteration cap before producing a final answer.]"
            final_text = (final_text or "(No final report was produced before the iteration cap was reached.)") + note

        # Root-span summary attributes: the two headline numbers for this
        # investigation, visible at a glance without walking every child span.
        root_span.set_attribute("rca.total_iterations", iterations_used)
        root_span.set_attribute("rca.total_cost_usd", total_cost)

    return final_text, total_cost, iterations_used


def write_report(text: str, cost: float, iterations: int) -> None:
    """Write the agent's RCA findings to REPORT_PATH as a markdown file.

    Parameters:
        text: the agent's final report text (already markdown-formatted by the model).
        cost: total USD cost of the investigation, for the metadata block.
        iterations: how many LLM round-trips the investigation took.
    """
    out = Path(REPORT_PATH)
    # Create output/ if this is a fresh checkout that doesn't have it yet.
    out.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    report = (
        "# Root Cause Analysis Report\n\n"
        "**Model:** " + MODEL + "  \n"
        "**Generated:** " + generated_at + "  \n"
        "**Iterations:** " + str(iterations) + "  \n"
        f"**Total cost (USD):** {cost:.6f}\n\n"
        "---\n\n"
        + text + "\n"
    )
    out.write_text(report, encoding="utf-8")


def main() -> None:
    """Entry point: validate the API key, run the investigation, and write both output artifacts."""
    # Read the key ONLY to check it's present -- never print it, never write
    # it anywhere, and never pass it through any other function's arguments.
    # If it's missing, fail fast with a clear message rather than letting the
    # Anthropic SDK raise its own (less obvious) error deep in the call stack.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: the ANTHROPIC_API_KEY environment variable is not set. "
              "Set it before running this agent.", file=sys.stderr)
        sys.exit(1)

    # Build a local telemetry pipeline for this run (see telemetry.py for why
    # this never touches the global trace provider).
    provider, tracer, memory_exporter = init_telemetry("rca-agent")

    final_text, cost, iterations = run_investigation(tracer)
    write_report(final_text, cost, iterations)

    # Flush the batched console exporter so every span it's holding gets
    # printed before we read spans back out. The in-memory exporter's
    # SimpleSpanProcessor already delivered its spans synchronously, so this
    # flush is only needed for the console side, but it's harmless to call
    # unconditionally here.
    provider.force_flush()

    span_count = dump_spans_to_json(memory_exporter, SPANS_PATH)

    print(f"Investigation complete: {iterations} iteration(s), ${cost:.6f} total cost.")
    print(f"Wrote {span_count} span(s) to {SPANS_PATH}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
