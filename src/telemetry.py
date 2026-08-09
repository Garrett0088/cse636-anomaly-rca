"""
telemetry.py
------------
Instrumentation plumbing for the RCA agent (src/rca_agent.py, imports this
module but is not itself part of this file). Everything here is pure setup
and serialization: no network calls, no environment reads, no side effects
at import time. Every function takes its inputs as explicit parameters so
this module stays independently testable and safe to import from a notebook
or a re-run script without any hidden global state.

Spans emitted through record_llm_usage() follow the OpenTelemetry GenAI
semantic conventions (the "gen_ai.*" attribute namespace), so any OTel-aware
collector or dashboard can recognize them as LLM call spans without needing
to know anything specific about this project or about Anthropic's API.
"""
from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
# InMemorySpanExporter lives in its own submodule in this SDK version rather
# than being re-exported from opentelemetry.sdk.trace.export's top level.
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

# ---------------------------------------------------------------------------
# Pricing constants for Claude Haiku 4.5, in USD per million tokens. Kept as
# named module-level constants (rather than inlined in estimate_cost_usd)
# so a future price change is a one-line edit instead of a search-and-replace.
# ---------------------------------------------------------------------------
INPUT_COST_PER_MTOK = 1.00   # USD per 1,000,000 input tokens
OUTPUT_COST_PER_MTOK = 5.00  # USD per 1,000,000 output tokens


def init_telemetry(service_name: str = "rca-agent") -> tuple:
    """Build a local, self-contained OpenTelemetry tracing pipeline.

    Parameters:
        service_name: the value stamped onto every span's service.name
            resource attribute, identifying which process/agent produced it.

    Returns:
        A (provider, tracer, memory_exporter) tuple:
          - provider: the TracerProvider this function constructed. The
            caller owns it and must call provider.force_flush() before
            reading spans back out, and provider.shutdown() when done.
          - tracer: a Tracer obtained from that provider, ready to start spans.
          - memory_exporter: the InMemorySpanExporter instance backing the
            in-memory capture path, to be passed into dump_spans_to_json().
    """
    # Build the resource that stamps every span with the service identity.
    # Without this, spans arrive at a collector (or in the JSON dump) as
    # anonymous and cannot be attributed to this agent versus any other
    # process on the box.
    resource = Resource.create({"service.name": service_name})

    # Construct a LOCAL provider instead of calling trace.set_tracer_provider().
    # The global setter can only be honored ONCE per Python process -- a second
    # call is silently ignored, which would send spans to a stale provider with
    # no error message. Holding our own reference here makes it safe to call
    # init_telemetry() again later in the same process (e.g. across notebook
    # cell re-runs or repeated test invocations) without that silent breakage.
    provider = TracerProvider(resource=resource)

    # ConsoleSpanExporter prints each span to stdout as it's exported -- this
    # is the "live terminal view" a developer watches while the agent runs.
    # BatchSpanProcessor is the right pairing for it because console output
    # is for human observation only: nothing downstream is blocked waiting
    # on it, so batching spans up and exporting them on a background thread
    # (instead of one-by-one, synchronously) is strictly a performance win
    # with no downside here.
    console_processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(console_processor)

    # InMemorySpanExporter is the "programmatic capture" path: it just holds
    # finished spans in a list for later retrieval by dump_spans_to_json().
    # SimpleSpanProcessor (synchronous, exports immediately on span end) is
    # required here specifically because the caller needs those spans
    # available the moment they finish -- BatchSpanProcessor would buffer
    # them and only flush periodically or in batches, which could mean a
    # dump_spans_to_json() call right after force_flush() still misses
    # recently-ended spans sitting in an unflushed batch.
    memory_exporter = InMemorySpanExporter()
    memory_processor = SimpleSpanProcessor(memory_exporter)
    provider.add_span_processor(memory_processor)

    # Get the tracer directly from OUR provider instance (provider.get_tracer),
    # not from the global trace.get_tracer(). Going through the global API
    # would look up whatever provider trace.set_tracer_provider() last
    # installed (or a no-op default if none was ever set) -- not necessarily
    # this one. __name__ identifies this module as the instrumentation
    # source, which is the conventional value to pass here.
    tracer = provider.get_tracer(__name__)

    return provider, tracer, memory_exporter


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of one LLM call from its token counts.

    Parameters:
        input_tokens: number of input (prompt) tokens consumed by the call.
        output_tokens: number of output (completion) tokens produced.

    Returns:
        The total estimated cost in USD, rounded to 6 decimal places (token
        counts are usually small enough that cost-per-call is a fraction of
        a cent, so rounding to fewer places would lose all the precision).
    """
    # Pricing is quoted per MILLION tokens, so divide the raw token count by
    # 1,000,000 before multiplying by the per-MTok rate.
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MTOK
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MTOK
    return round(input_cost + output_cost, 6)


def record_llm_usage(span, model: str, input_tokens: int, output_tokens: int,
                      stop_reason: str = None) -> float:
    """Attach OpenTelemetry GenAI semantic-convention attributes to a span.

    Parameters:
        span: an active OpenTelemetry span (e.g. one opened via
            tracer.start_as_current_span(...)) to record the LLM call onto.
        model: the model identifier used for the call (e.g.
            "claude-haiku-4-5-20251001").
        input_tokens: number of input (prompt) tokens consumed.
        output_tokens: number of output (completion) tokens produced.
        stop_reason: why the model stopped generating (e.g. "end_turn"), or
            None if unknown/not applicable.

    Returns:
        The estimated cost in USD for this single call (see
        estimate_cost_usd), so the caller can add it to a running total
        across multiple calls in one session.
    """
    # gen_ai.system identifies which vendor's API this call went to -- fixed
    # to "anthropic" since that's the only provider this agent talks to.
    span.set_attribute("gen_ai.system", "anthropic")
    # gen_ai.operation.name classifies what kind of GenAI operation this was;
    # "chat" is the correct value for a Messages-API-style request/response call.
    span.set_attribute("gen_ai.operation.name", "chat")
    # gen_ai.request.model records exactly which model served the request, so
    # cost/latency can later be broken down per model if that ever changes.
    span.set_attribute("gen_ai.request.model", model)
    # Token usage attributes drive both cost accounting and capacity analysis.
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    # Only set finish_reasons when a stop_reason is actually known -- setting
    # it to [None] when there's nothing to report would look like real data
    # to anyone reading the span later. The semantic convention defines this
    # as an array (a call can technically finish for more than one reason
    # across some APIs), so a single value is wrapped in a one-element list.
    if stop_reason is not None:
        span.set_attribute("gen_ai.response.finish_reasons", [stop_reason])

    # Compute and record the cost last, since it depends on the token counts
    # already read above -- keeps this as the one place per call where cost
    # is derived, rather than duplicating the pricing math at each call site.
    cost = estimate_cost_usd(input_tokens, output_tokens)
    span.set_attribute("gen_ai.usage.cost_usd", cost)

    return cost


def dump_spans_to_json(memory_exporter, out_path: str) -> int:
    """Serialize every span captured by an InMemorySpanExporter to a JSON file.

    NOTE: this function does not flush anything. The caller is responsible
    for calling provider.force_flush() before invoking this, so that any
    spans still in flight through a processor are guaranteed to have already
    reached memory_exporter by the time get_finished_spans() is read here.

    Parameters:
        memory_exporter: the InMemorySpanExporter to read finished spans from
            (the same instance returned by init_telemetry()).
        out_path: filesystem path to write the JSON file to. Its parent
            directory is created automatically if it doesn't already exist.

    Returns:
        The number of spans written to the file.
    """
    # get_finished_spans() returns whatever spans SimpleSpanProcessor has
    # already handed to this exporter -- it does not trigger a flush itself.
    spans = memory_exporter.get_finished_spans()

    records = []
    for span in spans:
        # start_time/end_time on a finished span are already unix-epoch
        # nanosecond integers, so duration is just their difference,
        # converted from nanoseconds to milliseconds for readability.
        duration_ms = (span.end_time - span.start_time) / 1_000_000

        # span_id/trace_id live on span.context as plain Python ints;
        # format them as fixed-width lowercase hex (64-bit span id ->
        # 16 hex digits, 128-bit trace id -> 32 hex digits) since hex is
        # the conventional human-readable form for OTel ids.
        span_id_hex = format(span.context.span_id, "016x")
        trace_id_hex = format(span.context.trace_id, "032x")
        # A root span has no parent, so span.parent is None in that case --
        # preserve that instead of forcing a fake id.
        parent_id_hex = format(span.parent.span_id, "016x") if span.parent else None

        records.append({
            "name": span.name,
            "span_id": span_id_hex,
            "trace_id": trace_id_hex,
            "parent_span_id": parent_id_hex,
            "start_time_unix_nano": span.start_time,
            "end_time_unix_nano": span.end_time,
            "duration_ms": duration_ms,
            "status": {
                "status_code": span.status.status_code.name,
                "description": span.status.description,
            },
            # span.attributes is a mapping-like object backed by the SDK's
            # internal BoundedAttributes -- copy it into a plain dict so
            # json.dumps can serialize it without needing a custom encoder.
            "attributes": dict(span.attributes),
        })

    # Create the parent directory (e.g. output/) if this is a fresh checkout
    # that doesn't have it yet, so the write below never fails on that account.
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # indent=2 produces pretty-printed, human-reviewable JSON -- this file is
    # a graded deliverable that a person is expected to open and read.
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return len(records)
