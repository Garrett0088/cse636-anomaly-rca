# Intelligent Anomaly Detection and AI-Generated Root Cause Analysis

**CSE636 — DevOps with AI, Summer 2026**
**Author:** Garrett
**Instructor:** Prof. Qingsong Zhang

A complete pipeline that detects anomalies in infrastructure metrics, groups the resulting
alerts into distinct incidents, and uses an LLM agent with tool access to investigate the most
severe incident and write a root cause analysis — with every LLM call instrumented using
OpenTelemetry GenAI semantic conventions.

---

## What this system does

The pipeline answers three questions in sequence, each one narrowing the problem for the next:

| Stage | Question it answers | Module |
|---|---|---|
| Detection | Which individual minutes look abnormal? | `src/anomaly_detector.py` |
| Grouping | Which of those minutes belong to the same real incident? | `src/alert_grouper.py` |
| Investigation | Why did the most severe incident happen? | `src/rca_agent.py` |

Detection produces hundreds of isolated flags. Grouping collapses them into a handful of
incidents with severity levels. Only the single `critical` incident is worth an LLM's attention —
which is what makes the agent's cost defensible.

---

## Repository structure

```
CSE636-anomaly-rca/
├── data/
│   ├── metrics_sample.csv          # Provenance record — never read downstream
│   ├── metrics_with_incident.csv   # Working dataset — every module reads this
│   └── logs_sample.txt             # 296 structured log lines across three services
├── src/
│   ├── generate_data.py            # Course starter file (see Attribution)
│   ├── inject_incident.py          # Injects a realistic 40-minute incident
│   ├── generate_logs.py            # Derives log evidence from the metric data
│   ├── anomaly_detector.py         # Isolation Forest + precision/recall/F1
│   ├── alert_grouper.py            # Rule-based grouping and severity
│   ├── telemetry.py                # OTel setup + GenAI span helpers
│   └── rca_agent.py                # Agentic RCA with native tool use
├── notebooks/
│   └── analysis.ipynb              # Three figures + narrative analysis
├── output/
│   ├── rca_report.md               # Generated RCA for the detected incident
│   ├── spans_sample.json           # 12 captured OTel spans from that run
│   ├── fig1_week_overview.png
│   ├── fig2_incident_zoom.png
│   └── fig3_group_severity.png
├── requirements.txt
└── README.md
```

---

## Setup and running

All commands run from the project root in PowerShell on Windows 11.

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

```powershell
pip install -r requirements.txt
```

The RCA agent is the only component that requires network access. Set the key in the shell —
it is never written to disk, and never read into a variable by application code (the Anthropic
SDK reads it from the environment itself):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Then run any component. Every module is designed to run from the project root:

```powershell
python -m src.anomaly_detector
```

```powershell
python -m src.alert_grouper
```

```powershell
python -m src.rca_agent
```

The notebook is opened separately and run with the `.venv` kernel:

```powershell
jupyter notebook notebooks/analysis.ipynb
```

### Regenerating the data

The committed CSVs are the exact files every result below was produced from, so regeneration is
not required. If you do regenerate, note that `generate_logs.py` refuses to overwrite an existing
output file — a deliberate guard, explained under "Failures worth naming."

---

## Data and provenance

### The two-CSV rule

`data/metrics_sample.csv` is the raw output of the course-provided generator. It is a **provenance
record**: it is committed, but no module reads it. Every downstream module reads
`data/metrics_with_incident.csv` instead.

The reason is auditability. If the working dataset is ever corrupted or a result becomes
impossible to reproduce, there is an untouched baseline to diff against. During development the
notebook and the generator briefly shared a default output filename, and running one silently
overwrote the other's 10,080-row dataset with a 500-row one — no error, no warning. The two-file
separation and an explicit `--out` argument are the fix.

To verify the rule holds:

```powershell
Select-String -Path src\*.py -Pattern "metrics_sample"
```

Only `generate_data.py` — the file that *writes* it — should match.

### Why the incident had to be injected

The course generator produces 185 labelled anomalies scattered across seven days as isolated
single-minute spikes. That is a perfectly good dataset for evaluating a detector, but random noise
has **no root cause**. It cannot support alert grouping (there is nothing to group) or agentic RCA
(there is nothing to explain).

`inject_incident.py` adds a 40-minute incident on 2025-10-05 with a realistic
onset → peak → recovery shape across four metrics. Two design constraints made it useful rather
than trivial:

- **Correlated, opposing movement.** During the incident `cpu_pct` rises while `req_per_sec`
  *falls*. That combination is the fingerprint of resource exhaustion rather than a traffic surge,
  and it is what lets the agent rule out "scale up" as a diagnosis. If both metrics moved the same
  direction, the data would tell the wrong story.
- **Magnitude deliberately not distinctive.** The incident's peak `error_rate` (5.0–8.3) overlaps
  the range background noise already reaches (~7.05). The incident is *not* separable by how large
  any single metric gets — only by how long it persists and how many metrics move together. This
  constraint is what forces the alert grouper to be more than a threshold.

`logs_sample.txt` is derived from the metric data at runtime rather than hardcoded: the generator
locates the incident window by finding the longest consecutive run of anomalies, then scales log
intensity from that window's own values. The three services tell a layered story — `payment-svc`
degrading then failing, `order-svc` timing out downstream of it, and `auth-svc` staying calm
throughout. That last one is negative evidence: a quiet service rules out a system-wide surge.

### Attribution

`src/generate_data.py` is **course starter material provided by Prof. Zhang**, copied unmodified
from the Week 5 lab folder. All other source files in this repository are original work.

---

## Components

### Anomaly detector — `src/anomaly_detector.py`

Isolation Forest over four features: `cpu_pct`, `mem_pct`, `req_per_sec`, `error_rate`. The model
is fit **blind to the label** — `is_anomaly` is read only inside `evaluate()`, never during
training or prediction.

Contamination sweep across the assignment's three suggested values:

| Contamination | Precision | Recall | F1 |
|---|---|---|---|
| 0.01 | 1.00 | 0.45 | 0.62 |
| 0.04 | 0.55 | 0.99 | **0.70** |
| 0.10 | 0.22 | 0.99 | 0.36 |

**Tuning reasoning.** Contamination is not really a tuning knob for accuracy — it is a direct
statement about how much of the data you *expect* to be anomalous, and it trades precision against
recall along a curve. At 0.01 the detector is nearly always right when it fires but misses more
than half of the real anomalies. At 0.10 it catches essentially everything at the cost of four
false alarms for every true one, which in production is alert fatigue: a queue nobody reads. F1
peaks at 0.04.

Accuracy is a trap on this dataset. Only 1.8% of rows are anomalous, so a detector that predicts
"normal" for every single row scores ~98% accuracy with a recall of zero. Precision and recall on
the anomaly class are the only numbers that mean anything here.

**Two different contamination values are used deliberately.** The scoring above uses 0.04, the
F1-optimal setting. The alert grouper consumes 0.01 instead. These are different tools with
different priorities: scoring wants a balanced view of detector quality, while grouping wants high
precision so that groups are not polluted by false positives before an LLM is asked to reason over
them. Using one value for both would compromise one of the two purposes.

### Alert grouper — `src/alert_grouper.py`

Takes the detector's per-minute flags and collapses them into incidents.

**Approach: rule-based, deliberately.** The rubric allows either, and this was the more defensible
choice:

| | Rule-based | LLM-based |
|---|---|---|
| Cost per run | $0 | ~$0.01+ |
| Determinism | Identical output every run | Varies |
| Testability | Trivial | Difficult |
| Failure surface | None — no network | API errors, rate limits, timeouts |

Clustering consecutive timestamps is arithmetic. Spending a model call on arithmetic adds cost and
nondeterminism for no analytical gain. The LLM is reserved for `rca_agent.py`, where the task is
genuinely open-ended reasoning and the model earns its place.

**Grouping rule.** Alerts within 5 minutes of each other join the same group. The tolerance bridges
single-minute gaps where the detector missed a row mid-incident, without merging genuinely
unrelated spikes hours apart. At 0 minutes tolerance one real event fractures into fragments; at 60
minutes, unrelated background blips merge into a fictitious incident.

**Severity rule.** Severity is derived from **duration and breadth, never magnitude**:

| Severity | Condition |
|---|---|
| `critical` | ≥ 10 minutes **and** ≥ 3 metrics deviating |
| `warning` | ≥ 3 minutes **or** ≥ 2 metrics deviating |
| `info` | Everything else |

A metric counts as deviating when it sits more than 2σ from a baseline computed from
**non-flagged rows only**. Excluding the flagged rows matters: a baseline that contains the
anomalies is inflated by them, and nothing can look like an outlier relative to a baseline that
already includes the outliers — the same reason a control group excludes the treatment.

The deviation check is **direction-agnostic** (absolute distance from the mean). A one-sided
"is it high?" check would miss the throughput collapse entirely, and that collapse is the single
most diagnostic signal in this dataset.

**Result:** 101 raw alerts → 76 groups, with exactly one `critical`:
`2025-10-05T14:10:00 → 14:33:00`, 23 minutes, 24 alerts, 4 of 4 metrics deviating. That window was
discovered entirely by the rules — it is not hardcoded anywhere in the module.

**Why severity ignores magnitude, demonstrated.** Group 75 is a single background-noise minute at
`2025-10-07T23:18:00` that reaches **4 of 4 metrics deviating** — the same breadth as the real
incident's worst minute. A magnitude-based or breadth-only rule would call it critical. It is
correctly classified `warning` because it lasts one minute. This is the strongest available
evidence that persistence, not extremity, separates a real incident from noise.

### Telemetry — `src/telemetry.py`

OpenTelemetry instrumentation for the agent's LLM calls, emitting GenAI semantic-convention
attributes: `gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, and a
computed `gen_ai.usage.cost_usd`.

Two exporters run in parallel — a `ConsoleSpanExporter` for live viewing during development, and an
`InMemorySpanExporter` whose contents are serialized to `output/spans_sample.json`.

**One architectural decision worth stating:** the module builds a **local** `TracerProvider` and
never calls `trace.set_tracer_provider()`. That global setter can only be honored once per Python
process — a second call is silently ignored with no error, which would send spans to a stale
provider during any notebook re-run. Holding an explicit reference makes re-execution safe.

A related principle: telemetry is an **observability side-channel, not a data path**. When
`run_investigation()` needed to report its iteration count, the tempting shortcut was to read it
back out of the exported span attributes. That would have made tracing load-bearing — turn
instrumentation off for a debug run and the report breaks. The function returns the value directly
instead.

### RCA agent — `src/rca_agent.py`

A native Anthropic tool-use loop. The agent is given two tools and told only that an incident
occurred in a specific window across four metrics. **It is never told the cause.**

| Tool | Purpose | Guardrail |
|---|---|---|
| `get_metrics` | Aggregate stats and sampled rows for a time window | Refuses windows > 120 minutes |
| `get_logs` | Log lines for a window, optionally filtered by service | Caps at 60 lines with an explicit truncation notice |

The loop sends the conversation, and if the model responds with `stop_reason == "tool_use"` it
executes the requested tool, appends the result, and sends again — up to 6 iterations. The model
chooses which tools to call, with what arguments, in what order. That choice is what makes the
system agentic rather than a scripted two-step pipeline.

**Prompt hygiene.** No system prompt, user prompt, or tool description mentions connection pools,
service names, exhaustion, or the directional relationship between CPU and throughput. Writing the
answer into the prompt would produce an impressive-looking report that demonstrates nothing.

**The run.** 4 iterations, 27.2 seconds, $0.0555, 12 spans captured in a correct parent-child tree
under a single `rca.investigation` root.

The agent's first move is telling: it requested metrics for `13:30–14:15` — reaching 30 minutes
*before* the incident to establish a baseline. Nothing instructed it to do that. It then pulled the
recovery side, then logs across the full span, then narrowed. That is an investigative strategy,
not prompt recitation.

**What it concluded.** Connection pool exhaustion in `payment-svc`, with `order-svc` timeouts as a
downstream cascade — the correct cause and the correct direction. `order-svc` also threw errors, and
a shallow analysis would have blamed the service with the loudest failures. It also hedged
appropriately on the trigger, listing a traffic spike, database degradation, and a connection leak
as candidates, because the logs establish *that* the pool exhausted but not *why*. Restraint under
genuine uncertainty is the right behavior.

### Visualization — `notebooks/analysis.ipynb`

Three figures, each making a specific point:

- **`fig1_week_overview.png`** — all four metrics across the full week, with detector-flagged points
  overlaid and critical groups shaded. Shows the incident in context against background noise.
- **`fig2_incident_zoom.png`** — the same view narrowed to the critical group's window plus 60
  minutes of padding on each side, so onset and recovery are visible. The window is **derived from
  the grouper's output**, not hardcoded — a hardcoded window would still produce a plausible-looking
  plot on data containing no incident at all.
- **`fig3_group_severity.png`** — every group plotted as duration against breadth, with the critical
  thresholds drawn as dashed lines. This figure is the argument of the whole grouping module in one
  image: the critical group sits alone in the upper-right quadrant, while background groups that
  reach identical breadth are stranded at zero duration on the left.

---

## Results summary

| Measure | Value |
|---|---|
| Dataset | 10,080 rows, 7 days at 1-minute cadence |
| Labelled anomalies | 224 (185 background + injected incident) |
| Detector F1 (contamination 0.04) | 0.70 (precision 0.55, recall 0.99) |
| Raw alerts (contamination 0.01) | 101 |
| Groups after grouping | 76 |
| Critical incidents identified | 1 |
| Detected incident window | 2025-10-05T14:10:00 → 14:33:00 |
| Agent iterations | 4 |
| Agent wall time | 27.2 s |
| Agent cost | $0.0555 |
| Spans captured | 12 |

---

## Failures worth naming

Three things did not work as expected. Each is a real limitation rather than a rough edge.

**1. The agent's top remediation contradicts its own evidence.** Its first recommendation is to
raise the connection pool from 41 to 100–150 connections to handle "400+ req/sec." But throughput
*collapsed* to 116 during the incident — demand fell, it did not spike. Raising the pool size treats
a capacity problem that the data shows did not exist; the likely outcome is 150 stuck connections
instead of 41.

The contradiction is visible inside the report itself: the Root Cause section correctly lists a
connection leak as a candidate, while Contributing Factors asserts that traffic overwhelmed
capacity. The model read the numbers correctly and still produced a remediation that contradicts
them, because "pool exhausted → make the pool bigger" is a strong pattern-match that overrode the
evidence in front of it. This is the most important finding in the project: an LLM's diagnostic
reasoning and its prescriptive reasoning can fail independently, and the prescription is the part a
human is most likely to act on without checking.

**2. The alert grouper's reduction ratio is too weak to be useful.** 101 alerts became 76 groups —
a ratio of 1.3. Of those, 73 are single-minute `warning`s. An on-call engineer receiving that queue
is still drowning. The cause is the `warning` rule using **or**: two metrics at 2σ is a low bar that
background noise clears constantly, so almost everything gets promoted above `info`. Requiring
duration **and** breadth, or raising the breadth threshold to 3, would push isolated blips down to
`info` where they belong. Critical classification is unaffected, since it already requires both
conditions. The correct detection masked a defective tier structure — the system found the right
answer while the mechanism underneath it was miscalibrated.

**3. The agent fetched overlapping data and paid for it twice.** On iteration 2 it requested three
log windows in one turn: `14:00–14:15`, `14:15–14:40`, and `14:00–14:40`. The third is a strict
superset of the first two. Because the Anthropic API is stateless and every turn resends the full
conversation, those duplicated lines rode along in every subsequent call. Input tokens went
1,140 → 5,316 → 16,140 across three iterations. This is visible only because the spans recorded it —
which is the entire argument for instrumenting agents rather than trusting them.

A smaller issue: the agent reported `error_rate` peaking at "8.3%" when the column is not a
percentage — it is an unbounded rate that reaches ~7 in background noise. Nothing told it the units,
and it inferred wrong. Tool descriptions should carry units explicitly.

---

## Requirements met

| Requirement | Points | Where |
|---|---|---|
| Anomaly detector with precision/recall | 20 | `src/anomaly_detector.py`, sweep table above |
| At least one visualization | 10 | `notebooks/analysis.ipynb`, 3 figures in `output/` |
| Alert grouping on 10+ alerts | 15 | `src/alert_grouper.py`, 101 alerts → 76 groups |
| Agentic RCA with 2+ tools | 25 | `src/rca_agent.py`, `get_metrics` + `get_logs` |
| Generated RCA report | 15 | `output/rca_report.md` |
| OTel GenAI spans captured | 10 | `src/telemetry.py`, `output/spans_sample.json` |
| README + reflection | 5 | This file, plus the separate reflection PDF |
