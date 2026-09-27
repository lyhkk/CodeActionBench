"""Small, best-effort run-version metadata captured when an episode starts.

`TRANSCRIPT_SCHEMA_VERSION` is the versioned contract of `transcript.jsonl` (the field-level
equivalent of Terminal-Bench's ATIF `schema_version`): both tracks stamp it on the first `meta`
record, and any consumer — report renderer, offline judge, SFT/RL export — can refuse a transcript
it does not understand instead of silently misreading it.

Bump rules:
  MINOR — new optional field on an existing record type (old readers stay correct).
  MAJOR — a field is removed, renamed, or changes meaning (old readers become wrong).

Record types at 2.1: `meta` (first line; run identity + declared scaffold parameters), `tool`
(one executed tool call), `done`, `no_tool_call`, `endpoint_failure`, `episode_fatal`, `end`.
Version 1.1 adds optional `sim_step_clock`, `sim_step_start`, `sim_step_end`, and `sim_step`
fields so continuous episode video can be synchronized to tool events without wall-clock gaps.
Version 1.2 adds the optional structured `failure` field to terminal and individual failure-bearing
records. Legacy status, step counting, budget, and termination semantics are unchanged.
Version 2.0 replaces track-specific step semantics with one control contract: `budget_used` counts
only chargeable calls, `total_calls` counts every external call, and `done` is free in both tracks.
The original 2.0 vendor-agent surface delivered the task through a first-free `get_task` control call;
the current instruction contract retires that tool and injects the same configuration before the
first model turn. It also records versioned tool and instruction surfaces. 1.x and 2.x trials are
readable but never aggregate.
Version 2.1 adds normalized `model_turn`, `context_retry`, `context_length_exceeded`,
`output_length_exceeded`, and `unsupported_stop_reason` records; model/tool records may carry
requested/effective output limits, deterministic context retention, and full-vs-model-visible
result projection metadata. Existing 2.0 budget fields do not change meaning.
Version 2.2 adds optional provider-native `reasoning_content` to `model_turn` and the first tool
record for a turn, plus `usage.reasoning_tokens`; reasoning is transcript metadata and is not
reinserted into the next provider prompt.
Version 2.3 adds the `model_refusal` terminal record and the optional `stop_details` field on
`model_turn`. A provider policy decline previously fell through to `unsupported_stop_reason`,
which attributed a model-side termination to the harness; it now carries the `model_refused`
failure code with the provider refusal category. Existing records do not change meaning.
Version 2.4 adds the `unintended_collision` terminal record. It carries the structured
`unintended_collision` failure and ends the tested attempt without permitting another tool call.
Version 2.5 adds optional `observed_before`, `observed_after`, `resulting_pose`, and `tick` fields
to that terminal payload so the collision-ending action retains the same raw boundary evidence as
returned ActionResults.
Version 2.6 adds optional `args`, `result`, `model_result`, and `result_projection` fields to the
`unintended_collision` terminal record. For `run_code` and `run_program`, the result retains the
completed sandbox `internal_trace` and execution error before the terminal error replaces it.
That record remains readable historical schema. Tool surface 9.0 stops emitting it: a new contact now appears
inside an ordinary tool result as a recoverable ABORTED ActionResult (or as a run-code
`interrupted_action`), which changes the versioned tool/instruction surface rather than the
transcript record grammar.
Version 2.7 adds the optional `reasoning_turns` counter to the `end` record: how many model turns
actually came back with reasoning text. `reasoning_content` has been an optional field since 2.2,
which made "the vendor returned none" and "this run predates the field" indistinguishable without
walking the whole transcript — and that ambiguity is what let every claude-opus-5 run record an
empty reasoning column unnoticed until 2026-08-08. The counter states it in one place, and the
registry's `reasoning_capture` declaration says whether zero is a defect. Existing records do not
change meaning.
Version 2.8 adds `provider_attempt_failed`, emitted for every failed provider request whether it is
retried successfully or terminates the turn. It records only the structured failure class, attempt
number, failed-request latency, retry decision/backoff, and observation time; raw provider prose,
request content, response content, and credentials are excluded. Existing records do not change
meaning.
Version 2.9 adds `provider_quota_wait`, `provider_response_telemetry`, and
`provider_error_telemetry`. The first records cross-process quota-limiter delay outside the
model-visible conversation and model wall budget; the latter two contain only allow-listed
rate-limit/reset/request-id response headers. Provider retry records may also include the selected
retry source and shared cooldown duration. Existing records do not change meaning.
Version 2.10 adds unambiguous image-transport counters to `result_projection` and `end`: image
messages, image blocks, unique images, repeated sends, and duplicate observation references that
were suppressed before transport. The compatibility `images_in_context` field keeps its historical
meaning (image-bearing message count). Existing records do not change meaning.
Version 2.11 adds request-context telemetry for the stable text history and request-local visual
tail: their separate estimates, image count, contributing message ids, and duplicate image payloads
suppressed across the retained tail. Existing records do not change meaning.
Version 2.12 adds Harness-only text-compaction counters, first turn, dropped message ids, ledger-entry
count, protected/recent-exact token estimates, visual-tail budget trimming, and context retry count.
None of these fields are inserted into the provider request. Existing records do not change meaning.
Version 2.13 adds `protected_units_eroded` and `force_compact_unreducible`: how many otherwise
protected turn units the budget ladder had to erode into the ledger, and whether a provider-driven
`force_compact` request was already at the floor (anchors + newest turn unit + newest visual
source), in which case the retry is not issued and the provider's own failure is reported. It also
RESTORES the historical meaning of `context_compacted` (any reduction of the recorded conversation,
image eviction included), which 2.12 had silently narrowed to text compaction only; `text_compacted`
remains the narrow fact. No other existing record changes meaning.
Version 2.14 adds the `physical_time_budget_exhausted` terminal record. It carries the typed
model-side, scoreable failure plus the simulator-step evidence; the out-of-band verifier still
judges the resulting state. Version 2.15 changes that record's step semantics from an interrupt
inside a primitive to the completed D0 tool boundary. ``physics_steps`` can therefore exceed the
declared threshold by the final atomic tool's advance; no subsequent tool is allowed to start.
Version 2.16 adds the historical `tool_cancelled_after_recoverable_abort` record for direct calls
submitted in the same assistant turn after a contact abort. Version 2.17 extends it to program
tools. Version 3.0 renames the record to `tool_cancelled_after_action_abort` and makes the barrier
reason-neutral: every structured ActionResult with status ABORTED cancels queued siblings.
Version 3.1 adds explicit resource-unit aliases to the `end` record: `tool_calls_used`,
`tool_call_budget`, `total_tool_dispatches`, and `model_turns`. The legacy `steps`, `budget_used`,
`total_calls`, and `turns` fields remain byte-compatible; the new names prevent a report from
dividing model turns by a tool-call ceiling.
"""
import subprocess
from functools import lru_cache
from pathlib import Path

TRANSCRIPT_SCHEMA_VERSION = "3.1"


@lru_cache(maxsize=8)
def git_version(path=None):
    """Return the current commit and dirty flag, or an empty dict outside a Git worktree."""
    cwd = Path(path or __file__).resolve()
    if cwd.is_file():
        cwd = cwd.parent
    try:
        root = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short=8", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {}


# --- Reference scaffold identity ---------------------------------------------------------
# Defined here, not in the harness package, because BOTH sides record it: the reference agent
# stamps it on its transcript, and the episode server's bridge records the scaffold identity
# it can attest without importing harness code (the benchmark layer must not depend on any
# agents/* module).
SCAFFOLD_NAME = "codeaction-reference"
# 0.8.0 records provider transport + rate-limit policy in `config_sha256` and validates every PNG
# before it reaches a multimodal endpoint. Runs made under different timeout/pacing policies must
# not aggregate silently, even though quota waiting is credited out of the model wall budget.
# 0.9.0 gives Gemini 3.6 Flash a native, stateless GenerateContent adapter. It replays complete
# context and relies on Google's implicit common-prefix cache; no fixed CachedContent is created.
# 0.10.0 makes provider response validation and failure attribution protocol-independent. Empty
# completed turns are provider retries, structured HTTP evidence outranks echoed prose, and output
# truncation is scoreable only at the model's declared native limit.
# 0.11.0 removes the historical global temperature=0 request override. Provider defaults now apply
# unless a frozen request profile explicitly declares a numeric sampling temperature.
# 0.12.0 makes the task-level physical execution budget model-visible and terminal at the exact
# simulator boundary while preserving out-of-band verifier finalization.
# 0.13.0 freezes connect/first-response/stream-idle/whole-response timeouts separately and adds an
# opt-in Kimi streaming A/B path without changing the registry-default buffered transport.
# 0.14.0 raises the per-turn output request from 8192 so that every model requests its own declared
# native output limit instead of a scaffold-chosen ceiling.
# 0.15.0 returned one post-settle head observation after a recoverable contact abort and prevented
# later calls from the same assistant turn from executing against the changed scene.
# 0.16.0 applies that same-turn barrier when the abort was reported by a run_code/run_program
# block rather than by a direct action call.
SCAFFOLD_VERSION = "0.16.0"
