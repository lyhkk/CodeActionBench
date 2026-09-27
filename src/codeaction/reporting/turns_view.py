"""One human-readable, turn-by-turn view of an episode: ``turns.v1.json``.

Every run flavor writes raw transcripts under its own contract (reference-agent transcript,
vendor stream sidecar, episode-server transcript), and those contracts version independently per
model and per harness. This module is the single projection that collapses all of them into one
stable shape, so a consumer (the site exporter, a human) reads ONE file per episode instead of
speaking N contract versions.

Ground rules:
 * The view is a PURE FUNCTION of the transcript files already in the episode directory. It is
   deterministic (no wall clock, no environment), so regenerating it can never produce new bytes
   from unchanged inputs.
 * It is a projection, never a second source of truth. Raw transcripts remain the evidence; every
   turn carries enough keys (turn index, call ids, step numbers) to find the raw record.
 * Coverage fails loudly: an assistant turn in the raw transcript that cannot be rendered as
   exactly one turn record raises instead of silently dropping. Per-call gaps (a call the loop
   never dispatched) are legal and are recorded, not fatal.
 * NEVER write this file into an already-sealed attempt: the artifact manifest verifier reports
   unlisted files as an integrity failure. At run time the controller writes it BEFORE sealing;
   for historical attempts use :func:`build_turns_view` in memory and do not persist.

Turn semantics (matches the runner): a turn is one model inference. The assistant message's
``tool_calls`` are that turn's actions; their interface results are aggregated under the call that
produced them, and each turn lists ``observations_in`` — the previous turn's call ids — because
those results are the INPUT this turn's thinking consumed. One numbering, both causal readings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TURNS_VIEW_SCHEMA = "turns-view.v1"
TURNS_VIEW_NAME = "turns.v1.json"
# One agent per episode today. The field exists so multi-agent episodes extend the schema by
# adding rows, not by changing shape.
DEFAULT_AGENT_ID = "agent-0"
# Guard for human readability: no single string value (a stray base64 payload, a runaway trace)
# may dominate the file. Raw transcripts keep the full value.
MAX_STRING_CHARS = 20000

# Reference-transcript events that report the outcome of one tool call and carry (turn,
# call_index). "tool" and "limit" are the normal path; the rest are terminal or cancellation
# outcomes that still answer exactly one call.
_CALL_OUTCOME_EVENTS = (
    "tool", "limit", "done", "episode_fatal", "tool_cancelled_after_action_abort",
)

# Per-turn bookkeeping and telemetry that must not surface as a terminal event. Unknown event
# names DO surface: a new terminal status appearing here is information, not noise.
_NON_TERMINAL_EVENTS = frozenset({
    "meta", "model_turn", "tool", "limit", "context_retry", "context_retry_declined",
    "provider_response_telemetry", "tool_cancelled_after_action_abort", "end",
})


class TurnsViewError(ValueError):
    """A raw transcript could not be projected without losing an assistant turn."""


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue                     # partial trailing line of a crashed run
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _bounded(value: Any) -> Any:
    """Deterministically truncate oversized strings anywhere in a JSON value."""
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        return value[:MAX_STRING_CHARS] + f"…[truncated {len(value) - MAX_STRING_CHARS} chars]"
    if isinstance(value, dict):
        return {key: _bounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_bounded(item) for item in value]
    return value


def _message_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [str(block.get("text") or "") for block in content
                 if isinstance(block, dict) and block.get("type") == "text"]
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return None


def _call_ids(turn: int, count: int) -> list[str]:
    return [f"t{turn}.c{index}" for index in range(count)]


def _reasoning_tokens(usage):
    """Reasoning spend from a provider usage record, wherever the adapter put it."""
    if not isinstance(usage, dict):
        return None
    value = usage.get("reasoning_tokens")
    if value is None and isinstance(usage.get("completion_tokens_details"), dict):
        value = usage["completion_tokens_details"].get("reasoning_tokens")
    return int(value) if isinstance(value, int) and value > 0 else None


def _thinking_summary(turns):
    """Per-episode reasoning accounting: one read of coverage instead of a scan over turns.

    Four disjoint states. Visible text counts as a reasoning signal but is never folded into the
    `thinking` field itself -- they are different channels (public statement vs thinking) and the
    projection must not reassign channel identity. The release-corpus analysis uses the same
    merge: Opus 5's reasoning gaps are short opening turns whose public text carries the plan.
    """
    think = [bool(t.get("thinking")) for t in turns]
    evid = [not a and bool(t.get("thinking_evidence")) for a, t in zip(think, turns)]
    text_only = [not a and not b and bool(t.get("text"))
                 for a, b, t in zip(think, evid, turns)]
    return {
        "turns_with_thinking_text": sum(think),
        "turns_with_thinking_evidence_only": sum(evid),
        "turns_with_visible_text_only": sum(text_only),
        "turns_without_thinking_or_text": len(turns) - sum(think) - sum(evid) - sum(text_only),
    }


# ---------------------------------------------------------------- reference


def _build_reference(rows: list[dict], source_files: list[str]) -> dict:
    meta = next((row for row in rows if row.get("event") == "meta"), {})
    model_turns = [row for row in rows if row.get("event") == "model_turn"]
    end = next((row for row in rows if row.get("event") == "end"), {})
    done = next((row for row in rows if row.get("event") == "done"), None)

    outcomes: dict[tuple[int, int], dict] = {}
    turnless_outcomes = []
    for row in rows:
        if row.get("event") not in _CALL_OUTCOME_EVENTS:
            continue
        turn, call_index = row.get("turn"), row.get("call_index")
        if isinstance(turn, int) and isinstance(call_index, int):
            outcomes[(turn, call_index)] = row
        else:
            turnless_outcomes.append(row)

    turns = []
    previous_ids: list[str] = []
    seen_turn_numbers = set()
    for record in model_turns:
        turn = record.get("turn")
        if not isinstance(turn, int) or turn in seen_turn_numbers:
            raise TurnsViewError(f"model_turn without a unique integer turn: {turn!r}")
        seen_turn_numbers.add(turn)
        message = record.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        calls = []
        for call_index, raw in enumerate(raw_calls):
            function = (raw or {}).get("function") or {}
            outcome = outcomes.pop((turn, call_index), None)
            args = outcome.get("args") if outcome is not None else None
            if args is None:
                try:
                    args = json.loads(str(function.get("arguments") or "{}"))
                except json.JSONDecodeError:
                    args = None
            call = {
                "id": f"t{turn}.c{call_index}",
                "provider_call_id": raw.get("id"),
                "tool": function.get("name"),
                "args": args,
                "dispatched": outcome is not None,
                "result": outcome.get("model_result", outcome.get("result"))
                if outcome is not None else None,
            }
            if outcome is not None:
                call["outcome_event"] = outcome.get("event")
                for key in ("charged", "step", "sim_step_start", "sim_step_end", "failure"):
                    if key in outcome:
                        call[key] = outcome[key]
                if outcome.get("event") == "done":
                    call["result"] = {"report": outcome.get("report")}
            calls.append(call)
        thinking = record.get("reasoning_content")
        thinking = thinking if isinstance(thinking, str) and thinking.strip() else None
        entry = {
            "turn": turn,
            "agent_id": DEFAULT_AGENT_ID,
            "observations_in": list(previous_ids),
            "thinking": thinking,
            "text": _message_text(message.get("content")),
            "tool_calls": calls,
            "usage": record.get("usage"),
            "latency_s": record.get("model_latency_s"),
            "stop_reason": record.get("stop_reason"),
            "provider_context_retry": bool(record.get("context_retry")),
        }
        if thinking is None:
            # Some providers never return the chain of thought (OpenAI's Responses API returns
            # only an optional summary, absent for short thoughts) while the spend is still
            # accounted. Token evidence keeps "the model thought here" distinguishable from
            # "the model did not think here".
            spent = _reasoning_tokens(record.get("usage"))
            if spent:
                entry["thinking_evidence"] = {
                    "kind": "withheld_by_provider", "reasoning_tokens": spent}
        turns.append(entry)
        # Every tool_call receives SOME answer message (the wire protocol requires it — an
        # undispatched call still gets a budget/cancellation payload), so every call id is an
        # observation the next turn consumed.
        previous_ids = [call["id"] for call in calls]

    if len(turns) != len(model_turns):
        raise TurnsViewError(
            f"rendered {len(turns)} turns from {len(model_turns)} model_turn events")

    scaffold = meta.get("scaffold") or {}
    return {
        "schema_version": TURNS_VIEW_SCHEMA,
        "source": {"kind": "reference", "files": source_files},
        "agents": [{
            "agent_id": DEFAULT_AGENT_ID,
            "scaffold": {key: scaffold.get(key) for key in ("name", "version", "model")
                         if key in scaffold},
        }],
        "task": meta.get("task"),
        "turns": turns,
        "final": {
            "done_report": done.get("report") if done is not None else None,
            "failure": end.get("failure"),
            "terminal_events": [row.get("event") for row in rows
                                if row.get("event") not in _NON_TERMINAL_EVENTS],
            "context_retry_count": end.get("context_retry_count"),
            "text_compaction_count": end.get("text_compaction_count"),
        },
        "coverage": {
            "model_turn_events": len(model_turns),
            "turns_rendered": len(turns),
            **_thinking_summary(turns),
            "call_outcomes_unattached": sorted(
                f"t{turn}.c{index}" for turn, index in outcomes),
            "call_outcomes_without_turn": [row.get("event") for row in turnless_outcomes],
        },
    }


# ------------------------------------------------------------------- vendor


def _base_tool_name(name: str) -> str:
    return name.rsplit("__", 1)[-1] if "__" in name else name


def _build_vendor(vendor_rows: list[dict], server_rows: list[dict],
                  source_files: list[str]) -> dict:
    init = next((row for row in vendor_rows if row.get("event") == "init"), {})
    result = next((row for row in vendor_rows if row.get("event") == "result"), {})
    assistants = [row for row in vendor_rows if row.get("event") == "assistant"]
    result_meta = {row.get("tool_use_id"): row for row in vendor_rows
                   if row.get("event") == "tool_result"}

    server_meta = next((row for row in server_rows if row.get("event") == "meta"), {})
    server_end = next((row for row in server_rows if row.get("event") == "end"), {})
    # Server outcomes in dispatch order. The MCP host serializes dispatch (one worker thread owns
    # the sim), and the CLI answers each tool_use before issuing the next, so stream order and
    # dispatch order agree; the name check below turns any violation into a recorded divergence
    # instead of a misattribution.
    server_queue = [row for row in server_rows
                    if row.get("event") in ("tool", "limit", "done", "episode_fatal",
                                            "unintended_collision")]
    queue_pos = 0
    join_diverged_at = None

    # Thinking-token telemetry attribution: `thinking_tokens` events appear in stream order
    # while an assistant message is being generated, so the deltas accumulated since the
    # previous assistant event belong to the next one.
    tokens_before_assistant: dict[int, int] = {}
    pending_tokens, assistant_seen = 0, 0
    for row in vendor_rows:
        event = row.get("event")
        if event == "thinking_tokens":
            delta = row.get("estimated_tokens_delta")
            if isinstance(delta, int) and delta > 0:
                pending_tokens += delta
        elif event == "assistant":
            assistant_seen += 1
            tokens_before_assistant[assistant_seen] = pending_tokens
            pending_tokens = 0
    # Trailing deltas (tokens streamed for the final message after its assistant event was
    # normalized) attach to the last turn rather than vanish.
    if pending_tokens and assistant_seen:
        tokens_before_assistant[assistant_seen] = \
            tokens_before_assistant.get(assistant_seen, 0) + pending_tokens

    turns = []
    previous_ids: list[str] = []
    for turn_number, record in enumerate(assistants, start=1):
        thinking_parts, text_parts, calls = [], [], []
        thinking_blocks = 0
        for block in record.get("content") or []:
            kind = block.get("type")
            if kind == "thinking":
                thinking_blocks += 1
                if str(block.get("text") or "").strip():
                    thinking_parts.append(str(block["text"]))
            elif kind == "text" and str(block.get("text") or ""):
                text_parts.append(str(block["text"]))
            elif kind == "tool_use":
                name = str(block.get("name") or "")
                call = {
                    "id": f"t{turn_number}.c{len(calls)}",
                    "provider_call_id": block.get("id"),
                    "tool": name,
                    "args": block.get("input"),
                    "dispatched": False,
                    "result": None,
                }
                # Codex's normalizer preserves bare MCP tool names. Restrict this form to
                # its declared stream and require literal argument equality before joining.
                bare_mcp = init.get("cli") == "codex" and name in server_meta.get("tools", [])
                if ((name.startswith("mcp__") and len(name.split("__")) >= 3)
                        or bare_mcp) and join_diverged_at is None:
                    if queue_pos < len(server_queue):
                        head = server_queue[queue_pos]
                        head_tool = head.get("tool") or (
                            "done" if head.get("event") == "done" else None)
                        expected_args = head.get("args", head.get("report")
                                                 if head.get("event") == "done" else None)
                        args_match = not bare_mcp or expected_args == block.get("input")
                        if head_tool == _base_tool_name(name) and args_match:
                            queue_pos += 1
                            call["dispatched"] = True
                            call["outcome_event"] = head.get("event")
                            call["result"] = (
                                {"report": head.get("report")}
                                if head.get("event") == "done"
                                else head.get("model_result", head.get("result")))
                            for key in ("charged", "step", "sim_step_start",
                                        "sim_step_end", "failure"):
                                if key in head:
                                    call[key] = head[key]
                        else:
                            join_diverged_at = call["id"]
                meta_row = result_meta.get(block.get("id"))
                if meta_row is not None:
                    call["result_summary"] = {
                        "is_error": meta_row.get("is_error"),
                        "image_count": meta_row.get("image_count"),
                    }
                calls.append(call)
        entry = {
            "turn": turn_number,
            "agent_id": DEFAULT_AGENT_ID,
            "observations_in": list(previous_ids),
            "thinking": "\n\n".join(thinking_parts) or None,
            "text": "\n".join(text_parts) or None,
            "tool_calls": calls,
            "usage": record.get("usage") or None,
            "latency_s": None,
            "stop_reason": None,
            "provider_context_retry": False,
        }
        if entry["thinking"] is None and (thinking_blocks or
                                          tokens_before_assistant.get(turn_number)):
            # The vendor stream carries thinking blocks and token telemetry but every
            # thinking/thinking_delta body is "" (measured across all 85 release episodes:
            # 2,914 empty blocks). Whether the provider omitted the text (Anthropic's
            # `thinking.display` defaults to `omitted` on these models -- the same mechanism
            # once hit the reference track until its registry flipped the display switch) or
            # the CLI dropped it cannot be determined from the artifacts, so the kind says
            # only where the words were lost relative to us: upstream.
            entry["thinking_evidence"] = {
                "kind": "withheld_upstream",
                "thinking_blocks": thinking_blocks,
                "estimated_tokens": tokens_before_assistant.get(turn_number, 0),
            }
        turns.append(entry)
        previous_ids = [call["id"] for call in calls]

    if len(turns) != len(assistants):
        raise TurnsViewError(
            f"rendered {len(turns)} turns from {len(assistants)} assistant events")

    return {
        "schema_version": TURNS_VIEW_SCHEMA,
        "source": {"kind": "vendor", "files": source_files, "turn_unit": "assistant_event"},
        "agents": [{
            "agent_id": DEFAULT_AGENT_ID,
            "runtime": {
                "model": init.get("model"),
                "claude_code_version": init.get("claude_code_version"),
                "cli": init.get("cli", "claude"),
                "cli_version": init.get("cli_version", init.get("claude_code_version")),
            },
        }],
        "task": server_meta.get("task"),
        "turns": turns,
        "final": {
            "done_report": next(
                (row.get("report") for row in server_rows if row.get("event") == "done"), None),
            "failure": server_end.get("failure"),
            "terminal_events": [result.get("subtype")] if result else [],
            "num_turns_reported": result.get("num_turns"),
        },
        "coverage": {
            "assistant_events": len(assistants),
            "turns_rendered": len(turns),
            **_thinking_summary(turns),
            "server_outcomes": len(server_queue),
            "server_outcomes_joined": queue_pos,
            "join_diverged_at": join_diverged_at,
        },
    }


# ----------------------------------------------------------- interface-only


def _build_interface_only(server_rows: list[dict], source_files: list[str]) -> dict:
    """No agent-side transcript: turn boundaries are unknown and are not invented."""
    meta = next((row for row in server_rows if row.get("event") == "meta"), {})
    end = next((row for row in server_rows if row.get("event") == "end"), {})
    calls = []
    for row in server_rows:
        if row.get("event") not in ("tool", "limit", "done", "episode_fatal",
                                    "unintended_collision"):
            continue
        call = {
            "id": f"c{len(calls)}",
            "tool": row.get("tool") or ("done" if row.get("event") == "done" else None),
            "args": row.get("args"),
            "dispatched": True,
            "outcome_event": row.get("event"),
            "result": ({"report": row.get("report")} if row.get("event") == "done"
                       else row.get("model_result", row.get("result"))),
        }
        for key in ("charged", "step", "sim_step_start", "sim_step_end", "failure"):
            if key in row:
                call[key] = row[key]
        calls.append(call)
    return {
        "schema_version": TURNS_VIEW_SCHEMA,
        "source": {"kind": "interface_only", "files": source_files},
        "agents": [{"agent_id": DEFAULT_AGENT_ID}],
        "task": meta.get("task"),
        "turns": None,
        "turns_unavailable_reason": "no agent-side transcript; turn boundaries unknown",
        "calls": calls,
        "final": {
            "done_report": next(
                (row.get("report") for row in server_rows if row.get("event") == "done"), None),
            "failure": end.get("failure"),
        },
        "coverage": {"server_outcomes": len(calls)},
    }


# -------------------------------------------------------------------- entry


def build_turns_view(run_dir: Path) -> dict:
    """Project whatever transcripts exist in ``run_dir`` into the turns-view shape."""
    run_dir = Path(run_dir)
    reference = run_dir / "reference_transcript.jsonl"
    plain = run_dir / "transcript.jsonl"
    vendor = run_dir / "vendor_transcript.jsonl"

    def is_reference_shaped(path: Path) -> bool:
        return path.is_file() and any(
            row.get("event") == "model_turn" for row in _read_jsonl(path))

    if is_reference_shaped(reference):
        return _bounded(_build_reference(_read_jsonl(reference), [reference.name]))
    if is_reference_shaped(plain):
        return _bounded(_build_reference(_read_jsonl(plain), [plain.name]))
    if vendor.is_file():
        server_rows = _read_jsonl(plain) if plain.is_file() else []
        files = [vendor.name] + ([plain.name] if plain.is_file() else [])
        return _bounded(_build_vendor(_read_jsonl(vendor), server_rows, files))
    if plain.is_file():
        return _bounded(_build_interface_only(_read_jsonl(plain), [plain.name]))
    raise TurnsViewError(f"no transcript found in {run_dir}")


def write_turns_view(run_dir: Path) -> Path:
    """Build and write ``turns.v1.json``. Call ONLY before the attempt is sealed."""
    run_dir = Path(run_dir)
    view = build_turns_view(run_dir)
    path = run_dir / TURNS_VIEW_NAME
    path.write_text(
        json.dumps(view, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


__all__ = [
    "TURNS_VIEW_NAME",
    "TURNS_VIEW_SCHEMA",
    "TurnsViewError",
    "build_turns_view",
    "write_turns_view",
]
