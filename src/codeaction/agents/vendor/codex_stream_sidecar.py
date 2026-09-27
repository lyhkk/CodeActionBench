"""Normalize Codex ``exec --json`` into the SAME audit artifacts the Claude seat writes.

Codex remains responsible for context, streaming assembly, and MCP dispatch.  This module reads
only the completed events the CLI emitted and writes bounded evidence: ``vendor_transcript.jsonl``,
``vendor_usage.json`` and ``vendor_runtime_attestation.json``, with the field names the reporting
layer already consumes.  Provider signatures and image bytes are deliberately excluded.

WHY THIS IS NOT A COPY OF ``stream_sidecar``.  The two CLIs do not describe an episode the same
way, and three of the differences change what may be CLAIMED rather than merely how it is parsed:

1. There is no init event.  Claude opens with ``system/init`` carrying the model, the CLI version,
   the tool roster and the MCP server list; Codex opens with a bare ``thread.started``.  So the
   roster check that the Claude sidecar performs against the stream is performed HERE against the
   episode server's own ``result.json`` -- our side of the wall, and therefore evidence that holds
   for any CLI.  What the model says about its own tools is self-report and is never read.
2. No event names the model.  Claude echoes it on init and on every assistant message.  Codex
   names it nowhere, so this sidecar records the model as launch-declared and does NOT assert it
   was observed.  Writing ``observed_assistant_models: [expected]`` here would have manufactured
   agreement out of an argv string.
3. A tool call and its result share one record.  Claude emits ``tool_use`` and a later
   ``tool_result`` joined by id, so an unreturned call is visible as an unmatched id; Codex emits
   a single ``item.completed`` holding both, so an unreturned call is simply absent.  The
   normalized transcript still splits them into the two-row shape the reporting layer expects,
   and the call count is cross-checked against the server's own count instead.

Unknown ``item.type`` values are recorded and fail the attestation.  A stack that grows a new
native tool surface in a point release must not be able to use it silently.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from codeaction.agents.vendor.clis import CODEX


SCHEMA_VERSION = "1.0"

# Item types this normalizer understands.  Everything else fails the attestation by name.
KNOWN_ITEM_TYPES = frozenset({"agent_message", "reasoning", "mcp_tool_call", "error"})

# Item types that would mean a banned capability was exercised.  Listed explicitly so the failure
# names the breach rather than a generic "unexpected item".  This is behavioural evidence: it
# shows a forbidden tool was USED, which is a stronger and narrower claim than showing none was
# offered -- the offered roster is established by the launch config and the server surface.
FORBIDDEN_ITEM_TYPES = {
    "command_execution": "shell_execution",
    "local_shell_call": "shell_execution",
    "unified_exec": "shell_execution",
    "file_change": "filesystem_read",
    "patch_apply": "filesystem_read",
    "web_search": "web_retrieval",
    "task_delegation": "subagent_delegation",
}

# Text Codex inserts when IT truncates a tool result (measured in the 0.154.0 binary). A
# result the model saw only part of is a result the benchmark did not deliver.
TRUNCATION_MARKERS = ("Warning: truncated output (original token count:",
                      "<truncated omitted_approx_tokens=")
_TERMINAL_OK = "turn.completed"
_TERMINAL_FAIL = ("turn.failed", "thread.failed")


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows, invalid = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows, invalid


def _image_count(value: Any) -> int:
    if isinstance(value, dict):
        return (1 if value.get("type") == "image" else 0) + sum(
            _image_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_image_count(item) for item in value)
    return 0


def _reasoning_text(item: dict[str, Any]) -> str:
    """Codex reasoning text, from whichever field this version carries it in.

    The shape is read, never assumed: a reasoning item that carries no text at all is counted as
    an invisible block rather than silently becoming an empty string, because the difference
    between 'thought and told us' and 'thought and did not' is the whole point of the rung.
    """
    for key in ("text", "content", "summary"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            parts = [
                part.get("text") for part in value
                if isinstance(part, dict) and isinstance(part.get("text"), str)]
            joined = "\n".join(part for part in parts if part and part.strip())
            if joined.strip():
                return joined
    return ""


def _server_status(output_dir: Path) -> str | None:
    try:
        result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
        return str(result["stats"]["status"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _server_tool_surface(output_dir: Path) -> tuple[list[str], str | None, int | None]:
    """The roster the EPISODE SERVER says it delivered, plus its own call count.

    Read from result.json rather than from the stream, so the claim rests on our side of the
    wall. Returns empty/None when the server wrote nothing, which is itself an attestation error
    raised by the caller -- never a silent pass.
    """
    try:
        result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return [], None, None
    if not isinstance(result, dict):
        return [], None, None
    surface = result.get("tool_surface") if isinstance(result.get("tool_surface"), dict) else {}
    names = surface.get("ordered_names")
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    total_calls = stats.get("total_calls")
    return (
        [name for name in names if isinstance(name, str)] if isinstance(names, list) else [],
        surface.get("delivered_sha256") if isinstance(surface.get("delivered_sha256"), str)
        else None,
        total_calls if isinstance(total_calls, int) and not isinstance(total_calls, bool)
        else None,
    )


def _feature_gate(stream_path: Path) -> tuple[dict[str, Any], list[str]]:
    """The in-container feature gate's report, from beside the raw stream.

    Written by codex_feature_gate.py before `codex exec` ran. Missing means the gate never ran,
    which is as disqualifying as a refusal: an episode with no proof of its feature set is an
    episode whose tool surface is unknown.
    """
    path = stream_path.parent / "vendor_feature_gate.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, ["vendor_feature_gate.json is missing: the feature gate never ran"]
    if not isinstance(report, dict):
        return {}, ["vendor_feature_gate.json is not an object"]
    errors = []
    if report.get("healthy") is not True:
        errors.append("feature gate refused the episode: "
                      + "; ".join(str(v) for v in report.get("violations") or ["no detail"]))
    return report, errors


_JS_MCP_REF = re.compile(r"tools\.mcp__codeaction__([A-Za-z0-9_]+)")
_WALL_TIME = re.compile(r"Wall time ([0-9.]+) seconds")


def _codemode_summary(state_dir: Path, output_dir: Path) -> dict[str, Any] | None:
    """What the model ran in the code-mode host, from the session rollout.

    Every action a code-mode model takes is a `custom_tool_call` named `exec` whose input is
    JavaScript; MCP tools appear inside it as `tools.mcp__codeaction__<name>(...)`. The JSON
    stream reports only the MCP calls that resulted, so this is the only place where an exec
    that called no tool -- pure computation, introspection of ALL_TOOLS -- is visible, and the
    only place where the HOST'S truncation of an exec output can be seen. Each exec is written
    to vendor_codemode.jsonl (the JavaScript is model output and is evidence, not vendor prose).
    """
    rollouts = sorted(state_dir.rglob("rollout-*.jsonl")) if state_dir.is_dir() else []
    if not rollouts:
        return None
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for path in rollouts:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("type") != "response_item":
                continue
            payload = row.get("payload") or {}
            kind = payload.get("type")
            call_id = str(payload.get("call_id") or "")
            if kind == "custom_tool_call":
                calls[call_id] = {"name": payload.get("name"), "js": str(payload.get("input") or ""),
                                  "output": ""}
                order.append(call_id)
            elif kind == "custom_tool_call_output" and call_id in calls:
                calls[call_id]["output"] = "".join(
                    str(block.get("text") or "") for block in (payload.get("output") or [])
                    if isinstance(block, dict))
    rows = []
    for index, call_id in enumerate(order):
        call = calls[call_id]
        out = call["output"]
        wall = _WALL_TIME.search(out)
        rows.append({
            "index": index,
            "call_id": call_id,
            "name": call["name"],
            "mcp_tools": _JS_MCP_REF.findall(call["js"]),
            "status": ("completed" if out.startswith("Script completed")
                       else "failed" if out.startswith("Script failed") else "unknown"),
            "host_wall_s": float(wall.group(1)) if wall else None,
            "truncated": any(marker in out for marker in TRUNCATION_MARKERS),
            "js": call["js"],
            "output_head": out[:400],
        })
    with (output_dir / "vendor_codemode.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "exec_calls": len(rows),
        "exec_with_mcp": sum(1 for r in rows if r["mcp_tools"]),
        "exec_without_mcp": sum(1 for r in rows if not r["mcp_tools"]),
        "exec_failed": sum(1 for r in rows if r["status"] == "failed"),
        # Two different things. A truncated exec that referenced a benchmark tool means the model
        # saw part of a RESULT THE BENCHMARK DELIVERED -- actionable, since the server's own
        # result cap exists to keep results under the host's limit. A truncated exec that called
        # no tool is the model's own introspection (text(ALL_TOOLS) and the like) hitting a cap
        # we cannot move: measured, the served truncation_policy keeps ~10,000 tokens whatever
        # tool_output_token_limit says (5000, 25000 and 60000 gave byte-identical output).
        "exec_truncated_by_host": sum(1 for r in rows if r["truncated"]),
        "benchmark_result_truncated": sum(1 for r in rows if r["truncated"] and r["mcp_tools"]),
        "introspection_truncated": sum(1 for r in rows if r["truncated"] and not r["mcp_tools"]),
        "mcp_references": sum(len(r["mcp_tools"]) for r in rows),
        "host_wall_s": round(sum(r["host_wall_s"] or 0.0 for r in rows), 1),
        "record": "vendor_codemode.jsonl",
    }


def _rollout_summary(state_dir: Path) -> dict[str, Any] | None:
    """What the throwaway home left behind: file inventory plus a type census of any JSONL.

    Presence is required; shape is only described. The rollout is the sole trace of code-mode
    execution and its format is the vendor's, so the first episodes record it before anything
    parses it in detail.
    """
    if not state_dir.is_dir():
        return None
    files = sorted(p for p in state_dir.rglob("*") if p.is_file())
    census: dict[str, int] = {}
    for path in files:
        if path.suffix != ".jsonl":
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                census["<invalid>"] = census.get("<invalid>", 0) + 1
                continue
            kind = str(row.get("type") or (row.get("payload") or {}).get("type") or "?") \
                if isinstance(row, dict) else "?"
            census[kind] = census.get(kind, 0) + 1
    return {
        "files": [{"path": str(p.relative_to(state_dir)), "bytes": p.stat().st_size}
                  for p in files],
        "jsonl_row_types": census,
    }


def _bare_tool_name(name: Any) -> str:
    """`mcp__codeaction__capture_head` and `capture_head` are the same tool, named twice."""
    text = str(name or "")
    return text.rsplit("__", 1)[-1] if text.startswith("mcp__") else text


def _normalized_usage(usage: dict[str, Any], turns: int) -> dict[str, Any]:
    """Codex usage in the shape the reporting layer already reads.

    `total_cost_usd` is null, not zero: Codex reports no currency, and a zero would read as a free
    episode. Uncached and cached input are kept apart because they are priced apart and move in
    opposite directions when the frame budget changes.
    """
    numbers = {
        key: usage.get(key) for key in (
            "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
            "output_tokens", "reasoning_output_tokens")
        if isinstance(usage.get(key), int) and not isinstance(usage.get(key), bool)
    }
    cached = numbers.get("cached_input_tokens", 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "duration_ms": None,
        "duration_api_ms": None,
        "num_turns": turns,
        "total_cost_usd": None,
        "usage": {
            "input_tokens": numbers.get("input_tokens"),
            "output_tokens": numbers.get("output_tokens"),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": numbers.get("cache_write_input_tokens", 0),
            "reasoning_output_tokens": numbers.get("reasoning_output_tokens"),
        },
        "model_usage": {},
        "vendor_usage_raw": numbers,
    }


def normalize_stream(
    stream_path: Path,
    output_dir: Path,
    *,
    expected_model: str,
    expected_effort: str,
    require_conformance: bool = False,
) -> dict[str, Any]:
    """Write vendor audit artifacts and return the runtime attestation.

    The signature is identical to the Claude seat's on purpose: the controller selects a sidecar
    by vendor and calls it the same way, so a third stack is a module reference and not a new
    branch at the call site.
    """
    rows, invalid_lines = _read_rows(stream_path)
    normalized: list[dict[str, Any]] = []
    thread_rows, terminal_rows, failed_rows = [], [], []
    turns_started = 0
    thinking_blocks = visible_thinking_blocks = image_results = 0
    tool_calls: list[dict[str, Any]] = []
    observed_servers: set[str] = set()
    unexpected_item_types: set[str] = set()
    forbidden_used: dict[str, str] = {}
    error_items = 0
    truncated_results = 0

    for index, row in enumerate(rows):
        kind = row.get("type")
        if kind == "thread.started":
            thread_rows.append(row)
            normalized.append({
                "event": "init",
                "model": None,
                "cli": CODEX.cli_label,
                "cli_version": CODEX.version,
                "tools": [],
                "mcp_servers": [],
                "session_id": row.get("thread_id"),
                "roster_source": "server",
                "model_evidence": CODEX.disposition("model_selection").evidence,
            })
        elif kind == "turn.started":
            turns_started += 1
        elif kind in ("item.started", "item.created", "item.updated"):
            continue                                    # partials; only completions are evidence
        elif kind == "item.completed":
            item = row.get("item") if isinstance(row.get("item"), dict) else {}
            item_type = str(item.get("type") or "")
            if item_type in FORBIDDEN_ITEM_TYPES:
                forbidden_used[item_type] = FORBIDDEN_ITEM_TYPES[item_type]
                continue
            if item_type not in KNOWN_ITEM_TYPES:
                unexpected_item_types.add(item_type or "<missing>")
                continue
            if item_type == "error":
                # The CLI's own runtime error (e.g. "code-mode host is disabled"). Counted and
                # fatal to the attestation; the message is vendor prose and stays in the raw
                # stream rather than being copied into evidence.
                error_items += 1
                continue
            if item_type == "agent_message":
                normalized.append({
                    "event": "assistant",
                    "model": None,
                    "content": [{"type": "text", "text": str(item.get("text") or "")}],
                    "usage": {},
                })
            elif item_type == "reasoning":
                thinking_blocks += 1
                text = _reasoning_text(item)
                if text.strip():
                    visible_thinking_blocks += 1
                normalized.append({
                    "event": "assistant",
                    "model": None,
                    "content": [{"type": "thinking", "text": text}],
                    "usage": {},
                })
            else:                                       # mcp_tool_call
                server = str(item.get("server") or "")
                observed_servers.add(server)
                call_id = str(item.get("id") or f"item_{index}")
                name = _bare_tool_name(item.get("tool"))
                call = {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": item.get("arguments")
                    if isinstance(item.get("arguments"), dict) else {},
                }
                tool_calls.append(call)
                normalized.append({
                    "event": "assistant", "model": None, "content": [call], "usage": {}})
                result = item.get("result") if isinstance(item.get("result"), dict) else {}
                result_text = "".join(
                    str(block.get("text") or "") for block in (result.get("content") or [])
                    if isinstance(block, dict) and block.get("type") == "text")
                if any(marker in result_text for marker in TRUNCATION_MARKERS):
                    truncated_results += 1
                images = _image_count(result.get("content"))
                image_results += images
                normalized.append({
                    "event": "tool_result",
                    "tool_use_id": call_id,
                    "is_error": bool(result.get("isError", False)),
                    "image_count": images,
                })
        elif kind == _TERMINAL_OK:
            terminal_rows.append(row)
            normalized.append({
                "event": "result",
                "subtype": "success",
                "is_error": False,
                "api_error_status": None,
                "num_turns": turns_started,
                "terminal_reason": _TERMINAL_OK,
            })
        elif kind in _TERMINAL_FAIL:
            failed_rows.append(row)
            error = row.get("error") if isinstance(row.get("error"), dict) else {}
            normalized.append({
                "event": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                # The message text is not retained; only a machine-usable code, for the same
                # reason normalize_exit refuses to classify on vendor prose.
                "api_error_status": error.get("code") or error.get("type"),
                "num_turns": turns_started,
                "terminal_reason": kind,
            })

    terminal = terminal_rows[-1] if terminal_rows else {}
    usage = terminal.get("usage") if isinstance(terminal.get("usage"), dict) else {}
    reasoning_tokens = usage.get("reasoning_output_tokens")
    thinking_tokens_observed = (
        reasoning_tokens if isinstance(reasoning_tokens, int)
        and not isinstance(reasoning_tokens, bool) and reasoning_tokens >= 0 else 0)
    delivered_names, delivered_sha256, server_calls = _server_tool_surface(output_dir)
    gate_report, gate_errors = _feature_gate(stream_path)
    delivered = {_bare_tool_name(name) for name in delivered_names}
    capture_calls = [call for call in tool_calls if call["name"] == "capture_head"]
    off_surface = sorted({
        call["name"] for call in tool_calls
        if delivered and call["name"] not in delivered})

    errors = list(gate_errors)
    if len(thread_rows) != 1:
        errors.append(f"expected one thread.started event, found {len(thread_rows)}")
    if len(terminal_rows) + len(failed_rows) != 1:
        errors.append(
            f"expected one terminal event, found "
            f"{len(terminal_rows) + len(failed_rows)}")
    if failed_rows:
        errors.append("Codex reported a failed turn")
    if invalid_lines:
        errors.append(f"stream contains {invalid_lines} invalid JSON lines")
    if forbidden_used:
        errors.append(
            "forbidden capability exercised: "
            + ", ".join(f"{item}->{cap}" for item, cap in sorted(forbidden_used.items())))
    if unexpected_item_types:
        errors.append(f"unrecognized stream item types: {sorted(unexpected_item_types)}")
    if error_items:
        errors.append(f"Codex emitted {error_items} runtime error item(s); see the raw stream")
    if truncated_results:
        errors.append(f"the CLI truncated {truncated_results} tool result(s) before the model "
                      f"saw them")
    rollout = _rollout_summary(stream_path.parent / "codex_home_state")
    if rollout is None:
        errors.append("codex_home_state is missing: the session rollout was not copied out")
    codemode = _codemode_summary(stream_path.parent / "codex_home_state", output_dir)
    if codemode is None:
        errors.append("no session rollout found: code-mode execution is unobservable")
    elif codemode["benchmark_result_truncated"]:
        # Only a truncated BENCHMARK result fails the episode. The host's ~10k-token cap on exec
        # output is served by the vendor and not configurable, so the benchmark's own result cap
        # is what must keep every delivered result under it; this is the check that it did.
        errors.append(f"the code-mode host truncated {codemode['benchmark_result_truncated']} "
                      f"benchmark tool result(s) before the model saw them")
    if not delivered_names:
        errors.append("episode server recorded no delivered tool surface to check against")
    if off_surface:
        errors.append(f"tools called that the server never delivered: {off_surface}")
    if tool_calls and observed_servers - {"codeaction"}:
        errors.append(f"MCP servers other than codeaction were called: "
                      f"{sorted(observed_servers - {'codeaction'})}")
    if tool_calls and "codeaction" not in observed_servers:
        errors.append("no call reached the codeaction MCP server")
    # Codex names no model anywhere in the stream, so there is no observation to compare against
    # the expectation. The claim is recorded as launch-declared below and is NOT asserted here;
    # an equality check would only ever compare argv with itself.
    # One trailing `done` after the server has already finalized on a budget is not a lost
    # call: the server stops counting at finalize and the agent's closing call lands after it.
    # Measured on a physical_time_budget_exhausted episode -- 38 in the stream, 37 recorded,
    # the 38th being `done`. Anything else that exceeds the server's count is still an error.
    trailing_done = (
        server_calls is not None
        and len(tool_calls) == server_calls + 1
        and tool_calls[-1]["name"] == "done"
        and _server_status(output_dir) not in (None, "done"))
    if server_calls is not None and len(tool_calls) > server_calls and not trailing_done:
        errors.append(
            f"stream shows {len(tool_calls)} tool calls but the server recorded {server_calls}")
    if require_conformance:
        if thinking_blocks < 1 or thinking_tokens_observed < 1:
            errors.append("conformance stream contains no measured thinking evidence")
        if thinking_blocks and not visible_thinking_blocks:
            errors.append(
                "reasoning items carried no text: model_reasoning_summary was left at the "
                "model's default of none")
        if len(capture_calls) != 1:
            errors.append(f"expected one capture_head call, found {len(capture_calls)}")
        if image_results < 1:
            errors.append("capture_head produced no image-bearing tool result")

    attestation = {
        "schema_version": SCHEMA_VERSION,
        "healthy": not errors,
        "errors": errors,
        "expected_model": expected_model,
        # Empty, and empty on purpose: nothing in a Codex stream names the model. The controller
        # stamps the model into the run identity from argv; this file must not imply the stream
        # agreed with it.
        "observed_assistant_models": [],
        "observed_usage_models": [],
        "expected_effort": expected_effort,
        "effort_control": "cli-config-flag",
        "thinking_blocks": thinking_blocks,
        "visible_thinking_blocks": visible_thinking_blocks,
        "thinking_tokens_observed": thinking_tokens_observed,
        "tool_calls": len(tool_calls),
        "capture_head_calls": len(capture_calls),
        "image_tool_results": image_results,
        "mcp_servers": sorted(name for name in observed_servers if name),
        "unexpected_tools": off_surface,
        "invalid_json_lines": invalid_lines,
        "raw_stream": stream_path.name,
        "raw_stream_retains_provider_blocks": True,
        # ---- vendor-neutral evidence annotations, so a comparison never reads a launch-declared
        # ---- claim as an observed one.
        "vendor_cli": CODEX.cli_label,
        "vendor_cli_version": CODEX.version,
        "tool_discovery": CODEX.tool_discovery,
        "model_evidence": CODEX.disposition("model_selection").evidence,
        "tool_roster_evidence": "server",
        "server_delivered_sha256": delivered_sha256,
        "server_total_calls": server_calls,
        "trailing_done_after_finalize": bool(trailing_done),
        "unexpected_item_types": sorted(unexpected_item_types),
        "forbidden_item_types": sorted(forbidden_used),
        "error_items": error_items,
        "truncated_results": truncated_results,
        "rollout": rollout,
        "codemode": codemode,
        "residual_banned": list(CODEX.residual),
        "feature_gate": {
            key: gate_report.get(key) for key in (
                "healthy", "violations", "enabled_features", "unexpected_enabled",
                "allowlisted_now_disabled", "feature_count", "home_entries_before",
                "home_entries_after", "text")
        } if gate_report else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vendor_transcript.jsonl").open("w", encoding="utf-8") as stream:
        for row in normalized:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "vendor_usage.json").write_text(
        json.dumps(_normalized_usage(usage, turns_started), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (output_dir / "vendor_runtime_attestation.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return attestation


__all__: Iterable[str] = ("normalize_stream", "SCHEMA_VERSION", "KNOWN_ITEM_TYPES",
                          "FORBIDDEN_ITEM_TYPES")
