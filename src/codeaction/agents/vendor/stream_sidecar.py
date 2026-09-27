"""Normalize Claude Code ``stream-json`` without participating in its agent loop.

Claude Code remains responsible for context, streaming assembly, and MCP dispatch.  This module
reads only the complete events the CLI emitted after the fact and writes bounded audit artifacts.
Provider signatures and image bytes are deliberately excluded from the normalized transcript.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
ALLOWED_BUILTIN_TOOLS = frozenset({"ToolSearch"})


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


def _content_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return [block for block in content if isinstance(block, dict)] \
        if isinstance(content, list) else []


def _image_count(value: Any) -> int:
    if isinstance(value, dict):
        return (1 if value.get("type") == "image" else 0) + sum(
            _image_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_image_count(item) for item in value)
    return 0


def _mcp_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def _normalized_usage(result: dict[str, Any]) -> dict[str, Any]:
    model_usage = result.get("modelUsage")
    return {
        "schema_version": SCHEMA_VERSION,
        "duration_ms": result.get("duration_ms"),
        "duration_api_ms": result.get("duration_api_ms"),
        "num_turns": result.get("num_turns"),
        "total_cost_usd": result.get("total_cost_usd"),
        "usage": result.get("usage") if isinstance(result.get("usage"), dict) else {},
        "model_usage": model_usage if isinstance(model_usage, dict) else {},
    }


def normalize_stream(
    stream_path: Path,
    output_dir: Path,
    *,
    expected_model: str,
    expected_effort: str,
    require_conformance: bool = False,
) -> dict[str, Any]:
    """Write vendor audit artifacts and return the runtime attestation."""
    rows, invalid_lines = _read_rows(stream_path)
    normalized: list[dict[str, Any]] = []
    init_rows, result_rows = [], []
    thinking_blocks = visible_thinking_blocks = image_results = 0
    thinking_tokens_observed = 0
    tool_calls: list[dict[str, Any]] = []
    tool_result_ids: set[str] = set()

    for row in rows:
        kind = row.get("type")
        if kind == "system" and row.get("subtype") == "init":
            init_rows.append(row)
            normalized.append({
                "event": "init",
                "model": row.get("model"),
                "claude_code_version": row.get("claude_code_version"),
                "tools": row.get("tools") if isinstance(row.get("tools"), list) else [],
                "mcp_servers": row.get("mcp_servers")
                if isinstance(row.get("mcp_servers"), list) else [],
                "session_id": row.get("session_id"),
            })
        elif kind == "system" and row.get("subtype") == "thinking_tokens":
            observed = row.get("estimated_tokens")
            if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                thinking_tokens_observed = max(thinking_tokens_observed, observed)
            normalized.append({
                "event": "thinking_tokens",
                "estimated_tokens": observed,
                "estimated_tokens_delta": row.get("estimated_tokens_delta"),
            })
        elif kind == "assistant":
            message = row.get("message")
            blocks_out = []
            for block in _content_blocks(message):
                block_type = block.get("type")
                if block_type == "thinking":
                    thinking_blocks += 1
                    text = str(block.get("thinking") or block.get("text") or "")
                    if text.strip():
                        visible_thinking_blocks += 1
                    blocks_out.append({"type": "thinking", "text": text})
                elif block_type == "text":
                    blocks_out.append({"type": "text", "text": str(block.get("text") or "")})
                elif block_type == "tool_use":
                    call = {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                    }
                    tool_calls.append(call)
                    blocks_out.append(call)
            normalized.append({
                "event": "assistant",
                "model": message.get("model") if isinstance(message, dict) else None,
                "content": blocks_out,
                "usage": message.get("usage") if isinstance(message, dict)
                and isinstance(message.get("usage"), dict) else {},
            })
        elif kind == "user":
            for block in _content_blocks(row.get("message")):
                if block.get("type") != "tool_result":
                    continue
                call_id = block.get("tool_use_id")
                if isinstance(call_id, str):
                    tool_result_ids.add(call_id)
                images = _image_count(block.get("content"))
                image_results += images
                normalized.append({
                    "event": "tool_result",
                    "tool_use_id": call_id,
                    "is_error": bool(block.get("is_error", False)),
                    "image_count": images,
                })
        elif kind == "result":
            result_rows.append(row)
            normalized.append({
                "event": "result",
                "subtype": row.get("subtype"),
                "is_error": row.get("is_error"),
                "api_error_status": row.get("api_error_status"),
                "num_turns": row.get("num_turns"),
                "terminal_reason": row.get("terminal_reason"),
            })

    init = init_rows[-1] if init_rows else {}
    result = result_rows[-1] if result_rows else {}
    tools = init.get("tools") if isinstance(init.get("tools"), list) else []
    unexpected_tools = sorted(
        name for name in tools
        if isinstance(name, str)
        and name not in ALLOWED_BUILTIN_TOOLS
        and not name.startswith("mcp__codeaction__"))
    mcp_names = _mcp_names(init.get("mcp_servers"))
    assistant_models = sorted({
        str(row.get("model")) for row in normalized
        if row.get("event") == "assistant" and row.get("model")
    })
    result_models = sorted((result.get("modelUsage") or {}).keys()) \
        if isinstance(result.get("modelUsage"), dict) else []
    capture_calls = [
        call for call in tool_calls
        if call.get("name") == "mcp__codeaction__capture_head"
        or str(call.get("name") or "").endswith("__capture_head")]
    unmatched_calls = sorted(
        str(call.get("id")) for call in tool_calls
        if isinstance(call.get("id"), str) and call["id"] not in tool_result_ids)

    errors = []
    if len(init_rows) != 1:
        errors.append(f"expected one init event, found {len(init_rows)}")
    if len(result_rows) != 1:
        errors.append(f"expected one result event, found {len(result_rows)}")
    if invalid_lines:
        errors.append(f"stream contains {invalid_lines} invalid JSON lines")
    if init.get("model") != expected_model and expected_model not in assistant_models:
        errors.append(f"expected model {expected_model!r} was not observed")
    if unexpected_tools:
        errors.append(f"unexpected built-in tools: {unexpected_tools}")
    if "codeaction" not in mcp_names:
        errors.append("codeaction MCP server was not observed")
    if unmatched_calls:
        errors.append(f"tool calls without results: {unmatched_calls}")
    if result.get("is_error") is True or result.get("subtype") != "success":
        errors.append("Claude Code result was not successful")
    if require_conformance:
        if thinking_blocks < 1 or thinking_tokens_observed < 1:
            errors.append("conformance stream contains no measured thinking evidence")
        if len(capture_calls) != 1:
            errors.append(f"expected one capture_head call, found {len(capture_calls)}")
        if image_results < 1:
            errors.append("capture_head produced no image-bearing tool result")
        if expected_model not in result_models:
            errors.append("expected model is absent from modelUsage")

    attestation = {
        "schema_version": SCHEMA_VERSION,
        "healthy": not errors,
        "errors": errors,
        "expected_model": expected_model,
        "observed_assistant_models": assistant_models,
        "observed_usage_models": result_models,
        "expected_effort": expected_effort,
        "effort_control": "cli-flag-and-environment",
        "thinking_blocks": thinking_blocks,
        "visible_thinking_blocks": visible_thinking_blocks,
        "thinking_tokens_observed": thinking_tokens_observed,
        "tool_calls": len(tool_calls),
        "capture_head_calls": len(capture_calls),
        "image_tool_results": image_results,
        "mcp_servers": mcp_names,
        "unexpected_tools": unexpected_tools,
        "invalid_json_lines": invalid_lines,
        "raw_stream": stream_path.name,
        "raw_stream_retains_provider_blocks": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vendor_transcript.jsonl").open("w", encoding="utf-8") as stream:
        for row in normalized:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "vendor_usage.json").write_text(
        json.dumps(_normalized_usage(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (output_dir / "vendor_runtime_attestation.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return attestation


__all__: Iterable[str] = ("normalize_stream", "SCHEMA_VERSION")
