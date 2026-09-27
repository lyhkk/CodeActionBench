#!/usr/bin/env python3
"""Audit cross-task isolation in a staged reference-scaffold matrix.

The runtime contract already creates a fresh container, provider adapter, and ``messages`` list
for every attempt.  This audit checks the evidence left by two consecutive task runs without
pretending that a transcript can expose a provider's private KV-cache implementation:

* every cell has its own complete controller/agent attestation and transcript;
* provider request IDs do not cross task boundaries;
* the second task's first model turn (before any tool/perception result) is surfaced for review;
* cached input tokens are reported as prefix-computation reuse, not classified as state leakage.

The lexical review is deliberately a warning, not a verdict.  A model may use a word such as
"block" innocently; structural failures and reused provider request IDs are hard failures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


PRIOR_TASK_MARKERS = {
    "blocks_ranking_size": ("block", "blocks", "ranking", "ranked"),
}
OBSERVATION_TOOLS = frozenset({
    "get_world_frame",
    "capture_head",
    "capture_wrist",
    "capture_evidence_views",
    "capture_motion_pair",
})
METRIC_EVIDENCE_TOOLS = frozenset({
    "plane_intersect",
    "probe_contact_along",
    "scale_from_gripper",
    "scale_from_object_size",
    "triangulate_correspondence",
})
PHYSICAL_ACTION_TOOLS = frozenset({
    "aim_camera",
    "close_gripper",
    "move_both_delta",
    "move_delta",
    "open_gripper",
    "probe_contact_along",
    "reach_both_tcp",
    "reach_tcp",
})


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _attempt_dir(run_dir: Path) -> Path | None:
    candidates = sorted(
        child for child in run_dir.glob("attempt-*-seed-*") if child.is_dir())
    return candidates[0] if len(candidates) == 1 else None


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content"):
                    if isinstance(item.get(key), str):
                        parts.append(item[key])
        return "\n".join(parts)
    if isinstance(value, dict):
        return _message_text(value.get("content"))
    return ""


def _first_turn_text(records: Iterable[dict[str, Any]]) -> str:
    for record in records:
        if record.get("event") != "model_turn":
            continue
        message = _message_text(record.get("message"))
        reasoning = str(record.get("reasoning_content") or "")
        return "\n".join(part for part in (reasoning, message) if part).strip()
    return ""


def _request_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(record["provider_request_id"])
        for record in records
        if record.get("event") == "model_turn" and record.get("provider_request_id")
    }


def _cached_tokens(records: Iterable[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        if record.get("event") != "model_turn":
            continue
        usage = record.get("usage") or {}
        for key in ("cached_tokens", "input_cached_tokens", "cache_read_input_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                total += value
                break
    return total


def _marker_hits(text: str, markers: Iterable[str]) -> list[str]:
    return sorted({
        marker for marker in markers
        if re.search(rf"(?<![A-Za-z]){re.escape(marker)}(?![A-Za-z])", text, re.IGNORECASE)
    })


def _tool_trace(records: Iterable[dict[str, Any]]) -> list[str]:
    """Return top-level tools plus sandbox-internal tools in execution order."""
    names: list[str] = []
    for record in records:
        if record.get("event") != "tool" or not isinstance(record.get("tool"), str):
            continue
        name = str(record["tool"])
        names.append(name)
        if name not in ("run_code", "run_program"):
            continue
        result = record.get("result")
        if not isinstance(result, dict):
            result = record.get("model_result")
        trace = result.get("internal_trace") if isinstance(result, dict) else None
        for item in trace if isinstance(trace, list) else ():
            nested = item.get("tool") if isinstance(item, dict) else None
            if isinstance(nested, str):
                names.append(nested)
    return names


def _acquisition_evidence(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    tools = _tool_trace(records)
    observations = [name for name in tools if name in OBSERVATION_TOOLS]
    metric = [name for name in tools if name in METRIC_EVIDENCE_TOOLS]
    first_action_index = next(
        (index for index, name in enumerate(tools) if name in PHYSICAL_ACTION_TOOLS), None)
    initial = tools if first_action_index is None else tools[:first_action_index + 1]
    return {
        "fresh_observation_observed": bool(observations),
        "observation_methods": list(dict.fromkeys(observations)),
        "explicit_metric_evidence_observed": bool(metric),
        "metric_evidence_methods": list(dict.fromkeys(metric)),
        "tools_through_first_physical_action": initial[:24],
        "first_physical_action": (
            tools[first_action_index] if first_action_index is not None else None),
        "interpretation": (
            "No explicit metric-evidence tool does not prove reuse: the model may compute scale "
            "inside run_code or reasoning. Review the recorded code/reasoning when this field is false."),
    }


def _attestation_errors(attempt: Path) -> list[str]:
    errors = []
    for name in (
            "reference_agent_attestation.json",
            "controller_identity_attestation.json",
            "gateway_attestation.json"):
        value = _read_json(attempt / name)
        if value is None:
            errors.append(f"missing or unreadable {name}")
        elif value.get("healthy") is not True:
            errors.append(f"unhealthy {name}: {value.get('error') or value.get('errors')}")
    status = _read_json(attempt / "controller_status.json")
    if status is None or status.get("state") != "complete":
        errors.append("controller_status is not complete")
    return errors


def audit_model(batch_dir: Path, model: str, tasks: tuple[str, str]) -> dict[str, Any]:
    task_records: dict[str, list[dict[str, Any]]] = {}
    task_attempts: dict[str, str] = {}
    errors: list[str] = []
    for task in tasks:
        run_dir = batch_dir / "runs" / model / task
        run = _read_json(run_dir / "run.json")
        if run is None:
            errors.append(f"{task}: missing run.json")
            continue
        if run.get("model") != model or run.get("task_name") != task:
            errors.append(f"{task}: run identity mismatch")
        attempt = _attempt_dir(run_dir)
        if attempt is None:
            errors.append(f"{task}: expected exactly one attempt directory")
            continue
        task_attempts[task] = str(attempt)
        errors.extend(f"{task}: {error}" for error in _attestation_errors(attempt))
        transcript = attempt / "reference_transcript.jsonl"
        records = _read_jsonl(transcript)
        if not records:
            errors.append(f"{task}: missing or empty reference transcript")
            continue
        if records[0].get("event") != "meta":
            errors.append(f"{task}: transcript does not begin with meta")
        if sum(record.get("event") == "meta" for record in records) != 1:
            errors.append(f"{task}: transcript must contain exactly one meta record")
        if not any(record.get("event") == "model_turn" for record in records):
            errors.append(f"{task}: transcript has no model turn")
        task_records[task] = records

    first, second = tasks
    first_ids = _request_ids(task_records.get(first, []))
    second_ids = _request_ids(task_records.get(second, []))
    overlap = sorted(first_ids & second_ids)
    if overlap:
        errors.append(f"provider request IDs cross task boundaries: {overlap[:3]}")

    second_first_turn = _first_turn_text(task_records.get(second, []))
    marker_hits = _marker_hits(second_first_turn, PRIOR_TASK_MARKERS.get(first, ()))
    warnings = []
    if marker_hits:
        warnings.append(
            "second task first turn mentions prior-task marker(s) before any tool result: "
            + ", ".join(marker_hits))

    cached_by_task = {
        task: _cached_tokens(records) for task, records in task_records.items()
    }
    acquisition_by_task = {
        task: _acquisition_evidence(records) for task, records in task_records.items()
    }
    for task, evidence in acquisition_by_task.items():
        if not evidence["fresh_observation_observed"]:
            warnings.append(
                f"{task}: no fresh observation tool was recorded; inspect the transcript")
        if not evidence["explicit_metric_evidence_observed"]:
            warnings.append(
                f"{task}: no explicit metric-evidence tool was recorded; inspect reasoning/code")
    return {
        "model": model,
        "tasks": list(tasks),
        "status": "fail" if errors else "pass_with_review" if warnings else "pass",
        "structural_errors": errors,
        "review_warnings": warnings,
        "provider_request_id_overlap": overlap,
        "cached_tokens_by_task": cached_by_task,
        "acquisition_evidence_by_task": acquisition_by_task,
        "cached_token_interpretation": (
            "Exact-prefix computation may be reused by the provider; this is not evidence that "
            "messages, reasoning, or perception from the prior task entered the new task."),
        "second_task_first_turn_excerpt": second_first_turn[:1200],
        "attempts": task_attempts,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# reference-scaffold Cross-task Isolation Audit",
        "",
        f"- Overall: **{report['status']}**",
        f"- Batch: `{report['batch_dir']}`",
        "- Scope: artifact isolation, request-ID separation, and pre-perception lexical review.",
        "- Limitation: a transcript cannot inspect a provider's private KV tensors; cached-token "
        "counts are reported separately and are not treated as conversation state.",
        "",
        "| Model | Status | Fresh observation (task 1 / task 2) | Explicit metric evidence "
        "(task 1 / task 2) | Cached tokens (task 1 / task 2) | Findings |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in report["models"]:
        tasks = item["tasks"]
        cached = item["cached_tokens_by_task"]
        acquisition = item["acquisition_evidence_by_task"]
        findings = item["structural_errors"] + item["review_warnings"]
        lines.append(
            f"| `{item['model']}` | {item['status']} | "
            f"{acquisition.get(tasks[0], {}).get('fresh_observation_observed', False)} / "
            f"{acquisition.get(tasks[1], {}).get('fresh_observation_observed', False)} | "
            f"{acquisition.get(tasks[0], {}).get('explicit_metric_evidence_observed', False)} / "
            f"{acquisition.get(tasks[1], {}).get('explicit_metric_evidence_observed', False)} | "
            f"{cached.get(tasks[0], 0):,} / {cached.get(tasks[1], 0):,} | "
            f"{'<br>'.join(findings) if findings else 'none'} |")
    lines.append("")
    lines.append("## First turn of task 2 (before any tool result)")
    lines.append("")
    for item in report["models"]:
        excerpt = item["second_task_first_turn_excerpt"] or "(empty)"
        lines.extend([f"### {item['model']}", "", "```text", excerpt, "```", ""])
    return "\n".join(lines)


def audit_batch(
    batch_dir: Path,
    models: Iterable[str],
    tasks: tuple[str, str],
    *,
    write: bool = True,
) -> dict[str, Any]:
    batch_dir = Path(batch_dir).resolve()
    items = [audit_model(batch_dir, model, tasks) for model in models]
    statuses = {item["status"] for item in items}
    status = "fail" if "fail" in statuses else (
        "pass_with_review" if "pass_with_review" in statuses else "pass")
    report = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "batch_dir": str(batch_dir),
        "tasks": list(tasks),
        "status": status,
        "models": items,
    }
    if write:
        audit_dir = batch_dir / "_isolation_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (audit_dir / "AUDIT.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit two staged reference-scaffold tasks for isolation.")
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tasks", nargs=2, required=True)
    args = parser.parse_args()
    report = audit_batch(args.batch_dir, args.models, tuple(args.tasks))
    print(json.dumps({"status": report["status"], "batch_dir": report["batch_dir"]}, indent=2))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
