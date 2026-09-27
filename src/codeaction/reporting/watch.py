#!/usr/bin/env python3
"""Formal test progress watcher & progress file writer.

Discovers, monitors, and summarizes formal test runs in real time:
- Scans run directories for completed attempts (result.json), running attempts (transcript.jsonl),
  and pending attempts.
- Computes success rates, Wilson CI95, step counts, durations, and estimated remaining time (ETA).
- Displays live terminal updates (--watch N or --once).
- Atomically writes real-time status to `progress.json` and `PROGRESS.md` inside each run directory,
  plus batch/per-model snapshots and `api_events.jsonl` under the selected output directory.

Usage:
  # Watch one formal batch whose run roots live below runs/<model>/<task>/:
  python -m codeaction.reporting.watch \
    runs/formal_20260811/runs \
    --watch 10 \
    --output-dir runs/formal_20260811/_monitor

  # Single-shot report for one codeaction run directory:
  python -m codeaction.reporting.watch <run-dir> --once

  # Watch all active run groups below CODEACTION_RUNS_ROOT:
  python -m codeaction.reporting.watch --watch 10

The batch snapshot directory contains progress.json, PROGRESS.md, api_events.jsonl, and one
models/<model-id>/{progress.json,api_events.jsonl} pair per model. Writes are atomic. Raw evidence
stays in each codeaction attempt directory and is never rewritten by this watcher.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from codeaction.paths import PROJECT_ROOT, REPOSITORY_ROOT, TASKS_ROOT
from codeaction.verification.metrics import resource_accounting


_DATA_ROOT = Path(os.environ.get("CODEACTION_RUNS_ROOT", PROJECT_ROOT / "runs")).resolve()
_LOGS_DIR = _DATA_ROOT / "_logs"

try:
    from codeaction.verification.metrics import wilson_ci95
except ImportError:
    def wilson_ci95(successes: int, n: int) -> Tuple[float, float]:
        if n <= 0:
            return (0.0, 0.0)
        p = successes / n
        z = 1.959964
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p + z2 / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
        return (max(0.0, center - half), min(1.0, center + half))

_ATTEMPT_DIR_PAT = re.compile(
    r"^(?:[as]\d+|seed_?\d+|attempt_?\d+|attempt-\d+-seed-\d+)$",
    re.IGNORECASE,
)
_REFERENCE_EVENT_MARK = "[reference-agent-event] "
_PROVIDER_FAILURE_EVENTS = {"endpoint_failure", "context_length_exceeded"}
_COMPLETE_ATTEMPT_CACHE: Dict[str, Tuple[tuple, dict]] = {}


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "N/A"
    sec = int(round(seconds))
    mins, s = divmod(sec, 60)
    hrs, m = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[dict]:
    records = []
    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
            except Exception:
                continue
    except Exception:
        pass
    return records


def _read_reference_events(attempt_dir: Path) -> Tuple[List[dict], Path]:
    """Read canonical events, falling back to the live compose stream while a run is active."""
    reference = attempt_dir / "reference_transcript.jsonl"
    if reference.is_file():
        return _read_jsonl(reference), reference

    compose = attempt_dir / "compose.log"
    records = []
    try:
        for line in compose.read_text(encoding="utf-8", errors="replace").splitlines():
            if _REFERENCE_EVENT_MARK not in line:
                continue
            try:
                value = json.loads(line.split(_REFERENCE_EVENT_MARK, 1)[1])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    except OSError:
        pass
    if records:
        return records, compose

    transcript = attempt_dir / "transcript.jsonl"
    return _read_jsonl(transcript), transcript


def _attempt_fingerprint(attempt_dir: Path) -> tuple:
    values = []
    for name in (
            "result.json", "run_meta.json", "controller_status.json",
            "reference_transcript.jsonl", "compose.log", "transcript.jsonl"):
        path = attempt_dir / name
        try:
            stat = path.stat()
            values.append((name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            values.append((name, None, None))
    return tuple(values)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _failure_events(records: List[dict]) -> List[dict]:
    explicit = [record for record in records
                if record.get("event") == "provider_attempt_failed"]
    if explicit:
        return explicit
    # Compatibility for older transcripts: a terminal provider failure was recorded, but
    # successful retries were not. Do not guess from arbitrary strings or double count it.
    return [record for record in records
            if record.get("event") in _PROVIDER_FAILURE_EVENTS
            and isinstance(record.get("failure"), dict)]


def _provider_failure_summary(records: List[dict]) -> Tuple[Dict[str, int], float]:
    counts: Dict[str, int] = {}
    retry_wait_s = 0.0
    for record in _failure_events(records):
        failure = record.get("failure") or {}
        code = str(failure.get("code") or "provider_unknown")
        counts[code] = counts.get(code, 0) + 1
        try:
            retry_wait_s += float(record.get("retry_delay_s") or 0.0)
        except (TypeError, ValueError):
            pass
    return counts, round(retry_wait_s, 3)


def _quota_wait_total(records: List[dict]) -> float:
    total = 0.0
    for record in records:
        if record.get("event") != "provider_quota_wait":
            continue
        try:
            total += float(record.get("quota_wait_s") or 0.0)
        except (TypeError, ValueError):
            pass
    return round(total, 3)


def _provider_traffic(records: List[dict]) -> dict:
    successful = [record for record in records if record.get("event") == "model_turn"]
    failed = _failure_events(records)
    usage: Dict[str, int] = {}
    for record in successful:
        for key, value in (record.get("usage") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    return {
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_requests": len(successful) + len(failed),
        "usage": usage,
    }


def parse_attempt_dir(attempt_dir: Path) -> dict:
    """Parse a single attempt directory for status, steps, timing, and verifier result."""
    cache_key = str(attempt_dir.resolve())
    fingerprint = _attempt_fingerprint(attempt_dir)
    cached = _COMPLETE_ATTEMPT_CACHE.get(cache_key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    result_path = attempt_dir / "result.json"
    meta_path = attempt_dir / "run_meta.json"
    status_path = attempt_dir / "controller_status.json"

    res_data = _read_json(result_path)
    run_meta = _read_json(meta_path) or {}
    controller = _read_json(status_path) or {}
    records, activity_path = _read_reference_events(attempt_dir)

    dir_name = attempt_dir.name
    now_ts = time.time()
    provider_failures, retry_wait_s = _provider_failure_summary(records)
    quota_wait_s = _quota_wait_total(records)
    provider_traffic = _provider_traffic(records)
    provider_failure_events = _failure_events(records)
    rate_limit_hits = provider_failures.get("provider_rate_limited", 0)

    max_budget_used = 0
    observed_steps = 0
    last_tool = None
    last_args = None
    for rec in records:
        if rec.get("event") in ("tool", "done", "limit"):
            last_tool = rec.get("tool") or rec.get("event")
            last_args = rec.get("args")
            if rec.get("charged", True):
                observed_steps += 1
        elif rec.get("event") in ("step", "tool_call"):
            observed_steps += 1
            last_tool = rec.get("tool") or rec.get("tool_name") or rec.get("action")
            last_args = rec.get("args") or rec.get("kwargs")
        try:
            max_budget_used = max(
                max_budget_used, int(rec.get("budget_used") or rec.get("steps") or 0))
        except (TypeError, ValueError):
            pass
    max_budget_used = max(max_budget_used, observed_steps)

    if res_data is not None:
        stats = res_data.get("stats") if isinstance(res_data.get("stats"), dict) else {}
        verifier = res_data.get("verifier") if isinstance(res_data.get("verifier"), dict) else {}
        is_success = bool(verifier.get("success", False))

        resources = resource_accounting(res_data, records)
        tool_calls_used = resources["tool_calls_used"]
        if tool_calls_used is None:
            tool_calls_used = max_budget_used
        timing = res_data.get("timing") if isinstance(res_data.get("timing"), dict) else {}
        duration_s = (
            timing.get("wall_time_s")
            or timing.get("elapsed_s")
            or stats.get("duration_s")
            or stats.get("wall_s")
            or stats.get("wall_time_s")
            or res_data.get("duration_s")
            or res_data.get("wall_s")
        )

        if duration_s is None and result_path.exists():
            duration_s = max(0.1, result_path.stat().st_mtime - attempt_dir.stat().st_mtime)

        failure = res_data.get("failure") if isinstance(res_data.get("failure"), dict) else {}
        failure_reason = None
        if not is_success:
            failure_reason = (
                verifier.get("failure_reason")
                or failure.get("code")
                or stats.get("status")
                or "unspecified_failure"
            )

        controller_state = controller.get("state")
        status = (
            "failed" if controller_state == "failed"
            else "finalizing" if controller and controller_state != "complete"
            else "completed"
        )

        parsed = {
            "name": dir_name,
            "path": str(attempt_dir),
            "status": status,
            "success": is_success,
            "tool_calls_used": tool_calls_used,
            "tool_call_budget": resources["tool_call_budget"],
            "model_turns": resources["model_turns"],
            # Compatibility for old progress.json readers. This value has always meant charged
            # outer tool calls, never provider model turns.
            "steps": tool_calls_used,
            "duration_s": round(float(duration_s), 2) if duration_s is not None else None,
            "failure_reason": failure_reason,
            "verifier": verifier,
            "rate_limit_hits": rate_limit_hits,
            "provider_failures": provider_failures,
            "retry_wait_s": retry_wait_s,
            "quota_wait_s": quota_wait_s,
            "provider_traffic": provider_traffic,
            "provider_failure_events": provider_failure_events,
            "controller_state": controller_state,
        }
        if status == "completed":
            _COMPLETE_ATTEMPT_CACHE[cache_key] = (fingerprint, parsed)
        return parsed

    has_activity = bool(records) or controller.get("state") == "starting"
    if has_activity:
        start_ts = attempt_dir.stat().st_mtime
        last_ts = activity_path.stat().st_mtime if activity_path.exists() else start_ts
        elapsed_s = max(0.0, now_ts - attempt_dir.stat().st_mtime)
        stale_threshold_s = 600.0  # 10 minutes without update means process likely died
        is_stale = (now_ts - last_ts) > stale_threshold_s
        status_str = (
            "failed" if controller.get("state") == "failed"
            else "interrupted" if is_stale
            else "running"
        )

        return {
            "name": dir_name,
            "path": str(attempt_dir),
            "status": status_str,
            "success": False if is_stale else None,
            "tool_calls_used": max_budget_used,
            "tool_call_budget": resource_accounting({}, records)["tool_call_budget"],
            "model_turns": resource_accounting({}, records)["model_turns"],
            "steps": max_budget_used,
            "elapsed_s": round(elapsed_s, 2),
            "last_tool": str(last_tool) if last_tool else "in_progress",
            "last_args": str(last_args)[:80] if last_args else None,
            "failure_reason": "stale_inactivity_timeout" if is_stale else None,
            "model": run_meta.get("model"),
            "interface": run_meta.get("interface"),
            "task_name": run_meta.get("task_name"),
            "rate_limit_hits": rate_limit_hits,
            "provider_failures": provider_failures,
            "retry_wait_s": retry_wait_s,
            "quota_wait_s": quota_wait_s,
            "provider_traffic": provider_traffic,
            "provider_failure_events": provider_failure_events,
            "controller_state": controller.get("state"),
        }

    return {
        "name": dir_name,
        "path": str(attempt_dir),
        "status": "pending",
        "success": None,
        "tool_calls_used": 0,
        "tool_call_budget": None,
        "model_turns": 0,
        "steps": 0,
        "duration_s": None,
        "rate_limit_hits": 0,
        "provider_failures": {},
        "retry_wait_s": 0.0,
        "quota_wait_s": 0.0,
        "provider_traffic": {
            "successful_requests": 0, "failed_requests": 0,
            "total_requests": 0, "usage": {},
        },
        "provider_failure_events": provider_failure_events,
        "controller_state": controller.get("state"),
    }


def inspect_run_group(group_dir: Path, declared_k: Optional[int] = None) -> dict:
    """Inspect a run directory and return structured progress metrics."""
    group_dir = group_dir.resolve()
    run_meta = _read_json(group_dir / "run_meta.json") or {}
    run_data = _read_json(group_dir / "run.json") or {}

    attempt_dirs = []
    if (group_dir / "result.json").exists() or any(
            (group_dir / name).exists() for name in (
                "reference_transcript.jsonl", "transcript.jsonl", "compose.log")):
        attempt_dirs.append(group_dir)
    else:
        for child in sorted(group_dir.iterdir()):
            if child.is_dir() and _ATTEMPT_DIR_PAT.fullmatch(child.name):
                attempt_dirs.append(child)

    attempts = [parse_attempt_dir(d) for d in attempt_dirs]

    # Infer model, interface, task_name from attempts if missing in top-level run_meta
    sample_res = None
    for d in attempt_dirs:
        r = _read_json(d / "result.json")
        if r:
            sample_res = r
            break

    task_name = str(
        run_data.get("task_name")
        or run_meta.get("task_name")
        or (sample_res.get("task_name") if sample_res else None)
        or group_dir.name
    )
    task_card_path = TASKS_ROOT / task_name / "task.json"
    card_data = _read_json(task_card_path) or {}
    comparison_model = (
        (((sample_res or {}).get("identity") or {}).get("comparison") or {}).get("model") or {})
    model = str(
        run_data.get("model")
        or comparison_model.get("id")
        or run_meta.get("model")
        or (sample_res.get("model") if sample_res else None)
        or "unknown_model"
    )
    interface = str(
        run_data.get("interface_profile")
        or run_meta.get("interface")
        or (sample_res.get("interface") if sample_res else None)
        or "unknown_interface"
    )

    k_target = (
        declared_k
        or run_data.get("attempts")
        or run_meta.get("attempts_declared")
        or card_data.get("protocol", {}).get("attempts_k")
        or 5
    )

    completed = [a for a in attempts if a["status"] == "completed"]
    running = [a for a in attempts if a["status"] in ("running", "finalizing")]
    interrupted = [a for a in attempts if a["status"] in ("failed", "interrupted")]

    n_completed = len(completed)
    n_running = len(running)
    n_interrupted = len(interrupted)
    n_total_found = len(attempts)
    n_declared = max(int(k_target), n_total_found)
    n_pending = max(0, n_declared - n_completed - n_running - n_interrupted)

    successes = sum(1 for a in completed if a["success"])
    failures = n_completed - successes

    success_rate = round(successes / n_completed, 4) if n_completed > 0 else 0.0
    ci_low, ci_high = wilson_ci95(successes, n_completed) if n_completed > 0 else (0.0, 0.0)

    durations = [a["duration_s"] for a in completed if a.get("duration_s") is not None]
    avg_dur = sum(durations) / len(durations) if durations else 120.0

    eta_s = max(0.0, (n_pending * avg_dur) + sum(max(0.0, avg_dur - a.get("elapsed_s", 0.0)) for a in running))

    total_rate_limits = sum(a.get("rate_limit_hits", 0) for a in attempts)
    provider_failures: Dict[str, int] = {}
    provider_usage: Dict[str, int] = {}
    for attempt in attempts:
        for code, count in attempt.get("provider_failures", {}).items():
            provider_failures[code] = provider_failures.get(code, 0) + int(count)
        for key, value in attempt.get("provider_traffic", {}).get("usage", {}).items():
            provider_usage[key] = provider_usage.get(key, 0) + int(value)

    return {
        "group_name": group_dir.name,
        "group_path": str(group_dir),
        "task_name": task_name,
        "model": model,
        "interface": interface,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_declared": n_declared,
        "n_completed": n_completed,
        "n_running": n_running,
        "n_interrupted": n_interrupted,
        "n_pending": n_pending,
        "n_success": successes,
        "n_fail": failures,
        "rate_limit_hits": total_rate_limits,
        "provider_failures": provider_failures,
        "retry_wait_s": round(sum(a.get("retry_wait_s", 0.0) for a in attempts), 3),
        "quota_wait_s": round(sum(a.get("quota_wait_s", 0.0) for a in attempts), 3),
        "provider_traffic": {
            "successful_requests": sum(
                a.get("provider_traffic", {}).get("successful_requests", 0)
                for a in attempts),
            "failed_requests": sum(
                a.get("provider_traffic", {}).get("failed_requests", 0)
                for a in attempts),
            "total_requests": sum(
                a.get("provider_traffic", {}).get("total_requests", 0)
                for a in attempts),
            "usage": provider_usage,
        },
        "success_rate": success_rate,
        "ci95": [round(ci_low, 4), round(ci_high, 4)],
        "avg_duration_s": round(avg_dur, 2) if durations else None,
        "eta_s": round(eta_s, 2) if (n_pending > 0 or n_running > 0) else 0.0,
        "attempts": attempts,
    }


def render_progress_md(summary: dict) -> str:
    """Render a clean Markdown progress report."""
    n_dec = summary["n_declared"]
    n_comp = summary["n_completed"]
    n_run = summary["n_running"]
    n_pend = summary["n_pending"]
    n_interrupted = summary.get("n_interrupted", 0)
    pct = round((n_comp / n_dec) * 100.0, 1) if n_dec > 0 else 0.0

    bar_len = 20
    filled = int(round((n_comp / n_dec) * bar_len)) if n_dec > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    if n_comp == n_dec and n_run == 0:
        status_badge = "✅ Complete"
    elif n_run > 0:
        status_badge = f"🏃 Running ({n_run} active)"
    else:
        status_badge = f"⏳ In Progress ({n_pend} pending)"

    ci_str = f"[{summary['ci95'][0]*100:.1f}% - {summary['ci95'][1]*100:.1f}%]"

    lines = [
        f"# Benchmark Progress: {summary['task_name']}",
        "",
        f"- **Model**: `{summary['model']}`",
        f"- **Interface**: `{summary['interface']}`",
        f"- **Status**: {status_badge}",
        f"- **Progress**: `[{bar}]` {pct}% ({n_comp}/{n_dec} completed, "
        f"{n_run} running, {n_pend} pending, {n_interrupted} interrupted)",
        f"- **Success Rate**: **{summary['success_rate']*100:.1f}%** ({summary['n_success']}/{n_comp}) | CI95: {ci_str}",
        f"- **Avg Attempt Duration**: {_format_duration(summary.get('avg_duration_s'))}",
        f"- **Estimated Time Remaining (ETA)**: {_format_duration(summary.get('eta_s'))}",
        f"- **Provider Failures**: `{json.dumps(summary.get('provider_failures', {}), sort_keys=True)}`",
        f"- **Provider Traffic**: `{json.dumps(summary.get('provider_traffic', {}), sort_keys=True)}`",
        f"- **Retry Backoff**: {_format_duration(summary.get('retry_wait_s'))}",
        f"- **Quota Pacing Wait**: {_format_duration(summary.get('quota_wait_s'))}",
        f"- **Last Updated**: {summary['timestamp']}",
        "",
    ]

    running_attempts = [a for a in summary["attempts"]
                        if a["status"] in ("running", "finalizing")]
    if running_attempts:
        lines.append("## ⚡ Active Attempts")
        lines.append("| Attempt | Charged tool calls | Model turns | Last Tool | Elapsed Time |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for a in running_attempts:
            lines.append(f"| `{a['name']}` | {a.get('tool_calls_used', a.get('steps', 0))} | "
                         f"{a.get('model_turns', '—')} | "
                         f"`{a.get('last_tool') or 'running'}` | "
                         f"{_format_duration(a.get('elapsed_s'))} |")
        lines.append("")

    lines.append("## 📊 Attempt History")
    lines.append("| Attempt | Status | Charged tool calls | Model turns | Duration | Details / Verifier |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for a in summary["attempts"]:
        name = a["name"]
        st = a["status"]
        if st == "completed":
            icon = "✅ Pass" if a["success"] else "❌ Fail"
            dur = _format_duration(a.get("duration_s"))
            reason = a.get("failure_reason") or "OK"
            lines.append(f"| `{name}` | {icon} | {a.get('tool_calls_used', a.get('steps', 0))} | "
                         f"{a.get('model_turns', '—')} | {dur} | `{reason}` |")
        elif st == "running":
            dur = _format_duration(a.get("elapsed_s"))
            lines.append(f"| `{name}` | 🏃 Running | {a.get('tool_calls_used', a.get('steps', 0))} | "
                         f"{a.get('model_turns', '—')} | {dur} | tool: `{a.get('last_tool')}` |")
        elif st == "finalizing":
            lines.append(f"| `{name}` | Finalizing | {a.get('tool_calls_used', a.get('steps', 0))} | "
                         f"{a.get('model_turns', '—')} | - | controller audit |")
        elif st in ("failed", "interrupted"):
            reason = a.get("failure_reason") or a.get("controller_state") or st
            lines.append(f"| `{name}` | Interrupted | {a.get('tool_calls_used', a.get('steps', 0))} | "
                         f"{a.get('model_turns', '—')} | - | `{reason}` |")
        else:
            lines.append(f"| `{name}` | ⏳ Pending | - | - | - | - |")

    lines.append("")
    return "\n".join(lines)


def update_progress_files(group_dir: Path, summary: dict) -> Tuple[Path, Path]:
    """Write progress.json and PROGRESS.md into the group directory."""
    group_dir.mkdir(parents=True, exist_ok=True)
    json_path = group_dir / "progress.json"
    md_path = group_dir / "PROGRESS.md"

    _atomic_write_text(json_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    md_content = render_progress_md(summary)
    _atomic_write_text(md_path, md_content + "\n")

    return json_path, md_path


def render_terminal_dashboard(summaries: List[dict], use_color: bool = True) -> str:
    """Format an interactive or scrollable terminal view of benchmark progress."""
    def c(code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    lines = []
    lines.append(c("1;36", "================================================================================"))
    lines.append(c("1;36", f"  ROBOTWIN BENCHMARK PROGRESS DASHBOARD  |  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    lines.append(c("1;36", "================================================================================"))

    total_dec = sum(s["n_declared"] for s in summaries)
    total_comp = sum(s["n_completed"] for s in summaries)
    total_succ = sum(s["n_success"] for s in summaries)
    total_run = sum(s["n_running"] for s in summaries)
    total_pend = sum(s["n_pending"] for s in summaries)
    total_interrupted = sum(s.get("n_interrupted", 0) for s in summaries)

    overall_rate = round(total_succ / total_comp, 4) if total_comp > 0 else 0.0
    overall_ci_low, overall_ci_high = wilson_ci95(total_succ, total_comp) if total_comp > 0 else (0.0, 0.0)

    total_rate_limits = sum(s.get("rate_limit_hits", 0) for s in summaries)

    lines.append(
        f" SUMMARY: Groups={len(summaries)} | "
        f"Completed={total_comp}/{total_dec} | Running={total_run} | "
        f"Pending={total_pend} | Interrupted={total_interrupted}"
    )
    lines.append(
        f" OVERALL SUCCESS RATE: {c('1;32' if overall_rate>0.5 else '1;33', f'{overall_rate*100:.1f}%')} "
        f"({total_succ}/{total_comp}) [CI95: {overall_ci_low*100:.1f}% - {overall_ci_high*100:.1f}%]"
    )
    if total_rate_limits > 0:
        lines.append(c("1;31", f" 🚨 API RATE LIMIT HITS (429): {total_rate_limits} event(s) detected across workers!"))
    else:
        lines.append(c("32", " API RATE LIMIT (429): None (0 events detected)"))
    all_provider_failures: Dict[str, int] = {}
    successful_requests = 0
    failed_requests = 0
    for summary in summaries:
        for code, count in summary.get("provider_failures", {}).items():
            all_provider_failures[code] = all_provider_failures.get(code, 0) + int(count)
        traffic = summary.get("provider_traffic", {})
        successful_requests += int(traffic.get("successful_requests", 0))
        failed_requests += int(traffic.get("failed_requests", 0))
    lines.append(f" PROVIDER FAILURES: {json.dumps(all_provider_failures, sort_keys=True)}")
    lines.append(
        f" PROVIDER REQUESTS: {successful_requests + failed_requests} total | "
        f"{successful_requests} successful | {failed_requests} failed")
    lines.append(c("36", "--------------------------------------------------------------------------------"))

    for s in summaries:
        n_dec = s["n_declared"]
        n_comp = s["n_completed"]
        n_run = s["n_running"]
        pct = (n_comp / n_dec) * 100.0 if n_dec > 0 else 0.0

        rate_str = f"{s['success_rate']*100:.1f}%" if n_comp > 0 else "N/A"
        rate_colored = c("1;32", rate_str) if s["success_rate"] >= 0.6 else c("1;31", rate_str) if n_comp > 0 else rate_str

        rl_hits = s.get("rate_limit_hits", 0)
        rl_tag = f" | {c('1;31', f'429 Hits: {rl_hits}')}" if rl_hits > 0 else ""

        status_tag = (
            c("1;32", "[DONE]") if n_comp == n_dec and n_run == 0
            else c("1;33", "[RUNNING]") if n_run > 0
            else c("1;31", "[INTERRUPTED]") if s.get("n_interrupted", 0) > 0
            else c("34", "[WAITING]")
        )

        lines.append(
            f"{status_tag} {c('1', s['task_name']):<25} | Model: {s['model']} | "
            f"Progress: {n_comp}/{n_dec} ({pct:.0f}%) | Success: {rate_colored} ({s['n_success']}/{n_comp}){rl_tag}"
        )

        active = [a for a in s["attempts"]
                  if a["status"] in ("running", "finalizing")]
        if active:
            for a in active:
                lines.append(
                    f"   └── ⚡ Active {a['name']}: tool calls "
                    f"{a.get('tool_calls_used', a.get('steps', 0))}"
                    f"/{a.get('tool_call_budget') or '?'} | model turns "
                    f"{a.get('model_turns', '?')} | last tool: {a.get('last_tool')} | "
                    f"elapsed: {_format_duration(a.get('elapsed_s'))}"
                )

    lines.append(c("36", "================================================================================"))
    return "\n".join(lines)


def _api_events(summaries: List[dict]) -> List[dict]:
    events = []
    for summary in summaries:
        for attempt in summary["attempts"]:
            for index, record in enumerate(
                    attempt.get("provider_failure_events", []), 1):
                events.append({
                    "model": summary["model"],
                    "task_name": summary["task_name"],
                    "group_path": summary["group_path"],
                    "attempt": attempt["name"],
                    "event_index": index,
                    **record,
                })
    return events


def _totals(summaries: List[dict]) -> dict:
    failures: Dict[str, int] = {}
    usage: Dict[str, int] = {}
    for summary in summaries:
        for code, count in summary.get("provider_failures", {}).items():
            failures[code] = failures.get(code, 0) + int(count)
        for key, value in summary.get("provider_traffic", {}).get("usage", {}).items():
            usage[key] = usage.get(key, 0) + int(value)
    return {
        "groups": len(summaries),
        "declared": sum(s["n_declared"] for s in summaries),
        "completed": sum(s["n_completed"] for s in summaries),
        "running": sum(s["n_running"] for s in summaries),
        "pending": sum(s["n_pending"] for s in summaries),
        "interrupted": sum(s.get("n_interrupted", 0) for s in summaries),
        "successes": sum(s["n_success"] for s in summaries),
        "failures": sum(s["n_fail"] for s in summaries),
        "provider_failures": failures,
        "provider_traffic": {
            "successful_requests": sum(
                s.get("provider_traffic", {}).get("successful_requests", 0)
                for s in summaries),
            "failed_requests": sum(
                s.get("provider_traffic", {}).get("failed_requests", 0)
                for s in summaries),
            "total_requests": sum(
                s.get("provider_traffic", {}).get("total_requests", 0)
                for s in summaries),
            "usage": usage,
        },
        "retry_wait_s": round(sum(s.get("retry_wait_s", 0.0) for s in summaries), 3),
        "quota_wait_s": round(sum(s.get("quota_wait_s", 0.0) for s in summaries), 3),
    }


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return normalized or "unknown-model"


def update_batch_files(output_dir: Path, summaries: List[dict]) -> None:
    """Write one batch snapshot plus isolated derived files for each model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    by_model: Dict[str, List[dict]] = {}
    for summary in summaries:
        by_model.setdefault(summary["model"], []).append(summary)
    snapshot = {
        "schema_version": "1.0",
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "totals": _totals(summaries),
        "groups": summaries,
        "models": {model: _totals(groups) for model, groups in sorted(by_model.items())},
    }
    _atomic_write_text(
        output_dir / "progress.json",
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write_text(
        output_dir / "PROGRESS.md",
        "# Formal Benchmark Progress\n\n```text\n"
        + render_terminal_dashboard(summaries, use_color=False)
        + "\n```\n",
    )
    all_events = _api_events(summaries)
    _atomic_write_text(
        output_dir / "api_events.jsonl",
        "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                for event in all_events),
    )
    for model, groups in by_model.items():
        model_dir = output_dir / "models" / _safe_component(model)
        model_snapshot = {
            "schema_version": "1.0",
            "updated_at": snapshot["updated_at"],
            "model": model,
            "totals": _totals(groups),
            "groups": groups,
        }
        _atomic_write_text(
            model_dir / "progress.json",
            json.dumps(model_snapshot, indent=2, ensure_ascii=False) + "\n",
        )
        model_events = [event for event in all_events if event["model"] == model]
        _atomic_write_text(
            model_dir / "api_events.jsonl",
            "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                    for event in model_events),
        )


def discover_run_groups(root: Path) -> List[Path]:
    """Find codeaction run roots recursively, with a legacy direct-attempt fallback."""
    groups: List[Path] = []
    if not root.is_dir():
        return groups
    if (root / "run.json").is_file() or (root / "result.json").is_file():
        return [root.resolve()]
    if any(child.is_dir() and _ATTEMPT_DIR_PAT.fullmatch(child.name)
           for child in root.iterdir()):
        return [root.resolve()]

    for run_file in root.rglob("run.json"):
        if any(part in {"_logs", "__pycache__", "runs_report", "_monitor"}
               for part in run_file.parts):
            continue
        groups.append(run_file.parent.resolve())
    if groups:
        return sorted(set(groups))

    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        if any(child.is_dir() and _ATTEMPT_DIR_PAT.fullmatch(child.name)
               for child in item.iterdir()):
            groups.append(item.resolve())
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch & report formal test run progress.")
    parser.add_argument(
        "groups",
        nargs="*",
        type=Path,
        help="Run directory paths. Default: discover all groups below CODEACTION_RUNS_ROOT.",
    )
    parser.add_argument("--watch", "-w", type=float, default=0, help="Watch interval in seconds (0 = single pass).")
    parser.add_argument("--once", action="store_true", help="Single pass, write progress files and exit.")
    parser.add_argument("--json", action="store_true", help="Dump JSON summary to stdout.")
    parser.add_argument("--k", type=int, default=None, help="Target attempts_k override.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Batch snapshot directory (default: CODEACTION_RUNS_ROOT/_logs).",
    )
    args = parser.parse_args()

    raw_groups = args.groups
    if raw_groups:
        roots = []
        for p in raw_groups:
            path_candidate = Path(p)
            if path_candidate.is_absolute() and path_candidate.exists():
                roots.append(path_candidate.resolve())
            elif (Path.cwd() / path_candidate).exists():
                roots.append((Path.cwd() / path_candidate).resolve())
            elif (PROJECT_ROOT / path_candidate).exists():
                roots.append((PROJECT_ROOT / path_candidate).resolve())
            elif (_DATA_ROOT / path_candidate).exists():
                roots.append((_DATA_ROOT / path_candidate).resolve())
            elif (REPOSITORY_ROOT / path_candidate).exists():
                roots.append((REPOSITORY_ROOT / path_candidate).resolve())
            else:
                roots.append((Path.cwd() / path_candidate).resolve())
    else:
        roots = [_DATA_ROOT]

    watch_interval = 0.0 if args.once else args.watch
    output_dir = (args.output_dir or _LOGS_DIR).resolve()

    while True:
        group_paths = sorted({
            group for root in roots for group in discover_run_groups(root)
        })
        if not group_paths and watch_interval <= 0:
            print("[watch_progress] No run group found.", file=sys.stderr)
            return 1
        summaries = []
        for g in group_paths:
            if not g.exists():
                continue
            summary = inspect_run_group(g, declared_k=args.k)
            summaries.append(summary)
            update_progress_files(g, summary)

        update_batch_files(output_dir, summaries)

        if args.json:
            payload = summaries[0] if len(summaries) == 1 else summaries
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            if watch_interval > 0:
                print("\033[H\033[J", end="")  # Clear terminal screen
            if summaries:
                print(render_terminal_dashboard(summaries, use_color=sys.stdout.isatty()))
            else:
                print(f"[watch_progress] Waiting for run.json under: "
                      f"{', '.join(str(root) for root in roots)}")

        if watch_interval <= 0:
            break
        time.sleep(watch_interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
