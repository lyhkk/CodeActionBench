#!/usr/bin/env python3
"""Read-only terminal watcher for durable codeaction batch state and attention."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

_AP = Path(__file__).resolve().parent.parent
if str(_AP) not in sys.path:
    sys.path.insert(0, str(_AP))

from codeaction.batch.attention import render_attention  # noqa: E402
from codeaction.batch.state import BatchStateError, BatchStateStore  # noqa: E402


def positive_finite_interval(value: str) -> float:
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return interval


def completed_model_tasks(state: Mapping[str, Any]) -> set[tuple[str, str]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for identifier in state["requested_cells"]:
        cell = state["cells"][identifier]
        grouped.setdefault((cell["model"], cell["task"]), []).append(cell["status"])
    return {
        key for key, statuses in grouped.items()
        if statuses and set(statuses) == {"accepted"}
    }


def summary_line(state: Mapping[str, Any]) -> str:
    counts = {status: 0 for status in (
        "queued", "running", "retry_wait", "accepted", "needs_attention")}
    for identifier in state["requested_cells"]:
        counts[state["cells"][identifier]["status"]] += 1
    return (
        f"[{state['active_stage']}] accepted={counts['accepted']}/{len(state['requested_cells'])} "
        f"running={counts['running']} queued={counts['queued']} retry_wait={counts['retry_wait']} "
        f"attention={counts['needs_attention']} status={state['status']} rev={state['revision']}"
    )


def summary_signature(state: Mapping[str, Any]) -> tuple[Any, ...]:
    counts = {status: 0 for status in (
        "queued", "running", "retry_wait", "accepted", "needs_attention")}
    for identifier in state["requested_cells"]:
        counts[state["cells"][identifier]["status"]] += 1
    return (
        state["active_stage"], state["status"], len(state["requested_cells"]),
        *(counts[status] for status in sorted(counts)),
    )


def read_watchdog_status(batch_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            (Path(batch_dir) / "watchdog_status.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def watchdog_line(status: Mapping[str, Any]) -> str:
    label = str(status.get("status") or "unknown").upper()
    return (
        f"[WATCHDOG {label}] reason={status.get('reason')} "
        f"restarts={status.get('restarts_used')}/{status.get('max_restarts')}"
    )


def watch(
    batch_dir: Path,
    *,
    interval_s: float | None,
    quiet_completions: bool,
    stream: TextIO,
    bell: bool,
    read_snapshot: Callable[[Path], Mapping[str, Any]] = BatchStateStore.read_snapshot,
    read_watchdog: Callable[[Path], Mapping[str, Any]] = read_watchdog_status,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    previous_revision = None
    previous_complete: set[tuple[str, str]] = set()
    previous_attention: set[str] = set()
    previous_summary = None
    previous_watchdog_status = None
    first = True
    while True:
        state = read_snapshot(batch_dir)
        if state["revision"] != previous_revision:
            complete = completed_model_tasks(state)
            open_attention = {
                attention_id: record
                for attention_id, record in state.get("attentions", {}).items()
                if record.get("status") == "open"
            }
            if not first and not quiet_completions:
                for model, task in sorted(complete - previous_complete):
                    stream.write(f"[COMPLETE] {task} / {model}\n")
            signature = summary_signature(state)
            if first or signature != previous_summary:
                stream.write(summary_line(state) + "\n")
            for attention_id in sorted(set(open_attention) - previous_attention):
                render_attention(open_attention[attention_id], stream=stream, bell=bell)
            if not first:
                for attention_id in sorted(previous_attention - set(open_attention)):
                    stream.write(f"[RESOLVED {attention_id}]\n")
            stream.flush()
            previous_revision = state["revision"]
            previous_summary = signature
            previous_complete = complete
            previous_attention = set(open_attention)
            first = False
        watchdog = read_watchdog(batch_dir)
        watchdog_status = watchdog.get("status")
        alert_statuses = {"restarting", "needs_attention", "stopped"}
        if watchdog_status != previous_watchdog_status and (
                watchdog_status in alert_statuses
                or previous_watchdog_status in alert_statuses):
            stream.write(watchdog_line(watchdog) + "\n")
            stream.flush()
        previous_watchdog_status = watchdog_status
        if interval_s is None:
            return 0
        sleep_fn(interval_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch one durable benchmark batch.")
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument(
        "--watch", type=positive_finite_interval, default=None, metavar="SECONDS")
    parser.add_argument("--quiet-completions", action="store_true")
    parser.add_argument("--no-bell", action="store_true")
    args = parser.parse_args(argv)
    try:
        return watch(
            args.batch_dir.resolve(), interval_s=args.watch,
            quiet_completions=args.quiet_completions,
            stream=sys.stdout, bell=not args.no_bell)
    except KeyboardInterrupt:
        return 0
    except BatchStateError as exc:
        print(f"watch_benchmark_batch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
