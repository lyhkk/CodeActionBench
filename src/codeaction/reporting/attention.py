"""What in a running agent matrix needs a human, and nothing else.

`reporting.watch` answers "how far along is it". This answers the different question "is anything
stuck or wrong", which is what an operator actually watches during a multi-hour batch. It reads
only files the batch already writes -- batch_state.json and the watcher's progress.json -- so it
never competes with the runner for the simulator or a provider.

An item is raised only when a person could act on it. A verifier failure is a RESULT, not an
attention item: the benchmark measuring a failed attempt is the system working.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# A cell running far past its own agent's estimate is the only stall signal available without
# reaching into the episode; the multiplier is deliberately loose so a slow-but-live cell does
# not cry wolf.
STALL_MULTIPLIER = 3.0
DEFAULT_EXPECTED_MINUTES = 20


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _age_minutes(started_at: Any, now: dt.datetime) -> float | None:
    if not isinstance(started_at, str):
        return None
    try:
        started = dt.datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=dt.timezone.utc)
    return (now - started).total_seconds() / 60.0


def _expected_minutes(state: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(entry.get("agent")): float(entry.get("expected_minutes_per_task")
                                       or DEFAULT_EXPECTED_MINUTES)
        for entry in state.get("queue", []) or []
        if isinstance(entry, Mapping) and entry.get("agent")
    }


def collect(batch_dir: Path, *, now: dt.datetime | None = None) -> dict:
    """Attention items for one batch directory, most actionable first."""
    batch_dir = Path(batch_dir)
    now = now or dt.datetime.now(dt.timezone.utc)
    state = _read_json(batch_dir / "batch_state.json")
    cells = state.get("cells") or {}
    expected = _expected_minutes(state)
    items: list[dict] = []

    for worker_error in state.get("worker_errors", []) or []:
        items.append({"severity": "critical", "kind": "worker_crashed",
                      "detail": str(worker_error),
                      "action": "a GPU worker thread died; the batch is short a lane"})

    for cell_id, cell in sorted(cells.items()):
        if not isinstance(cell, Mapping):
            continue
        status = cell.get("status")
        agent = str(cell_id).split("/", 1)[0]
        if status == "blocked":
            items.append({
                "severity": "critical", "kind": "cell_blocked", "cell": cell_id,
                "detail": str(cell.get("error") or f"rc={cell.get('return_code')}"),
                "log": cell.get("log"),
                "action": "inspect the log; this agent's remaining tasks were skipped"})
        elif status == "skipped":
            items.append({
                "severity": "warning", "kind": "cell_skipped", "cell": cell_id,
                "detail": str(cell.get("reason") or ""),
                "action": "re-queue after fixing the blocking cell"})
        elif status == "running":
            age = _age_minutes(cell.get("started_at"), now)
            budget = expected.get(agent, DEFAULT_EXPECTED_MINUTES) * STALL_MULTIPLIER
            if age is not None and age > budget:
                items.append({
                    "severity": "warning", "kind": "cell_stalled", "cell": cell_id,
                    "detail": f"running {age:.0f} min, over {budget:.0f} min for this agent",
                    "log": cell.get("log"),
                    "action": "check the log for a hung provider or simulator"})
            quota = float(cell.get("quota_wait_s") or 0)
            if quota > 600:
                items.append({
                    "severity": "info", "kind": "quota_pressure", "cell": cell_id,
                    "detail": f"{quota / 60:.0f} min waiting on provider quota",
                    "action": "expected under rate limits; raise the ceiling only if declared"})

    # A finished cell whose turn-by-turn view failed to build still has its raw evidence, but
    # the readable record everything downstream consumes is missing -- worth a person's attention
    # exactly once, at the end, rather than a silent error sidecar nobody opens.
    for status_path in sorted(batch_dir.glob("runs/*/*/attempt-*/controller_status.json")):
        verdict = str(_read_json(status_path).get("turns_view") or "")
        if verdict.startswith("error:"):
            attempt = status_path.parent
            items.append({
                "severity": "warning", "kind": "turns_view_failed",
                "cell": "/".join(attempt.parts[-3:-1]),
                "detail": verdict,
                "log": str(attempt / "turns.v1.error.json"),
                "action": "raw transcripts are intact; rebuild the view with "
                          "codeaction.reporting.turns_view.build_turns_view"})

    progress = _read_json(batch_dir / "_monitor" / "progress.json")
    order = {"critical": 0, "warning": 1, "info": 2}
    items.sort(key=lambda item: (order.get(item["severity"], 9), item.get("cell") or ""))
    counts: dict[str, int] = {}
    for cell in cells.values():
        if isinstance(cell, Mapping):
            counts[str(cell.get("status"))] = counts.get(str(cell.get("status")), 0) + 1
    return {
        "schema_version": "attention.v1",
        "batch_dir": str(batch_dir),
        "batch_status": state.get("status"),
        "observed_at": now.isoformat(),
        "cell_counts": counts,
        "total_cells": len(cells),
        "attempts_observed": (progress.get("totals") or {}).get("attempts"),
        "items": items,
    }


def render(report: Mapping[str, Any], *, use_color: bool = True) -> str:
    mark = {"critical": "!!", "warning": " !", "info": "  "}
    tint = {"critical": "\033[31m", "warning": "\033[33m", "info": "\033[2m"}
    reset = "\033[0m" if use_color else ""
    counts = report.get("cell_counts") or {}
    lines = [
        f"batch {report.get('batch_dir')}",
        f"  status={report.get('batch_status')} cells={report.get('total_cells')} "
        + " ".join(f"{name}={value}" for name, value in sorted(counts.items())),
        "",
    ]
    items = report.get("items") or []
    if not items:
        lines.append("  nothing needs attention")
        return "\n".join(lines)
    for item in items:
        colour = tint.get(item["severity"], "") if use_color else ""
        head = f"  {mark.get(item['severity'], '  ')} {item['kind']}"
        if item.get("cell"):
            head += f"  {item['cell']}"
        lines.append(f"{colour}{head}{reset}")
        lines.append(f"       {item.get('detail', '')}")
        lines.append(f"       -> {item.get('action', '')}")
        if item.get("log"):
            lines.append(f"       log: {item['log']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show only what needs a human in a running agent matrix.")
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--watch", "-w", type=float, default=0,
                        help="refresh interval in seconds (0 = single pass)")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    while True:
        report = collect(args.batch_dir)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        else:
            if args.watch:
                print("\033[2J\033[H", end="")
            print(render(report, use_color=not args.no_color), flush=True)
        if not args.watch:
            critical = any(item["severity"] == "critical" for item in report["items"])
            return 1 if critical else 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
