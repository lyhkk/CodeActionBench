#!/usr/bin/env python3
"""Replay a published tool ledger against a fresh scene, with nothing but the ledger.

The ledger is `benchmark/tasks/<task>/solution/oracle_calls.jsonl`: one line per call, each with
the tool's name, its literal arguments and the result it returned when the run was recorded.
This module reads the name and the arguments, boots the card's scene at the card's own seed,
dispatches every call in order through the same tool surface an agent gets, notes each place the
replayed result departs from the recorded one, and lets the verifier judge the end state.

It imports nothing from the library that produced the ledger -- no `sequence.py`, no `_common`,
no `_perception` -- and asks ideal perception nothing (`oracle_gate.n_oracle_queries` is 0 in its
result). That is the proof the bare sequence is sufficient: every scene answer the program ever
needed is already a literal in an argument, and no expert channel exists at replay time.

A replay is open-loop. The recorded program branched on what it saw; the ledger is what it did.
On the same seed the layout is identical and the cameras are fixed, so the pixel-derived
arguments hold; whether the motions land where they landed is a property of the simulator and the
planner, measured here per task and written to `replay_summary.json`, never assumed.

  PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl \\
  python oracle/replay.py click_bell --out <a fresh directory>

or `tools/run_oracle.sh replay click_bell`, which sets the environment above.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from codeaction.paths import PROJECT_ROOT, REPOSITORY_ROOT, ROBOTWIN_ROOT, TASKS_ROOT

SCHEMA_VERSION = "codeaction-solution-replay.v1"

# Result fields compared between the recorded and the replayed call. Exact for statuses,
# counts and ids; within a tolerance for the metre-valued reads, which are recomputed from a
# camera whose pose is fixed but whose renderer need not be bit-identical.
EXACT_FIELDS = ("status", "fingers_in_contact", "obs_id", "reachable", "ok")
METRIC_FIELDS = (("value", 0.005), ("tcp", 0.01))


def ledger_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ledger(path):
    """The calls, in order: name and arguments only. The recorded results ride along for the
    comparison and are never handed to the tool surface."""
    calls = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        calls.append({"i": int(rec["i"]), "tool": str(rec["tool"]),
                      "args": dict(rec.get("args") or {}), "recorded": rec.get("result")})
    if not calls:
        raise ValueError(f"{path}: the ledger is empty")
    if [c["i"] for c in calls] != list(range(1, len(calls) + 1)):
        raise ValueError(f"{path}: the ledger is not numbered 1..n without gaps")
    return calls


def _metric(payload, field):
    """The first metre-valued reading a payload carries under `field`, as a flat list."""
    if not isinstance(payload, dict):
        return None
    value = payload.get(field)
    if value is None and field == "tcp":
        value = (payload.get("resulting_pose") or {}).get("tcp") if isinstance(
            payload.get("resulting_pose"), dict) else None
    if isinstance(value, dict):
        value = value.get("xyz")
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (int, float)) for v in value[:3]):
        return [float(v) for v in value[:3]]
    return None


def divergences(recorded, replayed):
    """Where a replayed result departs from the recorded one, field by field."""
    out = []
    if not isinstance(recorded, dict) or not isinstance(replayed, dict):
        return out
    for field in EXACT_FIELDS:
        if field in recorded and recorded.get(field) != replayed.get(field):
            out.append({"field": field, "recorded": recorded.get(field),
                        "replayed": replayed.get(field)})
    for field, tol in METRIC_FIELDS:
        a, b = _metric(recorded, field), _metric(replayed, field)
        if a is None:
            continue
        if b is None:
            out.append({"field": field, "recorded": a, "replayed": None})
            continue
        gap = max(abs(x - y) for x, y in zip(a, b))
        if gap > tol:
            out.append({"field": field, "recorded": a, "replayed": b, "gap_m": round(gap, 4)})
    return out


def replay_calls(tools, calls):
    """Dispatch the ledger in order through `tools.call`. Stops at the first call the surface
    refuses (a schema error, a terminal episode); a departing result is noted, not fatal --
    the verifier is the judge of what the departures added up to."""
    segments, found = [], []
    for call in calls:
        try:
            payload = tools.call(call["tool"], **call["args"])
        except Exception as exc:
            segments.append({"name": f"call_{call['i']:03d}_{call['tool']}", "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"})
            return {"ok": False, "segments": segments,
                    "detail": {"n_replayed": len(segments) - 1, "n_calls": len(calls),
                               "failed_at": segments[-1]["name"], "divergences": found}}
        gaps = divergences(call["recorded"], payload)
        for gap in gaps:
            found.append({"i": call["i"], "tool": call["tool"], **gap})
        segments.append({"name": f"call_{call['i']:03d}_{call['tool']}", "ok": not gaps,
                         "status": payload.get("status") if isinstance(payload, dict) else None,
                         "divergences": gaps})
    return {"ok": True, "segments": segments,
            "detail": {"n_replayed": len(calls), "n_calls": len(calls), "divergences": found}}


def summarize(result, *, task, seed, ledger_path, calls, out_dir):
    detail = (result.get("reference") or {}).get("detail") or {}
    found = detail.get("divergences") or []
    verifier_ok = bool((result.get("verifier") or {}).get("success"))
    git = ((result.get("provenance") or {}).get("git") or {})
    return {"schema_version": SCHEMA_VERSION, "task": task, "seed": int(seed),
            "verdict": "PASS" if result.get("success") else "FAIL",
            "classification": result.get("classification"),
            "verifier_success": verifier_ok,
            "ledger": {"path": Path(ledger_path).name, "sha256": ledger_sha256(ledger_path),
                       "n_calls": len(calls)},
            "n_replayed": detail.get("n_replayed"), "failed_at": detail.get("failed_at"),
            "n_oracle_queries": (result.get("oracle_gate") or {}).get("n_oracle_queries"),
            "n_divergences": len(found), "divergences": found[:20],
            "commit": git.get("commit", ""), "source_dirty": git.get("dirty", True),
            "task_files_sha256": (result.get("provenance") or {}).get("task_files_sha256"),
            "task_pack_sha256": (result.get("provenance") or {}).get("task_pack_sha256"),
            "taskset_version": (result.get("provenance") or {}).get("taskset_version"),
            "initial_success": {
                name: event.get("true_at_entry")
                for name, event in (((result.get("latch") or {}).get("state") or {}).get("events") or {}).items()
            },
            "run_date": datetime.now(timezone.utc).date().isoformat(),
            "output_directory_name": Path(out_dir).name}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("task")
    parser.add_argument("--task-pack", type=Path, default=TASKS_ROOT)
    parser.add_argument("--ledger", type=Path, default=None,
                        help="default: the task's published solution/oracle_calls.jsonl")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: oracle/tasks/<task>/results/replay_<utc stamp>")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--recorder-every", type=int, default=10)
    args = parser.parse_args()
    args.task_pack = args.task_pack.resolve()
    if args.ledger is not None:
        args.ledger = args.ledger.resolve()
    if args.out is not None:
        args.out = args.out.resolve()

    rt = ROBOTWIN_ROOT
    for p in (rt, PROJECT_ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.chdir(rt)
    os.environ.setdefault("CODEACTION_CUROBO_BOUNDED_PLAN", "1")
    os.environ["CODEACTION_CUROBO_TABLE_WORLD"] = "0"

    from codeaction.benchmark.taskcard import load_task, validate_task_pack
    from codeaction.backends.robotwin.scene import load_task_scene
    import oracle._host as host

    task_pack = args.task_pack.resolve()
    pack = validate_task_pack(task_pack)
    if args.task not in pack["tasks"]:
        raise SystemExit(f"{args.task!r} is not registered in {task_pack}")
    card = load_task(args.task, tasks_root=task_pack)
    seed = int(card["protocol"]["scene_seeds"][0])
    ledger_path = (args.ledger or task_pack / args.task / "solution" / "oracle_calls.jsonl").resolve()
    calls = load_ledger(ledger_path)
    if len(calls) > card["budgets"]["max_tool_calls"]:
        raise SystemExit("ledger exceeds the task tool-call budget")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (PROJECT_ROOT / "oracle" / "tasks" / args.task / "results" / f"replay_{stamp}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out}")

    def replay(tools):
        return replay_calls(tools, calls)

    spec = {"module": "oracle.replay", "callable": "replay", "retries": 1,
            "card_key": "d0_ledger_replay", "round": "replay",
            "ledger": {"path": Path(ledger_path).name, "sha256": ledger_sha256(ledger_path),
                       "n_calls": len(calls)}}
    provenance = host._provenance(card, replay, spec, pack)
    scene = card["scene"]
    ctx = load_task_scene(scene["task_name"], seed=seed,
                          config=scene.get("config", "demo_clean_aloha"),
                          expected_env_source=scene.get("env_source"))
    try:
        result = host.run_replay_attempt(
            ctx, card, replay, out, seed=seed, attempt=1, provenance=provenance,
            recorder_every=args.recorder_every, camera=args.camera)
    finally:
        try:
            ctx["env"].close_env()
        except Exception:
            pass
    result["provenance"] = provenance
    summary = summarize(result, task=args.task, seed=seed, ledger_path=ledger_path, calls=calls,
                        out_dir=out.resolve())
    (out / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("task", "seed", "verdict", "n_replayed",
                                              "n_divergences", "n_oracle_queries", "output_directory_name")},
                     ensure_ascii=False), flush=True)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
