"""Replay host: scene observations, tool dispatch, recording, and out-of-band verification."""
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from codeaction.paths import PROJECT_ROOT, REPOSITORY_ROOT, ROBOTWIN_ROOT

_RT = ROBOTWIN_ROOT
for _p in (_RT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(_RT)
os.environ.setdefault("CODEACTION_CUROBO_BOUNDED_PLAN", "1")
os.environ["CODEACTION_CUROBO_TABLE_WORLD"] = "0"

from codeaction.runtime.episode_recorder import EpisodeRecorder                 # noqa: E402
from codeaction.runtime.step_observer import StepObserver                       # noqa: E402
from codeaction.benchmark.taskcard import (                                      # noqa: E402
    task_files_digest,
)
from codeaction.verification.verifiers import (                                      # noqa: E402
    LatchMonitor,
    check_destructive_actors,
    check_integrity_actors,
    run_task_verifier,
    select_latch_events,
    snapshot_actor_positions,
    snapshot_actor_quats,
    snapshot_final_state,
    snapshot_milestone_baseline,
)
from codeaction.interface.tools import ToolBox                                   # noqa: E402
from oracle._tools import ToolSurfaceCaller  # noqa: E402


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_state():
    from codeaction.launch import context
    frozen = context()
    if frozen is not None:
        return {"commit": frozen["source_commit"], "dirty": False, "dirty_paths": [],
                "execution_components": frozen["components"]}
    def _run(*args):
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], check=True,
                              text=True, capture_output=True).stdout

    if not (PROJECT_ROOT / ".git").exists():
        # Release images carry immutable build metadata instead of repository history.
        metadata = json.loads((PROJECT_ROOT / "build_info.json").read_text())
        commit = metadata["source_commit"]
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("invalid image source commit")
        return {"commit": commit, "dirty": False, "dirty_paths": []}
    # Do not strip the porcelain stream: its first leading column is the index status, and
    # stripping it corrupts the first path (" policy/..." -> "olicy/...").
    dirty_paths = [line[3:] for line in _run("status", "--porcelain").splitlines() if line]
    return {"commit": _run("rev-parse", "HEAD").strip(), "dirty": bool(dirty_paths),
            "dirty_paths": dirty_paths}


def _actor_names(verifier_spec):
    names = set(verifier_spec.get("destructive_actors") or [])
    for milestone in verifier_spec.get("milestones") or []:
        if "actor" in milestone:
            names.add(milestone["actor"])
    return sorted(names)


def _provenance(card, reference_fn, reference_spec, pack_info):
    source = inspect.getsourcefile(reference_fn)
    return {
        "git": _git_state(),
        "taskset_version": pack_info["taskset_version"],
        "task_pack_sha256": pack_info["sha256"],
        "task_card_sha256": _file_hash(Path(card["dir"]) / "task.json"),
        "task_files_sha256": task_files_digest(card["task"]["name"],
                                               tasks_root=Path(card["dir"]).parent),
        "reference": {**reference_spec,
                      "source": str(Path(source).resolve().relative_to(PROJECT_ROOT)),
                      "sha256": _file_hash(source)},
        "planner": {"CODEACTION_CUROBO_BOUNDED_PLAN":
                        os.environ.get("CODEACTION_CUROBO_BOUNDED_PLAN")},
    }


def _video_errors(out_dir, video_meta):
    errors = []
    if video_meta is None:
        return ["recorder returned no video metadata"]
    review = video_meta.get("review") or {}
    if review.get("error"):
        errors.append(f"review derivation failed: {review['error']}")
    for name in ("full.mp4", "review.mp4", "video_meta.json"):
        if not (Path(out_dir) / name).is_file():
            errors.append(f"missing required video artifact: {name}")
    return errors


def run_replay_attempt(ctx, card, replay_fn, out_dir, *, seed, attempt,
                       provenance, recorder_every=10, camera="head_camera"):
    """Replay one ledger with the standard recorder, latch, and verifier."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at, t0 = _utc_now(), time.monotonic()
    run_meta = {
        "schema_version": "0.1",
        "kind": "oracle_ledger_replay",
        "task": card["task"]["name"],
        "seed": int(seed), "attempt": int(attempt), "started_at": started_at,
        "status": "running", "scene": {"config": ctx["config"],
        "env_source": ctx["env_source"]}, **provenance,
    }
    _write_json(out_dir / "run_meta.json", run_meta)

    env, vspec = ctx["env"], card["verifier"]
    initial_poses = snapshot_actor_positions(env, _actor_names(vspec))
    # Every card actor, not just the integrity-flagged ones. Absolute orientation says nothing --
    # plenty of assets rest with their own +z lying sideways, so a bottle that never moved reads as
    # "tipped over" unless it is compared against how it STARTED. The integrity check reads this
    # dict by name and does not care that it now holds more names than it asks about.
    initial_quats = snapshot_actor_quats(env, _actor_names(vspec))
    # Entry reading of the end-state funnel, at the same boundary an agent episode takes it: a
    # milestone the initial layout already satisfies is not progress, and without this the
    # reference's funnel cannot be compared with a model's.
    milestone_baseline = snapshot_milestone_baseline(
        env, vspec.get("milestones") or [], initial_poses=initial_poses)
    latch_spec = dict(vspec.get("latch") or {})
    # Admission is where a card's own instrumentation gets exercised, so diagnostics run here
    # unconditionally: a scripted reference that reaches the goal is the positive control for
    # every mirrored latch, and its results are what justify trusting them in scored runs.
    latch_events = select_latch_events(latch_spec, diagnostics=True)
    monitor = (LatchMonitor(env, latch_events,
                            poll_every=latch_spec.get("poll_every", 5),
                            initial_poses=initial_poses)
               if latch_events else None)
    recorder = EpisodeRecorder(env, out_dir, every=recorder_every, camera=camera)
    observer = StepObserver(env)
    reference = {"ok": False, "segments": [], "detail": {}}
    infrastructure_errors = []
    video_meta = None
    attached = False

    tools_facade = None
    try:
        recorder.start()
        if monitor is not None:
            observer.add("latch", monitor.poll)
        observer.add("recorder", recorder.on_step).attach()
        attached = True
        toolbox = ToolBox(env, ctx["vp"], str(out_dir / "tools"))
        toolbox.attach_sim_step_source(lambda: observer.step_count)
        tools_facade = ToolSurfaceCaller(toolbox, log_path=out_dir / "oracle_calls.jsonl")
        value = replay_fn(tools_facade)
        if not isinstance(value, dict) or not isinstance(value.get("ok"), bool):
            raise TypeError("reference callable must return a dict with boolean 'ok'")
        reference = value
    except Exception as exc:                                           # evidence, not a lost batch
        reference["error"] = f"{type(exc).__name__}: {exc}"
        reference["traceback"] = traceback.format_exc(limit=12)
    finally:
        if attached:
            observer.detach()
        try:
            video_meta = recorder.finalize()
        except Exception as exc:
            infrastructure_errors.append(f"recorder finalize failed: {type(exc).__name__}: {exc}")

    observer_state = observer.state()
    infrastructure_errors.extend(
        f"step observer callback {name} failed: {error}"
        for name, error in observer_state.get("errors", {}).items())
    infrastructure_errors.extend(_video_errors(out_dir, video_meta))
    latch_state = monitor.state() if monitor is not None else None

    verifier = None
    try:
        verifier = run_task_verifier(vspec, env=env, initial_poses=initial_poses,
                                     latch_state=latch_state,
                                     milestone_baseline=milestone_baseline)
        if vspec.get("destructive_actors"):
            dspec = vspec.get("destructive_check") or {}
            verifier.update(check_destructive_actors(
                env, vspec["destructive_actors"], float(ctx.get("table_z", 0.74)),
                xy_bounds=dspec.get("xy_bounds"), initial_poses=initial_poses,
                min_z_m=dspec.get("min_z_m")))
        if vspec.get("integrity_actors"):
            verifier.update(check_integrity_actors(
                env, vspec["integrity_actors"], initial_quats,
                tilt_deg=float(vspec.get("integrity_tilt_deg", 60.0))))
    except Exception as exc:
        infrastructure_errors.append(f"verifier failed: {type(exc).__name__}: {exc}")

    # Read the end state at the same moment the verifier does, and never let it fail an episode:
    # this is an instrument for reading a failure, not a term in one.
    try:
        final_state = snapshot_final_state(env, _actor_names(vspec))
    except Exception as exc:                                # instrument, not a verdict
        final_state = {"error": f"{type(exc).__name__}: {exc}"}

    verifier_ok = bool(verifier and verifier.get("success"))
    destructive = bool(verifier and verifier.get("destructive"))
    success = (bool(reference.get("ok")) and verifier_ok and not destructive
               and not infrastructure_errors)
    classification = ("infrastructure_failed" if infrastructure_errors else
                      "reference_passed" if success else "reference_failed")
    result = {
        "schema_version": "0.1", "task": card["task"]["name"], "seed": int(seed),
        "attempt": int(attempt), "classification": classification, "success": success,
        "wall_s": round(time.monotonic() - t0, 3), "reference": reference,
        "initial_state": {"actor_positions": initial_poses, "actor_quaternions": initial_quats},
        "final_state": final_state,
        "verifier": verifier,
        "latch": {"required": bool(latch_spec),
                  "required_events": list(latch_spec.get("required") or []),
                  "state": latch_state},
        "observer": observer_state, "video": video_meta,
        "infrastructure_errors": infrastructure_errors,
    }
    budget = int(card["budgets"].get("max_tool_calls", card["budgets"].get("max_steps")))
    n_calls = tools_facade.n_calls if tools_facade is not None else 0
    result["oracle_gate"] = {
        "level": "replay", "n_oracle_queries": 0, "oracle_queries": [],
        "n_tool_calls": n_calls,
        "tool_calls": tools_facade.calls if tools_facade is not None else [],
        "budget_max_steps": budget, "within_tool_use_budget": n_calls <= budget,
    }
    _write_json(out_dir / "result.json", result)
    run_meta.update(status=classification, success=success, finished_at=_utc_now(),
                    wall_s=result["wall_s"])
    _write_json(out_dir / "run_meta.json", run_meta)
    # Render the calls and captured images for inspection.
    try:
        from oracle.transcript import write_run
        write_run(out_dir)
    except Exception as exc:                                       # never costs the verdict
        print(f"transcript: {type(exc).__name__}: {exc}", flush=True)
    return result
