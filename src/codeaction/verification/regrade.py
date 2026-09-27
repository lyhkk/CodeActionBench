"""Re-score a recorded attempt against the card as it stands now, with no simulator.

A corrected predicate should not cost a GPU re-run of every attempt already recorded. What makes
that possible is that `result.json` keeps the readings the verdict was computed from: where every
card actor started and ended, its contact and functional points, both TCPs, and each latch
event's outcome. A regrade rebuilds a *recorded scene* from those numbers and hands it to
`evaluate_milestones` -- the SAME function the live verifier calls, not a second implementation
of the predicates -- so a recomputed milestone agrees with a re-run by construction wherever the
reading it needs was recorded.

What is recomputed, and what is not:

- **milestones** -- recomputed for every predicate whose reading is in the record: the position
  family (`z_above`, `z_below`, `near_actor_point`, `moved_from_start`, `rise_from_start`,
  `z_offset_from_actor`, `ordered_axis`). A predicate that needs live articulation (`joint_above`)
  or the robot's gripper (`gripper_open`, `gripper_closed`) reads nothing and comes back
  `ok: null` with the reason -- never a guess. The recorded scene raises `NotRecorded`, which is
  deliberately not an `AttributeError`, so even a `getattr(env, name, default)` inside a predicate
  fails closed rather than silently reading a zero.
- **latch gating** -- recomputed from the recorded outcomes. Which events a card REQUIRES is a
  card decision, and the recording says which ones latched, so a changed `required` list is
  regraded with no simulator. A latch is a time series, though, and a recording holds only each
  polled event's outcome: an event the episode never polled has an UNKNOWN outcome, not a failed
  one, so a card that requires a new event refuses the run instead of failing it. A card that
  redefines an existing event's predicate cannot be detected from the record at all -- the
  outcome is reused and the record says so, and the card digests below say whether the card
  moved.
- **the environment's own success predicate** -- reused verbatim from the recording, never
  recomputed: `env.check_success()` is a live call. A regrade therefore answers "what would this
  card say about that episode", and the one change it cannot serve is a correction to
  `check_success` itself (an `envs_ext` edit), which needs the episode re-run. Every output says
  which of its terms was reused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from codeaction.verification.verifiers import (_merge_latch_verdict, evaluate_milestones,
                                               summarize_milestone_funnel)

SCHEMA_VERSION = "codeaction-regrade.v1"


class NotRecorded(RuntimeError):
    """A predicate asked the scene for something the recording does not hold.

    Not an ``AttributeError`` on purpose: `evaluate_predicate` reads one optional field with
    ``getattr(env, name, 0.0)``, and an AttributeError there would be swallowed into a default of
    zero -- a wrong number instead of a refusal.
    """


class _Point:
    def __init__(self, xyz, quat=None):
        self.p = np.asarray(xyz, float)
        self.q = np.asarray(quat if quat is not None else [1.0, 0.0, 0.0, 0.0], float)


class RecordedActor:
    """One actor as the recording left it: its pose, and the declared points that moved with it."""

    def __init__(self, name, position, quaternion=None, contact_points=(), functional_points=()):
        self._name = name
        if position is None:
            raise NotRecorded(f"{name}: no position was recorded")
        self._pose = _Point(position, quaternion)
        self._contact = list(contact_points or ())
        self._functional = list(functional_points or ())

    def get_pose(self):
        return self._pose

    def _point(self, points, index, kind, what):
        try:
            value = points[int(index)]
        except (IndexError, TypeError, ValueError):
            raise NotRecorded(
                f"{self._name}: {what} {index} is not in the recording") from None
        return _Point(value) if kind else np.asarray(value, float)

    def get_contact_point(self, index, kind=None):
        return self._point(self._contact, index, kind, "contact point")

    def get_functional_point(self, index, kind=None):
        return self._point(self._functional, index, kind, "functional point")

    def __getattr__(self, name):
        # An articulated actor answers `get_qpos()` live; a recording holds a pose, not a joint.
        raise NotRecorded(f"{self.__dict__.get('_name')}.{name} is not in the recording")


class _RecordedRobot:
    def __init__(self, tcp):
        self._tcp = tcp or {}

    def _pose(self, arm):
        value = self._tcp.get(arm)
        if value is None or arm == "unlabelled":
            raise NotRecorded(f"{arm} tcp pose was not recorded")
        return np.asarray(value, float)

    def get_left_tcp_pose(self):
        return self._pose("left")

    def get_right_tcp_pose(self):
        return self._pose("right")

    def __getattr__(self, name):
        raise NotRecorded(f"robot.{name} is not in the recording")


class RecordedScene:
    """A stand-in for the live env, answering only from a recorded state snapshot.

    `_resolve_actor` reaches an actor by attribute (`roller`) or by index (`bread[0]`), so the
    scene exposes each recorded name as an attribute and a list under the bare name. Anything
    else raises, which is what keeps a regrade from inventing a reading.
    """

    def __init__(self, state):
        state = state or {}
        self._actors = {}
        positions = state.get("actor_positions") or {}
        quats = state.get("actor_quaternions") or {}
        contacts = state.get("actor_contact_points") or {}
        functionals = state.get("actor_functional_points") or {}
        indexed = {}
        for name, position in positions.items():
            try:
                actor = RecordedActor(name, position, quats.get(name),
                                      contacts.get(name), functionals.get(name))
            except NotRecorded:
                continue
            self._actors[name] = actor
            if name.endswith("]") and "[" in name:
                base, _, rest = name.partition("[")
                indexed.setdefault(base, {})[int(rest[:-1])] = actor
        for base, by_index in indexed.items():
            self._actors[base] = [by_index[i] for i in sorted(by_index)]
        self.robot = _RecordedRobot(state.get("tcp"))

    @property
    def actor_names(self):
        return tuple(sorted(name for name in self._actors if not isinstance(
            self._actors[name], list)))

    def __getattr__(self, name):
        try:
            return self.__dict__["_actors"][name]
        except KeyError:
            raise NotRecorded(f"{name!r} is not an actor in the recording") from None


def _milestone_index(records):
    return {str(record.get("name")): record for record in records or []}


def _card_digest(card):
    """The card's own digest, read from the file the loader resolved it from.

    `load_task` returns the parsed card and its directory, not a hash; a regrade names the
    verifier version it graded under, so the digest is taken here the same way `run_meta` takes
    it -- over the card file's bytes."""
    directory = card.get("dir")
    if not directory:
        return None
    path = Path(directory) / "task.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _latch_blockers(spec, latch_state):
    """Required latch events the recording cannot answer.

    A latch is polled during the episode. The record holds each POLLED event's outcome, so an
    event a later card added was never watched: its outcome is unknown, and reading unknown as
    "did not latch" would fail a run that may well have satisfied it. Refuse instead.
    """
    required = [str(name) for name in (spec.get("latch") or {}).get("required") or []]
    if not required:
        return []
    polled = set(((latch_state or {}).get("events") or {}))
    missing = [name for name in required if name not in polled]
    if not missing:
        return []
    return [f"the card requires latch event(s) {missing} that this episode never polled; "
            f"their outcome is unknown, not false"]


def regrade_result(result, card, *, recorded_card_sha256=None, final_state=None,
                   state_provenance=None, initial_poses=None):
    """Re-score one recorded attempt against `card`. Returns the regrade record."""
    spec = dict(card.get("verifier") or {})
    recorded = dict(result.get("verifier") or {})
    final_state = final_state if final_state is not None else (result.get("final_state") or {})
    latch_state = ((result.get("latch") or {}).get("state")
                   if isinstance(result.get("latch"), dict) else None)
    card_sha256 = _card_digest(card)
    head = {"schema_version": SCHEMA_VERSION, "task": result.get("task_name") or result.get("task"),
            "seed": result.get("seed"), "attempt": result.get("attempt"),
            "card_sha256": {"recorded_under": recorded_card_sha256, "regraded_under": card_sha256},
            "card_changed": (None if not (recorded_card_sha256 and card_sha256)
                             else recorded_card_sha256 != card_sha256),
            "state_provenance": state_provenance or {}}
    blockers = _latch_blockers(spec, latch_state)
    if final_state.get("error"):
        blockers.append(f"the run could not read its end state: {final_state['error']}")
    if blockers:
        return {**head, "regradable": False, "blocked_by": blockers}
    scene = RecordedScene(final_state)
    if initial_poses is None:
        initial_poses = (result.get("initial_state") or {}).get("actor_positions") or None

    milestones = evaluate_milestones(scene, spec.get("milestones") or [],
                                     initial_poses=initial_poses)
    was = _milestone_index(recorded.get("milestones"))
    rows, changed, unevaluable = [], [], []
    for record in milestones:
        name = str(record.get("name"))
        before = was.get(name)
        # `true_at_entry` is an entry-time reading; the recording holds it and a regrade has no
        # entry scene to recompute it from, so it is carried, never invented.
        if before is not None and "true_at_entry" in before:
            record["true_at_entry"] = before["true_at_entry"]
        row = {"name": name, "ok": record.get("ok"),
               "was": None if before is None else before.get("ok"),
               "detail": record.get("detail")}
        if record.get("ok") is None:
            row["offline_evaluable"] = False
            unevaluable.append(name)
            row["reason"] = (record.get("detail") or {}).get("error") or "not evaluable"
        elif before is None:
            row["status"] = "new"
            changed.append(name)
        elif bool(before.get("ok")) != bool(record.get("ok")):
            row["status"] = "changed"
            changed.append(name)
        else:
            row["status"] = "same"
        rows.append(row)
    dropped = [name for name in was if name not in {row["name"] for row in rows}]

    # The base verdict is the environment's own reading, reused: a regrade never re-runs it.
    base = {"success": bool(recorded.get("success")), "score": recorded.get("score"),
            "kind": recorded.get("kind"), "detail": dict(recorded.get("detail") or {})}
    if milestones:
        base["milestones"] = milestones
        base["milestone_counts"] = summarize_milestone_funnel(
            milestones, entry_baseline_recorded=any("true_at_entry" in m for m in milestones))
    if spec.get("latch"):
        base = _merge_latch_verdict(base, spec["latch"], latch_state)
    return {
        **head,
        "regradable": True,
        "success": {"recorded": bool(recorded.get("success")),
                    "regraded": bool(base.get("success")),
                    "env_predicate": "reused_from_recording",
                    "changed": bool(recorded.get("success")) != bool(base.get("success"))},
        "milestones": rows,
        "milestones_changed": changed,
        "milestones_not_offline_evaluable": unevaluable,
        "milestones_dropped_from_card": dropped,
        "latch": base.get("latch"),
        "latch_outcomes": "reused_from_recording",
        "milestone_counts": base.get("milestone_counts"),
    }


def find_runs(root):
    """Every recorded attempt under `root`, a run directory or a tree of them."""
    root = Path(root)
    if (root / "result.json").is_file():
        return [root]
    return sorted(path.parent for path in root.rglob("result.json"))


_GT_SNAPSHOTS = "gt_snapshots.jsonl"
_TRANSCRIPT = "transcript.jsonl"


def _snapshots(run_dir):
    """Every scene snapshot the episode server logged, in order."""
    path = Path(run_dir) / _GT_SNAPSHOTS
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("actors"), dict):
            rows.append(row)
    return rows


def _last_snapshot(run_dir):
    rows = _snapshots(run_dir)
    return rows[-1] if rows else None


def recorded_initial_poses(run_dir, result):
    """Where each actor started, for the `moved_from_start` family.

    `result["initial_state"]` is preferred; the first logged snapshot is taken at episode entry
    before any tool ran, so it is the same reading by another route.
    """
    recorded = (result.get("initial_state") or {}).get("actor_positions")
    if recorded:
        return dict(recorded), "result.initial_state"
    rows = _snapshots(run_dir)
    if not rows:
        return None, None
    first = {name: actor["xyz"] for name, actor in (rows[0].get("actors") or {}).items()
             if isinstance(actor, dict) and actor.get("xyz")}
    return (first, _GT_SNAPSHOTS) if first else (None, None)


def _recorded_tcps(run_dir):
    """Each arm's last recorded TCP, from the server transcript's own primitive trace.

    Read from `internal_trace` rather than from the tool result the model chose to return: the
    trace is written by the server and names the arm, while the model's own `result` dict uses
    whatever key it liked and often carries no arm at all. Falls back to the top-level call args
    for a direct (non-run_code) primitive.
    """
    path = Path(run_dir) / _TRANSCRIPT
    if not path.is_file():
        return {}
    tcp = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        entries = [(row.get("args"), result)]
        entries += [(entry.get("args"), entry)
                    for entry in (result.get("internal_trace") or [])
                    if isinstance(entry, dict)]
        for args, payload in entries:
            if not isinstance(payload, dict):
                continue
            pose = _pose_in(payload)
            if pose is None:
                continue
            arm = (args or {}).get("arm") if isinstance(args, dict) else None
            if arm in ("left", "right"):
                tcp[arm] = pose
            else:
                # An achieved pose whose arm cannot be read. Before the sandbox recorded each
                # primitive's arguments, a pose reached from inside run_code carried no arm at
                # all -- the model named the key. Keep it under a reserved name so a predicate
                # that asks "which arm was nearest" can still answer, while one that asks about
                # a NAMED arm still finds nothing and refuses.
                tcp.setdefault("unlabelled", []).append(pose)
    return tcp


def _pose_in(payload, depth=0):
    """The first `tcp` triple under `payload`, however the producer nested it."""
    if depth > 6 or not isinstance(payload, (dict, list)):
        return None
    items = payload.items() if isinstance(payload, dict) else enumerate(payload)
    for key, value in items:
        if key in ("tcp", "tcp_pose") and isinstance(value, list) and len(value) >= 3:
            return [float(x) for x in value[:3]]
        found = _pose_in(value, depth + 1)
        if found is not None:
            return found
    return None


def recorded_final_state(run_dir, result):
    """The end state a regrade evaluates against, and where each part of it came from.

    `result["final_state"]` is preferred and is what the module was written for. The live
    verifier stopped writing it, so the evidence the episode DOES leave -- the server's
    gt_snapshots log and its transcript -- is assembled into the same shape. The provenance is
    returned beside the state so a regrade record never implies a reading came from somewhere
    it did not.
    """
    recorded = result.get("final_state")
    if isinstance(recorded, dict) and recorded.get("actor_positions"):
        return dict(recorded), {"actors": "result.final_state", "tcp": "result.final_state"}
    snapshot = _last_snapshot(run_dir)
    if snapshot is None:
        return {}, {"actors": None, "tcp": None}
    positions, quats, contacts = {}, {}, {}
    for name, actor in (snapshot.get("actors") or {}).items():
        if not isinstance(actor, dict) or not actor.get("xyz"):
            continue
        positions[name] = actor["xyz"]
        if actor.get("quat_wxyz"):
            quats[name] = actor["quat_wxyz"]
        points = actor.get("contact_points")
        if isinstance(points, list):
            ordered = sorted((p for p in points if isinstance(p, dict) and p.get("xyz")),
                             key=lambda p: p.get("index", 0))
            if ordered:
                contacts[name] = [p["xyz"] for p in ordered]
    tcp = _recorded_tcps(run_dir)
    state = {"actor_positions": positions, "actor_quaternions": quats,
             "actor_contact_points": contacts, "tcp": tcp}
    return state, {"actors": _GT_SNAPSHOTS, "tcp": _TRANSCRIPT if tcp else None,
                   "snapshot_tick": snapshot.get("tick"), "snapshot_step": snapshot.get("step")}


def regrade_run(run_dir, *, tasks_root=None, write=True):
    """Regrade one run directory against the current card; optionally write `regrade.json`."""
    from codeaction.benchmark.taskcard import TASKS_ROOT, load_task
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    # `result["task"]` is the INSTRUCTION TEXT, never a task name -- summarize.py says so in a
    # comment and this line read it anyway, so every regrade of a real run refused with "task
    # name contains unsupported characters". The name is `task_name`; the old field is kept as a
    # fallback for any record written before the split.
    task = result.get("task_name") or result.get("task")
    card = load_task(task, tasks_root=Path(tasks_root or TASKS_ROOT))
    meta_path = run_dir / "run_meta.json"
    recorded_card = None
    if meta_path.is_file():
        try:
            recorded_card = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "task_card_sha256")
        except (OSError, UnicodeError, json.JSONDecodeError):
            recorded_card = None
    state, provenance = recorded_final_state(run_dir, result)
    initial, initial_source = recorded_initial_poses(run_dir, result)
    provenance = {**provenance, "initial_poses": initial_source}
    record = regrade_result(result, card, recorded_card_sha256=recorded_card,
                            final_state=state, state_provenance=provenance,
                            initial_poses=initial)
    record["source_run"] = str(run_dir)
    if write:
        (run_dir / "regrade.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record
