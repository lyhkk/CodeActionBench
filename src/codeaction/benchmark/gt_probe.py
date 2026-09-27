"""Decision-time ground-truth snapshots — a DEV INSTRUMENT, never an agent-visible channel.

Why this exists. `tests/probes/grasp_decision_evidence.py` can settle the position half of "did the
agent have grounds" from run artefacts alone, and can show that a chosen orientation carries NO
object information (an exact world axis is the same vector whatever is on the table). It cannot
settle whether an orientation was RIGHT, because that is a comparison against the scene, and the
scene is exactly what the benchmark's GT wall keeps out of the agent's reach.

So the comparison is made here, on the host side, and written to a file the agent has no path to:

  * this module is imported ONLY by the sim-side bridge, never by the tool surface;
  * `snapshot()` returns a dict that its caller writes straight to disk and never merges into a
    tool result -- `test_codeaction_gt_probe.py` pins that no tool result can carry these keys;
  * the file lands beside `transcript.jsonl` in the attempt directory, which the model cannot read
    (the MCP tool surface exposes no filesystem, audited).

It is the same standing as the out-of-band verifier: privileged, analysis-only, and never a signal
the model can consume or optimize against.

Object-agnostic by construction. It records what any actor exposes -- world pose plus whatever
contact points the asset annotates -- and derives nothing task-specific. No object noun, no fixed
count, no per-task branch: every actor in every task goes through the same code path.
"""
from __future__ import annotations

import math


# An annotated contact frame is NOT a gripper frame. `_base_task.get_grasp_pose` turns one into
# the other with this fixed basis change before any planning happens, so a comparison against what
# the agent committed has to apply it too -- decoding the raw contact quaternion yields axes in a
# different frame and would produce a confident, wrong angle.
_CONTACT_TO_GRASP = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))


def _rotation(quat_wxyz):
    """Rotation matrix (world <- local) of a wxyz quaternion, as rows."""
    try:
        w, x, y, z = (float(v) for v in quat_wxyz)
    except (TypeError, ValueError):
        return None
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _column(matrix, index):
    return [matrix[row][index] for row in range(3)]


def _quat_axes(quat_wxyz):
    """Gripper-local +x (approach) and +y (opening) of a GRIPPER quaternion, in world coordinates.

    Same convention the tool surface documents: the quaternion rotates gripper-local axes into the
    world, local +x is the approach axis and local +y is the line joining the fingertips.
    """
    rotation = _rotation(quat_wxyz)
    if rotation is None:
        return None, None
    return _column(rotation, 0), _column(rotation, 1)


def _grasp_axes_from_contact(quat_wxyz):
    """Approach/opening axes of the grasp an annotated CONTACT frame stands for.

    Applies the same basis change the env applies (`_CONTACT_TO_GRASP`) so the result is in the
    gripper convention and is directly comparable with an orientation the agent committed.
    """
    contact = _rotation(quat_wxyz)
    if contact is None:
        return None, None
    grasp = tuple(
        tuple(sum(contact[r][k] * _CONTACT_TO_GRASP[k][c] for k in range(3)) for c in range(3))
        for r in range(3)
    )
    return _column(grasp, 0), _column(grasp, 1)


def _actor_entries(env):
    """Every task actor the env exposes, without knowing what any of them are."""
    seen, out = set(), []
    for name in dir(env):
        if name.startswith("_"):
            continue
        try:
            value = getattr(env, name)
        except Exception:
            continue
        if not hasattr(value, "get_pose") or not hasattr(value, "get_point"):
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        out.append((name, value))
    return out


def _contact_points(actor, limit=12):
    points = []
    for index in range(limit):
        try:
            value = actor.get_contact_point(index, "list")
        except Exception:
            break
        if value is None:
            break
        try:
            pose = [float(v) for v in value]
        except (TypeError, ValueError):
            break
        if len(pose) != 7:
            break
        approach, opening = _grasp_axes_from_contact(pose[3:])
        points.append({"index": index, "xyz": pose[:3], "quat_wxyz": pose[3:],
                       "frame": "grasp (contact annotation rebased as the env does)",
                       "approach_axis_world": approach, "opening_axis_world": opening})
    return points


def _span(points):
    """Longest separation between annotated contact points, and its direction.

    For an elongated object the annotated grasp points sit along the body, so this is a usable
    stand-in for the long axis -- stated as what it is (a span between annotations) rather than
    claimed to be a principal axis of the mesh.
    """
    best = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a, b = points[i]["xyz"], points[j]["xyz"]
            delta = [b[k] - a[k] for k in range(3)]
            norm = math.sqrt(sum(v * v for v in delta))
            if norm < 1e-9:
                continue
            if best is None or norm > best["separation_m"]:
                best = {"separation_m": norm,
                        "direction_world": [v / norm for v in delta],
                        "between": [points[i]["index"], points[j]["index"]]}
    return best


def snapshot(env, *, tick=None, step=None, tool=None):
    """One decision-time record of scene truth. Never returned to the model — see module docstring.

    Fail-safe by contract: any exception yields an `error` field instead of propagating, because a
    diagnostic must never be able to change what the episode does.
    """
    record = {"tick": tick, "step": step, "tool": tool, "actors": {}}
    try:
        for name, actor in _actor_entries(env):
            try:
                pose = actor.get_pose()
                entry = {"xyz": [float(v) for v in pose.p],
                         "quat_wxyz": [float(v) for v in pose.q]}
                points = _contact_points(actor)
                if points:
                    entry["contact_points"] = points
                    span = _span(points)
                    if span:
                        entry["annotated_span"] = span
                record["actors"][str(name)] = entry
            except Exception as exc:  # one bad actor must not lose the rest of the scene
                record["actors"][str(name)] = {"error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record
