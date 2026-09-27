#!/usr/bin/env python3
"""The readable twin of an oracle run: one turn per tool call, with everything it produced.

An agent episode leaves ``transcript.jsonl`` -- meta, one event per tool call, done, end -- and the
same video and observation images an oracle run leaves. What an oracle run lacked was the turn-by-
turn reading: ``oracle_calls.jsonl`` is the ledger, exact and unreadable, and the evidence the
sequence wrote lives in ``result.json`` with no link back to the call that produced it. This
writes both back together, the way Terminal-Bench keeps an oracle trial in the same shape as an
agent's:

    oracle_transcript.md      header (task, instruction, verdict, calls against target and
                              budget), then per turn: the call, a one-line reading of what came
                              back, the images it captured, the ideal-perception answers read off
                              those images, and the evidence records the sequence wrote before the
                              next call; then the verifier's terms and the end state
    oracle_transcript.jsonl   the same, one event per line, in the agent transcript's shape --
                              named so report tooling can never mistake it for an agent attempt

Turns own their evidence through the ``n_records`` stamp the ledger carries since 2026-09-02:
records written between two stamps belong to the earlier call. A run recorded before the stamp
existed gets its evidence in one block at the end instead, and says so.

  python oracle/transcript.py                      # every tasks/*/results/*
  python oracle/transcript.py tasks/lift_pot/results/s0
"""
import glob
import json
import os
import sys
from pathlib import Path

ORACLE = Path(__file__).resolve().parent
CARDS = ORACLE.parent / "benchmark" / "tasks"

MOTION = {"reach_tcp", "move_delta", "probe_contact_along", "set_gripper", "reach_both_tcp",
          "move_both_delta"}


def _load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def _lines(path):
    try:
        return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
                if l.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []


def _r(v, n=4):
    if isinstance(v, float):
        return round(v, n)
    if isinstance(v, (list, tuple)):
        return [_r(x, n) for x in v]
    if isinstance(v, dict):
        return {k: _r(x, n) for k, x in v.items()}
    return v


def _args_text(args):
    keep = {k: v for k, v in (args or {}).items() if k != "plane_offset_sigma_m"}
    return json.dumps(_r(keep), separators=(", ", ": "))


def _obs_ids(result):
    """Every observation id a tool result carries, in order."""
    out = []
    if not isinstance(result, dict):
        return out
    if result.get("obs_id"):
        out.append(result["obs_id"])
    for side in ("before", "after"):
        node = result.get(side)
        if isinstance(node, dict) and node.get("obs_id"):
            out.append(node["obs_id"])
    for node in result.get("observations") or []:
        if isinstance(node, dict) and node.get("obs_id"):
            out.append(node["obs_id"])
    for oid in result.get("obs_ids") or []:
        if isinstance(oid, str):
            out.append(oid)
    seen, unique = set(), []
    for oid in out:
        if oid not in seen:
            seen.add(oid)
            unique.append(oid)
    return unique


def _reading(tool, result):
    """One line on what a call came back with. Exact values, no interpretation."""
    if not isinstance(result, dict):
        return str(result)[:120]
    if tool in MOTION:
        tcp = (result.get("resulting_pose") or {}).get("tcp")
        text = f"{result.get('status')}"
        if result.get("abort_reason"):
            text += f" ({result['abort_reason']})"
        if tcp:
            text += f" -- tcp {_r(list(tcp[:3]), 3)}"
        achieved = result.get("achieved") or {}
        contact = achieved.get("contact") or achieved.get("end_contact") or {}
        fingers = contact.get("fingers_in_contact", contact.get("finger_count"))
        if fingers is not None:
            text += f", fingers in contact {fingers}"
        if achieved.get("finger_gap_m") is not None:
            text += f", finger gap {_r(achieved['finger_gap_m'], 4)} m"
        return text
    if tool in ("capture_head", "capture_wrist", "capture_evidence_views"):
        ids = _obs_ids(result)
        return f"{', '.join(ids)} (tick {result.get('tick')})" if ids else "no observation"
    if tool == "capture_motion_pair":
        v = result.get("validity") or {}
        return (f"{result.get('pair_id')} baseline {_r(result.get('camera_baseline_m'), 4)} m, "
                f"motion {'ok' if v.get('motion_succeeded') else 'failed'}, "
                f"views {', '.join(_obs_ids(result))}")
    if tool in ("plane_intersect", "triangulate_correspondence", "ray", "project",
                "scale_from_object_size", "scale_from_gripper"):
        value = result.get("value")
        unc = result.get("uncertainty")
        text = f"value {_r(value, 4)}"
        if isinstance(unc, (int, float)):
            text += f", sigma {_r(unc, 4)}"
        return text
    if tool == "check_tcp_pose_reachability":
        return (f"{'reachable' if result.get('planner_found_trajectory') else 'REFUSED'}"
                f" ({result.get('planner_status') or result.get('stage') or ''})")
    if tool == "grasp_quat_candidates":
        cands = result.get("candidates") or []
        return f"{len(cands)} candidates: " + "; ".join(
            f"{c.get('label')} {_r(c.get('quat_wxyz'), 3)}" for c in cands[:3])
    if tool == "get_grasp_contact":
        return f"fingers in contact {result.get('fingers_in_contact')}"
    if tool == "get_arm_pose":
        return f"tcp {_r(list((result.get('tcp_pose') or [])[:3]), 4)}"
    if tool == "get_gripper_state":
        return f"finger gap {_r(result.get('finger_gap_m'), 4)} m"
    if tool == "get_robot_state":
        return f"tick {result.get('tick')}"
    if tool == "draw_marks":
        return f"annotated {result.get('obs_id')}"
    if tool == "camera_aim_pose":
        pose = result.get("target_tcp_pose_world") or []
        return f"aim tcp {_r(list(pose[:3]), 3)}, valid {result.get('valid')}"
    if tool == "preview_tcp_pose":
        ann = result.get("annotation") or {}
        return f"annotated {result.get('obs_id')}, grasp centre px {_r(ann.get('grasp_center_px'), 1)}"
    if tool == "get_embodiment":
        tcp = result.get("tcp") or {}
        return f"tcp to fingertip {tcp.get('tcp_to_fingertip_plane_m')} m"
    return json.dumps(_r(result))[:140]


def _images(run_dir, obs_ids):
    """Files under tools/ for these observations, including annotated copies."""
    out = []
    for oid in obs_ids:
        for path in sorted(glob.glob(str(run_dir / "tools" / f"{oid}_*.png"))):
            out.append(os.path.relpath(path, run_dir))
    return out


def _record_text(record):
    rest = {k: v for k, v in record.items() if k not in ("name", "ok", "required")}
    body = json.dumps(_r(rest), separators=(", ", ": "))
    if len(body) > 220:
        body = body[:217] + "..."
    mark = "ok " if record.get("ok") else ("!! " if record.get("required") else "-- ")
    return f"{mark}{record['name']}  {body}"


def build(run_dir):
    run_dir = Path(run_dir)
    result = _load(run_dir / "result.json", {})
    meta = _load(run_dir / "run_meta.json", {})
    summary = _load(run_dir / "probe_summary.json", {})
    calls = _lines(run_dir / "oracle_calls.jsonl")
    if not calls or not result:
        return None
    task = result.get("task") or meta.get("task")
    card = _load(CARDS / str(task) / "task.json", {})
    instruction = (card.get("instructions") or {}).get("text") or ""
    if not instruction and (CARDS / str(task) / "instruction.md").exists():
        instruction = (CARDS / str(task) / "instruction.md").read_text(encoding="utf-8").strip()
    instruction = instruction.split("<!--")[0].strip()
    budget = (card.get("budgets") or {}).get("max_tool_calls")
    target = round(budget / 1.4) if budget else None
    records = (result.get("reference") or {}).get("segments") or []
    queries = (result.get("oracle_gate") or {}).get("oracle_queries") or []
    by_obs = {}
    for q in queries:
        if q.get("obs_id"):
            by_obs.setdefault(q["obs_id"], []).append(q)

    # The stamp appears once the sequence has handed its evidence list to a helper; the few
    # calls before that (the first capture, the embodiment card) carry none and own nothing.
    # A run with no stamp anywhere predates the mechanism.
    stamps = [c.get("n_records") for c in calls]
    stamped = any(v is not None for v in stamps)
    last = 0
    for k, v in enumerate(stamps):
        if v is None:
            stamps[k] = last
        else:
            last = v
    stamps.append(len(records))
    # An ideal-perception answer is read from an image and can always name the observation it
    # came from. One that names none was not read from anything the program could see -- the pose
    # an expert demonstration used, an orientation taken from a declared point. The renderer used
    # to index answers by observation and drop the rest, which hid them while the header went on
    # counting them, so they are collected here and shown under their own heading.
    placeless = [q for q in queries if not q.get("obs_id")]
    turns = []
    for k, c in enumerate(calls):
        owned = records[stamps[k]:stamps[k + 1]] if stamped else []
        obs = _obs_ids(c.get("result"))
        answers = []
        for oid in obs:
            for q in by_obs.get(oid, []):
                what = q.get("actor", "")
                if q.get("point_kind"):
                    what += f" {q['point_kind']} {q.get('point_index')}"
                elif q.get("semantic") == "actor_local_offset":
                    what += f" local offset {_r(q.get('local_offset'), 3)}"
                elif any(abs(v) > 1e-9 for v in (q.get("offset_world") or [])):
                    what += f" offset {_r(q.get('offset_world'), 3)}"
                text = f"{what} -> px {_r(q.get('px'), 1)} on {oid}"
                # The same point is often asked about twice -- once to see whether it is in
                # frame, once to read it -- and the second answer adds nothing to the reader.
                if text not in answers:
                    answers.append(text)
        turns.append({"step": c["i"], "tool": c["tool"], "args": c.get("args") or {},
                      "reading": _reading(c["tool"], c.get("result")),
                      "obs_ids": obs, "images": _images(run_dir, obs),
                      "oracle_answers": answers, "records": owned})
    unowned = records[:stamps[0]] if stamped else records
    return {"task": task, "seed": result.get("seed"), "round": (meta.get("reference") or {}).get("round"),
            "commit": str((meta.get("git") or {}).get("commit", ""))[:8],
            "instruction": instruction, "success": bool(result.get("success")),
            "classification": result.get("classification"),
            "level": (result.get("oracle_gate") or {}).get("level"),
            "n_calls": len(calls), "target": target, "budget": budget,
            "wall_s": result.get("wall_s"), "stamped": stamped,
            "turns": turns, "unowned_records": unowned, "placeless_answers": placeless,
            "milestones": (result.get("verifier") or {}).get("milestones") or [],
            "verifier_detail": (result.get("verifier") or {}).get("detail"),
            "final_state": result.get("final_state") or {},
            "videos": [n for n in ("review.mp4", "full.mp4") if (run_dir / n).exists()],
            "n_oracle_queries": len(queries)}


def render_markdown(t):
    verdict = "PASS" if t["success"] else "FAIL"
    lines = [f"# {t['task']} -- oracle reference run, {verdict}", ""]
    lines.append(f"- seed {t['seed']}, round {t['round']}, commit {t['commit']}, "
                 f"oracle level {t['level']}, {t['n_oracle_queries']} ideal-perception answers")
    lines.append(f"- {t['n_calls']} tool calls against a target of {t['target']} +-3 "
                 f"and a budget of {t['budget']}; wall {t['wall_s']} s")
    if t["instruction"]:
        lines.append(f"- instruction: {t['instruction']}")
    if t["videos"]:
        lines.append("- video: " + ", ".join(f"[{v}]({v})" for v in t["videos"]))
    if not t["stamped"]:
        lines.append("- recorded before the ledger stamped its evidence: the sequence's records "
                     "are listed after the turns rather than beside them")
    lines.append("")
    if t["unowned_records"] and t["stamped"]:
        lines.append("## Before the first call")
        lines += [f"- {_record_text(r)}" for r in t["unowned_records"]]
        lines.append("")
    lines.append("## Turns")
    lines.append("")
    for turn in t["turns"]:
        lines.append(f"### {turn['step']}. {turn['tool']}")
        lines.append(f"`{_args_text(turn['args'])}`  ")
        lines.append(f"-> {turn['reading']}")
        for img in turn["images"]:
            lines.append(f"![{os.path.basename(img)}]({img})")
        for a in turn["oracle_answers"]:
            lines.append(f"- perception: {a}")
        for r in turn["records"]:
            lines.append(f"- evidence: {_record_text(r)}")
        lines.append("")
    if t.get("placeless_answers"):
        lines.append("## Ideal-perception answers not read from any image")
        lines.append("")
        lines.append("These name no observation, which is the tell: a pixel is read from a frame "
                     "and can always say which one. What crossed here is orientation and offsets "
                     "from a declared point or an expert demonstration, never an absolute "
                     "position. They cost no tool call and so do not appear in the count above.")
        lines.append("")
        for q in t["placeless_answers"]:
            detail = ", ".join(f"{k} {v}" for k, v in sorted(q.items())
                               if k not in ("kind", "actor", "semantic"))
            lines.append(f"- not read from any image: {q.get('actor')} "
                         f"{q.get('semantic', q.get('kind'))} -- {detail}")
        lines.append("")
    if t["unowned_records"] and not t["stamped"]:
        lines.append("## Evidence the sequence wrote")
        lines += [f"- {_record_text(r)}" for r in t["unowned_records"]]
        lines.append("")
    lines.append("## Verdict")
    for m in t["milestones"]:
        lines.append(f"- {'ok ' if m.get('ok') else 'FAIL '}{m['name']}  "
                     f"{json.dumps(_r(m.get('detail')), separators=(', ', ': '))[:160]}")
    if t["verifier_detail"]:
        lines.append(f"- verifier: {json.dumps(_r(t['verifier_detail']))[:200]}")
    fs = t["final_state"]
    if fs.get("actor_positions"):
        lines.append(f"- final actor positions: {json.dumps(_r(fs['actor_positions']))}")
    if fs.get("tcp"):
        lines.append(f"- final tcp: {json.dumps(_r(fs['tcp']))}")
    lines.append("")
    return "\n".join(lines)


def render_jsonl(t):
    events = [{"event": "meta", "kind": "oracle_reference", "task": t["task"], "seed": t["seed"],
               "round": t["round"], "commit": t["commit"], "instruction": t["instruction"],
               "max_tool_calls": t["budget"], "target_calls": t["target"],
               "oracle_level": t["level"]}]
    for turn in t["turns"]:
        events.append({"event": "tool", "step": turn["step"], "tool": turn["tool"],
                       "args": turn["args"], "reading": turn["reading"],
                       "obs_ids": turn["obs_ids"], "images": turn["images"],
                       "oracle_answers": turn["oracle_answers"], "records": turn["records"]})
    events.append({"event": "end", "status": t["classification"], "success": t["success"],
                   "steps": t["n_calls"], "milestones": t["milestones"],
                   "final_state": t["final_state"]})
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"


def write_run(run_dir):
    t = build(run_dir)
    if t is None:
        return None
    run_dir = Path(run_dir)
    (run_dir / "oracle_transcript.md").write_text(render_markdown(t), encoding="utf-8")
    (run_dir / "oracle_transcript.jsonl").write_text(render_jsonl(t), encoding="utf-8")
    return run_dir / "oracle_transcript.md"


def main(argv):
    targets = argv or sorted(glob.glob(str(ORACLE / "tasks" / "*" / "results" / "*")))
    n = 0
    for d in targets:
        if (Path(d) / "oracle_calls.jsonl").exists():
            out = write_run(d)
            if out:
                n += 1
                print(out)
    print(f"{n} transcript(s) written")


if __name__ == "__main__":
    main(sys.argv[1:])
