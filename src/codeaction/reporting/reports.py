"""Generate per-run HTML execution-analysis reports + an index for the codeaction live runs.
Focus: actual execution (no theory) — every intermediate step, the model's per-turn text (when
recorded), per-turn images labeled by camera, scale derivations, failure→adjustment sequences,
and a computed metric card (spec §12 v5 subset). Images are referenced RELATIVELY so the whole
data/ tree can be rsynced to the local machine and browsed via file://.

AUTO-INGEST (no hardcoded run list): every run directory that contains a `transcript.jsonl` is
discovered automatically; its human-facing name/tag/verifier come from a sibling `run_meta.json`
that a run script writes via `finalize_run(...)`. A run script ends with ONE call —
`finalize_run(run_dir, name, tag, verifier)` — which stamps the meta and rebuilds the whole
report; nothing else is manual on the sim box.

Standalone rebuild: "$ROBOTWIN_PYTHON" .../make_reports.py
"""
import datetime as dt
import hashlib
import html
import json
import os
import sys
from pathlib import Path

from codeaction.paths import PROJECT_ROOT
from codeaction.verification.metrics import resource_accounting


BASE = Path(os.environ.get("CODEACTION_RUNS_ROOT", PROJECT_ROOT / "runs")).resolve()
OUT = BASE / "runs_report"
from codeaction.contracts.failures import failure_of
from codeaction.batch.results import (  # noqa: E402
    SUBMISSION_MANIFEST_NAME,
    BatchResultsError,
    load_submission_manifest,
)
CAM = {"head_camera": "头相机", "left_camera": "左腕相机", "right_camera": "右腕相机"}
# Milestone provenance glyphs, shared by the per-attempt funnel and the pooled cell row so the two
# can never drift: `=` mirrors an env.check_success conjunct verbatim, `~` was hand-derived from
# one and can disagree with it, `*` is our own instrument and makes no claim about the task's
# success condition. Vocabulary and card contract: codeaction.benchmark.taskcard.validate_card_verifier.
_PROV = {"check_success_conjunct": "=", "check_success_derived": "~",
         "benchmark_instrument": "*"}
SCALE_TOOLS = {"ray", "horizontal_plane_intersect", "table_plane_intersect",  # old name: pre-rename runs
               "capture_motion_pair", "scale_from_motion", "triangulate_correspondence",
               "scale_from_object_size", "scale_from_gripper", "project"}

MOTION_TOOLS = {"move_delta", "move_both_delta", "probe_contact_along", "probe_contact_z",
                "reach_tcp", "reach_both_tcp", "open_gripper", "close_gripper", "set_gripper",
                "aim_camera",
                "capture_motion_pair", "scale_from_motion"}

# Frozen backfill for the runs that pre-date run_meta.json (written once, idempotently, then the
# report is pure auto-discovery). NOT a list you edit for new runs — new runs self-register via
# finalize_run(); this only names the early ones that were produced before that existed.
_BACKFILL = {
    "stage3_v2": ("pick_red_box · qwen3.7-plus · tool-use · v2",
                  "全记录复跑(模型思考文本 + 每动作上帝视角)", None),
    "stage3": ("pick_red_box · qwen3.7-plus · tool-use", "stage-3 首个 live 成功",
               {"success": True, "score": 1.0, "detail": {"rise_m": 0.1519}}),
    "stage4/blue_box": ("pick_blue_box · qwen3.7-plus · tool-use", "stage-4 泛物体", None),
    "stage4/yellow_cylinder": ("pick_yellow_cylinder · qwen3.7-plus · tool-use",
                               "stage-4 泛物体(物体被打落)", None),
    "stage4/green_box": ("pick_green_box · qwen3.7-plus · tool-use", "stage-4 泛物体", None),
    "stage4_hybrid/blue_box": ("pick_blue_box · qwen3.7-plus · hybrid", "run_code 可用但 0 次采用", None),
    "stage4_hybrid/yellow_cylinder": ("pick_yellow_cylinder · qwen3.7-plus · hybrid",
                                      "run_code 可用但 0 次采用", None),
    "stage4_hybrid/green_box": ("pick_green_box · qwen3.7-plus · hybrid",
                                "run_code 可用但 0 次采用", None),
}


def register_run(run_dir, name, tag, verifier=None, interface=None, extra_meta=None):
    """Stamp a run directory with its human-facing metadata (auto-discovery reads this). Idempotent
    for callers that pass the same values."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                meta.update(existing)
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    if extra_meta is not None:
        if not isinstance(extra_meta, dict):
            raise TypeError("extra_meta must be a dict")
        meta.update(extra_meta)
    meta.update({"name": name, "tag": tag})
    if verifier is not None:
        meta["verifier"] = verifier
    if interface is not None:
        meta["interface"] = interface
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_name(sub):
    parts = sub.split("/")
    return " · ".join(reversed(parts)) if len(parts) > 1 else sub


def _is_codeaction_run(tp):
    """A codeaction-runner transcript starts with a `meta` event carrying `tools` — this signature
    excludes older servo/LoggingVLM logs that also happen to be named transcript.jsonl."""
    try:
        with open(tp, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    j = json.loads(line)
                    return j.get("event") == "meta" and "tools" in j
    except Exception:
        return False
    return False


def _preferred_transcript(run_dir):
    """Use the reference-agent transcript when present: it is the complete model+tool record.

    Docker reference runs also contain the episode-server ``transcript.jsonl`` used for the
    controller attestation.  That copy intentionally has no model reasoning or provider usage, so
    rendering it would produce a technically valid but materially incomplete detail page.
    """
    run_dir = Path(run_dir)
    reference = run_dir / "reference_transcript.jsonl"
    if reference.exists() and _is_codeaction_run(reference):
        return reference
    return run_dir / "transcript.jsonl"


def _identity_labels(meta, sub):
    """Canonical labels for identity-bearing attempts.

    These attempts may retain a generic host-authored name such as ``codeaction-reference``.  The
    report must instead show the tested model, scaffold version, delivered interface, and exact
    trial so five attempts never appear as five indistinguishable rows.
    """
    identity = meta.get("identity") or meta.get("expected_identity")
    if not isinstance(identity, dict):
        return None
    comparison = identity.get("comparison")
    trial = identity.get("trial")
    if not isinstance(comparison, dict) or not isinstance(trial, dict):
        return None
    task = (comparison.get("task") or {}).get("id") or meta.get("task_name")
    model = (comparison.get("model") or {}).get("id") or meta.get("model")
    tested = comparison.get("tested_unit") or {}
    driver = tested.get("driver") or {}
    driver_id = driver.get("id") or meta.get("agent_cli") or meta.get("agent_label")
    driver_version = driver.get("version") or meta.get("agent_cli_version")
    interface = tested.get("interface_profile") or meta.get("interface_profile") \
        or meta.get("interface")
    attempt = trial.get("attempt_index", meta.get("attempt_index"))
    seed = trial.get("scene_seed", meta.get("seed"))
    if not task or not model:
        return None
    scaffold = str(driver_id or "scaffold")
    if driver_version:
        scaffold += "@" + str(driver_version)
    trial_label = []
    if attempt is not None:
        trial_label.append(f"a{int(attempt) + 1}")
    if seed is not None:
        trial_label.append(f"s{seed}")
    name_parts = [str(task), str(model), scaffold, str(interface or "?")]
    if trial_label:
        name_parts.append("/".join(trial_label))
    driver_kind = ((tested.get("driver") or {}).get("kind")) or "?"
    profile = meta.get("profile") or "?"
    batch = Path(sub).parent.name if "/" in sub else sub
    tag_parts = [f"driver={driver_kind}", f"profile={profile}", f"batch={batch}"]
    blockers = tested.get("non_submittable_reasons") or meta.get("non_submittable_reasons") or []
    if blockers:
        tag_parts.append("release blocker=" + ",".join(map(str, blockers)))
    return " · ".join(name_parts), " · ".join(tag_parts)


def _release_set_root(run_dir):
    """The release set this attempt was filed into, if any.

    A release set is assembled by COPYING attempts, so one episode exists at its source batch and
    again under every set that filed it. Membership is what decides which copy the report shows.
    """
    for parent in Path(run_dir).parents:
        if parent == BASE:
            break
        if (parent / "release_set.v1.json").is_file():
            return parent
    return None


def _release_cell_of(run_dir, release_root):
    """(model, task) from a filed attempt's path: <set>/runs/<model>/<task>/attempt-.../run/..."""
    try:
        parts = Path(run_dir).relative_to(release_root).parts
    except ValueError:
        return None, None
    if len(parts) >= 3 and parts[0] == "runs":
        return parts[1], parts[2]
    return None, None


def _episode_identity(run_dir):
    """A key equal for copies of one episode and different for different episodes.

    Filing preserves the whole tail below the set root -- `<model>/<task>/attempt-NNN/
    execution-NNN/run/<attempt-dir>` -- so that tail is identical for copies and differs for a
    different attempt or execution of the same cell. It is paired with the `result.json` digest,
    which is copied byte-for-byte and carries this episode's own wall clock and verifier
    readings, so two unrelated runs that happen to share a tail still separate.

    Neither half is sufficient alone: the digest alone merged distinct attempts whose results
    were identical, and the tail alone would merge a re-run of the same cell.
    """
    run_dir = Path(run_dir)
    tail = "/".join(run_dir.parts[-6:])
    result = run_dir / "result.json"
    try:
        digest = hashlib.sha256(result.read_bytes()).hexdigest()
    except OSError:
        return "path:" + str(run_dir)
    return f"cell:{tail}:{digest}"


def _copy_rank(run_dir):
    """Which copy of one episode the report should keep: the release set the user browses.

    Ordering is (is release member, newest set) so the canonical row points at the set being
    published rather than at whichever source batch happened to be discovered first. "Newest" is
    the set's own recorded build time -- ranking by directory NAME put `codeaction-release-round2`
    above `codeaction-release-5x7`, because 'r' sorts after '5'.
    """
    root = _release_set_root(run_dir)
    if root is None:
        return (0, 0.0)
    document = root / "release_set.v1.json"
    try:
        generated = json.loads(document.read_text(encoding="utf-8")).get("generatedAt")
        built = dt.datetime.fromisoformat(str(generated)).timestamp()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        try:
            built = document.stat().st_mtime
        except OSError:
            built = 0.0
    return (1, built)


def discover_runs():
    """Every directory under data/ whose transcript.jsonl is a codeaction-runner log is a run (newest
    first by mtime). name/tag/verifier come from its run_meta.json; a run with no meta still shows
    with a path-derived name so nothing is ever silently dropped.

    One episode yields ONE row. Release sets file copies, so the same attempt used to appear once
    per set plus once at its source batch -- 457 rows for far fewer episodes, and the same video
    reachable under several path-mangled names.
    """
    seen, runs = set(), []
    candidates = []
    for tp in BASE.rglob("transcript.jsonl"):
        d = tp.parent
        if "runs_report" in d.parts or not _is_codeaction_run(tp):
            continue
        candidates.append((_episode_identity(d), _copy_rank(d), d, tp))
    chosen = {}
    for identity, rank, d, tp in candidates:
        if identity not in chosen or rank > chosen[identity][0]:
            chosen[identity] = (rank, d, tp)
    for _, d, tp in sorted(chosen.values(), key=lambda item: str(item[1])):
        if d in seen:
            continue
        seen.add(d)
        sub = str(d.relative_to(BASE))
        run = {"sub": sub, "name": _default_name(sub), "tag": ""}
        try:
            first = json.loads(tp.read_text(encoding="utf-8").splitlines()[0])
            if first.get("git_commit"):
                run["git_commit"] = first["git_commit"]
                run["git_dirty"] = bool(first.get("git_dirty"))
        except Exception:
            pass
        mp = d / "run_meta.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
                run["name"] = meta.get("name") or run["name"]
                run["tag"] = meta.get("tag") or ""
                identity_labels = _identity_labels(meta, sub)
                if identity_labels is not None:
                    run["name"], run["tag"] = identity_labels
                if "verifier" in meta:
                    run["verifier"] = meta["verifier"]
                # A run_meta version is accepted only when it was explicitly captured by the run;
                # report rebuilds never stamp their current HEAD onto an older episode.
                if meta.get("git_commit") and "git_commit" not in run:
                    run["git_commit"] = meta["git_commit"]
                    run["git_dirty"] = bool(meta.get("git_dirty"))
                elif meta.get("source_commit") and "git_commit" not in run:
                    run["git_commit"] = meta["source_commit"]
                    run["git_dirty"] = bool(meta.get("source_dirty"))
            except Exception:
                pass
        ap = d / "archive_meta.json"
        if ap.exists():
            try:
                archive_meta = json.loads(ap.read_text(encoding="utf-8"))
                if archive_meta.get("archive") is True:
                    run["archive_reason"] = str(archive_meta.get("reason") or
                                                "explicit_infra_archive")
                    run["archive_detail"] = str(archive_meta.get("detail") or "")
            except Exception:
                pass
        ep = d / "agent_exit.json"
        if ep.exists():
            try:
                agent_exit = json.loads(ep.read_text(encoding="utf-8"))
                if isinstance(agent_exit, dict):
                    run["agent_exit"] = agent_exit
            except Exception:
                pass
        # A release member is what the user actually browses, so it gets a short name led by the
        # set: the path-derived default reads
        # `benchmark_batches__codeaction-release-5x7__runs__gpt-5.6__scan_object__attempt-000__...`,
        # which buries the two fields anyone scans for.
        release_root = _release_set_root(d)
        if release_root is not None:
            model, task = _release_cell_of(d, release_root)
            if model and task:
                run["release_set"] = release_root.name
                run["name"] = f"{release_root.name} · {task} · {model}"
                run["tag"] = " · ".join(
                    part for part in (f"release={release_root.name}", run.get("tag")) if part)
        mt = tp.stat().st_mtime
        run["mtime"] = mt
        runs.append((mt, run))
    runs.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in runs]


def backfill_legacy():
    """Write run_meta.json for the pre-registration runs, only where missing (idempotent)."""
    for sub, (name, tag, ver) in _BACKFILL.items():
        d = BASE / sub
        if (d / "transcript.jsonl").exists() and not (d / "run_meta.json").exists():
            register_run(d, name, tag, ver)

CSS = """body{font-family:-apple-system,'PingFang SC',sans-serif;margin:0;background:#f5f6f8;color:#1c2733}
.wrap{max-width:1060px;margin:0 auto;padding:24px}h1{font-size:22px}h2{font-size:17px;margin-top:28px;
border-bottom:2px solid #dde3ea;padding-bottom:6px}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border:1px solid #dde3ea;padding:6px 9px;text-align:left;vertical-align:top}th{background:#eef1f5}
.step{background:#fff;border:1px solid #dde3ea;border-radius:8px;padding:10px 14px;margin:10px 0}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;color:#fff}
.S{background:#2e9e5b}.F{background:#d64545}.A{background:#e08a00}.E{background:#8a56c9}.I{background:#5b7b9d}
.args{color:#5b6b7c;font-size:12px;font-family:menlo,monospace;white-space:pre-wrap;word-break:break-all}
.res{font-size:12px;font-family:menlo,monospace;color:#31435a;white-space:pre-wrap;word-break:break-all;
background:#f2f5f9;padding:6px;border-radius:6px;margin-top:6px}
.think{background:#fffbe8;border-left:3px solid #e0c400;padding:6px 10px;font-size:13px;margin:6px 0}
img{max-width:315px;border:1px solid #c8d1dc;border-radius:6px;margin:4px}
.episode-video{display:block;width:100%;max-height:70vh;background:#111;border-radius:8px}
.video-links{margin:8px 0}.sim-range{color:#2a6fbb;font-size:11px;font-family:menlo,monospace}
.imglab{font-size:11px;color:#5b6b7c;text-align:center}.imgbox{display:inline-block;vertical-align:top}
.ok{color:#2e9e5b;font-weight:700}.bad{color:#d64545;font-weight:700}.note{color:#7a8a9a;font-size:12px}
a{color:#2a6fbb;text-decoration:none}a:hover{text-decoration:underline}
details{margin:6px 0}summary{cursor:pointer;font-weight:600;padding:6px 10px;border-radius:6px;
user-select:none;list-style:none}summary::-webkit-details-marker{display:none}
summary::before{content:'▸ ';color:#7a8a9a}details[open]>summary::before{content:'▾ '}
.lvl-date>summary{font-size:16px;background:#e7edf4;border:1px solid #d0dae6;margin-top:10px}
.lvl-task{margin-left:16px}.lvl-task>summary{font-size:13.5px;background:#f0f3f7}
.lvl-task table{margin:4px 0 4px 2px}.cnt{color:#7a8a9a;font-weight:400;font-size:12px;margin-left:8px}
.budget{color:#8a56c9;font-size:12px;font-weight:600}.entry{color:#1c2733}
.pill{display:inline-block;font-size:11px;padding:0 7px;border-radius:9px;margin-left:6px}
.pill.ok{background:#e3f4ea;color:#1d7a43}.pill.bad{background:#fbe6e6;color:#b02a2a}
.archive{margin-left:16px}.archive>summary{font-size:13.5px;background:#f4f0e8;color:#715c2c}
.pill.archive-pill{background:#eee7d9;color:#715c2c}"""


def esc(x):
    return html.escape(str(x))


def badge(status):
    m = {"SUCCESS": "S", "FAILED": "F", "ABORTED": "A", "ERR": "E"}
    return f'<span class="badge {m.get(status, "I")}">{esc(status)}</span>'


def result_status(tool, result):
    """Normalize wrapper results without confusing an explicit null error with a failure."""
    if result.get("status"):
        return result["status"]
    if tool == "run_code":
        return "SUCCESS" if result.get("ok") is True else "ERR"
    return "ERR" if "error" in result else "ok"


def result_is_error(tool, result):
    return result_status(tool, result) == "ERR"


def result_is_success(tool, result):
    return result_status(tool, result) == "SUCCESS"


def internal_trace_rows(result):
    """Normalize the compact run_code trace for reporting; old runs may only carry tool/ok.

    If an older trace lacks tick but its final action is also returned as ``value``, recover that
    final tick without inventing ticks for earlier internal calls.
    """
    rows = [dict(row) for row in (result.get("internal_trace") or [])
            if isinstance(row, dict)]
    value = result.get("value")
    if rows and isinstance(value, dict) and isinstance(value.get("tick"), int):
        last = rows[-1]
        if last.get("tick") is None and (last.get("action_id") is None
                                         or last.get("action_id") == value.get("action_id")):
            last["tick"] = value["tick"]
    return rows


CANCELLED_EVENTS = frozenset({
    "tool_cancelled_after_action_abort",
    "tool_cancelled_after_recoverable_abort",  # transcript schema <= 2.17
})


def executed_tool_records(lines):
    """Return executed calls, including collision-ending calls added by transcript schema 2.6."""
    return [
        line for line in lines
        if line.get("event") == "tool"
        or (
            line.get("event") == "unintended_collision"
            and isinstance(line.get("args"), dict)
            and isinstance(line.get("result"), dict)
        )
    ]


def cancelled_calls_by_preceding_record(lines):
    """Map each executed record to the same-turn calls the safety barrier cancelled after it.

    These calls consume episode budget but never reach the simulator, so they must NOT enter
    `executed_tool_records` (every motion, tick and redundancy aggregate reads that list). They
    are rendered as annotations on the aborting step instead, because a report whose step count
    silently disagrees with `budget_used` is unreadable.
    """
    out, last = {}, None
    for line in lines:
        if line.get("event") == "tool" or line.get("event") == "unintended_collision":
            last = line
        elif line.get("event") in CANCELLED_EVENTS and last is not None:
            out.setdefault(id(last), []).append(line)
    return out


def find_images(rundir, res):
    out = []
    if not isinstance(res, dict):
        return out
    observations = [("", res)] if "obs_id" in res else []
    if isinstance(res.get("before"), dict):
        observations.append(("运动前", res["before"]))
    if isinstance(res.get("after"), dict):
        observations.append(("运动后", res["after"]))
    post_collision = res.get("post_collision")
    if isinstance(post_collision, dict) and isinstance(
            post_collision.get("observation"), dict):
        observations.append(("碰撞后稳定", post_collision["observation"]))
    interrupted = res.get("interrupted_action")
    if isinstance(interrupted, dict):
        nested = interrupted.get("post_collision")
        if isinstance(nested, dict) and isinstance(nested.get("observation"), dict):
            observations.append(("碰撞后稳定", nested["observation"]))
    # Resolve from the detail-page directory instead of assuming a fixed run depth. Imported
    # codeaction attempts are one level deeper than the legacy batch/attempt layout.
    rel = Path(os.path.relpath(rundir, BASE / "runs_report")).as_posix()
    for phase, observation in observations:
        oid, cam = observation.get("obs_id"), observation.get("camera", "")
        if not oid:
            continue
        for fn in (f"{oid}_{cam}.png", f"{oid}.png"):
            p = rundir / "tools" / fn
            if p.exists():
                lab = CAM.get(cam, cam) + (" · 标注帧" if fn == f"{oid}.png" else "")
                phase_label = f"{phase} · " if phase else ""
                out.append((f"{rel}/tools/{fn}", f"{phase_label}{oid} · {lab}"))
    return out


def observer_images(rundir, tick, camera):
    """Return legacy and action-indexed evaluator frames for one action tick."""
    root = rundir / "tools" / "observer"
    legacy = root / f"tick_{tick:03d}_{camera}.png"
    found = [legacy] if legacy.is_file() else []
    found.extend(sorted(root.glob(f"tick_{tick:03d}_*_{camera}.png")))
    return list(dict.fromkeys(found))


def motion_result(tool, result):
    """Return the ActionResult payload for observer rendering, including pair acquisitions."""
    if tool in {"capture_motion_pair", "scale_from_motion"} \
            and isinstance(result.get("motion"), dict):
        return result["motion"]
    return result


def load(sub):
    d = BASE / sub
    lines = []
    tp = _preferred_transcript(d)
    if tp.exists():
        lines = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines()]
    return d, lines


def _batch_of_sub(sub):
    """Batch directory for both legacy ``batch/a1`` and nested imported result layouts."""
    parts = str(sub).split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else parts[0]


def verifier_of(run):
    if "verifier" in run:
        return run["verifier"]
    single = BASE / run["sub"] / "result.json"          # single-attempt layout
    if single.exists():
        return json.loads(single.read_text(encoding="utf-8")).get("verifier",
                                                                  {"success": None, "detail": {}})
    grp = run["sub"].split("/")[0]
    rj = BASE / grp / "results.json"                    # legacy variant-batch layout
    if rj.exists():
        for rec in json.loads(rj.read_text(encoding="utf-8")):
            if rec["variant"] == run["sub"].split("/")[1]:
                return rec["verifier"]
    return {"success": None, "detail": {}}


def result_of(run):
    """The whole result.json for a run (both layouts), or {} when absent. `verifier_of` returns
    only the verifier sub-object; report rows that read siblings of it (step_observer) need this."""
    single = BASE / run["sub"] / "result.json"
    if single.exists():
        try:
            return json.loads(single.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    grp, _, variant = run["sub"].partition("/")
    rj = BASE / grp / "results.json"
    if rj.exists():
        try:
            for rec in json.loads(rj.read_text(encoding="utf-8")):
                if rec.get("variant") == variant:
                    return rec
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def reason_of(tool, r):
    """A full, unabbreviated Chinese explanation for every non-SUCCESS action result."""
    if r.get("status") == "ABORTED":
        ar = str(r.get("abort_reason") or "")
        if ar.startswith("workspace"):
            return ("安全护栏拒绝:目标点在工作空间安全包围盒之外,动作在触碰仿真之前就被拒绝,"
                    "手臂完全没有移动(" + ar + ")")
        if ar == "arm_lock":
            return ("安全护栏拒绝:另一只手臂正被指令占用(默认一次只允许命令一只手臂),"
                    "本次动作未执行")
        if ar == "contact":
            return ("接触中止:受监护的移动过程中手指出现了新的物理接触,运动被立即停止。"
                    "这是安全护栏的合法行为——向下探底时,这正是期望的『碰到了』停止信号,"
                    "不是错误")
        return ar
    if r.get("status") == "FAILED":
        a = r.get("achieved") or {}
        why = a.get("reason")
        if tool == "aim_camera":
            return (f"俯仰角扫描中没有找到既可达、又能把目标点纳入腕相机视野的手臂位姿"
                    f"(plan_ok={a.get('plan_ok')}, in_view={a.get('in_view')}, "
                    f"目标距图像中心 {a.get('cdist_px')} 像素)")
        if why and "leg plan failed" in str(why):
            return ("某一段约束直线腿的运动规划失败:cuRobo 找不到无碰撞的关节轨迹"
                    "(常见原因:目标过低触及桌面碰撞模型 z≈0.76、该姿态接近关节极限)。"
                    "achieved 里报告了已真实走过的位移")
        if why and "stalled" in str(why):
            return ("手臂连续两腿几乎没有物理进展(顶住了碰撞代价墙或达到跟踪极限),"
                    "系统诚实报告为失败而不是假装完成")
        if why and "budget" in str(why):
            return ("在腿数预算内没能交付指令位移(每一腿都在向绝对目标重新瞄准,"
                    "但仍未收敛到目标点)")
        if why:
            return str(why)
        return ("cuRobo 运动规划失败:目标位姿不可达、接近关节极限、或与桌面碰撞模型/自身冲突。"
                "(这一批早期 run 未记录更细分的原因;新 run 已在 achieved.reason 里记录)")
    if tool == "run_code" and result_is_error(tool, r):
        err = r.get("error")
        return ("run_code 执行失败,原文:" + str(err)) if err else \
            "run_code 返回结构无效:缺少 ok:true,且没有可用的错误原文"
    if result_is_error(tool, r):
        return "工具调用本身无效(参数错误/引用了不存在的观测等),原文:" + str(r.get("error"))
    return None


def _attested_reference_stats(result):
    identity = result.get("identity_attestation")
    if not isinstance(identity, dict) or identity.get("reference_agent_match") is not True:
        return None
    reference = identity.get("reference_agent")
    if not isinstance(reference, dict) or reference.get("healthy") is not True:
        return None
    stats = reference.get("stats")
    return stats if isinstance(stats, dict) else None


def metrics_of(lines, ver, result=None, agent_exit=None, motion_records=None):
    """`result` = the whole result.json (optional). Needed for keys that are SIBLINGS of
    `verifier` rather than inside it — currently `step_observer`, whose disabled-callback list
    decides whether the latch readings below can be trusted at all."""
    result = result or {}
    tools = executed_tool_records(lines)
    end = next((l for l in lines if l["event"] == "end"), {})
    resource_stats = _attested_reference_stats(result) or end
    resources = resource_accounting(resource_stats, lines)
    done = next((l for l in lines if l["event"] == "done"), None)
    ab = {}
    reach_fail = {}
    fails = errs = 0
    rep = mx = 0
    prev = None
    for t in tools:
        r = t.get("result") or {}
        if r.get("status") == "ABORTED":
            ab[str(r.get("abort_reason"))] = ab.get(str(r.get("abort_reason")), 0) + 1
        if r.get("status") == "FAILED":
            fails += 1
            if t.get("tool") == "reach_tcp":
                cat = ((r.get("achieved") or {}).get("failure_category")
                       or (r.get("achieved") or {}).get("reason")
                       or "unknown")
                reach_fail[str(cat)] = reach_fail.get(str(cat), 0) + 1
        if result_is_error(t.get("tool"), r):
            errs += 1
        key = (t["tool"], json.dumps(t.get("args"), sort_keys=True))
        rep = rep + 1 if key == prev else 0
        mx = max(mx, rep + 1)
        prev = key
    claim = (done or {}).get("report", {}).get("success_claim") if done else None
    truth = ver.get("success")
    claim_row = ("—" if claim is None else
                 ('<span class="ok">一致</span>' if claim == truth else
                  '<span class="bad">虚报</span>' if claim and not truth else
                  '<span class="bad">漏报</span>'))
    rc = [t for t in tools if t["tool"] == "run_code"]
    extra = {}
    failure = failure_of(result, agent_exit)
    if failure is not None:
        disposition = "计分" if failure.scoreable else "排除"
        extra["失败归因"] = esc(
            f"{failure.origin.value}/{failure.code.value} · {disposition}"
            + (f" · {failure.detail_safe}" if failure.detail_safe else ""))
    try:                     # result-aware motion metrics (L2) — never break the report
        from codeaction.verification.metrics import (contact_pose_telemetry, contact_safety_stats, motion_stats,
                                     reachability_probe_telemetry, stall_telemetry)
        # Safety axis, reported beside the score and never inside it. The generic guard histogram
        # below cannot replace it: it reads top-level abort_reason only, so a run_code block that
        # the contact monitor killed does not appear there at all.
        cs = contact_safety_stats(lines)
        if cs["unexpected_contact_aborts"] or cs["legacy_collision_terminals"]:
            parts = []
            if cs["unexpected_contact_aborts"]:
                parts.append(f'{cs["unexpected_contact_aborts"]} 次' + (
                    f'(其中 run_code 内 {cs["run_code_unexpected_contact_interruptions"]})'
                    if cs["run_code_unexpected_contact_interruptions"] else ""))
            if cs["legacy_collision_terminals"]:
                # Tool surface 6.x-8.x semantics: the first contact ENDED the attempt.
                parts.append(f'历史终局记录 {cs["legacy_collision_terminals"]}')
            extra["接触中止(独立安全轴·不计入成功率)"] = " · ".join(parts)
        if cs["contact_read_unavailable_aborts"]:
            # Unknown contact state stopped the arm. Not a contact, and not evidence of none.
            extra["接触状态不可读中止"] = cs["contact_read_unavailable_aborts"]
        m = motion_stats(lines)
        if m["n_motion"]:
            extra["浪费动作率(FAILED或~0位移)"] = (f'{m["n_wasted"]}/{m["n_motion"]}'
                                                   f' = {m["wasted_motion_rate"]}')
            extra["最长无进展连段(result-aware)"] = m["max_no_progress_streak"]
        reach = reachability_probe_telemetry(lines)
        extra["可达性规划查询"] = reach["reachability_queries"]
        if reach["reachability_query_z_spread_m"] is not None:
            extra["可达性查询z跨度(m)"] = reach["reachability_query_z_spread_m"]
        contact_pose = contact_pose_telemetry(lines)
        if contact_pose["contact_pose_observations"]:
            extra["触觉接触位姿观察(分析项)"] = contact_pose["contact_pose_observations"]
            nonzero_parts = {
                part: count for part, count in
                contact_pose["contact_pose_observations_by_part"].items() if count}
            extra["触觉接触自身部位计数"] = nonzero_parts
            spans = contact_pose["contact_pose_axis_spread_m"]
            if any(value is not None for value in spans.values()):
                extra["触觉接触TCP轴向跨度(m)"] = spans
        stall = stall_telemetry(motion_records, ver)
        if stall["motion_trace_available"]:
            extra["stall physics steps"] = stall["stall_physics_steps"]
            extra["stall 后继续物理动作"] = (
                f'{stall["continued_physical_action_after_stall"]}'
                f' · 后续 {stall["physical_actions_after_first_stall"]} actions')
            extra["成功发生在 stalled action 内"] = (
                "未观察到 success latch" if stall["success_in_stalled_action"] is None
                else str(stall["success_in_stalled_action"]))
    except Exception:
        pass
    if isinstance(ver.get("milestones"), list):
        # `[入口即真]` marks a milestone the scene satisfied before the agent acted. It may still
        # be a legitimate TERMINAL requirement (a released gripper is a real check_success
        # conjunct) but it is never evidence of progress, which is why the row shows the marker
        # instead of folding it into a passed count.
        def _mark(mm):
            glyph = "✓" if mm.get("ok") else ("?" if mm.get("ok") is None else "✗")
            return (f'{_PROV.get(mm.get("provenance"), "")}{mm.get("name")}{glyph}'
                    + ("[入口即真]" if mm.get("true_at_entry") else ""))
        extra["milestone funnel(端态; =合取项镜像 ~派生 *自建仪器)"] = esc(
            " · ".join(_mark(mm) for mm in ver["milestones"]))
    counts = ver.get("milestone_counts")
    if isinstance(counts, dict):
        # Deliberately a partition, never a ratio: the removed `milestones_passed` numerator mixed
        # exact check_success mirrors, coarsened proximities, invented progress rungs and stages
        # that were free at entry, and got quoted as though it graded the attempt.
        labels = [("achieved_after_entry", "agent 达成"),
                  ("true_at_entry_and_at_end", "入口即真(终态要求·非进展证据)"),
                  ("true_at_entry_then_lost", "入口为真但被破坏"),
                  ("ok_true_unbaselined", "终态为真(无入口基线·未验证归因)"),
                  ("not_achieved", "未达成"),
                  ("entry_unknown", "入口读数不可用"),
                  ("not_evaluable", "不可求值"),
                  ("skipped", "actor 缺席跳过")]
        parts = [f"{zh} {counts[k]}" for k, zh in labels if counts.get(k)]
        parts.append(f'共声明 {counts.get("declared", 0)}')
        if counts.get("entry_baseline") != "recorded":
            parts.append("<span class=bad>入口基线缺失</span>")
        extra["milestone 分布(非比值)"] = " · ".join(parts)
    # In-episode latches, reported SEPARATELY from the end-state funnel: a latch says "held at
    # some polled instant", the funnel says "holds at the end". Showing them together is the
    # whole point — a subgoal that latched but failed at the end means reached-then-lost, which
    # the funnel alone cannot express. Marked ⏱ so the two are never read as the same claim.
    latch = ver.get("latch")
    if isinstance(latch, dict) and isinstance(latch.get("events"), dict):
        parts, void = [], []
        for name, st in latch["events"].items():
            if st.get("skipped"):
                parts.append(f'{name}⏱—')      # actor absent this seed (variable-count scene)
            elif st.get("void"):
                # Satisfied by the initial scene, so it measures the layout and not the agent.
                # The monitor never polls it; it produces no signal at all. This is a CARD
                # DEFECT to fix (retune the threshold or drop the event), not a result.
                void.append(name)
                parts.append(f'{name}⏱[void]')
            elif st.get("latched"):
                dwell = st.get("dwell_polls", 1)
                held = f'×{dwell}' if dwell > 1 else ''
                parts.append(f'{name}⏱✓@{st.get("step", "?")}{held}')
            elif st.get("error"):
                parts.append(f'{name}⏱!')      # predicate errored; stayed unlatched
            else:
                parts.append(f'{name}⏱✗')
        # Step indices order events WITHIN this episode only — different agents and seeds reach
        # the same subgoal at wildly different physics-step counts, so they are not comparable
        # across runs. An unfired latch means "not observed at the polling rate", not "did not
        # happen"; only the end-state funnel and the binary verdict are end-of-episode facts.
        extra["latch(期内·曾满足; 步号仅本 episode 内可比)"] = esc(" · ".join(parts))
        if void:
            extra["latch 作废(入口即真 → 卡缺陷)"] = (
                "<span class=bad>" + esc(" · ".join(void)) + "</span>")
        # The discrimination this whole mechanism exists for: latched during the episode but
        # false at the end = the subgoal was reached and then destroyed. Pairing is DECLARED by
        # the card (`mirrors: <milestone name>` on the event), never inferred from name equality
        # — the two namespaces are independent and a silent mismatch would hide exactly the
        # finding this row is for.
        end_ok = {mm.get("name"): mm.get("ok") for mm in (ver.get("milestones") or [])}
        lost, dangling = [], []
        for name, st in latch["events"].items():
            mirrors = (st.get("mirrors")
                       or next((e.get("mirrors") for e in (latch.get("declared_events") or [])
                                if e.get("name") == name), None))
            if not mirrors:
                continue
            if mirrors not in end_ok:
                dangling.append(f"{name}→{mirrors}")
            elif st.get("latched") and end_ok[mirrors] is False and not st.get("void"):
                lost.append(f'{name}(端态 {mirrors}✗)')
        if lost:
            extra["达成后丢失(latch✓但端态✗)"] = esc(" · ".join(lost))
        if dangling:
            extra["latch mirrors 指向不存在的 milestone"] = esc(" · ".join(dangling))
    elif isinstance(latch, dict) and latch.get("missing"):
        # No monitor state. Two very different situations, and calling both a failure was wrong:
        # a card that GATED on an event is unverified (fail closed), while a purely diagnostic
        # card simply ran with diagnostics off — the default, and not a problem at all.
        extra["latch(期内)"] = ("<span class=bad>已声明 required 事件但无 monitor 状态 → "
                                "fail-closed</span>" if latch.get("gated")
                                else "诊断未采集（--in-episode-diagnostics 未开启；不影响判决）")
    # step_observer is a SIBLING of `verifier` in result.json, not part of it. A disabled callback
    # means polling silently stopped, so any latch below it is "unknown", not "never happened".
    so = result.get("step_observer") or {}
    disabled = [k for k, v in (so.get("callbacks") or {}).items() if v != "enabled"]
    if disabled:
        extra["step observer 回调被禁用(latch 不可信)"] = esc(" · ".join(
            f'{k}: {(so.get("errors") or {}).get(k, "?")}' for k in disabled))
    if isinstance(so, dict) and so.get("physical_time_s") is not None:
        actual = so.get("physical_time_s")
        budget = so.get("physical_time_budget_s")
        expert = so.get("expert_sim_duration_s")
        ratio = so.get("physical_time_to_expert_ratio")
        extra["物理执行"] = esc(
            f"{actual:g}s / budget {budget:g}s · {so.get('steps', '?')}/"
            f"{so.get('threshold_physics_steps', so.get('max_physics_steps', '?'))} physics steps"
            if isinstance(actual, (int, float)) and isinstance(budget, (int, float))
            else f"{actual}s · {so.get('steps', '?')} physics steps")
        if isinstance(expert, (int, float)) and isinstance(ratio, (int, float)):
            extra["相对 expert 物理时长"] = esc(f"{expert:g}s · {ratio:.2f}×")
        overshoot_s = so.get("tool_boundary_overshoot_s")
        if isinstance(overshoot_s, (int, float)) and overshoot_s > 0:
            extra["最终原子工具越过 threshold"] = esc(
                f"{overshoot_s:g}s · {so.get('tool_boundary_overshoot_steps', '?')} physics steps")
    budget_used = resources["tool_calls_used"]
    budget_limit = resources["tool_call_budget"]
    budget_ratio = resources["tool_call_budget_utilization"]
    budget_text = (
        f"{budget_used} / {budget_limit} ({budget_ratio * 100:.1f}%)"
        if budget_used is not None and budget_limit is not None and budget_ratio is not None
        else str(budget_used if budget_used is not None else "?")
    )
    return {
        **extra,
        "episode 状态": end.get("status", "?"),
        "工具调用预算(已用/上限)": budget_text,
        "工具 dispatch 总数(含免费/超限)": (
            resources["total_tool_dispatches"]
            if resources["total_tool_dispatches"] is not None else "?"),
        "模型轮数(无预算分母)": (
            resources["model_turns"] if resources["model_turns"] is not None else "?"),
        "墙钟(s)": resource_stats.get("wall_s", end.get("wall_s", "?")),
        "tokens (prompt/completion)":
            f'{resource_stats.get("usage", {}).get("prompt_tokens", "?")} / '
            f'{resource_stats.get("usage", {}).get("completion_tokens", "?")}',
        "端点 infra 重试":
            resource_stats.get("infra_retries", end.get("infra_retries", 0)),
        "guard 中止(按原因)": esc(json.dumps(ab, ensure_ascii=False)) if ab else "0",
        "FAILED 动作数": fails, "工具错误(参数/未知)": errs,
        "reach_tcp 失败分类": esc(json.dumps(reach_fail, ensure_ascii=False)) if reach_fail else "0",
        "最长连续相同调用(冗余 round)": mx,
        "run_code 采用次数": len(rc),
        "声称 vs verifier": claim_row,
        "verifier kind": esc(ver.get("kind", "?")),
        "verifier": f'{"<span class=ok>SUCCESS</span>" if truth else "<span class=bad>FAILURE</span>"}'
                    f' · rise={ver.get("detail", {}).get("rise_m", "?")}m',
    }


def run_code_internal_charged(lines):
    """Return the recorded budget contract; legacy transcripts used parity charging."""
    meta = next((line for line in lines if line.get("event") == "meta"), {})
    return meta.get("run_code_internal_charged", True) is not False


def render_run(run):
    d, lines = load(run["sub"])
    ver = verifier_of(run)
    result = result_of(run)
    motion_records = []
    motion_trace = d / "tools" / "motion_trace.jsonl"
    if motion_trace.is_file():
        try:
            motion_records = [json.loads(line) for line in
                              motion_trace.read_text(encoding="utf-8").splitlines()
                              if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError):
            motion_records = []
    agent_exit = run.get("agent_exit") or {}
    tools = executed_tool_records(lines)
    cancelled_after = cancelled_calls_by_preceding_record(lines)
    done = next((l for l in lines if l["event"] == "done"), None)
    video_meta = {}
    try:
        video_meta = json.loads((d / "video_meta.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    h = [f"<html><head><meta charset='utf-8'><title>{esc(run['name'])}</title>",
         f"<style>{CSS}</style></head><body><div class='wrap'>",
         f"<p><a href='index.html'>← index</a></p><h1>{esc(run['name'])}</h1>",
         f"<p class='note'>{esc(run['tag'])} · 目录 <code>{esc(run['sub'])}</code>"
         f" · 运行版本 {_version_html(run)}</p>"]
    task = next((l.get("task") for l in lines if l["event"] == "meta"), "?")
    h.append(f"<p><b>任务:</b>{esc(task)}</p>")
    review_path = d / "review.mp4"
    full_path = d / "full.mp4"
    if review_path.exists():
        review = video_meta.get("review") or {}
        speedup = float(review.get("speedup") or 1.0)
        h.append("<h2>连续 episode 录像</h2>")
        h.append(f"<video class='episode-video' controls preload='metadata' "
                 f"src='../{esc(run['sub'])}/review.mp4'></video>")
        h.append("<p class='note'>head camera · sim-time 连续录像"
                 f" · {esc(video_meta.get('frames', '?'))} frames"
                 f" · review {esc(review.get('duration_s', '?'))} s"
                 f" · speedup {esc(speedup)}×</p>")
        links = [f"<a href='../{esc(run['sub'])}/review.mp4'>review.mp4</a>"]
        if full_path.exists():
            links.append(f"<a href='../{esc(run['sub'])}/full.mp4'>full.mp4</a>")
        h.append(f"<p class='video-links'>{' · '.join(links)}</p>")
    h.append("<h2>指标卡(§12 v5 可计算子集)</h2><table>")
    for k, v in metrics_of(lines, ver, result, agent_exit, motion_records).items():
        h.append(f"<tr><th style='width:270px'>{esc(k)}</th><td>{v}</td></tr>")
    h.append("</table>")

    entry_no = {id(t): i for i, t in enumerate(tools, 1)}   # contiguous #N per transcript entry
    charge_run_code_internals = run_code_internal_charged(lines)

    sc = [t for t in tools if t["tool"] in SCALE_TOOLS]
    h.append("<h2>尺度/几何推导(模型的度量链)</h2>")
    if sc:
        h.append("<table><tr><th>#</th><th>工具</th><th>输入</th><th>输出(value±σ)</th></tr>")
        for t in sc:
            r = t.get("result") or {}
            val = r.get("value")
            unc = r.get("uncertainty")
            h.append(f"<tr><td>#{entry_no[id(t)]}</td><td>{esc(t['tool'])}</td>"
                     f"<td class='args'>{esc(json.dumps(t.get('args'), ensure_ascii=False))}</td>"
                     f"<td class='args'>{esc(json.dumps(val, ensure_ascii=False))}"
                     f"{' ± ' + esc(round(unc, 4)) if isinstance(unc, (int, float)) else ''}</td></tr>")
        h.append("</table>")
    else:
        h.append("<p class='note'>本 run 未调用任何尺度/几何工具。</p>")

    h.append("<h2>逐步时间线(全部中间过程)</h2>")
    if charge_run_code_internals:
        h.append("<p class='note'>#N = 连续条目序号(第几次工具调用);<b>预算序号</b> = 累计消耗的"
                 "调用预算。该历史 run 使用 v0.3 预算语义:<code>run_code</code> 会把内部代理调用"
                 "一并计入,所以预算序号在 run_code 处按 (1+M) 跳增。</p>")
    else:
        h.append("<p class='note'>#N = 连续条目序号(第几次工具调用);<b>预算序号</b> = 累计消耗的"
                 "调用预算。该 run 使用 v0.5 预算语义:<code>run_code</code> 整块只计 1 次调用;"
                 "内部代理调用仅作遥测,不消耗 episode 预算。</p>")
    if tools and "model_text" not in tools[0]:
        h.append("<p class='note'>⚠ 该批 run 的 runner 尚未记录模型每轮伴随思考文本"
                 "(已修复,后续 run 将包含);以下为完整的工具调用/结果/图像流。</p>")
    elif tools and not any(t.get("reasoning_content") for t in tools):
        # The claude-opus-5 symptom: turns are recorded, thinking tokens are billed, and the
        # reasoning column is empty on every one of them. Say so on the page instead of rendering
        # a report that merely looks thinner than the Qwen one.
        h.append("<p class='note'>⚠ 该 run 没有任何一轮记录到 provider 的推理内容"
                 "(reasoning_content 全为空)。若该模型的 reasoning profile 本应开启思考,"
                 "说明请求没有要求供应商返回推理文本(例如 Anthropic 的 "
                 "<code>thinking.display</code> 默认为 <code>omitted</code>),"
                 "而不是模型没有思考。</p>")
    for i, t in enumerate(tools, 1):
        r = t.get("result") or {}
        st = result_status(t["tool"], r)
        head = (f"<div class='step'><b class='entry'>#{i}</b> · <code>{esc(t['tool'])}</code> "
                f"{badge(st)} <span class='note'>预算序号 {t['step']}</span>")
        sim_start = t.get("sim_step_start")
        sim_end = t.get("sim_step_end")
        if isinstance(sim_start, int) and isinstance(sim_end, int):
            sim_dt = float(video_meta.get("sim_dt") or 0.0)
            speedup = float((video_meta.get("review") or {}).get("speedup") or 1.0)
            sim_label = f"physics {sim_start}→{sim_end}"
            if sim_dt > 0:
                sim_label += f" · review {sim_start * sim_dt / speedup:.2f}→" \
                             f"{sim_end * sim_dt / speedup:.2f}s"
            head += f" <span class='sim-range'>{esc(sim_label)}</span>"
        if t["tool"] == "run_code":
            inner = int((r.get("tool_calls") or 0))
            if charge_run_code_internals:
                head += (f"<span class='budget'> · 消耗预算 {1 + inner} = 自身 1 + 内部 "
                         f"{inner} 次工具调用</span>")
            else:
                head += (f"<span class='budget'> · 消耗预算 1;内部 {inner} 次工具调用不计费</span>")
        h.append(head)
        # Two different things, previously conflated under one 模型思考 label: `reasoning_content`
        # is the provider's own reasoning channel (Qwen's reasoning_content, Anthropic's thinking
        # blocks), `model_text` is the assistant text that ships alongside the tool call.
        if t.get("reasoning_content"):
            h.append(f"<div class='think'><b>模型思考(推理链):</b>"
                     f"{esc(t['reasoning_content'])}</div>")
        if t.get("model_text"):
            h.append(f"<div class='think'><b>模型发言(伴随文本):</b>{esc(t['model_text'])}</div>")
        h.append(f"<div class='args'>args: {esc(json.dumps(t.get('args'), ensure_ascii=False))}</div>")
        why = reason_of(t["tool"], r)
        if why:
            h.append(f"<div style='color:#b03030;font-size:13px;margin:5px 0'>"
                     f"<b>状态解释:</b>{esc(why)}</div>")
        trace = internal_trace_rows(r) if t["tool"] == "run_code" else []
        if t["tool"] == "run_code":
            assigned = "是" if r.get("result_assigned") is True else "否"
            h.append(f"<div class='args'><b>result 已赋值:</b> {assigned}</div>")
        if trace:
            h.append(f"<details open><summary>内部工具序列（{len(trace)}）</summary>"
                     "<table><tr><th>#</th><th>工具</th><th>RPC</th><th>动作状态</th>"
                     "<th>执行证据</th></tr>")
            for row in trace:
                evidence = []
                for key in ("abort_reason", "failure_category", "stop_reason", "transition",
                            "action_id", "tick", "obs_ids", "error"):
                    if row.get(key) is not None:
                        evidence.append(f"{key}={row[key]}")
                h.append(f"<tr><td>{esc(row.get('index', '—'))}</td>"
                         f"<td><code>{esc(row.get('tool', '—'))}</code></td>"
                         f"<td>{'OK' if row.get('ok') else 'ERR'}</td>"
                         f"<td>{esc(row.get('status', '—'))}</td>"
                         f"<td class='args'>{esc('; '.join(evidence) or '—')}</td></tr>")
            h.append("</table></details>")
        rs = json.dumps({k: v for k, v in r.items() if k != 'cam_pose_snapshot'},
                        ensure_ascii=False)
        h.append(f"<div class='res'>{esc(rs[:900])}{'…' if len(rs) > 900 else ''}</div>")
        for src, lab in find_images(d, r):
            h.append(f"<div class='imgbox'><img src='{esc(src)}'>"
                     f"<div class='imglab'>{esc(lab)}</div></div>")
        # eval-side god-view after motion/gripper actions (the model did NOT see these)
        action = motion_result(t["tool"], r)
        if t["tool"] in MOTION_TOOLS and isinstance(action.get("tick"), int):
            tick = action["tick"]
            rel = f"../{run['sub']}/tools/observer"
            for cam in ("head_camera", "left_camera", "right_camera"):
                for p in observer_images(d, tick, cam):
                    h.append(f"<div class='imgbox'><img src='{rel}/{esc(p.name)}'>"
                             f"<div class='imglab'>动作后 · {CAM.get(cam, cam)} · "
                             f"上帝视角(模型未见)</div></div>")
        if t["tool"] == "run_code" and trace:
            ticks = list(dict.fromkeys(row.get("tick") for row in trace
                                       if isinstance(row.get("tick"), int)))
            if ticks:
                h.append(f"<details><summary>内部动作后上帝视角（{len(ticks)} ticks，模型未见）"
                         "</summary>")
                rel = f"../{run['sub']}/tools/observer"
                for tick in ticks:
                    for cam in ("head_camera", "left_camera", "right_camera"):
                        for p in observer_images(d, tick, cam):
                            h.append(f"<div class='imgbox'><img src='{rel}/{esc(p.name)}'>"
                                     f"<div class='imglab'>run_code 内部 tick {tick} · "
                                     f"{CAM.get(cam, cam)} · 上帝视角(模型未见)</div></div>")
                h.append("</details>")
        for cancelled in cancelled_after.get(id(t), []):
            h.append(
                "<div class='args' style='color:#8a6d00'>↳ 同轮安全屏障取消:"
                f"<code>{esc(str(cancelled.get('tool') or '?'))}</code> · 预算序号 "
                f"{esc(str(cancelled.get('step')))} · "
                f"{'已计费' if cancelled.get('charged') else '未计费'},未执行,"
                "未进入仿真</div>")
        h.append("</div>")
    if done:
        rep = done.get("report", {})
        h.append(f"<div class='step' style='border-color:#2e9e5b'><b>done · 模型报告</b> "
                 f"(success_claim={rep.get('success_claim')})<div class='think'>"
                 f"{esc(rep.get('report', ''))}</div></div>")

    h.append("<h2>失败后的反思与纠错</h2>")
    rows = []
    for i, t in enumerate(tools[:-1]):
        r = t.get("result") or {}
        if r.get("status") in ("FAILED", "ABORTED") or result_is_error(t["tool"], r):
            nxt = tools[i + 1]
            changed = (nxt["tool"] != t["tool"]) or (nxt.get("args") != t.get("args"))
            later_ok = any(result_is_success(x["tool"], x.get("result") or {})
                           and x["tool"] == t["tool"] for x in tools[i + 1:])
            rows.append((t, nxt, changed, later_ok))
    if rows:
        h.append("<table><tr><th>失败点</th><th>失败内容</th><th>下一步</th>"
                 "<th>调整?</th><th>同类动作后续成功?</th></tr>")
        for t, nxt, ch, ok2 in rows:
            r = t.get("result") or {}
            why = r.get("abort_reason") or r.get("error") or \
                (r.get("achieved") or {}).get("reason") or r.get("status")
            h.append(f"<tr><td>#{entry_no[id(t)]} <code>{esc(t['tool'])}</code></td>"
                     f"<td class='args'>{esc(str(why)[:120])}</td>"
                     f"<td>#{entry_no[id(nxt)]} <code>{esc(nxt['tool'])}</code></td>"
                     f"<td>{'<span class=ok>换目标/换工具</span>' if ch else '<span class=bad>原样重试</span>'}</td>"
                     f"<td>{'<span class=ok>是</span>' if ok2 else '<span class=bad>否</span>'}</td></tr>")
        h.append("</table>")
    else:
        h.append("<p class='note'>无失败动作。</p>")
    h.append("</div></body></html>")
    return "\n".join(h), ver, task


def _task_of(run):
    """Task label for grouping: first '·' segment of the human name; but if that is an attempt
    token (a1/s0…) or empty (meta-less orphan run), derive the base task from the batch dir name
    by stripping the track/version/model decorations."""
    import re as _re
    nm = (run.get("name") or "").split("·")[0].strip()
    if nm and not _re.match(r"^[as]\d+$", nm):
        return nm
    base = run["sub"].split("/")[0]
    base = _re.sub(r"^vendor_(opus\d*_|fable\d*_|sonnet\d*_|qwen\S*?_)?", "", base)   # unify vendor-agent under the real task
    base = _re.split(r"_track[AB]|_hybrid|_qwen|_v0\d|_stage|_\d{8}|_cuda", base)[0]
    return base or run["sub"].split("/")[0]


def _version_label(run):
    commit = run.get("git_commit")
    if not commit:
        return ""
    return str(commit) + ("+dirty" if run.get("git_dirty") else "")


def _version_html(run):
    label = _version_label(run)
    return f"<code>{esc(label)}</code>" if label else "<span class='note'>未记录</span>"


def _archive_reason(run, end):
    """Return an exclusion reason, or None for a scoreable tested-unit episode.

    Schema 1.2 uses only ``failure.scoreable``. The remaining archive/status cases are an explicit
    compatibility path for legacy 1.1 runs that cannot carry the new field.
    """
    failure = failure_of(run.get("result"), run.get("agent_exit"))
    if failure is not None:
        return None if failure.scoreable else failure.code.value
    explicit = run.get("archive_reason")
    if explicit:
        return str(explicit)
    status = str(end.get("status") or "")
    sub = str(run.get("sub") or "")
    if "quota_interrupted" in sub:
        return "subscription_quota_interrupted"
    if status in {"endpoint_failure", "tool_wedge", "episode_fatal"}:
        return status
    return None


def _aggregate_cells(meta_rows):
    """Group every VALID (non-archived) attempt by (task, model, interface, taskset version) and
    aggregate it — the report's headline numbers, produced on every rebuild instead of by hand.

    An attempt directory IS one attempt, so this regroups across batch directories: cells that
    merge several batches list them, because merged batches may differ in budget or harness commit
    and a reader must be able to see that. Archived (infrastructure-interrupted) attempts never
    enter the denominator, exactly as in the per-day tree below."""
    try:
        from codeaction.verification.summarize import attempts_declared_for, identity_of, summarize_records
    except Exception:
        return []
    cells = {}
    for r in meta_rows:
        if not r.get("result"):
            continue
        ident = identity_of(r["result"], r.get("run_meta"))
        task = ident["task_name"] or r["task"]
        key = (task, ident["model"] or "?", ident["interface"] or "?",
               ident["taskset_version"] or "?")
        cell = cells.setdefault(key, {"entries": [], "archived": [], "batches": set()})
        # The transcript is re-read here (rather than carried on every meta row) so the safety
        # axis has real input: `summarize_records` counts contact aborts from `tool_records`, and
        # an entry without them is coverage 0 — reported as unknown, never as a clean zero.
        entry = {"result": r["result"], "run_meta": r.get("run_meta") or {},
                 "agent_exit": r.get("agent_exit") or {},
                 "tool_records": _tool_records_of(r.get("sub"))}
        if r["archive_reason"]:
            entry["archive_reason"] = r["archive_reason"]
            failure = failure_of(entry["result"], entry["agent_exit"])
            if failure is not None and not failure.scoreable:
                entry["failure"] = failure.to_dict()
                entry["exclusion_kind"] = "classified_failure"
            else:
                entry["exclusion_kind"] = "legacy_archive"
            cell["archived"].append(entry)
        else:
            cell["entries"].append(entry)
            cell["batches"].add(r.get("managed_batch") or _batch_of_sub(r["sub"]))
    out = []
    for (task, model, interface, version), cell in cells.items():
        if not cell["entries"]:                 # an all-archived cell has no number to report
            continue
        try:
            summary = summarize_records(cell["entries"], cell["archived"],
                                        attempts_declared=attempts_declared_for(task))
        except Exception:
            continue
        batches = sorted(cell["batches"])
        summary.update({"task_name": task, "model": model, "interface": interface,
                        "taskset_version": version, "batches": batches})
        if len(batches) > 1:
            # Same identity, different batch directories: budgets, prompts or harness commit may
            # differ, so the pooled n is an analysis view — never a release number.
            summary["protocol_valid"] = False
            summary["validity_errors"] = summary["validity_errors"] + [
                f"pooled across {len(batches)} batch directories"]
        out.append(summary)
    out.sort(key=lambda s: (s["task_name"], s["model"], s["interface"]))
    return out


def _tool_records_of(sub):
    """One attempt's transcript for the aggregate, or [] when it cannot be read.

    Tolerant on purpose: an unreadable transcript must degrade to coverage 0 — reported as unknown
    — and never take the whole aggregate table down with it."""
    if not sub:
        return []
    try:
        return load(sub)[1]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []


def _contact_safety_cell(cell):
    """`总次数 · 出现该情况的 attempt 数/总 attempt 数`, or an explicit unknown.

    Never renders 0 for a cell whose transcripts could not be read: "we did not look" and "nothing
    happened" are different facts, and only one of them is good news."""
    safety = cell.get("contact_safety") or {}
    total = safety.get("unexpected_contact_aborts_total")
    legacy = safety.get("legacy_collision_terminals_total") or 0
    if total is None and not legacy:
        return "<span class='note'>无 transcript · 未知</span>"
    parts = []
    if total is not None:
        parts.append(
            f"{total} · {safety.get('attempts_with_unexpected_contact')}/{cell['n_attempts']}")
    if legacy:
        # Tool surface 6.x-8.x ended the attempt on the first contact, so those cells carry terminal records
        # and no recoverable aborts. Omitting them would render such a batch as a clean zero.
        parts.append(f"历史终局 {legacy}/{cell['n_attempts']}")
    body = " · ".join(parts)
    coverage = safety.get("transcript_coverage")
    if isinstance(coverage, (int, float)) and coverage < 1.0:
        body += f" <span class='note'>(仅覆盖 {coverage})</span>"
    return body


def _funnel_cell(cell):
    """One entry PER MILESTONE: `agent 达成 / 可归因 attempt 数`.

    Never a cross-milestone fraction. A card's milestones are heterogeneous claims (an exact
    `check_success` mirror, a coarsened proximity, our own progress rung), so "6/9" across them
    has no referent — that numerator was removed on 2026-08-09 and must not return through the
    aggregate. Across ATTEMPTS one milestone is the same predicate repeated, which is what makes
    each row's fraction readable. The denominator is the attempts whose entry baseline makes
    "the agent produced this" answerable at all, so it can be smaller than the cell's n."""
    funnel = cell.get("milestone_funnel") or {}
    authoritative = funnel.get("authoritative_milestones") or []
    instruments = funnel.get("development_instruments") or []
    unclassified = funnel.get("unclassified_milestones") or []
    if not (authoritative or instruments or unclassified):
        return ""

    def _rows(rows):
        parts = []
        for row in rows:
            attributable = row.get("attributable_attempts") or 0
            parts.append(f"{esc(str(row.get('name')))} "
                         + (f"{row.get('achieved_after_entry', 0)}/{attributable}"
                            if attributable else "—"))
        return " · ".join(parts)

    warn = (" <span class=bad>各 attempt 声明的 milestone 集合不一致</span>"
            if funnel.get("mixed_milestone_sets") else "")
    coverage = funnel.get("entry_baseline_coverage")
    note = ("" if coverage == 1.0 else
            f" <span class='note'>(入口基线覆盖 {coverage})</span>")

    # Two separate lines, never one. The first restates the authoritative predicate; the second is
    # our own instrumentation, several rows of which are explicitly not monotone toward the goal.
    # Rendering them apart is what stops a reader averaging one kind into the other.
    blocks = []
    if authoritative:
        blocks.append("进度漏斗（合取项镜像·agent 达成/可归因）：" + _rows(authoritative)
                      + note + warn)
    if instruments:
        blocks.append("<span class='note'>开发期仪器（非权威，不作对外数字）："
                      + _rows(instruments) + "</span>")
    if unclassified:
        blocks.append("<span class=bad>milestone 未声明 provenance："
                      + _rows(unclassified) + "</span>")
    return "<br>".join(blocks)


def _tokens_cell(cell):
    """Uncached / cached input, and output. Tokens, not currency: prices are dated external facts
    that belong to whoever publishes the table. Cached input bills at a fraction of uncached, so
    a prompt total alone cannot be priced — and the two move in opposite directions when frame
    retention changes, which is exactly the comparison this column exists to make visible."""
    tokens = cell.get("tokens") or {}
    if not tokens.get("prompt_total"):
        return "—"
    uncached, cached = tokens.get("uncached_prompt_total"), tokens.get("cached_prompt_total")
    if uncached is None:
        # Partial coverage: report what was measured and say the rest is unknown. A zero here
        # would read as "nothing was cached", which no provider actually told us.
        return (f"{tokens['prompt_total']:,} <span class='note'>(缓存拆分未知, 覆盖 "
                f"{tokens.get('cached_coverage')})</span> · 出 {tokens['completion_total']:,}")
    return (f"{uncached:,} / {cached:,} · 出 {tokens['completion_total']:,}")


def _aggregate_html(cells):
    if not cells:
        return ""
    rows = []
    for c in cells:
        lo, hi = c["ci95"]
        k = c["attempts_declared"]
        protocol = (f"<span class='pill ok'>完整 k={k}</span>" if c["protocol_complete"]
                    else f"<span class='pill'>dev {c['n_attempts']}/{k or '?'}</span>")
        blockers = "" if c["protocol_valid"] else (
            f"<div class='note'>协议不完整，不可提交：{esc('；'.join(c['validity_errors']))}</div>")
        excluded = c.get("attempts_excluded", 0)
        arch = f" <span class='note'>(+{excluded} 排除)</span>" if excluded else ""
        # pass^k at the largest k the cell actually ran: "did it work EVERY time", the number a
        # manipulation result is judged on, beside pass@1's "did it ever work".
        hat = c.get("pass_hat_k") or {}
        hat_k = max((int(key) for key in hat), default=None)
        hat_cell = f"{hat[str(hat_k)]} <span class='note'>(k={hat_k})</span>" if hat_k else "—"
        rows.append(
            f"<tr><td>{esc(c['task_name'])}{blockers}</td><td>{esc(c['model'])}</td>"
            f"<td>{esc(c['interface'])}</td><td><code>{esc(c['taskset_version'])}</code></td>"
            f"<td>{c['n_success']}/{c['n_attempts']}{arch}</td>"
            f"<td><b>{c['success_rate']}</b></td><td>[{lo}, {hi}]</td>"
            f"<td>{c['pass_at_k'].get('1', '—')}</td><td>{hat_cell}</td>"
            f"<td>{c['destructive_rate']}</td>"
            f"<td>{_contact_safety_cell(c)}</td>"
            f"<td class='note'>{_tokens_cell(c)}</td>"
            f"<td>{protocol}</td>"
            f"<td class='note'>{esc(', '.join(c['batches']))}</td></tr>")
        funnel = _funnel_cell(c)
        if funnel:
            rows.append(f"<tr><td colspan=14 class='note'>{funnel}</td></tr>")
    return ("<details open><summary style='background:#eef1f5;font-size:14px'>"
            "聚合成绩（任务 × 模型 × 接口 × 任务集版本）</summary>"
            "<p class='note'>成功率 = 带外 verifier 的二元判定在有效 attempt 上的比例；区间为 "
            "Wilson 95% CI（小 n 稳健）。failure.scoreable=false 的 attempt 不进分母；"
            "旧 1.1 run 才回退读取 archive_meta。一个 cell 若跨多个批次目录，"
            "批次列会全部列出——不同批次可能预算或 harness commit 不同，读者必须看得见。"
            "<b>协议</b>列标出该 cell 是否跑满任务卡声明的 <code>protocol.attempts_k</code>；"
            "未跑满的行是 dev 数字，不得作为发布成绩引用。"
            "<b>接触中止</b>是独立安全轴，<b>不参与成功率</b>：格式为"
            "「监控中止总次数 · 出现过的 attempt 数/总 attempt 数」，"
            "计自 transcript 中 <code>ABORTED/unexpected_contact</code>（含 run_code 内被中断的块）。"
            "一次中止只证明求解器在某个 physics step 产生了接触响应，不证明严重程度、持续性或任务失败。"
            "<b>pass^k</b> = 随机抽 k 次全部成功的概率 C(c,k)/C(n,k)，与 pass@1 的「至少一次」相对；"
            "固定场景重复协议下它才是可靠性数字。<b>进度漏斗</b>行按 milestone 逐条给出"
            "「agent 达成 / 可归因 attempt 数」，跨 milestone 求和无意义、故不提供；"
            "字形 = 合取项镜像 · ~ 派生 · * 自建仪器。<b>输入(未缓存/缓存)</b>是计价所需的拆分，"
            "本表只给 token，不给金额（价格是有时效的外部事实）。</p>"
            "<table><tr><th>任务</th><th>模型</th><th>接口</th><th>任务集</th><th>成功</th>"
            "<th>成功率</th><th>95% CI</th><th>pass@1</th><th>pass^k</th><th>破坏率</th>"
            "<th>接触中止</th><th>输入(未缓存/缓存)</th>"
            "<th>协议</th><th>批次</th></tr>" + "".join(rows) + "</table></details>")


_MANAGED_CELL_STATUSES = ("accepted", "needs_attention", "queued", "retry_wait", "running")


def _read_managed_state(batch_dir):
    state_path = Path(batch_dir) / "batch_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchResultsError(f"cannot read managed batch state: {exc}") from exc
    if not isinstance(state, dict):
        raise BatchResultsError("managed batch state must be an object")
    return state


def _managed_batch_snapshot(state, accepted_document, accepted_records):
    """Return report-safe state only after the accepted selection has been revalidated."""
    if state.get("revision") != accepted_document.get("state_revision") \
            or state.get("batch_id") != accepted_document.get("batch_id") \
            or state.get("active_stage") != accepted_document.get("active_stage"):
        raise BatchResultsError("managed batch state changed after accepted selection validation")

    requested = state.get("requested_cells")
    cells = state.get("cells")
    executions = state.get("executions")
    attentions = state.get("attentions")
    if not isinstance(requested, list) or len(requested) != len(set(requested)) \
            or not isinstance(cells, dict) or set(cells) != set(requested) \
            or not isinstance(executions, dict) or not isinstance(attentions, dict):
        raise BatchResultsError("managed batch state collections are invalid")

    counts = {status: 0 for status in _MANAGED_CELL_STATUSES}
    for identifier in requested:
        cell = cells.get(identifier)
        if not isinstance(cell, dict) or cell.get("status") not in counts:
            raise BatchResultsError(f"managed batch cell status is invalid: {identifier}")
        counts[cell["status"]] += 1
    if counts["accepted"] != accepted_document.get("attempt_count") \
            or len(accepted_records) != accepted_document.get("attempt_count"):
        raise BatchResultsError("managed batch accepted count differs from validated selection")

    open_attention = []
    for attention_id, record in attentions.items():
        if not isinstance(record, dict) or record.get("status") != "open":
            continue
        identifier = record.get("cell_id")
        cell = cells.get(identifier)
        execution_id = record.get("execution_id")
        execution = executions.get(f"{identifier}/{execution_id}")
        if record.get("attention_id") != attention_id or not isinstance(cell, dict) \
                or cell.get("status") != "needs_attention" \
                or cell.get("attention_id") != attention_id \
                or not isinstance(execution, dict) \
                or execution.get("status") != "needs_attention" \
                or record.get("task") != cell.get("task") \
                or record.get("model") != cell.get("model") \
                or record.get("attempt_index") != cell.get("attempt_index"):
            raise BatchResultsError(f"managed batch attention linkage is invalid: {attention_id}")
        for field in ("task", "model", "execution_id", "location"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise BatchResultsError(
                    f"managed batch attention lacks exact {field}: {attention_id}")
        attempt_index = record.get("attempt_index")
        if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) \
                or attempt_index < 0:
            raise BatchResultsError(
                f"managed batch attention lacks exact attempt: {attention_id}")
        open_attention.append(record)
    if len(open_attention) != counts["needs_attention"]:
        raise BatchResultsError("managed batch attention count differs from cell state")

    retry_executions = 0
    for execution_key, execution in executions.items():
        if not isinstance(execution, dict):
            raise BatchResultsError(f"managed batch execution is invalid: {execution_key}")
        if execution.get("status") == "retry_wait":
            retry_executions += 1
    return state, counts, open_attention, retry_executions


def _managed_batches():
    """Discover managed reference-scaffold batches and fail closed when their selection is stale or invalid."""
    batches = []
    base = Path(BASE).resolve()
    out = Path(OUT).resolve()
    for specification_path in sorted(base.rglob("batch_spec.json")):
        batch_dir = specification_path.parent.resolve()
        if batch_dir == out or out in batch_dir.parents:
            continue
        relative = batch_dir.relative_to(base).as_posix()
        view = {
            "root": batch_dir,
            "sub": relative,
            "valid": False,
            "accepted_attempt_dirs": frozenset(),
            "error": None,
        }
        try:
            manifest = batch_dir / SUBMISSION_MANIFEST_NAME
            state_before = _read_managed_state(batch_dir)
            accepted_document, accepted_records = load_submission_manifest(manifest)
            state = _read_managed_state(batch_dir)
            if state != state_before:
                raise BatchResultsError(
                    "managed batch state changed during accepted selection validation")
            state, counts, attention, retry_executions = _managed_batch_snapshot(
                state, accepted_document, accepted_records)
            accepted_dirs = frozenset(record.attempt_dir.resolve() for record in accepted_records)
            if any(batch_dir not in path.parents for path in accepted_dirs):
                raise BatchResultsError("validated accepted attempt escapes managed batch")
            view.update({
                "valid": True,
                "batch_id": state["batch_id"],
                "stage": state["active_stage"],
                "status": state.get("status"),
                "revision": state["revision"],
                "counts": counts,
                "total": len(state["requested_cells"]),
                "attempt_count": accepted_document["attempt_count"],
                "accepted_successes": sum(
                    1 for record in accepted_records
                    if (record.result.get("verifier") or {}).get("success") is True),
                "accepted_attempt_dirs": accepted_dirs,
                "attention": attention,
                "retry_executions": retry_executions,
            })
        except (BatchResultsError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            view["error"] = str(exc)
        batches.append(view)
    return batches


def _managed_run_role(run_dir, batches):
    """Return the closest containing managed batch and accepted/diagnostic disposition."""
    path = Path(run_dir).resolve()
    containing = [
        batch for batch in batches
        if path == batch["root"] or batch["root"] in path.parents
    ]
    if not containing:
        return None
    batch = max(containing, key=lambda item: len(item["root"].parts))
    role = (
        "accepted"
        if batch["valid"] and path in batch["accepted_attempt_dirs"]
        else "diagnostic"
    )
    return batch, role


def _managed_attention_html(record):
    return (
        f"<div><b>{esc(record['attention_id'])}</b> · "
        f"task=<code>{esc(record['task'])}</code> · "
        f"model=<code>{esc(record['model'])}</code> · "
        f"attempt=<code>attempt-{record['attempt_index']:03d}</code> · "
        f"execution=<code>{esc(record['execution_id'])}</code><br>"
        f"location=<code>{esc(record['location'])}</code><br>"
        f"reason={esc(record.get('reason', ''))} · "
        f"next={esc(record.get('suggested_action', ''))}</div>"
    )


def _managed_batches_html(batches):
    if not batches:
        return ""
    rows = []
    for batch in batches:
        if not batch["valid"]:
            rows.append(
                f"<tr><td><code>{esc(batch['sub'])}</code></td>"
                "<td colspan=4><span class='bad'>UNVALIDATED</span> · "
                f"{esc(batch['error'])}</td></tr>")
            continue
        counts = batch["counts"]
        attention = "".join(_managed_attention_html(record) for record in batch["attention"])
        rows.append(
            f"<tr><td><code>{esc(batch['sub'])}</code><br>"
            f"<span class='note'>{esc(batch['batch_id'])}</span></td>"
            f"<td>{esc(batch['stage'])}<br><span class='note'>"
            f"{esc(batch['status'])} · rev {batch['revision']}</span></td>"
            f"<td><b>{batch['attempt_count']}/{batch['total']} accepted</b><br>"
            f"{batch['accepted_successes']}✓ / "
            f"{batch['attempt_count'] - batch['accepted_successes']}✗</td>"
            f"<td>running={counts['running']} · queued={counts['queued']} · "
            f"retry_wait={counts['retry_wait']} · attention={counts['needs_attention']}<br>"
            f"<span class='note'>retry executions (diagnostic only): "
            f"{batch['retry_executions']}</span></td>"
            f"<td>{attention or '<span class=note>none</span>'}</td></tr>")
    return (
        "<details open><summary style='background:#eef1f5;font-size:14px'>"
        "Managed reference-scaffold batches</summary>"
        "<p class='note'>accepted headline 与分母只来自同时通过 durable state linkage 和 "
        "sealed artifact 重验的 <code>submission_manifest.v1.json</code>。重试及未选中的 execution "
        "仅保留诊断可见性，不进入成功率或聚合。</p>"
        "<table><tr><th>batch</th><th>stage</th><th>headline</th><th>state counts</th>"
        "<th>open attention</th></tr>" + "".join(rows) + "</table></details>"
    )


def rebuild():
    """Regenerate every per-run page + the index from auto-discovery. Safe to call any time.
    The index is a 3-tier collapsible tree: date → task → runs (newest first at every tier,
    newest date open)."""
    import datetime
    OUT.mkdir(parents=True, exist_ok=True)
    managed_batches = _managed_batches()
    meta = []                                   # per-run index metadata (discover order = newest first)
    for run in discover_runs():
        try:
            page, ver, task = render_run(run)
        except Exception as e:
            print(f"skip {run['sub']}: {e}")
            continue
        fn = run["sub"].replace("/", "__") + ".html"
        (OUT / fn).write_text(page, encoding="utf-8")
        _d, lines = load(run["sub"])
        end = next((l for l in lines if l["event"] == "end"), {})
        result = result_of(run)
        managed_role = _managed_run_role(BASE / run["sub"], managed_batches)
        managed_batch = managed_role[0]["sub"] if managed_role is not None else None
        managed_diagnostic = managed_role is not None and managed_role[1] == "diagnostic"
        managed_diagnostic_reason = None
        if managed_diagnostic:
            managed_diagnostic_reason = (
                "accepted manifest/state validation failed"
                if not managed_role[0]["valid"]
                else "execution is not in validated accepted selection"
            )
        archive_reason = (
            None if managed_role is not None
            else _archive_reason({**run, "result": result}, end)
        )
        failure = failure_of(result, run.get("agent_exit"))
        run_meta_path = BASE / run["sub"] / "run_meta.json"
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) \
                if run_meta_path.exists() else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            run_meta = {}
        meta.append({"fn": fn, "name": run["name"], "tag": run["tag"], "sub": run["sub"],
                     "task": _task_of(run), "mtime": run.get("mtime", 0),
                     "git_commit": run.get("git_commit"),
                     "git_dirty": run.get("git_dirty", False),
                     "result": result, "run_meta": run_meta,
                     "agent_exit": run.get("agent_exit") or {},
                     "failure": failure.to_dict() if failure is not None else None,
                     "ok": bool(ver.get("success")), "steps": end.get("steps", "?"),
                     "wall": end.get("wall_s", "?"), "status": end.get("status", "?"),
                     "archive_reason": archive_reason,
                     "managed_batch": managed_batch,
                     "managed_diagnostic": managed_diagnostic,
                     "managed_diagnostic_reason": managed_diagnostic_reason,
                     "archive_detail": (
                         failure.detail_safe if failure is not None
                         else run.get("archive_detail", ""))})
        print("wrote", fn)

    by_date = {}                                # date -> [runs] (insertion = newest first)
    for r in meta:
        day = datetime.datetime.fromtimestamp(r["mtime"]).strftime("%Y-%m-%d") if r["mtime"] else "undated"
        by_date.setdefault(day, []).append(r)
    tree = []
    for di, day in enumerate(sorted(by_date, reverse=True)):
        druns = by_date[day]
        active_runs = [
            r for r in druns
            if not r["archive_reason"] and not r["managed_diagnostic"]
        ]
        archived_runs = [r for r in druns if r["archive_reason"]]
        diagnostic_runs = [r for r in druns if r["managed_diagnostic"]]
        by_task = {}
        for r in active_runs:
            by_task.setdefault(r["task"], []).append(r)
        tasks = sorted(by_task, key=lambda tk: max(x["mtime"] for x in by_task[tk]), reverse=True)
        dsucc = sum(1 for r in active_runs if r["ok"])
        versions = list(dict.fromkeys(filter(None, (_version_label(r) for r in active_runs))))
        version_note = (" · commit " + ", ".join(versions)) if versions else " · commit 未记录"
        tree.append(f"<details class='lvl-date'{' open' if di == 0 else ''}>"
                    f"<summary>📅 {esc(day)}<span class='cnt'>{esc(version_note)} · "
                    f"{len(active_runs)} valid runs · {dsucc}✓ / "
                    f"{len(active_runs) - dsucc}✗ · {len(archived_runs)} excluded · "
                    f"{len(diagnostic_runs)} managed diagnostics</span></summary>")
        for tk in tasks:
            truns = by_task[tk]
            tsucc = sum(1 for r in truns if r["ok"])
            tree.append(f"<details class='lvl-task' open><summary>{esc(tk)}"
                        f"<span class='cnt'>{len(truns)} · {tsucc}✓</span></summary>"
                        f"<table><tr><th>run</th><th>特征</th><th>verifier</th>"
                        f"<th>运行版本</th><th>轮数</th><th>墙钟(s)</th></tr>")
            for r in truns:                     # already newest first
                pill = ("<span class='pill ok'>SUCCESS</span>" if r["ok"]
                        else "<span class='pill bad'>FAILURE</span>")
                st = "" if r["status"] in ("done", "?", None) else f" <span class='note'>({esc(r['status'])})</span>"
                tree.append(f"<tr><td><a href='{r['fn']}'>{esc(r['name'])}</a>{st}</td>"
                            f"<td class='note'>{esc(r['tag'])}</td><td>{pill}</td>"
                            f"<td>{_version_html(r)}</td><td>{esc(r['steps'])}</td>"
                            f"<td>{esc(r['wall'])}</td></tr>")
            tree.append("</table></details>")
        if diagnostic_runs:
            tree.append("<details class='archive'><summary>Managed diagnostics"
                        f"<span class='cnt'>{len(diagnostic_runs)} retry/unselected executions"
                        " · never counted</span></summary>"
                        "<table><tr><th>run</th><th>batch</th><th>reason</th>"
                        "<th>status</th><th>轮数</th><th>墙钟(s)</th></tr>")
            for r in diagnostic_runs:
                tree.append(
                    f"<tr><td><a href='{r['fn']}'>{esc(r['name'])}</a></td>"
                    f"<td><code>{esc(r['managed_batch'])}</code></td>"
                    f"<td>{esc(r['managed_diagnostic_reason'])}</td>"
                    f"<td>{esc(r['status'])}</td><td>{esc(r['steps'])}</td>"
                    f"<td>{esc(r['wall'])}</td></tr>")
            tree.append("</table></details>")
        if archived_runs:
            tree.append("<details class='archive'><summary>🗄 Excluded"
                        f"<span class='cnt'>{len(archived_runs)} 个非计分 attempt</span>"
                        "</summary><table><tr><th>run</th><th>归档原因</th><th>状态</th>"
                        "<th>轮数</th><th>墙钟(s)</th></tr>")
            for r in archived_runs:
                detail = f" · {r['archive_detail']}" if r.get("archive_detail") else ""
                tree.append(f"<tr><td><a href='{r['fn']}'>{esc(r['name'])}</a></td>"
                            f"<td><span class='pill archive-pill'>{esc(r['archive_reason'])}</span>"
                            f"<span class='note'>{esc(detail)}</span></td>"
                            f"<td>{esc(r['status'])}</td><td>{esc(r['steps'])}</td>"
                            f"<td>{esc(r['wall'])}</td></tr>")
            tree.append("</table></details>")
        tree.append("</details>")

    legend = """<details><summary style='background:#eef1f5;font-size:14px'>动作状态解释表(点击展开)</summary>
<table>
<tr><th style='width:180px'>状态</th><th>含义</th></tr>
<tr><td><span class='badge S'>SUCCESS</span></td><td>动作执行完成并交付。注意 achieved(实际执行量)
仍可能与 commanded(指令量)有毫米级差异——这是真实的执行器跟踪误差,判断时以 achieved 为准。</td></tr>
<tr><td><span class='badge F'>FAILED</span></td><td>动作尝试执行但没有交付。最常见原因是 cuRobo 运动
规划失败:目标位姿在该手臂的可达范围之外、接近关节极限、或与桌面碰撞模型(z≈0.76)/手臂自身冲突。
move_delta 还会细分:某段直线腿规划失败 / 连续两腿无物理进展(失速)/ 腿数预算内未收敛到目标。
手臂停留在失败时的位置,achieved 报告已真实走过的部分位移——模型可以据此换一个目标重试。</td></tr>
<tr><td><span class='badge A'>ABORTED / workspace</span></td><td>安全护栏在触碰仿真之前拒绝:目标点在
工作空间安全包围盒之外。手臂完全没有移动。</td></tr>
<tr><td><span class='badge A'>ABORTED / arm_lock</span></td><td>安全护栏拒绝:另一只手臂正被指令占用
(默认一次只允许命令一只手臂,双臂协同需要显式的成对调用)。动作未执行。</td></tr>
<tr><td><span class='badge A'>ABORTED / contact</span></td><td>接触中止:受监护的移动中手指出现了新的
物理接触,运动立即停止。这不是错误——向下探底时这正是期望的『碰到了』信号,是 depth-free 方法获得
高度信息的合法途径。仅当移动开始时无接触才布防(手里拿着东西移动属于预期接触,不会误触发)。</td></tr>
<tr><td><span class='badge E'>ERR</span></td><td>工具调用本身无效:参数类型错误/缺参、引用了不存在的
观测编号、观测已过期(手臂在拍摄后移动过)、几何退化(如极线需要两个不同视角)等。错误原文会返回给
模型,episode 继续。</td></tr></table>
<p class='note'><code>run_code</code> 的 SUCCESS 仅表示沙箱代码正常执行并返回;代码内部调用的
机器人动作仍可能返回 FAILED/ABORTED,应继续检查其 value 中保存的动作结果。</p>
<p class='note'>FAILED 与 ABORTED 的本质区别:FAILED = 执行器/规划器<b>尝试后</b>做不到;
ABORTED = 安全护栏<b>主动拒绝或中断</b>(其中 contact 中止往往是有用的信号而非失败)。
两者都把完整原因返回给模型;能否利用这些文字反馈调整策略,正是被测能力之一。</p></details>"""
    valid = [
        r for r in meta
        if not r["archive_reason"] and not r["managed_diagnostic"]
    ]
    archived = [r for r in meta if r["archive_reason"]]
    diagnostics = [r for r in meta if r["managed_diagnostic"]]
    legacy_valid = [r for r in valid if r["managed_batch"] is None]
    valid_managed_batches = [batch for batch in managed_batches if batch["valid"]]
    headline_count = len(legacy_valid) + sum(
        batch["attempt_count"] for batch in valid_managed_batches)
    nsucc = sum(1 for r in legacy_valid if r["ok"]) + sum(
        batch["accepted_successes"] for batch in valid_managed_batches)
    idx = (f"<html><head><meta charset='utf-8'><title>codeaction live runs</title><style>{CSS}</style>"
           f"</head><body><div class='wrap'><h1>Depth-free code-as-policy · live run 报告索引</h1>"
           f"<p class='note'>命名 = 任务 · 模型 · 接口特征。全部为真实模型驱动的 episode;"
           f"报告只做执行分析(无原理章节)。按 <b>日期 → 任务 → run</b> 三层折叠,最新在前;"
           f"最新日期默认展开。共 {headline_count} valid runs · {nsucc}✓ / "
           f"{headline_count - nsucc}✗；Excluded {len(archived)} 个非计分 attempt；"
           f"Managed diagnostics {len(diagnostics)} 个未选 execution，均不进入分母。</p>"
           f"{_managed_batches_html(managed_batches)}"
           f"{_aggregate_html(_aggregate_cells([r for r in meta if not r['managed_diagnostic']]))}"
           f"{''.join(tree)}<h2 style='margin-top:24px'>图例</h2>{legend}</div></body></html>")
    (OUT / "index.html").write_text(idx, encoding="utf-8")
    print("wrote index.html —", len(meta), "runs")


def finalize_run(run_dir, name, tag, verifier=None, interface=None, extra_meta=None):
    """Stamp this run's metadata, then rebuild the report if this process can write it.

    The two halves have different requirements and used to fail together. Stamping writes into
    the run directory, which is the writable output mount even inside the sim container; the
    rebuild writes `runs_report/` in the worktree, which that container mounts read-only. Raising
    from the rebuild therefore made a containerized caller log the whole call as skipped, and the
    index stopped advancing for nine days while every run's metadata was in fact being recorded.

    Returns whether the report itself was rebuilt. A caller that gets False is expected to have a
    host-side rebuild; `codeaction` does one when the attempt finishes.
    """
    register_run(run_dir, name, tag, verifier, interface, extra_meta=extra_meta)
    try:
        backfill_legacy()
        rebuild()
    except OSError:
        return False
    return True


def main():
    backfill_legacy()
    rebuild()


if __name__ == "__main__":
    main()
