"""Run-level metric aggregation — the production closure of the R5 protocol.

`metrics.py` holds the math (Wilson CI95, unbiased pass@k, per-attempt aggregation); this module
is what turns a RUN DIRECTORY on disk into the headline number, so no run can end without one.
It is deliberately track-agnostic: the reference scaffold writes `<out>/a<k>/result.json` (or `<out>/result.json`
for a single attempt), the vendor agent writes `<out>/s<seed>/result.json` — both are "one group = one
(task, model, interface) cell", and both are read here by the same walker.

Two rules that keep the number honest:

1. **Non-scoreable attempts never enter the denominator.** Transcript/result schema 1.2 makes
   `failure.scoreable` authoritative. Legacy `archive_meta.json` remains a read-only fallback for
   schema-1.1 runs and is reported separately instead of being silently reinterpreted.
2. **A number that did not run the declared protocol says so.** `attempts_declared` comes from the
   task card (`protocol.attempts_k`); `protocol_complete` is false whenever fewer valid attempts
   ran, and `protocol_valid` additionally requires one consistent identity (single model,
   interface and task-pack version across the group). A dev run at n=3 is still reported — it is
   just structurally marked as not a release number, so it can never be quoted as one.

Pure stdlib + `codeaction.verification.metrics`; no sim, no endpoint, locally testable.
"""
import json
import re
from pathlib import Path

from codeaction.contracts.failures import failure_from_legacy_status, failure_of
from codeaction.contracts.identity import comparison_key, validate_trial_against_randomness
from codeaction.verification.metrics import (aggregate_attempts, contact_pose_telemetry, contact_safety_stats,
                             milestone_funnel_across_attempts,
                             reachability_probe_telemetry, stall_telemetry)

# Bump rules mirror TRANSCRIPT_SCHEMA_VERSION: MINOR for a new field old readers can ignore,
# MAJOR when a field is removed, renamed, or changes meaning.
# 1.1 — additive `contact_safety`: contact-monitor interruptions counted per cell as an
# independent safety axis. It never enters `success_rate`. The version is what distinguishes a
# summary written BEFORE the axis existed from one whose run simply had no contact aborts.
# 1.2 — additive `milestone_funnel` (pooled per-milestone partition: partial progress a binary
# verdict cannot show) + `tokens` (uncached/cached prompt split: what makes a run priceable) +
# `pass_hat_k` beside `pass_at_k`. All three are read-only views over data the attempts already
# carried; no verdict, threshold or denominator changed. Same reason for the bump as 1.1: an
# absent block must be readable as "this summary predates the axis", never as a measured zero.
# 1.3 — additive analysis-only contact-pose channel telemetry: observation/part counts, maximum
# within-attempt xyz span, and transcript coverage so missing records remain unknown rather than 0.
# 1.4 — additive host-only stall attribution from motion_trace.jsonl: physics steps spent in
# stalled actions, whether later physical actions ran, and whether a success latch fell inside one.
SUMMARY_SCHEMA_VERSION = "1.6"
_ATTEMPT_DIR = re.compile(r"^[as]\d+$")      # the reference scaffold "a1", the vendor agent "s0"


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path):
    records = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    return records


def identity_of(result, run_meta=None) -> dict:
    """The (task, model, interface) cell an attempt belongs to, plus its version bindings.

    Field sources, in priority order: the controller-authored `provenance` block (the reference scaffold bare
    metal and the containerised the vendor agent both write one), then the attempt's own result keys, then
    run_meta.json. `result["task"]` is the INSTRUCTION TEXT in both tracks — never a task name —
    so it is deliberately not consulted here.
    """
    result = result or {}
    run_meta = run_meta or {}
    prov = result.get("provenance") or run_meta.get("provenance") or {}
    if not isinstance(prov, dict):
        prov = {}
    identity = result.get("identity") or run_meta.get("identity") or prov.get("expected_identity")
    if isinstance(identity, dict) and isinstance(identity.get("comparison"), dict):
        comparison = identity["comparison"]
        task = comparison.get("task") or {}
        model = comparison.get("model") or {}
        tested = comparison.get("tested_unit") or {}
        task_pack = comparison.get("task_pack") or {}
        environment = comparison.get("environment") or {}
        return {
            "task_name": task.get("id"),
            "model": model.get("id"),
            "interface": tested.get("interface_profile"),
            "taskset_version": task_pack.get("version"),
            "task_pack_sha256": task_pack.get("sha256"),
            "source_commit": environment.get("source_commit"),
            "comparison_sha256": comparison_key(identity),
            "comparison": comparison,
            "trial": identity.get("trial"),
        }

    def pick(*values):
        for v in values:
            if isinstance(v, str) and v:
                return v
        return None

    return {
        "task_name": pick(prov.get("task_name"), result.get("task_name"),
                          run_meta.get("task_name")),
        "model": pick(prov.get("model"), result.get("model"), run_meta.get("model")),
        "interface": pick(prov.get("interface"), result.get("interface"),
                          run_meta.get("interface")),
        "taskset_version": pick(prov.get("task_pack_version"), run_meta.get("task_pack_version")),
        "task_pack_sha256": pick(prov.get("task_pack_sha256"),
                                 run_meta.get("task_pack_sha256")),
        "source_commit": pick(prov.get("source_commit"), run_meta.get("source_commit"),
                              run_meta.get("git_commit")),
        "comparison_sha256": None,
        "comparison": None,
        "trial": None,
    }


def collect_attempts(run_dir) -> tuple:
    """Walk one run group → (scoreable_records, excluded_records).

    The tuple shape is unchanged for existing callers. Each excluded record states whether it came
    from the 1.2 failure contract or the legacy archive sidecar.
    """
    root = Path(run_dir)
    valid, archived = [], []
    if not root.is_dir():
        return valid, archived
    candidates = [root] if (root / "result.json").is_file() else []
    candidates.extend(sorted(d for d in root.iterdir()
                             if d.is_dir() and (d / "result.json").is_file()))
    for d in candidates:
        result = _read_json(d / "result.json")
        if result is None:
            continue
        entry = {"dir": d.name if d != root else ".", "result": result,
                 "run_meta": _read_json(d / "run_meta.json") or {},
                 "agent_exit": _read_json(d / "agent_exit.json") or {},
                 "tool_records": _read_jsonl(d / "transcript.jsonl"),
                 "motion_records": _read_jsonl(d / "tools" / "motion_trace.jsonl")}
        failure = failure_of(result, entry["agent_exit"])
        archive = _read_json(d / "archive_meta.json") or {}
        if failure is not None:
            entry["failure"] = failure.to_dict()
        if failure is not None and not failure.scoreable:
            entry["archive_reason"] = failure.code.value
            entry["exclusion_kind"] = "classified_failure"
            archived.append(entry)
        elif failure is None and archive.get("archive") is True:
            entry["archive_reason"] = str(archive.get("reason") or "explicit_infra_archive")
            entry["exclusion_kind"] = "legacy_archive"
            archived.append(entry)
        else:
            # Compatibility only: 1.1 result files have no structured failure. Preserve their
            # historical scoreability instead of retroactively changing old denominators.
            stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
            status = result.get("status") or stats.get("status")
            tested_origin = ("agent"
                             if str(result.get("interface") or "").lower() == "mcp-agent"
                             else "model")
            legacy_failure = failure_from_legacy_status(status, tested_origin=tested_origin)
            if legacy_failure is not None and not legacy_failure.scoreable:
                entry["failure"] = legacy_failure.to_dict()
                entry["archive_reason"] = legacy_failure.code.value
                entry["exclusion_kind"] = "legacy_status"
                archived.append(entry)
            elif status == "endpoint_failure":
                entry["archive_reason"] = "endpoint_failure"
                entry["exclusion_kind"] = "legacy_status"
                archived.append(entry)
            else:
                valid.append(entry)
    return valid, archived


def _consistent(values):
    known = {v for v in values if v}
    return (known.pop() if len(known) == 1 else None), len(known) > 1


def _contact_safety(entries) -> dict:
    """Per-cell safety axis: how often the contact monitor interrupted a command.

    Deliberately NOT part of `success_rate`. The 2026-08-07 Reflection retired collision-as-verdict
    precisely because one physics step of contact proves nothing about task failure — but it is
    still the closest analogue of a real cell's protective-stop count, and an attempt that reached
    the goal by repeatedly driving into things is not the same result as a clean one.

    `transcript_coverage` is load-bearing: an attempt whose transcript could not be read counts
    nothing, so without it a run with no transcripts would render as a flawless zero. At zero
    coverage every count is None — unknown, never zero.
    """
    per_attempt = [contact_safety_stats(e.get("tool_records"))
                   for e in entries if e.get("tool_records")]
    coverage = round(len(per_attempt) / len(entries), 4) if entries else 0.0
    note = ("counted from transcript contact-monitor aborts; an independent safety axis that "
            "never enters success_rate, and not a claim about severity or task failure")
    if not per_attempt:
        unknown = {name: None for name in (
            "unexpected_contact_aborts_total", "unexpected_contact_aborts_mean",
            "unexpected_contact_aborts_max", "attempts_with_unexpected_contact",
            "contact_read_unavailable_aborts_total", "legacy_collision_terminals_total",
        )}
        unknown["transcript_coverage"] = coverage
        unknown["note"] = note
        return unknown
    aborts = [item["unexpected_contact_aborts"] for item in per_attempt]
    return {
        "unexpected_contact_aborts_total": sum(aborts),
        "unexpected_contact_aborts_mean": round(sum(aborts) / len(aborts), 4),
        "unexpected_contact_aborts_max": max(aborts),
        "attempts_with_unexpected_contact": sum(1 for value in aborts if value),
        "contact_read_unavailable_aborts_total": sum(
            item["contact_read_unavailable_aborts"] for item in per_attempt),
        "legacy_collision_terminals_total": sum(
            item["legacy_collision_terminals"] for item in per_attempt),
        "transcript_coverage": coverage,
        "note": note,
    }


def summarize_records(entries, archived=(), attempts_declared=None) -> dict:
    """Aggregate scoreable entries; ``archived`` contains all excluded attempts for compatibility."""
    scoreable_entries = []
    excluded_entries = list(archived)
    for entry in entries:
        failure = failure_of(entry.get("result"), entry.get("agent_exit"))
        if failure is not None and not failure.scoreable:
            excluded = dict(entry)
            excluded["failure"] = failure.to_dict()
            excluded["archive_reason"] = failure.code.value
            excluded["exclusion_kind"] = "classified_failure"
            excluded_entries.append(excluded)
        else:
            scoreable_entries.append(entry)
    entries = scoreable_entries
    archived = excluded_entries
    records = [e["result"] for e in entries]
    ids = [identity_of(e["result"], e.get("run_meta")) for e in entries]
    summary = {"schema_version": SUMMARY_SCHEMA_VERSION}
    mixed = []
    for field in ("task_name", "model", "interface", "taskset_version", "task_pack_sha256"):
        value, is_mixed = _consistent(i[field] for i in ids)
        summary[field] = value
        if is_mixed:
            mixed.append(field)
    comparison_digest, comparison_mixed = _consistent(
        i["comparison_sha256"] for i in ids)
    summary["comparison_sha256"] = comparison_digest
    if comparison_mixed:
        mixed.append("identity.comparison")
    commits = sorted({i["source_commit"] for i in ids if i["source_commit"]})
    summary["source_commits"] = commits
    # The token split is only derivable once the protocol's prompt-token convention is known, and
    # that comes from the model. A group with a mixed model resolves to None above and correctly
    # leaves the derived billing fields unknown rather than picking one arm's convention.
    summary.update(aggregate_attempts(records, model=summary.get("model")))
    reachability = [reachability_probe_telemetry(e.get("tool_records")) for e in entries]
    summary["reachability_queries_total"] = sum(
        item["reachability_queries"] for item in reachability)
    summary["reachability_queries_by_tool"] = {
        name: sum(item["reachability_queries_by_tool"][name] for item in reachability)
        for name in ("check_tcp_pose_reachability", "check_direction_feasibility")
    }
    z_values = sum(item["reachability_query_z_values"] for item in reachability)
    summary["reachability_query_z_values"] = z_values
    # A pooled spread cannot be reconstructed from per-attempt spreads. Keep the attribution honest
    # by reporting the largest within-attempt bracket rather than inventing cross-scene geometry.
    spreads = [item["reachability_query_z_spread_m"] for item in reachability
               if item["reachability_query_z_spread_m"] is not None]
    summary["reachability_query_z_spread_m_max"] = max(spreads) if spreads else None
    contact_pose_records = [
        e.get("tool_records") for e in entries if isinstance(e.get("tool_records"), list)]
    contact_poses = [contact_pose_telemetry(records) for records in contact_pose_records]
    summary["contact_pose_transcript_coverage"] = (
        round(len(contact_pose_records) / len(entries), 4) if entries else None)
    summary["contact_pose_observations_total"] = (
        sum(item["contact_pose_observations"] for item in contact_poses)
        if contact_poses else None)
    summary["contact_pose_observations_by_part"] = {
        part: (sum(item["contact_pose_observations_by_part"][part] for item in contact_poses)
               if contact_poses else None)
        for part in ("fingertip", "arm_link", "robot_self")
    }
    contact_pose_spreads = {}
    for axis in ("x", "y", "z"):
        values = [
            item["contact_pose_axis_spread_m"][axis] for item in contact_poses
            if item["contact_pose_axis_spread_m"][axis] is not None]
        contact_pose_spreads[axis] = max(values) if values else None
    summary["contact_pose_axis_spread_m_max"] = contact_pose_spreads
    summary["contact_safety"] = _contact_safety(entries)
    stall_rows = [
        stall_telemetry(
            entry.get("motion_records"),
            (entry.get("result") or {}).get("verifier") or {},
        )
        for entry in entries
    ]
    traced_stalls = [row for row in stall_rows if row["motion_trace_available"]]
    stall_success_rows = [
        row for row in traced_stalls if row["success_in_stalled_action"] is not None]
    summary["stall_analysis"] = {
        "motion_trace_coverage": (
            round(len(traced_stalls) / len(entries), 4) if entries else None),
        "stall_physics_steps_total": (
            sum(row["stall_physics_steps"] for row in traced_stalls)
            if traced_stalls else None),
        "stall_physics_steps_mean": (
            round(sum(row["stall_physics_steps"] for row in traced_stalls)
                  / len(traced_stalls), 4) if traced_stalls else None),
        "stall_actions_total": (
            sum(row["stall_actions"] for row in traced_stalls) if traced_stalls else None),
        "attempts_with_stall": (
            sum(1 for row in traced_stalls if row["stall_actions"])
            if traced_stalls else None),
        "attempts_continued_after_stall": (
            sum(1 for row in traced_stalls
                if row["continued_physical_action_after_stall"])
            if traced_stalls else None),
        "success_action_coverage": (
            round(len(stall_success_rows) / len(entries), 4) if entries else None),
        "successes_observed_in_stalled_action": (
            sum(1 for row in stall_success_rows if row["success_in_stalled_action"])
            if stall_success_rows else None),
        "note": (
            "analysis-only; physical actions come from host-only motion_trace records and "
            "success boundaries from the card-declared env_success_observed latch"),
    }
    # Partial-progress attribution for the whole cell. Pooled here rather than in the per-attempt
    # report because a leaderboard row is a group: five 0.0 attempts that all reached the last
    # milestone and five that never moved an object are the same headline number and completely
    # different results.
    summary["milestone_funnel"] = milestone_funnel_across_attempts(
        [record.get("verifier") or {} for record in records])
    summary["attempts_declared"] = (int(attempts_declared)
                                    if isinstance(attempts_declared, int) else None)
    legacy_archived = [e for e in archived
                       if e.get("exclusion_kind") in (None, "legacy_archive")]
    legacy_status = [e for e in archived if e.get("exclusion_kind") == "legacy_status"]
    classified_excluded = [e for e in archived
                           if e.get("exclusion_kind") == "classified_failure"]
    summary["attempts_excluded"] = len(archived)
    summary["attempts_archived"] = len(legacy_archived)
    summary["attempts_excluded_legacy_status"] = len(legacy_status)
    summary["attempts_excluded_non_scoreable"] = len(classified_excluded)
    if legacy_archived:
        summary["archive_reasons"] = sorted(
            {e.get("archive_reason", "unknown") for e in legacy_archived})
    if classified_excluded:
        counts = {}
        for entry in classified_excluded:
            failure = failure_of(entry.get("result"), entry.get("agent_exit"))
            key = (f"{failure.origin.value}/{failure.code.value}"
                   if failure is not None else "unknown/unknown")
            counts[key] = counts.get(key, 0) + 1
        summary["excluded_failure_counts"] = counts
    if legacy_status:
        summary["legacy_status_exclusion_counts"] = {
            reason: sum(1 for e in legacy_status if e.get("archive_reason") == reason)
            for reason in sorted({e.get("archive_reason", "unknown") for e in legacy_status})
        }
    dirs = [e["dir"] for e in entries if e.get("dir")]   # absent for in-memory callers
    if dirs:
        summary["attempt_dirs"] = dirs
    statuses = {}
    for r in records:
        s = str((r.get("stats") or {}).get("status", "unknown"))
        statuses[s] = statuses.get(s, 0) + 1
    summary["episode_status_counts"] = statuses
    failure_counts = {}
    for entry in entries:
        failure = failure_of(entry.get("result"), entry.get("agent_exit"))
        if failure is None:
            continue
        key = f"{failure.origin.value}/{failure.code.value}"
        failure_counts[key] = failure_counts.get(key, 0) + 1
    summary["scoreable_failure_counts"] = failure_counts
    claims = []
    for record in records:
        done_report = (record.get("stats") or {}).get("done_report")
        claim = done_report.get("success_claim") if isinstance(done_report, dict) else None
        claims.append((claim if isinstance(claim, bool) else None,
                       bool((record.get("verifier") or {}).get("success"))))
    scored = [(c, v) for c, v in claims if c is not None]
    if records:
        summary["claim_coverage"] = round(len(scored) / len(records), 4)
    if scored:                                    # truthfulness axis: claim vs out-of-band fact
        summary["claim_accuracy"] = round(sum(1 for c, v in scored if c == v) / len(scored), 4)
        summary["false_claim_rate"] = round(sum(1 for c, v in scored if c and not v)
                                            / len(scored), 4)
    # Verdict-sampling evidence (taskset 0.14.0): the binary verdict is one end-state read of the
    # env's own predicate, so an attempt that reached the goal state and then lost it reads exactly
    # like one that never reached it. `env_success_observed` is the in-episode record of that same
    # predicate. Coverage is reported beside the count because pre-0.14.0 runs have no record and
    # an absent record must never be counted as "did not happen".
    observed = [(r.get("verifier") or {}).get("env_success_observed") for r in records]
    monitored = [(o, bool((r.get("verifier") or {}).get("success")))
                 for o, r in zip(observed, records)
                 if isinstance(o, dict) and o.get("monitored")]
    if records:
        summary["env_success_observed_coverage"] = round(len(monitored) / len(records), 4)
    if monitored:
        summary["n_env_success_lost_before_end"] = sum(
            1 for o, success in monitored if o.get("latched") and not success)
        voided = sum(1 for o, _ in monitored if o.get("void"))
        if voided:                       # the scene satisfied the task before the agent acted
            summary["env_success_void_at_entry"] = voided
    identity_errors = []
    post_migration = [identity for identity in ids if identity["comparison"] is not None]
    if post_migration and len(post_migration) != len(ids):
        identity_errors.append("mixed legacy and current identities")
    seen_attempts = set()
    seen_seeds = set()
    for identity in post_migration:
        randomness = identity["comparison"]["randomness_protocol"]
        try:
            validate_trial_against_randomness(
                identity["trial"], randomness)
        except (KeyError, TypeError, ValueError) as exc:
            identity_errors.append(str(exc))
            continue
        attempt_index = identity["trial"]["attempt_index"]
        scene_seed = identity["trial"]["scene_seed"]
        if attempt_index in seen_attempts:
            identity_errors.append(f"duplicate attempt_index {attempt_index}")
        fixed_scene = randomness.get("seed_policy") in ("fixed_scene_seed_attempts", "fixed_scene_seed_repeats")
        if scene_seed in seen_seeds and not fixed_scene:
            identity_errors.append(f"duplicate scene_seed {scene_seed}")
        seen_attempts.add(attempt_index)
        seen_seeds.add(scene_seed)
        if identity["comparison"]["environment"].get("submittable") is False:
            identity_errors.append("runtime identity is dev-only")
    complete = bool(summary["attempts_declared"]) and \
        summary["n_attempts"] == summary["attempts_declared"]
    summary["protocol_complete"] = complete
    blockers = []
    if not complete:
        blockers.append(f"ran {summary['n_attempts']} of "
                        f"{summary['attempts_declared'] or 'an undeclared number of'} attempts")
    if mixed:
        blockers.append(f"mixed identity across attempts: {mixed}")
    blockers.extend(sorted(set(identity_errors)))
    if post_migration and not summary["comparison_sha256"]:
        blockers.append("no stable comparison identity")
    if not summary["taskset_version"]:
        blockers.append("no task-pack version recorded")
    if post_migration and not mixed:
        randomness = post_migration[0]["comparison"]["randomness_protocol"]
        if randomness.get("seed_policy") in ("fixed_scene_seed_attempts", "fixed_scene_seed_repeats"):
            primary = randomness.get("scene_seeds") or []
            summary["fixed_scene_reliability"] = {
                "primary_scene_seed": primary[0] if len(primary) == 1 else None,
                "backup_scene_seeds": list(
                    randomness.get("backup_scene_seeds") or []),
                "attempts": int(randomness.get("attempts_k", 0)),
                "successes": summary["n_success"],
                # Operational reliability claim: every declared independent attempt completed
                # successfully. The raw successes/attempts remain beside it; this is not a
                # population-probability estimate from five samples.
                "reliably_solved": bool(
                    complete and summary["n_success"] == summary["attempts_declared"]),
            }
    summary["protocol_valid"] = not blockers
    summary["validity_errors"] = blockers
    # Submission eligibility is a separate question from protocol validity: a complete, internally
    # consistent run of a scripted model is valid but not submittable. Each attempt records its own
    # verdict beside its identity; the group reports the union.
    non_submittable = sorted({
        reason
        for record in records
        if record.get("submittable") is False
        for reason in (record.get("non_submittable_reasons") or ["unspecified"])
    })
    summary["submittable"] = not non_submittable and not blockers
    summary["non_submittable_reasons"] = non_submittable
    return summary


def attempts_declared_for(task_name, tasks_root=None):
    """The card's declared attempts_k, or None when the task is not in the given pack."""
    if not task_name:
        return None
    try:
        from codeaction.benchmark.taskcard import TASKS_ROOT, load_task
        card = load_task(task_name, tasks_root=tasks_root or TASKS_ROOT)
    except Exception:
        return None
    k = (card.get("protocol") or {}).get("attempts_k")
    return int(k) if isinstance(k, int) and not isinstance(k, bool) and k > 0 else None


def summarize_run_dir(run_dir, attempts_declared=None, tasks_root=None, write=True):
    """Walk one run group, aggregate it, and (by default) write `<run_dir>/summary.json`.
    Returns None when the directory holds no attempt at all."""
    root = Path(run_dir)
    valid, archived = collect_attempts(root)
    if not valid and not archived:
        return None
    if attempts_declared is None and valid:
        task_name = identity_of(valid[0]["result"], valid[0].get("run_meta"))["task_name"]
        attempts_declared = attempts_declared_for(task_name, tasks_root=tasks_root)
    summary = summarize_records(valid, archived, attempts_declared=attempts_declared)
    summary["run_dir"] = str(root)
    if write:
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def discover_groups(base):
    """Every run group under `base`: a directory that itself holds result.json, or whose attempt
    subdirectories do. An attempt directory is never reported as a group of its own."""
    base = Path(base)
    if not base.is_dir():
        return []
    groups, attempt_dirs = [], set()
    for d in sorted(p for p in base.rglob("*") if p.is_dir()):
        if "runs_report" in d.parts or d.name == "tools":
            continue
        children = [c for c in sorted(d.iterdir())
                    if c.is_dir() and (c / "result.json").is_file()]
        if children:
            groups.append(d)
            attempt_dirs.update(children)
        elif (d / "result.json").is_file() and d not in attempt_dirs \
                and not _ATTEMPT_DIR.match(d.name):
            groups.append(d)
    return [g for g in groups if g not in attempt_dirs]
