"""Benchmark metrics protocol (convention adopted from Terminal-Bench):
the headline number is the resolved rate over a FIXED
number of attempts per task, reported with a 95% confidence interval; the unbiased pass@k
estimator (Chen et al. 2021 — the same one TB1's harness ships) is the auxiliary view. Pure math,
no sim, locally testable. Wilson is used for the interval because attempt counts are small."""
import math
from collections.abc import Mapping, Sequence

from codeaction.motion.action_contact_policy import PHYSICS_ACTION_TOOLS

_Z95 = 1.959963984540054


def _nonnegative_int(value):
    return (int(value) if isinstance(value, int) and not isinstance(value, bool)
            and value >= 0 else None)


def resource_accounting(record_or_stats, transcript_records=(), *, max_tool_calls=None) -> dict:
    """Return one attempt's resource counters without mixing their units.

    `budget_used`/legacy `steps` count charged outer tool calls. `total_calls` also includes free
    control and rejected over-limit dispatches. `turns` counts provider model turns. Only the first
    quantity may be divided by `max_tool_calls`; a model-turn percentage against that ceiling is
    dimensionally invalid.
    """
    value = record_or_stats if isinstance(record_or_stats, Mapping) else {}
    stats = value.get("stats") if isinstance(value.get("stats"), Mapping) else value
    attestation = value.get("identity_attestation")
    if isinstance(attestation, Mapping) and attestation.get("reference_agent_match") is True:
        reference = attestation.get("reference_agent")
        if isinstance(reference, Mapping) and reference.get("healthy") is True \
                and isinstance(reference.get("stats"), Mapping):
            stats = reference["stats"]
    records = (list(transcript_records)
               if isinstance(transcript_records, Sequence) and not isinstance(
                   transcript_records, (str, bytes)) else [])
    meta = next((row for row in records
                 if isinstance(row, Mapping) and row.get("event") == "meta"), {})

    used = _nonnegative_int(stats.get("tool_calls_used"))
    if used is None:
        used = _nonnegative_int(stats.get("budget_used"))
    if used is None:
        used = _nonnegative_int(stats.get("steps"))
    budget = _nonnegative_int(stats.get("tool_call_budget"))
    if budget is None:
        budget = _nonnegative_int(max_tool_calls)
    if budget is None:
        budget = _nonnegative_int(meta.get("max_tool_calls"))
    if budget is None:
        budget = _nonnegative_int(meta.get("max_steps"))
    dispatches = _nonnegative_int(stats.get("total_tool_dispatches"))
    if dispatches is None:
        dispatches = _nonnegative_int(stats.get("total_calls"))
    model_turns = _nonnegative_int(stats.get("model_turns"))
    if model_turns is None:
        model_turns = _nonnegative_int(stats.get("turns"))
    if model_turns is None and records:
        model_turns = sum(
            1 for row in records
            if isinstance(row, Mapping) and row.get("event") == "model_turn")
    return {
        "budget_unit": "tool_calls",
        "tool_calls_used": used,
        "tool_call_budget": budget,
        "tool_call_budget_utilization": (
            round(used / budget, 4)
            if used is not None and budget is not None and budget > 0 else None),
        "total_tool_dispatches": dispatches,
        "model_turns": model_turns,
    }


def wilson_ci95(successes: int, n: int):
    """Wilson score 95% interval for a binomial proportion (well-behaved at small n / 0% / 100%,
    unlike the normal approximation)."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (_Z95 / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-c, k)/C(n, k) for n attempts with c successes."""
    if k > n:
        raise ValueError(f"k={k} exceeds the number of attempts n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k: C(c, k)/C(n, k) — the probability that ALL k trials of a randomly drawn
    size-k subset succeeded (the reliability counterpart of pass@k, named as in tau-bench).

    pass@k rewards a model that solves a task at least once in k tries; pass^k only rewards one
    that solves it every time. On a fixed-scene protocol (the same seed repeated k times) pass^k
    is the operational number: it answers "can this be relied on", which is what a manipulation
    task cares about. At k=1 the two estimators coincide."""
    if k > n:
        raise ValueError(f"k={k} exceeds the number of attempts n={n}")
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def _usage_total(records, key):
    values = [_usage_of(record).get(key, 0) for record in records]
    return None if any(value is None for value in values) else sum(values)


def _usage_of(record):
    identity = record.get("identity_attestation")
    if isinstance(identity, dict) and identity.get("reference_agent_match") is True:
        reference = identity.get("reference_agent")
        if isinstance(reference, dict) and reference.get("healthy") is True:
            stats = reference.get("stats")
            if isinstance(stats, dict) and isinstance(stats.get("usage"), dict):
                return stats["usage"]
    return record.get("stats", {}).get("usage", {})


MOTION_TOOLS = {"move_delta", "move_both_delta", "reach_tcp", "reach_both_tcp",
                "probe_contact_along", "probe_contact_z", "capture_motion_pair",
                "scale_from_motion"}
_WASTED_EPS_M = 0.001


def _delta_mag(achieved):
    """Max commanded-arm displacement magnitude from an ActionResult.achieved payload
    (single-arm `delta` or dual-arm `left/right.delta`); None when unreadable."""
    if not isinstance(achieved, dict):
        return None
    mags = []
    if isinstance(achieved.get("delta"), (list, tuple)):
        mags.append(math.sqrt(sum(float(v) ** 2 for v in achieved["delta"])))
    for side in ("left", "right"):
        s = achieved.get(side)
        if isinstance(s, dict) and isinstance(s.get("delta"), (list, tuple)):
            mags.append(math.sqrt(sum(float(v) ** 2 for v in s["delta"])))
    return max(mags) if mags else None


def _motion_calls(events):
    """Yield every executed motion call, whether issued directly or from inside ``run_code``.

    A ``run_code`` call is one transcript event however many robot commands it executed, and its
    commands appear only in the result's ``internal_trace``. Counting the top level alone therefore
    scored a model that drives the robot from the sandbox as having made no motion at all: on the
    2026-08-15 `scan_object` set it reported 0 motion calls for claude-opus-5 and gemini-3.6-flash,
    both of which had moved objects by more than 0.2 m.
    """
    for e in events or []:
        if e.get("event") != "tool":
            continue
        if e.get("tool") in MOTION_TOOLS:
            yield e.get("tool"), (e.get("result") or {})
        result = e.get("result")
        trace = result.get("internal_trace") if isinstance(result, dict) else None
        for item in trace or ():
            if isinstance(item, dict) and item.get("tool") in MOTION_TOOLS:
                yield item["tool"], (item.get("result") or {})


def motion_stats(events) -> dict:
    """Result-aware wasted-motion metrics over transcript `tool` events (L2, 2026-07-04: naive
    repeated-call detectors miss our loops — thrash lives in RESULT space). Wasted = a motion
    command that FAILED, or 'succeeded' with ~zero achieved displacement and no new contact.
    ABORTED-on-contact is EVIDENCE, not waste. Analysis only — never an in-loop gate (L3)."""
    n_motion = n_wasted = streak = max_streak = 0
    for tool, res in _motion_calls(events):
        if tool in {"capture_motion_pair", "scale_from_motion"} \
                and isinstance(res.get("motion"), dict):
            res = res["motion"]
        if not isinstance(res, dict) or "status" not in res:
            continue
        n_motion += 1
        status = res.get("status")
        ach = res.get("achieved") or {}
        contact = bool(ach.get("contact_after")) or res.get("abort_reason") == "contact" \
            or ach.get("stop_reason") == "contact"
        mag = _delta_mag(ach)
        wasted = (status == "FAILED") or (
            status == "SUCCESS" and mag is not None and mag < _WASTED_EPS_M and not contact)
        if wasted:
            n_wasted += 1
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {"n_motion": n_motion, "n_wasted": n_wasted,
            "wasted_motion_rate": round(n_wasted / n_motion, 4) if n_motion else None,
            "max_no_progress_streak": max_streak}


def stall_telemetry(motion_records, verifier=None) -> dict:
    """Attribute stalls and a polled success event to host-only physical-action boundaries.

    ``motion_trace.jsonl`` contains direct and ``run_code`` internal actions in one tick order.
    Its physics-step counts partition StepObserver time, so the absolute step recorded by
    ``env_success_observed`` can be located without exposing extra state to the agent. Missing
    trace/latch evidence remains ``None``, never a fabricated false or zero.
    """
    rows = [row for row in (motion_records or []) if isinstance(row, dict)]
    if not rows:
        return {
            "motion_trace_available": False,
            "physical_actions": None,
            "stall_actions": None,
            "stall_physics_steps": None,
            "physical_actions_after_first_stall": None,
            "continued_physical_action_after_stall": None,
            "success_latch_step": None,
            "success_action": None,
            "success_in_stalled_action": None,
        }

    ordered = sorted(rows, key=lambda row: (
        row.get("tick") if isinstance(row.get("tick"), int) else math.inf))
    stalls = [row for row in ordered if row.get("stop_condition") == "stalled"]
    first_stall_tick = min(
        (row.get("tick") for row in stalls if isinstance(row.get("tick"), int)),
        default=None,
    )
    after = [
        row for row in ordered
        if first_stall_tick is not None
        and isinstance(row.get("tick"), int)
        and row["tick"] > first_stall_tick
        and isinstance(row.get("physics_steps"), int)
        and row["physics_steps"] > 0
    ]

    observed = (verifier or {}).get("env_success_observed")
    success_step = (
        observed.get("step")
        if isinstance(observed, dict)
        and observed.get("monitored") is True
        and observed.get("latched") is True
        and not observed.get("void")
        and isinstance(observed.get("step"), int)
        else None
    )
    success_action = None
    cumulative = 0
    for row in ordered:
        steps = row.get("physics_steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            continue
        start, cumulative = cumulative + 1, cumulative + steps
        if success_step is not None and start <= success_step <= cumulative:
            success_action = {
                "action_id": row.get("action_id"),
                "tick": row.get("tick"),
                "tool": row.get("tool"),
                "start_step": start,
                "end_step": cumulative,
                "stalled": row.get("stop_condition") == "stalled",
            }
            break

    return {
        "motion_trace_available": True,
        "physical_actions": len(ordered),
        "stall_actions": len(stalls),
        "stall_physics_steps": sum(int(row.get("physics_steps") or 0) for row in stalls),
        "physical_actions_after_first_stall": len(after),
        "continued_physical_action_after_stall": bool(after) if stalls else False,
        "success_latch_step": success_step,
        "success_action": success_action,
        "success_in_stalled_action": (
            success_action["stalled"] if success_action is not None else None),
    }


def token_billing(recs, model=None) -> dict:
    """The billing split of one run group: uncached prompt, cached prompt, completion.

    A prompt-token count alone cannot be priced. Every provider we run bills cached input at a
    fraction of the uncached rate, so two runs with the same prompt total can differ several-fold
    in cost, and a scaffold change (image retention is the live example) moves the total and the
    cached share in OPPOSITE directions. This function therefore reports the split, never a
    currency amount: prices are external, dated facts that belong to whoever publishes the table,
    not to this module.

    `prompt_total` is a VENDOR-REPORTED sum and its meaning depends on the protocol: Anthropic
    counts only the tokens billed at the full input rate, everyone else counts the whole prompt
    with the cached part inside it. Deriving the split therefore needs `model`, which
    `codeaction.evidence.cost_ledger` resolves to a convention through the registry. Subtracting blind is what
    reported `uncached_prompt_total = -3,502,714` for claude-opus-5 on the 2026-08-13 canary and
    a `cache_hit_rate` of 4.0615. `input_total` is the protocol-neutral denominator -- uncached +
    cache write + cache read -- and is what `cache_hit_rate` divides by, so the ratio is
    comparable across vendors instead of meaning two different things per column.

    Anything that would have to be guessed stays None: providers that never report `cached_tokens`
    leave the split unknown (`cached_coverage` says how many attempts did report it), and so does
    a group whose model is unknown here. A silent zero would read as "nothing was cached", which
    is a fabricated measurement."""
    from codeaction.evidence.cost_ledger import EXCLUDES_CACHED, prompt_tokens_convention

    usages = [_usage_of(r) for r in recs]
    prompt_total = _usage_total(recs, "prompt_tokens")
    reported = [u for u in usages if u.get("cached_tokens") is not None]
    coverage = round(len(reported) / len(usages), 4) if usages else None
    cached_total = (sum(int(u["cached_tokens"]) for u in reported) if reported else None)
    creation = [u for u in usages if u.get("cache_creation_tokens") is not None]
    creation_total = (sum(int(u["cache_creation_tokens"]) for u in creation) if creation else None)
    convention = prompt_tokens_convention(model) if model else None
    complete = bool(usages) and len(reported) == len(usages) and convention is not None and prompt_total is not None

    uncached_total = input_total = None
    if complete:
        if convention == EXCLUDES_CACHED:
            uncached_total = prompt_total
            input_total = prompt_total + cached_total + (creation_total or 0)
        else:
            uncached_total = prompt_total - cached_total - (creation_total or 0)
            input_total = prompt_total
        if uncached_total < 0:
            # The record contradicts its own protocol's convention; pricing it would launder a
            # bookkeeping error into a plausible number.
            uncached_total = input_total = None
    return {
        "prompt_total": prompt_total,
        "prompt_tokens_convention": convention,
        "completion_total": _usage_total(recs, "completion_tokens"),
        "cached_prompt_total": cached_total,
        "uncached_prompt_total": uncached_total,
        "input_total": input_total,
        "cache_hit_rate": (round(cached_total / input_total, 4)
                           if input_total else None),
        "cached_coverage": coverage,
        "cache_creation_total": creation_total,
    }


_FUNNEL_BUCKETS = ("achieved_after_entry", "true_at_entry_and_at_end", "true_at_entry_then_lost",
                   "not_achieved", "entry_unknown", "ok_true_unbaselined",
                   "ok_false_unbaselined", "not_evaluable", "skipped")
# The four states that a recorded entry baseline makes attributable. Only these form the
# denominator of a per-milestone rate; the rest are states in which "did the AGENT do it" has no
# answer, and folding them in would manufacture one.
_FUNNEL_ATTRIBUTABLE = ("achieved_after_entry", "true_at_entry_and_at_end",
                        "true_at_entry_then_lost", "not_achieved")


def milestone_funnel_across_attempts(verdicts) -> dict:
    """Pool the end-state milestone funnel over a run group — ONE ROW PER MILESTONE.

    The binary verdict cannot distinguish an attempt that never moved an object from one that
    assembled everything and lost it on the final move. `verifiers.summarize_milestone_funnel`
    answers that per attempt; this is the run-group view of the same partition, and it keeps each
    milestone in its own row for the reason that removed the old `milestones_passed` ratio on
    2026-08-09: a card's milestones are heterogeneous claims (an exact `check_success` conjunct,
    a coarsened proximity, an invented progress rung), so any sum ACROSS milestones is a number
    with no referent. Across ATTEMPTS, one milestone is the same predicate repeated, so a rate is
    meaningful — and that rate is reported only for the attempts whose entry baseline makes
    "the agent produced this" answerable at all.

    `mixed_milestone_sets` marks a group whose attempts declared different milestone names: that
    is a card/identity fault, and the rows below it are not comparable."""
    verdicts = list(verdicts or [])
    order, rows, name_sets = [], {}, []
    baselined_attempts = 0
    with_milestones = 0
    for verdict in verdicts:
        records = (verdict or {}).get("milestones")
        if not isinstance(records, list) or not records:
            continue
        with_milestones += 1
        name_sets.append(tuple(str(rec.get("name")) for rec in records))
        if any("true_at_entry" in rec for rec in records):
            baselined_attempts += 1
        for rec in records:
            name = str(rec.get("name"))
            if name not in rows:
                order.append(name)
                rows[name] = {"name": name, "provenance": rec.get("provenance"),
                              "attempts_reporting": 0,
                              **{bucket: 0 for bucket in _FUNNEL_BUCKETS}}
            row = rows[name]
            row["attempts_reporting"] += 1
            if row["provenance"] is None and rec.get("provenance"):
                row["provenance"] = str(rec["provenance"])
            if (rec.get("detail") or {}).get("skipped"):
                row["skipped"] += 1
                continue
            ok = rec.get("ok")
            if ok is None:
                row["not_evaluable"] += 1
                continue
            if "true_at_entry" not in rec:
                row["ok_true_unbaselined" if ok else "ok_false_unbaselined"] += 1
                continue
            entry = rec["true_at_entry"]
            if entry is None:
                row["entry_unknown"] += 1
            elif entry:
                row["true_at_entry_and_at_end" if ok else "true_at_entry_then_lost"] += 1
            else:
                row["achieved_after_entry" if ok else "not_achieved"] += 1
    for row in rows.values():
        attributable = sum(row[bucket] for bucket in _FUNNEL_ATTRIBUTABLE)
        row["attributable_attempts"] = attributable
        row["achieved_after_entry_rate"] = (
            round(row["achieved_after_entry"] / attributable, 4) if attributable else None)
    ordered = [rows[name] for name in order]
    # Partitioned by provenance, never merged into one list. A `check_success_conjunct` row
    # restates the authoritative predicate the verdict itself reads; a `benchmark_instrument` row
    # is our own probe, declared with its own limits (several are explicitly not monotone toward
    # the goal). Handing a reader one interleaved array invites exactly the cross-kind average
    # that removed `milestones_passed` on 2026-08-09. An unclassified row is a card fault: it
    # cannot say which of the two it is, so it is surfaced separately rather than counted.
    return {
        "attempts": len(verdicts),
        "attempts_with_milestones": with_milestones,
        "entry_baseline_coverage": (round(baselined_attempts / with_milestones, 4)
                                    if with_milestones else None),
        "mixed_milestone_sets": len(set(name_sets)) > 1,
        "authoritative_milestones": [
            row for row in ordered if row.get("provenance") == "check_success_conjunct"],
        "development_instruments": [
            row for row in ordered if row.get("provenance") == "benchmark_instrument"],
        "unclassified_milestones": [
            row for row in ordered
            if row.get("provenance") not in ("check_success_conjunct", "benchmark_instrument")],
    }


def aggregate_attempts(recs, model=None) -> dict:
    """Per-task summary over attempt records [{"stats": {...}, "verifier": {...}}, ...] —
    the shape every episode host writes into result.json. Success/destructive come from the
    out-of-band verifier.

    `model` is what lets the token split be derived at all: it names the protocol whose
    prompt-token convention the usage numbers were recorded under (see `token_billing`)."""
    n = len(recs)
    c = sum(1 for r in recs if r.get("verifier", {}).get("success"))
    destr = sum(1 for r in recs if r.get("verifier", {}).get("destructive"))
    lo, hi = wilson_ci95(c, n)
    resources = [resource_accounting(record) for record in recs]
    return {
        "n_attempts": n,
        "n_success": c,
        "success_rate": round(c / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
        "pass_at_k": {str(k): round(pass_at_k(n, c, k), 4) for k in (1, 2, 5) if k <= n},
        # The reliability counterpart. Reported beside pass@k rather than instead of it: on the
        # fixed-scene attempt protocol they answer different questions ("ever" vs "always").
        "pass_hat_k": {str(k): round(pass_hat_k(n, c, k), 4) for k in (1, 2, 5) if k <= n},
        "n_destructive": destr,
        "destructive_rate": round(destr / n, 4) if n else None,
        "tool_calls_used_mean": _mean(
            [resource["tool_calls_used"] for resource in resources]),
        "tool_call_budget_mean": _mean(
            [resource["tool_call_budget"] for resource in resources]),
        "tool_call_budget_utilization_mean": _mean(
            [resource["tool_call_budget_utilization"] for resource in resources]),
        "total_tool_dispatches_mean": _mean(
            [resource["total_tool_dispatches"] for resource in resources]),
        "model_turns_mean": _mean([resource["model_turns"] for resource in resources]),
        "wall_s_mean": _mean([r.get("stats", {}).get("wall_s") for r in recs]),
        "prompt_tokens_total": _usage_total(recs, "prompt_tokens"),
        "completion_tokens_total":
            _usage_total(recs, "completion_tokens"),
        "tokens": token_billing(recs, model=model),
    }


# Abort reasons produced by the per-physics-step contact monitor (`codeaction/collision_monitor.py`
# via the ToolBox registry guard). Each one is ONE INTERRUPTED COMMAND — the monitor stops at the
# first physics step whose solver produced a contact response, so a count is not a claim about
# severity, persistence, stuck state, or task failure.
_CONTACT_ABORT_COUNTERS = {
    "unexpected_contact": "unexpected_contact_aborts",
    "contact_read_unavailable": "contact_read_unavailable_aborts",
}
_CODE_TOOLS = ("run_code", "run_program")


def _code_block_contact_reason(result):
    """The contact reason that stopped one sandbox block, or None.

    At most one per block: the sandbox kills the child at the first contact abort, so a block
    cannot carry two. Reading `interrupted_action` rather than `internal_trace` is what keeps the
    count right — the interrupted call appears in BOTH (`codeaction/sandbox.py`).
    """
    action = result.get("interrupted_action")
    if isinstance(action, dict):
        reason = str(action.get("abort_reason"))
        return reason if reason in _CONTACT_ABORT_COUNTERS else None
    # Fixture/custom registries can raise past the ToolBox conversion: the block still stops, but
    # with no ActionResult to read. The trace row is then the only surviving evidence.
    for row in result.get("internal_trace") or []:
        if isinstance(row, dict) and row.get("error") == "UnintendedCollision":
            return "unexpected_contact"
    return None


def contact_safety_stats(tool_records) -> dict:
    """Count contact-monitor interruptions in ONE attempt's transcript.

    This is the safety axis the 2026-08-07 Reflection asked for, deliberately kept OUT of the
    success rate: an attempt that reached the goal after repeatedly tripping the monitor and one
    that reached it cleanly are the same success and different numbers here.

    Unknown contact state (`contact_read_unavailable`) is counted separately and never pooled with
    an observed contact — a sensor that could not be read is not evidence of a contact, and not
    evidence of its absence either. ANALYSIS ONLY: never a gate, never agent-visible.
    """
    out = {name: 0 for name in _CONTACT_ABORT_COUNTERS.values()}
    # Provenance for the observed-contact count only. A block stopped by an UNREADABLE sensor is
    # counted in its own total and deliberately not here: it is not a subset of observed contact.
    out["run_code_unexpected_contact_interruptions"] = 0
    out["legacy_collision_terminals"] = 0
    for record in tool_records or []:
        if not isinstance(record, dict):
            continue
        # Transcript 2.4-2.6 ended the attempt on contact with its own terminal record. Tool surface 9.0 no
        # longer emits one; historical runs still count, under their own name, because a terminal
        # and a recoverable interruption are different facts about different tested units.
        if record.get("event") == "unintended_collision":
            out["legacy_collision_terminals"] += 1
            continue
        if record.get("event") not in (None, "tool"):
            continue
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        if record.get("tool") in _CODE_TOOLS:
            reason = _code_block_contact_reason(result)
            if reason is not None:
                out[_CONTACT_ABORT_COUNTERS[reason]] += 1
                if reason == "unexpected_contact":
                    out["run_code_unexpected_contact_interruptions"] += 1
            continue
        counter = _CONTACT_ABORT_COUNTERS.get(str(result.get("abort_reason")))
        if counter and result.get("status") == "ABORTED":
            out[counter] += 1
    return out


_REACHABILITY_TOOLS = ("check_tcp_pose_reachability", "check_direction_feasibility")


def reachability_probe_telemetry(tool_records) -> dict:
    """How much the agent interrogated the planner, and over what vertical span.

    Why this is counted at all: the planner's world model contains one obstacle, the table cuboid
    (`envs/robot/planner.py`), and no task objects. A refusal therefore cannot leak an object's
    position — but it does encode the table plane, which D0 nominally requires the agent to obtain
    by touching. Since these two tools execute nothing, that plane can be bracketed by querying
    heights until the answer flips, at no motion cost.

    The channel is dominated (descend-to-contact reads the plane directly off forward kinematics,
    while a planner sweep returns a whole-arm collision boundary offset by gripper geometry and
    confounded by IK and joint limits), so it is declared rather than closed. This makes it
    observable instead of merely known: a run that bracketed the table this way shows up as many
    queries across a wide z span.

    ANALYSIS ONLY -- never a gate, never agent-visible (§0.1 corollary 7). Pure: takes decoded
    transcript `tool` records, touches no simulator.
    """
    zs = []
    calls = 0
    per_tool = {name: 0 for name in _REACHABILITY_TOOLS}
    def observe(record):
        nonlocal calls
        if not isinstance(record, dict):
            return
        name = record.get("tool") or record.get("name")
        if name not in _REACHABILITY_TOOLS:
            return
        calls += 1
        per_tool[name] += 1
        arguments = record.get("args") or record.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        target = arguments.get("target_xyz")
        if isinstance(target, (list, tuple)) and len(target) == 3:
            try:
                zs.append(float(target[2]))
            except (TypeError, ValueError):
                pass
        result = record.get("result")
        if name == "check_direction_feasibility" and isinstance(result, dict):
            targets = [
                item.get("target_xyz") for item in (result.get("results") or [])
                if isinstance(item, dict)]
            for result_target in targets:
                if isinstance(result_target, (list, tuple)) and len(result_target) == 3:
                    try:
                        zs.append(float(result_target[2]))
                    except (TypeError, ValueError):
                        pass

    for record in tool_records or []:
        observe(record)
        if not isinstance(record, dict) or record.get("tool") not in ("run_code", "run_program"):
            continue
        result = record.get("result")
        trace = result.get("internal_trace") if isinstance(result, dict) else None
        for nested in trace or []:
            observe(nested)
    return {
        "reachability_queries": calls,
        "reachability_queries_by_tool": per_tool,
        "reachability_query_z_values": len(zs),
        "reachability_query_z_spread_m": (
            round(max(zs) - min(zs), 6) if len(zs) >= 2 else None),
    }


_CONTACT_PARTS = ("fingertip", "arm_link", "robot_self")


def _contact_pose_results(tool_records):
    """Yield physical-action results once, including actions nested in code execution.

    ``interrupted_action`` repeats the same action already present in ``internal_trace`` and is
    deliberately ignored. Counting both would make the telemetry depend on whether the model used
    direct dispatch or ``run_code`` for the same physical observation.
    """
    for record in tool_records or ():
        if not isinstance(record, dict) or record.get("event") not in (None, "tool"):
            continue
        name = record.get("tool") or record.get("name")
        result = record.get("result")
        if name in PHYSICS_ACTION_TOOLS and isinstance(result, dict):
            yield result
        if name not in ("run_code", "run_program") or not isinstance(result, dict):
            continue
        for nested in result.get("internal_trace") or ():
            if not isinstance(nested, dict):
                continue
            nested_name = nested.get("tool") or nested.get("name")
            nested_result = nested.get("result")
            if nested_name in PHYSICS_ACTION_TOOLS:
                yield nested_result if isinstance(nested_result, dict) else nested


def contact_pose_telemetry(tool_records) -> dict:
    """Count metric contact-pose observations exposed by physical action results.

    A usable observation requires both a non-empty own-part report and the same-boundary robot
    snapshot. The contacted entity remains unavailable. This is an analysis-only declaration of
    the tactile channel: it never scores, gates, or feeds a result back to the agent.
    """
    observations = 0
    by_part = {part: 0 for part in _CONTACT_PARTS}
    positions = {axis: [] for axis in range(3)}
    for result in _contact_pose_results(tool_records):
        achieved = result.get("achieved")
        if not isinstance(achieved, dict):
            continue
        parts = achieved.get("contact_parts")
        pose = achieved.get("pose_at_contact")
        if not isinstance(parts, list) or not parts or not isinstance(pose, dict):
            continue
        arms_state = pose.get("arms")
        if not isinstance(arms_state, dict):
            continue
        contacted_arms = set()
        observation_parts = {part: 0 for part in _CONTACT_PARTS}
        for item in parts:
            if not isinstance(item, dict):
                continue
            part = item.get("part")
            if part in observation_parts:
                observation_parts[part] += 1
            for arm in item.get("arms") or ():
                if arm in ("left", "right"):
                    contacted_arms.add(arm)
        observation_xyz = []
        for arm in contacted_arms:
            arm_state = arms_state.get(arm)
            tcp = arm_state.get("tcp_pose") if isinstance(arm_state, dict) else None
            if not isinstance(tcp, (list, tuple)) or len(tcp) < 3:
                continue
            try:
                xyz = [float(tcp[index]) for index in range(3)]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in xyz):
                continue
            observation_xyz.append(xyz)
        if not observation_xyz:
            continue
        observations += 1
        for part, count in observation_parts.items():
            by_part[part] += count
        for xyz in observation_xyz:
            for axis, value in enumerate(xyz):
                positions[axis].append(value)
    names = ("x", "y", "z")
    return {
        "contact_pose_observations": observations,
        "contact_pose_observations_by_part": by_part,
        "contact_pose_axis_spread_m": {
            names[axis]: (round(max(values) - min(values), 6)
                          if len(values) >= 2 else None)
            for axis, values in positions.items()
        },
    }
