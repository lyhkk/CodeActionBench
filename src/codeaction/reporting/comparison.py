"""One offline comparison report from a supplied collection or validated batch selections.

Run: python -m codeaction.reporting.comparison INPUT [INPUT ...] --out comparison.html
No Docker, network, model calls, or inferred success labels are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from html import escape
from itertools import combinations
import json
import math
from pathlib import Path
import re
import statistics


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_creation_tokens")
NUMERIC_FIELDS = ("wall_s", "physical_s", "charged_tool_calls", "run_code_calls",
                  "internal_tool_calls", "provider_s", "input_tokens_total", *TOKEN_FIELDS, "api_equivalent_cost_usd")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def complete_sum(values):
    values = list(values)
    return sum(values) if values and all(number(v) is not None for v in values) else None


def _request(raw: dict, *, vendor: bool = False) -> dict:
    names = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens",
             "cached_tokens": "cache_read_input_tokens", "cache_creation_tokens": "cache_creation_input_tokens"}
    result = {name: number(raw.get(names[name] if vendor else name)) for name in TOKEN_FIELDS}
    split = raw.get("cache_creation") or {}
    result["cache_creation_5m_tokens"] = number(split.get("ephemeral_5m_input_tokens"))
    result["cache_creation_1h_tokens"] = number(split.get("ephemeral_1h_input_tokens"))
    return result


def usage_evidence(directory: Path) -> dict:
    """Use reference requests, vendor final totals, or explicitly estimated Claude output."""
    reference = read_jsonl(directory / "reference_transcript.jsonl")
    turns = [r for r in reference if r.get("event") == "model_turn"]
    if turns:
        return {"source": "provider_requests", "output_kind": "reported", "per_request": True,
                "requests": [_request(r.get("usage") or {}) for r in turns],
                "provider_s": complete_sum(number(r.get("model_latency_s")) for r in turns)}
    vendor_file = directory / "vendor_usage.json"
    if vendor_file.is_file():
        raw = json.loads(vendor_file.read_text())
        usage = raw.get("usage") or {}
        if number(usage.get("input_tokens")) is not None and number(usage.get("output_tokens")) is not None:
            return {"source": "vendor_episode_total", "output_kind": "reported", "per_request": False,
                    "requests": [_request(usage, vendor=True)], "provider_s": None}
    rows = read_jsonl(directory / "vendor_transcript.jsonl")
    groups, current, pending, thinking = [], [], 0, 0
    for row in rows:
        event = row.get("event")
        if event == "thinking_tokens":
            if current:
                groups.append((current, thinking))
                current, thinking = [], 0
            pending += number(row.get("estimated_tokens_delta")) or 0
        elif event == "assistant":
            if not current:
                thinking, pending = pending, 0
            current.append(row)
        elif current:
            groups.append((current, thinking))
            current, thinking = [], 0
    if current:
        groups.append((current, thinking))
    requests = []
    for group, thinking in groups:
        raw = group[0].get("usage") or {}
        if number(raw.get("input_tokens")) is None:
            return {"source": "unavailable", "output_kind": "unavailable", "requests": [],
                    "per_request": False, "provider_s": None}
        request = _request(raw, vendor=True)
        blocks = [b for row in group for b in row.get("content", [])
                  if isinstance(b, dict) and b.get("type") != "thinking"]
        encoded = json.dumps(blocks, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        request["reported_partial_output_tokens"] = request["completion_tokens"]
        request["completion_tokens"] = thinking + math.ceil(len(encoded) / 4)
        requests.append(request)
    return {"source": "deduplicated_vendor_stream" if requests else "unavailable",
            "output_kind": "estimated" if requests else "unavailable", "per_request": True,
            "requests": requests, "provider_s": None}


def extract_attempt(directory: Path, *, task: str, agent: str, attempt: int,
                    success: bool, result: dict | None = None) -> dict:
    """Export numeric evidence only; prompts, credentials, paths and source IDs stay out."""
    result = result if result is not None else json.loads((directory / "result.json").read_text())
    if type(success) is not bool:
        raise ValueError("each selected attempt must have a boolean verifier verdict")
    stats = result.get("stats") or {}
    tool_rows = [r for r in read_jsonl(directory / "transcript.jsonl") if r.get("event") == "tool"]
    tools = Counter("run_code" if r["tool"] == "run_program" else r["tool"]
                    for r in tool_rows if isinstance(r.get("tool"), str) and r["tool"] != "write_file")
    internal = [i for r in tool_rows for i in (r.get("result") or {}).get("internal_trace", [])
                if isinstance(i, dict)]
    charged = number(stats.get("tool_calls_used"))
    if charged is None:
        charged = number(stats.get("budget_used"))
    status = stats.get("status")
    if not isinstance(status, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", status):
        status = "unavailable"
    return {"task": task, "agent": agent, "attempt": attempt, "success": success,
            "termination": status, "wall_s": number(stats.get("wall_s")),
            "physical_s": number((result.get("step_observer") or {}).get("physical_time_s")),
            "charged_tool_calls": charged,
            "run_code_calls": tools.get("run_code", 0) if tool_rows else None,
            "internal_tool_calls": len(internal) if tool_rows else None,
            "tool_counts": dict(sorted(tools.items())), "usage": usage_evidence(directory)}


def price_usage(agent: str, evidence: dict, pricing: dict) -> tuple[float | None, str]:
    model = {"codex-astra": "gpt-6-astra", "claude-code-opus-5": "claude-opus-5"}.get(agent, agent)
    rates = pricing.get("models", {}).get(model)
    requests = evidence.get("requests") or []
    if not rates or not requests or rates.get("price_available") is False:
        return None, "unavailable"
    if rates.get("long_context") and not evidence.get("per_request"):
        return None, "per-request usage required for context tiers"
    costs, assumed_cache = [], False
    for request in requests:
        prompt, output, cached = (request.get(f) for f in TOKEN_FIELDS[:3])
        if any(number(x) is None for x in (prompt, output)):
            return None, "incomplete usage"
        if cached is None:
            cached, assumed_cache = 0, True
        explicit = rates.get("evaluation_cache_mode") == "explicit"
        written = request.get("cache_creation_tokens")
        # Implicit caching has no separately billed cache write; missing output is never zero.
        if not explicit and written is None:
            written = 0
        if written is None:
            written, assumed_cache = 0, True
        if rates.get("prompt_convention") == "includes_cached":
            uncached, context = prompt - cached - written, prompt
        elif rates.get("prompt_convention") == "excludes_cached":
            uncached, context = prompt, prompt + cached + written
        else:
            return None, "unknown token convention"
        if uncached < 0:
            return None, "inconsistent token counters"
        write5, write1 = request.get("cache_creation_5m_tokens"), request.get("cache_creation_1h_tokens")
        if write5 is None and write1 is None:
            write5, write1 = written, 0
        if number(write5) is None or number(write1) is None or write5 + write1 != written:
            return None, "cache duration split unavailable"
        read_rate = rates.get("explicit_cache_read" if explicit else "implicit_cache_read")
        write5_rate = rates.get("explicit_cache_write_5m") if explicit else rates.get("input")
        write1_rate = rates.get("explicit_cache_write_1h") if explicit else rates.get("input")
        parts = [(uncached, rates.get("input"), "input_multiplier"),
                 (output, rates.get("output"), "output_multiplier"),
                 (cached, read_rate, "cache_multiplier"),
                 (write5, write5_rate, "cache_multiplier"),
                 (write1, write1_rate, "cache_multiplier")]
        tier = rates.get("long_context") or {}
        long = context > tier.get("input_tokens_gt", math.inf)
        if any(count and number(rate) is None for count, rate, _ in parts):
            return None, "price unavailable"
        costs.append(sum(count * (rate or 0) * (tier.get(mult, 1) if long else 1)
                         for count, rate, mult in parts) / 1_000_000)
    kind = "estimated output" if evidence.get("output_kind") == "estimated" else "reported usage"
    return round(sum(costs), 6), "API-equivalent, " + kind + ("; missing cache counters assumed zero for pricing" if assumed_cache else "")


def describe(values) -> dict:
    present = sorted(number(v) for v in values if number(v) is not None)
    def quantile(p):
        if not present:
            return None
        position = (len(present) - 1) * p
        lo, hi = math.floor(position), math.ceil(position)
        return round(present[lo] + (present[hi] - present[lo]) * (position - lo), 6)
    return {"available": len(present), "total": round(sum(present), 6) if present else None,
            "mean": round(statistics.fmean(present), 6) if present else None,
            "median": quantile(.5), "p90": quantile(.9)}


def _condition(result: dict) -> dict:
    comparison = (result.get("identity") or {}).get("comparison") or {}
    environment = comparison.get("environment") or {}
    # Agent implementations/interfaces may differ. Task, scoring and simulator conditions may not.
    components = environment.get("code_components") or {}
    return {"task_pack": comparison.get("task_pack"),
            "tool_set": (comparison.get("tool_surface") or {}).get("tool_set"),
            "environment": {k: environment.get(k) for k in (
                "runtime", "sim_image_digest", "environment_lock_sha256", "embodiment")},
            "code": {k: v for k, v in components.items() if k not in ("agents", "reporting", "documentation", "tests")},
            "legacy_source": environment.get("source_commit") if not components else None}


def load_input(path: Path) -> tuple[list[dict], dict]:
    if (path / "measurements.jsonl").is_file():
        rows = read_jsonl(path / "measurements.jsonl")
        for row in rows:
            if type(row.get("attempt")) is not int or row["attempt"] < 1:
                raise ValueError("invalid trajectory attempt")
            if type(row.get("success")) is not bool:
                raise ValueError("missing verifier verdict")
            if any(not isinstance(row.get(k), str) or
                   not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row[k]) for k in ("task", "agent")):
                raise ValueError("invalid trajectory identity")
            relative = Path(row["task"]) / row["agent"] / f"attempt-{row['attempt']:03d}.json"
            view = json.loads((path / "trajectories" / relative).read_text())
            if any(view.get(k) != row[k] for k in ("task", "agent", "attempt")) or \
                    view.get("verifier", {}).get("success") is not row["success"]:
                raise ValueError("measurement and trajectory verdict differ")
            row["_condition"] = "supplied-evaluation"
            row["_task_condition"] = row["task"]
            # A display projection cannot attest the original execution configuration.
            row["_comparison_identity"] = None
            row["_trial_identity"] = None
        return rows, {"selected": len(rows), "requested": None, "kind": "supplied evaluation collection"}
    from codeaction.batch.results import load_submission_manifest
    manifest, selected = load_submission_manifest(path)
    rows = []
    for record in selected:
        entry = record.entry
        row = extract_attempt(record.attempt_dir, task=entry["task_name"], agent=entry["model"],
                              attempt=entry["attempt_index"] + 1, success=record.result["verifier"]["success"],
                              result=record.result)
        row["_condition"] = json.dumps(_condition(record.result), sort_keys=True)
        comparison = record.result["identity"]["comparison"]
        row["_comparison_identity"] = comparison
        row["_trial_identity"] = record.result["identity"]["trial"]
        row["_task_condition"] = json.dumps({"scene_seed": entry["scene_seed"],
            **{k: comparison.get(k) for k in ("task", "verifier", "budgets")}}, sort_keys=True)
        rows.append(row)
    batch_dir = path if path.is_dir() else path.parent
    state = json.loads((batch_dir / "batch_state.json").read_text())
    requested = [{"task": state["cells"][key]["task"], "agent": state["cells"][key]["model"],
                  "attempt": state["cells"][key]["attempt_index"] + 1} for key in state["requested_cells"]]
    return rows, {"selected": len(rows), "requested": len(state["requested_cells"]),
                  "requested_cells": requested,
                  "kind": "validated accepted executions", "status": state["status"]}


def summarize(rows: list[dict], pricing: dict) -> dict:
    seen = set()
    groups = defaultdict(list)
    for original in rows:
        row = dict(original)
        condition = row.pop("_condition")
        key = (condition, row["task"], row["agent"], row["attempt"])
        if key in seen:
            raise ValueError("duplicate task/agent/attempt under the same evaluation conditions")
        if type(row["success"]) is not bool:
            raise ValueError("missing verifier verdict")
        seen.add(key)
        usage = row.pop("usage")
        row["usage_source"], row["output_kind"] = usage["source"], usage["output_kind"]
        row["provider_s"] = usage.get("provider_s")
        for field in TOKEN_FIELDS:
            row[field] = complete_sum(request.get(field) for request in usage.get("requests", []))
        model = {"codex-astra": "gpt-6-astra", "claude-code-opus-5": "claude-opus-5"}.get(row["agent"], row["agent"])
        convention = pricing.get("models", {}).get(model, {}).get("prompt_convention")
        row["input_tokens_total"] = (row["prompt_tokens"] if convention == "includes_cached" else
            complete_sum(row[f] for f in ("prompt_tokens", "cached_tokens", "cache_creation_tokens"))
            if convention == "excludes_cached" else None)
        row["api_equivalent_cost_usd"], row["cost_basis"] = price_usage(row["agent"], usage, pricing)
        groups[condition].append(row)
    result = []
    for index, group in enumerate(groups.values(), 1):
        agents = sorted({r["agent"] for r in group})
        summary, task_rows, pairs = {}, [], []
        for agent in agents:
            subset = [r for r in group if r["agent"] == agent]
            success = sum(r["success"] for r in subset)
            per_task = defaultdict(list)
            for row in subset:
                per_task[row["task"]].append(row)
            summary[agent] = {"attempts": len(subset), "successes": success,
                "success_rate": success / len(subset), "tasks": len(per_task),
                "tasks_solved": sum(any(r["success"] for r in rs) for rs in per_task.values()),
                "complete_three_attempt_tasks": sum(len(rs) == 3 for rs in per_task.values()),
                "macro_task_success_rate": statistics.fmean(sum(r["success"] for r in rs) / len(rs) for rs in per_task.values()),
                "metrics": {f: describe(r.get(f) for r in subset) for f in NUMERIC_FIELDS},
                "termination": dict(Counter(r["termination"] for r in subset)),
                "output_kind": dict(Counter(r["output_kind"] for r in subset)),
                "cost_basis": dict(Counter(r["cost_basis"] for r in subset)),
                "tool_counts": dict(sum((Counter(r["tool_counts"]) for r in subset), Counter()))}
            for task, rs in sorted(per_task.items()):
                task_rows.append({"task": task, "agent": agent, "attempts": len(rs),
                                  "successes": sum(r["success"] for r in rs)})
        for left, right in combinations(agents, 2):
            selected = {a: {(r["task"], r["attempt"], r["_task_condition"]): r["success"]
                           for r in group if r["agent"] == a} for a in (left, right)}
            keys = selected[left].keys() & selected[right].keys()
            counts = Counter((selected[left][k], selected[right][k]) for k in keys)
            pairs.append({"left": left, "right": right, "matched_attempts": len(keys),
                          "both_success": counts[True, True], "left_only": counts[True, False],
                          "right_only": counts[False, True], "both_fail": counts[False, False],
                          "success_rate_difference": (counts[True, False] - counts[False, True]) / len(keys) if keys else None})
        result.append({"group": f"evaluation-{index}", "agents": summary, "by_task": task_rows,
                       "paired_comparisons": pairs,
                       "attempts": [{k: v for k, v in r.items() if not k.startswith("_")} for r in group]})
    return {"groups": result, "pricing": pricing}


def write_report(document: dict, path: Path) -> None:
    def table(headers, rows):
        return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{escape(h)}</th>" for h in headers) + \
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"
    def value(v):
        return "unavailable" if v is None else f"{v:,.2f}"
    sections = []
    for group in document["groups"]:
        summary = dict(sorted(group["agents"].items(), key=lambda item: (-item[1]["success_rate"], item[0])))
        sections.append(f"<h2>{escape(group['group'])}</h2>" + table(
            ["Agent", "Success / attempts", "Success rate", "Tasks solved", "Complete 3-attempt tasks"],
            [[a, f"{s['successes']} / {s['attempts']}", f"{100*s['success_rate']:.1f}%", f"{s['tasks_solved']} / {s['tasks']}", s['complete_three_attempt_tasks']] for a, s in summary.items()]))
        sections.append("<h3>Resources and usage</h3>" + table(
            ["Agent", "Wall time median (s)", "Physical time median (s)", "Charged calls median", "Total input tokens mean", "Output tokens mean", "API-equivalent USD total"],
            [[a, *[value(s["metrics"][f][stat]) for f, stat in (("wall_s", "median"), ("physical_s", "median"), ("charged_tool_calls", "median"), ("input_tokens_total", "mean"), ("completion_tokens", "mean"), ("api_equivalent_cost_usd", "total"))]] for a, s in summary.items()]))
        sections.append("<details><summary>Measurement coverage and full statistics</summary><p>Coverage is the number of attempts with each measurement. Token and cost totals use only available attempts; they are incomplete if coverage is below the attempt count.</p>" + table(
            ["Agent", "Metric", "Coverage", "Total", "Mean", "Median", "P90"],
            [[a, f, f"{s['metrics'][f]['available']} / {s['attempts']}", *[value(s['metrics'][f][k]) for k in ("total", "mean", "median", "p90")]] for a, s in summary.items() for f in NUMERIC_FIELDS]) + "</details>")
        sections.append("<details><summary>Paired agent comparisons</summary><p>Pairs share the same task and attempt index under matching task/scoring conditions. These are descriptive comparisons, not significance tests.</p>" + table(
            ["Left", "Right", "Pairs", "Both succeed", "Left only", "Right only", "Both fail"],
            [[p[k] for k in ("left", "right", "matched_attempts", "both_success", "left_only", "right_only", "both_fail")] for p in group["paired_comparisons"]]) + "</details>")
        cells = {(r["task"], r["agent"]): f"{r['successes']} / {r['attempts']}" for r in group["by_task"]}
        sections.append("<h3>Per-task results</h3><p>Each cell is successful attempts / selected attempts.</p>" + table(["Task", *summary],
            [[task, *[cells.get((task, a), "unavailable") for a in summary]] for task in sorted({r["task"] for r in group["by_task"]})]))
        sections.append("<details><summary>Termination, tool use and estimate assumptions</summary>" + table(["Agent", "Termination counts", "Tool calls", "Output token basis", "Cost basis"],
            [[a, *[json.dumps(s[k], sort_keys=True) for k in ("termination", "tool_counts", "output_kind", "cost_basis")]] for a, s in summary.items()]) + "</details>")
    payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><title>CodeActionBench results</title>
<style>body{font:15px system-ui;margin:36px auto;max-width:1300px;padding:0 20px;color:#182337}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:900px;margin:18px 0 36px;font-variant-numeric:tabular-nums}td,th{border-bottom:1px solid #dae0e8;padding:9px;text-align:left;min-width:70px;max-width:420px;overflow-wrap:break-word}td:first-child{white-space:nowrap}th{background:#edf2f7}h2{margin-top:48px}button{padding:10px;cursor:pointer}p{max-width:950px;line-height:1.6}details{border:1px solid #dae0e8;border-radius:8px;padding:16px;margin:16px 0;overflow:auto}summary{cursor:pointer;font-weight:600}</style>
<h1>CodeActionBench result comparison</h1>
<p>Verifier verdicts determine success. Only selected accepted executions contribute to new-batch results. Missing diagnostic data stays unavailable. Groups with different evaluation conditions remain separate. Task coverage and repetitions are shown explicitly.</p>
<p>Total input tokens include cache reads and writes for every provider. Native prompt counters remain available separately because Anthropic excludes cache tokens from that field, while the other providers include them. Output counters include reasoning where the provider reports it.</p>
<p>Costs are API-equivalent estimates at the bundled price snapshot, not subscription bills. Missing cache counters are assumed zero only for pricing; the token measurements remain unavailable. Missing input/output usage prevents pricing. Claude stream output, where no final usage is available, is estimated as recorded thinking-token estimates plus ceil(UTF-8 bytes of non-thinking blocks / 4); repeated assistant stream records count as one request. The embedded data records this basis per attempt. Wall time includes provider waits; physical time comes from simulator steps.</p>
<button id="download">Download comparison data (JSON)</button>
""" + "".join(sections) + '<script type="application/json" id="data">' + payload + """</script>
<script>document.getElementById('download').onclick=()=>{const b=new Blob([JSON.stringify(JSON.parse(document.getElementById('data').textContent),null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};</script></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared parser surface for the module CLI and front-door report commands."""
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, help="write an offline HTML report")
    parser.add_argument("--pricing", type=Path, default=Path(__file__).resolve().parents[3] / "configs/analysis-pricing.json")
    parser.add_argument("--reference", type=Path, action="append",
                        help="explicit reference collection or accepted batch; repeat for multiple sources")
    parser.add_argument("--match-map", type=Path, help="explicit candidate/reference selectors in JSON")
    parser.add_argument("--inventory-out", type=Path,
                        help="write available task/agent/configuration selectors before choosing matches")


def run(args: argparse.Namespace) -> int:
    """Generate a report from parsed arguments without launching any evaluation."""
    if args.reference or args.match_map or args.inventory_out:
        from codeaction.reporting.reference_comparison import run_reference
        return run_reference(args)
    if args.out is None:
        raise ValueError("--out is required for a summary report")
    rows, sources = [], []
    for path in args.inputs:
        selected, source = load_input(path)
        rows.extend(selected)
        sources.append(source)
    pricing = json.loads(args.pricing.read_text())
    document = summarize(rows, pricing)
    document["sources"] = sources
    write_report(document, args.out)
    print(f"Wrote {args.out}: {len(rows)} selected attempts, {len(document['groups'])} evaluation groups")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    from codeaction.batch.results import BatchResultsError
    try:
        return run(args)
    except (OSError, ValueError, KeyError, BatchResultsError) as exc:
        parser.exit(1, f"Comparison: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
