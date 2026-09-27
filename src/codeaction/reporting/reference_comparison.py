"""Explicit, offline comparison against a selected reference collection.

Selectors identify recorded groups. They never assert that two configurations match.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import statistics

from codeaction.contracts.identity import comparison_key
from codeaction.reporting.comparison import NUMERIC_FIELDS, load_input, summarize


SELECTOR_FIELDS = ("source", "task", "agent", "comparison_key")
CONDITION_FIELDS = ("task", "task_pack", "environment", "tool_surface", "instruction_surface",
                    "verifier", "budgets", "randomness_protocol")


def _key(selector: dict) -> tuple:
    if not isinstance(selector, dict) or set(selector) != set(SELECTOR_FIELDS):
        raise ValueError(f"each selector must contain exactly {', '.join(SELECTOR_FIELDS)}")
    if any(not isinstance(selector[k], str) or not selector[k] for k in SELECTOR_FIELDS[:-1]):
        raise ValueError("selector source, task and agent must be nonempty strings")
    if selector["comparison_key"] is not None and not isinstance(selector["comparison_key"], str):
        raise ValueError("selector comparison_key must be a recorded key or null for unknown identity")
    return tuple(selector[k] for k in SELECTOR_FIELDS)


def collect(paths: list[Path], role: str, pricing: dict) -> dict:
    """Retain source boundaries and full configurations; reject duplicate trial selections."""
    sources, groups, seen = [], [], set()
    for index, path in enumerate(paths, 1):
        rows, source = load_input(path)
        source_id = f"{role}-{index}"
        observed = {(row["task"], row["agent"], row["attempt"]) for row in rows}
        requested = source.get("requested_cells")
        coverage = []
        for task, agent in sorted({(row["task"], row["agent"]) for row in rows + (requested or [])}):
            planned = [row["attempt"] for row in requested if row["task"] == task and row["agent"] == agent] \
                if requested is not None else None
            accepted = sorted(n for t, a, n in observed if t == task and a == agent)
            coverage.append({"task": task, "agent": agent, "selected_attempts": accepted,
                             "requested_attempts": planned,
                             "requested_without_accepted": sorted(set(planned) - set(accepted))
                             if planned is not None else None})
        sources.append({"source": source_id, **source, "task_model_coverage": coverage})
        selected = defaultdict(list)
        for row in rows:
            identity = row.get("_comparison_identity")
            digest = comparison_key({"comparison": identity}) if identity else None
            key = (row["task"], row["agent"], digest)
            trial = (*key, row["attempt"])
            if trial in seen:
                raise ValueError(f"duplicate task/agent/attempt/configuration in {role} selection")
            seen.add(trial)
            selected[key].append(row)
        for (task, agent, digest), subset in sorted(selected.items(), key=lambda item: str(item[0])):
            identity = subset[0].get("_comparison_identity")
            summary = summarize(subset, pricing)["groups"][0]
            declared = (identity or {}).get("randomness_protocol", {}).get("attempts_k")
            declared = declared if type(declared) is int and declared > 0 else None
            groups.append({
                "selector": dict(zip(SELECTOR_FIELDS, (source_id, task, agent, digest))),
                "identity_status": "recorded" if identity else "unknown",
                "identity": identity,
                "protocol_attempts": declared,
                "protocol_coverage": len(subset) / declared if declared else None,
                "summary": summary["agents"][agent],
                "attempts": summary["attempts"],
                "trials": {str(row["attempt"]): row.get("_trial_identity") for row in subset},
            })
    return {"sources": sources, "groups": groups}


def inventory(candidate: dict, reference: dict) -> dict:
    def public_side(side):
        return {"sources": side["sources"], "groups": [
            {k: v for k, v in group.items() if k not in ("attempts", "trials")}
            for group in side["groups"]]}
    return {"schema": "codeaction-comparison-inventory.v1",
            "candidate": public_side(candidate), "reference": public_side(reference)}


def _identity_status(candidate: dict | None, reference: dict | None, fields: tuple) -> dict:
    unknown = [field for field in fields if not (candidate or {}).get(field)
               or not (reference or {}).get(field)]
    if "model" in fields and any((side or {}).get("model", {}).get("evidence") == "unavailable"
                                 for side in (candidate, reference)) and "model" not in unknown:
        unknown.append("model")
    different = [field for field in fields if field not in unknown and
                 candidate[field] != reference[field]]
    return {"status": "mismatched" if different else "unknown" if unknown else "matched",
            "different_fields": different, "unknown_fields": unknown}


def _rate(rows: list[dict]) -> float | None:
    return sum(row["success"] for row in rows) / len(rows) if rows else None


def _delta(candidate: float | None, reference: float | None) -> float | None:
    return candidate - reference if candidate is not None and reference is not None else None


def compare(candidate: dict, reference: dict, mapping: dict) -> dict:
    if not isinstance(mapping, dict) or set(mapping) != {"schema", "pairs"} or \
            mapping["schema"] != "codeaction-reference-matches.v1" or \
            not isinstance(mapping["pairs"], list) or not mapping["pairs"]:
        raise ValueError("match map requires schema codeaction-reference-matches.v1 and nonempty pairs")
    indexes = [{_key(group["selector"]): group for group in side["groups"]}
               for side in (candidate, reference)]
    used = [set(), set()]
    comparisons = []
    for pair in mapping["pairs"]:
        if not isinstance(pair, dict) or set(pair) != {"candidate", "reference"}:
            raise ValueError("each match must contain exactly candidate and reference selectors")
        selected = []
        for i, role in enumerate(("candidate", "reference")):
            key = _key(pair[role])
            if key not in indexes[i]:
                raise ValueError(f"{role} selector not found; regenerate the inventory: {pair[role]}")
            if key in used[i]:
                raise ValueError(f"{role} group is used more than once in the match map")
            used[i].add(key)
            selected.append(indexes[i][key])
        left, right = selected
        fields = tuple(sorted(set(CONDITION_FIELDS) | (
            (left["identity"] or {}).keys() | (right["identity"] or {}).keys()) - {"model", "tested_unit"}))
        conditions = _identity_status(left["identity"], right["identity"], fields)
        model = _identity_status(left["identity"], right["identity"], ("model",))
        configuration = _identity_status(left["identity"], right["identity"], ("tested_unit",))
        statuses = [s["status"] for s in (conditions, model, configuration)]
        a, b = ({row["attempt"]: row for row in group["attempts"]} for group in selected)
        common = sorted(a.keys() & b.keys())
        unknown_trials = [n for n in common if not left["trials"].get(str(n)) or not right["trials"].get(str(n))]
        different_trials = [n for n in common if n not in unknown_trials and
                            left["trials"][str(n)] != right["trials"][str(n)]]
        trials = {"status": "mismatched" if different_trials else
                  "unknown" if unknown_trials or not common else "matched",
                  "different_attempts": different_trials, "unknown_attempts": unknown_trials}
        verified = [n for n in common if all(s == "matched" for s in statuses) and
                    n not in unknown_trials and n not in different_trials]
        statuses.append(trials["status"])
        status = "mismatched" if "mismatched" in statuses else "unknown" if "unknown" in statuses else "matched"
        metrics = {}
        for field in NUMERIC_FIELDS:
            available = [n for n in common if a[n].get(field) is not None and b[n].get(field) is not None]
            ca = statistics.fmean(a[n][field] for n in available) if available else None
            rb = statistics.fmean(b[n][field] for n in available) if available else None
            metrics[field] = {"paired_available": len(available), "common_attempts": len(common),
                              "candidate_mean": ca, "reference_mean": rb, "observed_delta": _delta(ca, rb)}
        comparisons.append({
            **pair, "identity_status": status, "conditions": conditions, "model": model,
            "configuration": configuration, "trials": trials,
            "candidate_attempts": len(a), "reference_attempts": len(b),
            "common_attempt_numbers": common,
            "candidate_only_attempts": sorted(a.keys() - b.keys()),
            "reference_only_attempts": sorted(b.keys() - a.keys()),
            "candidate_success_rate": _rate(list(a.values())),
            "reference_success_rate": _rate(list(b.values())),
            "observed_success_rate_delta": _delta(_rate(list(a.values())), _rate(list(b.values()))),
            "common_attempt_success_rate_delta": _delta(_rate([a[n] for n in common]), _rate([b[n] for n in common])),
            "verified_paired_attempts": len(verified),
            "verified_success_rate_delta": _delta(_rate([a[n] for n in verified]), _rate([b[n] for n in verified])),
            "metrics": metrics,
        })
    return {**inventory(candidate, reference), "schema": "codeaction-reference-comparison.v1",
            "comparisons": comparisons,
            "unmapped": {role: [group["selector"] for key, group in indexes[i].items() if key not in used[i]]
                         for i, role in enumerate(("candidate", "reference"))}}


def write_reference_report(document: dict, path: Path) -> None:
    def value(v):
        return "unavailable" if v is None else str(v)

    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + \
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(
                f"<td>{escape(value(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"

    def selector(s):
        return f"{s['source']} / {s['task']} / {s['agent']} / {s['comparison_key'] or 'unknown configuration'}"

    def pp(v):
        return None if v is None else f"{100 * v:+.2f} pp"

    sections = ["<h2>Comparison summary</h2>" + table(
        ["Candidate task / agent", "Reference task / agent", "Identity", "Candidate / reference attempts",
         "Observed success delta", "Verified-pair success delta"], [
            [f"{p['candidate']['task']} / {p['candidate']['agent']}",
             f"{p['reference']['task']} / {p['reference']['agent']}", p["identity_status"],
             f"{p['candidate_attempts']} / {p['reference_attempts']}",
             pp(p["observed_success_rate_delta"]), pp(p["verified_success_rate_delta"])]
            for p in document["comparisons"]])]
    for role in ("candidate", "reference"):
        side = document[role]
        sections.append(f"<h2>{role.title()} coverage</h2>" + table(
            ["Source", "Evidence", "Selected attempts", "Requested attempts", "Status"],
            [[s["source"], s["kind"], s["selected"], s.get("requested"), s.get("status")]
             for s in side["sources"]]))
        sections.append(table(["Source", "Task", "Agent/model label", "Selected attempt numbers",
                               "Requested attempt numbers", "Requested without accepted execution"], [
            [s["source"], c["task"], c["agent"], c["selected_attempts"], c["requested_attempts"],
             c["requested_without_accepted"]] for s in side["sources"] for c in s["task_model_coverage"]]))
        sections.append(table(["Task / agent / configuration", "Recorded model", "Identity", "Success / selected",
                               "Protocol attempts", "Protocol coverage", "Mapped"], [
            [selector(g["selector"]), (g["identity"] or {}).get("model", {}).get("id"), g["identity_status"],
             f"{g['summary']['successes']} / {g['summary']['attempts']}", g["protocol_attempts"],
             None if g["protocol_coverage"] is None else f"{100 * g['protocol_coverage']:.1f}%",
             "no" if g["selector"] in document["unmapped"][role] else "yes"] for g in side["groups"]]))
    for pair in document["comparisons"]:
        sections.append("<details><summary>" + escape(
            f"{pair['candidate']['task']} / {pair['candidate']['agent']}: {pair['identity_status']}") +
                        "</summary><p>Candidate: " + escape(selector(pair["candidate"])) +
                        "<br>Reference: " + escape(selector(pair["reference"])) + "</p>" +
                        f"<p><strong>Identity: {escape(pair['identity_status'])}</strong></p>")
        sections.append(table(["Identity component", "Status", "Different fields", "Unknown fields"], [
            [name, pair[name]["status"], ", ".join(pair[name]["different_fields"]),
             ", ".join(pair[name]["unknown_fields"])] for name in ("conditions", "model", "configuration")]))
        sections.append(table(["Candidate attempts", "Reference attempts", "Common attempt numbers",
                               "Candidate only", "Reference only", "Verified pairs"], [[
            pair[k] for k in ("candidate_attempts", "reference_attempts", "common_attempt_numbers",
                             "candidate_only_attempts", "reference_only_attempts", "verified_paired_attempts")]]))
        sections.append(table(["Trial identity", "Different attempt identities", "Unknown attempt identities"], [[
            pair["trials"]["status"], pair["trials"]["different_attempts"], pair["trials"]["unknown_attempts"]]]))
        sections.append(table(["Candidate success rate", "Reference success rate", "Observed delta",
                               "Common-attempt observed delta", "Verified-pair delta"], [[
            f"{100 * pair['candidate_success_rate']:.2f}%", f"{100 * pair['reference_success_rate']:.2f}%",
            *[pp(pair[k]) for k in ("observed_success_rate_delta", "common_attempt_success_rate_delta",
                                   "verified_success_rate_delta")]]]))
        sections.append(table(["Metric", "Available pairs / common attempts", "Candidate mean",
                               "Reference mean", "Observed delta"], [
            [name, f"{m['paired_available']} / {m['common_attempts']}", m["candidate_mean"],
             m["reference_mean"], m["observed_delta"]] for name, m in pair["metrics"].items()]))
        sections.append("</details>")
    sections.append("<details><summary>Recorded identities, coverage and measurement basis</summary><pre>" +
                    escape(json.dumps(document, ensure_ascii=False, indent=2)) + "</pre></details>")
    payload = json.dumps(document, ensure_ascii=False).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>CodeActionBench reference comparison</title>
<style>body{font:15px system-ui;margin:32px;color:#182337}table{border-collapse:collapse;width:100%;margin:18px 0 32px}th,td{padding:9px;border-bottom:1px solid #dae0e8;text-align:left;overflow-wrap:anywhere}th{background:#edf2f7}p{max-width:1000px;line-height:1.6}pre{white-space:pre-wrap}details{padding:12px;border:1px solid #dae0e8}</style>
<h1>Reference result comparison</h1>
<p>All deltas are candidate minus reference. Explicit matches select groups; they do not establish
equivalent identities. Observed deltas are descriptive, including when identity is unknown or mismatched.
Verified-pair deltas require equal recorded comparison and trial identities. Common attempt numbers
alone do not prove equal scenes or configurations. No significance test is performed.</p>
<p>Only selected accepted executions contribute to batch results. Supplied projections retain unknown
execution identities and unknown requested/protocol coverage. Missing attempts are not failures;
missing measurements remain unavailable. Metric deltas use only common attempt numbers with measurements
on both sides. Token estimates and API-equivalent costs retain their measurement basis below.</p>
<button id="download">Download reference comparison (JSON)</button>
""" + "".join(sections) + '<script type="application/json" id="data">' + payload + """</script>
<script>document.getElementById('download').onclick=()=>{const b=new Blob([JSON.stringify(JSON.parse(document.getElementById('data').textContent),null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='reference-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};</script></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def run_reference(args: argparse.Namespace) -> int:
    if not args.reference:
        raise ValueError("reference comparison requires explicit --reference input(s)")
    if bool(args.out) != bool(args.match_map):
        raise ValueError("a reference report requires both --out and --match-map")
    if not args.out and not args.inventory_out:
        raise ValueError("choose --inventory-out, or --match-map with --out")
    candidate_paths = [p.resolve() for p in args.inputs]
    reference_paths = [p.resolve() for p in args.reference]
    if len(set(candidate_paths)) != len(candidate_paths) or len(set(reference_paths)) != len(reference_paths):
        raise ValueError("duplicate source path in candidate or reference selection")
    if set(candidate_paths) & set(reference_paths):
        raise ValueError("candidate and reference sources must be separate selections")
    outputs = [p.resolve() for p in (args.out, args.inventory_out) if p is not None]
    roots = [p if p.is_dir() else p.parent for p in candidate_paths + reference_paths]
    if len(outputs) != len(set(outputs)):
        raise ValueError("report and inventory outputs must differ")
    for output in outputs:
        if any(output == root or root in output.parents for root in roots) or \
                (args.match_map and output == args.match_map.resolve()) or output == args.pricing.resolve():
            raise ValueError("comparison outputs must not overwrite or sit inside input evidence")
    pricing = json.loads(args.pricing.read_text())
    candidate = collect(args.inputs, "candidate", pricing)
    reference = collect(args.reference, "reference", pricing)
    document = compare(candidate, reference, json.loads(args.match_map.read_text())) if args.match_map else None
    if args.inventory_out:
        args.inventory_out.parent.mkdir(parents=True, exist_ok=True)
        args.inventory_out.write_text(json.dumps(inventory(candidate, reference), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote comparison inventory: {args.inventory_out}")
    if document is not None:
        write_reference_report(document, args.out)
        print(f"Wrote {args.out}: {len(document['comparisons'])} explicit reference matches")
    return 0
