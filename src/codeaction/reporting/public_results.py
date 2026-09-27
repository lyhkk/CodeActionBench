"""Export a selected result set as portable turn records and an offline HTML reader."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from codeaction.reporting.turns_view import build_turns_view

ASSETS = Path(__file__).with_name("public_assets")
SCHEMA = "codeaction-public-results.v2"
# Remove machine-specific locations from the public projection; raw evidence remains unchanged.
PRIVATE_PATH = re.compile(r"/(?:Users|home|Robotwin|private|tmp)/[^\s\"'<>`;,\]\)\}]*")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def public_value(value, counts: Counter):
    if isinstance(value, dict):
        return {key: public_value(item, counts) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item, counts) for item in value]
    if isinstance(value, str):
        value, n = PRIVATE_PATH.subn("[local-path]", value)
        counts["local_paths"] += n
        return value
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checked_path(root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"unsafe result path: {relative!r}")
    path = (root / part).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"result path escapes input tree: {relative!r}")
    return path


def page(title: str, data, *, prefix: str, script: str, body: str) -> str:
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><link rel="stylesheet" href="{prefix}assets/report.css">
<body>{body}<script id="data" type="application/json">{encoded}</script>
<script src="{prefix}assets/{script}"></script></body></html>'''


INDEX_BODY = '''<main><header><span class="eyebrow">CODEACTION BENCHMARK · RESULTS</span>
<h1>675 attempts.<br>Every interaction, readable.</h1>
<p class="intro">Browse model decisions, tool calls, observations, and recorded outcomes.</p>
<nav><a href="README.md">Dataset notes</a><a href="summary.csv">Download CSV</a>
<a href="index.json">Dataset JSON</a><a href="validation.json">Validation</a></nav></header>
<section id="metrics" class="metrics"></section>
<p id="provenance-note" class="note"></p><section class="panel">
<div class="filters"><label>Search<input id="search" placeholder="Task or agent"></label>
<label>Agent<select id="agent"><option value="">All agents</option></select></label>
<label>Outcome<select id="outcome"><option value="">All outcomes</option><option value="pass">Pass</option>
<option value="fail">Fail</option><option value="changed">Changed by regrade</option></select></label>
<label>Scoring<select id="scoring"><option value="corrected">Corrected predicates</option>
<option value="original">Original verdict</option></select></label></div>
<div class="table-wrap"><table><thead><tr><th>Task</th><th>Agent</th><th>Attempt</th><th>Outcome</th>
<th>Turns</th><th>Tool calls</th><th>Time</th></tr></thead><tbody id="rows"></tbody></table></div>
<p id="count" class="note"></p></section></main>'''

EPISODE_BODY = '''<main><nav><a href="../../../index.html">← All attempts</a>
<a href="turns.v1.json">Turn JSON</a><a href="episode.json">Episode summary</a>
<a href="source.json">Evidence hashes</a></nav><header><span class="eyebrow" id="label"></span>
<h1 id="title"></h1><p class="intro" id="instruction"></p><div id="badges" class="badges"></div></header>
<section class="episode-layout"><aside><video id="video" controls preload="metadata"></video>
<p class="note">Recorded head-camera review. Turn images appear with the calls that produced them.</p>
<div id="coverage" class="panel"></div></aside><section><div class="filters">
<label>Find in turns<input id="search" placeholder="Tool, text, or observation"></label></div>
<div id="turns"></div></section></section></main>'''


def export_results(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("output must be separate from the sealed input tree")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    manifest_path = source / "MANIFEST.json"
    original_manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("input manifest must contain a nonempty cells list")
    paths = [cell["path"] for cell in cells]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate episode paths in manifest")
    normalized = []
    for cell in cells:
        if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cell[key]) for key in ("task", "agent")):
            raise ValueError("task and agent identifiers must be simple path components")
        run = checked_path(source, cell["path"])
        # Sealed historical manifests keep their paths. New public output uses attempt only.
        key = "attempt" if "attempt" in cell else "repeat"
        if "attempt" in cell and "repeat" in cell:
            raise ValueError("episode declares both current and historical attempt fields")
        attempt = cell.get(key)
        if type(attempt) is not int or attempt < 1:
            raise ValueError("episode attempt must be a positive integer")
        expected = f"{cell['task']}/{cell['agent']}/{key}-{attempt:03d}"
        if cell["path"] != expected or not run.is_dir():
            raise ValueError(f"invalid episode path: {cell['path']!r}")
        if not isinstance(cell.get("success"), bool) or not isinstance(cell.get("success_corrected"), bool):
            raise ValueError(f"both recorded verdicts are required: {cell['path']}")
        normalized.append({**cell, "attempt": attempt, "source_path": cell["path"],
                           "path": f"{cell['task']}/{cell['agent']}/attempt-{attempt:03d}"})
    cells = normalized
    if len({cell["path"] for cell in cells}) != len(cells):
        raise ValueError("duplicate attempts after reading historical paths")
    output.mkdir(parents=True)
    shutil.copytree(ASSETS, output / "assets")
    rows, redactions, issues = [], Counter(), []
    counts = Counter()
    for cell in cells:
        run = checked_path(source, cell["source_path"])
        target = output / cell["path"]
        target.mkdir(parents=True)
        view = build_turns_view(run)
        coverage = view.get("coverage") or {}
        if (coverage.get("join_diverged_at") or coverage.get("call_outcomes_unattached")
                or coverage.get("call_outcomes_without_turn")
                or (view["source"]["kind"] == "vendor" and
                    coverage.get("server_outcomes") != coverage.get("server_outcomes_joined"))):
            issues.append({"path": cell["path"], "coverage": coverage})
        counts[view["source"]["kind"]] += 1
        input_files = list(view["source"]["files"])
        input_files += [name for name in ("result.json", "provenance.json", "run_meta.json",
                                         "artifact_manifest.v1.json") if (run / name).is_file()]
        before = {name: digest(checked_path(run, name)) for name in input_files}
        view = public_value(view, redactions)
        view["public_export"] = {"schema_version": SCHEMA,
                                  "local_paths": "replaced by [local-path]",
                                  "evidence": "source.json"}
        write_json(target / "turns.v1.json", view)
        media = []
        images = {}
        for path in sorted((run / "tools").glob("obs_*.png")):
            rel = path.relative_to(run).as_posix()
            origin = checked_path(run, rel)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, dest)
            media.append(rel)
            match = re.match(r"(obs_\d+(?:_a\d+)?)", path.stem)
            if match:
                images.setdefault(match[1], []).append(rel)
        video = None
        if (run / "review.mp4").is_file():
            shutil.copyfile(checked_path(run, "review.mp4"), target / "review.mp4")
            video = "review.mp4"
            media.append(video)
        turns = view.get("turns")
        row = {key: cell.get(key) for key in (
            "task", "agent", "attempt", "path", "seed", "success", "success_corrected",
            "tool_calls_used", "wall_s", "status", "task_pack_version")}
        row.update(turn_count=len(turns) if turns is not None else None,
                   source_kind=view["source"]["kind"],
                   changed_by_regrade=cell["success"] != cell["success_corrected"],
                   page=f"{cell['path']}/index.html")
        episode = {"schema_version": SCHEMA, **row, "coverage": coverage,
                   "regrade": {"predicate_changed": (cell.get("regrade") or {}).get("predicate_changed"),
                               "flipped": cell["success"] != cell["success_corrected"]},
                   "video": video, "observations": images}
        write_json(target / "episode.json", episode)
        for rel in media:
            before[rel] = digest(run / rel)
            if digest(target / rel) != before[rel]:
                raise ValueError(f"media copy differs: {cell['path']}/{rel}")
        if any(digest(run / rel) != value for rel, value in before.items()):
            raise ValueError(f"source changed during export: {cell['path']}")
        write_json(target / "source.json", {
            "schema_version": SCHEMA, "source_dataset": source.name,
            "episode_path": cell["source_path"], "manifest_sha256": original_manifest_hash,
            "sha256": before, "turns_sha256": digest(target / "turns.v1.json"),
        })
        (target / "index.html").write_text(page(
            f"{cell['task']} · {cell['agent']} · {cell['attempt']}",
            {"episode": episode, "view": view}, prefix="../../../", script="episode.js",
            body=EPISODE_BODY), encoding="utf-8")
        rows.append(row)
        if len(rows) % 75 == 0:
            print(f"exported {len(rows)}/{len(cells)}", flush=True)
    if digest(manifest_path) != original_manifest_hash:
        raise ValueError("source manifest changed during export")
    summary = {
        "schema_version": SCHEMA, "source_dataset": source.name,
        "source_manifest_sha256": original_manifest_hash,
        "episodes": len(rows), "tasks": len({r["task"] for r in rows}),
        "agents": len({r["agent"] for r in rows}),
        "original_successes": sum(r["success"] for r in rows),
        "corrected_successes": sum(r["success_corrected"] for r in rows),
        "changed_verdicts": sum(r["changed_by_regrade"] for r in rows),
        "cells": rows,
    }
    write_json(output / "index.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    body = INDEX_BODY.replace("675 attempts.", f"{len(rows)} attempts.")
    (output / "index.html").write_text(page("CodeAction · Results", summary, prefix="",
                                            script="index.js", body=body), encoding="utf-8")
    validation = {"schema_version": SCHEMA, "episodes_exported": len(rows),
                  "source_kinds": dict(counts), "coverage_issues": issues,
                  "redactions": dict(redactions), "source_hashes_unchanged": True,
                  "media_hashes_verified": True, "complete": not issues}
    write_json(output / "validation.json", validation)
    (output / "README.md").write_text(f'''# CodeAction result collection

Open `index.html` in a browser. The reader works offline and needs no server.

This collection contains {len(rows)} historical attempts across {summary['tasks']} tasks and
{summary['agents']} agents. The index records both the original verdict and the corrected-predicate
verdict; the reader defaults to corrected verdicts. Original successes: {summary['original_successes']}.
Corrected successes: {summary['corrected_successes']}. Changed verdicts: {summary['changed_verdicts']}.

The selection keeps three attempts per task and agent. Their original task-pack versions and
identities are historical; exporting them does not turn them into new runs of the release protocol.

## Files

- `index.html`, `summary.csv`, `index.json`: browsing and aggregate metadata.
- `<task>/<agent>/attempt-NNN/turns.v1.json`: decisions, tool calls, and returned observations.
- `episode.json`: original and corrected verdicts, timing, and transcript coverage.
- `source.json`: relative source locations and SHA-256 hashes tying the view to raw evidence.
- `review.mp4`, `tools/obs_*.png`: recorded media, copied and hash-verified.
- `validation.json`: coverage, source integrity, and redaction counts.

Reference turns correspond to model inferences. Vendor turns correspond to normalized assistant
stream events, which may separate text, thinking, and tool calls; counts across those representations
are not inference-count comparisons. Missing reasoning text is labelled as unavailable; token-only
evidence remains separate. Long strings use the existing turn-view truncation marker.

Machine-specific absolute paths in the public projection are replaced with `[local-path]`.
The sealed input tree is read-only during export. Source transcripts remain the authoritative
record and can be matched using `source.json`. Ground-truth snapshots and credential files are
excluded from this collection.
''', encoding="utf-8")
    if issues:
        raise ValueError(f"{len(issues)} episodes have incomplete transcript coverage; see validation.json")
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_results(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
