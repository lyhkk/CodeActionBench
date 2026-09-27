#!/usr/bin/env python3
"""Refuse to start a Codex episode unless what the model can USE and what it can SEE are ours.

Runs INSIDE the agent container, after the throwaway CODEX_HOME is built and before `codex exec`.
Standard library only: the image carries no codeaction package.

Three fail-closed checks, all against the CLI's OWN report of effective state rather than our
intent:

1. HOME PURITY. The home holds nothing but auth.json and config.toml before Codex is invoked, and
   nothing but Codex's own scratch afterwards. `$CODEX_HOME/AGENTS.md` and `$CODEX_HOME/skills/*`
   are injected into the prompt on 0.154.0 REGARDLESS of project_doc_max_bytes=0 and
   features.skill_search=false (measured with canary files), so the only thing keeping them out
   is that the home is ours -- and this is what proves it.
2. TOOL PURITY. `codex features list` shows no enabled feature outside the allowlist.
   codex_config.toml is a deny list and fails open; this inverts it. The MCP half of the surface
   is attested by the episode server, which is the other side of the same wall.
3. TEXT PURITY. `codex debug prompt-input` renders the model-visible input list. Every developer
   block must match, in order, a preamble the manifest already knows by hash; the environment
   block must match structurally; the final user block must be OUR prompt byte for byte; and
   nothing else may be present. A CLI upgrade that changes the vendor preamble, or any channel
   that adds a sentence we did not write, fails here instead of in a published number.
   The render uses a copy of the home with the MCP table removed: tool schemas are not part of
   this render anyway, and a required-but-unreachable server would abort it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GATE_SCHEMA_VERSION = "1.1"
HOME_BEFORE = {"auth.json", "config.toml"}
# Directories Codex creates for itself when invoked. Anything else appearing is a foreign file.
HOME_SCRATCH = {"tmp", ".tmp", "log", "logs"}
_TAG = re.compile(r"^<([a-z_]+)")


def _load_allowlist(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def _run(argv, env, cwd=None, timeout=120):
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=timeout)


def _features(codex_bin: str, env: dict) -> tuple[dict[str, dict], list[str]]:
    completed = _run([codex_bin, "features", "list"], env)
    if completed.returncode != 0:
        return {}, [f"codex features list exited {completed.returncode}: "
                    f"{completed.stderr.strip()[:300]}"]
    rows, problems = {}, []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[-1] not in ("true", "false"):
            if line.strip():
                problems.append(f"unparseable features row: {line.strip()[:120]}")
            continue
        rows[parts[0]] = {"stage": " ".join(parts[1:-1]), "enabled": parts[-1] == "true"}
    if not rows:
        problems.append("codex features list reported no features")
    return rows, problems


def _strip_mcp(config_text: str) -> str:
    out, skip = [], False
    for line in config_text.splitlines():
        if line.startswith("["):
            skip = line.startswith("[mcp_servers")
        if not skip:
            out.append(line)
    return "\n".join(out) + "\n"


def _normalize(text: str, home: str, cwd: str) -> str:
    text = text.replace(home, "<CODEX_HOME>").replace(cwd, "<CWD>")
    text = re.sub(r"<current_date>[^<]*</current_date>", "<current_date>*</current_date>", text)
    text = re.sub(r"<timezone>[^<]*</timezone>", "<timezone>*</timezone>", text)
    return text


def _render(codex_bin: str, home: Path, model: str, prompt: str, scratch_root: Path):
    """Render the model-visible input list in a disposable copy of the home."""
    gate_home = Path(tempfile.mkdtemp(prefix="gate-home.", dir=scratch_root))
    gate_cwd = Path(tempfile.mkdtemp(prefix="gate-cwd.", dir=scratch_root))
    try:
        os.chmod(gate_home, 0o700)
        os.chmod(gate_cwd, 0o700)
        shutil.copy2(home / "auth.json", gate_home / "auth.json")
        (gate_home / "config.toml").write_text(
            _strip_mcp((home / "config.toml").read_text(encoding="utf-8")), encoding="utf-8")
        env = dict(os.environ, CODEX_HOME=str(gate_home))
        completed = _run([codex_bin, "debug", "prompt-input", "-c", f"model={model}", "--",
                          prompt], env, cwd=str(gate_cwd), timeout=180)
        if completed.returncode != 0:
            return None, [f"prompt-input exited {completed.returncode}: "
                          f"{completed.stderr.strip()[:300]}"]
        try:
            items = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return None, ["prompt-input did not return JSON"]
        blocks = []
        for item in items if isinstance(items, list) else []:
            text = "".join(
                c.get("text", "") for c in (item.get("content") or []) if isinstance(c, dict))
            match = _TAG.match(text)
            blocks.append({
                "role": item.get("role"),
                "tag": match.group(1) if match else "(plain)",
                "bytes": len(text.encode("utf-8")),
                "raw": text,
                "sha256": hashlib.sha256(
                    _normalize(text, str(gate_home), str(gate_cwd)).encode("utf-8")).hexdigest(),
            })
        return blocks, []
    finally:
        shutil.rmtree(gate_home, ignore_errors=True)
        shutil.rmtree(gate_cwd, ignore_errors=True)


def _check_text(blocks, prompt: str, manifest: dict) -> list[str]:
    problems = []
    if not blocks:
        return ["prompt-input rendered no blocks"]
    expected = manifest.get("blocks") or []
    if len(blocks) != len(expected) + 1:
        problems.append(
            f"rendered {len(blocks)} blocks, manifest expects {len(expected)} preamble blocks "
            f"plus our prompt")
    for index, want in enumerate(expected):
        if index >= len(blocks):
            break
        got = blocks[index]
        if (got["role"], got["tag"], got["sha256"]) != (
                want["role"], want["tag"], want["sha256"]):
            problems.append(
                f"block {index} is {got['role']}/{got['tag']}/{got['sha256'][:12]}, manifest "
                f"expects {want['role']}/{want['tag']}/{want['sha256'][:12]}")
    last = blocks[-1]
    if last["role"] != "user" or last["raw"] != prompt:
        problems.append("the final user block is not our prompt byte for byte")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex-home", required=True, type=Path)
    ap.add_argument("--allowlist", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--model", default=None, help="enables the text-purity check")
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--preamble-manifest", type=Path, default=None)
    ap.add_argument("--model-catalog", type=Path, default=None,
                    help="record the bundled catalog used by this isolated session")
    ap.add_argument("--write-manifest", type=Path, default=None,
                    help="measure the preamble and write it as the manifest instead of checking")
    ap.add_argument("--scratch-root", type=Path, default=None)
    args = ap.parse_args(argv)
    scratch_root = args.scratch_root or args.codex_home.parent

    violations: list[str] = []
    before = sorted(p.name for p in args.codex_home.iterdir())
    foreign_before = sorted(set(before) - HOME_BEFORE)
    if foreign_before:
        violations.append(f"CODEX_HOME holds foreign entries before start: {foreign_before}")
    if not HOME_BEFORE <= set(before):
        violations.append(f"CODEX_HOME is missing {sorted(HOME_BEFORE - set(before))}")

    allowlist = _load_allowlist(args.allowlist)
    env = dict(os.environ, CODEX_HOME=str(args.codex_home))
    rows, problems = _features(args.codex_bin, env)
    violations.extend(problems)
    enabled = sorted(name for name, row in rows.items() if row["enabled"])
    unexpected = sorted(set(enabled) - allowlist)
    if unexpected:
        violations.append(f"enabled features outside the allowlist: {unexpected}")

    text_report = None
    if args.model:
        prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else ""
        blocks, problems = _render(args.codex_bin, args.codex_home, args.model, prompt,
                                   scratch_root)
        violations.extend(problems)
        if blocks is not None:
            public = [{k: b[k] for k in ("role", "tag", "bytes", "sha256")} for b in blocks]
            if args.write_manifest:
                manifest = {"schema_version": GATE_SCHEMA_VERSION, "model": args.model,
                            "blocks": public[:-1]}
                args.write_manifest.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                print(f"wrote {args.write_manifest}", file=sys.stderr)
            elif args.preamble_manifest:
                manifest = json.loads(args.preamble_manifest.read_text(encoding="utf-8"))
                violations.extend(_check_text(blocks, prompt, manifest))
            else:
                violations.append("text-purity check requested without a manifest")
            text_report = {
                "blocks": public,
                "preamble_bytes": sum(b["bytes"] for b in public[:-1]),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_bytes": len(prompt.encode("utf-8")),
            }

    after = sorted(p.name for p in args.codex_home.iterdir())
    foreign_after = sorted(set(after) - HOME_BEFORE - HOME_SCRATCH)
    if foreign_after:
        violations.append(f"CODEX_HOME grew foreign entries: {foreign_after}")

    report = {
        "schema_version": GATE_SCHEMA_VERSION,
        "healthy": not violations,
        "violations": violations,
        "enabled_features": enabled,
        "unexpected_enabled": unexpected,
        # Allowlisted but currently off: informational. A later CLI turning one off changes the
        # stack under test and deserves a look, but is not a breach.
        "allowlisted_now_disabled": sorted(allowlist - set(enabled)),
        "feature_count": len(rows),
        "home_entries_before": before,
        "home_entries_after": after,
        "text": text_report,
    }
    if args.model_catalog is not None:
        report["model_catalog_sha256"] = hashlib.sha256(args.model_catalog.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if violations:
        print("codex gate REFUSED:", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 2
    print(f"codex gate ok: {len(enabled)} features enabled, all allowlisted"
          + (f"; {len(text_report['blocks'])} prompt blocks verified" if text_report else ""),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
