#!/usr/bin/env python3
"""Check public source syntax, links and declarations without Docker or model calls."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from codeaction.cli.maintenance import CHECKS, changes


def source_paths(root: Path) -> list[Path]:
    """Include a source archive as well as a checkout, excluding local installation data."""
    excluded = {".git", ".release", ".venv", ".codeaction-env", "__pycache__",
                "tests", "assets", "runs", "demos", "build", "dist", ".pytest_cache"}
    paths = []
    def visit(directory: Path):
        for path in sorted(directory.iterdir()):
            if path.name in excluded or path.name.endswith(".egg-info") or path.is_symlink():
                continue
            if path == root / "configs/local":
                continue
            if path.is_dir():
                visit(path)
            elif path.is_file():
                paths.append(path)
    visit(root)
    return paths


def check_links(path: Path, root: Path) -> list[str]:
    failures = []
    for raw in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        target = raw.split(' "', 1)[0].strip("<>")
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        destination = path.parent / unquote(parts.path)
        if not destination.exists():
            failures.append(f"{path.relative_to(root)}: missing link {target}")
    return failures


def run_checks(root: Path, checks: list[str], names: list[str] | None = None) -> list[str]:
    failures = []
    paths = source_paths(root) if names is None else [root / name for name in names]
    paths = [path for path in paths if path.is_file() and not path.is_symlink()]
    for path in paths:
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if "documentation_links" in checks and path.suffix == ".md":
                failures.extend(check_links(path, root))
            if path.suffix == ".sh":
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                if result.returncode:
                    failures.append(f"{path.relative_to(root)}: invalid shell syntax")
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            failures.append(f"{path.relative_to(root)}: {exc}")
    if "component_files" in checks:
        try:
            declared = json.loads((root / "src/codeaction/component_files.json").read_text())
            for role, files in declared.items():
                for name in files:
                    path = Path(name)
                    if path.is_absolute() or ".." in path.parts or not (root / path).is_file() \
                            or (root / path).is_symlink():
                        failures.append(f"{role}: missing or unsafe component file {name}")
        except (OSError, ValueError, TypeError) as exc:
            failures.append(f"component declarations: {exc}")
    if "task_pack" in checks:
        try:
            from codeaction.benchmark.taskcard import validate_task_pack
            validate_task_pack(root / "benchmark/tasks", strict_pins=False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"task pack: {exc}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--base", help="review changes relative to a Git revision or release manifest")
    selection.add_argument("--all", action="store_true", help="check a complete checkout or source archive")
    args = parser.parse_args()
    if args.all:
        checks = sorted({check for values in CHECKS.values() for check in values})
        names = None
    else:
        report = changes(ROOT, args.base)
        checks = report["checks"]
        names = [name for group, files in report["components"].items()
                 if group != "tests" for name in files]
    failed = run_checks(ROOT, checks, names)
    if failed:
        print("Failed checks:\n" + "\n".join(failed), file=sys.stderr)
        return 1
    print("Public source checks passed: " + (", ".join(checks) or "no public source changes"))
    print("Behavioral and container validation are separate maintainer checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
