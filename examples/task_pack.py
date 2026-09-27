"""Create a separate task pack from a task card, or validate an existing pack on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from codeaction.benchmark.taskcard import CANARY_MARK, load_registry, load_task, validate_task_pack


def create_pack(source: Path, destination: Path, name: str) -> dict:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise ValueError("destination already exists; choose a new task pack directory")
    load_task(source)
    card = json.loads((source / "task.json").read_text())
    card["task"] = {"name": name}
    registry = {"schema_version": card["schema_version"], "taskset_version": "local.1",
                "_canary": CANARY_MARK, "tasks": [name]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".task-pack-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        (staging / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
        load_registry(staging)  # Validate the ID before using it as a directory name.
        task = staging / name
        task.mkdir()
        (task / "task.json").write_text(json.dumps(card, indent=2) + "\n")
        shutil.copyfile(source / "instruction.md", task / "instruction.md")
        validate_task_pack(staging)
        # mkdir refuses an intervening creator; never merge into another task pack.
        destination.mkdir()
        try:
            shutil.copytree(staging, destination, dirs_exist_ok=True)
        except Exception:
            shutil.rmtree(destination)
            raise
    return validate_task_pack(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="copy only task.json and instruction.md into a new pack")
    create.add_argument("--source", type=Path, required=True, help="source task directory")
    create.add_argument("--name", required=True, help="new registered task ID")
    create.add_argument("--out", type=Path, required=True, help="new pack directory")
    validate = sub.add_parser("validate", help="validate the registered cards without a simulator")
    validate.add_argument("pack", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (create_pack(args.source, args.out, args.name) if args.command == "create"
                  else validate_task_pack(args.pack))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Task pack: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
