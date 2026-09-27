"""Explicit local extension declarations, loaded only in their execution process."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil

KINDS = {"provider", "agent", "tool", "verifier"}
_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]*\Z")


def read_declarations(paths) -> list[dict]:
    import yaml
    entries, seen = [], set()
    for path in paths:
        path = Path(path).expanduser().resolve()
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError(f"unsupported extension declaration: {path}")
        for raw in value.get("extensions", []):
            entry = dict(raw)
            kind, name = entry.get("kind"), entry.get("name", "")
            if kind not in KINDS or not _NAME.fullmatch(name) or (kind, name) in seen:
                raise ValueError(f"invalid or duplicate extension: {kind}/{name}")
            seen.add((kind, name))
            if not isinstance(entry.get("files"), list) or not entry["files"]:
                raise ValueError(f"extension {name} must explicitly list its source files")
            if not isinstance(entry.get("entrypoint"), str) or ":" not in entry["entrypoint"]:
                raise ValueError(f"extension {name} needs module:callable entrypoint")
            if entry.get("permissions") not in (None, {}):
                raise ValueError("additional filesystem/network permissions need an environment adapter; cannot be granted by an extension")
            if kind == "tool":
                if name in {"done", "run_code", "write_file", "read_file", "list_files", "run_program"}:
                    raise ValueError("episode control/composition requires a protocol change, not a tool override")
                required = {"description", "input_schema", "output_schema", "returns"}
                if not required.issubset(entry) or not all(isinstance(entry[k], dict) for k in ("input_schema", "output_schema")):
                    raise ValueError(f"tool {name} requires input/output schemas, description and returns")
            if kind == "agent" and "credential" in entry and (not isinstance(entry["credential"], str) or not re.fullmatch(r"[a-z][a-z0-9-]*", entry["credential"])):
                raise ValueError("agent credential must name a local alias")
            if "replace" in entry and type(entry["replace"]) is not bool:
                raise ValueError("replace must be boolean")
            entry["source"] = str((path.parent / entry.get("source", ".")).resolve())
            for filename in entry["files"]:
                relative = Path(filename)
                if relative.is_absolute() or ".." in relative.parts or relative.suffix not in {".py", ".json"}:
                    raise ValueError(f"unsupported extension source file: {filename}")
                original = Path(entry["source"]) / relative
                if not original.is_file() or any(p.is_symlink() for p in (original, *original.parents)):
                    raise ValueError(f"extension must name regular files: {filename}")
            entries.append(entry)
    return entries


def freeze_extensions(entries: list[dict], destination: Path, *, kinds=KINDS) -> list[dict]:
    selected = []
    for entry in entries:
        if entry["kind"] not in kinds:
            continue
        item = dict(entry)
        directory = Path("extensions") / (item["kind"] + "_" + item["name"])
        for name in item["files"]:
            origin = Path(item["source"]) / name
            target = destination / directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            import hashlib
            before = hashlib.sha256(origin.read_bytes()).hexdigest()
            shutil.copy2(origin, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != before or hashlib.sha256(origin.read_bytes()).hexdigest() != before:
                raise ValueError(f"extension changed while copying: {name}")
        item["source"] = directory.as_posix()
        selected.append(item)
    config = destination / "config/extensions.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def declarations(kind: str) -> dict[str, dict]:
    path = os.environ.get("CODEACTION_EXTENSIONS_FILE")
    if not path:
        return {}
    values = json.loads(Path(path).read_text())
    return {item["name"]: item for item in values if item["kind"] == kind}


def entrypoint(kind: str, name: str, *, builtin=None):
    entry = declarations(kind).get(name)
    if entry is None:
        return builtin
    if builtin is not None and entry.get("replace") is not True:
        raise ValueError(f"extension {kind}/{name} must declare replace: true")
    # Requirements are checked, never installed during a run.
    from packaging.requirements import Requirement
    for requirement in entry.get("requirements", []):
        req = Requirement(requirement)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"extension {name} requires {req}; update its environment") from exc
        if version not in req.specifier:
            raise ValueError(f"extension {name} requires {req}, installed {version}")
    import sys
    root = Path(os.environ["CODEACTION_EXTENSIONS_FILE"]).parent.parent
    path = (root / entry["source"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("extension escaped its snapshot")
    module, attribute = entry["entrypoint"].split(":", 1)
    # Each extension has its own package namespace, even if both use main.py.
    import importlib.util
    import types
    import hashlib
    namespace = "_codeaction_extension_" + kind + "_" + name + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(path)]
        sys.modules[namespace] = package
    return getattr(importlib.import_module(namespace + "." + module), attribute)
