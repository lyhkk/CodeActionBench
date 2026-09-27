"""Prepare verified release runtime copies without requiring a clean Git checkout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from codeaction.release import load_release, sha256, source_files


def snapshot_files(root: Path) -> dict:
    """All prepared inputs, including role projections and non-secret configuration."""
    return {p.relative_to(root).as_posix(): [sha256(p), bool(p.stat().st_mode & 0o111)]
            for p in sorted(root.rglob("*")) if p.is_file() and p != root / "snapshot.json"}


def verify_snapshot(root: Path) -> dict:
    root = root.resolve()
    record = json.loads((root / "snapshot.json").read_text())
    if record.get("schema_version") != "codeaction-snapshot.v2":
        raise ValueError("unsupported execution snapshot")
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    if root.name != identity:
        raise ValueError("snapshot record identity differs from its content address")
    if any(p.is_symlink() for p in root.rglob("*")) or snapshot_files(root) != record["files"]:
        raise ValueError("execution snapshot is missing or changed; restore the original snapshot")
    return record


def prepare_snapshot(source: Path, cache: Path, *, task_pack: Path | None = None,
                     extension_paths=(), model_registry: Path | None = None,
                     baseline: Path | None = None, require_match=False,
                     configuration_files: dict | None = None, include_model_overlay=True, provider_env_file: Path | None = None) -> tuple[Path, dict]:
    """Freeze current inputs, with an optional immutable release used for comparison."""
    from codeaction.components import prepare_components
    from codeaction.extensions import read_declarations, freeze_extensions
    from codeaction.providers import model_registry as models
    from codeaction.environments import component
    source = source.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=cache))
    try:
        before = source_files(source)
        for name, digest in before.items():
            dest = staging / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, dest)
            if sha256(dest) != digest:
                raise ValueError(f"source changed while copying: {name}")
        if source_files(source) != before:
            raise ValueError("source changed while preparing snapshot")
        if task_pack is not None and task_pack.resolve() != source / "benchmark/tasks":
            shutil.rmtree(staging / "benchmark/tasks")
            pack = task_pack.resolve()
            if any(p.is_symlink() for p in pack.rglob("*")):
                raise ValueError("task package must not contain symlinks")
            shutil.copytree(pack, staging / "benchmark/tasks")
            if snapshot_files(pack) != snapshot_files(staging / "benchmark/tasks"):
                raise ValueError("task pack changed while copying")
        entries = read_declarations(extension_paths)
        config = staging / "config"
        config.mkdir()
        raw = json.loads((source / "src/codeaction/providers/models/registry.json").read_text())
        overlay = (json.loads(model_registry.expanduser().read_text())["models"]
                   if model_registry else models._overlay_models() if include_model_overlay else {})
        raw["models"].update(overlay)
        def reject_secrets(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in {"api_key", "access_token", "refresh_token", "password", "secret"}:
                        raise ValueError("credentials must be stored separately from model/run configuration")
                    reject_secrets(item)
            elif isinstance(value, list):
                for item in value: reject_secrets(item)
        reject_secrets(raw)
        for entry in entries:
            reject_secrets(entry.get("config", {}))
        # Parse before writing; neither credentials nor arbitrary environment entries are copied.
        for name, entry in raw["models"].items():
            if entry.get("protocol") not in {e["name"] for e in entries if e["kind"] == "provider"}:
                models._parse_entry(name, entry)
        endpoints = {}
        if provider_env_file is not None and provider_env_file.expanduser().is_file():
            from urllib.parse import urlsplit, parse_qsl
            endpoints = {key: value for key, value in models.read_credential_file(provider_env_file.expanduser()).items()
                         if key.endswith("_BASE_URL")}
            for value in endpoints.values():
                parsed = urlsplit(value)
                if parsed.username or parsed.password or any(key.lower() in {"key", "api_key", "apikey", "token", "access_token"} for key, _ in parse_qsl(parsed.query)):
                    raise ValueError("endpoint credentials must use the separate KEY field, not the URL")
        (config / "endpoints.json").write_text(json.dumps(endpoints, sort_keys=True) + "\n")
        (config / "models.json").write_text(json.dumps(raw, indent=2) + "\n")
        for name, original in (configuration_files or {}).items():
            if name not in {"rate-limits.json", "agents.json"}:
                raise ValueError("only non-secret account and rate configuration may be frozen")
            content = Path(original).read_bytes()
            reject_secrets(json.loads(content))
            (config / name).write_bytes(content)
        freeze_extensions(entries, staging)
        (staging / "backend/robotwin/assets").mkdir(exist_ok=True)
        prepare_components(staging)
        for role in ("reference-agent", "claude-agent", "codex-agent"):
            public = freeze_extensions(entries, staging / "components" / role, kinds={"provider", "agent"})
            public += [{k: e[k] for k in ("kind", "name", "description", "input_schema", "output_schema", "returns", "replace") if k in e}
                       for e in entries if e["kind"] == "tool"]
            (staging / "components" / role / "config/extensions.json").write_text(json.dumps(public, indent=2) + "\n")
        import subprocess
        commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                text=True, capture_output=True, check=False).stdout.strip()
        if len(commit) != 40:
            commit = "0" * 40
        actual = source_files(staging)
        model_identity = sha256(config / "models.json")
        extension_identity = hashlib.sha256(json.dumps({
            name: value for name, value in snapshot_files(staging).items()
            if name.startswith("extensions/") or name == "config/extensions.json"
        }, sort_keys=True).encode()).hexdigest()
        changes = []
        original = {}
        if baseline:
            original = json.loads(baseline.read_text())
            expected = original.get("source_files", {})
            changes = [{"path": name, "component": component(name)}
                       for name in sorted(set(expected) | set(actual)) if expected.get(name) != actual.get(name)]
            if (original.get("model_definitions_sha256") != model_identity
                    if "model_definitions_sha256" in original else bool(overlay)):
                changes.append({"path": "model definitions", "component": "agents"})
            if (original.get("extensions_sha256") != extension_identity
                    if "extensions_sha256" in original else bool(entries)):
                for kind in sorted({e["kind"] for e in entries} or {"agent"}):
                    changes.append({"path": "local extensions/" + kind,
                                    "component": {"tool": "protocol", "verifier": "scoring"}.get(kind, "agents")})
            (config / "baseline.json").write_text(json.dumps(original, indent=2) + "\n")
        if require_match and (not baseline or changes):
            raise ValueError("strict reproduction requires a matching release; " + json.dumps(changes))
        context = {"schema_version": "codeaction-context.v2", "source_commit": commit,
                   "baseline": "matching" if baseline and not changes else "modified" if baseline else "unregistered",
                   "changes": changes, "require_release_match": bool(require_match),
                   "model_definitions_sha256": model_identity, "extensions_sha256": extension_identity,
                   "contracts_from_runtime": original.get("contracts_from_runtime", False)}
        groups = {}
        for name, value in snapshot_files(staging).items():
            if name.startswith("components/"):
                continue
            group = component(name)
            if name.startswith("extensions/"):
                kind = Path(name).parts[1].split("_", 1)[0]
                group = {"agent": "agents", "provider": "agents", "tool": "protocol", "verifier": "scoring"}[kind]
            groups.setdefault(group, {})[name] = value
        context["components"] = {key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                                 for key, value in groups.items()}
        (config / "context.json").write_text(json.dumps(context, indent=2) + "\n")
        record = {"schema_version": "codeaction-snapshot.v2", "files": snapshot_files(staging)}
        identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        target = cache.resolve() / identity
        (staging / "snapshot.json").write_text(json.dumps(record, sort_keys=True) + "\n")
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        if target.exists():
            if verify_snapshot(target) != record:
                raise ValueError("cached execution differs")
        else:
            os.rename(staging, target)
        return target, context
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare_release_snapshot(source: Path, manifest: Path, cache: Path) -> tuple[Path, Path]:
    """Publish a complete copy atomically; validate cached copies before every reuse.

    Only runtime files named by the validated release are copied. Local configuration,
    credentials and results outside that selection never enter the execution directory.
    """
    source = source.resolve()
    manifest_bytes = manifest.read_bytes()
    release = load_release(manifest, source)
    if manifest.read_bytes() != manifest_bytes:
        raise ValueError("release manifest changed while preparing execution")
    identity = hashlib.sha256(manifest_bytes).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache.resolve() / identity
    pinned = target / "release-manifest.json"
    if target.exists():
        if pinned.is_symlink() or pinned.read_bytes() != manifest_bytes:
            raise ValueError("cached execution manifest differs; remove the damaged snapshot")
        load_release(pinned, target)
        return target, pinned
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=cache))
    try:
        for name, digest in release["source_files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("release contains an unsafe source path")
            original = source / relative
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if original.is_symlink():
                raise ValueError(f"source changed into a symlink: {name}")
            shutil.copyfile(original, destination)
            destination.chmod(0o555 if original.stat().st_mode & 0o111 else 0o444)
            if sha256(destination) != digest:
                raise ValueError(f"source changed while copying: {name}")
        if source_files(source, selection=release.get("source_selection", 1)) != release["source_files"]:
            raise ValueError("source changed while preparing execution")
        (staging / "release-manifest.json").write_bytes(manifest_bytes)
        load_release(staging / "release-manifest.json", staging)
        # Copies are read-only to prevent accidental edits; reuse also verifies hashes.
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        try:
            os.rename(staging, target)
        except OSError:
            if not target.exists():
                raise
            if pinned.read_bytes() != manifest_bytes:
                raise ValueError("concurrent snapshot preparation disagreed")
            load_release(pinned, target)
        return target, pinned
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def save_launch_record(path: Path, value: dict) -> None:
    """Publish a complete launch record without replacing another launch's record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".launch-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
