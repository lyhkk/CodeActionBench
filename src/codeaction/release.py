"""Portable release locks for source, simulator images, and separately installed assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from codeaction.paths import PROJECT_ROOT

SOURCE_DIRS = ("src", "oracle", "backend", "docker", "configs", "tools", "benchmark")
SOURCE_FILES = ("pyproject.toml", "VERSION", "benchmark_package.json")
IMAGE_ARGS = {"sim": "sim_image", "gateway": "gateway_image",
              "reference-agent": "reference_agent_image", "fixture-agent": "fixture_agent_image",
              "claude-agent": "claude_agent_image", "codex-agent": "codex_agent_image",
              "scratch": "scratch_image", "scratch-launcher": "launcher_image"}
PINNED_IMAGE = re.compile(r"(?:sha256:[0-9a-f]{64}|[a-z0-9][^\s@]*@sha256:[0-9a-f]{64})\Z")
SOURCE_LABEL = "org.codeaction.source-sha256"
RUNTIME_LABEL = "org.codeaction.runtime-sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(root: Path, *, selection: int = 2, include_tasks: bool = True) -> dict[str, str]:
    """Runtime-affecting source only; documentation edits do not invalidate release images."""
    if selection not in (1, 2):
        raise ValueError("unsupported source selection")
    paths = [root / name for name in SOURCE_FILES]
    for name in SOURCE_DIRS:
        paths.extend((root / name).rglob("*"))
    result = {}
    for path in sorted(paths):
        relative = path.relative_to(root)
        if not include_tasks and (relative.is_relative_to("benchmark/tasks")
                                  or relative == Path("oracle/manifest.json")):
            continue
        if any(part in {"__pycache__", "assets", "results"} or part.endswith(".egg-info")
               for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if selection == 2:
            # User recipes and authentication are not simulator runtime inputs.
            if relative.parts[0] == "configs" and not relative.is_relative_to("configs/robotwin"):
                continue
            if path.suffix.lower() in {".env", ".token"} or path.name.startswith(".env"):
                continue
            if path.suffix.lower() == ".md" and not (
                    relative.is_relative_to("benchmark/tasks") and path.name == "instruction.md"):
                continue
        if path.is_symlink():
            raise ValueError(f"runtime source must not be a symlink: {relative}")
        if path.is_file():
            result[relative.as_posix()] = sha256(path)
    return result


def source_identity(root: Path) -> str:
    """Identify the selected source, including uncommitted content and executable bits.

    This is provenance, not a Docker cache key. Docker owns layer invalidation.
    """
    files = {name: [digest, bool((root / name).stat().st_mode & 0o111)]
             for name, digest in source_files(root).items()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def runtime_identity(root: Path) -> str:
    """Code/configuration baked into images; task packages are mounted separately."""
    files = {name: [digest, bool((root / name).stat().st_mode & 0o111)]
             for name, digest in source_files(root, include_tasks=False).items()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def image_matches_runtime(labels: dict, expected: str, *, role: str | None = None) -> bool:
    return (labels.get(RUNTIME_LABEL) == expected
            and (role is None or labels.get("org.opencontainers.image.title") == f"codeaction-{role}"))


def prepare_build_context(root: Path, output: Path) -> None:
    """Copy only declared runtime inputs into a fresh Docker build context."""
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("build context must be separate from the source")
    files = source_files(root)
    identity = source_identity(root)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, digest in files.items():
            original, target = root / name, output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if original.is_symlink():
                raise ValueError("source changed into a symlink")
            shutil.copy2(original, target)
            if sha256(target) != digest:
                raise ValueError(f"source changed while preparing build: {name}")
        if source_identity(root) != identity or source_identity(output) != identity:
            raise ValueError("source changed while preparing build")
    except BaseException:
        shutil.rmtree(output)
        raise


def asset_files(root: Path) -> dict[str, dict]:
    """Hash the three runtime resource trees, excluding download archives and metadata."""
    result = {}
    for tree in ("objects", "embodiments", "background_texture"):
        directory = root / tree
        if not directory.is_dir():
            raise ValueError(f"missing asset directory: {tree}")
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                result[path.relative_to(root).as_posix()] = {
                    "bytes": path.stat().st_size, "sha256": sha256(path)}
    if not result:
        raise ValueError("empty asset tree")
    return result


def write_json(path: Path, value: dict, *, overwrite: bool = False) -> None:
    """Expose complete JSON atomically; published manifests are create-only by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"manifest already exists: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Linking is atomic and refuses a target created by a concurrent publisher.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_assets(root: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != "codeaction-assets.v1":
        raise ValueError("unsupported asset manifest")
    if asset_files(root) != manifest.get("files"):
        raise ValueError("assets differ from the release asset manifest")


def verify_release_source(path: Path, root: Path) -> dict:
    """Verify immutable files without interpreting a saved runtime's task schema."""
    value = json.loads(path.read_text())
    if value.get("schema_version") not in {"codeaction-release.v1", "codeaction-release.v2"}:
        raise ValueError("unsupported release manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", value.get("source_commit", "")):
        raise ValueError("release source commit is invalid")
    if source_files(root, selection=value.get("source_selection", 1)) != value.get("source_files"):
        raise ValueError("runtime source differs from the release (including tasks); use matching source or publish a new manifest")
    if "runtime_sha256" in value and runtime_identity(root) != value["runtime_sha256"]:
        raise ValueError("runtime identity differs from the release")
    images = value.get("images", {})
    if not isinstance(images, dict) or not {"sim", "gateway"}.issubset(images):
        raise ValueError("release must pin sim and gateway images")
    for role, ref in images.items():
        if role not in IMAGE_ARGS or not isinstance(ref, str) or not PINNED_IMAGE.fullmatch(ref):
            raise ValueError(f"invalid immutable image reference for {role}")
    return value


def load_release(path: Path, root: Path) -> dict:
    value = verify_release_source(path, root)
    from codeaction.benchmark.taskcard import validate_task_pack
    options = {"strict_pins": False} if value.get("contracts_from_runtime") else {}
    pack = validate_task_pack(root / "benchmark/tasks", **options)
    if pack["sha256"] != value.get("task_pack_sha256"):
        raise ValueError("task pack differs from the release")
    return value


def apply_release(args) -> dict:
    value = load_release(args.release_manifest, args.source_root.resolve())
    from codeaction.benchmark.taskcard import validate_task_pack
    selected_pack = getattr(args, "task_pack", None) or args.source_root / "benchmark/tasks"
    if (selected_pack.resolve() != (args.source_root / "benchmark/tasks").resolve()
            and validate_task_pack(selected_pack)["sha256"] != value["task_pack_sha256"]):
        raise ValueError("selected task pack differs from the release")
    role = {"reference": "reference-agent", "fixture": "fixture-agent",
            "claude": "claude-agent", "codex": "codex-agent"}[args.agent_mode]
    required = {"sim", "gateway", role}
    if args.interface_profile == "vendor-mcp-gateway":
        required |= {"scratch", "scratch-launcher"}
    if not required.issubset(value["images"]):
        raise ValueError(f"release does not include the requested images: {sorted(required)}")
    for name, ref in value["images"].items():
        setattr(args, IMAGE_ARGS[name], ref)
    # Full resource verification is explicit and happens before Docker starts.
    assets = value.get("assets")
    if assets is None:
        raise ValueError("release has no asset manifest")
    verify_assets(args.assets_root.resolve(), assets)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    context = sub.add_parser("build-context", help="prepare an explicit, credential-free Docker context")
    context.add_argument("root", type=Path)
    context.add_argument("output", type=Path)
    identity = sub.add_parser("source-id", help="identify actual selected source content")
    identity.add_argument("root", type=Path)
    runtime = sub.add_parser("runtime-id", help="identify runtime independently of task packages")
    runtime.add_argument("root", type=Path)
    assets = sub.add_parser("assets", help="write a portable resource manifest")
    assets.add_argument("root", type=Path)
    assets.add_argument("output", type=Path)
    export = sub.add_parser("export-assets", help="copy resources and normalize robot paths")
    export.add_argument("root", type=Path)
    export.add_argument("output", type=Path)
    verify = sub.add_parser("verify-assets")
    verify.add_argument("root", type=Path)
    verify.add_argument("manifest", type=Path)
    pull = sub.add_parser("pull", help="pull the immutable registry images in a release")
    pull.add_argument("manifest", type=Path)
    pull.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    if args.command == "build-context":
        prepare_build_context(args.root, args.output)
    elif args.command == "runtime-id":
        print(runtime_identity(args.root))
        return 0
    elif args.command == "source-id":
        print(source_identity(args.root.resolve()))
        return 0
    elif args.command == "assets":
        write_json(args.output, {"schema_version": "codeaction-assets.v1", "files": asset_files(args.root)})
    elif args.command == "export-assets":
        import yaml
        from codeaction.backends.robotwin.asset_config import relative_robot_config
        source, output = args.root.resolve(), args.output.resolve()
        if output == source or source in output.parents or output in source.parents or output.exists():
            raise ValueError("asset output must be new and separate from the source")
        for tree in ("objects", "embodiments", "background_texture"):
            if not (source / tree).is_dir():
                raise ValueError(f"missing source assets: {tree}")
        output.mkdir(parents=True)
        for tree in ("objects", "embodiments", "background_texture"):
            shutil.copytree(source / tree, output / tree)
        for path in sorted((output / "embodiments").rglob("*.yml")):
            data = yaml.safe_load(path.read_text())
            if isinstance(data, dict) and isinstance(data.get("robot_cfg"), dict):
                normalized = relative_robot_config(data, path.parent)
                if normalized != data:
                    path.write_text(yaml.safe_dump(normalized, sort_keys=False))
        write_json(output / "assets-manifest.json", {
            "schema_version": "codeaction-assets.v1", "files": asset_files(output)})
    elif args.command == "verify-assets":
        verify_assets(args.root, json.loads(args.manifest.read_text()))
    elif args.command == "pull":
        value = load_release(args.manifest, args.source_root.resolve())
        for ref in dict.fromkeys(value["images"].values()):
            command = ["docker", "image", "inspect", ref] if ref.startswith("sha256:") else ["docker", "pull", ref]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    print("verified" if args.command in {"verify-assets", "pull"} else str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
