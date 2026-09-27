"""Readable impact reports and immutable release preparation; never launches models."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from codeaction.environments import component, environment_files, environment_identity, ROLES, validate_environment
from codeaction.release import source_files, write_json

CHECKS = {
    "tasks": ["task_pack"],
    "scoring": ["python_syntax", "task_pack"],
    "protocol": ["python_syntax", "component_files", "task_pack"],
    "agents": ["python_syntax", "component_files"],
    "environment": ["python_syntax", "component_files"],
    "documentation": ["documentation_links"],
    "tests": [],
    "reporting": ["python_syntax"],
    "shared": ["python_syntax", "documentation_links", "component_files", "task_pack"],
}


def changes(root: Path, base: str) -> dict:
    candidate = Path(base).expanduser()
    if not candidate.is_file() and (root / ".release" / (base + ".json")).is_file():
        candidate = root / ".release" / (base + ".json")
    if candidate.is_file():
        old = json.loads(candidate.read_text())["source_files"]
        current = source_files(root)
        names = sorted(name for name in set(old) | set(current) if old.get(name) != current.get(name))
    else:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", base + "^{commit}"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        names = subprocess.run(["git", "-C", str(root), "diff", "--name-only", revision, "--"],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        names += subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
                                capture_output=True, text=True, check=True).stdout.splitlines()
    groups = {}
    for name in sorted(set(names)):
        groups.setdefault(component(name), []).append(name)
    environments = [role for role in ROLES if set(names) & set(environment_files(role))]
    tasks = sorted({Path(name).parts[2] for name in names
                    if name.startswith("benchmark/tasks/") and len(Path(name).parts) > 3})
    checks = sorted({check for group in groups for check in CHECKS[group]})
    return {"base": base, "components": groups, "affected_tasks": tasks,
            "environments_to_build": environments, "checks": checks,
            "semantic_compatibility": "review required" if groups.keys() - {"documentation", "tests"} else "no runtime changes"}


def require_public_checkout(root: Path) -> None:
    """A release commit must belong to this export, not an enclosing development repo."""
    if (root / "tests").exists():
        raise ValueError("prepare the release from the public export; keep maintenance tests outside it")
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True, check=False)
    if result.returncode or Path(result.stdout.strip()).resolve() != root.resolve():
        raise ValueError("the reviewed public source must have its own Git checkout before release preparation")


def prepare_release(args, root: Path) -> Path:
    root = root.resolve()
    require_public_checkout(root)
    from codeaction.cli.main import _git_state, _inspect_image
    from codeaction.benchmark.taskcard import validate_task_pack
    commit, dirty = _git_state(root)
    if dirty:
        raise ValueError("commit the reviewed source before publishing a release manifest")
    if not args.version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in args.version):
        raise ValueError("release version must be a simple version name")
    assets_path = args.assets_manifest or root / ".release/assets.json"
    assets = json.loads(assets_path.read_text())
    if assets.get("schema_version") != "codeaction-assets.v1":
        raise ValueError("unsupported asset manifest")
    images, environments = {}, {}
    from codeaction.image_distribution import PROFILES, load_manifest
    manifest_path = getattr(args, "images_manifest", None)
    if manifest_path and args.image:
        raise ValueError("choose --images-manifest or --image")
    if manifest_path is None and not args.image and (root / "docker/images.json").is_file():
        manifest_path = root / "docker/images.json"
    if manifest_path:
        declared = load_manifest(manifest_path, root)
        image_items = [f"{role}={declared['images'][role]['ref']}" for role in PROFILES["all"]]
    else:
        image_items = args.image or [f"{role}=codeaction-{role}:dev" for role in PROFILES["all"]]
    for item in image_items:
        role, ref = item.split("=", 1)
        if role in images:
            raise ValueError(f"duplicate image role: {role}")
        image = _inspect_image(ref)
        validate_environment(image.labels, root, role)
        images[role] = image.digest or image.image_id
        environments[role] = environment_identity(root, role)
    if not {"sim", "gateway"}.issubset(images):
        raise ValueError("release requires sim and gateway environments")
    output = args.out or root / ".release" / (args.version + ".json")
    value = {"schema_version": "codeaction-release.v2", "version": args.version,
             "source_commit": commit, "source_selection": 2, "source_files": source_files(root),
             "images": images, "environments": environments, "assets": assets,
             "task_pack_sha256": validate_task_pack(root / "benchmark/tasks", strict_pins=False)["sha256"],
             "contracts_from_runtime": True}
    from codeaction.execution_snapshot import prepare_snapshot
    import tempfile
    # Publishing never accidentally includes the maintainer's home model overlay.
    for name in args.extensions or ():
        path = name.expanduser().resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("published extension declarations must be committed within the repository")
    from codeaction.extensions import read_declarations
    files = list(args.extensions or ()) + ([args.model_registry] if args.model_registry else [])
    for entry in read_declarations(args.extensions or ()):
        files += [Path(entry["source"]) / filename for filename in entry["files"]]
    for filename in files:
        filename = Path(filename).expanduser().resolve()
        if not filename.is_relative_to(root.resolve()):
            raise ValueError("published configuration and extension code must be inside the repository")
        subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(filename.relative_to(root))],
                       check=True, capture_output=True)
    with tempfile.TemporaryDirectory() as temporary:
        _, resolved = prepare_snapshot(root, Path(temporary), extension_paths=args.extensions or (),
                                       model_registry=args.model_registry, include_model_overlay=False)
    value["model_definitions_sha256"] = resolved["model_definitions_sha256"]
    value["extensions_sha256"] = resolved["extensions_sha256"]
    if args.base:
        value["changes"] = changes(root, args.base)
    if _git_state(root) != (commit, False):
        raise ValueError("source changed during release preparation")
    write_json(output, value)
    return output


def add_parsers(sub):
    parser = sub.add_parser("changes", help="show component changes and relevant checks")
    parser.add_argument("--base", required=True)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    release = sub.add_parser("release", help="prepare versioned release artifacts")
    prepare = release.add_subparsers(dest="release_command", required=True).add_parser("prepare")
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--assets-manifest", type=Path, default=None,
                         help="defaults to .release/assets.json in the repository")
    prepare.add_argument("--image", action="append")
    prepare.add_argument("--images-manifest", type=Path,
                         help="published dependency images; defaults to docker/images.json when present")
    prepare.add_argument("--out", type=Path)
    prepare.add_argument("--base")
    prepare.add_argument("--extensions", nargs="+", type=Path)
    prepare.add_argument("--model-registry", type=Path)


def main(args, root):
    if args.command == "release":
        print(prepare_release(args, root))
    else:
        value = changes(root, args.base)
        if args.format == "json":
            print(json.dumps(value, indent=2))
        else:
            for group, names in value["components"].items():
                print(group + ": " + ", ".join(names))
            print("Environment rebuilds: " + (", ".join(value["environments_to_build"]) or "none"))
            print("Public checks: " + (", ".join(value["checks"]) or "none; review private maintenance changes"))
            print("Compatibility: " + value["semantic_compatibility"])
    return 0
