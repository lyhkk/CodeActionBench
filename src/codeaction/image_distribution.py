"""Publish and install dependency images independently of application source revisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from codeaction.environments import ENVIRONMENT_LABEL, environment_identity, validate_environment
from codeaction.paths import PROJECT_ROOT
from codeaction.release import IMAGE_ARGS, PINNED_IMAGE, write_json

SCHEMA = "codeaction-images.v1"
PROFILES = {
    "all": ("sim", "gateway", "reference-agent", "fixture-agent", "claude-agent", "codex-agent"),
    "reference-mcp": ("sim", "gateway", "reference-agent"),
    "vendor-mcp-direct": ("sim", "gateway", "fixture-agent", "claude-agent", "codex-agent"),
}
PUBLISH_ROLES = (*PROFILES["all"], "sim-base")
DIGEST = re.compile(r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")


def inspect(ref: str) -> dict:
    result = subprocess.run(["docker", "image", "inspect", ref], check=True,
                            capture_output=True, text=True)
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("expected exactly one local image")
    return rows[0]


def check_image(root: Path, role: str, ref: str, *, image_id: str | None = None) -> dict:
    image = inspect(ref)
    labels = (image.get("Config") or {}).get("Labels") or {}
    if role == "sim-base":
        if labels.get(ENVIRONMENT_LABEL) != environment_identity(root, role):
            raise ValueError("sim-base: dependency declarations differ")
    else:
        validate_environment(labels, root, role)
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        raise ValueError(f"{role}: expected linux/amd64 environment")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("Id", ""))):
        raise ValueError(f"{role}: invalid local image identity")
    if image_id is not None and image["Id"] != image_id:
        raise ValueError(f"{role}: pulled image differs from the published image ID")
    return image


def _read_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA
            or value.get("platform") != "linux/amd64"):
        raise ValueError("unsupported image manifest")
    images = value.get("images")
    if not isinstance(images, dict):
        raise ValueError("invalid image declarations")
    for role, entry in images.items():
        if role not in (*IMAGE_ARGS, "sim-base") or not isinstance(entry, dict):
            raise ValueError("unknown image role or invalid image declaration")
        if not isinstance(entry.get("ref"), str) or not DIGEST.fullmatch(entry["ref"]):
            raise ValueError(f"{role}: a pullable registry digest is required")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(entry.get("image_id", ""))):
            raise ValueError(f"{role}: missing image ID")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("environment", ""))):
            raise ValueError(f"{role}: missing environment identity")
    return value


def _validate_environments(root: Path, images: dict) -> None:
    for role, entry in images.items():
        if entry["environment"] != environment_identity(root, role):
            raise ValueError(f"{role}: dependency declarations changed; use --build or a matching manifest")


def load_manifest(path: Path, root: Path, profile: str = "all") -> dict:
    value = _read_manifest(path)
    if not set(PROFILES[profile]).issubset(value["images"]):
        raise ValueError(f"image manifest does not cover {profile}")
    _validate_environments(root, {role: value["images"][role] for role in PROFILES[profile]})
    return value


def _read_selection(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    images = value.get("images") if isinstance(value, dict) else None
    if not isinstance(images, dict) or any(
            role not in IMAGE_ARGS or not isinstance(ref, str) or not PINNED_IMAGE.fullmatch(ref)
            for role, ref in images.items()):
        raise ValueError(f"invalid installed image selection: {path}")
    return images


def save_selection(path: Path, images: dict[str, str]) -> None:
    selected = _read_selection(path)
    selected.update(images)
    write_json(path, {"images": selected}, overwrite=True)


def install(root: Path, manifest: Path, profile: str, output: Path) -> None:
    value = load_manifest(manifest, root, profile)
    _read_selection(output)
    selected = {}
    for role in PROFILES[profile]:
        entry = value["images"][role]
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", entry["ref"]], check=True)
        image = check_image(root, role, entry["ref"], image_id=entry["image_id"])
        selected[role] = entry["ref"]
        # Low-level developer commands retain their conventional local aliases.
        subprocess.run(["docker", "tag", image["Id"], f"codeaction-{role}:dev"], check=True)
    save_selection(output, selected)


def capture(root: Path, profile: str, output: Path) -> None:
    _read_selection(output)
    images = {role: check_image(root, role, f"codeaction-{role}:dev")["Id"]
              for role in PROFILES[profile]}
    save_selection(output, images)


def publish(root: Path, namespace: str, version: str, output: Path, *, push: bool = False,
            resume: bool = False, base_manifest: Path | None = None,
            roles: list[str] | tuple[str, ...] | None = None) -> dict:
    """The default is a read-only plan. Only --push writes to Docker Hub."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,38}", namespace):
        raise ValueError("namespace must be a Docker Hub user or organization")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", version) or version in {"latest", "dev"}:
        raise ValueError("use a versioned tag, not latest or dev")
    if output.exists() or output.is_symlink():
        raise FileExistsError("image manifest already exists; choose a new version")
    if (base_manifest is None) != (roles is None):
        raise ValueError("use --base-manifest and --roles together")
    selected_roles = tuple(PUBLISH_ROLES if roles is None else roles)
    if (not selected_roles or len(set(selected_roles)) != len(selected_roles)
            or set(selected_roles) - set(PUBLISH_ROLES)):
        raise ValueError("choose distinct, known image roles")
    reused = {}
    if base_manifest is not None:
        previous = _read_manifest(base_manifest)
        reused = {role: entry for role, entry in previous["images"].items() if role not in selected_roles}
        if not set(PUBLISH_ROLES).issubset(set(reused) | set(selected_roles)):
            raise ValueError("base manifest and selected roles must cover all seven published environments")
        _validate_environments(root, reused)
    environments = {role: environment_identity(root, role) for role in selected_roles}
    inspected = {role: check_image(root, role, f"codeaction-{role}:dev") for role in selected_roles}
    destinations = {role: f"docker.io/{namespace}/codeaction-{role}:{version}" for role in selected_roles}
    planned = {role: {**entry, "action": "reuse"} for role, entry in reused.items()}
    planned.update({
        role: {"target": destinations[role], "image_id": inspected[role]["Id"],
               "action": "publish", "environment": environments[role]} for role in selected_roles})
    _validate_environments(root, planned)
    plan = {"version": version, "platform": "linux/amd64", "images": planned}
    if not push:
        return plan
    # Refuse existing tags and ambiguous registry failures before publishing any image.
    existing = {}
    for role, ref in destinations.items():
        result = subprocess.run(["docker", "manifest", "inspect", ref], capture_output=True, text=True)
        if result.returncode == 0:
            if not resume:
                raise ValueError(f"published tag already exists: {ref}")
            remote = json.loads(result.stdout)
            config = remote.get("config") if isinstance(remote, dict) else None
            if not isinstance(config, dict) or config.get("digest") != inspected[role]["Id"]:
                raise ValueError(f"existing tag has different content: {ref}")
            existing[role] = True
            continue
        if not any(term in result.stderr.lower() for term in ("no such manifest", "manifest unknown")):
            raise ValueError(f"cannot establish that {ref} is unused; check login and repository access")
    value = {"schema_version": SCHEMA, "version": version, "platform": "linux/amd64", "images": dict(reused)}
    for role, ref in destinations.items():
        if role in existing:
            subprocess.run(["docker", "pull", "--platform", "linux/amd64", ref], check=True)
            check_image(root, role, ref, image_id=inspected[role]["Id"])
        else:
            subprocess.run(["docker", "tag", inspected[role]["Id"], ref], check=True)
            subprocess.run(["docker", "push", ref], check=True)
        repository = ref.rsplit(":", 1)[0]
        aliases = {repository, repository.removeprefix("docker.io/")}
        candidates = [d for d in inspect(ref).get("RepoDigests", []) if d.split("@")[0] in aliases]
        if len(candidates) != 1:
            raise ValueError(f"cannot resolve the pushed digest for {role}")
        digest = candidates[0]
        if not DIGEST.fullmatch(digest):
            raise ValueError(f"invalid pushed digest for {role}")
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", digest], check=True)
        check_image(root, role, digest, image_id=inspected[role]["Id"])
        value["images"][role] = {"ref": digest, "image_id": inspected[role]["Id"],
                                  "environment": environments[role]}
    _validate_environments(root, value["images"])
    write_json(output, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pull", "capture", "check"):
        command = sub.add_parser(name)
        command.add_argument("--profile", choices=PROFILES, default="all")
        if name != "capture":
            command.add_argument("manifest", type=Path)
        if name != "check":
            command.add_argument("--out", type=Path, default=Path("configs/local/images.json"))
    command = sub.add_parser("publish")
    command.add_argument("--namespace", required=True)
    command.add_argument("--version", required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--base-manifest", type=Path,
                         help="reuse unchanged roles from this published image manifest")
    command.add_argument("--roles", nargs="+", choices=PUBLISH_ROLES,
                         help="roles to publish; requires --base-manifest (default: publish all seven)")
    command.add_argument("--push", action="store_true")
    command.add_argument("--resume", action="store_true",
                         help="reuse existing version tags only when their content IDs match exactly")
    args = parser.parse_args(argv)
    try:
        if args.command == "publish":
            print(json.dumps(publish(args.root, args.namespace, args.version, args.out,
                                     push=args.push, resume=args.resume,
                                     base_manifest=args.base_manifest, roles=args.roles), indent=2))
        elif args.command == "capture":
            capture(args.root, args.profile, args.out)
        elif args.command == "check":
            load_manifest(args.manifest, args.root, args.profile)
            print("image manifest matches the selected dependency environments")
        else:
            install(args.root, args.manifest, args.profile, args.out)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Images: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
