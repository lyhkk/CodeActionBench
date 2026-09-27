"""Dependency compatibility is independent of task and Python source revisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ENVIRONMENT_LABEL = "org.codeaction.environment-sha256"
LAUNCH_ABI = "snapshot.v1"
ROLES = ("sim", "reference-agent", "gateway", "fixture-agent", "claude-agent", "codex-agent",
         "scratch", "scratch-launcher")


def environment_files(role: str) -> tuple[str, ...]:
    if role not in (*ROLES, "sim-base"):
        raise ValueError(f"unknown environment role: {role}")
    files = [f"docker/{role}.Dockerfile"]
    if role in ("sim", "sim-base"):
        files = ["docker/sim-base.Dockerfile", "docker/requirements.lock", *files]
    if role == "reference-agent":
        files.append("docker/reference-agent.requirements.lock")
    files += [name + ".dockerignore" for name in list(files) if name.endswith(".Dockerfile")]
    return tuple(sorted(set(files)))


def environment_identity(root: Path, role: str) -> str:
    value = {"launch_abi": LAUNCH_ABI, "files": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in environment_files(role)}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_environment(labels: dict, root: Path, role: str | None = None) -> None:
    title = labels.get("org.opencontainers.image.title", "")
    actual_role = title.removeprefix("codeaction-")
    role = role or actual_role
    if actual_role != role or labels.get(ENVIRONMENT_LABEL) != environment_identity(root, role):
        raise ValueError(f"environment {role} is incompatible; build that environment with tools/build_images.sh")


def component(name: str) -> str:
    """Conservative ownership for change reports; shared code never looks task-only."""
    if name in {"docker/images.json", "src/codeaction/image_distribution.py", "tools/images.sh"}:
        return "environment"
    if name in {"config/models.json", "config/endpoints.json", "config/agents.json",
                "config/rate-limits.json"}:
        return "agents"
    if name.startswith("benchmark/tasks/"):
        return "tasks"
    if name.startswith("src/codeaction/verification/") or "/envs/" in name:
        return "scoring"
    if name.startswith(("src/codeaction/interface/", "src/codeaction/runtime/", "src/codeaction/motion/")):
        return "protocol"
    if name.startswith(("src/codeaction/agents/", "src/codeaction/providers/")):
        return "agents"
    if name.startswith("docker/") and (name.endswith(".Dockerfile") or name.endswith(".lock") or name.endswith(".dockerignore")):
        return "environment"
    if name.startswith(("docs/", "tests/")) or name.endswith(".md"):
        return "documentation" if not name.startswith("tests/") else "tests"
    if name.startswith("src/codeaction/reporting/"):
        return "reporting"
    return "shared"
