"""Source roots are independent of the checkout name and the working directory."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("CODEACTION_ROOT", PACKAGE_ROOT.parents[1])).resolve()
REPOSITORY_ROOT = PROJECT_ROOT
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", PROJECT_ROOT / "backend/robotwin")).resolve()
TASKS_ROOT = PROJECT_ROOT / "benchmark/tasks"
DOCKER_ROOT = PROJECT_ROOT / "docker"
ROBOTWIN_CONFIG_ROOT = ROBOTWIN_ROOT / "task_config"
CODEACTION_CONFIG_ROOT = PROJECT_ROOT / "configs/robotwin"


def resolve_source_path(path: str) -> Path:
    """Resolve source citations from both current and historical task cards."""
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source citations must be repository-relative")
    legacy = Path("policy") / "CodeAction"
    if relative.is_relative_to(legacy):
        return PROJECT_ROOT / relative.relative_to(legacy)
    if relative.parts and relative.parts[0] in {"envs", "task_config", "script", "description"}:
        return ROBOTWIN_ROOT / relative
    return PROJECT_ROOT / relative
