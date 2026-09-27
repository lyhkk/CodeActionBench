"""Resolve robot resource references relative to the installed embodiment directory."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PATH_FIELDS = ("urdf_path", "asset_root_path", "collision_spheres", "usd_path")


def relative_robot_config(value: dict, config_dir: Path) -> dict:
    """Remove old installation prefixes without changing any kinematic/planner parameter."""
    result = copy.deepcopy(value)
    kinematics = result["robot_cfg"]["kinematics"]
    for key in PATH_FIELDS:
        raw = kinematics.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw)
        parts = path.parts
        marker = ("assets", "embodiments", config_dir.name)
        matches = [i for i in range(len(parts) - 2) if parts[i:i + 3] == marker]
        if matches:
            path = Path(*parts[matches[-1] + 3:])
        elif path.is_absolute():
            try:
                path = path.relative_to(config_dir)
            except ValueError as exc:
                raise ValueError(f"{key} is outside the installed embodiment: {raw}") from exc
        if ".." in path.parts:
            raise ValueError(f"{key} escapes the embodiment directory")
        kinematics[key] = path.as_posix()
    return result


def load_robot_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file() and path.name in ("curobo_left.yml", "curobo_right.yml"):
        path = path.with_name(path.stem + "_tmp.yml")
    value = relative_robot_config(yaml.safe_load(path.read_text()), path.parent)
    for key in PATH_FIELDS:
        raw = value["robot_cfg"]["kinematics"].get(key)
        if isinstance(raw, str) and raw:
            target = path.parent / raw
            if not target.exists():
                raise FileNotFoundError(f"robot resource {key} is missing: {target}")
            value["robot_cfg"]["kinematics"][key] = str(target)
    return value
