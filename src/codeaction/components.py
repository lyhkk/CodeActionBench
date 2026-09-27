"""Explicit process file sets. Agents never receive simulator or scoring implementations."""
from __future__ import annotations

import json
from pathlib import Path
import shutil


def prepare_components(source: Path) -> None:
    declared = json.loads((source / "src/codeaction/component_files.json").read_text())
    for role, names in declared.items():
        target = source / "components" / role
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe component file: {name}")
            origin = source / name
            if origin.is_symlink() or not origin.is_file():
                raise ValueError(f"missing declared {role} component file: {name}")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            if destination.suffix in {".sh", ".py"}:
                destination.chmod(0o555)
        # Agent-visible configuration is generated from public declarations only.
        config = target / "config"
        config.mkdir(parents=True, exist_ok=True)
        if role == "reference-agent":
            shutil.copy2(source / "config/models.json", config / "models.json")
            shutil.copy2(source / "config/endpoints.json", config / "endpoints.json")


def compose_mounts(source: Path) -> dict[str, str]:
    values = {"CODEACTION_SIM_CODE": str(source),
              "CODEACTION_BACKEND_CODE": str(source / "backend/robotwin")}
    for role in json.loads((source / "src/codeaction/component_files.json").read_text()):
        values["CODEACTION_" + role.replace("-", "_").upper() + "_CODE"] = str(source / "components" / role)
    return values
