"""Rendered primitive-library reference for the code-first interface.

On the default surface every D0 primitive arrives as a model-facing tool schema. On the code-first
surface the same primitives are reachable only inside `run_code`, so their semantics have to arrive
some other way — as a document the model reads with `read_file`.

Single source: the text is rendered from `schemas.TOOL_SPECS`, the same dict the default surface
delivers, so the two arms cannot drift into describing different capabilities. Nothing is added
here beyond formatting — no ordering, no recommendation, no worked example that would amount to a
procedure (§0.1 corollary 4).
"""
from __future__ import annotations

import json
from typing import Iterable, Mapping

from codeaction.interface.schemas import TOOL_SPECS

API_REFERENCE_PATH = "api_reference.md"
API_REFERENCE_VERSION = "1.0.0"

_HEADER = """# Primitive library

These functions are available inside `run_code` and `run_program`. Call them by name with keyword
arguments; each returns the plain Python value documented below. They are the same primitives, with
the same semantics and the same failure modes, that the default interface delivers as tool schemas.

`load_image(obs_id)` returns an observation's RGB as an HxWx3 uint8 numpy array (a copy). `np` and
`math` are already imported. Assign to `result` to return a value from the block.
"""


def _render_parameters(schema: Mapping) -> str:
    properties = (schema or {}).get("properties") or {}
    if not properties:
        return "- (no parameters)\n"
    required = set((schema or {}).get("required") or ())
    lines = []
    for name, spec in properties.items():
        spec = spec or {}
        kind = spec.get("type", "any")
        if spec.get("enum"):
            kind = " | ".join(json.dumps(v) for v in spec["enum"])
        flag = "required" if name in required else "optional"
        description = str(spec.get("description") or "").strip()
        suffix = f" — {description}" if description else ""
        lines.append(f"- `{name}` ({kind}, {flag}){suffix}")
    return "\n".join(lines) + "\n"


def render_api_reference(names: Iterable[str] = None) -> str:
    """Markdown reference for `names` (default: every tool in the pinned specs)."""
    names = list(names) if names is not None else list(TOOL_SPECS)
    unknown = [name for name in names if name not in TOOL_SPECS]
    if unknown:
        raise ValueError(f"no tool spec for: {unknown}")
    blocks = [_HEADER]
    for name in names:
        description, schema = TOOL_SPECS[name]
        blocks.append(f"\n## `{name}`\n\n{description.strip()}\n\n{_render_parameters(schema)}")
    return "".join(blocks)


def library_names(delivered: Iterable[str], *, all_names: Iterable[str] = None) -> list[str]:
    """Primitives reachable ONLY through code on this surface = pinned set minus delivered.

    Control and composition tools are excluded: they are never callable from inside the sandbox.
    """
    from codeaction.interface.schemas import PROGRAM_TOOL_NAMES
    excluded = set(delivered) | set(PROGRAM_TOOL_NAMES) | {"run_code", "done"}
    pool = list(all_names) if all_names is not None else list(TOOL_SPECS)
    return [name for name in pool if name not in excluded]
