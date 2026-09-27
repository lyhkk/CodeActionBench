"""Static GT-leak audits (spec §11 stage 1; §7 v4). Three layers, none needing the sim:
 1. audit_registry — the exposed tool NAMES contain no deny-listed name.
 2. audit_source — a source file's CODE references no deny-listed symbol. This catches what the name
    audit cannot: a benignly-named tool that reads GT in its body. Comments (after '#') are exempt;
    strings/docstrings are scanned. The audit-definition files themselves are excluded by the caller.
 3. audit_instruction_text — the task instruction is harness surface too: it must carry no metric
    hint (sizes, distances, coordinates) — "20 cm ahead" in the instruction is the same leak as a
    GT tool.
Module-graph isolation (a tool module importing a module that CONTAINS privileged functions) is the
run_code sandbox's job (spec §5), not this static pass — stated, not hidden."""
import re

from codeaction.interface.registry import DENY


def audit_registry(exposed_tools) -> dict:
    """No exposed tool name may be deny-listed. DENY covers object ground truth and the retired
    workspace/depth rungs alike, so this single check replaces the former per-rung comparisons."""
    v = [name for name in exposed_tools if name in DENY]
    return {"clean": not v, "violations": sorted(set(v))}


# Symbols that must never appear in agent-facing tool source (object GT, scene lists, rendered
# depth, privileged accessors, scene-actor handles).
_SOURCE_DENY = (
    "actor_center", "get_object_pose", "get_scene_objects", "get_segmentation",
    "get_depth", "get_point_cloud", "privileged_perception", "env.pot", ".actor",
    "get_contact_point",
)


def audit_source(text: str, filename: str = "") -> dict:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0]
        for tok in _SOURCE_DENY:
            if tok in code:
                hits.append({"file": filename, "line": i, "token": tok})
    return {"clean": not hits, "violations": hits}


# Metric hints in instruction text: "20 cm", "0.3 m", "z=0.74", "[0.25, -0.1, ...]".
_METRIC_HINT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:cm|mm|meters?|m)\b"
    r"|[xyz]\s*=\s*-?\d"
    r"|\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d"
)


def audit_instruction_text(text: str) -> dict:
    hits = [m.group(0) for m in _METRIC_HINT.finditer(text or "")]
    return {"clean": not hits, "violations": hits}
