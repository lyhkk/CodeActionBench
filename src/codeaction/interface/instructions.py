"""Single authored source for benchmark-owned scaffold instruction fragments.

Task prose remains in each task package and tool descriptions remain in ``schemas.py`` through the
pinned tool registry.  This module owns only scaffold-level prompt/payload text and exposes exact
surface records for hashing and preflight.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from codeaction.contracts.identity import sha256_json
from codeaction.contracts.harness_parameters import declared_harness_parameters
from codeaction.runtime.composition import declared_composition_contract, model_composition_contract
from codeaction.interface.schemas import GRASP_EVIDENCE_NOTE, NONCONTACT_SCALE_NOTE, RUN_CODE_INTERFACE_NOTE


INSTRUCTION_CONTRACT_ID = "codeaction-shared"
# 4.2.0 delivers the common ActionResult/contact boundary once instead of copying it into every
# motion tool description and return summary.
# 4.3.0 states the boundary-snapshot ACCESS PATH and says joint state is a pull. The old wording
# ("including both arms' joints, EE/TCP, gripper, contact") named the contents of a four-level
# object without ever giving the path, and the delivered tool definitions carry no outputSchema —
# measured 2026-08-13, a model spent three consecutive turns guessing key names
# (`observed_after['arm_tcp_pose']`, `['gripper']`, `['left_arm']`) and then blew the whole 8000-char
# stdout budget dumping the object before giving up on it.
INSTRUCTION_CONTRACT_VERSION = "6.0.0"
CONTROL_POLICY = {
    "done_pre_finalize_max": 1,
}

ACTION_RESULT_INTERFACE_NOTE = (
    "Common ActionResult contract: execution.physics_steps is this call's physics advance; "
    "execution.partial says whether a non-completing call already caused physical side effects "
    "(null means unavailable). Raw observed_before/observed_after are the boundary robot states "
    "for both arms; read one as observed_after.arms.left (or .right), whose fields are ee_pose, "
    "tcp_pose, orientation, opening_m, finger_gap_m, gripper_val, drive_commanded_closed, contact "
    "and read_failures, with frame and units at the top level. Joint qpos/qvel is not in them; "
    "get_robot_state returns it on request. Contact identity does not permit or interrupt a "
    "motion; task intent cannot be inferred from which robot part touched. "
    "When a transport stops because its measured TCP pose made no progress while its target "
    "remained outstanding, "
    "achieved.blocked_in_contact reports whether the commanded arm had robot contact above the "
    "declared impulse threshold at that stall boundary (null means the contact read was "
    "unavailable), achieved.contact_parts names only your own "
    "parts then in contact, and achieved.pose_at_contact is your same-boundary robot state or "
    "null when no contact pose was available. Contact coexisting with a stall does not by itself "
    "prove that contact caused the stall. Any atomic action that returns ABORTED cancels calls "
    "submitted after it in the same assistant turn, including inside run_code; the "
    "episode remains active for a new decision on the next turn."
)


@dataclass(frozen=True)
class Fragment:
    id: str
    version: str
    template: str


FRAGMENTS = {
    "role": Fragment(
        "role", "1.0.0",
        "You control a dual-arm robot through the provided benchmark tools.",
    ),
    "environment_facts": Fragment(
        "environment_facts", "4.2.0",
        "Facts: there is no depth sensor and no object ground truth; any metric value you use "
        "must come from your own tool evidence. A single RGB ray has no scene depth, and coordinate "
        "axes contain no scene geometry. Tool results are typed; `achieved` can differ from "
        "`commanded` — trust `achieved`. Physical safety guards may clamp a step, reject an "
        "out-of-workspace target, or stop execution. " + ACTION_RESULT_INTERFACE_NOTE + " Tool "
        "calls execute strictly sequentially. "
        "Ordinary motion tools command one arm; explicitly paired tools such as "
        "`move_both_delta` and `reach_both_tcp` may command both arms in one synchronized action. "
        "The harness parameters that actually govern this episode are "
        "{harness_parameters_json}.",
    ),
    "budget": Fragment(
        "budget", "1.1.0",
        "This episode has a budget of {max_tool_calls} charged tool calls. Every direct tool call "
        "is charged. `run_code` counts as one charged call; its internal primitive calls are "
        "recorded but not charged.",
    ),
    "control_policy": Fragment(
        "control_policy", "2.0.0",
        "Control calls: `done` is free and may be used once to finalize the attempt. The task, "
        "budget, composition limits, safety rules, and harness parameters are delivered in the "
        "initial episode configuration; no tool call is required to retrieve them.",
    ),
    "tool_interface": Fragment("tool_interface", "3.0.0", RUN_CODE_INTERFACE_NOTE),
    "run_code_limit": Fragment(
        "run_code_limit", "3.0.0",
        "Composition contract (authoritative JSON): {composition_contract_json}.",
    ),
    "noncontact_scale": Fragment(
        "noncontact_scale", "1.0.0", NONCONTACT_SCALE_NOTE),
    "grasp_evidence": Fragment(
        "grasp_evidence", "1.0.0", GRASP_EVIDENCE_NOTE),
    "termination": Fragment(
        "termination", "1.0.0",
        "Respond with one or more tool calls per turn; they execute strictly in order. Finish by "
        "calling `done(report=..., success_claim=...)` when the task is complete or you cannot "
        "proceed.",
    ),
    "task": Fragment("task", "1.0.0", "{task_text}"),
}

# Namespace-free on purpose (3.0.0, 2026-09-03). Both earlier versions named the MCP server in
# the prompt, and the server has been renamed once already: the earlier line said `ap_bench`
# while the tools the model actually saw were `mcp__codeaction__*`, so the one concrete fact the
# sentence carried was the one that had gone stale. A model discovers the tools with ToolSearch
# either way, which is what the sentence is for. This text is concatenated into the VENDOR
# controller prompt, so editing it moves `instruction_surface_sha256` for the vendor path only --
# vendor attempts recorded before and after do not pool. The reference path never carries it.
TRANSPORT_NOTE = Fragment(
    "transport_note", "3.0.0",
    "Use `ToolSearch` to discover the benchmark MCP tools, and use only those tools for the "
    "robot task.",
)
# The sentence above is true of Claude Code and false of Codex: Codex has no ToolSearch and
# shows the whole surface from turn one. Measured on the first Codex episodes, the model spent
# its first three actions searching for a tool called ToolSearch and then dumped ALL_TOOLS --
# 12,232 tokens, truncated by the host -- before touching the robot. The discovery sentence is
# therefore selected by the seat's declared tool_discovery. The 3.0.0 text is byte-identical for
# the deferred seat, so its recorded instruction surface does not move.
TRANSPORT_NOTE_EAGER = Fragment(
    "transport_note", "3.1.0",
    "The benchmark MCP tools are available directly in this session; use only those tools "
    "for the robot task.",
)
TRANSPORT_NOTES = {
    "deferred_toolsearch": TRANSPORT_NOTE,
    "eager_all": TRANSPORT_NOTE_EAGER,
}

SHARED_FRAGMENT_IDS = tuple(FRAGMENTS)
REFERENCE_SYSTEM_FRAGMENT_IDS = (
    "role", "environment_facts", "budget", "control_policy", "tool_interface",
    "run_code_limit", "noncontact_scale", "grasp_evidence", "termination",
)
VENDOR_CONTROLLER_FRAGMENT_IDS = ("role",)
VENDOR_TASK_FRAGMENT_IDS = tuple(fragment_id for fragment_id in SHARED_FRAGMENT_IDS
                         if fragment_id != "role")


def fragment_versions() -> dict:
    return {fragment_id: fragment.version for fragment_id, fragment in FRAGMENTS.items()}


def instruction_reference() -> dict:
    return {
        "contract_id": INSTRUCTION_CONTRACT_ID,
        "contract_version": INSTRUCTION_CONTRACT_VERSION,
        "fragments": fragment_versions(),
    }


def validate_instruction_reference(value: Mapping[str, Any]) -> dict:
    expected = instruction_reference()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError(f"unknown or stale instruction contract: expected {expected}")
    return expected


def instruction_drift(value) -> dict | None:
    """None when the card's pinned instruction contract matches the live code; else both sides.
    Same role as tool_surface.tool_set_drift: eligibility signal, never a runnability gate."""
    expected = instruction_reference()
    if isinstance(value, Mapping) and dict(value) == expected:
        return None
    return {"card": dict(value) if isinstance(value, Mapping) else None, "code": expected}


def _render(fragment_id: str, values: Mapping[str, Any]) -> str:
    fragment = FRAGMENTS[fragment_id]
    return fragment.template.format(**values)


def _values(*, task_text: str, max_tool_calls: int,
            physical_time_budget_s=900.0,
            run_code_max_internal_calls: int, harness_parameters=None,
            composition_contract=None, include_program_workspace=False) -> dict:
    if not isinstance(max_tool_calls, int) or isinstance(max_tool_calls, bool) \
            or max_tool_calls < 1:
        raise ValueError("max_tool_calls must be a positive integer")
    if physical_time_budget_s is not None:
        physical_time_budget_s = float(physical_time_budget_s)
        if not math.isfinite(physical_time_budget_s) or physical_time_budget_s <= 0:
            raise ValueError("physical_time_budget_s must be positive and finite")
    if not isinstance(run_code_max_internal_calls, int) \
            or isinstance(run_code_max_internal_calls, bool) \
            or run_code_max_internal_calls < 1:
        raise ValueError("run_code_max_internal_calls must be a positive integer")
    parameters = (declared_harness_parameters()
                  if harness_parameters is None else dict(harness_parameters))
    composition = (
        declared_composition_contract(
            max_internal_tool_calls=run_code_max_internal_calls)
        if composition_contract is None else dict(composition_contract))
    composition = model_composition_contract(
        composition, include_program_workspace=include_program_workspace)
    if composition.get("max_internal_tool_calls") != run_code_max_internal_calls:
        raise ValueError("composition_contract max_internal_tool_calls disagrees with renderer")
    return {
        "task_text": str(task_text),
        "max_tool_calls": max_tool_calls,
        "run_code_max_internal_calls": run_code_max_internal_calls,
        "harness_parameters_json": json.dumps(
            parameters, sort_keys=True, separators=(",", ":")),
        "composition_contract_json": json.dumps(
            composition, sort_keys=True, separators=(",", ":")),
    }


def render_fragments(fragment_ids, **values) -> list[dict]:
    return [
        {"id": fragment_id, "version": FRAGMENTS[fragment_id].version,
         "text": _render(fragment_id, values)}
        for fragment_id in fragment_ids
    ]


def instruction_contract_payload() -> dict:
    return {fragment_id: fragment.template for fragment_id, fragment in FRAGMENTS.items()}


INSTRUCTION_CONTRACT_SHA256 = sha256_json(instruction_contract_payload())


def reference_instruction_surface(*, task_text: str, max_tool_calls: int,
                    run_code_max_internal_calls: int, physical_time_budget_s=900.0,
                    harness_parameters=None,
                    composition_contract=None, include_program_workspace=False) -> dict:
    values = _values(
        task_text=task_text,
        max_tool_calls=max_tool_calls,
        physical_time_budget_s=physical_time_budget_s,
        run_code_max_internal_calls=run_code_max_internal_calls,
        harness_parameters=harness_parameters,
        composition_contract=composition_contract,
        include_program_workspace=include_program_workspace,
    )
    system_fragments = render_fragments(REFERENCE_SYSTEM_FRAGMENT_IDS, **values)
    task_fragment = render_fragments(("task",), **values)
    records = [
        {
            "channel": "system",
            "source": "reference_agent",
            "fragment_ids": list(REFERENCE_SYSTEM_FRAGMENT_IDS),
            "text_or_payload": "\n".join(item["text"] for item in system_fragments),
        },
        {
            "channel": "user",
            "source": "reference_agent",
            "fragment_ids": ["task"],
            "text_or_payload": task_fragment[0]["text"],
        },
    ]
    return {
        "instruction_contract_sha256": INSTRUCTION_CONTRACT_SHA256,
        "instruction_surface_sha256": sha256_json(records),
        "fragment_ids": list(SHARED_FRAGMENT_IDS),
        "fragment_manifest": [
            {"id": fragment_id, "version": FRAGMENTS[fragment_id].version,
             "applied_by": "reference_agent"}
            for fragment_id in SHARED_FRAGMENT_IDS
        ],
        "records": records,
        "system_prompt": records[0]["text_or_payload"],
        "task_message": records[1]["text_or_payload"],
    }


def vendor_instruction_surface(*, task_text: str, max_tool_calls: int,
                    run_code_max_internal_calls: int, physical_time_budget_s=900.0,
                    harness_parameters=None,
                    composition_contract=None, include_program_workspace=False,
                    tool_discovery: str = "deferred_toolsearch") -> dict:
    try:
        transport_note = TRANSPORT_NOTES[tool_discovery]
    except KeyError:
        raise ValueError(f"unknown tool discovery {tool_discovery!r}; "
                         f"known: {sorted(TRANSPORT_NOTES)}") from None
    values = _values(
        task_text=task_text,
        max_tool_calls=max_tool_calls,
        physical_time_budget_s=physical_time_budget_s,
        run_code_max_internal_calls=run_code_max_internal_calls,
        harness_parameters=harness_parameters,
        composition_contract=composition_contract,
        include_program_workspace=include_program_workspace,
    )
    payload_instruction_ids = tuple(
        fragment_id for fragment_id in VENDOR_TASK_FRAGMENT_IDS if fragment_id != "task")
    task_fragments = render_fragments(payload_instruction_ids, **values)
    task_text_rendered = _render("task", values)
    episode_config = {
        "instruction_contract": instruction_reference(),
        "task": task_text_rendered,
        "budget_tool_calls": max_tool_calls,
        "run_code_max_internal_calls": run_code_max_internal_calls,
        "run_code_internal_charged": False,
        "harness_parameters": (declared_harness_parameters()
                               if harness_parameters is None
                               else dict(harness_parameters)),
        "filesystem_policy": "structurally_denied",
        "control_policy": dict(CONTROL_POLICY),
        "instruction_fragments": task_fragments,
    }
    controller_fragments = render_fragments(VENDOR_CONTROLLER_FRAGMENT_IDS, **values)
    controller_text = "\n".join([
        *[item["text"] for item in controller_fragments],
        transport_note.template,
        "Initial episode configuration (authoritative JSON):",
        json.dumps(episode_config, sort_keys=True, separators=(",", ":")),
    ])
    records = [
        {
            "channel": "controller",
            "source": "controller",
            "fragment_ids": ["role", "transport_note", *VENDOR_TASK_FRAGMENT_IDS],
            "text_or_payload": controller_text,
        },
    ]
    manifest = [
        {"id": "role", "version": FRAGMENTS["role"].version,
         "applied_by": "controller"},
        {"id": "transport_note", "version": transport_note.version,
         "applied_by": "controller"},
    ]
    manifest.extend(
        {"id": fragment_id, "version": FRAGMENTS[fragment_id].version,
         "applied_by": "controller"}
        for fragment_id in VENDOR_TASK_FRAGMENT_IDS
    )
    return {
        "instruction_contract_sha256": INSTRUCTION_CONTRACT_SHA256,
        "instruction_surface_sha256": sha256_json(records),
        "fragment_ids": list(SHARED_FRAGMENT_IDS),
        "fragment_manifest": manifest,
        "records": records,
        "controller_prompt": controller_text,
        "episode_config": episode_config,
    }


def controller_prompt(*, task_text: str, max_tool_calls: int,
                      run_code_max_internal_calls: int, harness_parameters=None,
                      composition_contract=None,
                      tool_discovery: str = "deferred_toolsearch") -> str:
    """Render the complete vendor-agent initial prompt; no later control call completes it."""
    return vendor_instruction_surface(
        task_text=task_text,
        max_tool_calls=max_tool_calls,
        run_code_max_internal_calls=run_code_max_internal_calls,
        harness_parameters=harness_parameters,
        composition_contract=composition_contract,
        tool_discovery=tool_discovery,
    )["controller_prompt"]
