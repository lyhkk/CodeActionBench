"""Repository-wide, simulator-free validation for the release task-pack contract."""
from __future__ import annotations

import json
from pathlib import Path

from codeaction.contracts.identity import (budget_identity, build_identity_from_card, comparison_key,
                              sha256_json)
from codeaction.interface.instructions import (SHARED_FRAGMENT_IDS, instruction_reference,
                                  reference_instruction_surface, vendor_instruction_surface)
from codeaction.benchmark.taskcard import (
    TASKS_ROOT, declared_scene_seeds, instruction_for_scene_seed, load_task,
    taskset_profile, validate_task_pack,
)
from codeaction.interface.tool_surface import (INTERFACE_PROFILES, INTERFACE_REFERENCE,
                                  INTERFACE_VENDOR_DIRECT, INTERFACE_VENDOR_GATEWAY,
                                  assert_surface_preflight, surface_identity)


def _instruction_summary(surface):
    return {
        key: surface[key] for key in (
            "instruction_contract_sha256", "instruction_surface_sha256",
            "fragment_ids", "fragment_manifest")
    }


EXPECTED_TASKSET_VERSION = "0.34.0"


def validate_repository_contract(tasks_root=TASKS_ROOT) -> dict:
    root = Path(tasks_root).resolve()
    pack = validate_task_pack(root)
    # The one place a task-pack bump must be reflected. Kept explicit rather than derived so a
    # pack change is a deliberate edit here, not something that silently rides along. The message
    # quotes the constant: it previously named 0.27.0 while the check demanded 0.28.0, so a real
    # mismatch would have reported the wrong required version.
    if pack["taskset_version"] != EXPECTED_TASKSET_VERSION:
        raise ValueError(
            f"harness-only physical-time task pack must be version {EXPECTED_TASKSET_VERSION}, "
            f"got {pack['taskset_version']}")

    surfaces = {
        profile: surface_identity(profile, hybrid=True)
        for profile in INTERFACE_PROFILES
    }
    base_hashes = {surface["base_tool_set_sha256"] for surface in surfaces.values()}
    if len(base_hashes) != 1:
        raise ValueError("interface base tool hashes differ")
    if "bash_exec" in surfaces[INTERFACE_REFERENCE]["ordered_names"] or \
            "bash_exec" in surfaces[INTERFACE_VENDOR_DIRECT]["ordered_names"]:
        raise ValueError("release interface unexpectedly exposes bash_exec")
    if surfaces[INTERFACE_VENDOR_GATEWAY]["extras"] != ["bash_exec"] or \
            surfaces[INTERFACE_VENDOR_GATEWAY]["submittable"]:
        raise ValueError("gateway extra/profile declaration is invalid")
    if any("get_task" in surface["ordered_names"] for surface in surfaces.values()):
        raise ValueError("retired get_task remains on an active model-facing surface")
    for surface in surfaces.values():
        assert_surface_preflight(surface, json.loads(json.dumps(surface)))

    card_reports = []
    for task_name in pack["tasks"]:
        card = load_task(task_name, tasks_root=root)
        if card["schema_version"] != "0.5":
            raise ValueError(f"{task_name} did not migrate to task-card 0.5")
        if card["instructions"] != instruction_reference():
            raise ValueError(f"{task_name} instruction reference drifted")
        budgets = card["budgets"]
        budget = budget_identity(budgets)
        max_calls = int(budgets["max_tool_calls"])
        run_code_calls = int(budgets["run_code_max_internal_calls"])
        seeds = declared_scene_seeds(card)
        task_instruction = instruction_for_scene_seed(card, seeds[0])
        reference_surface = reference_instruction_surface(
            task_text=task_instruction,
            max_tool_calls=max_calls,
            physical_time_budget_s=float(budgets["physical_time_budget_s"]),
            run_code_max_internal_calls=run_code_calls,
        )
        vendor_surface = vendor_instruction_surface(
            task_text=task_instruction,
            max_tool_calls=max_calls,
            physical_time_budget_s=float(budgets["physical_time_budget_s"]),
            run_code_max_internal_calls=run_code_calls,
        )
        if vendor_surface["episode_config"]["task"] != task_instruction:
            raise ValueError(
                f"{task_name} initial episode configuration changed the official instruction")
        if reference_surface["instruction_contract_sha256"] != \
                vendor_surface["instruction_contract_sha256"]:
            raise ValueError(f"{task_name} shared instruction contract differs by track")
        if reference_surface["fragment_ids"] != list(SHARED_FRAGMENT_IDS) or \
                vendor_surface["fragment_ids"] != list(SHARED_FRAGMENT_IDS):
            raise ValueError(f"{task_name} shared instruction fragments differ")

        environment = {
            "runtime": "fixture",
            "source_commit": "0" * 40,
            "asset_manifest_sha256": sha256_json(card["scene"].get("asset_pins") or {}),
            "embodiment": card["scene"].get("embodiment"),
            "task_config_sha256": sha256_json(card["scene"]),
        }
        common = {
            "card": card,
            "pack_info": pack,
            "environment": environment,
            "source_commit": "0" * 40,
            "declared_scene_seeds": seeds,
        }
        first = build_identity_from_card(
            **common,
            tool_surface=surfaces[INTERFACE_REFERENCE],
            instruction_surface=_instruction_summary(reference_surface),
            tested_unit={
                "interface_profile": INTERFACE_REFERENCE,
                "driver": {"kind": "reference_scaffold", "id": "codeaction-reference",
                           "version": "0.6.0", "image_digest": None,
                           "config_sha256": "1" * 64},
            },
            model={"provider": "fixture", "id": "fixture", "reasoning": "none",
                   "temperature": 0, "requested_output_tokens": 1,
                   "effective_output_tokens": 1},
            scene_seed=seeds[0],
            attempt_index=0,
        )
        second_index = 1 if len(seeds) > 1 else 0
        second = build_identity_from_card(
            **common,
            tool_surface=surfaces[INTERFACE_REFERENCE],
            instruction_surface=_instruction_summary(reference_surface),
            tested_unit=first["comparison"]["tested_unit"],
            model=first["comparison"]["model"],
            scene_seed=seeds[second_index],
            attempt_index=second_index,
        )
        if first["comparison"] != second["comparison"]:
            raise ValueError(f"{task_name} comparison identity varies by trial")
        if len(seeds) > 1 and first["trial"] == second["trial"]:
            raise ValueError(f"{task_name} trial identity did not vary")
        if comparison_key(first) != comparison_key(second):
            raise ValueError(f"{task_name} comparison key is unstable")

        verifier = card["verifier"]
        if verifier.get("kind") != "env_check_success":
            raise ValueError(
                f"{task_name} release verifier must use the task environment's check_success")
        # The verdict is one end-state read of that predicate, so every release card must also
        # RECORD whether the predicate held earlier in the episode; without it, reached-then-lost
        # and never-reached are indistinguishable in the artifacts.
        events = (verifier.get("latch") or {}).get("events") or []
        if sum(1 for event in events if event.get("type") == "env_check_success") != 1:
            raise ValueError(
                f"{task_name} must declare exactly one env_check_success latch event")
        card_reports.append({
            "task": task_name,
            "scoring": card["metadata"]["scoring"],
            "budget_sha256": budget["sha256"],
            "instruction_contract_sha256": reference_surface["instruction_contract_sha256"],
            "reference_instruction_surface_sha256": reference_surface["instruction_surface_sha256"],
            "vendor_instruction_surface_sha256": vendor_surface["instruction_surface_sha256"],
            "comparison_sha256": comparison_key(first),
        })

    from codeaction.paths import PROJECT_ROOT

    project_root = PROJECT_ROOT
    authored_targets = (
        project_root / "src" / "codeaction" / "agents" / "vendor" / "render_prompt.py",
        project_root / "src" / "codeaction" / "cli" / "main.py",
    )
    forbidden = (
        "You are operating a real dual-arm robot",
        "Load them with ToolSearch, call get_task first",
        "Finish by calling done(report=",
    )
    for path in authored_targets:
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            if fragment in text:
                raise ValueError(f"scaffold instruction literal remains in {path.name}")

    # The scored denominator is derived from the cards, never from a count kept elsewhere: a card
    # enters the leaderboard total only by declaring `scoring: "leaderboard"`.
    scoring = [report["scoring"] for report in card_reports]
    return {
        "ok": True,
        "task_pack": {
            "version": pack["taskset_version"],
            "sha256": pack["sha256"],
            "tasks": len(pack["tasks"]),
            "scored": sum(1 for value in scoring if value == "leaderboard"),
            "calibration_only": sum(1 for value in scoring if value == "calibration_only"),
            # Published by the admission gate itself: difficulty bands, categories, arms and
            # budget totals of the scored set. A leaderboard cannot be read without knowing the
            # spread of the set under it, and a derived profile cannot go stale.
            "profile": taskset_profile(root),
        },
        "base_tool_set_sha256": next(iter(base_hashes)),
        "delivered_tool_sha256": {
            profile: value["delivered_sha256"] for profile, value in surfaces.items()
        },
        "cards": card_reports,
    }


def main() -> int:
    report = validate_repository_contract()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
