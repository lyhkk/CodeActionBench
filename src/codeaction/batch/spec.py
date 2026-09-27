"""Resolve one manually selected stability/release stage against the pinned task pack.

Stages are targets, not an automatic pipeline.  A caller launches exactly one stage, reviews its
result, and may later launch a larger target against the same durable batch directory.  Prefix
nesting makes every accepted logical episode from a smaller cumulative target reusable by a larger
one when the task-pack and harness identities still match.  An explicitly named review target is
self-contained and is never treated as a cumulative prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from codeaction.benchmark.taskcard import (
    TASKS_ROOT,
    declared_scene_seeds,
    load_task,
    taskset_profile,
    validate_task_pack,
)


@dataclass(frozen=True)
class StageProfile:
    task_count: int
    episodes_per_task: int
    submission_target: bool = False
    # Where in LEADERBOARD_TASK_ORDER the stage starts. Cumulative stages start at 0; a round
    # stage starts where the previous round stopped, so an operator runs only the new tasks and
    # the already-accepted ones are not re-run.
    task_offset: int = 0
    # A deliberately non-prefix task target.  It is for a reviewed, standalone batch, not an
    # implicit expansion of any earlier stability prefix.
    task_names: tuple[str, ...] | None = None

# Stability grows by prefixes of one explicit order.  The first five retain the existing reference-scaffold
# pilot's first cells and span all currently declared primary categories plus both arm modes.  The
# resolver proves that this list is exactly the pack's leaderboard set, so a task-pack change fails
# closed instead of silently dropping or enrolling a task.
LEADERBOARD_TASK_ORDER = (
    "blocks_ranking_size",
    "handover_mic",
    "scan_object",
    "dump_bin_bigbin",
    "open_laptop_setup_arm",
    "place_cans_plasticbox",
    "handover_block",
    "click_bell",
    "lift_pot",
    "open_microwave",
    "place_bread_basket",
    "press_stapler",
    "move_can_pot",
    "grab_roller_dual_contact",
    "place_mouse_pad",
    "put_bottles_dustbin",
    "rotate_qrcode",
    "beat_block_hammer",
    "pick_diverse_bottles",
    "stack_blocks_three",
    "place_dual_shoes",
    "stack_bowls_three",
    "hanging_mug",
    "place_object_basket",
    "place_bread_skillet",
)


MANUAL_STAGES: Mapping[str, StageProfile] = {
    "stability-1x1": StageProfile(1, 1),
    # Round two: the four tasks stability-5x1 adds over stability-1x1, run on their own so an
    # increment costs four cells per model instead of five.
    "stability-4x1": StageProfile(4, 1, task_offset=1),
    "stability-5x1": StageProfile(5, 1),
    # Standalone contact/stall review selected by the release owner.  Its results retain their
    # own batch provenance when they are reported beside an earlier release prefix.
    "stability-stall-4x1": StageProfile(
        4,
        1,
        task_names=(
            "press_stapler",
            "click_bell",
            "beat_block_hammer",
            "grab_roller_dual_contact",
        ),
    ),
    # Third paid round: the 16 leaderboard tasks not covered by the accepted five-task prefix or
    # the standalone four-task stall review.  Keep this explicit: resolving stability-20x1 here
    # would repeat paid cells because the stall review lives in a separate durable batch.
    "stability-16x1": StageProfile(
        16,
        1,
        task_names=(
            "place_cans_plasticbox", "handover_block", "lift_pot", "open_microwave",
            "place_bread_basket", "move_can_pot", "place_mouse_pad", "put_bottles_dustbin",
            "rotate_qrcode", "pick_diverse_bottles", "stack_blocks_three", "place_dual_shoes",
            "stack_bowls_three", "hanging_mug", "place_object_basket", "place_bread_skillet",
        ),
    ),
    "stability-20x1": StageProfile(20, 1),
    "stability-25x1": StageProfile(25, 1),
    "stability-25x3": StageProfile(25, 3),
    "release-25x3": StageProfile(25, 3, submission_target=True),
}


@dataclass(frozen=True)
class EpisodeTarget:
    task: str
    attempt_index: int
    scene_seed: int

    @property
    def key(self) -> str:
        return f"{self.task}/attempt-{self.attempt_index:03d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "attempt_index": self.attempt_index,
            "scene_seed": self.scene_seed,
        }


@dataclass(frozen=True)
class ExactCellTarget:
    model: str
    task: str
    attempt_index: int
    scene_seed: int

    @property
    def key(self) -> str:
        return f"{self.model}/{self.task}/attempt-{self.attempt_index:03d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "attempt_index": self.attempt_index,
            "scene_seed": self.scene_seed,
        }


@dataclass(frozen=True)
class StageTarget:
    stage: str
    task_pack_root: str
    taskset_version: str
    task_pack_sha256: str
    tasks: tuple[str, ...]
    episodes_per_task: int | None
    episodes: tuple[EpisodeTarget, ...]
    submission_target: bool
    exact_cells: tuple[ExactCellTarget, ...] | None = None
    manifest_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = {
            "stage": self.stage,
            "task_pack": {
                "root": self.task_pack_root,
                "taskset_version": self.taskset_version,
                "sha256": self.task_pack_sha256,
            },
            "tasks": list(self.tasks),
            "episodes_per_task": self.episodes_per_task,
            "total_episodes_per_model": len(self.episodes),
            "submission_target": self.submission_target,
            "episodes": [episode.as_dict() for episode in self.episodes],
        }
        if self.exact_cells is not None:
            value.update({
                "selection": "exact_cells",
                "exact_cells": [cell.as_dict() for cell in self.exact_cells],
                "total_cells": len(self.exact_cells),
                "manifest_sha256": self.manifest_sha256,
            })
        return value


EXACT_CELL_MANIFEST_SCHEMA = "exact-cell-manifest.v1"
_MANIFEST_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


def target_cell_rows(
    target: StageTarget, models: tuple[str, ...] | list[str],
) -> tuple[ExactCellTarget, ...]:
    """Resolve the immutable scored cells without conflating them with dispatch order."""
    selected = tuple(models)
    if target.exact_cells is None:
        return tuple(
            ExactCellTarget(model, episode.task, episode.attempt_index, episode.scene_seed)
            for model in selected for episode in target.episodes
        )
    manifest_models = tuple(dict.fromkeys(cell.model for cell in target.exact_cells))
    if selected != manifest_models:
        raise ValueError(
            "selected models must exactly match first appearance in the exact-cell manifest")
    return target.exact_cells


def resolve_cell_manifest(
    path: Path, *, tasks_root: Path = TASKS_ROOT,
) -> StageTarget:
    """Resolve an operator-authored exact cell list against the pinned task pack."""
    source = Path(path).resolve()
    try:
        raw_bytes = source.read_bytes()
        value = json.loads(raw_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"exact-cell manifest is unreadable: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != EXACT_CELL_MANIFEST_SCHEMA:
        raise ValueError(f"exact-cell manifest must use {EXACT_CELL_MANIFEST_SCHEMA}")
    name = value.get("name")
    if not isinstance(name, str) or not _MANIFEST_NAME.fullmatch(name):
        raise ValueError("exact-cell manifest name must be a safe lowercase identifier")
    raw_cells = value.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("exact-cell manifest cells must be a non-empty list")

    root = Path(tasks_root).resolve()
    pack = validate_task_pack(root)
    # Any task the pack REGISTERS, not only the scored ones. A manifest is a reviewed standalone
    # target (submission_target is False for it), and the agent-axis runner already accepts any
    # registered task -- restricting the manifest to the leaderboard made a calibration card
    # unrunnable through it, which is exactly the card a machinery test should use.
    registered = set(pack["tasks"])
    cells: list[ExactCellTarget] = []
    seen = set()
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping) or set(raw) != {
                "model", "task", "attempt_index", "scene_seed"}:
            raise ValueError(f"exact-cell manifest cell {index} has invalid fields")
        model, task = raw["model"], raw["task"]
        attempt, seed = raw["attempt_index"], raw["scene_seed"]
        if not isinstance(model, str) or not model or "/" in model:
            raise ValueError(f"exact-cell manifest cell {index} has invalid model")
        if not isinstance(task, str) or task not in registered:
            raise ValueError(
                f"exact-cell manifest cell {index} names a task the pack does not register: "
                f"{task!r}")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0 \
                or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"exact-cell manifest cell {index} has invalid attempt or seed")
        card = load_task(task, tasks_root=root)
        seeds = declared_scene_seeds(card)
        if attempt >= int(card["protocol"]["attempts_k"]) or attempt >= len(seeds) \
                or seeds[attempt] != seed:
            raise ValueError(
                f"exact-cell manifest cell {index} disagrees with the task-card seed")
        key = (model, task, attempt)
        if key in seen:
            raise ValueError(f"exact-cell manifest duplicates {model}/{task}/{attempt}")
        seen.add(key)
        cells.append(ExactCellTarget(model, task, attempt, seed))

    episodes: list[EpisodeTarget] = []
    seen_episodes = set()
    for cell in cells:
        key = (cell.task, cell.attempt_index)
        if key not in seen_episodes:
            seen_episodes.add(key)
            episodes.append(EpisodeTarget(cell.task, cell.attempt_index, cell.scene_seed))
    return StageTarget(
        stage=name,
        task_pack_root=str(root),
        taskset_version=str(pack["taskset_version"]),
        task_pack_sha256=str(pack["sha256"]),
        tasks=tuple(dict.fromkeys(cell.task for cell in cells)),
        episodes_per_task=None,
        episodes=tuple(episodes),
        submission_target=False,
        exact_cells=tuple(cells),
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def resolve_stage(stage: str, *, tasks_root: Path = TASKS_ROOT) -> StageTarget:
    """Resolve one named stage; never select or launch a following stage."""
    try:
        profile = MANUAL_STAGES[stage]
    except KeyError as exc:
        raise ValueError(
            f"unknown manual stage {stage!r}; expected one of {list(MANUAL_STAGES)}") from exc

    root = Path(tasks_root).resolve()
    pack = validate_task_pack(root)
    declared_leaderboard = tuple(taskset_profile(root)["scored"]["tasks"])
    if len(declared_leaderboard) != 25:
        raise ValueError(
            "manual benchmark stages require exactly 25 leaderboard tasks; "
            f"task pack declares {len(declared_leaderboard)}")
    if len(LEADERBOARD_TASK_ORDER) != len(set(LEADERBOARD_TASK_ORDER)) or \
            set(LEADERBOARD_TASK_ORDER) != set(declared_leaderboard):
        raise ValueError(
            "frozen stability task order must exactly match the task pack's leaderboard set")
    if profile.task_names is None:
        start = profile.task_offset
        tasks = LEADERBOARD_TASK_ORDER[start:start + profile.task_count]
        if len(tasks) != profile.task_count:
            raise ValueError(
                f"stage {stage} requests tasks [{start}:{start + profile.task_count}] but the frozen "
                f"order declares only {len(LEADERBOARD_TASK_ORDER)}")
    else:
        tasks = profile.task_names
        if len(tasks) != profile.task_count:
            raise ValueError(
                f"stage {stage} declares {len(tasks)} named tasks but requests "
                f"task_count={profile.task_count}")
        if len(tasks) != len(set(tasks)) or not set(tasks) <= set(LEADERBOARD_TASK_ORDER):
            raise ValueError(
                f"stage {stage} declares duplicate or non-leaderboard named tasks: {tasks}")

    episodes: list[EpisodeTarget] = []
    for task in tasks:
        card = load_task(task, tasks_root=root)
        declared_attempts = int(card["protocol"]["attempts_k"])
        if declared_attempts < profile.episodes_per_task:
            raise ValueError(
                f"stage {stage} requests {profile.episodes_per_task} episodes for {task}, "
                f"but its card declares only {declared_attempts}")
        seeds = declared_scene_seeds(card)
        for attempt_index in range(profile.episodes_per_task):
            episodes.append(EpisodeTarget(
                task=task,
                attempt_index=attempt_index,
                scene_seed=seeds[attempt_index],
            ))

    if profile.submission_target:
        invalid = [
            task for task in tasks
            if load_task(task, tasks_root=root)["protocol"]["attempts_k"] != 3
        ]
        if invalid:
            raise ValueError(
                "release-25x3 requires attempts_k=3 for every leaderboard task; "
                f"invalid={invalid}")

    return StageTarget(
        stage=stage,
        task_pack_root=str(root),
        taskset_version=str(pack["taskset_version"]),
        task_pack_sha256=str(pack["sha256"]),
        tasks=tasks,
        episodes_per_task=profile.episodes_per_task,
        episodes=tuple(episodes),
        submission_target=profile.submission_target,
    )
