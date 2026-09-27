"""Canonical identity helpers for CodeAction trials.

The benchmark hashes structured contracts, never Python source files.  Every string is normalized
to NFC with LF line endings before compact, sorted-key JSON encoding.  Task-card digests use the
same filename/length/content framing as the task-pack digest.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping


IDENTITY_SCHEMA_VERSION = "0.3"
RANDOMNESS_PROTOCOL_ID = "declared-scene-seeds"
RANDOMNESS_PROTOCOL_VERSION_DISTINCT = "1.0"
RANDOMNESS_PROTOCOL_VERSION_FIXED = "3.0"
SEED_POLICY_DISTINCT = "fixed_per_attempt_index"
SEED_POLICY_FIXED = "fixed_scene_seed_attempts"


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", str(value).replace("\r\n", "\n").replace("\r", "\n"))


def canonical_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, Mapping):
        return {normalize_text(str(key)): canonical_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"identity value is not JSON-safe: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def framed_files_sha256(entries: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, content in sorted(entries):
        name = normalize_text(relative).encode("utf-8")
        digest.update(name + b"\0")
        digest.update(str(len(content)).encode("ascii") + b"\0")
        digest.update(content)
    return digest.hexdigest()


def task_card_identity(card_dir: str | Path) -> dict:
    root = Path(card_dir)
    entries = []
    for name in ("task.json", "instruction.md"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"task identity entry must be a regular file: {name}")
        entries.append((name, path.read_bytes()))
    try:
        card = json.loads(entries[0][1].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid task.json while building identity: {exc}") from exc
    task_id = (card.get("task") or {}).get("name")
    schema_version = card.get("schema_version")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task identity requires task.name")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("task identity requires schema_version")
    return {
        "id": task_id,
        "card_schema_version": schema_version,
        "card_sha256": framed_files_sha256(entries),
    }


def budget_identity(budgets: Mapping[str, Any]) -> dict:
    value = canonical_value(dict(budgets))
    version = value.get("contract_version")
    if not isinstance(version, str) or not version:
        raise ValueError("budgets.contract_version is required")
    return {"contract_version": version, "sha256": sha256_json(value)}


def randomness_protocol(card: Mapping[str, Any], scene_seeds: Iterable[int]) -> dict:
    protocol = card.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("task card protocol must be an object")
    attempts_k = protocol.get("attempts_k")
    seeds = list(scene_seeds)
    if not isinstance(attempts_k, int) or isinstance(attempts_k, bool) or attempts_k < 1:
        raise ValueError("protocol.attempts_k must be a positive integer")
    if len(seeds) != attempts_k or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("declared scene seeds must contain attempts_k non-negative integers")
    seed_policy = protocol.get("seed_policy")
    if seed_policy == SEED_POLICY_DISTINCT:
        if len(seeds) != len(set(seeds)):
            raise ValueError("declared scene seeds must be unique")
        fixed_seeds = protocol.get("scene_seeds")
        if fixed_seeds is not None and seeds != fixed_seeds:
            raise ValueError(
                "declared scene seeds disagree with protocol.scene_seeds")
        value = {
            "id": RANDOMNESS_PROTOCOL_ID,
            "version": RANDOMNESS_PROTOCOL_VERSION_DISTINCT,
            "attempts_k": attempts_k,
            "scene_seeds": seeds,
            "model_seed_policy": "unsupported",
        }
    elif seed_policy == SEED_POLICY_FIXED:
        primary = protocol.get("scene_seeds")
        attempts_per_seed = protocol.get("attempts_per_seed")
        backups = protocol.get("backup_scene_seeds")
        if not isinstance(primary, list) or len(primary) != 1 or \
                not isinstance(primary[0], int) or isinstance(primary[0], bool) or \
                primary[0] < 0:
            raise ValueError(
                "fixed-scene randomness requires exactly one primary scene seed")
        if not isinstance(attempts_per_seed, int) or isinstance(attempts_per_seed, bool) or attempts_per_seed < 1:
            raise ValueError("fixed-scene randomness requires positive attempts_per_seed")
        if attempts_k != attempts_per_seed or seeds != [primary[0]] * attempts_per_seed:
            raise ValueError(
                "declared attempt scene seeds disagree with fixed-scene attempt protocol")
        if not isinstance(backups, list) or len(backups) != 1 or \
                not isinstance(backups[0], int) or isinstance(backups[0], bool) or \
                backups[0] < 0 or backups[0] == primary[0]:
            raise ValueError(
                "fixed-scene randomness requires one disjoint backup scene seed")
        value = {
            "id": RANDOMNESS_PROTOCOL_ID,
            "version": RANDOMNESS_PROTOCOL_VERSION_FIXED,
            "attempts_k": attempts_k,
            "seed_policy": seed_policy,
            "scene_seeds": list(primary),
            "attempt_scene_seeds": seeds,
            "attempts_per_seed": attempts_per_seed,
            "backup_scene_seeds": list(backups),
            "model_seed_policy": "unsupported",
        }
    else:
        raise ValueError("unsupported task-card seed policy")
    value["sha256"] = sha256_json(value)
    return value


def trial_identity(*, scene_seed: int, attempt_index: int, model_seed=None,
                   seed_support="unsupported") -> dict:
    if not isinstance(scene_seed, int) or isinstance(scene_seed, bool) or scene_seed < 0:
        raise ValueError("trial scene_seed must be a non-negative integer")
    if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) or attempt_index < 0:
        raise ValueError("trial attempt_index must be a non-negative integer")
    if seed_support == "unsupported" and model_seed is not None:
        raise ValueError("unsupported model seeding requires model_seed=null")
    value = {
        "scene_seed": scene_seed,
        "attempt_index": attempt_index,
        "model_seed": model_seed,
        "seed_support": seed_support,
    }
    return value


def validate_trial_against_randomness(trial: Mapping[str, Any],
                                      declared: Mapping[str, Any]) -> None:
    fixed_scene = declared.get("seed_policy") in (SEED_POLICY_FIXED, "fixed_scene_seed_repeats")
    seeds = (declared.get("attempt_scene_seeds") if fixed_scene
             else declared.get("scene_seeds"))
    attempt_index = trial.get("attempt_index")
    if not isinstance(seeds, list) or not isinstance(attempt_index, int):
        raise ValueError("invalid randomness/trial identity")
    if attempt_index < 0 or attempt_index >= len(seeds):
        raise ValueError("trial attempt_index is outside the declared randomness protocol")
    if trial.get("scene_seed") != seeds[attempt_index]:
        raise ValueError("trial scene_seed disagrees with its declared attempt index")
    # Read sealed version-2 evidence without rewriting its identity or hash.
    if declared.get("seed_policy") == "fixed_scene_seed_repeats":
        repeat_index = trial.get("repeat_index")
        if repeat_index != attempt_index:
            raise ValueError("trial repeat_index disagrees with its declared attempt index")
    elif "repeat_index" in trial:
        raise ValueError("current trial uses attempt_index only")


def make_identity(comparison: Mapping[str, Any], trial: Mapping[str, Any]) -> dict:
    comparison_copy = canonical_value(dict(comparison))
    trial_copy = canonical_value(dict(trial))
    randomness = comparison_copy.get("randomness_protocol")
    if not isinstance(randomness, Mapping):
        raise ValueError("identity.comparison.randomness_protocol is required")
    validate_trial_against_randomness(trial_copy, randomness)
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "comparison": comparison_copy,
        "trial": trial_copy,
    }


_COMPARISON_KEYS = {
    "task", "task_pack", "environment", "tool_surface", "instruction_surface",
    "verifier", "tested_unit", "model", "budgets", "randomness_protocol",
}


def comparison_identity(**components) -> dict:
    missing = sorted(_COMPARISON_KEYS - set(components))
    unknown = sorted(set(components) - _COMPARISON_KEYS)
    if missing or unknown:
        raise ValueError(
            f"comparison identity keys mismatch: missing={missing}, unknown={unknown}")
    value = canonical_value(components)
    for key in _COMPARISON_KEYS:
        if not isinstance(value.get(key), Mapping):
            raise ValueError(f"identity.comparison.{key} must be an object")
    task = value["task"]
    for key in ("id", "card_schema_version", "card_sha256"):
        if not task.get(key):
            raise ValueError(f"identity.comparison.task.{key} is required")
    return value


def build_identity_from_card(
    card: Mapping[str, Any],
    pack_info: Mapping[str, Any],
    *,
    environment: Mapping[str, Any],
    tool_surface: Mapping[str, Any],
    instruction_surface: Mapping[str, Any],
    tested_unit: Mapping[str, Any],
    model: Mapping[str, Any],
    source_commit: str,
    declared_scene_seeds: Iterable[int],
    scene_seed: int,
    attempt_index: int,
) -> dict:
    card_dir = card.get("dir")
    if not isinstance(card_dir, str) or not card_dir:
        raise ValueError("loaded task card must carry its package directory")
    comparison = comparison_identity(
        task=task_card_identity(card_dir),
        task_pack={
            "id": "robotwin-codeaction",
            "version": pack_info["taskset_version"],
            "sha256": pack_info["sha256"],
        },
        environment=dict(environment),
        tool_surface=dict(tool_surface),
        instruction_surface=dict(instruction_surface),
        verifier={
            "kind": card["verifier"]["kind"],
            "config_sha256": sha256_json(card["verifier"]),
            "source_commit": source_commit,
        },
        tested_unit=dict(tested_unit),
        model=dict(model),
        budgets=budget_identity(card["budgets"]),
        randomness_protocol=randomness_protocol(card, declared_scene_seeds),
    )
    return make_identity(
        comparison,
        trial_identity(
            scene_seed=scene_seed,
            attempt_index=attempt_index,
        ),
    )


def comparison_key(identity: Mapping[str, Any]) -> str:
    comparison = identity.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("identity.comparison is required")
    return sha256_json(comparison)


# Fields 0.2 carried inside tested_unit that 0.3 removes.
#
# `track` was a pure projection of driver.kind ("A" for benchmark_reference_scaffold, "B"
# otherwise), so dropping it removes no information: every 0.2 artifact records driver.kind.
# The scoring fields were a POLICY VERDICT about a run, not a property of what was tested; two
# attempts identical in every tested respect must pool regardless of them.
# Named exactly as 0.2 artifacts spell them: the projection reads historical documents, so it
# must not be renamed along with the current vocabulary.
_RETIRED_TESTED_UNIT_FIELDS_0_2 = (   # structure-guard: historical name
    "track", "release_stage", "release_eligible", "release_blockers",   # structure-guard: historical name
    "release_qualification_sha256",   # structure-guard: historical name
)


def project_comparison_to_current(comparison, *, schema_version: str) -> dict:
    """Project one recorded comparison identity onto the current schema.

    The projection is a pure deletion, so an attempt recorded under 0.2 can be compared with one
    recorded under 0.3 without rewriting the stored artifact and without inventing any value.
    """
    if schema_version == IDENTITY_SCHEMA_VERSION:
        return canonical_value(dict(comparison))
    if schema_version != "0.2":
        raise ValueError(
            f"cannot project identity schema {schema_version!r} onto {IDENTITY_SCHEMA_VERSION}")
    value = canonical_value(dict(comparison))
    tested_unit = value.get("tested_unit")
    if not isinstance(tested_unit, Mapping):
        raise ValueError("identity.comparison.tested_unit must be an object")
    driver = tested_unit.get("driver")
    if not isinstance(driver, Mapping) or not driver.get("kind"):
        raise ValueError(
            "a 0.2 comparison without tested_unit.driver.kind cannot be projected: the retired "
            "`track` field would be the only record of which agent stack ran")
    value["tested_unit"] = {
        key: item for key, item in tested_unit.items()
        if key not in _RETIRED_TESTED_UNIT_FIELDS_0_2
    }
    return value


# The agent stack a run reports under tested_unit.driver.kind. Defined here, next to the identity
# they belong to, because BOTH the run that emits one and the batch that freezes one must use the
# same strings -- a batch froze "vendor_agent_cli" against a run that reported
# "vendor_agent_runtime" and the mismatch only surfaced when results were written.
REFERENCE_DRIVER_KIND = "benchmark_reference_scaffold"
VENDOR_DRIVER_KIND = "vendor_agent_runtime"
FIXTURE_DRIVER_KIND = "offline_fixture"

_DRIVER_KIND_BY_AGENT_MODE = {
    "reference": REFERENCE_DRIVER_KIND,
    # Every third-party agent CLI is a vendor agent runtime. The KIND is the class of stack; WHICH
    # stack is `driver.id`, taken from the agent image's own org.codeaction.agent-cli label. That
    # split is why seating a second vendor did not need a second kind -- and why the reverse
    # lookup below returns every mode a kind admits instead of guessing the first one.
    "claude": VENDOR_DRIVER_KIND,
    "codex": VENDOR_DRIVER_KIND,
    "fixture": FIXTURE_DRIVER_KIND,
}


def agent_modes_for_driver_kind(kind: str) -> tuple[str, ...]:
    """Every agent mode a batch frozen on this driver kind could have launched."""
    if kind == "local_agent":
        return ("reference",)
    modes = tuple(
        mode for mode, value in _DRIVER_KIND_BY_AGENT_MODE.items() if value == kind)
    if not modes:
        raise ValueError(f"unknown driver kind: {kind!r}")
    return modes


def agent_mode_for_driver_kind(kind: str) -> str:
    """The single agent mode this kind admits.

    Raises when the kind admits more than one, rather than returning whichever was declared
    first: a caller that needs one answer from an ambiguous kind is asking the wrong question and
    should compare `driver.id`.
    """
    modes = agent_modes_for_driver_kind(kind)
    if len(modes) != 1:
        raise ValueError(
            f"driver kind {kind!r} admits {list(modes)}; compare driver.id instead")
    return modes[0]


def driver_kind_for_agent_mode(agent_mode: str) -> str:
    """The class of agent stack this mode runs; the offline fixture is its own kind."""
    try:
        return _DRIVER_KIND_BY_AGENT_MODE[agent_mode]
    except KeyError:
        raise ValueError(f"unknown agent mode: {agent_mode!r}") from None


def agent_identity(tested_unit, model) -> dict:
    """The unit this benchmark measures: one agent is one (driver, model) pair.

    Derived, never stored: both halves already live in the comparison identity, and a stored copy
    would repeat the mistake the retired `track` field made.
    """
    driver = tested_unit.get("driver")
    if not isinstance(driver, Mapping) or not driver.get("kind") or not driver.get("id"):
        raise ValueError("agent identity requires tested_unit.driver.kind and .id")
    if not model.get("id"):
        raise ValueError("agent identity requires model.id")
    return {
        "driver_kind": normalize_text(str(driver["kind"])),
        "driver_id": normalize_text(str(driver["id"])),
        "model_id": normalize_text(str(model["id"])),
        "reasoning": normalize_text(str(model.get("reasoning") or "unspecified")),
    }


def agent_label(agent) -> str:
    """One readable, stable name per agent, e.g. `claude-code/claude-opus-5@effort-high`."""
    return f"{agent['driver_id']}/{agent['model_id']}@{agent['reasoning']}"
