"""Validated accepted-attempt selection and batch-level benchmark aggregation.

The durable batch snapshot remains the source of truth.  This module derives two read-only views:

* ``submission_manifest.v1.json`` names the one immutable execution accepted for every accepted cell;
* ``results/submission_results.v1.json`` groups those attempts by task, model, and comparison.

Directory discovery is deliberately absent.  Every accepted path comes from the exact execution
linked by ``batch_state.json`` and is revalidated against its sealed reference-scaffold artifacts.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from codeaction.interface.tool_surface import surface_is_submittable

from codeaction.evidence.artifacts import (
    MANIFEST_NAME,
    ArtifactManifestError,
    verify_artifact_manifest,
)
from codeaction.contracts.failures import failure_of
from codeaction.agents.vendor.clis import VENDOR_AGENT_MODES
from codeaction.contracts.identity import (
    IDENTITY_SCHEMA_VERSION,
    agent_modes_for_driver_kind,
    comparison_key,
    sha256_json,
    validate_trial_against_randomness,
)
from codeaction.verification.metrics import resource_accounting, stall_telemetry
from codeaction.evidence.provenance import validate_provenance
from codeaction.verification.summarize import summarize_records
from codeaction.benchmark.taskcard import (
    SEED_BOUND_TASK_CARD_SCHEMA_VERSIONS,
    TASK_CARD_SCHEMA_VERSION,
)
from codeaction.contracts.version import TRANSCRIPT_SCHEMA_VERSION


SUBMISSION_MANIFEST_NAME = "submission_manifest.v1.json"
SUBMISSION_MANIFEST_SCHEMA = "submission-manifest.v1"
ACCEPTED_EVIDENCE_SCHEMA = "accepted-attempt-evidence.v1"
SUBMISSION_RESULTS_NAME = "submission_results.v1.json"
SUBMISSION_RESULTS_SCHEMA = "submission-results.v1"
_IMAGE_DIGEST = re.compile(r"^(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})$")
_IMAGE_DIGEST_FIELDS = frozenset({
    "sim_image_digest", "agent_image_digest", "gateway_image_digest",
})
# Stages an accepted attempt may carry. "official-eval" and "qualified" are retired names for the
# same verdict this package now calls "submittable"; historical attempts keep them, and dropping
# them here would make those attempts unexportable for no reason.
SUBMITTABLE_STAGES = ("submittable", "official-eval", "qualified")


class BatchResultsError(RuntimeError):
    """Accepted state and immutable execution evidence disagree."""


# What a step observer recorded before task cards carried a physical-time budget. Such a run had
# no budget to enforce and no expert duration to compare against, so the fields are absent rather
# than lost; accepting the shape states that, where demanding the current one would have made every
# pre-budget attempt permanently unexportable.
_PRE_BUDGET_STEP_OBSERVER_FIELDS = {"steps", "callbacks", "errors"}


def _validate_physical_execution(value: Any, failure: Any) -> None:
    record = _require_mapping(value, "result.step_observer")
    required = {
        "steps", "sim_dt", "physical_time_s", "physical_time_budget_s",
        "threshold_physics_steps", "effective_physical_time_threshold_s",
        "budget_reached_step", "tool_boundary_overshoot_steps",
        "tool_boundary_overshoot_s", "expert_sim_duration_s",
        "physical_time_to_expert_ratio", "callbacks", "errors",
    }
    if set(record) == _PRE_BUDGET_STEP_OBSERVER_FIELDS:
        steps = record["steps"]
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise BatchResultsError("result.step_observer has invalid step counters")
        if not isinstance(record["callbacks"], Mapping) \
                or not isinstance(record["errors"], Mapping):
            raise BatchResultsError("result.step_observer callback evidence is invalid")
        if failure is not None and failure.code.value == "physical_time_budget_exhausted":
            raise BatchResultsError(
                "physical-time failure requires the tool-boundary step observer that measured it")
        return
    if set(record) != required:
        raise BatchResultsError(
            "result.step_observer is neither the current tool-boundary schema nor a pre-budget "
            f"observer: missing={sorted(required - set(record))}, "
            f"unknown={sorted(set(record) - required)}")
    steps = record["steps"]
    threshold = record["threshold_physics_steps"]
    overshoot = record["tool_boundary_overshoot_steps"]
    if any(not isinstance(item, int) or isinstance(item, bool) for item in (
            steps, threshold, overshoot)) or steps < 0 or threshold < 1:
        raise BatchResultsError("result.step_observer has invalid step counters")
    numeric = (
        "sim_dt", "physical_time_s", "physical_time_budget_s",
        "effective_physical_time_threshold_s", "tool_boundary_overshoot_s",
        "expert_sim_duration_s", "physical_time_to_expert_ratio",
    )
    if any(isinstance(record[key], bool) or not isinstance(record[key], (int, float))
           or not math.isfinite(float(record[key])) for key in numeric):
        raise BatchResultsError("result.step_observer has invalid numeric evidence")
    dt = float(record["sim_dt"])
    budget = float(record["physical_time_budget_s"])
    expert = float(record["expert_sim_duration_s"])
    if dt <= 0 or budget <= 0 or expert <= 0:
        raise BatchResultsError("result.step_observer durations must be positive")
    expected_threshold = math.floor(budget / dt)
    expected_overshoot = max(0, steps - expected_threshold)
    expected_reached = expected_threshold if steps >= expected_threshold else None
    checks = (
        threshold == expected_threshold,
        overshoot == expected_overshoot,
        record["budget_reached_step"] == expected_reached,
        math.isclose(float(record["physical_time_s"]), round(steps * dt, 6), abs_tol=1e-6),
        math.isclose(float(record["effective_physical_time_threshold_s"]),
                     threshold * dt, abs_tol=1e-9),
        math.isclose(float(record["tool_boundary_overshoot_s"]),
                     round(overshoot * dt, 6), abs_tol=1e-6),
        math.isclose(float(record["physical_time_to_expert_ratio"]),
                     round(float(record["physical_time_s"]) / expert, 6), abs_tol=1e-6),
    )
    if not all(checks):
        raise BatchResultsError("result.step_observer arithmetic is inconsistent")
    if not isinstance(record["callbacks"], Mapping) or not isinstance(record["errors"], Mapping):
        raise BatchResultsError("result.step_observer callback evidence is invalid")
    physically_exhausted = (
        failure is not None
        and failure.code.value == "physical_time_budget_exhausted")
    if physically_exhausted != (steps >= threshold):
        raise BatchResultsError(
            "physical-time failure disagrees with the completed tool-boundary evidence")


@dataclass(frozen=True)
class SubmissionAttempt:
    """One validated manifest entry together with the private source records it names."""

    entry: dict[str, Any]
    run_dir: Path
    attempt_dir: Path
    result: dict[str, Any]
    run_meta: dict[str, Any]
    agent_exit: dict[str, Any]
    tool_records: tuple[dict[str, Any], ...]
    motion_records: tuple[dict[str, Any], ...]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path, *, optional: bool = False) -> dict[str, Any]:
    if optional and not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchResultsError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchResultsError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BatchResultsError(f"cannot read JSONL from {path}: {exc}") from exc
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BatchResultsError(
                f"invalid JSONL record at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise BatchResultsError(f"JSONL record at {path}:{line_number} is not an object")
        records.append(value)
    return tuple(records)


def _parse_schema_version(value: Any) -> tuple[int, int] | None:
    major, _, minor = str(value or "").partition(".")
    if not major.isdigit() or not minor.isdigit():
        return None
    return int(major), int(minor)


def _transcript_schema_readable(observed: Any) -> bool:
    """Accept a transcript this reader understands, not only one stamped with today's version.

    `codeaction/version.py` defines a MINOR bump as a new optional field that leaves old readers
    correct, and reserves MAJOR for changed meaning ("1.x and 2.x trials are readable but never
    aggregate"). Requiring exact equality therefore rejected transcripts that parse correctly here,
    purely for predating the newest MINOR -- which is what stopped the 2.13 attempts from being
    exported at all. A newer MINOR is still refused, because this reader cannot know what it added,
    and a different MAJOR is still refused, because its records mean something else.
    """
    current = _parse_schema_version(TRANSCRIPT_SCHEMA_VERSION)
    parsed = _parse_schema_version(observed)
    if current is None or parsed is None:
        return False
    return parsed[0] == current[0] and parsed[1] <= current[1]


def _validate_current_transcript(attempt_dir: Path, name: str) -> None:
    records = _read_jsonl(attempt_dir / name)
    meta = [record for record in records if record.get("event") == "meta"]
    if not records or len(meta) != 1 or records[0] is not meta[0] \
            or not _transcript_schema_readable(meta[0].get("schema_version")):
        raise BatchResultsError(
            f"{name} must start with exactly one transcript meta record whose schema shares the "
            f"major version of {TRANSCRIPT_SCHEMA_VERSION} and is no newer than it")


def _validate_vendor_transcript(attempt_dir: Path, name: str, *, model: str) -> None:
    """The vendor CLI's own session record, checked on ITS shape rather than ours.

    A reference transcript opens with a meta record we wrote, so its schema version is ours to
    check. A vendor transcript is the CLI's record of one session: it opens with the `init` event
    the CLI emits and closes with the `result` event that ends it. Checking those two, and that
    `init` names the model the batch declared, proves the same three things the meta check proves
    for the reference file -- one complete session, not truncated, of the thing the cell claims.
    """
    records = _read_jsonl(attempt_dir / name)
    init = [record for record in records if record.get("event") == "init"]
    results = [record for record in records if record.get("event") == "result"]
    if not records or len(init) != 1 or records[0] is not init[0]:
        raise BatchResultsError(
            f"{name} must start with exactly one vendor init record")
    if init[0].get("model") is None:
        # A seat whose stream never names the model (Codex emits no init event and no model
        # field) records null here on purpose rather than copying argv into an "observed"
        # slot. The claim then rests on the attestation, which the controller stamped from the
        # launch and which says so: model_evidence = launch. Accepting a null against that is
        # the honest check; demanding a stream-observed model from a CLI that has none rejected
        # every Codex cell of the first 25-task batch after all 25 had physically completed.
        attestation = _read_json(attempt_dir / "vendor_runtime_attestation.json")
        if attestation.get("model_evidence") != "launch":
            raise BatchResultsError(
                f"{name} init names no model and the attestation does not declare launch "
                f"evidence for it")
        if attestation.get("expected_model") != model:
            raise BatchResultsError(
                f"{name} attestation declares model {attestation.get('expected_model')!r}, "
                f"not the declared {model!r}")
    elif init[0].get("model") != model:
        raise BatchResultsError(
            f"{name} init names model {init[0].get('model')!r}, not the declared {model!r}")
    if len(results) != 1 or records[-1] is not results[0]:
        raise BatchResultsError(
            f"{name} must end with exactly one vendor result record; a session that stopped "
            f"without one was truncated")
    if results[0].get("is_error") is True:
        raise BatchResultsError(f"{name} records a failed vendor session")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BatchResultsError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _path_under(
    batch_dir: Path,
    value: str | Path,
    *,
    field: str,
    require_directory: bool,
) -> tuple[Path, str]:
    lexical_root = Path(batch_dir).absolute()
    root = lexical_root.resolve()
    raw = Path(value)
    if raw.is_absolute():
        candidate_root = lexical_root
        try:
            relative = raw.relative_to(lexical_root)
        except ValueError:
            candidate_root = next(
                (ancestor for ancestor in raw.parents
                 if ancestor.resolve(strict=False) == root), None)
            if candidate_root is None:
                raise BatchResultsError(f"{field} escapes batch directory: {raw}")
            relative = raw.relative_to(candidate_root)
    else:
        candidate_root = lexical_root
        relative = raw
    if relative == Path(".") or any(part in {"", ".", ".."} for part in relative.parts):
        raise BatchResultsError(f"{field} must be a non-empty normalized batch-relative path")
    candidate = candidate_root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise BatchResultsError(f"{field} traverses a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BatchResultsError(f"{field} does not exist: {candidate}") from exc
    if resolved != root and root not in resolved.parents:
        raise BatchResultsError(f"{field} escapes batch directory: {candidate}")
    if require_directory and not resolved.is_dir():
        raise BatchResultsError(f"{field} is not a directory: {candidate}")
    if not require_directory and not resolved.is_file():
        raise BatchResultsError(f"{field} is not a file: {candidate}")
    return resolved, relative.as_posix()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BatchResultsError(f"{field} must be an object")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BatchResultsError(f"{field} must be a non-empty string")
    return value


def _validated_image_digests(value: Any, field: str) -> dict[str, str]:
    images = _require_mapping(value, field)
    if set(images) != _IMAGE_DIGEST_FIELDS:
        raise BatchResultsError(f"{field} must contain the exact three runtime image digests")
    invalid = sorted(
        key for key, digest in images.items()
        if not isinstance(digest, str) or not _IMAGE_DIGEST.fullmatch(digest)
    )
    if invalid:
        raise BatchResultsError(f"{field} contains invalid image identities: {invalid}")
    return {key: str(images[key]) for key in sorted(_IMAGE_DIGEST_FIELDS)}


def _validate_specification(specification: Mapping[str, Any]) -> None:
    if not isinstance(specification, Mapping):
        raise BatchResultsError("batch specification must be an object")
    if specification.get("schema_version") != "1.0":
        raise BatchResultsError("unsupported batch specification schema")
    frozen = _require_mapping(
        specification.get("comparison_identity"), "batch_spec.comparison_identity")
    if specification.get("comparison_sha256") != sha256_json(dict(frozen)):
        raise BatchResultsError("batch_spec comparison_sha256 does not match comparison_identity")
    task_pack = _require_mapping(specification.get("task_pack"), "batch_spec.task_pack")
    _require_string(task_pack.get("taskset_version"), "batch_spec.task_pack.taskset_version")
    _require_string(task_pack.get("sha256"), "batch_spec.task_pack.sha256")
    models = specification.get("models")
    if not isinstance(models, list) or not models:
        raise BatchResultsError("batch_spec.models must be a non-empty unique list")
    for index, model in enumerate(models):
        _require_string(model, f"batch_spec.models[{index}]")
    if len(models) != len(set(models)):
        raise BatchResultsError("batch_spec.models must be a non-empty unique list")
    for field in ("source_commit", "interface_profile", "run_profile"):
        _require_string(frozen.get(field), f"comparison_identity.{field}")
    driver_identity = _require_mapping(frozen.get("driver"), "comparison_identity.driver")
    if not driver_identity:
        raise BatchResultsError("comparison_identity.driver must be non-empty")
    submittable = frozen.get("submittable")
    if not isinstance(submittable, bool):
        raise BatchResultsError("comparison_identity.submittable must be boolean")
    images = _validated_image_digests(
        frozen.get("image_digests"), "comparison_identity.image_digests")
    if driver_identity.get("image_digest") != images["agent_image_digest"]:
        raise BatchResultsError(
            "comparison_identity driver image differs from frozen reference-agent image")
    profiles = _require_mapping(
        frozen.get("model_profiles"), "comparison_identity.model_profiles")
    # Every batch model must appear. An agent with no provider call behind it -- a subscription
    # agent has no API key and no per-key window, a scripted agent issues no request at all --
    # appears with an EMPTY object, which says "nothing about a provider request is frozen here"
    # and compares vacuously below. Omitting it instead would make a genuinely missing profile
    # indistinguishable from one that never existed.
    if set(profiles) != set(models):
        raise BatchResultsError(
            "comparison_identity.model_profiles must exactly cover batch models")
    required_projection = {
        "reasoning", "request_profile", "transport_profile", "rate_limit_policy",
    }
    for model in models:
        projection = _require_mapping(
            profiles.get(model), f"comparison_identity.model_profiles.{model}")
        if not projection:
            continue        # explicit "no provider request is frozen for this agent"
        missing = required_projection - set(projection)
        if missing:
            raise BatchResultsError(
                f"frozen model profile {model} lacks fields: {sorted(missing)}")


def _validate_frozen_comparison(
    specification: Mapping[str, Any], comparison: Mapping[str, Any],
    cell_model: str | None = None,
) -> None:
    frozen = _require_mapping(
        specification.get("comparison_identity"), "batch_spec.comparison_identity")
    supported = {
        "source_commit", "interface_profile", "run_profile", "submittable", "driver",
        "model_profiles", "image_digests",
    }
    unknown = set(frozen) - supported
    if unknown:
        raise BatchResultsError(
            f"unsupported frozen comparison identity fields: {sorted(unknown)}")
    environment = _require_mapping(
        comparison.get("environment"), "result.identity.comparison.environment")
    tested_unit = _require_mapping(
        comparison.get("tested_unit"), "result.identity.comparison.tested_unit")
    if frozen.get("source_commit") != environment.get("source_commit"):
        raise BatchResultsError("result source_commit differs from frozen batch identity")
    if frozen.get("interface_profile") != tested_unit.get("interface_profile"):
        raise BatchResultsError("result interface profile differs from frozen batch identity")
    expected_driver = _require_mapping(frozen.get("driver"), "comparison_identity.driver")
    driver = _require_mapping(tested_unit.get("driver"), "result tested_unit.driver")
    images = _validated_image_digests(
        frozen.get("image_digests"), "comparison_identity.image_digests")
    if environment.get("sim_image_digest") != images["sim_image_digest"]:
        raise BatchResultsError("result sim image differs from the frozen batch identity")
    if driver.get("image_digest") != images["agent_image_digest"]:
        raise BatchResultsError("result agent image differs from the frozen batch identity")
    mismatched = [key for key, value in expected_driver.items() if driver.get(key) != value]
    if mismatched:
        raise BatchResultsError(
            f"result driver differs from frozen batch identity: {sorted(mismatched)}")
    expected_profiles = frozen.get("model_profiles")
    if expected_profiles is not None:
        profiles = _require_mapping(expected_profiles, "comparison_identity.model_profiles")
        model = _require_mapping(comparison.get("model"), "result identity model")
        model_id = _require_string(model.get("id"), "result identity model.id")
        # model_profiles is keyed by the BATCH's model key, which on the agent axis is the agent
        # label and not the model id: one model can be reached by two different agents, and the
        # frozen profile belongs to the agent. Callers that know the cell pass it; the id is the
        # fallback for the classic batches where the two happen to be the same string.
        profile_key = cell_model or model_id
        expected_model = _require_mapping(
            profiles.get(profile_key), f"comparison_identity.model_profiles.{profile_key}")
        profile_mismatches = [
            key for key, value in expected_model.items() if model.get(key) != value
        ]
        if profile_mismatches:
            raise BatchResultsError(
                "result model profile differs from frozen batch identity: "
                f"{sorted(profile_mismatches)}")


def _validate_randomness(randomness: Mapping[str, Any]) -> None:
    sealed = randomness.get("sha256")
    unhashed = dict(randomness)
    unhashed.pop("sha256", None)
    if not isinstance(sealed, str) or sealed != sha256_json(unhashed):
        raise BatchResultsError("randomness protocol sha256 is invalid")
    attempts = randomness.get("attempts_k")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise BatchResultsError("randomness protocol attempts_k is invalid")
    if randomness.get("id") != "declared-scene-seeds" \
            or randomness.get("model_seed_policy") != "unsupported":
        raise BatchResultsError("randomness protocol identity is invalid")
    seeds = randomness.get("scene_seeds")
    if not isinstance(seeds, list) or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise BatchResultsError("randomness protocol scene_seeds are invalid")
    policy = randomness.get("seed_policy")
    if policy in ("fixed_scene_seed_attempts", "fixed_scene_seed_repeats"):
        # Historical evidence is validated in its original form and keeps its original hash.
        version, count_key = (("3.0", "attempts_per_seed") if policy == "fixed_scene_seed_attempts"
                              else ("2.0", "repeats_per_seed"))
        attempt_seeds = randomness.get("attempt_scene_seeds")
        backups = randomness.get("backup_scene_seeds")
        if randomness.get("version") != version or len(seeds) != 1 \
                or randomness.get(count_key) != attempts \
                or attempt_seeds != [seeds[0]] * attempts \
                or not isinstance(backups, list) or len(backups) != 1 \
                or not isinstance(backups[0], int) or isinstance(backups[0], bool) \
                or backups[0] < 0 or backups[0] == seeds[0]:
            raise BatchResultsError("fixed-scene randomness protocol is inconsistent")
    elif policy is None:
        if randomness.get("version") != "1.0" or len(seeds) != attempts \
                or len(seeds) != len(set(seeds)):
            raise BatchResultsError("distinct-scene randomness protocol is inconsistent")
    else:
        raise BatchResultsError("randomness protocol seed_policy is invalid")


def _artifact_result_sha256(attempt_dir: Path) -> str:
    manifest = _read_json(attempt_dir / MANIFEST_NAME)
    matches = [
        entry for entry in manifest.get("files", [])
        if isinstance(entry, dict) and entry.get("path") == "result.json"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise BatchResultsError(f"{MANIFEST_NAME} does not contain one hashed result.json")
    return str(matches[0]["sha256"])


def validate_execution_attempt(
    batch_dir: Path,
    specification: Mapping[str, Any],
    cell: Mapping[str, Any],
    run_dir: Path,
    *,
    allow_manual_attestation_acceptance: bool = False,
) -> dict[str, Any]:
    """Validate one sealed reference-scaffold execution and return state-safe accepted evidence."""
    _validate_specification(specification)
    root = Path(batch_dir).resolve()
    run_path, run_relative = _path_under(
        Path(batch_dir), run_dir, field="run_dir", require_directory=True)
    task = _require_string(cell.get("task"), "cell.task")
    # NAMED `agent`, not `model`: the cell key is the AGENT LABEL, and calling it model is what
    # produced a run of mismatches against run.json, the result identity and the provenance --
    # every one of which records the MODEL under that name.
    agent = _require_string(cell.get("model"), "cell.model")
    attempt_index = cell.get("attempt_index")
    scene_seed = cell.get("scene_seed")
    if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) or attempt_index < 0:
        raise BatchResultsError("cell.attempt_index must be a non-negative integer")
    if not isinstance(scene_seed, int) or isinstance(scene_seed, bool) or scene_seed < 0:
        raise BatchResultsError("cell.scene_seed must be a non-negative integer")

    run = _read_json(run_path / "run.json")
    if run.get("schema_version") != "0.2":
        raise BatchResultsError("unsupported run.json schema")
    # The agent mode must be the one this batch froze, not literally "reference": a batch of
    # vendor-CLI cells is as legitimate as a reference one, and hardcoding the mode here rejected
    # it outright.
    frozen_driver = _require_mapping(
        (specification.get("comparison_identity") or {}).get("driver"),
        "comparison_identity.driver")
    try:
        expected_modes = agent_modes_for_driver_kind(str(frozen_driver.get("kind")))
    except ValueError as exc:
        raise BatchResultsError(str(exc)) from None
    # A kind can admit more than one mode -- every vendor CLI is a vendor agent runtime -- so
    # this guard keeps a reference batch from being filled with vendor cells and leaves telling
    # the two vendor seats apart to driver.id, which the frozen identity also carries.
    if run.get("agent_mode") not in expected_modes:
        raise BatchResultsError(
            f"run agent_mode {run.get('agent_mode')!r} is not one the batch driver admits "
            f"({list(expected_modes)})")
    # Everything below asks the same question the seven `== "claude"` tests used to ask badly:
    # did a vendor CLI drive this cell, or our own reference scaffold? Which vendor CLI is
    # settled by driver.id and by the identity comparison, never by re-testing a mode name here.
    vendor_driven = run.get("agent_mode") in VENDOR_AGENT_MODES
    if run.get("attempts") != 1:
        raise BatchResultsError("one batch execution must contain exactly one attempt")
    # The cell's key is the AGENT, which is the agent label and not the model id -- one model can
    # be reached by two different agents. `model` is the fallback for runs recorded before the
    # label existed, where the two were the same string.
    run_agent = run.get("agent_label") or run.get("model")
    if run.get("task_name") != task or run_agent != agent:
        raise BatchResultsError(
            f"run.json task/agent ({run.get('task_name')!r}/{run_agent!r}) differs from the "
            f"logical cell ({task!r}/{agent!r})")
    frozen = _require_mapping(
        specification.get("comparison_identity"), "batch_spec.comparison_identity")
    frozen_images = _validated_image_digests(
        frozen.get("image_digests"), "comparison_identity.image_digests")
    if run.get("profile") != frozen.get("run_profile"):
        raise BatchResultsError("run.json profile differs from the frozen batch identity")
    if run.get("attempt_indices") != [attempt_index]:
        raise BatchResultsError("run.json attempt_indices differs from the logical cell")
    expected_name = f"attempt-{attempt_index:03d}-seed-{scene_seed:06d}"
    candidates = sorted(run_path.glob("attempt-*-seed-*"))
    if len(candidates) != 1 or candidates[0].name != expected_name:
        raise BatchResultsError(
            f"execution must contain only the exact declared attempt {expected_name}")
    attempt_path, attempt_relative = _path_under(
        root, candidates[0], field="attempt_dir", require_directory=True)
    # Which SECOND transcript exists is a property of the driver, not of the batch: the reference
    # scaffold writes the model exchange it drove, and the vendor CLI writes its own. Requiring
    # the reference one of every execution is what rejected a vendor batch whose episodes had run
    # and passed -- the agent_mode check above already accepts such a batch, and acceptance has to
    # agree with it.
    _validate_current_transcript(attempt_path, "transcript.jsonl")
    if vendor_driven:
        _validate_vendor_transcript(attempt_path, "vendor_transcript.jsonl",
                                    model=_require_string(run.get("model"), "run.model"))
    else:
        _validate_current_transcript(attempt_path, "reference_transcript.jsonl")

    status = _read_json(attempt_path / "controller_status.json")
    manual_attestation_override = (
        allow_manual_attestation_acceptance
        and not vendor_driven
        and status.get("state") == "failed"
        and status.get("result_present") is True
        and status.get("lifecycle_error") == "reference-agent transcript attestation failed"
        and status.get("reference_agent_attestation") == "mismatch"
    )
    if (status.get("state") != "complete" and not manual_attestation_override) \
            or status.get("result_present") is not True:
        raise BatchResultsError("controller did not complete the attempt with a result")
    if status.get("compose_up_exit") != 0 or status.get("compose_down_exit") != 0:
        raise BatchResultsError("controller did not cleanly start and stop the isolated execution")
    if status.get("attempt_index") != attempt_index or status.get("seed") != scene_seed:
        raise BatchResultsError("controller status trial differs from the logical cell")
    if status.get("lifecycle_error") and not manual_attestation_override:
        raise BatchResultsError("controller status contains a lifecycle_error")
    # Same split for the harness attestation: each driver proves its own agent runtime, and the
    # OTHER field must read not_applicable rather than be ignored, so a run recorded under one
    # driver cannot be accepted as the other.
    driver_attestation = ("vendor_runtime_attestation" if vendor_driven
                          else "reference_agent_attestation")
    idle_attestation = ("reference_agent_attestation" if vendor_driven
                        else "vendor_runtime_attestation")
    for field in (
            "controller_identity_attestation", "filesystem_audit", "gateway_attestation",
            driver_attestation):
        expected = "clean" if field == "filesystem_audit" else "healthy"
        if field == driver_attestation and manual_attestation_override:
            expected = "mismatch"
        if status.get(field) != expected:
            raise BatchResultsError(f"controller status {field} is not {expected}")
    if status.get(idle_attestation) not in (None, "not_applicable"):
        raise BatchResultsError(
            f"controller status {idle_attestation} is {status.get(idle_attestation)!r} on a "
            f"{run.get('agent_mode')} execution; the run was not driven by the batch's "
            f"driver")

    result = _read_json(attempt_path / "result.json")
    if "failure" not in result:
        raise BatchResultsError("a current result must carry an explicit failure field")
    # The interface a result records is the driver's own: the reference scaffold reports the
    # agent it ran, the vendor CLI reports the MCP agent it is. One name was hardcoded here.
    expected_interface = "mcp-agent" if vendor_driven else "reference-agent"
    if result.get("task_name") != task or result.get("interface") != expected_interface:
        raise BatchResultsError(
            f"result task/interface ({result.get('task_name')!r}/{result.get('interface')!r}) "
            f"differs from the logical cell ({task!r}/{expected_interface!r})")
    identity = _require_mapping(result.get("identity"), "result.identity")
    if identity.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise BatchResultsError("accepted result requires the current identity schema")
    comparison = _require_mapping(identity.get("comparison"), "result.identity.comparison")
    trial = _require_mapping(identity.get("trial"), "result.identity.trial")
    task_identity = _require_mapping(comparison.get("task"), "result identity task")
    # The loader itself declares which seed-bound card schemas it supports, and 0.3 differs from
    # 0.4 only by fields that were added (expert_sim_duration_s, physical_time_budget_s). Demanding
    # the newest one here made every attempt run against a still-supported card unexportable.
    if task_identity.get("card_schema_version") not in SEED_BOUND_TASK_CARD_SCHEMA_VERSIONS:
        raise BatchResultsError(
            "accepted result requires a supported seed-bound task-card schema "
            f"({', '.join(SEED_BOUND_TASK_CARD_SCHEMA_VERSIONS)}); newest is "
            f"{TASK_CARD_SCHEMA_VERSION}")
    model_identity = _require_mapping(comparison.get("model"), "result identity model")
    tested_unit = _require_mapping(comparison.get("tested_unit"), "result tested_unit")
    driver = _require_mapping(tested_unit.get("driver"), "result tested_unit.driver")
    if not driver.get("kind"):
        raise BatchResultsError("submission results require identity tested_unit.driver.kind")
    agent_label = _require_string(run.get("agent_label"), "run.agent_label")
    # The result's `model` names the AGENT, which is the label the run was launched under.
    # `driver.id` is a different thing -- the harness or vendor CLI behind it ("codeaction-
    # reference", "claude-code") -- and comparing it to the agent label only ever held because
    # the classic batches left --agent-label at a default that happened to equal the harness id.
    if result.get("model") != agent_label:
        raise BatchResultsError(
            f"result model {result.get('model')!r} is not the agent this run was launched as "
            f"({agent_label!r})")
    _require_string(driver.get("id"), "result tested_unit.driver.id")
    # The identity's model is the MODEL the agent drove, which is not the cell key: the cell key
    # is the agent. The chain that pins it is cell -> agent_label -> run.json.model ->
    # identity.model.id, and every link is checked, here and just above.
    if task_identity.get("id") != task or model_identity.get("id") != run.get("model"):
        raise BatchResultsError(
            f"result identity task/model ({task_identity.get('id')!r}/"
            f"{model_identity.get('id')!r}) differs from the launch "
            f"({task!r}/{run.get('model')!r})")
    if trial.get("attempt_index") != attempt_index or trial.get("scene_seed") != scene_seed:
        raise BatchResultsError("result identity trial differs from the logical cell")
    randomness = _require_mapping(
        comparison.get("randomness_protocol"), "result randomness protocol")
    _validate_randomness(randomness)
    try:
        validate_trial_against_randomness(trial, randomness)
    except (KeyError, TypeError, ValueError) as exc:
        raise BatchResultsError(f"invalid result trial identity: {exc}") from exc
    declared = randomness.get("attempts_k")
    if not isinstance(declared, int) or isinstance(declared, bool) or declared <= attempt_index:
        raise BatchResultsError("result randomness protocol has an invalid attempts_k")

    expected_pack = _require_mapping(specification.get("task_pack"), "batch_spec.task_pack")
    result_pack = _require_mapping(comparison.get("task_pack"), "result identity task_pack")
    if result_pack.get("version") != expected_pack.get("taskset_version") \
            or result_pack.get("sha256") != expected_pack.get("sha256"):
        raise BatchResultsError("result task pack differs from the frozen batch task pack")
    _validate_frozen_comparison(specification, comparison, cell.get("model"))
    if run.get("source_commit") != comparison.get("environment", {}).get("source_commit") \
            or run.get("interface_profile") != tested_unit.get("interface_profile"):
        raise BatchResultsError("run.json source/interface differs from result identity")
    if run.get("task_pack_version") != result_pack.get("version") \
            or run.get("task_pack_sha256") != result_pack.get("sha256"):
        raise BatchResultsError("run.json task pack differs from result identity")

    raw_provenance = _read_json(attempt_path / "provenance.json")
    try:
        provenance = validate_provenance(raw_provenance)
    except (KeyError, TypeError, ValueError) as exc:
        raise BatchResultsError(f"invalid execution provenance: {exc}") from exc
    if provenance.get("schema_version") != "0.2" or provenance.get("runtime") != "docker":
        raise BatchResultsError("accepted batch result requires Docker provenance schema 0.2")
    expected_provenance = {
        "profile": run.get("profile"),
        "task_name": task,
        # The provenance records the MODEL the agent drove, and the agent separately.
        "model": run.get("model"),
        "agent_label": run.get("agent_label"),
        # Same driver-owned name as the result's: hardcoding one of the two made a vendor
        # execution's provenance disagree with itself.
        "interface": expected_interface,
        "interface_profile": tested_unit.get("interface_profile"),
        "seed": scene_seed,
        "attempt_index": attempt_index,
        "source_commit": comparison.get("environment", {}).get("source_commit"),
        "task_pack_version": result_pack.get("version"),
        "task_pack_sha256": result_pack.get("sha256"),
        "expected_identity": identity,
    }
    provenance_mismatches = [
        field for field, expected in expected_provenance.items()
        if provenance.get(field) != expected
    ]
    if provenance_mismatches:
        raise BatchResultsError(
            f"provenance differs from accepted execution: {sorted(provenance_mismatches)}")
    provenance_images = {
        field: provenance.get(field) for field in _IMAGE_DIGEST_FIELDS
    }
    if provenance_images != frozen_images:
        raise BatchResultsError(
            "provenance runtime images differ from the frozen batch identity")

    run_meta = _read_json(attempt_path / "run_meta.json")
    run_meta_images = {
        field: run_meta.get(field) for field in _IMAGE_DIGEST_FIELDS
    }
    if run_meta_images != frozen_images:
        raise BatchResultsError(
            "run_meta runtime images differ from the frozen batch identity")
    meta_identity = run_meta.get("expected_identity") or run_meta.get("identity")
    if meta_identity != identity:
        raise BatchResultsError("run_meta expected identity differs from result identity")
    identity_sha256 = sha256_json(dict(identity))
    attestation = _read_json(attempt_path / "controller_identity_attestation.json")
    if attestation.get("healthy") is not True \
            or attestation.get("expected_identity_sha256") != identity_sha256 \
            or attestation.get("observed_identity_sha256") != identity_sha256:
        raise BatchResultsError("controller identity attestation differs from result identity")
    filesystem_audit = _read_json(attempt_path / "filesystem_audit.json")
    gateway_attestation = _read_json(attempt_path / "gateway_attestation.json")
    # Each driver files its own runtime attestation under its own name; the controller status
    # checked above already agreed which one this execution is expected to have.
    agent_attestation = _read_json(attempt_path / f"{driver_attestation}.json")
    if filesystem_audit.get("outcome") != "clean":
        raise BatchResultsError("filesystem audit is not clean")
    if gateway_attestation.get("healthy") is not True:
        raise BatchResultsError("gateway attestation is not healthy")
    if manual_attestation_override:
        errors = agent_attestation.get("errors")
        if agent_attestation.get("healthy") is not False \
                or not isinstance(errors, list) or not errors \
                or any(not isinstance(error, str) or not error for error in errors):
            raise BatchResultsError(
                "manual acceptance requires a concrete reference-agent attestation mismatch")
    elif agent_attestation.get("healthy") is not True:
        raise BatchResultsError(f"{driver_attestation} is not healthy")
    tool_surface = _require_mapping(comparison.get("tool_surface"), "result tool surface")
    delivered_sha256 = _require_string(
        tool_surface.get("delivered_sha256"), "result tool surface delivered_sha256")
    if gateway_attestation.get("interface_profile") != tested_unit.get("interface_profile") \
            or gateway_attestation.get("expected_delivered_sha256") != delivered_sha256 \
            or gateway_attestation.get("observed_delivered_sha256") != delivered_sha256:
        raise BatchResultsError("gateway attestation differs from the delivered tool surface")
    scaffold_sha256 = _require_string(
        driver.get("config_sha256"), "result tested_unit.driver.config_sha256")
    if vendor_driven:
        # A vendor runtime attests something else, because there is no scaffold config of ours to
        # hash: it attests WHAT RAN -- the model and effort the cell declared, only the
        # benchmark's own MCP server, no tool outside the surface, and a stream it could parse
        # end to end. The delivered tool surface is attested by the gateway, checked above, which
        # is the same evidence the reference branch reads from its own attestation.
        # A seat whose stream names no model at all reports an EMPTY observed list, and an
        # empty list must not read as agreement: the subtraction below only rejects a DIFFERENT
        # model, so the positive claim rests on expected_model, which the seat's own attestation
        # marks as launch-declared rather than observed.
        observed_models = set(agent_attestation.get("observed_assistant_models") or [])
        if agent_attestation.get("expected_model") != run.get("model") \
                or (observed_models - {run.get("model")}) \
                or agent_attestation.get("unexpected_tools") not in ([], None) \
                or agent_attestation.get("invalid_json_lines") not in (0, None) \
                or agent_attestation.get("errors") != []:
            raise BatchResultsError(
                "vendor runtime attestation differs from the declared cell")
    elif agent_attestation.get("expected_scaffold_config_sha256") != scaffold_sha256 \
            or agent_attestation.get("observed_scaffold_config_sha256") != scaffold_sha256 \
            or agent_attestation.get("expected_delivered_sha256") != delivered_sha256 \
            or agent_attestation.get("observed_delivered_sha256") != delivered_sha256 \
            or (not manual_attestation_override and agent_attestation.get("errors") != []):
        raise BatchResultsError("reference-agent attestation differs from the frozen scaffold")

    agent_exit = _read_json(attempt_path / "agent_exit.json", optional=True)
    failure = failure_of(result, agent_exit)
    if failure is not None and not failure.scoreable and not manual_attestation_override:
        raise BatchResultsError(
            f"non-scoreable execution cannot be accepted: {failure.origin.value}/{failure.code.value}")
    verifier = _require_mapping(result.get("verifier"), "result.verifier")
    if not isinstance(verifier.get("success"), bool):
        raise BatchResultsError("accepted result verifier.success must be boolean")
    _validate_physical_execution(result.get("step_observer"), failure)

    release_required = bool(frozen.get("submittable"))
    try:
        artifact_check = verify_artifact_manifest(attempt_path)
    except ArtifactManifestError as exc:
        raise BatchResultsError(f"invalid artifact manifest in {attempt_path}: {exc}") from exc
    if artifact_check.get("reasoning_replay_error"):
        raise BatchResultsError(
            "accepted execution has invalid reasoning replay evidence: "
            f"{artifact_check['reasoning_replay_error']}")
    if not artifact_check.get("integrity_ok") or not artifact_check.get("evidence_complete"):
        raise BatchResultsError(f"attempt evidence is incomplete or changed: {artifact_check}")
    # The tested unit's submittability is DERIVED from the interface it was tested on, not read
    # off it: `_build_tested_unit` deliberately carries no verdict about a run, so demanding a
    # stamped flag there could never be satisfied by any writer -- reference or vendor -- and made
    # this whole release path unreachable. The interface profile is a property OF the tested unit,
    # which is what the check wanted to ask; a non-submittable arm (reference-code-first,
    # vendor-mcp-gateway) is still refused, now for its own reason.
    if release_required and (
            run.get("submittable") is not True
            or run.get("submission_stage") not in SUBMITTABLE_STAGES
            or result.get("submittable") is not True
            or not surface_is_submittable(
                _require_string(tested_unit.get("interface_profile"),
                                "tested_unit.interface_profile"))
            or artifact_check.get("submittable") is not True):
        raise BatchResultsError(
            "accepted execution is not a submittable artifact "
            f"(stage must be one of {', '.join(SUBMITTABLE_STAGES)})")
    result_sha256 = _sha256_file(attempt_path / "result.json")
    if _artifact_result_sha256(attempt_path) != result_sha256:
        raise BatchResultsError("artifact manifest result.json hash disagrees with the file")
    artifact_manifest_sha256 = _sha256_file(attempt_path / MANIFEST_NAME)
    run_sha256 = _sha256_file(run_path / "run.json")

    return {
        "schema_version": ACCEPTED_EVIDENCE_SCHEMA,
        "run_dir": run_relative,
        "attempt_dir": attempt_relative,
        "artifact_manifest": f"{attempt_relative}/{MANIFEST_NAME}",
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "run_sha256": run_sha256,
        "result_sha256": result_sha256,
        "identity_sha256": identity_sha256,
        "comparison_sha256": comparison_key(identity),
    }


def validate_submission_evidence(
    batch_dir: Path,
    specification: Mapping[str, Any],
    cell: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    allow_manual_attestation_acceptance: bool = False,
) -> dict[str, Any]:
    """Re-read an accepted execution and prove its persisted evidence still names it exactly."""
    expected_fields = {
        "schema_version", "run_dir", "attempt_dir", "artifact_manifest",
        "artifact_manifest_sha256", "run_sha256", "result_sha256", "identity_sha256",
        "comparison_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_fields:
        raise BatchResultsError("accepted evidence fields are incomplete or unknown")
    if evidence.get("schema_version") != ACCEPTED_EVIDENCE_SCHEMA:
        raise BatchResultsError("unsupported accepted evidence schema")
    actual = validate_execution_attempt(
        batch_dir,
        specification,
        cell,
        Path(str(evidence.get("run_dir"))),
        allow_manual_attestation_acceptance=allow_manual_attestation_acceptance,
    )
    if dict(evidence) != actual:
        raise BatchResultsError("persisted accepted evidence differs from immutable execution")
    return actual


def _active_target_cells(
    specification: Mapping[str, Any], state: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[str]]:
    history = state.get("stage_history")
    if not isinstance(history, list) or not history:
        raise BatchResultsError("batch_state.stage_history must be non-empty")
    latest = _require_mapping(history[-1], "latest batch stage history entry")
    target = _require_mapping(latest.get("target"), "latest batch stage target")
    active_stage = _require_string(state.get("active_stage"), "batch_state.active_stage")
    if latest.get("stage") != active_stage or target.get("stage") != active_stage:
        raise BatchResultsError("active stage differs from the latest stage target")
    target_pack = _require_mapping(target.get("task_pack"), "active target task_pack")
    frozen_pack = _require_mapping(specification.get("task_pack"), "batch_spec.task_pack")
    if target_pack.get("taskset_version") != frozen_pack.get("taskset_version") \
            or target_pack.get("sha256") != frozen_pack.get("sha256"):
        raise BatchResultsError("active stage task pack differs from the batch specification")
    tasks = target.get("tasks")
    episodes = target.get("episodes")
    episodes_per_task = target.get("episodes_per_task")
    if not isinstance(tasks, list) or not tasks \
            or any(not isinstance(task, str) or not task for task in tasks) \
            or len(tasks) != len(set(tasks)) or not isinstance(episodes, list):
        raise BatchResultsError("active stage target shape is invalid")
    if target.get("selection") == "exact_cells":
        raw_cells = target.get("exact_cells")
        manifest_sha256 = target.get("manifest_sha256")
        if not isinstance(raw_cells, list) or not raw_cells \
                or target.get("total_cells") != len(raw_cells) \
                or not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64 \
                or target.get("submission_target") is not False \
                or episodes_per_task is not None:
            raise BatchResultsError("exact-cell active target shape is invalid")
        normalized_cells = []
        expected_ids = []
        seen = set()
        for index, raw in enumerate(raw_cells):
            item = _require_mapping(raw, f"exact target cell {index}")
            model = _require_string(item.get("model"), f"exact target cell {index}.model")
            task = _require_string(item.get("task"), f"exact target cell {index}.task")
            attempt = item.get("attempt_index")
            seed = item.get("scene_seed")
            if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0 \
                    or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise BatchResultsError(f"exact target cell {index} is invalid")
            identifier = f"{model}/{task}/attempt-{attempt:03d}"
            if identifier in seen:
                raise BatchResultsError(f"exact target duplicates cell {identifier}")
            seen.add(identifier)
            expected_ids.append(identifier)
            normalized_cells.append((model, task, attempt, seed))
        if tasks != list(dict.fromkeys(item[1] for item in normalized_cells)) \
                or list(specification.get("models") or []) != list(dict.fromkeys(
                    item[0] for item in normalized_cells)):
            raise BatchResultsError("exact target task/model order is inconsistent")
        requested = state.get("requested_cells")
        cells = _require_mapping(state.get("cells"), "batch_state.cells")
        if requested != expected_ids or set(cells) != set(expected_ids):
            raise BatchResultsError("requested cells differ from exact-cell active target")
        for identifier, (model, task, attempt, seed) in zip(expected_ids, normalized_cells):
            cell = _require_mapping(cells.get(identifier), f"batch cell {identifier}")
            expected = {
                "model": model, "task": task,
                "attempt_index": attempt, "scene_seed": seed,
            }
            if any(cell.get(field) != value for field, value in expected.items()):
                raise BatchResultsError(f"batch cell differs from active target: {identifier}")
        return target, expected_ids

    if not isinstance(episodes_per_task, int) \
            or isinstance(episodes_per_task, bool) or episodes_per_task < 1:
        raise BatchResultsError("active stage target shape is invalid")
    expected_episode_keys = [
        (task, attempt_index)
        for task in tasks
        for attempt_index in range(episodes_per_task)
    ]
    observed_episode_keys = []
    normalized_episodes = []
    for index, episode_value in enumerate(episodes):
        episode = _require_mapping(episode_value, f"active target episode {index}")
        task = _require_string(episode.get("task"), f"active target episode {index}.task")
        attempt_index = episode.get("attempt_index")
        scene_seed = episode.get("scene_seed")
        if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) \
                or attempt_index < 0 or not isinstance(scene_seed, int) \
                or isinstance(scene_seed, bool) or scene_seed < 0:
            raise BatchResultsError(f"active target episode {index} is invalid")
        observed_episode_keys.append((task, attempt_index))
        normalized_episodes.append((task, attempt_index, scene_seed))
    if observed_episode_keys != expected_episode_keys:
        raise BatchResultsError("active stage episodes are not the exact task/index prefix")
    if target.get("total_episodes_per_model") != len(normalized_episodes) \
            or not isinstance(target.get("submission_target"), bool):
        raise BatchResultsError("active stage episode totals or release target are invalid")
    models = list(specification.get("models") or [])
    expected_ids = [
        f"{model}/{task}/attempt-{attempt_index:03d}"
        for model in models
        for task, attempt_index, _ in normalized_episodes
    ]
    requested = state.get("requested_cells")
    if requested != expected_ids:
        raise BatchResultsError("requested cells differ from the exact active stage target")
    cells = _require_mapping(state.get("cells"), "batch_state.cells")
    if set(cells) != set(expected_ids):
        raise BatchResultsError("batch cells differ from the exact active stage target")
    for identifier, (model, episode) in zip(
            expected_ids,
            ((model, episode) for model in models for episode in normalized_episodes)):
        cell = _require_mapping(cells.get(identifier), f"batch cell {identifier}")
        expected = {
            "model": model,
            "task": episode[0],
            "attempt_index": episode[1],
            "scene_seed": episode[2],
        }
        if any(cell.get(field) != value for field, value in expected.items()):
            raise BatchResultsError(f"batch cell differs from active target: {identifier}")
    return target, expected_ids


# Two operator decisions can put a sealed execution into `accepted`, and they must be verified
# differently. Accepting a harness_attestation_failed execution overrides an attestation, so the
# evidence is re-verified with that override allowed. Accepting an accepted_artifact_invalid
# execution overrides NOTHING: it says the validator that rejected the artifact was itself wrong
# and has since been corrected, so the evidence must re-verify under the CURRENT validator with
# no allowance at all -- the sealed artifacts pass on their own merits or the entry is refused.
# The first 25-task Codex batch is the case: every artifact rejected for a validator defect,
# re-accepted after the fix, and republished only because each one re-verifies strictly.
_MANUAL_ACCEPTANCE_CATEGORIES = {
    "harness_attestation_failed": True,      # attestation override allowed at re-verification
    "accepted_artifact_invalid": False,      # strict re-verification, no override
}


def manual_acceptance_attestation_override(manual, attention, resolution, *, cell_id,
                                           execution_id):
    """Whether this manual acceptance may override an attestation; None if it is not valid."""
    category = manual.get("category")
    if category not in _MANUAL_ACCEPTANCE_CATEGORIES \
            or attention.get("category") != category \
            or attention.get("cell_id") != cell_id \
            or attention.get("execution_id") != execution_id \
            or attention.get("status") != "resolved" \
            or resolution.get("action") != "accept" \
            or manual.get("note") != resolution.get("note"):
        return None
    return _MANUAL_ACCEPTANCE_CATEGORIES[category]


def _accepted_entries(
    batch_dir: Path,
    specification: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_specification(specification)
    if specification.get("batch_id") != state.get("batch_id"):
        raise BatchResultsError("batch specification and state IDs disagree")
    revision = state.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise BatchResultsError("batch state revision must be a positive integer")
    _, expected_ids = _active_target_cells(specification, state)
    cells = _require_mapping(state.get("cells"), "batch_state.cells")
    executions = _require_mapping(state.get("executions"), "batch_state.executions")
    requested = state.get("requested_cells")
    if requested != expected_ids:
        raise BatchResultsError("batch_state requested cells differ from active target")
    requested_set = set(requested)
    if not requested_set <= set(cells):
        raise BatchResultsError("batch_state requested_cells reference unknown cells")

    linked_execution_keys = set()
    entries = []
    identity_hashes = set()
    for identifier in requested:
        cell = _require_mapping(cells[identifier], f"batch cell {identifier}")
        if cell.get("status") != "accepted":
            continue
        execution_id = _require_string(
            cell.get("accepted_execution_id"), f"accepted execution for {identifier}")
        execution_key = f"{identifier}/{execution_id}"
        execution = _require_mapping(
            executions.get(execution_key), f"batch execution {execution_key}")
        if execution.get("status") != "accepted" or execution.get("cell_id") != identifier \
                or execution.get("execution_id") != execution_id:
            raise BatchResultsError(f"accepted execution linkage is invalid: {execution_key}")
        detail = _require_mapping(execution.get("detail"), f"execution detail {execution_key}")
        evidence = _require_mapping(
            detail.get("accepted_attempt"), f"accepted_attempt evidence {execution_key}")
        manual = detail.get("manual_acceptance")
        allow_manual_attestation_acceptance = False
        if manual is not None:
            manual = _require_mapping(manual, f"manual acceptance {execution_key}")
            attention_id = _require_string(
                manual.get("attention_id"), f"manual acceptance {execution_key}.attention_id")
            attention = _require_mapping(
                _require_mapping(state.get("attentions"), "batch_state.attentions").get(attention_id),
                f"manual acceptance attention {attention_id}")
            resolution = _require_mapping(
                attention.get("resolution"), f"manual acceptance resolution {attention_id}")
            override = manual_acceptance_attestation_override(
                manual, attention, resolution, cell_id=identifier, execution_id=execution_id)
            if override is None:
                raise BatchResultsError(
                    f"manual acceptance is not a resolved operator decision: {execution_key}")
            allow_manual_attestation_acceptance = override
        verified = validate_submission_evidence(
            batch_dir,
            specification,
            cell,
            evidence,
            allow_manual_attestation_acceptance=allow_manual_attestation_acceptance,
        )
        identity_hash = verified["identity_sha256"]
        if identity_hash in identity_hashes:
            raise BatchResultsError(f"duplicate accepted result identity: {identity_hash}")
        identity_hashes.add(identity_hash)
        linked_execution_keys.add(execution_key)
        entries.append({
            "cell_id": identifier,
            "execution_id": execution_id,
            "task_name": cell.get("task"),
            "model": cell.get("model"),
            "attempt_index": cell.get("attempt_index"),
            "scene_seed": cell.get("scene_seed"),
            **{key: verified[key] for key in (
                "run_dir", "attempt_dir", "artifact_manifest", "artifact_manifest_sha256",
                "run_sha256", "result_sha256", "identity_sha256", "comparison_sha256")},
        })

    accepted_execution_keys = {
        key for key, execution in executions.items()
        if isinstance(execution, Mapping) and execution.get("status") == "accepted"
    }
    if accepted_execution_keys != linked_execution_keys:
        raise BatchResultsError(
            "accepted execution records must exactly match accepted requested cells")
    accepted_outside_target = [
        identifier for identifier, cell in cells.items()
        if identifier not in requested_set and isinstance(cell, Mapping)
        and cell.get("status") == "accepted"
    ]
    if accepted_outside_target:
        raise BatchResultsError(
            f"accepted cells exist outside the active target: {accepted_outside_target}")
    entries.sort(key=lambda entry: entry["cell_id"])
    return entries


def build_submission_manifest_document(
    batch_dir: Path,
    specification: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete accepted selection for one atomic batch-state snapshot."""
    entries = _accepted_entries(batch_dir, specification, state)
    return {
        "schema_version": SUBMISSION_MANIFEST_SCHEMA,
        "batch_id": specification.get("batch_id"),
        "state_revision": state.get("revision"),
        "active_stage": state.get("active_stage"),
        "batch_comparison_sha256": specification.get("comparison_sha256"),
        "task_pack": copy.deepcopy(dict(specification.get("task_pack") or {})),
        "generated_at": _utc_now(),
        "attempt_count": len(entries),
        "selection_sha256": sha256_json(entries),
        "attempts": entries,
    }


def validate_submission_manifest_document(
    batch_dir: Path,
    specification: Mapping[str, Any],
    state: Mapping[str, Any],
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a persisted selection against current state and every sealed attempt."""
    if not isinstance(document, Mapping) \
            or document.get("schema_version") != SUBMISSION_MANIFEST_SCHEMA:
        raise BatchResultsError("unsupported accepted-attempts document")
    expected = build_submission_manifest_document(batch_dir, specification, state)
    for field in (
            "schema_version", "batch_id", "state_revision", "active_stage",
            "batch_comparison_sha256", "task_pack", "attempt_count", "selection_sha256",
            "attempts"):
        if document.get(field) != expected[field]:
            raise BatchResultsError(f"accepted-attempts document differs from state: {field}")
    if not isinstance(document.get("generated_at"), str) or not document["generated_at"]:
        raise BatchResultsError("accepted-attempts generated_at must be recorded")
    return copy.deepcopy(dict(document))


def _batch_documents(batch_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(batch_dir).resolve()
    return _read_json(root / "batch_spec.json"), _read_json(root / "batch_state.json")


def write_submission_manifest(
    batch_dir: Path,
    specification: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(batch_dir).resolve()
    if specification is None or state is None:
        disk_specification, disk_state = _batch_documents(root)
        specification = disk_specification if specification is None else specification
        state = disk_state if state is None else state
    document = build_submission_manifest_document(root, specification, state)
    _atomic_json(root / SUBMISSION_MANIFEST_NAME, document)
    return document


def _record_for_entry(batch_dir: Path, entry: Mapping[str, Any]) -> SubmissionAttempt:
    root = Path(batch_dir).resolve()
    run_dir, _ = _path_under(
        root, str(entry.get("run_dir")), field="manifest run_dir", require_directory=True)
    attempt_dir, _ = _path_under(
        root, str(entry.get("attempt_dir")), field="manifest attempt_dir", require_directory=True)
    result = _read_json(attempt_dir / "result.json")
    run_meta = _read_json(attempt_dir / "run_meta.json")
    agent_exit = _read_json(attempt_dir / "agent_exit.json", optional=True)
    return SubmissionAttempt(
        entry=copy.deepcopy(dict(entry)),
        run_dir=run_dir,
        attempt_dir=attempt_dir,
        result=result,
        run_meta=run_meta,
        agent_exit=agent_exit,
        tool_records=_read_jsonl(attempt_dir / "transcript.jsonl"),
        motion_records=(
            _read_jsonl(attempt_dir / "tools" / "motion_trace.jsonl")
            if (attempt_dir / "tools" / "motion_trace.jsonl").is_file() else ()),
    )


def load_submission_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], tuple[SubmissionAttempt, ...]]:
    """Load a managed selection and revalidate its state linkage and immutable artifacts."""
    raw = Path(manifest_path)
    if raw.is_dir():
        raw = raw / SUBMISSION_MANIFEST_NAME
    if raw.is_symlink():
        raise BatchResultsError("accepted-attempts manifest must not be a symlink")
    document = _read_json(raw)
    batch_dir = raw.parent.resolve()
    specification, state = _batch_documents(batch_dir)
    validate_submission_manifest_document(batch_dir, specification, state, document)
    records = tuple(_record_for_entry(batch_dir, entry) for entry in document.get("attempts", []))
    return document, records


def build_submission_results_document(
    batch_dir: Path,
    specification: Mapping[str, Any],
    state: Mapping[str, Any],
    accepted_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Group the exact accepted selection by task/model/comparison without path discovery."""
    root = Path(batch_dir).resolve()
    accepted = (
        build_submission_manifest_document(root, specification, state)
        if accepted_document is None else
        validate_submission_manifest_document(root, specification, state, accepted_document)
    )
    return _build_submission_results_from_manifest(
        root, specification, state, accepted)


def _build_submission_results_from_manifest(
    root: Path,
    specification: Mapping[str, Any],
    state: Mapping[str, Any],
    accepted: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate a selection validated in the current call path."""
    records = [_record_for_entry(root, entry) for entry in accepted["attempts"]]
    grouped: dict[tuple[str, str, str], list[SubmissionAttempt]] = {}
    for record in records:
        entry = record.entry
        key = (entry["task_name"], entry["model"], entry["comparison_sha256"])
        grouped.setdefault(key, []).append(record)

    tasks: dict[str, Any] = {}
    for (task, model, comparison_sha256), group in sorted(grouped.items()):
        declared_values = {
            (((record.result.get("identity") or {}).get("comparison") or {})
             .get("randomness_protocol") or {}).get("attempts_k")
            for record in group
        }
        if len(declared_values) != 1:
            raise BatchResultsError(
                f"accepted attempts disagree on declared protocol: {task}/{model}")
        attempts_declared = next(iter(declared_values))
        if not isinstance(attempts_declared, int) or isinstance(attempts_declared, bool) \
                or attempts_declared < 1:
            raise BatchResultsError(f"invalid attempts_k for {task}/{model}")
        summary_entries = [{
            "dir": record.entry["attempt_dir"],
            "result": record.result,
            "run_meta": record.run_meta,
            "agent_exit": record.agent_exit,
            "tool_records": list(record.tool_records),
            "motion_records": list(record.motion_records),
        } for record in group]
        summary = summarize_records(
            summary_entries, attempts_declared=attempts_declared)
        attempt_rows = []
        for record in sorted(group, key=lambda item: item.entry["attempt_index"]):
            verifier = record.result.get("verifier") or {}
            failure = failure_of(record.result, record.agent_exit)
            physical_execution = record.result.get("step_observer")
            stall_analysis = stall_telemetry(
                record.motion_records, verifier)
            resource_usage = resource_accounting(record.result)
            attempt_rows.append({
                **copy.deepcopy(record.entry),
                "success": verifier.get("success") is True,
                "score": verifier.get("score")
                    if isinstance(verifier.get("score"), (int, float)) else None,
                "failure": failure.to_dict() if failure is not None else None,
                "physical_execution": (
                    copy.deepcopy(physical_execution)
                    if isinstance(physical_execution, Mapping) else None),
                "stall_analysis": stall_analysis,
                "resource_usage": resource_usage,
            })
        task_node = tasks.setdefault(task, {"models": {}})
        model_node = task_node["models"].setdefault(model, {"comparisons": {}})
        if comparison_sha256 in model_node["comparisons"]:
            raise BatchResultsError(
                f"duplicate task/model/comparison aggregate: {task}/{model}/{comparison_sha256}")
        model_node["comparisons"][comparison_sha256] = {
            "interface": summary.get("interface"),
            "attempts_declared": attempts_declared,
            "attempts": attempt_rows,
            "summary": summary,
        }

    target, expected_ids = _active_target_cells(specification, state)
    requested = list(state.get("requested_cells") or [])
    attempt_count = len(accepted["attempts"])
    stage_complete = requested == expected_ids and attempt_count == len(expected_ids) and all(
        (state.get("cells") or {}).get(identifier, {}).get("status") == "accepted"
        for identifier in expected_ids
    )
    release_shape = (
        (state.get("active_stage"), target.get("episodes_per_task"))
        in {("release-25x3", 3), ("release-25x5", 5)}
        and target.get("submission_target") is True
        and isinstance(target.get("tasks"), list)
        and len(target["tasks"]) == 25
    )
    protocol_complete = stage_complete and release_shape
    if protocol_complete:
        for task in target["tasks"]:
            task_node = tasks.get(task)
            if not isinstance(task_node, Mapping):
                protocol_complete = False
                break
            for model in specification["models"]:
                model_node = (task_node.get("models") or {}).get(model)
                comparisons = (
                    model_node.get("comparisons")
                    if isinstance(model_node, Mapping) else None)
                if not isinstance(comparisons, Mapping) or len(comparisons) != 1:
                    protocol_complete = False
                    break
                comparison = next(iter(comparisons.values()))
                attempt_rows = comparison.get("attempts") or []
                indices = [row.get("attempt_index") for row in attempt_rows]
                summary = comparison.get("summary") or {}
                if comparison.get("attempts_declared") != target["episodes_per_task"] \
                        or indices != list(range(target["episodes_per_task"])) \
                        or summary.get("protocol_complete") is not True \
                        or summary.get("protocol_valid") is not True:
                    protocol_complete = False
                    break
            if not protocol_complete:
                break
    return {
        "schema_version": SUBMISSION_RESULTS_SCHEMA,
        "batch_id": specification.get("batch_id"),
        "state_revision": state.get("revision"),
        "generated_at": _utc_now(),
        "accepted_selection_sha256": accepted["selection_sha256"],
        "stage": {
            "name": state.get("active_stage"),
            "requested_cells": len(requested),
            "accepted_cells": attempt_count,
            "complete": stage_complete,
        },
        "release_protocol": {
            "tasks": len(target.get("tasks") or []),
            "episodes_per_task": target.get("episodes_per_task"),
            "submission_target": bool(target.get("submission_target")),
            "complete": protocol_complete,
        },
        "tasks": tasks,
    }


def write_submission_results(
    batch_dir: Path,
    specification: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    accepted_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(batch_dir).resolve()
    if specification is None or state is None:
        disk_specification, disk_state = _batch_documents(root)
        specification = disk_specification if specification is None else specification
        state = disk_state if state is None else state
    document = build_submission_results_document(
        root, specification, state, accepted_document=accepted_document)
    _atomic_json(root / "results" / SUBMISSION_RESULTS_NAME, document)
    return document


def write_batch_result_views(
    batch_dir: Path,
    specification: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write both views from one caller-supplied state snapshot."""
    root = Path(batch_dir).resolve()
    if specification is None or state is None:
        disk_specification, disk_state = _batch_documents(root)
        specification = disk_specification if specification is None else specification
        state = disk_state if state is None else state
    accepted = write_submission_manifest(root, specification, state)
    benchmark = _build_submission_results_from_manifest(
        root, specification, state, accepted)
    _atomic_json(root / "results" / SUBMISSION_RESULTS_NAME, benchmark)
    return {"submission_manifest": accepted, "submission_results": benchmark}
