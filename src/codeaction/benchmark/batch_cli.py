#!/usr/bin/env python3
"""Manually launch exactly one durable official reference-scaffold benchmark stage.

This file is a thin host control-plane CLI.  Episode processes are owned exclusively by
``BatchController``, which launches ``codeaction.py run`` and isolated Compose projects.  The
CLI never invokes a bare-metal episode driver and never advances to another stage.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from codeaction.paths import PROJECT_ROOT, REPOSITORY_ROOT, TASKS_ROOT

_RT = PROJECT_ROOT
_DATA = Path(os.environ.get("CODEACTION_RUNS_ROOT", PROJECT_ROOT / "runs")).resolve()

from codeaction.benchmark.agents import reference_agent  # noqa: E402
from codeaction.batch.controller import (  # noqa: E402
    ControllerConfig,
    BatchController,
    build_execution_plan,
    validate_controller_identity,
)
from codeaction.batch.results import BatchResultsError, validate_submission_evidence  # noqa: E402
from codeaction.batch.controller import (  # noqa: E402
    build_comparison_identity, inspect_official_images as _inspect_official_images)
from codeaction.batch.scheduler import BatchScheduler  # noqa: E402
from codeaction.batch.spec import (  # noqa: E402
    MANUAL_STAGES,
    StageTarget,
    resolve_cell_manifest,
    resolve_stage,
    target_cell_rows,
)
from codeaction.batch.state import (  # noqa: E402
    BATCH_SPEC_SCHEMA_VERSION,
    BATCH_STATE_SCHEMA_VERSION,
    BatchStateError,
    BatchStateStore,
    cell_id as durable_cell_id,
)
from codeaction.contracts.identity import sha256_json  # noqa: E402
from codeaction.providers.model_registry import (  # noqa: E402
    RegistryError,
    known_models,
    read_credential_file,
    resolve_model,
)
from codeaction.providers.provider_runtime import (  # noqa: E402
    ProviderRuntimeConfigError,
    load_rate_limit_config,
    rate_limit_policy_for,
    strict_rate_limit_coverage,
)
from codeaction.benchmark.matrix import (  # noqa: E402
    DEFAULT_CREDENTIAL_CONCURRENCY,
    DEFAULT_MODELS,
    _git_preflight,
    model_credentials,
    parse_credential_limits,
    parse_reasoning_profiles,
)


INTERFACE_PROFILE = "reference-mcp"
RUN_PROFILE = "eval"
_IMAGE_DIGEST = re.compile(r"^(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})$")
_IMAGE_DIGEST_FIELDS = (
    "sim_image_digest", "agent_image_digest", "gateway_image_digest",
)
DEFAULT_IMAGE_REFS = {
    "sim_image_digest": "codeaction-sim:dev",
    "agent_image_digest": "codeaction-reference-agent:dev",
    "gateway_image_digest": "codeaction-gateway:dev",
}
_SUBPROCESS_ENV_ALLOWLIST = (
    "CODEACTION_ASSETS_ROOT",
    "CODEACTION_RELEASE_MANIFEST",
    "PATH",
    "HOME",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
    "DOCKER_API_VERSION",
    "SSH_AUTH_SOCK",
    "XDG_RUNTIME_DIR",
)


class BatchLaunchError(RuntimeError):
    """A read-only preflight failed before durable state or a GPU lease was acquired."""


@dataclass(frozen=True)
class PreparedBatch:
    repo_root: Path
    batch_dir: Path
    source_commit: str
    source_snapshot: Mapping[str, str]
    target: StageTarget
    models: tuple[str, ...]
    gpus: tuple[int, ...]
    credentials: Mapping[str, str | None]
    credential_limits: Mapping[str, int]
    reasoning_profiles: Mapping[str, str]
    rate_limit_policies: Mapping[str, Mapping[str, Any]]
    comparison_identity: Mapping[str, Any]
    image_identity: Mapping[str, str]
    config: ControllerConfig
    existing_state: Mapping[str, Any] | None


def _read_object(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink():
        raise BatchLaunchError(f"{field} must not be a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchLaunchError(f"cannot read valid {field}: {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise BatchLaunchError(f"{field} must be a JSON object: {path}")
    return value


def _require_directory(path: Path, field: str) -> Path:
    raw = Path(path).absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise BatchLaunchError(f"{field} must be an existing non-symlink directory: {raw}")
    return raw.resolve()


def _require_regular_file(path: Path, field: str) -> Path:
    raw = Path(path).absolute()
    if raw.is_symlink():
        raise BatchLaunchError(f"{field} must not be a symlink: {raw}")
    try:
        mode = raw.stat().st_mode
    except OSError as exc:
        raise BatchLaunchError(f"{field} is not readable: {raw}") from exc
    if not stat.S_ISREG(mode):
        raise BatchLaunchError(f"{field} must be a regular file: {raw}")
    return raw.resolve()


def _resolve_python(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise BatchLaunchError("--python must be non-empty")
    candidate = Path(value) if "/" in value or "\\" in value else None
    if candidate is None:
        found = shutil.which(value)
        candidate = Path(found) if found is not None else None
    if candidate is None:
        raise BatchLaunchError(f"Python executable is not on PATH: {value}")
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise BatchLaunchError(f"Python executable is not readable: {candidate}") from exc
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise BatchLaunchError(f"Python executable is not an executable file: {resolved}")
    return str(resolved)


def _controller_environment(repo_root: Path) -> dict[str, str]:
    """Return the minimal host environment needed by codeaction and recovery subprocesses."""
    environment = {
        key: os.environ[key]
        for key in _SUBPROCESS_ENV_ALLOWLIST
        if isinstance(os.environ.get(key), str) and os.environ[key]
    }
    environment.setdefault(
        "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(
        (Path(repo_root).resolve() / "src").resolve())
    return environment


def _credential_prefix(alias: str) -> str:
    return alias.replace("-", "_").upper()


def _validate_credentials(path: Path, credentials: Mapping[str, str | None]) -> None:
    all_aliases = {
        entry.credential
        for entry in (resolve_model(model) for model in known_models())
        if entry.credential is not None
    }
    try:
        values = read_credential_file(path, allowed_aliases=all_aliases)
    except RegistryError as exc:
        raise BatchLaunchError(str(exc)) from exc
    missing = sorted({
        alias for alias in credentials.values()
        if alias is not None and f"{_credential_prefix(alias)}_KEY" not in values
    })
    if missing:
        raise BatchLaunchError(
            f"provider credential file lacks selected aliases: {missing}")
    # ``values`` contains secrets.  It is deliberately neither returned nor included in a plan.


def _comparison_identity(
    *,
    source_commit: str,
    models: tuple[str, ...],
    reasoning_profiles: Mapping[str, str],
    rate_limit_policies: Mapping[str, Mapping[str, Any]],
    image_digests: Mapping[str, str],
) -> dict[str, Any]:
    """The frozen identity for a reference-scaffold stage, via the one shared builder.

    It used to write its own, under a `harness` key that batch results reject as an unsupported
    identity field -- so every stage this module launched would have failed on its first result
    write. One builder now, and batch results are what defines the shape.
    """
    from codeaction.benchmark.agents import reference_agent

    return build_comparison_identity(
        source_commit=source_commit,
        agents={model: reference_agent(model, reasoning=reasoning_profiles[model])
                for model in models},
        rate_limit_policies=rate_limit_policies,
        image_digests=image_digests,
        run_profile=RUN_PROFILE,
    )


def _contained_attempt(batch_dir: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BatchLaunchError("accepted attempt evidence lacks attempt_dir")
    raw = Path(relative)
    if raw.is_absolute() or raw == Path(".") or any(
            part in {"", ".", ".."} for part in raw.parts):
        raise BatchLaunchError("accepted attempt_dir is not a normalized batch-relative path")
    candidate = batch_dir
    for part in raw.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise BatchLaunchError("accepted attempt_dir traverses a symlink")
    resolved = candidate.resolve()
    if batch_dir != resolved and batch_dir not in resolved.parents:
        raise BatchLaunchError("accepted attempt_dir escapes the batch directory")
    if not resolved.is_dir():
        raise BatchLaunchError("accepted attempt_dir does not exist")
    return resolved


def _accepted_image_identities(
    batch_dir: Path,
    specification: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[dict[str, str]]:
    identities = []
    for identifier in state["requested_cells"]:
        cell = state["cells"][identifier]
        if cell.get("status") != "accepted":
            continue
        execution_id = cell.get("accepted_execution_id")
        execution = state["executions"].get(f"{identifier}/{execution_id}") or {}
        detail = execution.get("detail") or {}
        evidence = detail.get("accepted_attempt") or {}
        manual = detail.get("manual_acceptance")
        allow_counter_mismatch = False
        if manual is not None:
            attention_id = manual.get("attention_id") if isinstance(manual, Mapping) else None
            attention = (state.get("attentions") or {}).get(attention_id) or {}
            resolution = attention.get("resolution") or {}
            allow_counter_mismatch = (
                manual.get("category") == "harness_attestation_failed"
                and attention.get("category") == "harness_attestation_failed"
                and attention.get("cell_id") == identifier
                and attention.get("execution_id") == execution_id
                and attention.get("status") == "resolved"
                and resolution.get("action") == "accept"
                and manual.get("note") == resolution.get("note")
            )
            if not allow_counter_mismatch:
                raise BatchLaunchError(
                    f"manual acceptance audit is invalid for {identifier}")
        try:
            validate_submission_evidence(
                batch_dir,
                specification,
                cell,
                evidence,
                allow_reference_agent_counter_mismatch=allow_counter_mismatch,
            )
        except BatchResultsError as exc:
            raise BatchLaunchError(
                f"accepted episode evidence is invalid for {identifier}: {exc}") from exc
        attempt = _contained_attempt(batch_dir, evidence.get("attempt_dir"))
        provenance = _read_object(
            attempt / "provenance.json", f"accepted provenance for {identifier}")
        identities.append({field: provenance.get(field) for field in _IMAGE_DIGEST_FIELDS})
    return identities


def _read_existing_batch(
    *,
    batch_dir: Path,
    target: StageTarget,
    models: tuple[str, ...],
    comparison_identity: Mapping[str, Any],
    image_digests: Mapping[str, str],
) -> Mapping[str, Any] | None:
    raw = Path(batch_dir).absolute()
    if raw.is_symlink():
        raise BatchLaunchError("batch directory must not be a symlink")
    if not raw.exists():
        return None
    if not raw.is_dir():
        raise BatchLaunchError("batch path exists but is not a directory")
    root = raw.resolve()
    spec_path = root / "batch_spec.json"
    state_path = root / "batch_state.json"
    if not spec_path.exists() and not state_path.exists():
        unexpected = sorted(
            path.name for path in root.iterdir() if path.name != ".controller.lock")
        if unexpected:
            raise BatchLaunchError(
                f"new batch directory is not empty: {unexpected}")
        return None
    if spec_path.exists() != state_path.exists():
        raise BatchLaunchError("batch_spec.json and batch_state.json must both exist")
    specification = _read_object(spec_path, "batch specification")
    state = _read_object(state_path, "batch state")
    second_state = _read_object(state_path, "batch state")
    if state != second_state:
        raise BatchLaunchError("batch state changed during read-only preflight")
    if specification.get("schema_version") != BATCH_SPEC_SCHEMA_VERSION \
            or state.get("schema_version") != BATCH_STATE_SCHEMA_VERSION:
        raise BatchLaunchError("batch schema version is unsupported")
    if specification.get("batch_id") != state.get("batch_id"):
        raise BatchLaunchError("batch specification/state IDs disagree")
    if tuple(specification.get("models") or ()) != models:
        raise BatchLaunchError("selected models differ from the frozen batch")
    if specification.get("comparison_identity") != comparison_identity \
            or specification.get("comparison_sha256") != sha256_json(
                dict(comparison_identity)):
        raise BatchLaunchError("comparison identity differs from the frozen batch")
    if specification.get("task_pack") != {
            "taskset_version": target.taskset_version,
            "sha256": target.task_pack_sha256,
    }:
        raise BatchLaunchError("task pack differs from the frozen batch")
    try:
        BatchStateStore._assert_invariants(specification, state)
    except BatchStateError as exc:
        raise BatchLaunchError(f"existing batch state is invalid: {exc}") from exc

    desired = {
        durable_cell_id(row.model, row.task, row.attempt_index): row
        for row in target_cell_rows(target, models)
    }
    existing = set(state["requested_cells"])
    if not existing <= set(desired):
        raise BatchLaunchError("manual stage target would remove existing requested cells")
    for identifier in existing:
        cell = state["cells"][identifier]
        episode = desired[identifier]
        if (cell.get("task"), cell.get("attempt_index"), cell.get("scene_seed")) != (
                episode.task, episode.attempt_index, episode.scene_seed):
            raise BatchLaunchError(f"existing cell identity drifted: {identifier}")
    accepted_images = _accepted_image_identities(root, specification, state)
    if any(identity != image_digests for identity in accepted_images):
        raise BatchLaunchError(
            "current official images differ from an already accepted episode image identity")
    return state


def _build_config(args: argparse.Namespace, *, repo_root: Path,
                  batch_dir: Path, task_pack: Path, expected_stage: str,
                  image_digests: Mapping[str, str],
                  reasoning_profiles: Mapping[str, str],
                  rate_limit_policies: Mapping[str, Mapping[str, Any]]) -> ControllerConfig:
    return ControllerConfig(
        expected_stage=expected_stage,
        repo_root=repo_root,
        batch_dir=batch_dir,
        python_bin=_resolve_python(args.python),
        controller_module="codeaction.cli.main",
        task_pack=task_pack,
        provider_env_file=_require_regular_file(
            args.provider_env_file, "provider credential file"),
        provider_rate_limit_file=_require_regular_file(
            args.provider_rate_limit_file, "provider rate-limit file"),
        image_digests=dict(image_digests),
        agents={model: reference_agent(model, reasoning=rung)
                for model, rung in reasoning_profiles.items()},
        rate_limit_policies={
            model: copy.deepcopy(dict(policy))
            for model, policy in rate_limit_policies.items()
        },
        gpus=tuple(args.gpus),
        environment=_controller_environment(repo_root),
        hard_timeout_s=args.hard_timeout_s,
        cleanup_timeout_s=args.cleanup_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )


def prepare_batch(
    args: argparse.Namespace,
    *,
    repo_root: Path = _RT,
    source_preflight: Callable[[Path], tuple[str, Mapping[str, str]]] = _git_preflight,
    image_preflight: Callable[[Mapping[str, str], str], Mapping[str, str]] =
        _inspect_official_images,
) -> PreparedBatch:
    """Complete every read-only check before a state lock or GPU lease can exist."""
    root = _require_directory(repo_root, "source root")
    source_commit, source_snapshot = source_preflight(root)
    if not isinstance(source_commit, str) or not source_commit:
        raise BatchLaunchError("source preflight returned no commit")
    task_pack = _require_directory(args.task_pack, "task pack")
    target = (
        resolve_cell_manifest(args.cell_manifest, tasks_root=task_pack)
        if args.cell_manifest is not None
        else resolve_stage(args.stage, tasks_root=task_pack)
    )
    manifest_models = (
        tuple(dict.fromkeys(cell.model for cell in target.exact_cells))
        if target.exact_cells is not None else None
    )
    models = tuple(
        args.models if args.models is not None
        else (manifest_models or tuple(DEFAULT_MODELS))
    )
    if not models or len(models) != len(set(models)):
        raise BatchLaunchError("models must be non-empty and contain no duplicates")
    gpus = tuple(args.gpus)
    if not gpus or len(gpus) != len(set(gpus)) or any(
            not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0 for gpu in gpus):
        raise BatchLaunchError("GPUs must be unique non-negative integers")
    try:
        target_cell_rows(target, models)
    except ValueError as exc:
        raise BatchLaunchError(str(exc)) from exc
    reasoning_profiles = parse_reasoning_profiles(
        models, args.reasoning_profile, args.reasoning_profile_override)
    credentials = model_credentials(models)
    if any(credential is None for credential in credentials.values()):
        raise BatchLaunchError("reference-scaffold provider batch models must declare credential aliases")
    credential_limits = parse_credential_limits(
        args.credential_limit, credentials, args.default_credential_limit)

    provider_env_file = _require_regular_file(
        args.provider_env_file, "provider credential file")
    _validate_credentials(provider_env_file, credentials)
    rate_limit_file = _require_regular_file(
        args.provider_rate_limit_file, "provider rate-limit file")
    rate_config = load_rate_limit_config(rate_limit_file)
    rate_policies = {
        model: rate_limit_policy_for(rate_config, credentials[model], model)
        for model in models
    }
    missing_rate_limits = sorted(
        model for model, policy in rate_policies.items()
        if not strict_rate_limit_coverage(policy)
    )
    if missing_rate_limits:
        raise BatchLaunchError(
            f"models lack strict request/token rate limits: {missing_rate_limits}")

    image_refs = {
        "sim_image_digest": args.sim_image,
        "agent_image_digest": args.reference_agent_image,
        "gateway_image_digest": args.gateway_image,
    }
    image_digests = dict(image_preflight(image_refs, source_commit))
    if set(image_digests) != set(_IMAGE_DIGEST_FIELDS) or any(
            not isinstance(value, str) or not _IMAGE_DIGEST.fullmatch(value)
            for value in image_digests.values()):
        raise BatchLaunchError("official image preflight returned invalid content IDs")
    comparison_identity = _comparison_identity(
        source_commit=source_commit,
        models=models,
        reasoning_profiles=reasoning_profiles,
        rate_limit_policies=rate_policies,
        image_digests=image_digests,
    )
    batch_dir = Path(args.batch_dir).absolute()
    existing_state = _read_existing_batch(
        batch_dir=batch_dir,
        target=target,
        models=models,
        comparison_identity=comparison_identity,
        image_digests=image_digests,
    )
    config = _build_config(
        args,
        repo_root=root,
        batch_dir=batch_dir,
        task_pack=task_pack,
        expected_stage=target.stage,
        image_digests=image_digests,
        reasoning_profiles=reasoning_profiles,
        rate_limit_policies=rate_policies,
    )
    synthetic_specification = {
        "batch_id": "preflight-only",
        "models": list(models),
        "comparison_identity": comparison_identity,
    }
    validate_controller_identity(synthetic_specification, config)
    return PreparedBatch(
        repo_root=root,
        batch_dir=batch_dir,
        source_commit=source_commit,
        source_snapshot=dict(source_snapshot),
        target=target,
        models=models,
        gpus=gpus,
        credentials=credentials,
        credential_limits=credential_limits,
        reasoning_profiles=reasoning_profiles,
        rate_limit_policies=rate_policies,
        comparison_identity=comparison_identity,
        image_identity=image_digests,
        config=config,
        existing_state=existing_state,
    )


def _command_example(prepared: PreparedBatch) -> list[str]:
    row = target_cell_rows(prepared.target, prepared.models)[0]
    model = row.model
    episode = row
    identifier = durable_cell_id(model, episode.task, episode.attempt_index)
    prior_cell = (prepared.existing_state or {}).get("cells", {}).get(identifier, {})
    execution_number = int(prior_cell.get("execution_count") or 0) + 1
    plan = build_execution_plan(
        specification={
            "batch_id": (prepared.existing_state or {}).get("batch_id", "dry-run-preview"),
            "models": list(prepared.models),
            "comparison_identity": prepared.comparison_identity,
        },
        cell_id=identifier,
        cell={
            "model": model,
            "task": episode.task,
            "attempt_index": episode.attempt_index,
            "scene_seed": episode.scene_seed,
        },
        lease={
            "lease_id": "dry-run-preview",
            "cell_id": identifier,
            "execution_id": f"execution-{execution_number:03d}",
            "gpu": prepared.gpus[0],
        },
        config=prepared.config,
    )
    command = list(plan.command)
    provider_flag = command.index("--provider-env-file")
    command[provider_flag + 1] = "<redacted-provider-env-file>"
    return command


def _dry_run_document(prepared: PreparedBatch) -> dict[str, Any]:
    state = prepared.existing_state or {}
    cells = state.get("cells") or {}
    requested = state.get("requested_cells") or []
    accepted = sum(cells.get(identifier, {}).get("status") == "accepted"
                   for identifier in requested)
    return {
        "schema_version": "manual-reference-batch-plan.v1",
        "dry_run": True,
        "stage": prepared.target.stage,
        "auto_advance": False,
        "runtime": "codeaction-compose",
        "track": "A",
        "batch_dir": str(prepared.batch_dir),
        "source_commit": prepared.source_commit,
        "source_snapshot": dict(prepared.source_snapshot),
        "models": list(prepared.models),
        "gpus": list(prepared.gpus),
        "tasks": list(prepared.target.tasks),
        "episodes_per_task": prepared.target.episodes_per_task,
        "target_cells": len(target_cell_rows(prepared.target, prepared.models)),
        "existing_requested_cells": len(requested),
        "existing_accepted_cells": accepted,
        "credential_aliases": dict(prepared.credentials),
        "credential_limits": dict(prepared.credential_limits),
        "official_image_identity": dict(prepared.image_identity),
        "comparison_sha256": sha256_json(dict(prepared.comparison_identity)),
        "launch_contract": {
            "agent_mode": "reference",
            "reference_model_mode": "provider",
            "profile": RUN_PROFILE,
            "interface_profile": INTERFACE_PROFILE,
            "attempts_per_execution": 1,
            "command_example": _command_example(prepared),
        },
    }


def execute_prepared(
    prepared: PreparedBatch,
    *,
    store_open: Callable[..., Any] = BatchStateStore.open_or_create,
    scheduler_type: Callable[..., Any] = BatchScheduler,
    controller_type: Callable[..., Any] = BatchController,
    stdout: TextIO = sys.stdout,
) -> int:
    """Create/resume state only after preflight, then delegate the one selected stage."""
    with store_open(
        prepared.batch_dir,
        target=prepared.target,
        models=prepared.models,
        comparison_identity=prepared.comparison_identity,
    ) as store:
        scheduler = scheduler_type(
            store,
            model_credentials=prepared.credentials,
            credential_limits=prepared.credential_limits,
        )
        controller = controller_type(store, scheduler, config=prepared.config)
        stdout.write(
            f"[batch] stage={prepared.target.stage} cells="
            f"{len(target_cell_rows(prepared.target, prepared.models))} "
            f"batch={prepared.batch_dir}\n")
        stdout.flush()
        return int(controller.run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one manually selected official reference-scaffold batch stage.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--stage", choices=tuple(MANUAL_STAGES))
    target.add_argument("--cell-manifest", type=Path)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--reasoning-profile", default="high")
    parser.add_argument(
        "--reasoning-profile-override", action="append", default=[], metavar="MODEL=RUNG")
    parser.add_argument(
        "--default-credential-limit", type=int,
        default=DEFAULT_CREDENTIAL_CONCURRENCY)
    parser.add_argument(
        "--credential-limit", action="append", default=[], metavar="ALIAS=N")
    parser.add_argument(
        "--provider-env-file", type=Path,
        default=Path.home() / ".codeaction_provider.env")
    parser.add_argument(
        "--provider-rate-limit-file", type=Path,
        default=Path.home() / ".codeaction" / "provider_rate_limits.json")
    parser.add_argument("--sim-image", default=DEFAULT_IMAGE_REFS["sim_image_digest"])
    parser.add_argument(
        "--reference-agent-image", default=DEFAULT_IMAGE_REFS["agent_image_digest"])
    parser.add_argument("--gateway-image", default=DEFAULT_IMAGE_REFS["gateway_image_digest"])
    parser.add_argument("--task-pack", type=Path, default=TASKS_ROOT)
    parser.add_argument(
        "--python", default=os.environ.get(
            "CODEACTION_PYTHON", sys.executable))
    parser.add_argument("--hard-timeout-s", type=float, default=7200.0)
    parser.add_argument("--cleanup-timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_cli(
    argv: list[str] | None = None,
    *,
    repo_root: Path = _RT,
    source_preflight: Callable[[Path], tuple[str, Mapping[str, str]]] = _git_preflight,
    image_preflight: Callable[[Mapping[str, str], str], Mapping[str, str]] =
        _inspect_official_images,
    store_open: Callable[..., Any] = BatchStateStore.open_or_create,
    scheduler_type: Callable[..., Any] = BatchScheduler,
    controller_type: Callable[..., Any] = BatchController,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepared = prepare_batch(
            args, repo_root=repo_root, source_preflight=source_preflight,
            image_preflight=image_preflight)
        if args.dry_run:
            document = _dry_run_document(prepared)
            stdout.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stdout.write("# " + shlex.join(document["launch_contract"]["command_example"]) + "\n")
            stdout.flush()
            return 0
        return execute_prepared(
            prepared,
            store_open=store_open,
            scheduler_type=scheduler_type,
            controller_type=controller_type,
            stdout=stdout,
        )
    except (
            BatchLaunchError,
            BatchStateError,
            ProviderRuntimeConfigError,
            RegistryError,
            ValueError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
    ) as exc:
        stderr.write(f"ERROR: {type(exc).__name__}: {exc}\n")
        stderr.flush()
        return 2


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
