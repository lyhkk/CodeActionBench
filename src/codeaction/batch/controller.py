"""Single-stage reference-scaffold batch control over isolated ``codeaction`` executions.

The host process is only a control plane.  Every episode is launched through ``codeaction.py run``
and therefore receives its own Compose project; this module never invokes the legacy bare-metal
episode driver.  A controller instance owns exactly the stage already opened in durable state and
never selects or expands to a following stage.
"""
from __future__ import annotations

import concurrent.futures
import copy
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from codeaction.benchmark.agents import AgentConfig
from codeaction.evidence.artifacts import MANIFEST_NAME
from codeaction.batch.attention import AttentionReporter
from codeaction.batch.control import apply_control_requests
from codeaction.batch.outcomes import OutcomePolicy, classify_execution
from codeaction.batch.recovery import reconcile_active_leases
from codeaction.batch.results import (
    BatchResultsError,
    validate_execution_attempt,
    write_batch_result_views,
)
from codeaction.batch.state import BatchStateError, BatchStateStore, cell_id as durable_cell_id
from codeaction.batch.supervisor import ProcessResult, ProcessSupervisor
from codeaction.providers.model_registry import resolve_model


_EXECUTION_ID = re.compile(r"^execution-(\d{3,})$")
_REASONING_RUNG = re.compile(r"^[a-z][a-z0-9-]*$")
_IMAGE_DIGEST = re.compile(r"^(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})$")
_IMAGE_DIGEST_FIELDS = frozenset({
    "sim_image_digest", "agent_image_digest", "gateway_image_digest",
})


class BatchControllerError(RuntimeError):
    """The controller cannot safely continue the requested manual stage."""


@dataclass(frozen=True)
class ControllerConfig:
    """Frozen launch inputs for one invocation of an already-open manual stage."""

    expected_stage: str
    repo_root: Path
    batch_dir: Path
    python_bin: str
    controller_module: str
    task_pack: Path
    provider_env_file: Path
    provider_rate_limit_file: Path
    image_digests: Mapping[str, str]
    agents: Mapping[str, AgentConfig]
    rate_limit_policies: Mapping[str, Mapping[str, Any]]
    gpus: tuple[int, ...]
    environment: Mapping[str, str]
    token_file: Path | None = None
    # Which run tier the cells launch under. `eval` keeps every strict gate and is what a
    # release stage must use; `dev` records the same findings as non-submittable reasons instead
    # of refusing, which is the right tier for an experiment batch.
    run_profile: str = "eval"
    hard_timeout_s: float = 7200.0
    cleanup_timeout_s: float = 120.0
    poll_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.expected_stage, str) or not self.expected_stage:
            raise ValueError("expected_stage must be non-empty")
        if not isinstance(self.python_bin, str) or not self.python_bin:
            raise ValueError("python_bin must be non-empty")
        if self.controller_module != "codeaction.cli.main":
            raise ValueError("controller_module must be 'codeaction.cli.main'")
        if not self.gpus or len(self.gpus) != len(set(self.gpus)):
            raise ValueError("gpus must be non-empty and unique")
        if any(not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0
               for gpu in self.gpus):
            raise ValueError("gpus must contain non-negative integers")
        for value, field in (
                (self.hard_timeout_s, "hard_timeout_s"),
                (self.cleanup_timeout_s, "cleanup_timeout_s"),
                (self.poll_interval_s, "poll_interval_s")):
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be a finite positive number")
        if not isinstance(self.image_digests, Mapping) \
                or set(self.image_digests) != _IMAGE_DIGEST_FIELDS \
                or any(not isinstance(value, str) or not _IMAGE_DIGEST.fullmatch(value)
                       for value in self.image_digests.values()):
            raise ValueError(
                "image_digests must contain exact digest-pinned sim, agent, and gateway images")
        if not isinstance(self.agents, Mapping) or not self.agents:
            raise ValueError("agents must be a non-empty mapping")
        for label, agent in self.agents.items():
            if not isinstance(agent, AgentConfig):
                raise ValueError("agents must contain AgentConfig values")
            if label != agent.label:
                raise ValueError("agent mapping keys must match agent labels")
            if not isinstance(agent.reasoning, str) or not _REASONING_RUNG.fullmatch(agent.reasoning):
                raise ValueError("agent reasoning must be a valid rung")
        profiles = {agent.interface_profile for agent in self.agents.values()}
        if len(profiles) != 1:
            raise ValueError(
                "a batch runs one interface profile; got "
                f"{sorted(profiles)} -- run one batch per profile")
        # Subscription agents do not use provider-key rate-limit policies.
        expected_policies = {label for label, agent in self.agents.items()
                             if agent.has_provider_profile}
        if not isinstance(self.rate_limit_policies, Mapping) \
                or set(self.rate_limit_policies) != expected_policies \
                or any(not isinstance(policy, Mapping) or not policy
                       for policy in self.rate_limit_policies.values()):
            raise ValueError(
                "rate_limit_policies must exactly cover the provider-backed models with "
                "non-empty objects")
        if not isinstance(self.environment, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in self.environment.items()):
            raise ValueError("environment must map strings to strings")
        if self.run_profile not in ("dev", "eval"):
            raise ValueError(f"run_profile must be dev or eval, got {self.run_profile!r}")


@dataclass(frozen=True)
class ExecutionPlan:
    """Paths and process identity for one immutable execution record."""

    cell_id: str
    lease_id: str
    execution_id: str
    execution_dir: Path
    run_dir: Path
    log_path: Path
    run_id: str
    compose_project: str
    command: tuple[str, ...]
    cleanup_command: tuple[str, ...]


@dataclass(frozen=True)
class _PendingExecution:
    plan: ExecutionPlan
    lease: Mapping[str, Any]
    cell: Mapping[str, Any]
    credential: str | None


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _safe_hash(value: str, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _execution_number(execution_id: Any) -> int:
    match = _EXECUTION_ID.fullmatch(str(execution_id or ""))
    if match is None:
        raise BatchControllerError(f"invalid durable execution ID: {execution_id!r}")
    value = int(match.group(1))
    if value < 1:
        raise BatchControllerError("durable execution number must be positive")
    return value


def inspect_official_images(
    refs: Mapping[str, str], source_commit: str,
) -> dict[str, str]:
    """Freeze images by content ID and verify the actual source used by the build."""
    if set(refs) != set(_IMAGE_DIGEST_FIELDS):
        raise BatchControllerError("official image refs must cover sim, reference-agent, and gateway")
    identities: dict[str, str] = {}
    from codeaction.release import image_matches_runtime, runtime_identity
    from codeaction.paths import PROJECT_ROOT
    expected = runtime_identity(PROJECT_ROOT)
    for field in _IMAGE_DIGEST_FIELDS:
        ref = refs[field]
        if not isinstance(ref, str) or not ref:
            raise BatchControllerError(f"official image ref is empty: {field}")
        try:
            process = subprocess.run(
                ["docker", "image", "inspect", ref], check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise BatchControllerError("docker image inspection is unavailable") from exc
        if process.returncode:
            raise BatchControllerError(f"official image is unavailable locally: {ref}")
        try:
            values = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise BatchControllerError(f"docker returned invalid inspection JSON: {ref}") from exc
        if not isinstance(values, list) or len(values) != 1:
            raise BatchControllerError(f"docker did not resolve exactly one image: {ref}")
        value = values[0]
        image_id = value.get("Id")
        labels = ((value.get("Config") or {}).get("Labels") or {})
        if not isinstance(image_id, str) or not _IMAGE_DIGEST.fullmatch(image_id):
            raise BatchControllerError(f"official image has an invalid content ID: {ref}")
        from codeaction.launch import context
        if context() is not None:
            from codeaction.environments import validate_environment
            try:
                validate_environment(labels, PROJECT_ROOT)
            except ValueError as exc:
                raise BatchControllerError(str(exc)) from exc
        elif not isinstance(labels, Mapping) or not image_matches_runtime(labels, expected):
            raise BatchControllerError(f"image differs from current source; run tools/build_images.sh: {ref}")
        identities[field] = image_id
    return identities


def build_comparison_identity(
    *,
    source_commit: str,
    agents: Mapping[str, Any],
    rate_limit_policies: Mapping[str, Mapping[str, Any]],
    image_digests: Mapping[str, str],
    run_profile: str = "eval",
) -> dict[str, Any]:
    """The frozen identity a batch is created with, derived from its agents.

    One builder for both runners: the supervisor's own stages and the agent-axis matrix. Option B
    means every agent in a batch delivers the same interface profile, which is what makes a single
    batch-level `interface_profile` truthful. Provider-backed agents contribute a request/rate
    profile; a subscription agent has neither and contributes none."""
    from codeaction.providers.model_registry import resolve_model

    profiles = {agent.interface_profile for agent in agents.values()}
    if len(profiles) != 1:
        raise BatchControllerError(
            f"a batch runs one interface profile; got {sorted(profiles)}")
    interface_profile = profiles.pop()
    drivers = {agent.driver for agent in agents.values()}
    projections = {}
    for label, agent in agents.items():
        if not agent.has_provider_profile:
            # Present and empty on purpose: see the model_profiles check in batch.results.
            projections[label] = {}
            continue
        entry = resolve_model(agent.model)
        request_profile = entry.profile(agent.reasoning)
        projections[label] = {
            "reasoning": request_profile["reasoning"],
            "request_profile": request_profile,
            "transport_profile": entry.transport_profile(),
            "rate_limit_policy": copy.deepcopy(dict(rate_limit_policies[label])),
        }
    from codeaction.contracts.identity import driver_kind_for_agent_mode

    if len(drivers) != 1:
        raise BatchControllerError(f"a batch runs one driver; got {sorted(drivers)}")
    # EXACTLY the keys a run reports under tested_unit.driver, and no others: batch results
    # compare every frozen key against the run's own driver object, so an extra key here (the
    # `mcp_profile` this used to carry) is a guaranteed mismatch, and a key named `harness`
    # instead of `driver` is rejected outright as an unsupported identity field.
    return {
        "source_commit": source_commit,
        "interface_profile": interface_profile,
        "run_profile": run_profile,
        "submittable": run_profile == "eval",
        "image_digests": copy.deepcopy(dict(image_digests)),
        "driver": {
            "kind": ("local_agent" if all(a.reference_model_mode == "local" for a in agents.values())
                     else driver_kind_for_agent_mode(drivers.pop())),
            "image_digest": image_digests["agent_image_digest"],
        },
        "model_profiles": projections,
    }


def validate_controller_identity(
    specification: Mapping[str, Any], config: ControllerConfig,
) -> None:
    """Prove the launch rung and reference-scaffold interface are part of the frozen comparison."""
    models = tuple(specification.get("models") or ())
    if not isinstance(specification.get("batch_id"), str) \
            or not specification.get("batch_id"):
        raise BatchControllerError("batch specification lacks a non-empty batch_id")
    if set(config.agents) != set(models):
        raise BatchControllerError(
            "agents must exactly cover the frozen batch model set")
    comparison = specification.get("comparison_identity")
    if not isinstance(comparison, Mapping):
        raise BatchControllerError("batch specification lacks comparison_identity")
    agents = config.agents
    delivered = {agent.interface_profile for agent in agents.values()}
    if comparison.get("interface_profile") not in delivered:
        raise BatchControllerError(
            f"frozen interface_profile {comparison.get('interface_profile')!r} is not what this "
            f"batch's agents deliver ({sorted(delivered)})")
    # A release stage must be eval AND submittable; an experiment batch declares dev, and then
    # `submittable` must be False rather than an unearned True.
    if comparison.get("run_profile") != config.run_profile:
        raise BatchControllerError(
            f"frozen run_profile {comparison.get('run_profile')!r} differs from the controller's "
            f"{config.run_profile!r}")
    if comparison.get("submittable") is not (config.run_profile == "eval"):
        raise BatchControllerError(
            "submittable must be True for an eval batch and False for a dev batch")
    source_commit = comparison.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise BatchControllerError("batch comparison must freeze source_commit")
    if not isinstance(comparison.get("driver"), Mapping) or not comparison["driver"]:
        raise BatchControllerError("batch comparison must freeze the driver identity")
    frozen_images = comparison.get("image_digests")
    if not isinstance(frozen_images, Mapping) \
            or dict(frozen_images) != dict(config.image_digests):
        raise BatchControllerError(
            "controller image digests differ from the frozen batch comparison")
    # `agent_image_digest` names whichever agent container this batch runs: the reference
    # scaffold's image for a reference batch, the vendor CLI's for a vendor batch.
    if comparison["driver"].get("image_digest") != frozen_images.get(
            "agent_image_digest"):
        raise BatchControllerError(
            "frozen driver image differs from the agent image digest")
    # EVERY model appears, exactly as batch.results requires -- an agent with no provider call
    # behind it (no API key, or no request at all) appears with an EMPTY profile, which states
    # that nothing about a provider request is frozen for it. Only the provider-backed ones have
    # a request and a rate-limit window to check.
    frozen_profiles = comparison.get("model_profiles")
    if not isinstance(frozen_profiles, Mapping) or set(frozen_profiles) != set(models):
        raise BatchControllerError(
            "batch comparison model_profiles must exactly cover the batch models")
    provider_models = tuple(model for model in models
                            if agents[model].has_provider_profile)
    empty_but_provider_backed = [model for model in provider_models
                                 if not frozen_profiles.get(model)]
    if empty_but_provider_backed:
        raise BatchControllerError(
            f"provider-backed models froze an empty profile: {sorted(empty_but_provider_backed)}")
    for label in provider_models:
        profile = frozen_profiles.get(label)
        # The batch axis is the agent LABEL; the request profile is a property of the MODEL that
        # label runs. They are the same string for most seats, which is why resolving the label
        # worked until a seat published under one name and ran another -- and then this check,
        # correctly, refused a batch it could not vouch for. Resolve the model, as the identity
        # builder already does.
        entry = resolve_model(agents[label].model)
        expected_request = entry.profile(agents[label].reasoning)
        if not isinstance(profile, Mapping) \
                or set(profile) != {
                    "request_profile", "reasoning", "transport_profile",
                    "rate_limit_policy",
                } \
                or profile.get("request_profile") != expected_request \
                or profile.get("reasoning") != expected_request.get("reasoning") \
                or profile.get("transport_profile") != entry.transport_profile() \
                or profile.get("rate_limit_policy") != config.rate_limit_policies[label]:
            raise BatchControllerError(
                f"launch reasoning/request profile differs from frozen agent {label!r} "
                f"(model {agents[label].model!r})")


def build_execution_plan(
    *,
    specification: Mapping[str, Any],
    cell_id: str,
    cell: Mapping[str, Any],
    lease: Mapping[str, Any],
    config: ControllerConfig,
) -> ExecutionPlan:
    """Build one side-effect-free, isolated reference-scaffold ``codeaction`` launch."""
    validate_controller_identity(specification, config)
    if lease.get("cell_id") != cell_id:
        raise BatchControllerError("lease and selected cell disagree")
    model = cell.get("model")
    task = cell.get("task")
    attempt_index = cell.get("attempt_index")
    if not isinstance(model, str) or model not in config.agents:
        raise BatchControllerError("cell model is not frozen in the controller configuration")
    if not isinstance(task, str) or not task:
        raise BatchControllerError("cell task must be non-empty")
    if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) \
            or attempt_index < 0:
        raise BatchControllerError("cell attempt_index must be a non-negative integer")
    try:
        expected_cell_id = durable_cell_id(model, task, attempt_index)
    except BatchStateError as exc:
        raise BatchControllerError(f"cell has unsafe durable identity: {exc}") from exc
    if cell_id != expected_cell_id:
        raise BatchControllerError("cell ID differs from its model/task/attempt identity")
    gpu = lease.get("gpu")
    if gpu not in config.gpus:
        raise BatchControllerError("lease GPU is outside the controller GPU set")
    execution_id = str(lease.get("execution_id") or "")
    execution_number = _execution_number(execution_id)
    batch_root = Path(config.batch_dir).resolve()
    execution_dir = (
        batch_root / "runs" / model / task / f"attempt-{attempt_index:03d}" / execution_id
    )
    if execution_dir.exists():
        raise BatchControllerError(
            f"immutable execution directory already exists: {execution_dir}")
    run_dir = execution_dir / "run"
    run_id = (
        f"b-{_safe_hash(str(specification.get('batch_id')), 12)}-"
        f"{_safe_hash(cell_id, 12)}-e{execution_number:03d}"
    )
    compose_project = f"codeaction-{run_id}-a{attempt_index:03d}"
    # ONE command builder for both drivers (codeaction.benchmark.agents.build_agent_command).
    # Before this the supervisor wrote its own reference-only tuple while the agent-axis runner
    # wrote another, and the two had already drifted -- the supervisor never passed
    # --agent-label, so every reference cell it launched was filed under the CLI's default label.
    from codeaction.benchmark.agents import build_agent_command

    agent = config.agents[model]
    command = tuple(build_agent_command(
        agent,
        python_bin=str(config.python_bin),
        controller_module=config.controller_module,
        task=task,
        gpu=gpu,
        run_dir=run_dir,
        task_pack=Path(config.task_pack).resolve(),
        profile=config.run_profile,
        provider_env_file=Path(config.provider_env_file).resolve(),
        provider_rate_limit_file=Path(config.provider_rate_limit_file).resolve(),
        token_file=config.token_file,
        attempt_index=attempt_index,
        run_id=run_id,
        source_root=Path(config.repo_root).resolve(),
        image_digests=config.image_digests,
    ))
    cleanup = (
        str(config.python_bin), "-m", "codeaction.batch.recovery", "cleanup-project",
        "--project", compose_project,
    )
    return ExecutionPlan(
        cell_id=cell_id,
        lease_id=str(lease.get("lease_id") or ""),
        execution_id=execution_id,
        execution_dir=execution_dir,
        run_dir=run_dir,
        log_path=execution_dir / "controller.log",
        run_id=run_id,
        compose_project=compose_project,
        command=command,
        cleanup_command=cleanup,
    )


class BatchController:
    """Run only the currently requested stage and settle every lease durably."""

    def __init__(
        self,
        store: BatchStateStore,
        scheduler: Any,
        *,
        config: ControllerConfig,
        reporter: AttentionReporter | None = None,
        runner: Callable[..., Any] | None = None,
        classifier: Callable[..., Any] = classify_execution,
        validator: Callable[..., Mapping[str, Any]] = validate_execution_attempt,
        result_writer: Callable[..., Any] = write_batch_result_views,
        reconciler: Callable[..., Any] = reconcile_active_leases,
        control_applier: Callable[..., Any] = apply_control_requests,
        outcome_policy: OutcomePolicy | None = None,
        now: Callable[[], dt.datetime] = _utc_now,
        thread_pool_factory: Callable[..., Any] = concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        self.store = store
        self.scheduler = scheduler
        self.config = config
        self.reporter = reporter or AttentionReporter(store)
        self.runner = runner or ProcessSupervisor(store).run
        self.classifier = classifier
        self.validator = validator
        self.result_writer = result_writer
        self.reconciler = reconciler
        self.control_applier = control_applier
        self.outcome_policy = outcome_policy or OutcomePolicy()
        self.now = now
        self.thread_pool_factory = thread_pool_factory
        self.last_error: str | None = None
        self._owner_thread = threading.get_ident()
        if Path(store.batch_dir).resolve() != Path(config.batch_dir).resolve():
            raise BatchControllerError("controller batch_dir differs from its durable state store")
        validate_controller_identity(store.specification, config)

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise BatchControllerError("durable transitions must run on the controller thread")

    def _assert_stage(self) -> Mapping[str, Any]:
        snapshot = self.store.snapshot()
        if snapshot.get("active_stage") != self.config.expected_stage:
            raise BatchControllerError(
                "durable active_stage differs from this manual controller invocation")
        history = snapshot.get("stage_history") or []
        if not history or history[-1].get("stage") != self.config.expected_stage:
            raise BatchControllerError("latest stage history entry differs from expected_stage")
        return snapshot

    def _deadline_at(self, now: dt.datetime) -> str:
        if now.tzinfo is None:
            raise BatchControllerError("controller clock must include a timezone")
        return (now + dt.timedelta(seconds=self.config.hard_timeout_s)).isoformat()

    def _launch(self, pending: _PendingExecution) -> Any:
        plan = pending.plan
        return self.runner(
            plan.lease_id,
            plan.command,
            log_path=plan.log_path,
            cwd=Path(self.config.repo_root).resolve(),
            env=dict(self.config.environment),
            hard_timeout_s=self.config.hard_timeout_s,
            run_dir=plan.run_dir,
            compose_project=plan.compose_project,
            cleanup_command=plan.cleanup_command,
            cleanup_timeout_s=self.config.cleanup_timeout_s,
        )

    @staticmethod
    def _process_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, ProcessResult):
            return value.as_dict()
        if isinstance(value, Mapping):
            return dict(value)
        method = getattr(value, "as_dict", None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return dict(result)
        raise BatchControllerError("supervisor runner returned no structured process result")

    def _open_attention(
        self,
        pending: _PendingExecution,
        *,
        category: str,
        reason: str,
        scope: Mapping[str, str] | None = None,
        location: Path | None = None,
        suggested_action: str,
        evidence: Mapping[str, Any] | None = None,
        execution_detail: Mapping[str, Any] | None = None,
    ) -> None:
        self._assert_owner()
        self.reporter.open_for_lease(
            pending.plan.lease_id,
            category=category,
            reason=reason,
            scope=dict(scope or {"kind": "cell", "id": pending.plan.cell_id}),
            location=str(location or pending.plan.execution_dir),
            suggested_action=suggested_action,
            evidence=dict(evidence or {}),
            execution_detail=dict(execution_detail or {}),
        )

    def _settle_worker_exception(
        self, pending: _PendingExecution, error: BaseException,
    ) -> None:
        self._open_attention(
            pending,
            category="supervisor_exception",
            reason=f"worker supervisor raised {type(error).__name__}",
            suggested_action=(
                "inspect the immutable execution and recorded compose project before requeue"),
            evidence={"exception_type": type(error).__name__},
            execution_detail={"controller": {"worker_exception": type(error).__name__}},
        )

    def _apply_window_governor(self, pending: "_PendingExecution") -> None:
        """Read what this episode's vendor stream said about the account's rolling window.

        Called for every settled execution, whatever its verdict: the window was spent either
        way, and an episode that failed late spent as much of it as one that passed. A credential
        with no window policy -- every provider key, and any subscription whose plan declares
        none -- returns here immediately.
        """
        from codeaction.batch import window_governor
        from codeaction.benchmark.agent_config import account_for
        alias = self.scheduler.model_credentials.get(pending.cell.get("model"))
        if not alias:
            return
        try:
            account = account_for(alias)
        except Exception:                                   # a local config error is not a verdict
            return
        policy = getattr(account, "window", None)
        if policy is None:
            return
        run_dir = Path(pending.plan.run_dir)
        streams = sorted(run_dir.glob("attempt-*-seed-*/vendor/vendor_stream.jsonl"))
        if not streams:
            return
        reading = window_governor.read_window(streams[0])
        decision = window_governor.after_episode(
            policy, reading, now=self.now().timestamp())
        if decision.action != "pause_credential":
            return
        self.store.open_credential_pause(
            alias, resume_at=float(decision.resume_at or 0.0), note=decision.reason,
            utilization=decision.utilization)

    def _settle_result(self, pending: _PendingExecution, raw_process: Any) -> None:
        self._assert_owner()
        # Before the outcome branches, because each of them returns and the window was spent
        # regardless of which one this execution takes.
        try:
            self._apply_window_governor(pending)
        except Exception:                 # governing must never turn a settled run into a crash
            pass
        process = self._process_mapping(raw_process)
        # A stopped attempt is never scored or automatically retried, even if partial
        # evidence contains a verifier value. The original execution remains inspectable.
        stop = process.get("operator_stop")
        if not stop:
            stop = next((lease.get("operator_stop") for lease in self.store.active_leases()
                         if lease["lease_id"] == pending.plan.lease_id), None)
        if stop:
            self._open_attention(
                pending, category="operator_stopped", reason="operator interrupted this attempt",
                suggested_action="requeue after inspection, then resume the paused queue",
                evidence={"stop_request": dict(stop)}, execution_detail={"process": process})
            return
        process_abnormal = process.get("return_code") != 0 \
            or process.get("timed_out") is True or bool(process.get("error"))
        cleanup_proven = process.get("cleanup_return_code") == 0 \
            and process.get("cleanup_timed_out") is False
        if process_abnormal and cleanup_proven:
            try:
                accepted_attempt = self.validator(
                    self.config.batch_dir,
                    self.store.specification,
                    pending.cell,
                    pending.plan.run_dir,
                )
            except BatchResultsError:
                # A complete non-scoreable provider result is not acceptable evidence, but its
                # structured failure still determines retry/attention below.
                pass
            except Exception as exc:
                self._open_attention(
                    pending,
                    category="preserved_artifact_validation_failed",
                    reason=(
                        "abnormal process left evidence whose validator failed "
                        f"with {type(exc).__name__}"),
                    suggested_action=(
                        "inspect the sealed reference-scaffold evidence before requeueing this episode"),
                    evidence={"exception_type": type(exc).__name__},
                    execution_detail={"process": process},
                )
                return
            else:
                self.store.finish_accepted_execution(
                    pending.plan.lease_id,
                    accepted_attempt=dict(accepted_attempt),
                    detail={
                        "process": process,
                        "settlement": "preserved_sealed_attempt_after_abnormal_process",
                    },
                )
                return
        try:
            decision = self.classifier(
                cell_id=pending.plan.cell_id,
                cell=pending.cell,
                process=process,
                run_dir=pending.plan.run_dir,
                credential=pending.credential,
                policy=self.outcome_policy,
                now=self.now(),
            )
        except Exception as exc:
            self._open_attention(
                pending,
                category="outcome_classification_failed",
                reason=f"execution outcome classifier raised {type(exc).__name__}",
                suggested_action="inspect preserved process and result artifacts before requeue",
                evidence={"exception_type": type(exc).__name__},
                execution_detail={"process": process},
            )
            return

        if decision.outcome == "accepted":
            try:
                accepted_attempt = self.validator(
                    self.config.batch_dir,
                    self.store.specification,
                    pending.cell,
                    pending.plan.run_dir,
                )
            except Exception as exc:
                self._open_attention(
                    pending,
                    category="accepted_artifact_invalid",
                    reason=f"accepted artifact validation raised {type(exc).__name__}",
                    location=Path(decision.attempt_dir or pending.plan.run_dir),
                    suggested_action=(
                        "inspect the sealed reference-scaffold evidence; repair the harness before requeue"),
                    evidence={"exception_type": type(exc).__name__},
                    execution_detail={"outcome": decision.as_dict(), "process": process},
                )
                return
            self.store.finish_accepted_execution(
                pending.plan.lease_id,
                accepted_attempt=dict(accepted_attempt),
                detail=dict(decision.detail),
            )
            return

        if decision.outcome == "retry_wait":
            self.store.finish_execution(
                pending.plan.lease_id,
                outcome="retry_wait",
                detail=dict(decision.detail),
                next_retry_at=decision.next_retry_at,
            )
            return

        if decision.outcome == "needs_attention" and isinstance(decision.attention, Mapping):
            attention = decision.attention
            self._open_attention(
                pending,
                category=str(attention.get("category") or decision.category),
                reason=str(attention.get("reason") or decision.reason),
                scope=attention.get("scope"),
                location=Path(str(attention.get("location") or pending.plan.execution_dir)),
                suggested_action=str(
                    attention.get("suggested_action") or "inspect and requeue after repair"),
                evidence=attention.get("evidence"),
                execution_detail=dict(decision.detail),
            )
            return

        self._open_attention(
            pending,
            category="outcome_contract_invalid",
            reason=f"classifier returned unsupported outcome {decision.outcome!r}",
            suggested_action="inspect the controller outcome contract before requeue",
            execution_detail={"process": process},
        )

    def _recovered_pending(
        self, lease: Mapping[str, Any], cell: Mapping[str, Any],
    ) -> _PendingExecution:
        cell_id = str(lease.get("cell_id") or "")
        model = str(cell.get("model") or "")
        task = str(cell.get("task") or "")
        attempt_index = cell.get("attempt_index")
        execution_id = str(lease.get("execution_id") or "")
        if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) \
                or attempt_index < 0:
            raise BatchControllerError("recovered cell attempt_index is invalid")
        if durable_cell_id(model, task, attempt_index) != cell_id:
            raise BatchControllerError("recovered lease and cell identity disagree")
        execution_number = _execution_number(execution_id)
        execution_dir = (
            Path(self.config.batch_dir).resolve() / "runs" / model / task
            / f"attempt-{attempt_index:03d}" / execution_id
        )
        run_dir = execution_dir / "run"
        persisted_run = lease.get("run_dir")
        if not isinstance(persisted_run, str) \
                or Path(persisted_run).resolve() != run_dir.resolve():
            raise BatchControllerError("recovered lease run_dir differs from durable identity")
        run_id = (
            f"b-{_safe_hash(str(self.store.specification.get('batch_id')), 12)}-"
            f"{_safe_hash(cell_id, 12)}-e{execution_number:03d}"
        )
        compose_project = f"codeaction-{run_id}-a{attempt_index:03d}"
        if lease.get("compose_project") != compose_project:
            raise BatchControllerError(
                "recovered lease Compose project differs from durable identity")
        plan = ExecutionPlan(
            cell_id=cell_id,
            lease_id=str(lease.get("lease_id") or ""),
            execution_id=execution_id,
            execution_dir=execution_dir,
            run_dir=run_dir,
            log_path=execution_dir / "controller.log",
            run_id=run_id,
            compose_project=compose_project,
            command=(),
            cleanup_command=(),
        )
        credential = self.scheduler.model_credentials.get(model)
        return _PendingExecution(plan, dict(lease), dict(cell), credential)

    def _settle_preserved(
        self,
        lease: Mapping[str, Any],
        cell: Mapping[str, Any],
        recovery_detail: Mapping[str, Any],
    ) -> str | None:
        """Settle a completed pre-crash artifact, or leave an unfinished run for retry."""
        self._assert_owner()
        pending = self._recovered_pending(lease, cell)
        attempt_index = int(cell["attempt_index"])
        scene_seed = cell.get("scene_seed")
        if not isinstance(scene_seed, int) or isinstance(scene_seed, bool) or scene_seed < 0:
            raise BatchControllerError("recovered cell scene_seed is invalid")
        attempt_dir = (
            pending.plan.run_dir
            / f"attempt-{attempt_index:03d}-seed-{scene_seed:06d}"
        )
        batch_root = Path(self.config.batch_dir).resolve()
        try:
            relative_attempt = attempt_dir.relative_to(batch_root)
        except ValueError as exc:
            raise BatchControllerError("preserved attempt path escapes the batch directory") from exc
        cursor = batch_root
        for part in relative_attempt.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise BatchControllerError("preserved attempt path traverses a symlink")
        status_path = attempt_dir / "controller_status.json"
        manifest_path = attempt_dir / MANIFEST_NAME
        if status_path.is_symlink() or manifest_path.is_symlink():
            raise BatchControllerError("preserved evidence marker is a symlink")
        if lease.get("operator_stop"):
            self._settle_result(pending, {"operator_stop": lease["operator_stop"],
                                         "return_code": -1, "recovery": dict(recovery_detail)})
            return "needs_attention"
        if not status_path.exists() and not manifest_path.exists():
            return None
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BatchControllerError("preserved controller status is unreadable") from exc
        if not isinstance(status, Mapping):
            raise BatchControllerError("preserved controller status is not an object")
        if status.get("state") != "complete":
            if manifest_path.exists():
                raise BatchControllerError(
                    "preserved artifact manifest exists without a complete controller status")
            return None
        self._settle_result(
            pending,
            {
                "return_code": -1,
                "timed_out": False,
                "cleanup_return_code": 0,
                "cleanup_timed_out": False,
                "error": "controller_recovered_after_crash",
                "recovery": dict(recovery_detail),
            },
        )
        current = self.store.snapshot()["cells"].get(pending.plan.cell_id, {})
        status_value = current.get("status")
        if status_value not in {"accepted", "retry_wait", "needs_attention"}:
            raise BatchControllerError("preserved artifact was not durably settled")
        return str(status_value)

    def _reconcile(self) -> Any:
        self._assert_owner()
        return self.reconciler(
            self.store,
            self.reporter,
            settle_preserved=self._settle_preserved,
        )

    def _write_result_views(self) -> None:
        self._assert_owner()
        self.result_writer(
            self.config.batch_dir,
            self.store.specification,
            self.store.snapshot(),
        )

    def _pending_for_lease(
        self, cell_id: str, lease: Mapping[str, Any], snapshot: Mapping[str, Any],
    ) -> _PendingExecution:
        cell = dict(snapshot["cells"][cell_id])
        plan = build_execution_plan(
            specification=self.store.specification,
            cell_id=cell_id,
            cell=cell,
            lease=lease,
            config=self.config,
        )
        credential = self.scheduler.model_credentials.get(cell["model"])
        return _PendingExecution(plan, dict(lease), cell, credential)

    def run(self) -> int:
        """Return 0 for this stage complete, 1 for attention-only, or 2 for controller error."""
        self._assert_owner()
        futures: dict[concurrent.futures.Future, _PendingExecution] = {}
        gpu_futures: dict[int, concurrent.futures.Future] = {}
        executor = None
        try:
            self._assert_stage()
            self._reconcile()
            self.reporter.sync_outputs()
            self._write_result_views()
            executor = self.thread_pool_factory(
                max_workers=len(self.config.gpus), thread_name_prefix="reference-supervisor")
            while True:
                self._assert_stage()
                self.control_applier(self.store, self.reporter)

                completed = [future for future in futures if future.done()]
                for future in completed:
                    pending = futures.pop(future)
                    gpu_futures.pop(int(pending.lease["gpu"]), None)
                    try:
                        process = future.result()
                    except BaseException as exc:
                        self._settle_worker_exception(pending, exc)
                        if not isinstance(exc, Exception):
                            raise
                        continue
                    try:
                        self._settle_result(pending, process)
                    except Exception as exc:
                        active = {
                            lease["lease_id"] for lease in self.store.active_leases()
                        }
                        if pending.plan.lease_id not in active:
                            raise
                        self._open_attention(
                            pending,
                            category="controller_settlement_failed",
                            reason=(
                                "controller could not durably settle a completed supervisor "
                                f"result: {type(exc).__name__}"),
                            suggested_action=(
                                "inspect the state transition and preserved execution before "
                                "requeue"),
                            evidence={"exception_type": type(exc).__name__},
                        )

                self._assert_stage()
                counts = self.store.stage_counts()
                if counts["accepted"] == counts["total"]:
                    self._write_result_views()
                    return 0

                decisions = []
                for gpu in self.config.gpus:
                    if gpu in gpu_futures:
                        continue
                    now = self.now()
                    decision = self.scheduler.try_acquire(
                        worker_id=f"gpu-{gpu}",
                        gpu=gpu,
                        now=now,
                        deadline_at=self._deadline_at(now),
                    )
                    decisions.append(decision)
                    if decision.status != "acquired":
                        continue
                    acquired_snapshot = self._assert_stage()
                    try:
                        pending = self._pending_for_lease(
                            str(decision.cell_id), decision.lease, acquired_snapshot)
                        future = executor.submit(self._launch, pending)
                    except Exception as exc:
                        cell = acquired_snapshot["cells"][str(decision.cell_id)]
                        fallback_plan = self._fallback_plan(
                            str(decision.cell_id), decision.lease, cell)
                        self._settle_worker_exception(
                            _PendingExecution(
                                fallback_plan, dict(decision.lease), dict(cell),
                                self.scheduler.model_credentials.get(cell["model"])),
                            exc,
                        )
                        continue
                    futures[future] = pending
                    gpu_futures[gpu] = future

                if futures:
                    concurrent.futures.wait(
                        tuple(futures),
                        timeout=self.config.poll_interval_s,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    continue

                counts = self.store.stage_counts()
                if counts["needs_attention"] and not counts["queued"] \
                        and not counts["retry_wait"] and not counts["running"]:
                    self._write_result_views()
                    return 1
                if decisions and all(
                        decision.status in {"needs_attention", "stage_complete"}
                        for decision in decisions):
                    self._write_result_views()
                    return 1 if counts["needs_attention"] else 0
                # No process is active.  retry_wait cooldown and operator control requests are the
                # only expected reasons to wait; this sleep does not own any GPU lease.
                threading.Event().wait(self.config.poll_interval_s)
        except BaseException as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                if self.store.active_leases():
                    self._reconcile()
                    self.reporter.sync_outputs()
            except Exception as recovery_exc:
                self.last_error += (
                    "; recovery failed: "
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
            if isinstance(exc, Exception):
                return 2
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    def _fallback_plan(
        self,
        cell_id: str,
        lease: Mapping[str, Any],
        cell: Mapping[str, Any],
    ) -> ExecutionPlan:
        """Name a failed pre-launch lease without creating or trusting a process command."""
        execution_id = str(lease.get("execution_id") or "execution-unknown")
        attempt_index = int(cell.get("attempt_index", 0))
        execution_dir = (
            Path(self.config.batch_dir).resolve() / "runs" / str(cell.get("model"))
            / str(cell.get("task")) / f"attempt-{attempt_index:03d}" / execution_id
        )
        return ExecutionPlan(
            cell_id=cell_id,
            lease_id=str(lease.get("lease_id") or ""),
            execution_id=execution_id,
            execution_dir=execution_dir,
            run_dir=execution_dir / "run",
            log_path=execution_dir / "controller.log",
            run_id="not-launched",
            compose_project="not-launched",
            command=(),
            cleanup_command=(),
        )


__all__ = [
    "BatchControllerError",
    "ControllerConfig",
    "ExecutionPlan",
    "BatchController",
    "build_execution_plan",
    "validate_controller_identity",
]
