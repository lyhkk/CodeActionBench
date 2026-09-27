"""reference-scaffold execution outcome policy for the durable benchmark controller."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from codeaction.contracts.failures import FailureCode, FailureRecord, failure_from_dict, failure_of


_HTTP_STATUS = re.compile(r"(?:^|; )http_status=(\d{3})(?:;|$)")


@dataclass(frozen=True)
class OutcomePolicy:
    """Bounded controller retries; provider adapters already own request-level retries."""

    # Provider adapters already retry the exact failed request without discarding the episode.
    # Replaying a whole paid episode after that bounded path fails is expensive and has no better
    # evidence that the provider recovered, so operator attention owns every full-run retry.
    rate_limit_max_executions: int = 1
    provider_max_executions: int = 1
    lifecycle_max_executions: int = 2
    rate_limit_delays_s: tuple[float, ...] = ()
    provider_delays_s: tuple[float, ...] = ()
    lifecycle_delays_s: tuple[float, ...] = (15.0,)

    def __post_init__(self) -> None:
        policies = (
            ("rate_limit", self.rate_limit_max_executions, self.rate_limit_delays_s),
            ("provider", self.provider_max_executions, self.provider_delays_s),
            ("lifecycle", self.lifecycle_max_executions, self.lifecycle_delays_s),
        )
        for name, maximum, delays in policies:
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
                raise ValueError(f"{name}_max_executions must be a positive integer")
            if len(delays) != maximum - 1 \
                    or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                           or not math.isfinite(value) or value < 0 for value in delays):
                raise ValueError(
                    f"{name}_delays_s must contain one finite non-negative delay per retry")


@dataclass(frozen=True)
class ExecutionOutcome:
    outcome: str
    category: str
    reason: str
    detail: Mapping[str, Any]
    next_retry_at: str | None = None
    attention: Mapping[str, Any] | None = None
    attempt_dir: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _retry_at(now: dt.datetime, seconds: float) -> str:
    if now.tzinfo is None:
        raise ValueError("outcome time must include a timezone")
    return (now + dt.timedelta(seconds=seconds)).isoformat()


def _http_status(failure: FailureRecord | None) -> int | None:
    if failure is not None and failure.http_status is not None:
        return failure.http_status
    detail = failure.detail_safe if failure is not None else None
    match = _HTTP_STATUS.search(detail or "")
    return int(match.group(1)) if match else None


def _attention_scope(
    failure: FailureRecord | None,
    *,
    cell_id: str,
    model: str,
    credential: str | None,
) -> dict[str, str]:
    code = failure.code if failure is not None else None
    if code in {
            FailureCode.PROVIDER_AUTH_FAILED,
            FailureCode.PROVIDER_QUOTA_EXHAUSTED,
            FailureCode.PROVIDER_RATE_LIMITED,
    } and credential:
        return {"kind": "credential", "id": credential}
    if code in {
            FailureCode.PROVIDER_AUTH_FAILED,
            FailureCode.PROVIDER_QUOTA_EXHAUSTED,
            FailureCode.PROVIDER_TIMEOUT,
            FailureCode.PROVIDER_SERVER_ERROR,
    }:
        return {"kind": "model", "id": model}
    if failure is not None \
            and failure.code == FailureCode.HARNESS_CONTRACT_VIOLATION \
            and failure.stage == "model_call":
        return {"kind": "model", "id": model}
    return {"kind": "cell", "id": cell_id}


def _needs_attention(
    *,
    category: str,
    reason: str,
    location: Path,
    scope: Mapping[str, str],
    detail: Mapping[str, Any],
    suggested_action: str,
    evidence: Mapping[str, Any] | None = None,
    attempt_dir: Path | None = None,
) -> ExecutionOutcome:
    return ExecutionOutcome(
        outcome="needs_attention",
        category=category,
        reason=reason,
        detail=dict(detail),
        attention={
            "category": category,
            "reason": reason,
            "scope": dict(scope),
            "location": str(location),
            "suggested_action": suggested_action,
            "evidence": dict(evidence or {}),
        },
        attempt_dir=str(attempt_dir) if attempt_dir is not None else None,
    )


def _retry_or_attention(
    *,
    category: str,
    reason: str,
    location: Path,
    cell_id: str,
    model: str,
    credential: str | None,
    execution_count: int,
    max_executions: int,
    delays: tuple[float, ...],
    now: dt.datetime,
    policy: OutcomePolicy,
    detail: Mapping[str, Any],
    failure: FailureRecord | None = None,
    attempt_dir: Path | None = None,
) -> ExecutionOutcome:
    if execution_count < max_executions:
        delay = delays[min(execution_count - 1, len(delays) - 1)]
        return ExecutionOutcome(
            outcome="retry_wait",
            category=category,
            reason=reason,
            detail=dict(detail),
            next_retry_at=_retry_at(now, delay),
            attempt_dir=str(attempt_dir) if attempt_dir is not None else None,
        )
    evidence = {"executions_used": execution_count}
    status = _http_status(failure)
    if status is not None:
        evidence.update({"http_status": status, "status_source": "provider_http"})
    return _needs_attention(
        category=category,
        reason=f"{reason}; retry budget exhausted",
        location=location,
        scope=_attention_scope(
            failure, cell_id=cell_id, model=model, credential=credential),
        detail=detail,
        suggested_action="inspect the preserved execution, repair the cause, then requeue",
        evidence=evidence,
        attempt_dir=attempt_dir,
    )


def classify_execution(
    *,
    cell_id: str,
    cell: Mapping[str, Any],
    process: Mapping[str, Any],
    run_dir: Path,
    credential: str | None,
    policy: OutcomePolicy,
    now: dt.datetime,
) -> ExecutionOutcome:
    """Classify one completed reference-scaffold process without inferring errors from HTTP prose."""
    run_dir = Path(run_dir).resolve()
    model = str(cell.get("model") or "")
    execution_count = cell.get("execution_count")
    if not isinstance(execution_count, int) or isinstance(execution_count, bool) \
            or execution_count < 1:
        raise ValueError("running cell must have a positive execution_count")
    process_detail = {str(key): value for key, value in process.items()}

    return_code = process.get("return_code")
    process_abnormal = return_code != 0 or process.get("timed_out") is True \
        or bool(process.get("error"))
    cleanup_proven = process.get("cleanup_return_code") == 0 \
        and process.get("cleanup_timed_out") is False
    cleanup_failed = process.get("cleanup_timed_out") is True \
        or process.get("cleanup_return_code") not in (None, 0) \
        or (process_abnormal and not cleanup_proven)
    if cleanup_failed:
        return _needs_attention(
            category="cleanup_failed",
            reason="the exact compose cleanup did not complete",
            location=run_dir,
            scope={"kind": "cell", "id": cell_id},
            detail={"process": process_detail},
            suggested_action=(
                "verify the recorded compose project is gone before requeueing this episode"),
            evidence={
                "cleanup_return_code": process.get("cleanup_return_code"),
                "cleanup_timed_out": bool(process.get("cleanup_timed_out")),
            },
        )

    attempt_index = cell.get("attempt_index")
    scene_seed = cell.get("scene_seed")
    if not isinstance(attempt_index, int) or not isinstance(scene_seed, int):
        raise ValueError("cell attempt_index and scene_seed must be integers")
    attempt_dir = run_dir / f"attempt-{attempt_index:03d}-seed-{scene_seed:06d}"
    controller = _read_object(attempt_dir / "controller_status.json")
    controller_complete = bool(controller) and controller.get("state") == "complete"
    agent_exit = _read_object(attempt_dir / 'agent_exit.json') or {}
    vendor_failure = failure_from_dict(agent_exit.get('failure'))
    if not controller_complete and vendor_failure is not None and vendor_failure.code in {
            FailureCode.PROVIDER_AUTH_FAILED, FailureCode.PROVIDER_QUOTA_EXHAUSTED}:
        # A rejected vendor request cannot produce a complete scoring attestation. Preserve
        # that failed evidence, but stop the paying account instead of retrying the episode.
        return _needs_attention(
            category=vendor_failure.code.value,
            reason=f'vendor access stopped: {vendor_failure.code.value}',
            location=attempt_dir,
            scope=_attention_scope(vendor_failure, cell_id=cell_id, model=model, credential=credential),
            detail={'process': process_detail, 'controller_status': controller or {},
                    'failure': vendor_failure.to_dict()},
            suggested_action='restore provider access or quota, then requeue the exact episode',
            evidence={'failure': vendor_failure.to_dict(), 'error_source': agent_exit.get('error_source')},
            attempt_dir=attempt_dir,
        )
    if not controller_complete:
        if process.get("timed_out") is True:
            return _retry_or_attention(
                category="execution_timeout",
                reason="the reference-scaffold execution exceeded its hard deadline",
                location=run_dir,
                cell_id=cell_id,
                model=model,
                credential=credential,
                execution_count=execution_count,
                max_executions=policy.lifecycle_max_executions,
                delays=policy.lifecycle_delays_s,
                now=now,
                policy=policy,
                detail={"process": process_detail},
            )
        if process.get("error"):
            return _retry_or_attention(
                category="supervisor_error",
                reason="the reference-scaffold process supervisor reported a local error",
                location=run_dir,
                cell_id=cell_id,
                model=model,
                credential=credential,
                execution_count=execution_count,
                max_executions=policy.lifecycle_max_executions,
                delays=policy.lifecycle_delays_s,
                now=now,
                policy=policy,
                detail={"process": process_detail},
            )
        mismatch = bool(controller) and any(
            controller.get(field) == "mismatch"
            for field in (
                "controller_identity_attestation", "gateway_attestation",
                "reference_agent_attestation", "raw_mcp_surface")
        )
        if mismatch:
            return _needs_attention(
                category="harness_attestation_failed",
                reason="controller or reference-scaffold harness attestation mismatched",
                location=attempt_dir,
                scope={"kind": "cell", "id": cell_id},
                detail={"process": process_detail, "controller_status": controller or {}},
                suggested_action="inspect attestation evidence and fix the harness before requeue",
            )
        lifecycle = (controller or {}).get("lifecycle_error") or (
            f"codeaction return code {return_code}")
        return _retry_or_attention(
            category="execution_lifecycle_failed",
            reason=str(lifecycle)[:500],
            location=attempt_dir,
            cell_id=cell_id,
            model=model,
            credential=credential,
            execution_count=execution_count,
            max_executions=policy.lifecycle_max_executions,
            delays=policy.lifecycle_delays_s,
            now=now,
            policy=policy,
            detail={"process": process_detail, "controller_status": controller or {}},
            attempt_dir=attempt_dir,
        )

    result = _read_object(attempt_dir / "result.json")
    agent_exit = _read_object(attempt_dir / "agent_exit.json") or {}
    stats = result.get("stats") if isinstance(result, Mapping) else None
    has_failure_contract = isinstance(result, Mapping) and (
        "failure" in result or (isinstance(stats, Mapping) and "failure" in stats)
    )
    if result is None or not has_failure_contract:
        return _needs_attention(
            category="harness_artifact_invalid",
            reason="complete reference-scaffold execution lacks a readable structured result",
            location=attempt_dir,
            scope={"kind": "cell", "id": cell_id},
            detail={"process": process_detail, "controller_status": controller},
            suggested_action="inspect result.json generation, then requeue this episode",
        )

    failure = failure_of(result, agent_exit)
    detail = {
        "process": process_detail,
        "controller_status": controller,
        "failure": failure.to_dict() if failure is not None else None,
    }
    if failure is None or failure.scoreable:
        return ExecutionOutcome(
            outcome="accepted",
            category="scoreable_episode",
            reason=("episode completed" if failure is None
                    else f"scoreable model outcome: {failure.code.value}"),
            detail=detail,
            attempt_dir=str(attempt_dir),
        )

    if failure.code in {
            FailureCode.PROVIDER_AUTH_FAILED,
            FailureCode.PROVIDER_QUOTA_EXHAUSTED,
    }:
        status = _http_status(failure)
        evidence = {"failure": failure.to_dict()}
        if status is not None:
            evidence.update({"http_status": status, "status_source": "provider_http"})
        return _needs_attention(
            category=failure.code.value,
            reason=f"non-retryable provider failure: {failure.code.value}",
            location=attempt_dir,
            scope=_attention_scope(
                failure, cell_id=cell_id, model=model, credential=credential),
            detail=detail,
            suggested_action="repair provider access or quota, then requeue the exact episode",
            evidence=evidence,
            attempt_dir=attempt_dir,
        )

    if failure.retryable:
        rate_limited = failure.code == FailureCode.PROVIDER_RATE_LIMITED
        delays = policy.rate_limit_delays_s if rate_limited else policy.provider_delays_s
        max_executions = (
            policy.rate_limit_max_executions
            if rate_limited else policy.provider_max_executions)
        return _retry_or_attention(
            category=failure.code.value,
            reason=f"retryable provider failure: {failure.code.value}",
            location=attempt_dir,
            cell_id=cell_id,
            model=model,
            credential=credential,
            execution_count=execution_count,
            max_executions=max_executions,
            delays=delays,
            now=now,
            policy=policy,
            detail=detail,
            failure=failure,
            attempt_dir=attempt_dir,
        )

    status = _http_status(failure)
    evidence = {"failure": failure.to_dict()}
    if status is not None:
        evidence.update({"http_status": status, "status_source": "provider_http"})
    return _needs_attention(
        category=failure.code.value,
        reason=f"non-scoreable {failure.origin.value} failure: {failure.code.value}",
        location=attempt_dir,
        scope=_attention_scope(
            failure, cell_id=cell_id, model=model, credential=credential),
        detail=detail,
        suggested_action="inspect the preserved failure evidence, repair, then requeue",
        evidence=evidence,
        attempt_dir=attempt_dir,
    )
