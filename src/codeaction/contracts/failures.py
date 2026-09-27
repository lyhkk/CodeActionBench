"""Structured failure contract shared by both benchmark tracks.

The legacy ``status`` strings remain the control-flow contract.  This module adds
an orthogonal, serializable explanation of *whose failure it was* and whether the attempt belongs
in a score denominator.  Callers must pass only bounded, credential-free text to ``detail_safe``;
raw provider responses and exception messages are intentionally never serialized here.
"""
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional


class FailureOrigin(str, Enum):
    MODEL = "model"
    AGENT = "agent"
    PROVIDER_INFRA = "provider_infra"
    HARNESS = "harness"
    ENVIRONMENT = "environment"


class FailureCode(str, Enum):
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    OUTPUT_LENGTH_EXCEEDED = "output_length_exceeded"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    INVALID_VIRTUAL_FILE_ARGUMENT = "invalid_virtual_file_argument"
    VIRTUAL_FILE_NOT_FOUND = "virtual_file_not_found"
    VIRTUAL_WORKSPACE_LIMIT = "virtual_workspace_limit"
    NO_TOOL_CALL = "no_tool_call"
    MODEL_REFUSED = "model_refused"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PHYSICAL_TIME_BUDGET_EXHAUSTED = "physical_time_budget_exhausted"
    WALL_BUDGET_EXHAUSTED = "wall_budget_exhausted"
    PROVIDER_AUTH_FAILED = "provider_auth_failed"
    PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_SERVER_ERROR = "provider_server_error"
    AGENT_TIMEOUT = "agent_timeout"
    NO_DONE_NORMAL_EXIT = "no_done_normal_exit"
    AGENT_CRASHED = "agent_crashed"
    AGENT_KILLED_BY_ENVIRONMENT = "agent_killed_by_environment"
    TOOL_RUNTIME_ERROR = "tool_runtime_error"
    CONTROL_CALL_WEDGE = "control_call_wedge"
    TOOL_WEDGE = "tool_wedge"
    VERIFIER_ERROR = "verifier_error"
    HARNESS_CONTRACT_VIOLATION = "harness_contract_violation"
    UNINTENDED_COLLISION = "unintended_collision"


class PhysicalTimeBudgetExhausted(RuntimeError):
    """Internal terminal signal raised after the threshold-crossing D0 tool has returned."""

    def __init__(self, *, physics_step, threshold_physics_steps, budget_s, sim_dt):
        self.physics_step = int(physics_step)
        self.threshold_physics_steps = int(threshold_physics_steps)
        self.budget_s = float(budget_s)
        self.sim_dt = float(sim_dt)
        super().__init__(
            f"physical execution time threshold {self.threshold_physics_steps} was reached; "
            f"atomic tool ended at step {self.physics_step}")


class ContactReadUnavailable(RuntimeError):
    """Internal signal that the contact state could not be measured.

    This is distinct from an empty contact list: callers may safely report zero only after the
    backend returned a readable empty list. The original backend exception stays chained for host
    debugging but is never copied into a model-facing payload.
    """

    def __init__(self):
        super().__init__("contact state unavailable")


class ProgramWorkspaceError(ValueError):
    """Recoverable virtual-workspace error with a stable model-visible subtype."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class FailureRecord:
    code: FailureCode
    origin: FailureOrigin
    stage: str
    retryable: bool
    scoreable: bool
    detail_safe: Optional[str] = None
    http_status: Optional[int] = None

    def to_dict(self) -> dict:
        out = {
            "code": self.code.value,
            "origin": self.origin.value,
            "stage": self.stage,
            "retryable": self.retryable,
            "scoreable": self.scoreable,
        }
        if self.detail_safe:
            out["detail_safe"] = self.detail_safe
        if self.http_status is not None:
            out["http_status"] = self.http_status
        return out


# One explicit default for every initial code.  Context/output-length defaults are conservative:
# later batches may promote them to scoreable model outcomes only after the effective limits have
# been recorded and shown to equal the declared limits.
_DEFAULTS = {
    FailureCode.CONTEXT_LENGTH_EXCEEDED:
        (FailureOrigin.HARNESS, "model_call", False, False),
    FailureCode.OUTPUT_LENGTH_EXCEEDED:
        (FailureOrigin.HARNESS, "model_turn", False, False),
    FailureCode.INVALID_TOOL_ARGUMENTS:
        (FailureOrigin.MODEL, "model_turn", False, True),
    FailureCode.INVALID_VIRTUAL_FILE_ARGUMENT:
        (FailureOrigin.MODEL, "tool_dispatch", False, True),
    FailureCode.VIRTUAL_FILE_NOT_FOUND:
        (FailureOrigin.MODEL, "tool_dispatch", False, True),
    FailureCode.VIRTUAL_WORKSPACE_LIMIT:
        (FailureOrigin.MODEL, "tool_dispatch", False, True),
    FailureCode.NO_TOOL_CALL:
        (FailureOrigin.MODEL, "model_turn", False, True),
    # A safety-classifier decline is the tested unit declining the task, so it is scored like any
    # other model-side termination: excluding it would let a model that refuses hard tasks look
    # better than one that attempts and fails them.  The provider's refusal category is recorded
    # in `detail_safe` so a false positive on a benign manipulation task stays identifiable.
    FailureCode.MODEL_REFUSED:
        (FailureOrigin.MODEL, "model_turn", False, True),
    FailureCode.BUDGET_EXHAUSTED:
        (FailureOrigin.MODEL, "episode", False, True),
    FailureCode.PHYSICAL_TIME_BUDGET_EXHAUSTED:
        (FailureOrigin.MODEL, "physical_execution", False, True),
    FailureCode.WALL_BUDGET_EXHAUSTED:
        (FailureOrigin.MODEL, "episode", False, True),
    FailureCode.PROVIDER_AUTH_FAILED:
        (FailureOrigin.PROVIDER_INFRA, "model_call", False, False),
    FailureCode.PROVIDER_QUOTA_EXHAUSTED:
        (FailureOrigin.PROVIDER_INFRA, "model_call", False, False),
    FailureCode.PROVIDER_RATE_LIMITED:
        (FailureOrigin.PROVIDER_INFRA, "model_call", True, False),
    FailureCode.PROVIDER_TIMEOUT:
        (FailureOrigin.PROVIDER_INFRA, "model_call", True, False),
    FailureCode.PROVIDER_SERVER_ERROR:
        (FailureOrigin.PROVIDER_INFRA, "model_call", True, False),
    FailureCode.AGENT_TIMEOUT:
        (FailureOrigin.AGENT, "agent_runtime", False, True),
    FailureCode.NO_DONE_NORMAL_EXIT:
        (FailureOrigin.AGENT, "agent_runtime", False, True),
    FailureCode.AGENT_CRASHED:
        (FailureOrigin.AGENT, "agent_runtime", False, True),
    FailureCode.AGENT_KILLED_BY_ENVIRONMENT:
        (FailureOrigin.ENVIRONMENT, "agent_runtime", False, False),
    FailureCode.TOOL_RUNTIME_ERROR:
        (FailureOrigin.ENVIRONMENT, "tool_dispatch", False, False),
    FailureCode.CONTROL_CALL_WEDGE:
        (FailureOrigin.AGENT, "control_loop", False, True),
    FailureCode.TOOL_WEDGE:
        (FailureOrigin.ENVIRONMENT, "tool_dispatch", False, False),
    FailureCode.VERIFIER_ERROR:
        (FailureOrigin.ENVIRONMENT, "verification", False, False),
    FailureCode.HARNESS_CONTRACT_VIOLATION:
        (FailureOrigin.HARNESS, "harness", False, False),
    # Historical tool surface 6.x-8.x terminal records remain parseable. Tool surface 9.0 emits the action-local
    # string code UNEXPECTED_CONTACT inside ActionResult instead of finalizing the episode.
    FailureCode.UNINTENDED_COLLISION:
        (FailureOrigin.MODEL, "tool_execution", False, True),
}

_DETAIL_LIMIT = 240
_CREDENTIALISH = re.compile(
    r"(?i)(bearer\s+|api[_-]?key\s*[=:]\s*|token\s*[=:]\s*)[^\s,;]+"
)


def bounded_detail(value: Any) -> Optional[str]:
    """Normalize caller-supplied safe evidence and cap it before persistence."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    text = _CREDENTIALISH.sub(r"\1[REDACTED]", text)
    return text[:_DETAIL_LIMIT] or None


def _validated_http_status(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 599:
        raise ValueError("http_status must be an integer from 100 through 599")
    return value


def default_failure(
    code: FailureCode | str,
    *,
    detail_safe: Any = None,
    origin: FailureOrigin | str | None = None,
    stage: str | None = None,
    retryable: bool | None = None,
    scoreable: bool | None = None,
    http_status: int | None = None,
) -> FailureRecord:
    """Construct a validated record, optionally overriding an explicitly context-dependent field."""
    code = code if isinstance(code, FailureCode) else FailureCode(code)
    base_origin, base_stage, base_retryable, base_scoreable = _DEFAULTS[code]
    if origin is not None:
        origin = origin if isinstance(origin, FailureOrigin) else FailureOrigin(origin)
    return FailureRecord(
        code=code,
        origin=origin or base_origin,
        stage=str(stage or base_stage),
        retryable=base_retryable if retryable is None else bool(retryable),
        scoreable=base_scoreable if scoreable is None else bool(scoreable),
        detail_safe=bounded_detail(detail_safe),
        http_status=_validated_http_status(http_status),
    )


def failure_from_dict(value: Any) -> Optional[FailureRecord]:
    """Read the additive 1.2 field. Unknown/invalid records fail closed as unreadable, not scoreable."""
    if not isinstance(value, Mapping):
        return None
    try:
        return FailureRecord(
            code=FailureCode(value["code"]),
            origin=FailureOrigin(value["origin"]),
            stage=str(value["stage"]),
            retryable=bool(value["retryable"]),
            scoreable=bool(value["scoreable"]),
            detail_safe=bounded_detail(value.get("detail_safe")),
            http_status=_validated_http_status(value.get("http_status")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def failure_of(result: Any, agent_exit: Any = None) -> Optional[FailureRecord]:
    """Resolve a persisted failure. Agent-exit evidence is authoritative for an ambiguous no_done."""
    if isinstance(agent_exit, Mapping):
        raw = agent_exit.get("failure")
        if raw is not None:
            parsed = failure_from_dict(raw)
            return parsed or default_failure(
                FailureCode.HARNESS_CONTRACT_VIOLATION,
                detail_safe="invalid failure record in agent_exit.json",
            )
    if not isinstance(result, Mapping):
        return None
    raw = result.get("failure")
    if raw is not None:
        parsed = failure_from_dict(raw)
        return parsed or default_failure(
            FailureCode.HARNESS_CONTRACT_VIOLATION,
            detail_safe="invalid failure record in result.json",
        )
    stats = result.get("stats")
    if isinstance(stats, Mapping) and stats.get("failure") is not None:
        return failure_from_dict(stats["failure"]) or default_failure(
            FailureCode.HARNESS_CONTRACT_VIOLATION,
            detail_safe="invalid failure record in result.stats",
        )
    status = (stats.get("status") if isinstance(stats, Mapping) else None) or result.get("status")
    has_12_contract = "failure" in result or (
        isinstance(stats, Mapping) and "failure" in stats)
    if has_12_contract and status == "no_done":
        return default_failure(
            FailureCode.HARNESS_CONTRACT_VIOLATION,
            stage="agent_runtime",
            detail_safe="schema 1.2 no_done is missing agent_exit classification",
        )
    return None


def _exception_chain(exc: BaseException) -> Iterable[BaseException]:
    """Yield an exception and its explicit/implicit causes once each."""
    pending = [exc]
    seen = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException) and context is not cause:
            pending.append(context)


def _exception_status(exc: BaseException) -> Optional[int]:
    for current in _exception_chain(exc):
        for value in (
            getattr(current, "status_code", None),
            getattr(current, "http_status", None),
            getattr(getattr(current, "response", None), "status_code", None),
        ):
            try:
                if value is not None:
                    parsed = int(value)
                    if 100 <= parsed <= 599:
                        return parsed
            except (TypeError, ValueError):
                pass
    return None


def _exception_code(exc: BaseException) -> str:
    for current in _exception_chain(exc):
        value = getattr(current, "code", None)
        if value is None:
            body = getattr(current, "body", None)
            if isinstance(body, Mapping):
                error = body.get("error")
                value = error.get("code") if isinstance(error, Mapping) else body.get("code")
        if value is not None:
            return str(value).strip().lower()
    return ""


class EmptyProviderCompletion(RuntimeError):
    """A 200 response that carried no completion choice.

    Its own class is what makes it classifiable: an unguarded index into the missing choice list
    raises TypeError or IndexError, both of which land in the catch-all below as a non-retryable
    provider server error and end the episode. An empty completion is transient, so it belongs on
    the ordinary bounded-retry path instead.
    """


def ensure_usable_provider_completion(
    message: Mapping[str, Any], stop_reason: str, *, provider: str,
    auxiliary_content: Any = None,
) -> None:
    """Keep empty provider turns out of the loop's model-owned ``no_tool_call`` path.

    Every adapter normalizes to the same canonical message before calling this function. A
    length/refusal response is terminal evidence even when it has no content; every other turn
    must contain text or at least one tool call. Missing provider output is transient and follows
    the ordinary bounded provider retry policy.
    """
    content = message.get("content")
    has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
    has_auxiliary = (
        bool(auxiliary_content.strip())
        if isinstance(auxiliary_content, str)
        else bool(auxiliary_content)
    )
    if has_content or bool(message.get("tool_calls")) or has_auxiliary:
        return
    if str(stop_reason) in ("length", "refusal"):
        return
    raise EmptyProviderCompletion(f"{provider} returned no usable completion")


def _raise_site(exc: BaseException) -> str:
    """`file.py:line in function` for the frame that raised, or "" when unavailable.

    Provider error BODIES stay unpersisted because they can echo prompt text back. A local
    exception's own code location carries no run data at all -- it names our file and our line --
    so recording it is free of that risk and turns a bare `detail_safe: "TypeError"` into
    something diagnosable without a second paid run.
    """
    tb = getattr(exc, "__traceback__", None)
    if tb is None:
        return ""
    while tb.tb_next is not None:
        tb = tb.tb_next
    frame = tb.tb_frame
    return f"{Path(frame.f_code.co_filename).name}:{tb.tb_lineno} in {frame.f_code.co_name}"


def classify_provider_exception(exc: BaseException) -> FailureRecord:
    """Classify without persisting the raw exception message.

    The message is consulted only because OpenAI-compatible gateways do not consistently expose a
    typed error code.  Persisted detail is restricted to exception class, HTTP status, and typed
    provider code.
    """
    status = _exception_status(exc)
    provider_code = _exception_code(exc)
    chain = tuple(_exception_chain(exc))
    name = type(exc).__name__
    evidence = " ".join(
        [type(current).__name__ for current in chain]
        + [provider_code]
        + [str(current) for current in chain]
    ).lower()
    detail = f"{name}" + (f"; http_status={status}" if status is not None else "") + (
        f"; provider_code={provider_code}" if provider_code else "")

    def classified(code: FailureCode, **overrides: Any) -> FailureRecord:
        return default_failure(code, http_status=status, **overrides)

    # Structured HTTP evidence always outranks provider prose. Gateways routinely echo request
    # text, so a 503 whose body mentions "context window" or "rate limit" is still a 503. Allowing
    # the echoed text to win can force irreversible context compaction or the wrong cooldown.
    # Several OpenAI-compatible providers report a depleted paid balance as HTTP 429.  Exact
    # machine-readable billing codes outrank the transport status: retrying those responses as an
    # RPM event can waste the rest of a long batch, whereas an unqualified 429 remains transient.
    if provider_code in {
            "billing_hard_limit_reached",
            "credit_balance_insufficient",
            "credit_balance_too_low",
            "insufficient_quota",
            "quota_exhausted",
    }:
        return classified(FailureCode.PROVIDER_QUOTA_EXHAUSTED, detail_safe=detail)
    if status == 402:
        return classified(FailureCode.PROVIDER_QUOTA_EXHAUSTED, detail_safe=detail)
    if status in (401, 403):
        return classified(FailureCode.PROVIDER_AUTH_FAILED, detail_safe=detail)
    if status == 408:
        return classified(FailureCode.PROVIDER_TIMEOUT, detail_safe=detail)
    if status == 429:
        return classified(FailureCode.PROVIDER_RATE_LIMITED, detail_safe=detail)
    if status is not None and 500 <= status <= 599:
        return classified(FailureCode.PROVIDER_SERVER_ERROR, detail_safe=detail)

    # Some streaming SDKs surface capacity/quota pressure only as the structured gRPC-style code
    # and provide no HTTP status. This is provider evidence, not prose: treating it as an unknown
    # local exception incorrectly attributes a transient provider failure to the harness.
    if provider_code in {"resource_exhausted", "resource-exhausted"}:
        return classified(FailureCode.PROVIDER_RATE_LIMITED, detail_safe=detail)

    if any(isinstance(current, EmptyProviderCompletion) for current in chain):
        return classified(
            FailureCode.PROVIDER_SERVER_ERROR, retryable=True, detail_safe=detail)

    typed_context_evidence = (
        "context_length" in provider_code
        or "maximum_context" in provider_code
    )
    explicit_context_text = bool(re.search(
        r"\b(?:maximum context length(?: is)? exceeded|"
        r"context length(?: is)? (?:exceeded|too (?:large|long))|"
        r"context window(?: is)? (?:exceeded|too (?:large|long))|"
        r"too many (?:input )?tokens)\b",
        evidence,
    ))
    if typed_context_evidence or explicit_context_text:
        return classified(FailureCode.CONTEXT_LENGTH_EXCEEDED, detail_safe=detail)

    # DashScope performs data inspection before Qwen inference. Once the harness-side PNG/Base64
    # preflight has accepted the payload, this provider code is a content-policy refusal rather
    # than malformed transport. Retrying the same bytes does not repair it, and calling it a
    # harness contract violation incorrectly blames the request schema.
    if provider_code in ("data_inspection_failed", "datainspectionfailed"):
        return classified(
            FailureCode.MODEL_REFUSED,
            stage="model_call",
            detail_safe=detail,
        )
    if status in (400, 404, 405, 409, 415, 422):
        return classified(
            FailureCode.HARNESS_CONTRACT_VIOLATION,
            stage="model_call",
            detail_safe=detail,
        )

    if status is None and (
            "authentication" in evidence or "permissiondenied" in evidence):
        return classified(FailureCode.PROVIDER_AUTH_FAILED, detail_safe=detail)
    if status is None and ("ratelimit" in evidence or "rate limit" in evidence):
        return classified(FailureCode.PROVIDER_RATE_LIMITED, detail_safe=detail)
    if status is None and (
            any(isinstance(current, TimeoutError) for current in chain)
            or any("timeout" in type(current).__name__.lower() for current in chain)
    ):
        phase = next(
            (str(getattr(current, "timeout_phase")) for current in chain
             if getattr(current, "timeout_phase", None)),
            None,
        )
        return classified(
            FailureCode.PROVIDER_TIMEOUT,
            stage=f"model_call.{phase}" if phase else "model_call",
            detail_safe=detail,
        )

    # A dropped connection is typed evidence, and it is exactly the transient failure the retry
    # budget exists for: DNS, TCP reset, TLS, a socket closed mid-response. Without this branch it
    # fell through to the catch-all below and came back as a harness contract violation with
    # retryable=False, so one network blip ended an episode with all three attempts unspent.
    # Ordered AFTER the timeout branch on purpose: the OpenAI SDK's APITimeoutError subclasses
    # APIConnectionError and httpx's ConnectTimeout is a ConnectError, so a timeout stays a
    # timeout by having already returned above.
    if status is None and (
            any(isinstance(current, ConnectionError) for current in chain)
            or any("connect" in type(current).__name__.lower() for current in chain)
    ):
        return classified(FailureCode.PROVIDER_SERVER_ERROR, detail_safe=detail)

    # A typed client status is enough to identify a request-contract failure. Without a status,
    # text fallback is deliberately narrow: the word "unsupported" alone also appears in
    # temporary region failover and capacity messages.
    specific_contract_text = bool(
        re.search(r"\b(?:bad|invalid) request\b", evidence)
        or re.search(r"\bunsupported (?:argument|field|parameter|schema|model)\b", evidence)
        or re.search(r"\b(?:argument|field|parameter|schema)\b.{0,40}\bnot supported\b", evidence)
    )
    if status is None and specific_contract_text:
        return classified(
            FailureCode.HARNESS_CONTRACT_VIOLATION,
            stage="model_call",
            detail_safe=detail,
        )
    # With no HTTP/code/typed evidence, this is a local boundary failure, not proof of provider
    # infrastructure trouble. It remains non-scoreable but is attributed to the harness. Put the
    # raise site first so a long provider_code cannot truncate the only actionable evidence.
    site = _raise_site(exc)
    return classified(
        FailureCode.HARNESS_CONTRACT_VIOLATION,
        origin=FailureOrigin.HARNESS,
        stage="model_call",
        retryable=False,
        detail_safe=(
            f"{name}; raise_site={site}"
            + (f"; provider_code={provider_code}" if provider_code else "")
        ) if site else detail,
    )


class ProviderCallError(RuntimeError):
    """Safe terminal wrapper used across the provider/episode boundary."""

    def __init__(self, failure: FailureRecord):
        self.failure = failure
        super().__init__(f"provider call failed: {failure.code.value}")


def failure_from_legacy_status(status: Any, *, tested_origin: str) -> Optional[FailureRecord]:
    """Explicit Batch-1 mapping that preserves every legacy terminal status.

    ``no_done`` is deliberately unresolved: only the external controller knows whether the vendor
    CLI exited normally, crashed, or was killed by its environment.
    """
    status = str(status or "")
    tested_origin = (FailureOrigin.AGENT if str(tested_origin).lower() == "agent"
                     else FailureOrigin.MODEL)
    if status in ("", "running", "done"):
        return None
    if status == "budget_exhausted":
        return default_failure(FailureCode.BUDGET_EXHAUSTED, origin=tested_origin)
    if status == "physical_time_budget_exhausted":
        return default_failure(
            FailureCode.PHYSICAL_TIME_BUDGET_EXHAUSTED,
            origin=tested_origin,
        )
    if status == "wall_budget":
        return default_failure(FailureCode.WALL_BUDGET_EXHAUSTED, origin=tested_origin)
    if status == "no_tool_calls":
        return default_failure(FailureCode.NO_TOOL_CALL)
    if status == "episode_fatal":
        return default_failure(FailureCode.TOOL_RUNTIME_ERROR)
    if status == "tool_wedge":
        return default_failure(FailureCode.TOOL_WEDGE)
    if status == "control_call_wedge":
        return default_failure(FailureCode.CONTROL_CALL_WEDGE)
    if status == "unintended_collision":
        return default_failure(FailureCode.UNINTENDED_COLLISION, origin=tested_origin)
    if status == "agent_timeout":
        return default_failure(FailureCode.AGENT_TIMEOUT)
    if status == "verifier_error":
        return default_failure(FailureCode.VERIFIER_ERROR)
    if status in ("no_done", "endpoint_failure"):
        return None
    return default_failure(
        FailureCode.HARNESS_CONTRACT_VIOLATION,
        detail_safe=f"unmapped legacy status: {status}",
    )


def classify_agent_exit(record: Mapping[str, Any]) -> Optional[FailureRecord]:
    """Map structured client/container/server evidence to the vendor-agent terminal failure."""
    server_status = str(record.get("server_status") or "")
    container = record.get("container") if isinstance(record.get("container"), Mapping) else {}
    if server_status == "tool_wedge":
        return default_failure(FailureCode.TOOL_WEDGE)
    if server_status == "verifier_error":
        return default_failure(FailureCode.VERIFIER_ERROR)
    if any(bool(record.get(k)) for k in ("transport_lost", "controller_killed")) \
            or any(bool(container.get(k)) for k in ("oom_killed", "killed", "transport_lost")):
        return default_failure(FailureCode.AGENT_KILLED_BY_ENVIRONMENT)
    if server_status in (
            "done", "budget_exhausted", "wall_budget", "episode_fatal",
            "unintended_collision"):
        # The episode server already finalized these states and owns their classification.
        return None

    status = record.get("api_error_status")
    subtype = str(record.get("client_subtype") or "").strip().lower()
    if subtype == 'usage_limit_reached' or status in ('usage_limit_reached', 'insufficient_quota', 'quota_exhausted'):
        return default_failure(FailureCode.PROVIDER_QUOTA_EXHAUSTED, stage='agent_runtime')
    if subtype in ("authentication_failed", "not_logged_in", "permission_denied"):
        return default_failure(FailureCode.PROVIDER_AUTH_FAILED, stage="agent_runtime")
    if subtype in ("rate_limited", "quota_exceeded", "session_limit"):
        return default_failure(FailureCode.PROVIDER_RATE_LIMITED, stage="agent_runtime")
    try:
        status = int(status) if status not in (None, "") else None
    except (TypeError, ValueError):
        status = None
    if status in (401, 403):
        return default_failure(FailureCode.PROVIDER_AUTH_FAILED, stage="agent_runtime")
    if status == 429:
        return default_failure(FailureCode.PROVIDER_RATE_LIMITED, stage="agent_runtime")
    if status == 408:
        return default_failure(FailureCode.PROVIDER_TIMEOUT, stage="agent_runtime")
    if status is not None and 500 <= status <= 599:
        return default_failure(FailureCode.PROVIDER_SERVER_ERROR, stage="agent_runtime")
    if bool(record.get("timed_out")):
        return default_failure(FailureCode.AGENT_TIMEOUT)

    try:
        cli_exit_code = int(record.get("cli_exit_code"))
    except (TypeError, ValueError):
        cli_exit_code = None
    if server_status == "no_done" and cli_exit_code == 0:
        return default_failure(FailureCode.NO_DONE_NORMAL_EXIT)
    if cli_exit_code not in (None, 0):
        return default_failure(
            FailureCode.AGENT_CRASHED,
            detail_safe=f"cli_exit_code={cli_exit_code}",
        )
    return None


def with_detail(failure: FailureRecord, detail_safe: Any) -> FailureRecord:
    """Return a copy with bounded safe evidence; useful at a boundary that adds fixed context."""
    return replace(failure, detail_safe=bounded_detail(detail_safe))


def is_recoverable_action_abort(payload) -> bool:
    """Whether one structured action returned ABORTED while the episode remains active."""
    if not isinstance(payload, dict):
        return False
    if payload.get("status") != "ABORTED":
        return False
    if isinstance(payload.get("action_id"), str):
        return True
    interrupted = payload.get("interrupted_action")
    return isinstance(interrupted, dict) and interrupted.get("status") == "ABORTED"
