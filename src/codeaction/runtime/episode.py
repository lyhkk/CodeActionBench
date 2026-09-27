"""Sequential episode execution under the current release contract.

Reference and vendor profiles share budget, argument and termination rules. They differ
in task delivery, image transport and the attribution of agent/model failures.
"""
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional

from codeaction.runtime.argcheck import ArgumentContractError, validate_arguments
from codeaction.contracts.failures import (FailureCode, FailureRecord, PhysicalTimeBudgetExhausted,
                              ProgramWorkspaceError, default_failure,
                              failure_from_legacy_status, is_recoverable_action_abort)
from codeaction.contracts.result_contracts import validate_result
from codeaction.contracts.harness_parameters import declared_harness_parameters
from codeaction.interface.schemas import (GRASP_EVIDENCE_NOTE, NONCONTACT_SCALE_NOTE, PROGRAM_TOOL_NAMES,
                             RUN_CODE_INTERFACE_NOTE)
from codeaction.contracts.tool_results import (FOLLOWUP_IMAGE_TRANSPORT, INLINE_IMAGE_TRANSPORT,
                                  RUN_CODE_RESULT_MAX_IMAGES, ToolProjection,
                                  payload_projection, project_tool_result)


@dataclass(frozen=True)
class EpisodeContractProfile:
    name: str
    # Which side owns a terminal status recorded before the failure
    # taxonomy existed: the vendor agent stack, or the model itself.
    tested_origin: str
    task_delivery: str
    image_transport: str
    contract_version: str = "2.0.1"


CANONICAL_DONE_ACK = {"ok": True, "note": "episode ended; attempt recorded."}

REFERENCE_EPISODE_CONTRACT = EpisodeContractProfile(
    name="reference_episode_contract_2_0_1",
    tested_origin="model",
    task_delivery="initial_user_message",
    image_transport=FOLLOWUP_IMAGE_TRANSPORT,
    contract_version="2.0.1",
)

VENDOR_EPISODE_CONTRACT = EpisodeContractProfile(
    name="vendor_episode_contract_2_0_1",
    tested_origin="agent",
    task_delivery="initial_controller_message",
    image_transport=INLINE_IMAGE_TRANSPORT,
    contract_version="2.0.1",
)


# Episode control, not capability: reachable on every surface, so the delivered-tool allowlist
# never applies to them.
CONTROL_TOOL_NAMES = frozenset({"done"})


@dataclass(frozen=True)
class RuntimeCall:
    name: str
    args: Any
    projection: ToolProjection
    step: int
    charged: bool
    kind: str = "tool"
    failure: Optional[FailureRecord] = None
    error_detail: Optional[str] = None
    # What an interrupted code block had already executed when an episode terminal cut it off.
    # Recorded beside the terminal, never inside its payload: `payload_projection` builds the
    # model-visible copy from the same dict, so anything placed there would reach the model.
    interrupted_partial: Optional[dict] = None

    @property
    def payload(self) -> dict:
        return self.projection.payload


class EpisodeRuntime:
    """Own common dispatch, locking, budgets, state, and exactly-once finalization."""

    def __init__(
        self,
        toolbox,
        task_text,
        *,
        profile: EpisodeContractProfile,
        sandbox=None,
        max_steps=40,
        max_tool_calls=None,
        wall_budget_s=900.0,
        delivered_tools: Optional[Iterable[str]] = None,
        on_finalize: Optional[Callable[[str, Any], None]] = None,
        before_finalize: Optional[Callable[[str, dict], None]] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.toolbox = toolbox
        self.task_text = str(task_text)
        self.profile = profile
        self.sandbox = sandbox
        self.max_steps = int(max_steps if max_tool_calls is None else max_tool_calls)
        self.max_tool_calls = self.max_steps
        self.wall_budget_s = float(wall_budget_s)
        self.registry = toolbox.registry_map()
        # The delivered surface must be an allowlist, not a hint. The registry backs every
        # primitive whatever the interface delivers, so without this a model that merely GUESSES
        # an undelivered name gets it executed: the code-first arm reached the robot 110 times
        # that way (14 real motions) through an interface that delivered two camera tools, which
        # is why its "documented library vs schema" contrast never actually ran. Only names on
        # the delivered surface reach the registry; `run_code` still binds the full library
        # inside the sandbox, which is the point of a code-first surface.
        self.delivered_tools = (
            None if delivered_tools is None else frozenset(str(n) for n in delivered_tools))
        self.steps = 0
        self.total_calls = 0
        # Calls the reference agent charged but never sent to the simulator: contract-rejected
        # arguments and the budget-exhausting call. Recorded so the two counters can differ without
        # the difference being silent.
        self.reference_only_calls = 0
        self.reference_only_budget = 0
        self.status = "running"
        self.done_report = None
        self.failure = None
        self._finalized = False
        self._on_finalize = on_finalize
        self._before_finalize = before_finalize
        self._clock = clock
        # `_t0` is the budget origin and moves forward when provider-infra time is credited back;
        # `_t0_true` never moves, so `wall_s` stays an honest stopwatch reading.
        self._t0 = clock()
        self._t0_true = self._t0
        self.wall_credit_s = 0.0
        self._lock = threading.RLock()

    def restrict_to(self, delivered_tools: Iterable[str]) -> None:
        """Pin the delivered surface once the caller knows what it handed the model.

        The surface is usually derived from the registry, which the runtime owns and reads exactly
        once, so the allowlist cannot be a constructor argument without a second `registry_map()`
        call. Pinning is only legal before the first dispatch.
        """
        with self._lock:
            if self.steps or self.total_calls:
                raise ValueError("the delivered surface must be pinned before the first call")
            self.delivered_tools = frozenset(str(n) for n in delivered_tools)

    def _is_delivered(self, name: str) -> bool:
        """Whether an outer call names a tool this interface actually handed the model.

        ``done`` stays reachable whatever the surface delivers.
        under the frozen vendor-agent legacy profile; current profiles receive the same information in
        their initial episode configuration and do not expose that control tool.
        """
        if self.delivered_tools is None:
            return True
        name = str(name)
        return (
            name in self.delivered_tools
            or name in CONTROL_TOOL_NAMES
        )

    @property
    def budget_used(self) -> int:
        return self.steps

    @property
    def over(self) -> bool:
        return self.status != "running"

    @property
    def wall_s(self) -> float:
        """True elapsed time, credits included: what a stopwatch on the attempt would read."""
        return round(self._clock() - self._t0_true, 1)

    @property
    def budget_wall_s(self) -> float:
        """Elapsed time the episode is answerable for -- what the wall budget is compared against."""
        return round(self._clock() - self._t0, 1)

    def credit_wall_budget(self, seconds: float) -> float:
        """Give back time a provider outage stole, and report how much was actually given.

        The wall budget exists to bound the EPISODE -- the model's own deliberation and the sim's
        execution. Time the harness spends asleep between provider retries is neither, and charging
        it produced a failure classified `origin=model, scoreable=True`: a vendor's rate limiter
        could exhaust the model's budget and then be scored as the model's inability to finish.

        The credit is bounded rather than open-ended. An unbounded one would let a provider outage
        extend an attempt indefinitely, which is exactly what an unattended batch must not do, so
        an attempt can be extended by at most its own wall budget and no further.
        """
        with self._lock:
            amount = float(seconds)
            if not amount > 0.0 or self.over:
                return 0.0
            headroom = float(self.wall_budget_s) - self.wall_credit_s
            applied = min(amount, max(headroom, 0.0))
            if applied <= 0.0:
                return 0.0
            self.wall_credit_s = round(self.wall_credit_s + applied, 3)
            self._t0 += applied
            return applied

    def reconcile_external_total_calls(self, total_calls: int) -> None:
        """Adopt the reference-side call count unconditionally; record the difference.

        The budget is denominated in what the AGENT spent, so the reference agent's count is the
        record and the simulator aligns to it. The agent charges calls the simulator never sees:
        every call it rejects against the argument contract before dispatch, plus the one that
        exhausts the budget. Those are the agent's own doing and belong in its budget.

        Nothing here raises. Reconciliation runs inside the finalize control, so an exception
        would fail the control, leave the simulator to finalize itself as ``no_done``, and lose a
        completed episode to an attestation mismatch -- twice already the cost of validating here
        (a malformed call before budget exhaustion, then the same on the budget counter). The
        difference is kept as SIGNED evidence instead: positive is the normal excess, and negative
        would mean the simulator executed calls the agent never made, which is a defect the
        attestation reports rather than a reason to discard the attempt.
        """
        with self._lock:
            observed = int(total_calls)
            self.reference_only_calls = observed - self.total_calls
            self.total_calls = observed

    def reconcile_external_budget_used(self, budget_used: int) -> None:
        """Adopt the reference-side chargeable-call count unconditionally; record the difference.

        Same rule and same non-raising contract as ``reconcile_external_total_calls``: a call
        rejected against the argument contract is charged by the agent and never reaches the
        simulator, and the agent's number is the budget record.
        """
        with self._lock:
            observed = int(budget_used)
            self.reference_only_budget = observed - self.steps
            self.steps = observed

    def _successful_control_projection(self, name: str, payload: dict) -> ToolProjection:
        """Validate control results for the current episode contract."""
        if getattr(self.toolbox, "enforce_result_contracts", False):
            validate_result(name, payload)
        return payload_projection(payload)

    def check_wall_budget(self) -> bool:
        """Finalize on wall exhaustion. Returns True only when the episode may continue."""
        with self._lock:
            if self.over:
                return False
            if self._clock() - self._t0 <= self.wall_budget_s:
                return True
            self.finalize("wall_budget")
            return False

    @staticmethod
    def _canonical_done_report(args: Any) -> dict:
        """Validate the irreversible control input and preserve the caller's exact values."""
        if not isinstance(args, Mapping):
            raise ArgumentContractError(
                f"done: arguments must be object, got {type(args).__name__}")
        validate_arguments("done", args)
        return dict(args)

    def _already_over_payload(self, *, done=False) -> dict:
        if done:
            return dict(CANONICAL_DONE_ACK)
        return {"error": f"EPISODE_OVER ({self.status}): the attempt has ended; "
                         "no further tool calls are executed."}

    def finalize(
        self,
        status,
        *,
        failure: Optional[FailureRecord] = None,
        context: Optional[dict] = None,
    ) -> bool:
        """Finalize once; return whether this invocation performed the transition."""
        with self._lock:
            if self._finalized:
                return False
            if self._before_finalize is not None:
                self._before_finalize(str(status), dict(context or {}))
            self._finalized = True
            self.status = str(status)
            self.failure = failure or failure_from_legacy_status(
                self.status,
                tested_origin=self.profile.tested_origin,
            )
            if self._on_finalize is not None:
                self._on_finalize(self.status, self.done_report)
            return True

    def disconnect(self) -> bool:
        return self.finalize("no_done")

    def _physical_time_terminal_call(self, name, args, exc):
        # A terminal raised from inside run_code discarded everything the block had already done:
        # `Sandbox.run` never returns, so its trace -- a local -- went with it, and the primitives
        # that moved the scene before the threshold left no record. The sandbox now carries them
        # on the exception.
        partial = {key: value for key, value in (
            ("internal_trace", getattr(exc, "internal_trace", None)),
            ("tool_calls", getattr(exc, "tool_calls", None)),
            ("obs_ids", getattr(exc, "obs_ids", None)),
        ) if value is not None}
        failure = default_failure(
            FailureCode.PHYSICAL_TIME_BUDGET_EXHAUSTED,
            detail_safe=(
                f"physics_steps={exc.physics_step}; "
                f"threshold_physics_steps={exc.threshold_physics_steps}; "
                f"budget_s={exc.budget_s:g}; sim_dt={exc.sim_dt:g}"),
        )
        call = RuntimeCall(
            name,
            args,
            payload_projection({
                "error": (
                    "EPISODE_OVER (physical_time_budget_exhausted): "
                    "the task-level physical execution time is exhausted."),
            }),
            self.steps,
            True,
            kind="terminal",
            failure=failure,
            interrupted_partial=partial or None,
        )
        self.finalize(
            "physical_time_budget_exhausted",
            failure=failure,
            context={"call": call},
        )
        return call

    def cancel_after_recoverable_abort(self, name, args) -> RuntimeCall:
        """Account for one same-turn direct call without executing it in the simulator."""
        with self._lock:
            name = str(name)
            self.total_calls += 1
            charged = name != "done" and self.steps < self.max_tool_calls
            if charged:
                self.steps += 1
            return RuntimeCall(
                name,
                args,
                payload_projection({
                    # Reason-neutral on purpose: every structured action abort uses this barrier.
                    "error": (
                        "not executed: an earlier call in this assistant turn returned a "
                        "structured action abort; inspect that result and replan in the next "
                        "turn"),
                    "executed": False,
                    "cancelled_by": "action_abort",
                }),
                self.steps,
                charged,
                kind="cancelled",
            )

    def dispatch(self, name, args, *, malformed_arguments=False) -> RuntimeCall:
        """Execute one call under the current episode contract."""
        with self._lock:
            name = str(name)
            self.total_calls += 1
            done_argument_error = None
            if name == "done":
                if self.over:
                    return RuntimeCall(
                        name, args, payload_projection(self._already_over_payload(done=True)),
                        self.steps, False, kind="over")
                if not malformed_arguments:
                    try:
                        done_report = self._canonical_done_report(args)
                    except ArgumentContractError as exc:
                        done_argument_error = exc
                    else:
                        self.done_report = done_report
                        ack = dict(CANONICAL_DONE_ACK)
                        call = RuntimeCall(
                            name, args, self._successful_control_projection(name, ack),
                            self.steps, False, kind="done")
                        self.finalize("done", context={"call": call})
                        return call

            if self.over:
                return RuntimeCall(
                    name, args, payload_projection(self._already_over_payload()),
                    self.steps, False, kind="over")

            if self._clock() - self._t0 > self.wall_budget_s:
                self.finalize("wall_budget")
                return RuntimeCall(
                    name, args,
                    payload_projection({
                        "error": "EPISODE_OVER (wall_budget): the time budget is exhausted."}),
                    self.steps, False, kind="limit", failure=self.failure)
            if self.steps >= self.max_tool_calls:
                self.finalize("budget_exhausted")
                return RuntimeCall(
                    name, args,
                    payload_projection({
                        "error": "EPISODE_OVER (budget_exhausted): the step budget "
                                 f"({self.max_tool_calls} tool calls) is exhausted."}),
                    self.steps, False, kind="limit", failure=self.failure)

            self.steps += 1
            result_obj = None
            code_observation_ids = ()
            call_failure = None
            undelivered_detail = None
            if malformed_arguments:
                payload = {"error": "tool arguments were not valid JSON"}
                call_failure = default_failure(FailureCode.INVALID_TOOL_ARGUMENTS)
                projection = payload_projection(payload)
            elif done_argument_error is not None:
                projection = payload_projection({
                    "error": f"bad arguments: {done_argument_error}",
                })
                call_failure = default_failure(FailureCode.INVALID_TOOL_ARGUMENTS)
            elif not self._is_delivered(name):
                # Model-visible text is the SAME as for a name that exists nowhere. Saying "this
                # tool exists but was not delivered to you" would tell the model the registry
                # holds tools its surface withheld; the precise reason goes to the transcript.
                payload = {"error": f"unknown tool {name!r}"}
                call_failure = default_failure(FailureCode.INVALID_TOOL_ARGUMENTS)
                projection = payload_projection(payload)
                undelivered_detail = (
                    f"{name}: not on the delivered tool surface"
                    + (" (backed by the runtime registry)" if name in self.registry else ""))
            elif name == "run_code" and self.sandbox is not None:
                try:
                    payload = self.sandbox.run(args.get("code", ""))
                except PhysicalTimeBudgetExhausted as exc:
                    return self._physical_time_terminal_call(name, args, exc)
                code_observation_ids = payload.get("obs_ids") or ()
                projection = project_tool_result(
                    payload,
                    toolbox=self.toolbox,
                    code_observation_ids=code_observation_ids,
                    image_transport=self.profile.image_transport,
                    run_code_result_max_images=RUN_CODE_RESULT_MAX_IMAGES,
                    serialize_result=False,
                    tool_name=name,
                )
            elif name in PROGRAM_TOOL_NAMES and self.sandbox is not None:
                try:
                    payload = getattr(self.sandbox, name)(**args)
                    code_observation_ids = payload.get("obs_ids") or ()
                    projection = project_tool_result(
                        payload,
                        toolbox=self.toolbox,
                        code_observation_ids=code_observation_ids,
                        image_transport=self.profile.image_transport,
                        run_code_result_max_images=RUN_CODE_RESULT_MAX_IMAGES,
                        serialize_result=False,
                        tool_name=name,
                    )
                except PhysicalTimeBudgetExhausted as exc:
                    # `run_program` executes primitives through the same sandbox as `run_code`, so
                    # it can reach the task physical-time threshold the same way. Without this the
                    # terminal escaped dispatch() entirely -- no host catches it -- and the episode
                    # ended up attributing its own terminal to whatever tool ran next.
                    return self._physical_time_terminal_call(name, args, exc)
                except ProgramWorkspaceError as exc:
                    payload = {"error": str(exc), "error_code": exc.code,
                               "filesystem_policy": "structurally_denied"}
                    call_failure = default_failure(FailureCode(exc.code))
                    projection = payload_projection(payload)
                except (TypeError, ValueError) as exc:
                    payload = {"error": str(exc), "error_code": "invalid_virtual_file_argument",
                               "filesystem_policy": "structurally_denied"}
                    call_failure = default_failure(FailureCode.INVALID_VIRTUAL_FILE_ARGUMENT)
                    projection = payload_projection(payload)
            elif name not in self.registry:
                projection = payload_projection({"error": f"unknown tool {name!r}"})
            else:
                try:
                    result_obj = self.registry[name](**args)
                    projection = project_tool_result(
                        result_obj,
                        toolbox=self.toolbox,
                        image_transport=self.profile.image_transport,
                        tool_name=name,
                    )
                except (TypeError, ArgumentContractError) as exc:
                    # A declared-contract violation is an argument error, not a tool error: it
                    # must land in the same taxonomy as a Python signature mismatch.
                    projection = payload_projection({"error": f"bad arguments: {exc}"})
                    call_failure = default_failure(FailureCode.INVALID_TOOL_ARGUMENTS)
                except (KeyError, ValueError) as exc:
                    projection = payload_projection({"error": str(exc)})
                    call_failure = default_failure(FailureCode.INVALID_TOOL_ARGUMENTS)
                except PhysicalTimeBudgetExhausted as exc:
                    return self._physical_time_terminal_call(name, args, exc)
                except Exception as exc:
                    failure = failure_from_legacy_status(
                        "episode_fatal",
                        tested_origin=self.profile.tested_origin)
                    event_step = self.steps
                    call = RuntimeCall(
                        name, args, payload_projection({"error": "episode aborted"}),
                        event_step, True, kind="fatal", failure=failure,
                        error_detail=f"{name}: {exc}")
                    self.finalize("episode_fatal", failure=failure, context={"call": call})
                    return call

            kind = ("recoverable_abort"
                    if is_recoverable_action_abort(projection.full_payload) else "tool")
            return RuntimeCall(
                name, args, projection, self.steps, True, kind=kind,
                failure=call_failure, error_detail=undelivered_detail,
            )
