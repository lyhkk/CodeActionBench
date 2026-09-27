"""Agent-visible execution accounting for every sim-mutating D0 tool. Pure: no simulator import.

Two facts decide what a motion call may CLAIM to the model, and before this module each tool
answered them differently or not at all:

* did the physics engine advance at all — a planner refusal that never reached ``env.move`` must
  not advance the episode state clock, must not stale existing observations, and must not be
  reported as an execution. Reproduced across four tools: a `move_delta` whose plan was refused
  (outer sim_step 8381 -> 8381, TCP unchanged) still returned action tick 34 -> 35;
* if the call did not complete, had it already moved the robot — `FAILED` alone cannot distinguish
  "nothing happened" from "the arm is now somewhere else". A paired reach recorded FAILED while
  advancing 104 physics steps.

Neither is derivable by the caller. Everything that IS derivable stays out: the residual to the
commanded target, the net displacement, and which internal leg stopped the motion all follow from
`commanded` + `observed_before/after`, and this repository's standing rule is that the harness does
not pre-chew them ("No position/angle error or intermediate waypoint is returned; compare
commanded with observed_before/observed_after if needed").

The internal stage-by-stage record — fallback legs, bounded corrections, per-leg deviation
measurements — is auditor-grade, not agent-grade: it changes nothing the model can do next, and it
lives in the host-only `codeaction.motion.motion_trace` file instead of in the model's context. The status
vocabulary is unchanged (`result_contracts._action`); what changes is that ONE rule table produces
it for every tool.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExecutionEvidence:
    """What the harness actually observed about one call, before any wording is chosen."""

    attempted: bool                    # at least one command was issued to the simulator
    physics_steps: Optional[int]       # steps the physics engine advanced during this call
    state_changed: Optional[bool]      # measured robot-state change; the clockless fallback
    interrupted: bool                  # a guard stopped the command before its endpoint
    completed: bool                    # the tool's declared procedure ran to its end
    post_state_observed: bool = True

    @property
    def determinable(self) -> bool:
        """Whether ANY evidence exists about physical advance.

        Without a clock and without a state read, the honest answer to "did it move" is neither
        yes nor no. Collapsing that into "nothing happened" would be the same unearned claim this
        module exists to prevent, one direction over.
        """
        return self.physics_steps is not None or self.state_changed is not None

    @property
    def advanced(self) -> bool:
        """Whether the robot physically did anything.

        The physics clock wins wherever it exists. Without it — unit fixtures and probes, which
        attach no `StepObserver` — the measured state comparison is the only evidence there is.
        """
        if self.physics_steps is not None:
            return int(self.physics_steps) > 0
        return bool(self.state_changed)


def execution_status(evidence: ExecutionEvidence) -> str:
    """The single rule table every sim-mutating tool now shares.

    A planning refusal is `NOT_STARTED`, not `FAILED`: nothing executed, so there is no execution
    to have failed. `planning.status` carries the refusal, which is what keeps a planner refusal,
    a guard abort and a real execution failure from collapsing into one boolean.
    """
    if not evidence.attempted:
        return "NOT_STARTED"
    if not evidence.determinable or not evidence.post_state_observed:
        return "UNKNOWN"
    if not evidence.advanced:
        return "NOT_STARTED"
    if evidence.interrupted:
        return "INTERRUPTED"
    return "COMPLETED" if evidence.completed else "FAILED"


def execution_block(evidence: ExecutionEvidence) -> dict:
    """The top-level ``execution`` block.

    ``partial`` is true exactly when the call did not complete and the robot moved anyway, so a
    caller never has to infer prior side effects from two giant state snapshots. It is null — not
    false — when no clock and no state read were available, because "no side effects" and "we
    could not tell" are different facts and only one of them is safe to plan against.
    """
    return {
        "status": execution_status(evidence),
        "post_state_observed": bool(evidence.post_state_observed),
        "state_changed": evidence.state_changed,
        "physics_steps": (None if evidence.physics_steps is None
                          else int(evidence.physics_steps)),
        "partial": (None if evidence.attempted and not evidence.determinable
                    else bool(evidence.advanced and not evidence.completed)),
    }


__all__ = ["ExecutionEvidence", "execution_block", "execution_status"]
