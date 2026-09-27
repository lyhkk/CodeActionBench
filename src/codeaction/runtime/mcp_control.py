"""Transport-only controls for the containerized reference scaffold.

These names are never returned by ``tools/list`` and therefore are not model capabilities.  They
let the benchmark-owned reference runtime wait for scene readiness and finalize a classified
attempt when the model loop ends without calling ``done``.
"""

REFERENCE_READY_TOOL = "__codeaction_reference_ready__"
REFERENCE_FINALIZE_TOOL = "__codeaction_reference_finalize__"
# Every status the reference model loop may end on. The episode server validates the finalize
# control against this set, so a status added in `reference_agent` but not here would be rejected
# server-side and lose every episode that ends that way -- with nothing local to catch it, since
# the check runs inside the sim container. One definition, imported by both sides;
# `tests/test_agent_container_contract.py` asserts the agent's finalize literals are covered.
REFERENCE_FINALIZE_STATUSES = frozenset({
    "budget_exhausted",
    "wall_budget",
    "no_tool_calls",
    "endpoint_failure",
    "physical_time_budget_exhausted",
    "context_length_exceeded",
    "output_length_exceeded",
    "unsupported_stop_reason",
    "model_refusal",
    "episode_fatal",
})
REFERENCE_MALFORMED_TOOL = "__codeaction_reference_malformed__"
# Mirror a same-turn call cancelled by the reference loop after a structured action abort. The
# host records the charge and explicit non-execution without invoking the named public tool.
REFERENCE_CANCELLED_TOOL = "__codeaction_reference_cancelled__"
# Returns the seconds a provider outage stole from the episode. Both sides of a reference-scaffold attempt
# enforce the wall budget on their own clock -- the agent before each model turn, the sim host on
# each tool dispatch -- and neither can tell "the model is thinking" from "we are waiting out
# someone's rate limiter" by looking at the gap between calls. Only the agent knows, so it says so,
# and this is the channel. Without it a vendor's 429 storm is spent from the model's time budget
# and then scored against the model as `wall_budget` (origin=model, scoreable=True).
REFERENCE_WALL_CREDIT_TOOL = "__codeaction_reference_wall_credit__"
MALFORMED_ARGUMENTS_MARKER = "__codeaction_malformed_tool_arguments__"
