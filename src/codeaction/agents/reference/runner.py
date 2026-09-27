"""Compatibility/development entry point over the single Python reference-agent loop."""

from codeaction.interface.instructions import reference_instruction_surface
from codeaction.agents.reference.reference_agent import (
    DEFAULT_CONTEXT_POLICY,
    FRAME_RETENTION,
    MAX_TOKENS_PER_TURN,
    ModelAdapter,
    REQUESTED_OUTPUT_TOKENS,
    SCAFFOLD_NAME,
    SCAFFOLD_VERSION,
    ScriptedModel,
    run_episode,
    scaffold_card,
)
from codeaction.providers.model_adapter import build_provider


def system_prompt(budget=None, hybrid=False, run_code_max_calls=None) -> str:
    """Compatibility entry point over the single authored instruction renderer."""
    return reference_instruction_surface(
        task_text="",
        max_tool_calls=int(budget or 1),
        run_code_max_internal_calls=int(run_code_max_calls or 1),
    )["system_prompt"]


from codeaction.runtime.sandbox import make_sandbox  # noqa: F401  (canonical home)


__all__ = [
    "DEFAULT_CONTEXT_POLICY",
    "FRAME_RETENTION",
    "MAX_TOKENS_PER_TURN",
    "ModelAdapter",
    "REQUESTED_OUTPUT_TOKENS",
    "SCAFFOLD_NAME",
    "SCAFFOLD_VERSION",
    "ScriptedModel",
    "build_provider",
    "make_sandbox",
    "run_episode",
    "scaffold_card",
    "system_prompt",
]
