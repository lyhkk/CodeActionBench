"""Capability and evidence contract for vendor agent CLIs.

Vendor stacks use the same benchmark tools and simulator. Comparable execution requires the
same capabilities, isolation restrictions and evidence for each claim. This module defines
that contract for CLI declarations in ``clis.py``, runtime attestations and the generated
``docs/vendor-cli-onboarding.md`` checklist.

Evidence sources differ by CLI. Claude Code's ``system/init`` event reports its tool roster;
Codex starts with ``thread.started`` without that roster. Model self-report cannot establish
which tools were exposed. Server records and container boundaries support common claims;
CLI-specific stream events provide additional evidence where available. Launch configuration
establishes what was requested and must be paired with execution evidence where required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------------------------
# Evidence kinds ordered by independence from vendor reporting: container boundaries and
# episode-server records establish properties outside the CLI's own event stream.
# ---------------------------------------------------------------------------------------------
EVIDENCE_KINDS: dict[str, str] = {
    "container": (
        "The container and its namespaces. What is not mounted cannot be read, and no vendor "
        "flag can undo that. Valid for every CLI without asking the CLI anything."),
    "server": (
        "The episode server's own record -- result.json (tool_surface, stats) and the transcript "
        "it wrote. Our side of the wall, so it establishes a claim identically for every CLI."),
    "launch": (
        "The argv and config file WE hand the CLI. It establishes what was requested, not what "
        "happened; pair it with stream or server evidence wherever the CLI can report back."),
    "stream": (
        "The CLI's own machine-readable event stream. Strong when present, but each CLI emits a "
        "different set of events, so a claim resting on stream alone is not cross-CLI evidence."),
}

# A CLI can declare a required capability unmet; that declaration disqualifies the stack.
DISPOSITIONS: dict[str, str] = {
    "satisfied": "The required capability is present and evidenced.",
    "denied": "The banned capability is structurally unavailable, and something proves it.",
    "residual": (
        "The banned capability could not be removed from the CLI, and is instead neutralized "
        "(there is nothing for it to reach) and DECLARED. Never silently tolerated."),
    "unmet": "The capability is required and this stack cannot provide it. Disqualifying.",
}


@dataclass(frozen=True)
class Capability:
    """One line of the contract."""

    id: str
    summary: str
    why: str
    # Evidence kinds that may establish this item, strongest acceptable first.  A CLI declaring a
    # kind outside this tuple is a declaration error, not a weaker proof.
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").isalnum():
            raise ValueError(f"capability id must be a bare identifier: {self.id!r}")
        unknown = [kind for kind in self.evidence if kind not in EVIDENCE_KINDS]
        if unknown:
            raise ValueError(f"{self.id}: unknown evidence kind(s) {unknown}")
        if not self.evidence:
            raise ValueError(f"{self.id}: a contract item needs at least one evidence kind")


# ---------------------------------------------------------------------------------------------
# Required capabilities for scoreable, auditable episodes.
# ---------------------------------------------------------------------------------------------
REQUIRED: tuple[Capability, ...] = (
    Capability(
        id="machine_readable_stream",
        summary="Emits one JSONL event stream on stdout, ending in a terminal event.",
        why=("Human-readable vendor prose cannot be audited, diffed, or classified. The terminal "
             "event is what separates 'the model stopped' from 'the process died'."),
        evidence=("stream",),
    ),
    Capability(
        id="per_call_tool_record",
        summary="Every tool call and its result appear as distinct, parseable records.",
        why=("Tool-call count and the call/result pairing are the episode's spine: an unmatched "
             "call is a lost turn, and a run that cannot show the pairing cannot show it did "
             "not lose one."),
        evidence=("stream", "server"),
    ),
    Capability(
        id="token_usage_totals",
        summary="Reports input and output token totals for the episode.",
        why=("Cost is compared in tokens, never currency -- prices are dated external facts and "
             "one of these CLIs reports no currency at all."),
        evidence=("stream",),
    ),
    Capability(
        id="reasoning_spend_counter",
        summary="Reports reasoning tokens as a number distinct from output tokens.",
        why=("A thinking rung that is configured but never funded looks identical to one that "
             "worked, unless the spend is counted separately."),
        evidence=("stream",),
    ),
    Capability(
        id="reasoning_text_capture",
        summary="Can be configured to RETURN reasoning summary text, not merely spend it.",
        why=("Both stacks default to withholding it, by different switches -- Anthropic's "
             "thinking.display=omitted, OpenAI's default_reasoning_summary=none. Spending "
             "reasoning tokens is not evidence that any reasoning came back, so the switch is "
             "part of the contract rather than a per-run detail someone remembers."),
        evidence=("launch", "stream"),
    ),
    Capability(
        id="model_selection",
        summary="The model is selected by us and is nameable in the result identity.",
        why=("The unit under test is (driver, model, rung). A CLI that silently substitutes or "
             "upgrades the model publishes a number attributed to a model that never ran."),
        evidence=("launch", "stream"),
    ),
    Capability(
        id="effort_selection",
        summary="The reasoning rung is selected by us from the model's declared ladder.",
        why="Same reason as model_selection: the rung is half of what a comparison holds fixed.",
        evidence=("launch", "stream"),
    ),
    Capability(
        id="mcp_stdio_client",
        summary="Attaches an MCP server over stdio and calls its tools.",
        why=("The tool surface reaches the agent ONLY through MCP. A stack that cannot speak it "
             "cannot be given the benchmark's tools at all."),
        evidence=("server", "stream"),
    ),
    Capability(
        id="image_tool_results",
        summary="An MCP image result reaches the model as an image, not as a dropped block.",
        why=("The benchmark is depth-free and image-driven. A stack that silently discards "
             "tool-returned images scores the task blind and the score means nothing. This is "
             "the exact defect that ruled out one candidate CLI before it was ever integrated."),
        evidence=("server", "stream"),
    ),
    Capability(
        id="tool_allowlist",
        summary="Native tools can be restricted to an explicit allowlist.",
        why=("Everything in BANNED below is enforced first by this switch and only then by the "
             "container. A stack with no allowlist puts the entire wall on the container."),
        evidence=("launch", "stream"),
    ),
    Capability(
        id="nonpersistent_session",
        summary="Keeps no session, history, or memory outside the episode.",
        why=("Episodes are independent samples. Carry-over between them turns an attempt into a "
             "second look at the same scene and inflates every rate computed from them."),
        evidence=("launch", "container"),
    ),
    Capability(
        id="pinned_version",
        summary="Runs at a version we pin, and does not update itself.",
        why=("The CLI is half the tested unit. A stack that self-updates mid-batch makes the "
             "batch a mixture of two agents reported as one."),
        evidence=("container", "launch"),
    ),
    Capability(
        id="deterministic_exit",
        summary="Exits with a code that, with the terminal event, classifies the ending.",
        why=("A scoreable model failure and an infrastructure failure must not be recorded as "
             "the same outcome; one is a result and the other is a retry."),
        evidence=("stream", "server"),
    ),
)


# ---------------------------------------------------------------------------------------------
# Banned capabilities either expose private task information or change the evaluated interface.
# For example, a vendor code sandbox differs from the benchmark's run_code tool.
# ---------------------------------------------------------------------------------------------
BANNED: tuple[Capability, ...] = (
    Capability(
        id="filesystem_read",
        summary="Reading any host path: the repository, task solutions, prior runs, secrets.",
        why=("The ground-truth wall. Task cards, verifiers and oracle solutions are all on disk; "
             "a single successful read turns the benchmark into an open-book exam."),
        evidence=("container", "launch"),
    ),
    Capability(
        id="shell_execution",
        summary="Running host commands.",
        why="A shell is filesystem_read plus network_egress with extra steps.",
        evidence=("container", "launch"),
    ),
    Capability(
        id="network_egress",
        summary="Any network destination other than the vendor's own model endpoint.",
        why=("Egress is an exfiltration path for the scene and an import path for a solution. "
             "The model endpoint is the one exception, because without it there is no agent."),
        evidence=("container", "launch"),
    ),
    Capability(
        id="web_retrieval",
        summary="Web search or fetch tools.",
        why=("A published task name is searchable. This is network_egress dressed as a feature, "
             "and it is on by default in both candidate stacks."),
        evidence=("launch", "stream"),
    ),
    Capability(
        id="subagent_delegation",
        summary="Spawning subagents, teams, or delegated turns.",
        why=("The tested unit is one agent at one rung. A stack that fans out is a different "
             "unit, spends a different amount, and cannot be compared to one that does not. "
             "One candidate model's top rung enables delegation implicitly -- hence the ban is "
             "on the capability, not on a flag."),
        evidence=("launch", "stream"),
    ),
    Capability(
        id="persistent_memory",
        summary="Memory, notes, or goals that outlive the episode.",
        why="Same reason as nonpersistent_session, stated as the thing to switch off.",
        evidence=("launch", "container"),
    ),
    Capability(
        id="instruction_injection",
        summary="Plugins, skills, hooks, project docs, rules files, slash commands.",
        why=("The instruction surface is hashed into the result identity. Any channel that can "
             "add instructions we did not write makes that hash a claim about a prompt the "
             "model did not receive."),
        evidence=("launch", "container"),
    ),
    Capability(
        id="background_execution",
        summary="Background tasks, cron, or work that outlives the turn.",
        why=("An episode is bounded by a wall clock and a tool budget. Work that escapes the "
             "turn escapes both."),
        evidence=("launch",),
    ),
    Capability(
        id="vendor_code_sandbox",
        summary="The CLI's own code-execution surface standing in for the benchmark's run_code.",
        why=("The benchmark DELIVERS run_code as an MCP tool, identically to every agent. A "
             "vendor-supplied code mode is a different interface -- different sandbox, different "
             "budget accounting, different failure taxonomy -- so a seat using it is not running "
             "the same benchmark. This is a fairness ban, not a security one: it is exactly why "
             "the surface is pinned to direct tool exposure even for a model whose native mode "
             "is code mode."),
        evidence=("launch", "server", "container"),
    ),
)


REQUIRED_IDS: tuple[str, ...] = tuple(item.id for item in REQUIRED)
BANNED_IDS: tuple[str, ...] = tuple(item.id for item in BANNED)
ALL_CAPABILITIES: tuple[Capability, ...] = REQUIRED + BANNED


def capability(capability_id: str) -> Capability:
    for item in ALL_CAPABILITIES:
        if item.id == capability_id:
            return item
    raise KeyError(f"unknown capability {capability_id!r}")


def _duplicates(ids: Iterable[str]) -> list[str]:
    seen, dupes = set(), []
    for value in ids:
        if value in seen:
            dupes.append(value)
        seen.add(value)
    return dupes


_dupe = _duplicates(REQUIRED_IDS + BANNED_IDS)
if _dupe:
    raise ValueError(f"duplicate capability id(s): {_dupe}")


__all__ = (
    "ALL_CAPABILITIES", "BANNED", "BANNED_IDS", "Capability", "DISPOSITIONS",
    "EVIDENCE_KINDS", "REQUIRED", "REQUIRED_IDS", "capability",
)
