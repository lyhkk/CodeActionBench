"""Agent configurations a batch can schedule.

An agent is (driver, model, reasoning rung, delivered interface) -- never a bare model id. The
released roster mixes drivers deliberately: the same model appears on our reference scaffold and
on Claude Code, which is the one measurement that separates the scaffold's contribution from the
model's.

Concurrency is expressed through the credential axis the scheduler already understands. A
provider-backed agent shares its provider's alias, so two agents on the same key serialize
against that provider's rate limits.

A vendor agent's quota is its SUBSCRIPTION ACCOUNT, and each account is its own credential with
its own declared concurrency: two accounts run side by side without contending, while two agents
on the same account share that account's lane count.

The roster names the account by ALIAS only. Which account that alias denotes -- its plan, its lane
count, the path to its token, how its rolling window is governed -- is local configuration read
from `agent_config` (`~/.config/codeaction/agents.json`), never a committed fact: the published
roster must be the same eight agents in every checkout, while whose subscription pays for the
vendor seat is nobody else's business. An alias with no configuration leaves the agent in the
roster and unrunnable, which is what lets a checkout with no subscription still run the other
seven and still describe the full benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from codeaction.agents.vendor.clis import VENDOR_AGENT_MODES, vendor_cli
from codeaction.benchmark.agent_config import (AgentConfigError, account_for,
                                               unconfigured_message)
from codeaction.providers.model_registry import resolve_model

REFERENCE_DRIVER = "reference"
# Every seated vendor CLI, from the seat table rather than a second list here. A roster that kept
# its own copy would have gone stale the first time a stack was added, and the failure would have
# been an agent that schedules but cannot launch.
VENDOR_DRIVERS: tuple[str, ...] = VENDOR_AGENT_MODES
# Kept as the name of the FIRST vendor seat, because the released roster and its recorded results
# refer to it. It is not a synonym for "the vendor driver" any more.
VENDOR_DRIVER = "claude"


# Aliases the roster declares for its vendor seats. Each is a NAME, not an account: what it
# denotes is whatever the local `agents.json` binds it to.
CLAUDE_SUBSCRIPTION = "claude-subscription"
CODEX_SUBSCRIPTION = "codex-subscription"


@dataclass(frozen=True)
class AgentConfig:
    """One schedulable agent. ``label`` is the identity a batch reports and files results under."""

    label: str
    driver: str
    model: str
    reasoning: str
    interface_profile: str
    # Set only for the vendor driver: which subscription account pays for this agent.
    account: str | None = None
    # How many episodes of THIS agent may run at once. It is a property of what pays for the
    # agent, not of the task: a provider key that tolerates one in-flight request gets 1, and a
    # subscription with several lanes on one account gets as many as the account declares. The
    # batch freezes it as `model_lanes`, and the credential cap still binds above it.
    lanes: int = 1
    # Reference driver only. `scripted` replaces the provider call with a fixed tool sequence, so
    # the agent costs nothing and needs no key -- the only way to exercise the whole batch
    # machinery end to end without spending a subscription or a provider quota.
    reference_model_mode: str = "provider"

    def __post_init__(self) -> None:
        if self.driver not in (REFERENCE_DRIVER, *VENDOR_DRIVERS):
            raise ValueError(f"unknown driver {self.driver!r}")
        if self.driver in VENDOR_DRIVERS:
            if self.interface_profile != "vendor-mcp-direct":
                raise ValueError("the vendor agent runs on vendor-mcp-direct")
            # Only that the alias IS one. Whether it resolves is a question about the machine, and
            # asking it at import would make the roster -- a published fact -- depend on a file.
            if not isinstance(self.account, str) or not self.account:
                raise ValueError(
                    f"vendor agent needs a subscription alias, got {self.account!r}")
        elif self.account is not None:
            raise ValueError("only a vendor agent has a subscription account")
        if not self.label or "/" in self.label:
            raise ValueError(f"agent label must be non-empty and path-safe: {self.label!r}")
        if self.reference_model_mode not in ("provider", "scripted", "local"):
            raise ValueError(
                f"reference_model_mode must be provider or scripted, "
                f"got {self.reference_model_mode!r}")
        if self.driver in VENDOR_DRIVERS and self.reference_model_mode != "provider":
            raise ValueError("only the reference driver has a model mode")
        if not isinstance(self.lanes, int) or isinstance(self.lanes, bool) or self.lanes < 1:
            raise ValueError(f"agent lanes must be a positive integer, got {self.lanes!r}")

    @property
    def credential(self) -> str | None:
        """Which shared quota this agent consumes.

        For a vendor agent this is the ALIAS, available with or without configuration: grouping
        two agents onto one quota does not require knowing what the quota is.
        """
        if self.driver in VENDOR_DRIVERS:
            return self.account
        if self.reference_model_mode == "local":
            from codeaction.extensions import declarations
            return declarations("agent")[self.label].get("credential")
        if self.is_scripted:
            return None          # a scripted agent spends nothing shared
        return resolve_model(self.model).credential

    @property
    def subscription(self):
        """The configured account behind the alias, or None when nothing declares it."""
        if not self.account:
            return None
        return account_for(self.account)

    def require_subscription(self):
        """The account, or a refusal that names the alias and the file that would declare it."""
        account = self.subscription
        if account is None:
            raise AgentConfigError(unconfigured_message(self.account))
        return account

    @property
    def uses_subscription(self) -> bool:
        return self.driver in VENDOR_DRIVERS

    @property
    def is_scripted(self) -> bool:
        return self.reference_model_mode == "scripted"

    @property
    def has_provider_profile(self) -> bool:
        """Whether a frozen request/rate-limit profile exists for this agent at all.

        False for a subscription agent (no key, no per-key window) and for a scripted one (no
        provider call). Both are scheduled like any other agent; neither has a profile to freeze,
        and fabricating one would be a claim about a request that never happens."""
        return not self.uses_subscription and not self.is_scripted and self.reference_model_mode != "local"


def reference_agent(model: str, *, reasoning: str = "high",
                    interface_profile: str = "reference-mcp",
                    lanes: int = 1, label: str | None = None) -> AgentConfig:
    """A reference agent; an optional label distinguishes configurations of the same model."""
    return AgentConfig(label=label or model, driver=REFERENCE_DRIVER, model=model,
                       reasoning=reasoning, interface_profile=interface_profile, lanes=lanes)


def scripted_agent(label: str = "scripted", *, lanes: int = 4) -> AgentConfig:
    """A zero-cost agent that drives the real tool surface from a fixed sequence.

    Not part of the released roster: it scores nothing and belongs to no leaderboard. It exists so
    the batch machinery -- queue, leases, attentions, requeue, directory layout -- can be proven
    on the real sim before a paid batch is launched against it."""
    return AgentConfig(label=label, driver=REFERENCE_DRIVER, model="scripted-model",
                       reasoning="none", interface_profile="reference-mcp",
                       lanes=lanes, reference_model_mode="scripted")


def vendor_agent(model: str, *, reasoning: str = "high", label: str | None = None,
                 account: str = CLAUDE_SUBSCRIPTION, lanes: int = 1) -> AgentConfig:
    return AgentConfig(label=label or f"claude-code-{model}", driver=VENDOR_DRIVER,
                       model=model, reasoning=reasoning,
                       interface_profile="vendor-mcp-direct", account=account, lanes=lanes)


def codex_agent(model: str, *, reasoning: str = "high", label: str | None = None,
                account: str = CODEX_SUBSCRIPTION, lanes: int = 1) -> AgentConfig:
    """A seat on the Codex CLI. Same interface, same server, a different agent stack."""
    return AgentConfig(label=label or f"codex-{model}", driver="codex",
                       model=model, reasoning=reasoning,
                       interface_profile="vendor-mcp-direct", account=account, lanes=lanes)


# The published roster: seven models on the reference scaffold plus Claude Code, with
# claude-opus-5 on both so the scaffold's contribution is directly measurable. Every agent runs
# the `high` rung except qwen3.8-max, whose registry entry tops out below it.
RELEASE_AGENTS: tuple[AgentConfig, ...] = (
    # LANES, i.e. how many episodes of one agent run at once. Everything on a third-party
    # provider key stays at 1. The two Anthropic-API agents take 2, and Claude Code takes the
    # 4 its subscription account declares -- that is the only agent that can fill a four-GPU box
    # on its own.
    reference_agent("claude-opus-5", lanes=2),
    reference_agent("claude-sonnet-5", lanes=2),
    reference_agent("gpt-5.6"),
    reference_agent("gemini-3.6-flash"),
    reference_agent("grok-4.6"),
    reference_agent("kimi-k3"),
    reference_agent("qwen3.8-max", reasoning="medium"),
    # Label matches the published identity, not the model id.
    vendor_agent("claude-opus-5", label="claude-code-opus-5", lanes=4,
                 account=CLAUDE_SUBSCRIPTION),
)


# Names that stand for a group rather than one agent. Selecting "the released roster" or "every
# reference seat" is the common case and had no spelling: a seven-agent batch meant seven --model
# flags, and the roster -- which is right here -- could not be referred to at all.
AGENT_SELECTORS = ("all-agents", "reference-agents", "vendor-agents")


def expand_agent_selection(names, roster=None) -> tuple[str, ...]:
    """Resolve selectors to agent labels, order-preserving and duplicate-free.

    A name that is not a selector passes through untouched, so a selector and a single label can
    be mixed, and an unknown label still fails later where the roster is validated.
    """
    roster = tuple(RELEASE_AGENTS if roster is None else roster)
    groups = {
        "all-agents": [a.label for a in roster],
        "reference-agents": [a.label for a in roster if a.driver == REFERENCE_DRIVER],
        "vendor-agents": [a.label for a in roster if a.driver in VENDOR_DRIVERS],
    }
    out: list[str] = []
    for name in names or ():
        for label in groups.get(str(name), [str(name)]):
            if label not in out:
                out.append(label)
    return tuple(out)



# Agents that can be selected by label but score nothing: they exist to prove the machinery.
TEST_AGENTS: tuple[AgentConfig, ...] = (scripted_agent(),)


# Real, scoreable agents that are NOT part of the published roster yet. They are selectable by
# label and by no selector, so `all-agents` keeps meaning the eight agents every published result
# was produced by. Promoting one is a deliberate edit to RELEASE_AGENTS, made when its seat has a
# run behind it -- not a side effect of adding the seat.
CANDIDATE_AGENTS: tuple[AgentConfig, ...] = (
    # Four lanes, like the released Claude Code seat: one subscription account can fill the box.
    codex_agent("gpt-6-astra", label="codex-astra", account=CODEX_SUBSCRIPTION, lanes=4),
    # The same stack one rung up. An agent IS (driver, model, rung, interface), so a rung is a
    # SEAT rather than a flag: the label carries it into the run directory, the batch cell id and
    # the identity hash, which is what keeps a high episode and an xhigh episode from ever
    # pooling. gpt-6-astra is the only model in the registry with a rung above `high`.
    # Both seats spend the same subscription, so a batch that selects both must still cap the
    # credential at the four lanes the account declares, not eight.
    codex_agent("gpt-6-astra", label="codex-astra-xhigh", reasoning="xhigh",
                account=CODEX_SUBSCRIPTION, lanes=4),
)


def agents_by_label(agents: Iterable[AgentConfig] = RELEASE_AGENTS) -> dict[str, AgentConfig]:
    table: dict[str, AgentConfig] = {}
    for agent in agents:
        if agent.label in table:
            raise ValueError(f"duplicate agent label {agent.label!r}")
        table[agent.label] = agent
    return table


def resolve_agents(labels: Iterable[str] | None,
                   roster: Iterable[AgentConfig] = RELEASE_AGENTS) -> tuple[AgentConfig, ...]:
    # The released roster is the default selection; the test agents are selectable BY NAME only,
    # never part of "all agents", so a bare run can never quietly include a scoreless one.
    table = agents_by_label(roster)
    selected = tuple(labels) if labels else tuple(table)
    for agent in TEST_AGENTS + CANDIDATE_AGENTS:
        table.setdefault(agent.label, agent)
    unknown = [label for label in selected if label not in table]
    if unknown:
        raise ValueError(f"unknown agent label(s): {unknown}; known: {sorted(table)}")
    if len(set(selected)) != len(selected):
        raise ValueError("agent selection repeats a label")
    return tuple(table[label] for label in selected)


def agent_credentials(agents: Iterable[AgentConfig]) -> dict[str, str | None]:
    """Scheduler input: agent label -> the quota it consumes."""
    return {agent.label: agent.credential for agent in agents}


def agent_credential_limits(agents: Iterable[AgentConfig],
                            overrides: Mapping[str, int] | None = None,
                            default: int = 1) -> dict[str, int]:
    """Per-credential concurrency.

    A subscription account's lane count comes from its own declaration, not from ``default``;
    an override may lower it (to be gentle on a shared window) but never raise it above what the
    plan sustains, because the extra lane would not queue -- it would degrade the others.
    """
    agents = tuple(agents)
    limits = {agent.credential: default for agent in agents if agent.credential}
    for agent in agents:
        # An unconfigured alias keeps the conservative default rather than the lane count the
        # roster hopes for: a batch that cannot see the account must not assume its width.
        account = agent.subscription
        if account is not None:
            limits[account.alias] = account.max_concurrency
    for alias, limit in (overrides or {}).items():
        if alias not in limits:
            continue
        limit = int(limit)
        # The ceiling is the configured account's own declaration; with no configuration there
        # is no declaration to exceed, and the override stands as given.
        account = account_for(alias)
        if account is not None and limit > account.max_concurrency:
            raise ValueError(
                f"{alias} sustains {account.max_concurrency} parallel session(s) on its "
                f"{account.plan} plan; {limit} would degrade every lane rather than queue")
        limits[alias] = limit
    return limits


def build_agent_command(
    agent: AgentConfig,
    *,
    python_bin: str,
    controller_module: str,
    task: str,
    gpu: int | str,
    run_dir,
    task_pack,
    profile: str = "dev",
    provider_env_file=None,
    provider_rate_limit_file=None,
    token_file=None,
    attempts: int = 1,
    start_seed: int = 0,
    attempt_index: int | None = None,
    run_id: str | None = None,
    source_root=None,
    image_digests=None,
) -> list[str]:
    """The controller invocation for one (agent, task) cell.

    The driver decides which credential material is passed: a reference agent needs a provider
    key and its rate-limit declaration, the vendor agent needs the subscription token file and
    accepts neither.
    """
    command = [
        python_bin, "-m", controller_module, "run",
        "--profile", profile,
        "--agent-mode", agent.driver,
        "--interface-profile", agent.interface_profile,
        "--model", agent.model,
        "--reasoning-profile", agent.reasoning,
        "--task", task,
        "--attempts", str(attempts),
        "--gpu", str(gpu),
        "--task-pack", str(task_pack),
        "--run-dir", str(run_dir),
        "--agent-label", agent.label,
    ]
    # A supervised cell names the exact declared attempt it is replaying; a one-off run walks
    # forward from a start seed. They are alternatives, never both.
    if attempt_index is not None:
        command.extend(["--attempt-index", str(attempt_index)])
    else:
        command.extend(["--start-seed", str(start_seed)])
    if run_id is not None:
        command.extend(["--run-id", str(run_id)])
    if source_root is not None:
        command.extend(["--source-root", str(source_root)])
    if image_digests:
        # One digest field per image; which CLI flag carries the agent image depends on the
        # driver, because the two drivers run different agent containers.
        agent_image_flag = ("--reference-agent-image" if agent.driver == REFERENCE_DRIVER
                            else vendor_cli(agent.driver).image_cli_flag)
        command.extend(["--sim-image", str(image_digests["sim_image_digest"]),
                        agent_image_flag, str(image_digests["agent_image_digest"]),
                        "--gateway-image", str(image_digests["gateway_image_digest"])])
    if agent.driver == REFERENCE_DRIVER:
        command.extend(["--reference-model-mode", agent.reference_model_mode])
        if provider_env_file is not None and (agent.has_provider_profile or
                (agent.reference_model_mode == "local" and agent.credential is not None)):
            command.extend(["--provider-env-file", str(provider_env_file)])
        if agent.has_provider_profile and provider_rate_limit_file is not None:
            command.extend(["--provider-rate-limit-file", str(provider_rate_limit_file)])
    else:
        # Each account has its own token file; an explicit override still wins so a batch can
        # point at a relocated credential without editing the roster.
        resolved = (Path(token_file) if token_file is not None
                    else agent.require_subscription().token_path)
        command.extend(["--token-file", str(resolved)])
    return command


__all__ = [
    "AGENT_SELECTORS", "AgentConfig", "CANDIDATE_AGENTS", "CLAUDE_SUBSCRIPTION",
    "CODEX_SUBSCRIPTION", "RELEASE_AGENTS",
    "TEST_AGENTS", "codex_agent", "expand_agent_selection",
    "scripted_agent", "REFERENCE_DRIVER",
    "VENDOR_DRIVER", "VENDOR_DRIVERS", "agent_credential_limits", "agent_credentials",
    "agents_by_label", "build_agent_command", "reference_agent", "resolve_agents", "vendor_agent",
]


def default_reasoning_for_model(model: str) -> str:
    """Published settings remain fixed; new model definitions own their default."""
    for agent in RELEASE_AGENTS:
        if agent.label == model and agent.driver == REFERENCE_DRIVER:
            return agent.reasoning
    return resolve_model(model).default_reasoning_profile
