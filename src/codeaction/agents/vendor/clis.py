"""The vendor agent CLIs this benchmark can seat, and how each one meets the contract.

One entry per third-party agent stack.  Everything that differs between stacks lives HERE -- the
image, the credential shape, the compose service, the stream normalizer, and one declaration per
line of ``contract.py``.  Nothing else in the controller may branch on which vendor is running:
the previous shape of this code was twenty ``agent_mode == "claude"`` tests scattered through
``cli/main.py``, and every one of them was a place a second stack could be forgotten.

The declarations are not documentation. ``__post_init__`` refuses a CLI that leaves any contract
item undeclared, that claims a required capability with an evidence kind the contract does not
accept for it, or that marks a banned capability anything but denied/residual.  Onboarding a new
stack therefore cannot skip a line: the import fails until all of them are answered.

ASYMMETRIES ARE DECLARED, NOT SMOOTHED.  Two stacks meeting the same item by different mechanisms
is normal and fine.  Two stacks meeting it with different evidence STRENGTH is the thing that
would quietly corrupt a comparison, so the evidence kind is part of the declaration and the
cross-CLI attestation reports the weakest kind that established each claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from codeaction.agents.vendor.contract import (BANNED_IDS, DISPOSITIONS, EVIDENCE_KINDS,
                                               REQUIRED_IDS, capability)


@dataclass(frozen=True)
class Disposition:
    """How one CLI answers one line of the contract."""

    capability: str
    state: str
    evidence: str
    mechanism: str
    # Written only where the mechanism alone would overstate the claim.  Every note here was a
    # thing someone would otherwise have had to rediscover from a failed run.
    note: str = ""

    def __post_init__(self) -> None:
        capability(self.capability)          # raises on an unknown id
        if self.state not in DISPOSITIONS:
            raise ValueError(f"{self.capability}: unknown state {self.state!r}")
        if self.evidence not in EVIDENCE_KINDS:
            raise ValueError(f"{self.capability}: unknown evidence kind {self.evidence!r}")
        if self.evidence not in capability(self.capability).evidence:
            raise ValueError(
                f"{self.capability}: the contract does not accept {self.evidence!r} evidence "
                f"for this item; accepted={list(capability(self.capability).evidence)}")
        if not self.mechanism:
            raise ValueError(f"{self.capability}: a disposition must name its mechanism")


@dataclass(frozen=True)
class CredentialMount:
    """What a subscription seat mounts into the agent container.

    The two stacks differ in SHAPE, not merely in path: Claude Code reads one shell file that
    exports a token, while Codex needs a whole writable CODEX_HOME (auth.json plus the sqlite
    files it opens at startup).  A seat that assumed 'a token file' is why this is a type.
    """

    kind: str                # token_file | config_home
    container_path: str
    container_env: str       # the variable inside the container that names it
    compose_env: str         # the compose variable the controller sets to the host path

    def __post_init__(self) -> None:
        if self.kind not in ("token_file", "config_home"):
            raise ValueError(f"unknown credential kind {self.kind!r}")


@dataclass(frozen=True)
class VendorCli:
    """One seatable agent stack."""

    name: str
    agent_mode: str
    cli_label: str            # image label org.codeaction.agent-cli
    version: str
    compose_profile: str
    compose_service: str
    image_compose_env: str
    # The controller flag a batch uses to pin this seat's image by digest. It lives here for the
    # same reason the compose service does: the batch must not hold a table of its own that can
    # drift from this one.
    image_cli_flag: str
    dockerfile: str
    credential: CredentialMount
    sidecar_module: str
    effort_compose_env: str
    # How the episode's tools become visible to the model.  Claude Code defers MCP schemas behind
    # ToolSearch and eagerly loads a declared subset; Codex has no deferral and shows the whole
    # surface from turn one.  Same tools, same delivered_sha256, different discovery -- so it is
    # recorded on the run rather than left as an unstated difference between two scores.
    tool_discovery: str
    dispositions: tuple[Disposition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.tool_discovery not in ("deferred_toolsearch", "eager_all"):
            raise ValueError(f"{self.name}: unknown tool discovery {self.tool_discovery!r}")
        by_id = {}
        for item in self.dispositions:
            if item.capability in by_id:
                raise ValueError(f"{self.name}: duplicate disposition for {item.capability}")
            by_id[item.capability] = item
        missing = [key for key in REQUIRED_IDS + BANNED_IDS if key not in by_id]
        if missing:
            raise ValueError(
                f"{self.name} leaves contract item(s) undeclared: {missing}. A stack is seated "
                f"only when every line of the contract has an answer.")
        for key in REQUIRED_IDS:
            if by_id[key].state not in ("satisfied", "unmet"):
                raise ValueError(
                    f"{self.name}.{key} is required; it is satisfied or unmet, not "
                    f"{by_id[key].state!r}")
        for key in BANNED_IDS:
            if by_id[key].state not in ("denied", "residual"):
                raise ValueError(
                    f"{self.name}.{key} is banned; it is denied or residual, not "
                    f"{by_id[key].state!r}")

    def disposition(self, capability_id: str) -> Disposition:
        for item in self.dispositions:
            if item.capability == capability_id:
                return item
        raise KeyError(f"{self.name} declares nothing for {capability_id!r}")

    @property
    def unmet(self) -> tuple[str, ...]:
        """Required capabilities this stack cannot provide. A non-empty tuple disqualifies it."""
        return tuple(item.capability for item in self.dispositions if item.state == "unmet")

    @property
    def residual(self) -> tuple[str, ...]:
        """Banned capabilities that survive in the CLI and are neutralized rather than removed."""
        return tuple(item.capability for item in self.dispositions if item.state == "residual")

    def summary(self) -> dict:
        """The declaration, as it is stamped into a run's vendor attestation."""
        return {
            "cli": self.cli_label,
            "cli_version": self.version,
            "agent_mode": self.agent_mode,
            "tool_discovery": self.tool_discovery,
            "unmet_required": list(self.unmet),
            "residual_banned": list(self.residual),
            "evidence_by_capability": {
                item.capability: {"state": item.state, "evidence": item.evidence}
                for item in self.dispositions
            },
        }


# =============================================================================================
# Claude Code.  The first seat; these declarations describe what it has been doing since the
# vendor path was built, written down rather than newly imposed.
# =============================================================================================
CLAUDE = VendorCli(
    name="claude",
    agent_mode="claude",
    cli_label="claude-code",
    version="2.1.212",
    compose_profile="provider",
    compose_service="claude-agent",
    image_compose_env="CODEACTION_CLAUDE_AGENT_IMAGE",
    image_cli_flag="--claude-agent-image",
    dockerfile="docker/claude-agent.Dockerfile",
    credential=CredentialMount(
        kind="token_file",
        container_path="/run/secrets/codeaction/claude_oauth_token.sh",
        container_env="CLAUDE_OAUTH_TOKEN_FILE",
        compose_env="CODEACTION_TOKEN_FILE"),
    sidecar_module="codeaction.agents.vendor.stream_sidecar",
    effort_compose_env="CODEACTION_CLAUDE_EFFORT",
    tool_discovery="deferred_toolsearch",
    dispositions=(
        Disposition("machine_readable_stream", "satisfied", "stream",
                    "--output-format stream-json --verbose; terminal event type=result"),
        Disposition("per_call_tool_record", "satisfied", "stream",
                    "assistant.content[].tool_use paired to user.content[].tool_result by "
                    "tool_use_id",
                    note="The pairing is explicit, so a call that never returned is directly "
                         "observable as an unmatched tool_use_id."),
        Disposition("token_usage_totals", "satisfied", "stream",
                    "result.usage and result.modelUsage"),
        Disposition("reasoning_spend_counter", "satisfied", "stream",
                    "system/thinking_tokens estimated_tokens"),
        Disposition("reasoning_text_capture", "satisfied", "launch",
                    "--thinking-display summarized",
                    note="Without it Opus 5 emits thinking blocks with empty bodies: measured "
                         "2,914 empty blocks across 85 release episodes before the flag was "
                         "added. The spend was identical; only the text was missing."),
        Disposition("model_selection", "satisfied", "stream",
                    "--model, echoed in system/init.model and every assistant.message.model"),
        Disposition("effort_selection", "satisfied", "launch",
                    "--effort plus CLAUDE_CODE_EFFORT_LEVEL"),
        Disposition("mcp_stdio_client", "satisfied", "server",
                    "--mcp-config /run/codeaction/mcp.json --strict-mcp-config"),
        Disposition("image_tool_results", "satisfied", "stream",
                    "image blocks inside user.content[].tool_result, counted by the sidecar"),
        Disposition("tool_allowlist", "satisfied", "stream",
                    "--tools ToolSearch restricts the BUILT-IN set to one tool; --allowedTools "
                    "mcp__codeaction__*,ToolSearch gates execution; the roster is echoed in "
                    "system/init.tools and the sidecar fails the attestation on anything else",
                    note="FAILS CLOSED, which is the property that matters and the reason this "
                         "is stream evidence rather than launch. `--tools` is a roster "
                         "allowlist over the built-in set, so a tool this CLI grows in a later "
                         "version is absent by default rather than newly permitted. "
                         "--disallowedTools is also passed and is the weakest of the three: a "
                         "deny list cannot name a tool that does not exist yet."),
        Disposition("nonpersistent_session", "satisfied", "launch",
                    "--no-session-persistence plus a fresh HOME per invocation"),
        Disposition("pinned_version", "satisfied", "container",
                    "npm install of an exact version, asserted at build time; "
                    "DISABLE_AUTOUPDATER=1"),
        Disposition("deterministic_exit", "satisfied", "stream",
                    "result.subtype and result.api_error_status, with the process exit code"),

        Disposition("filesystem_read", "denied", "container",
                    "the agent image contains no RoboTwin source or assets, and the compose "
                    "service mounts none; Read/Glob/Grep are additionally disallowed"),
        Disposition("shell_execution", "denied", "launch",
                    "--disallowedTools Bash (also NotebookEdit, Write, Edit)"),
        Disposition("network_egress", "residual", "container",
                    "the container joins the unrestricted `egress` bridge network; what the "
                    "MODEL can reach is bounded by the fails-closed roster above, and what the "
                    "CLI PROCESS reaches is bounded only by DISABLE_TELEMETRY, "
                    "DISABLE_ERROR_REPORTING, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC and "
                    "DISABLE_AUTOUPDATER",
                    note="The residual is the PROCESS, not the model: the model cannot emit "
                         "anything but a tool call, and no tool it has reaches the network. The "
                         "process controls are environment variables -- cooperative, "
                         "launch-evidence, and they fail open on a version bump. What that "
                         "risks is not ground truth (this container holds no repository, task "
                         "pack or solution) and not the episode reaching the vendor (it must, "
                         "to run at all), but the episode ADDITIONALLY reaching a third party. "
                         "The strongest argument for closing it is consistency: this design "
                         "says the wall is the mount table rather than a promise about a tool "
                         "list, and applying container evidence to the filesystem while "
                         "accepting launch evidence for the network is the same promise in a "
                         "different place. Left open deliberately, because narrowing it changes "
                         "this seat's environment and therefore its comparability with the "
                         "published baseline -- measure first with a logging proxy, then "
                         "enforce on BOTH seats together."),
        Disposition("web_retrieval", "denied", "stream",
                    "no retrieval tool is in the built-in roster --tools admits, none is in "
                    "--allowedTools, and the observed roster in system/init.tools is checked "
                    "against that every episode",
                    note="Credited to the roster allowlist, not to --disallowedTools "
                         "WebFetch,WebSearch, which is also passed. The distinction is not "
                         "pedantic: this repository publishes task names, so a retrieval tool "
                         "is a live leak path, and a deny list would not cover a differently "
                         "named one added in a point release."),
        Disposition("subagent_delegation", "denied", "launch",
                    "--disallowedTools Task; --no-chrome; agent-teams env unset"),
        Disposition("persistent_memory", "denied", "launch",
                    "vendor_settings.json autoMemoryEnabled=false; "
                    "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1; isolated HOME"),
        Disposition("instruction_injection", "denied", "launch",
                    "--settings with enabledPlugins={}; --disable-slash-commands; "
                    "--disallowedTools Skill,Workflow; CLAUDE_CODE_DISABLE_CLAUDE_MDS=1; "
                    "an empty temporary cwd"),
        Disposition("background_execution", "denied", "launch",
                    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1; CLAUDE_CODE_DISABLE_CRON=1"),
        Disposition("vendor_code_sandbox", "denied", "launch",
                    "Claude Code has no code-execution surface separate from Bash, which is "
                    "disallowed; run_code arrives only as an MCP tool"),
    ),
)


# =============================================================================================
# Codex.  Second seat.  Where a mechanism differs from Claude's it is because the CLI is
# genuinely different, and where the EVIDENCE differs the declaration says so instead of
# borrowing the stronger word.
# =============================================================================================
CODEX = VendorCli(
    name="codex",
    agent_mode="codex",
    cli_label="codex",
    version="0.154.0",
    compose_profile="codex",
    compose_service="codex-agent",
    image_compose_env="CODEACTION_CODEX_AGENT_IMAGE",
    image_cli_flag="--codex-agent-image",
    dockerfile="docker/codex-agent.Dockerfile",
    credential=CredentialMount(
        kind="config_home",
        container_path="/run/secrets/codeaction/codex-home",
        container_env="CODEACTION_CODEX_AUTH_DIR",
        compose_env="CODEACTION_CODEX_HOME"),
    sidecar_module="codeaction.agents.vendor.codex_stream_sidecar",
    effort_compose_env="CODEACTION_CODEX_EFFORT",
    tool_discovery="eager_all",
    dispositions=(
        Disposition("machine_readable_stream", "satisfied", "stream",
                    "codex exec --json; terminal event turn.completed, turn.failed or "
                    "thread.failed"),
        Disposition("per_call_tool_record", "satisfied", "stream",
                    "one item.completed of type mcp_tool_call carrying arguments AND result",
                    note="Call and result share a single record, so there is no unmatched-call "
                         "state to observe the way Claude's tool_use_id pairing gives one. A "
                         "call that never returned is absent from the stream entirely, which is "
                         "why the server's own tool_surface and call count are the cross-check."),
        Disposition("token_usage_totals", "satisfied", "stream",
                    "turn.completed.usage: input_tokens, cached_input_tokens, "
                    "cache_write_input_tokens, output_tokens",
                    note="No currency field exists, unlike Claude Code's total_cost_usd. "
                         "Comparison is in tokens for both seats, which is the repository's "
                         "existing position anyway."),
        Disposition("reasoning_spend_counter", "satisfied", "stream",
                    "turn.completed.usage.reasoning_output_tokens"),
        Disposition("reasoning_text_capture", "satisfied", "launch",
                    "-c model_reasoning_summary=auto",
                    note="gpt-6-astra's served model record declares "
                         "default_reasoning_summary=\"none\": without this the rung is funded "
                         "and the transcript is empty. Same failure mode as Claude's "
                         "thinking.display, different switch."),
        Disposition("model_selection", "satisfied", "launch",
                    "codex exec -m gpt-6-astra",
                    note="WEAKER EVIDENCE THAN CLAUDE. Codex emits no init event and no model "
                         "field on any item, so nothing in the stream confirms which model "
                         "answered. The claim rests on argv alone. Do not report it as "
                         "stream-verified."),
        Disposition("effort_selection", "satisfied", "launch",
                    "-c model_reasoning_effort=high",
                    note="Astra's ladder is low/medium/high/xhigh/max/ultra. The registry "
                         "declares low/medium/high/xhigh; the released roster runs high, and "
                         "xhigh is opt-in for thinking-depth comparisons. `max` is not declared, "
                         "and `ultra` is additionally barred by subagent_delegation."),
        Disposition("mcp_stdio_client", "satisfied", "server",
                    "-c mcp_servers.codeaction.command pointed at the stdio-to-TCP client"),
        Disposition("image_tool_results", "satisfied", "stream",
                    "image content blocks inside the mcp_tool_call result, counted by the "
                    "sidecar exactly as the first seat counts blocks inside tool_result",
                    note="Verified live before integration: on a randomized six-colour panel "
                         "served over MCP, both subscription accounts returned the colours in "
                         "correct row-major order. This is the capability whose absence ruled "
                         "out a different candidate CLI."),
        Disposition("tool_allowlist", "satisfied", "launch",
                    "features.* disabled one native surface at a time, then "
                    "codex_feature_gate.py reads `codex features list` at container start and "
                    "aborts before the first model call if anything enabled is outside "
                    "codex_feature_allowlist.txt; the codeaction server's own delivered "
                    "surface is the MCP half",
                    note="WEAKER THAN CLAUDE IN TWO SEPARATE WAYS, and the second is the "
                         "sharper one. (1) Evidence: Codex publishes no roster event, so the "
                         "offered list rests on the launch config and the server's delivered "
                         "surface, never on the model's account of its own tools, which is "
                         "self-report. (2) Fail direction: Codex has no equivalent of `--tools`, "
                         "so its native surface is governed by a DENY LIST OF FEATURE NAMES, "
                         "which fails OPEN -- a native tool added under a new feature name in a "
                         "point release is enabled by default and named in no list here. That "
                         "is the same structure that let features.unified_exec stay true. "
                         "The startup gate inverts it: the CLI's own report of EFFECTIVE state "
                         "is compared to an allowlist and a new enabled feature aborts the "
                         "episode before any model call, so the fail direction is now closed "
                         "at startup rather than only detected after use. The sidecar still "
                         "fails any stream item type it does not recognize as a second layer."),
        Disposition("nonpersistent_session", "satisfied", "container",
                    "a per-invocation throwaway CODEX_HOME that is destroyed with the "
                    "container; the session rollout it accumulates is copied into the vendor "
                    "output as evidence (auth.json excluded) and never reaches a later episode",
                    note="Not --ephemeral. The rollout is the only trace of what the model "
                         "executed in the code-mode host between MCP calls, so it is kept -- "
                         "inside a home no other episode can see."),
        Disposition("pinned_version", "satisfied", "container",
                    "the release tarball is fetched by exact tag and verified against a pinned "
                    "SHA-256 at build time; no auto-update path is installed"),
        Disposition("deterministic_exit", "satisfied", "stream",
                    "turn.completed versus turn.failed / thread.failed, with the process exit "
                    "code"),

        Disposition("filesystem_read", "residual", "container",
                    "features.view_image=false removes the image reader; apply_patch and the "
                    "MCP resource readers have no flag and remain; the agent image and its "
                    "service mount no RoboTwin source, assets, task pack or results, so none of "
                    "them resolves anything",
                    note="view_image IS suppressible -- it is a stable feature flag and "
                         "`codex features list` confirms it goes false -- which is why this "
                         "residual is narrower than the admission probe's. What remains has no "
                         "switch. It is neutralized by the mount table, and that was checked "
                         "live during admission: an actual view_image attempt against a "
                         "host-only canary returned ENOENT from inside the namespace. The wall "
                         "is what is not mounted, never a promise that the tool list is empty."),
        Disposition("shell_execution", "residual", "container",
                    "features.shell_tool=false sticks; sandbox_mode=read-only; the agent image "
                    "carries no RoboTwin source, assets, task pack or results, and the service "
                    "mounts none",
                    note="features.unified_exec=false DOES NOT STICK on 0.154.0. Measured three "
                         "ways -- config file, -c override and --disable -- and `codex features "
                         "list` reports it `true` in all three, while all 36 other flags this "
                         "seat sets do take effect. So the shell channel is bounded by the "
                         "container and the read-only sandbox, not by a flag, and the sidecar "
                         "fails any attempt that reaches the stream. Claiming `denied` here "
                         "would have been a claim about a flag that does nothing."),
        Disposition("network_egress", "residual", "container",
                    "the container joins the same unrestricted `egress` bridge network the "
                    "Claude seat uses",
                    note="Deliberately the SAME posture as the first seat rather than a "
                         "stricter one: an environment difference between two seats is a "
                         "confound in every number they are compared on. A standalone "
                         "exact-host CONNECT proxy was built and verified during admission "
                         "(example.com, 127.0.0.1 and a look-alike host all denied, "
                         "chatgpt.com:443 allowed), so tightening BOTH seats is a known, "
                         "available follow-up."),
        Disposition("web_retrieval", "denied", "launch",
                    "-c web_search=disabled, features.web_search_request=false, "
                    "supports_search_tool left unused"),
        Disposition("subagent_delegation", "residual", "launch",
                    "features.multi_agent=false, features.multi_agent_v2=false, and the `ultra` "
                    "rung -- documented as maximum reasoning WITH automatic task delegation -- "
                    "is never selected",
                    note="The flags do not remove the preamble. gpt-6-astra's served record "
                         "declares multi_agent_version=v2, and a `<multi_agent_role>` developer "
                         "message (2,429 bytes, naming spawn_agent / followup_task / "
                         "send_message) is still rendered with both flags false -- measured "
                         "offline via `codex debug prompt-input`. A following "
                         "`<multi_agent_mode>` message tells the model not to spawn unless "
                         "asked. Delegation is therefore discouraged and unconfigured rather "
                         "than absent, and an actual delegation item in the stream fails the "
                         "attestation by name."),
        Disposition("persistent_memory", "denied", "launch",
                    "features.memories=false, features.goals=false, --ephemeral, and a "
                    "throwaway CODEX_HOME"),
        Disposition("instruction_injection", "denied", "launch",
                    "a throwaway CODEX_HOME holding only auth.json and the image's own "
                    "config.toml; project_doc_max_bytes=0; --ignore-rules; features.plugins, "
                    "plugin_sharing, remote_plugin, hooks, skill_search, "
                    "skill_mcp_dependency_install and apps all false",
                    note="The ban is on channels that could inject instructions WE DID NOT "
                         "WRITE -- host skills, user config, project docs, rules files, "
                         "plugins -- and a throwaway CODEX_HOME closes all of them. Measured "
                         "with canary files via `codex debug prompt-input` on 0.154.0: "
                         "project_doc_max_bytes=0 blocks ./AGENTS.md but NOT $CODEX_HOME/"
                         "AGENTS.md, and features.skill_search=false does NOT block a user "
                         "skill under $CODEX_HOME/skills/ -- both are injected. So the throwaway "
                         "home is load-bearing, and codex_feature_gate.py proves it holds only "
                         "auth.json and config.toml before every start, then renders the "
                         "model-visible input via `codex debug prompt-input` for THIS episode's "
                         "prompt and refuses unless every preamble block matches the baked "
                         "manifest by hash and the final block is our prompt byte for byte. "
                         "Codex still "
                         "renders its own built-in preamble: measured offline on 0.154.0, three "
                         "developer messages totalling 7,141 bytes (a skills roster, the "
                         "multi-agent role, the multi-agent retraction) that neither "
                         "features.skill_search=false nor the model record's own "
                         "include_skills_usage_instructions=false removes. That is the vendor's "
                         "system prompt, i.e. the stack under test, exactly as Claude Code's own "
                         "preamble is; it is declared here so a token comparison is read with it "
                         "in view, not treated as a leak."),
        Disposition("background_execution", "denied", "launch",
                    "features.deferred_executor=false and a non-interactive exec invocation "
                    "that ends with the turn"),
        Disposition("vendor_code_sandbox", "residual", "container",
                    "features.code_mode_host=true with the host binary installed; "
                    "features.code_mode and code_mode_only stay false; the host runs inside the "
                    "same read-only, repository-free container as the CLI",
                    note="CANNOT BE DENIED FOR THIS MODEL. gpt-6-astra's served record declares "
                         "tool_mode=\"code_mode_only\" and the CLI enforces it: on the first "
                         "live attempt with the host disabled, the tool router failed closed "
                         "('Code Mode is unavailable because code-mode host is disabled') and "
                         "the model was handed no tools at all -- not even the benchmark's. The "
                         "host is therefore this model's ONLY tool transport, and the interface "
                         "difference is real and declared: the model can compute in the host's "
                         "runtime in addition to the benchmark's run_code, so run_code usage "
                         "counts on this seat are not comparable to a direct-exposure seat's. "
                         "The earlier reading of this ban as a no-op precaution was wrong."),
    ),
)


VENDOR_CLIS: dict[str, VendorCli] = {cli.agent_mode: cli for cli in (CLAUDE, CODEX)}
VENDOR_AGENT_MODES: tuple[str, ...] = tuple(VENDOR_CLIS)


def vendor_cli(agent_mode: str) -> VendorCli:
    try:
        return VENDOR_CLIS[agent_mode]
    except KeyError:
        raise ValueError(
            f"{agent_mode!r} is not a vendor agent mode; known: {list(VENDOR_CLIS)}") from None


def is_vendor_mode(agent_mode: str) -> bool:
    return agent_mode in VENDOR_CLIS


__all__ = ("CLAUDE", "CODEX", "CredentialMount", "Disposition", "VENDOR_AGENT_MODES",
           "VENDOR_CLIS", "VendorCli", "is_vendor_mode", "vendor_cli")
