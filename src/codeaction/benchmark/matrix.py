#!/usr/bin/env python3
"""Run selected agents/tasks through the durable, credential-aware batch scheduler.

The public eval command supplies resolved options. Dry-run prints the plan without
creating run directories; actual launches pin image identities and preserve batch state.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from codeaction.paths import PROJECT_ROOT


_RT = PROJECT_ROOT
_DATA = Path(os.environ.get("CODEACTION_RUNS_ROOT", PROJECT_ROOT / "runs")).resolve()

from codeaction.providers.model_registry import RegistryError, resolve_model
from codeaction.batch.controller import (  # noqa: E402
    BatchController, ControllerConfig, build_comparison_identity, inspect_official_images)
from codeaction.batch.scheduler import BatchScheduler  # noqa: E402
from codeaction.batch.spec import EXACT_CELL_MANIFEST_SCHEMA, resolve_cell_manifest  # noqa: E402
from codeaction.batch.state import BatchStateStore  # noqa: E402
from codeaction.benchmark.agents import (
    RELEASE_AGENTS, TEST_AGENTS, VENDOR_DRIVERS, agent_credential_limits, agent_credentials,
    build_agent_command, reference_agent, resolve_agents)
from codeaction.providers.provider_runtime import (
    load_rate_limit_config,
    rate_limit_policy_for,
    strict_rate_limit_coverage,
)
from codeaction.benchmark.taskcard import (
    TASKS_ROOT,
    declared_scene_seeds,
    load_task,
    validate_task_pack,
)
from codeaction.interface.tool_surface import INTERFACE_REFERENCE, INTERFACE_REFERENCE_CODE_FIRST


DEFAULT_MODELS = (
    "gpt-5.6",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.6-flash",
    "kimi-k3",
    "qwen3.8-max",
    "grok-4.6",
)
# Every entry must be in the released task set; `pick_red_box` left it in taskset 0.29.0.
ALL_TASKS = (
    "stack_blocks_three",
    "blocks_ranking_size",
    "handover_mic",
    "scan_object",
    "dump_bin_bigbin",
)
# Named rather than sliced: the pilot pair is a decision, not two offsets that silently move when
# a task joins or leaves the tuple above.
PILOT_TASKS = ("blocks_ranking_size", "handover_mic")
REMAINDER_TASKS = tuple(task for task in ALL_TASKS if task not in PILOT_TASKS)

# One-task estimates from existing stack_blocks_three runs.  They are scheduling hints only and
# never enter benchmark identity or results. Unknown Qwen/Grok revisions are conservatively placed
# between the measured medium and fast rows.
EXPECTED_MINUTES = {
    "claude-opus-5": 34,
    "gemini-3.6-flash": 33,
    "claude-sonnet-5": 17,
    "qwen3.8-max": 16,
    "qwen3.7-plus": 15,
    "kimi-k3": 13,
    "gpt-5.6": 8,
    # Latest measured blocks_ranking_size run: 21.63 minutes.  This is only a
    # longest-first scheduling hint and never enters benchmark identity.
    "grok-4.6": 22,
}
DEFAULT_CREDENTIAL_CONCURRENCY = 1


@dataclass(frozen=True)
class WorkItem:
    """One dynamically dispatched task while retaining model/task ordering identity."""

    gpu: int
    dispatch_index: int
    model: str
    task: str
    task_index: int
    credential: str | None


def released_tasks(task_pack: Path = TASKS_ROOT) -> tuple[str, ...]:
    """Every task the pack registers, in registry order -- the benchmark itself.

    The phases below are one historical staging of a five-task pilot. Evaluating a model on the
    benchmark means running the released set, so that has to be sayable without naming 25 tasks
    by hand."""
    return tuple(validate_task_pack(Path(task_pack))["tasks"])


def roster_with_models(models, roster=RELEASE_AGENTS):
    """The released roster plus a plain reference-scaffold agent for each named model.

    A model that is not on the roster is still a model the benchmark can measure: `--model` is
    how someone tests their own, registry overlay included, without editing a released file. The
    agent it synthesizes is the default seat -- reference driver, `reference-mcp`, reasoning at
    the top rung -- labelled by the model id.

    A name that ALREADY denotes an agent is never synthesized: `scripted` is the zero-cost agent
    that proves the machinery, and turning it into a provider-backed model would break the one
    selection someone can run for free."""
    from codeaction.benchmark.agents import CANDIDATE_AGENTS
    known = {agent.label for agent in tuple(roster) + TEST_AGENTS + CANDIDATE_AGENTS}
    from codeaction.extensions import declarations
    from codeaction.benchmark.agents import AgentConfig
    from codeaction.benchmark.agents import default_reasoning_for_model
    local = declarations("agent")
    conflicts = [name for name in local if name in known and not local[name].get("replace")]
    if conflicts:
        raise ValueError(f"agent names already exist; declare replace: true: {conflicts}")
    def configured(name):
        return AgentConfig(label=name, driver="reference", model="scripted-model", reasoning="none",
                           interface_profile="reference-mcp", reference_model_mode="local")
    selected_roster = tuple(configured(agent.label) if agent.label in local else agent for agent in roster)
    extra = tuple(configured(name) if name in local else reference_agent(name, reasoning=default_reasoning_for_model(name))
                  for name in dict.fromkeys(models or ())
                  if name not in known or (name in local and name not in {a.label for a in roster}))
    return selected_roster + extra


def phase_tasks(phase: str) -> tuple[str, ...]:
    if phase == "pilot":
        return PILOT_TASKS
    if phase == "remainder":
        return REMAINDER_TASKS
    raise ValueError(f"unknown phase {phase!r}")


def selected_tasks(phase: str, explicit_tasks: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return an explicit ordered task subset or the frozen phase selection.

    An explicit list is NOT restricted to the staged phase tasks: any task the selected pack
    registers may be scheduled, and `_validate` is what rejects one the pack does not have. The
    phases remain the default staging for a full matrix.
    """
    tasks = tuple(explicit_tasks or ())
    if not tasks:
        return phase_tasks(phase)
    if len(set(tasks)) != len(tasks):
        raise ValueError("task list contains duplicates")
    return tasks


def model_credentials(models: Iterable[str]) -> dict[str, str | None]:
    return {model: resolve_model(model).credential for model in models}


from codeaction.config_paths import (
    PROVIDER_ENV_VAR, PROVIDER_RATE_LIMIT_VAR, PROVIDER_ENV_CANDIDATES,
    PROVIDER_RATE_LIMIT_CANDIDATES, resolve_provider_file as _resolve_provider_file)


def parse_credential_limits(
    specifications: Iterable[str],
    credentials: Mapping[str, str | None],
    default_limit: int,
) -> dict[str, int]:
    if (not isinstance(default_limit, int) or isinstance(default_limit, bool)
            or default_limit < 1):
        raise ValueError("default credential concurrency limit must be a positive integer")
    selected = {credential for credential in credentials.values() if credential is not None}
    # ONLY the aliases someone named. Pre-filling every selected credential with the default made
    # this an override of all of them, and an override outranks the account's own declaration --
    # so a subscription that declares four lanes was silently run at one. The caller passes the
    # same default separately, and `agent_credential_limits` applies it only where no account and
    # no override says otherwise.
    limits: dict[str, int] = {}
    seen: set[str] = set()
    for specification in specifications:
        alias, separator, raw_limit = specification.partition("=")
        if not separator or not alias or not raw_limit:
            raise ValueError(
                f"credential limit must use ALIAS=N syntax, got {specification!r}")
        if alias not in selected:
            raise ValueError(
                f"credential limit names unused/unknown alias {alias!r}; selected={sorted(selected)}")
        if alias in seen:
            raise ValueError(f"credential limit duplicates alias {alias!r}")
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(f"credential limit must be an integer: {specification!r}") from exc
        if limit < 1:
            raise ValueError(f"credential limit must be positive: {specification!r}")
        limits[alias] = limit
        seen.add(alias)
    return limits


def parse_reasoning_profiles(
    models: Iterable[str], default_profile: str, specifications: Iterable[str],
) -> dict[str, str]:
    selected = tuple(models)
    profiles = {model: str(default_profile) for model in selected}
    seen = set()
    for specification in specifications:
        model, separator, profile = specification.partition("=")
        if not separator or model not in profiles or not profile:
            raise ValueError(
                f"reasoning override must use selected MODEL=RUNG syntax, got {specification!r}")
        if model in seen:
            raise ValueError(f"reasoning override duplicates model {model!r}")
        profiles[model] = profile
        seen.add(model)
    for model, profile in profiles.items():
        resolve_model(model).profile(profile)
    return profiles


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_preflight(root: Path = _RT) -> tuple[str, dict[str, Any]]:
    from codeaction.launch import context
    frozen = context()
    if frozen is not None:
        return frozen["source_commit"], {"kind": "execution_snapshot", "path": str(root)}
    if not (root / ".git").exists():
        from codeaction.release import source_identity
        return "0" * 40, {"kind": "source_archive", "source_sha256": source_identity(root),
                          "dirty": False}
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True, text=True, capture_output=True).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True).stdout.strip()
    from codeaction.release import source_identity
    return commit, {
        "kind": "git_checkout",
        "path": str(root.resolve()),
        "commit": commit,
        "dirty": bool(status),
        "source_sha256": source_identity(root),
    }


def _validate(agents, tasks: tuple[str, ...], task_pack: Path) -> None:
    labels = [agent.label for agent in agents]
    if len({agent.reference_model_mode == "local" for agent in agents}) > 1:
        raise ValueError("use separate output batches for local agents and built-in drivers")
    if len(set(labels)) != len(labels):
        raise ValueError("agent selection contains duplicates")
    for agent in agents:
        # Every agent's model must exist and expose the rung asked of it, whichever driver runs it.
        resolve_model(agent.model).profile(agent.reasoning)
    pack = validate_task_pack(task_pack)
    missing = [task for task in tasks if task not in pack.get("tasks", {})]
    if missing:
        raise ValueError(f"task pack does not register: {missing}")


def manifest_cell_order(
    agent_labels: Sequence[str],
    tasks: Sequence[str],
    attempts: int,
    seeds_by_task: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """The cell list for a manifest batch, in DISPATCH order.

    For a manifest target the manifest order IS the dispatch queue, so this decides scheduling.
    Rotating attempt -> task -> agent keeps consecutive cells on different agents: written
    agent-major, one agent's cells would sit at the head of the queue and serialise behind each
    other at one lane apiece while the box still had room for the others. It also means an
    interrupted batch holds one attempt for EVERY agent rather than every attempt for the first two,
    which is the more useful thing to be left holding.
    """
    return [
        {"model": label, "task": task, "attempt_index": attempt_index,
         "scene_seed": int(seeds_by_task[task][attempt_index])}
        for attempt_index in range(int(attempts))
        for task in tasks
        for label in agent_labels
    ]


def task_primary_seeds(tasks: Iterable[str], task_pack: Path) -> dict[str, int]:
    return {
        task: declared_scene_seeds(load_task(task, tasks_root=task_pack))[0]
        for task in tasks
    }


def _plan_payload(
    phase: str,
    tasks: tuple[str, ...],
    agents,
    gpus: tuple[int, ...],
    credential_limits: Mapping[str, int],
    rate_limit_policies: Mapping[str, Mapping[str, Any]],
    task_pack: Path,
    attempts: int = 1,
) -> dict[str, Any]:
    agents = tuple(agents)
    return {
        "phase": phase,
        "tasks_per_agent": list(tasks),
        "primary_scene_seeds": task_primary_seeds(tasks, task_pack),
        "attempts_per_cell": int(attempts),
        "total_cells": len(agents) * len(tasks) * int(attempts),
        "source_snapshot_required": True,
        "scheduler": {
            "kind": "credential-aware-dynamic-v1",
            "gpus": list(gpus),
            "agent_concurrency_limits": {
                agent.label: min(len(gpus), agent.lanes,
                                 credential_limits.get(agent.credential, agent.lanes))
                for agent in agents
            },
            "credential_concurrency_limits": dict(sorted(credential_limits.items())),
            "strict_rate_limit_coverage": {
                label: strict_rate_limit_coverage(policy)
                for label, policy in sorted(rate_limit_policies.items())
            },
            "priority": "continue-started-agent-then-longest-estimate",
        },
        "queue": [
            {
                "agent": agent.label,
                "driver": agent.driver,
                "model": agent.model,
                "reasoning_profile": agent.reasoning,
                "interface_profile": agent.interface_profile,
                "credential": agent.credential,
                "tasks": list(tasks),
                "expected_minutes_per_task": EXPECTED_MINUTES.get(agent.model, 20),
            }
            for agent in agents
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Staged agent-matrix runner: any mix of drivers, models and rungs.")
    parser.add_argument("--phase", choices=("pilot", "remainder"), default="pilot")
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="explicit ordered task subset; overrides --phase task selection")
    parser.add_argument(
        "--all-tasks", action="store_true",
        help="every task the pack registers, in registry order -- the released benchmark. "
             "Overrides --phase; cannot be combined with --tasks")
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="model ids to measure on the default seat (reference scaffold, reference-mcp, top "
             "reasoning rung). A model that is not on the released roster is synthesized, so "
             "testing your own needs no released file edited -- only a model-registry entry")
    parser.add_argument(
        "--agents", nargs="+", default=None, metavar="LABEL",
        help="agent labels from the released roster; default is the whole roster. An agent is "
             "(driver, model, rung, interface), so selecting by label is how a batch mixes "
             "harnesses as well as models")
    parser.add_argument(
        "--attempts", type=int, default=1,
        help="attempts per (agent, task) cell, each on its own declared scene seed")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--default-credential-limit", type=int, default=DEFAULT_CREDENTIAL_CONCURRENCY,
        help="maximum concurrent cells per selected credential alias (default: 1)")
    parser.add_argument(
        "--credential-limit", action="append", default=[], metavar="ALIAS=N",
        help="override one selected credential alias; repeat for multiple aliases")
    parser.add_argument("--provider-env-file", type=Path,
                        default=_resolve_provider_file(PROVIDER_ENV_VAR, PROVIDER_ENV_CANDIDATES))
    parser.add_argument(
        "--provider-rate-limit-file", type=Path,
        default=_resolve_provider_file(
            PROVIDER_RATE_LIMIT_VAR, PROVIDER_RATE_LIMIT_CANDIDATES),
        help="non-secret exact request/token windows; required for a paid matrix run")
    parser.add_argument(
        # No default: which file holds a vendor token is the account's declaration, resolved from
        # `agents.json` when a cell is built. A path under someone's home directory is not a
        # property of the benchmark.
        "--token-file", type=Path, default=None,
        help="subscription token for vendor-driver agents; unused by reference agents")
    parser.add_argument("--task-pack", type=Path, default=TASKS_ROOT)
    parser.add_argument("--run-profile", choices=("dev", "eval"), default="dev",
                        help="dev records strict-gate findings as non-submittable reasons; "
                             "eval refuses on them and is what a release stage must use")
    parser.add_argument("--assets-root", type=Path,
                        default=Path(os.environ.get("CODEACTION_ASSETS_ROOT", _RT / "assets")))
    parser.add_argument("--release-manifest", type=Path,
                        default=os.environ.get("CODEACTION_RELEASE_MANIFEST"))
    parser.add_argument("--sim-image", default="codeaction-sim:dev")
    parser.add_argument("--claude-agent-image", "--agent-image", dest="claude_agent_image", default="codeaction-claude-agent:dev")
    parser.add_argument("--codex-agent-image", default="codeaction-codex-agent:dev")
    parser.add_argument("--reference-agent-image", default="codeaction-reference-agent:dev")
    parser.add_argument("--gateway-image", default="codeaction-gateway:dev")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--python", default=os.environ.get(
        "CODEACTION_PYTHON", sys.executable))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and commands without creating run directories")
    args = parser.parse_args(argv)
    if args.release_manifest is not None:
        from codeaction.release import IMAGE_ARGS, load_release
        release = load_release(args.release_manifest, _RT)
        for role, ref in release["images"].items():
            setattr(args, IMAGE_ARGS[role], ref)

    if args.all_tasks and args.tasks:
        raise ValueError("--all-tasks and --tasks name two different selections; pass one")
    task_pack = args.task_pack.resolve()
    roster = roster_with_models(args.models)
    chosen_agents = list(args.agents or ()) + list(args.models or ())
    agents = resolve_agents(chosen_agents or None, roster)
    agents_by_name = {agent.label: agent for agent in agents}
    labels = tuple(agents_by_name)
    gpus = tuple(args.gpus)
    tasks = (released_tasks(task_pack) if args.all_tasks
             else selected_tasks(args.phase, args.tasks))
    # What the plan reports as its selection has to be what was actually selected: a batch over
    # the whole pack is not the "pilot" phase, and reading otherwise in a manifest is a lie.
    selection = ("released-pack" if args.all_tasks
                 else "explicit" if args.tasks else args.phase)
    if args.attempts < 1:
        raise ValueError("--attempts must be positive")
    _validate(agents, tasks, task_pack)
    credentials = agent_credentials(agents)
    # The scheduler axis is the agent label; the queue never interprets it, so mixing drivers
    # needs no scheduler change. Only the subscription's fixed limit is not overridable.
    credential_limits = agent_credential_limits(
        agents,
        parse_credential_limits(args.credential_limit, credentials,
                                args.default_credential_limit),
        default=args.default_credential_limit)
    needs_provider_credentials = any(agent.has_provider_profile for agent in agents)
    rate_limit_config = (
        load_rate_limit_config(args.provider_rate_limit_file.resolve())
        if needs_provider_credentials and args.provider_rate_limit_file.is_file() else None)
    # A vendor-driver agent spends a subscription, which declares no request/token windows; its
    # pacing is the fixed single-lane limit above, so it is exempt from the coverage requirement.
    rate_limit_policies = {
        agent.label: rate_limit_policy_for(rate_limit_config, agent.credential, agent.model)
        for agent in agents if agent.has_provider_profile
    }
    missing_rate_limits = [
        label for label, policy in rate_limit_policies.items()
        if not strict_rate_limit_coverage(policy)
    ]
    # Provider credential material is required only when a provider-backed agent is scheduled.
    # A subscription-only batch declares no request/token windows and reads no provider env: its
    # pacing is the fixed single lane, so demanding those files would block a valid batch.
    plan = _plan_payload(
        selection, tasks, agents, gpus, credential_limits,
        rate_limit_policies, task_pack, attempts=args.attempts)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # One directory per batch, and inside it agent / task / attempt -- the aggregation axis.
    batch_dir = (args.out_dir or (_DATA / f"batch_{stamp}")).resolve()
    controller_module = "codeaction.cli.main"

    if args.dry_run:
        if needs_provider_credentials and rate_limit_config is None:
            raise RuntimeError(
                f"provider rate-limit file does not exist: {args.provider_rate_limit_file}")
        if missing_rate_limits:
            raise RuntimeError(
                "matrix requires explicit request/token windows for every model; missing="
                f"{missing_rate_limits}")
        print(json.dumps({"batch_dir": str(batch_dir), **plan}, indent=2))
        print("# One cell per (agent, task, attempt); GPU is chosen when a cell becomes eligible.")
        print("# Image refs below are the tags; a real launch pins their content digests.")
        for agent in agents:
            for task in tasks:
                seeds = declared_scene_seeds(load_task(task, tasks_root=task_pack))
                for attempt_index in range(args.attempts):
                    run_dir = (batch_dir / "runs" / agent.label / task
                               / f"attempt-{attempt_index:03d}" / "execution-001" / "run")
                    command = build_agent_command(
                        agent, python_bin=args.python, controller_module=controller_module,
                        task=task, gpu="<dynamic>", run_dir=run_dir,
                        task_pack=task_pack, profile=args.run_profile,
                        attempt_index=attempt_index,
                        provider_env_file=(args.provider_env_file.resolve()
                                           if needs_provider_credentials else None),
                        provider_rate_limit_file=(args.provider_rate_limit_file.resolve()
                                                  if needs_provider_credentials else None),
                        token_file=(args.token_file.resolve() if args.token_file else None))
                    print(f"# seed {seeds[attempt_index]}")
                    print(shlex.join(command))
        return 0

    if args.release_manifest is not None:
        source_commit = release["source_commit"]
        source_snapshot = {"kind": "release_snapshot", "path": str(_RT),
                           "commit": source_commit}
    else:
        source_commit, source_snapshot = _git_preflight()
    # RESUME. A batch that stopped on an attention is meant to be inspected, resolved and picked
    # up again -- without this the whole requeue path is unreachable, because nothing would ever
    # execute the cell a human just put back in the queue. A directory holding batch_spec.json is
    # that batch; anything else non-empty is still refused.
    resuming = (batch_dir / "batch_spec.json").is_file()
    if batch_dir.exists() and not resuming and any(batch_dir.iterdir()):
        raise RuntimeError(f"batch directory already exists and is not a batch: {batch_dir}")
    if needs_provider_credentials and not args.provider_env_file.is_file():
        raise RuntimeError(f"provider environment file does not exist: {args.provider_env_file}")
    if needs_provider_credentials and rate_limit_config is None:
        raise RuntimeError(
            f"provider rate-limit file does not exist: {args.provider_rate_limit_file}")
    if missing_rate_limits:
        raise RuntimeError(
            "paid matrix requires explicit request/token windows for every model; missing="
            f"{missing_rate_limits}")

    batch_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------
    # From here the batch runs on the DURABLE state machine (codeaction.batch), not on an
    # in-memory queue. What that buys, and why the bespoke queue this replaced could not:
    #   - every (agent, task, attempt) is its own cell with its own lease, so a crash is
    #     recoverable and a single attempt can be re-run without touching its siblings;
    #   - a failure opens an ATTENTION with a scope, and the queue holds exactly what that
    #     scope names while a human decides (python -m codeaction.batch.control <dir> list);
    #   - dispatch stays fully dynamic: a GPU is refilled the moment its own episode ends,
    #     never at a barrier across the four.
    # Layout under the batch directory is agent / task / attempt, which is the aggregation
    # axis: runs/<agent>/<task>/attempt-NNN/<execution-id>/run. The execution level exists
    # because a requeued attempt must not overwrite the evidence of the one it replaces.
    # ------------------------------------------------------------------------------------
    seeds_by_task = {}
    for task in tasks:
        seeds = declared_scene_seeds(load_task(task, tasks_root=task_pack))
        if args.attempts > len(seeds):
            raise RuntimeError(
                f"{task} declares {len(seeds)} attempts; --attempts {args.attempts} exceeds it")
        seeds_by_task[task] = seeds
    manifest_cells = manifest_cell_order(
        [agent.label for agent in agents], tasks, args.attempts, seeds_by_task)

    if resuming:
        # The frozen manifest is the batch's own; rebuilding it from today's flags could quietly
        # change the cell set the store already owns.
        manifest_path = batch_dir / "cell_manifest.json"
    else:
        manifest_path = batch_dir.parent / f".{batch_dir.name}.cell_manifest.json"
        _atomic_json(manifest_path, {
            "schema_version": EXACT_CELL_MANIFEST_SCHEMA,
            "name": batch_dir.name.replace("-", "_").lower(),
            "cells": manifest_cells,
        })
    target = resolve_cell_manifest(manifest_path, tasks_root=task_pack)

    drivers = {agent.driver for agent in agents}
    # One agent image per batch: the frozen identity carries a single agent_image_digest, so a
    # batch mixing two vendor CLIs would freeze one image and launch the other. The reference
    # scaffold may share a batch with nothing, and each vendor seat is its own batch.
    vendor_drivers_selected = drivers & set(VENDOR_DRIVERS)
    if len(vendor_drivers_selected) > 1 or (vendor_drivers_selected and drivers != vendor_drivers_selected):
        raise SystemExit(
            f"a batch runs one agent image; selected drivers {sorted(drivers)} would need more")
    from codeaction.agents.runtime_registry import execution_driver
    driver_name = next(iter(drivers))
    image_refs = {
        "sim_image_digest": args.sim_image,
        "agent_image_digest": getattr(args, execution_driver(driver_name).image_option),
        "gateway_image_digest": args.gateway_image,
    }
    if resuming:
        # Both come from the batch, never re-derived: the store refuses a resume whose identity
        # differs, and re-inspecting images after a rebuild would be exactly such a difference.
        identity = json.loads(
            (batch_dir / "batch_spec.json").read_text())["comparison_identity"]
        image_digests = dict(identity["image_digests"])
    else:
        image_digests = inspect_official_images(image_refs, source_commit)
        identity = build_comparison_identity(
            source_commit=source_commit,
            agents=agents_by_name,
            rate_limit_policies=rate_limit_policies,
            image_digests=image_digests,
            run_profile=args.run_profile,
        )
    config = ControllerConfig(
        expected_stage=target.stage,
        repo_root=_RT,
        batch_dir=batch_dir,
        python_bin=args.python,
        controller_module=controller_module,
        task_pack=task_pack,
        provider_env_file=args.provider_env_file.resolve(),
        provider_rate_limit_file=args.provider_rate_limit_file.resolve(),
        image_digests=image_digests,
        rate_limit_policies=rate_limit_policies,
        gpus=gpus,
        environment={
            **{key: value for key, value in os.environ.items()
               if key in ("PATH", "HOME", "LANG", "PYTHONPATH", "CODEACTION_ROOT",
                          "CODEACTION_RUN_CONTEXT", "CODEACTION_RUNS_ROOT", "CODEACTION_AGENTS_CONFIG", "CODEACTION_MODEL_REGISTRY_FILE", "CODEACTION_EXTENSIONS_FILE", "PYTHONDONTWRITEBYTECODE")},
            "CODEACTION_ASSETS_ROOT": str(args.assets_root.expanduser().resolve()),
            **({"CODEACTION_RELEASE_MANIFEST": str(args.release_manifest.expanduser().resolve())}
               if args.release_manifest is not None else {}),
        },
        agents=agents_by_name,
        token_file=(args.token_file.resolve() if args.token_file else None),
        run_profile=args.run_profile,
    )
    with BatchStateStore.open_or_create(
            batch_dir, target=target, models=labels, comparison_identity=identity,
            model_lanes={agent.label: agent.lanes for agent in agents}) as store:
        # Now that the store owns the directory, everything else may live beside its state.
        if not resuming:
            manifest_path.replace(batch_dir / "cell_manifest.json")
        _atomic_json(batch_dir / "batch_meta.json", {
            **plan, "source_commit": source_commit, "source_snapshot": source_snapshot,
            "run_profile": args.run_profile, "image_digests": image_digests,
            "model_lanes": {agent.label: agent.lanes for agent in agents},
            "provider_env_file": "redacted",
        })
        # The progress monitor is kept from the previous runner: the durable state machine
        # records what happened, and this renders what is happening while it happens.
        monitor_dir = batch_dir / "_monitor"
        monitor_dir.mkdir(exist_ok=True)
        monitor_log = (monitor_dir / "watcher.log").open("w", encoding="utf-8")
        watcher = subprocess.Popen([
            args.python, "-m", "codeaction.reporting.watch", str(batch_dir / "runs"),
            "--watch", "10", "--output-dir", str(monitor_dir),
        ], stdout=monitor_log, stderr=subprocess.STDOUT)
        scheduler = BatchScheduler(
            store, model_credentials=credentials, credential_limits=credential_limits)
        controller = BatchController(store, scheduler, config=config)
        print(f"[matrix] {'resuming' if resuming else 'batch'}={batch_dir} "
              f"cells={len(store.snapshot()['requested_cells'])} "
              f"lanes={ {agent.label: agent.lanes for agent in agents} }", flush=True)
        try:
            code = int(controller.run())
        finally:
            watcher.terminate()
            try:
                watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                watcher.kill()
                watcher.wait()
            # The periodic watcher can be up to ten seconds behind the last cell; write one
            # final snapshot so a finished batch never renders as still running.
            subprocess.run([
                args.python, "-m", "codeaction.reporting.watch", str(batch_dir / "runs"),
                "--once", "--output-dir", str(monitor_dir),
            ], stdout=monitor_log, stderr=subprocess.STDOUT, check=False)
            monitor_log.close()
    if code == 1:
        print("[matrix] the batch stopped on an attention. Inspect and resolve it with:\n"
              f"  python -m codeaction.batch.control {batch_dir} list", file=sys.stderr)
    elif code == 0:
        print(f"[matrix] complete: {batch_dir}")
    else:
        # A controller error is recorded on the controller and was going nowhere: the batch
        # exited with its cells still queued and no line saying why, which cost four rounds of
        # reproducing it by hand.
        print(f"[matrix] controller error: {controller.last_error}", file=sys.stderr)
        print(f"[matrix] batch left intact for inspection: {batch_dir}", file=sys.stderr)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, RegistryError) as exc:
        print(f"[matrix] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
