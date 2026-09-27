"""One prepared execution path for single runs, batches and reference replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from codeaction.execution_snapshot import prepare_snapshot, save_launch_record, verify_snapshot


def context() -> dict | None:
    path = os.environ.get("CODEACTION_RUN_CONTEXT")
    return json.loads(Path(path).read_text()) if path else None


def _argv(parser, args) -> list[str]:
    command_parser = next(action for action in parser._actions
                          if isinstance(action, argparse._SubParsersAction)).choices[args.command]
    values = vars(args)
    result = [args.command]
    for action in command_parser._actions:
        if not action.option_strings or action.dest in {"help", "config", "extensions", "model_registry",
                                                       "require_release_match", "release_manifest"}:
            continue
        value = values.get(action.dest)
        if value is None or value == []:
            continue
        flag = next((x for x in action.option_strings if x.startswith("--")), action.option_strings[0])
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(flag)
        elif isinstance(value, list):
            if isinstance(action, argparse._AppendAction):
                for item in value:
                    result.extend([flag, str(item)])
            else:
                result.extend([flag, *map(str, value)])
        else:
            result.extend([flag, str(value)])
    return result


def preview(parser, args, root: Path) -> int:
    """Validate local definitions without Docker, permanent snapshots or output directories."""
    import copy
    args = copy.copy(args)
    with tempfile.TemporaryDirectory(prefix="codeaction-plan-") as temporary:
        baseline = getattr(args, "release_manifest", None) or os.environ.get("CODEACTION_RELEASE_MANIFEST")
        snapshot, info = prepare_snapshot(root, Path(temporary), task_pack=getattr(args, "task_pack", None),
            extension_paths=getattr(args, "extensions", []), model_registry=getattr(args, "model_registry", None),
            baseline=Path(baseline) if baseline else None, require_match=args.require_release_match)
        if args.command != "eval":
            print(json.dumps({"command": args.command, "task": getattr(args, "task", None),
                              "tasks": getattr(args, "tasks", None), "baseline": info["baseline"],
                              "changes": info["changes"], "containers_started": False}))
            return 0
        args.task_pack = snapshot / "benchmark/tasks"
        for key, value in vars(args).items():
            if isinstance(value, Path):
                setattr(args, key, value.expanduser().resolve())
        env = dict(os.environ, PYTHONPATH=str(snapshot / "src"), CODEACTION_ROOT=str(snapshot),
                   CODEACTION_RUN_CONTEXT=str(snapshot / "config/context.json"),
                   CODEACTION_MODEL_REGISTRY_FILE=str(snapshot / "config/models.json"),
                   CODEACTION_EXTENSIONS_FILE=str(snapshot / "config/extensions.json"), PYTHONDONTWRITEBYTECODE="1")
        env.pop("CODEACTION_RELEASE_MANIFEST", None)
        return subprocess.run([sys.executable, "-m", "codeaction.cli.main", *_argv(parser, args)],
                              env=env, cwd=snapshot, check=False).returncode


def launch(parser, args, root: Path) -> int:
    from codeaction.cli.main import _inspect_image
    from codeaction.agents.runtime_registry import execution_driver
    command = args.command
    output = getattr(args, "run_dir", None) if command == "run" else getattr(args, "out_dir", None)
    if output is None:
        parent = root / "runs"
        parent.mkdir(parents=True, exist_ok=True)
        # Reserve a unique name; the actual controller still owns directory creation.
        temporary = Path(tempfile.mkdtemp(prefix=command + "-", dir=parent))
        temporary.rmdir()
        output = temporary
    output = Path(output).expanduser().resolve()
    record_path = output.parent / f".{output.name}.execution.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("schema_version") == "codeaction-execution.v1" and command == "eval":
            from codeaction.cli.main import run_eval
            return run_eval(args)
        if record.get("schema_version") != "codeaction-execution.v2" or record.get("command") != command:
            raise ValueError("output belongs to a different execution; choose a new output directory")
        snapshot = Path(record["source_root"])
        verify_snapshot(snapshot)
        if record["output"] != str(output):
            raise ValueError("saved output differs from the requested output")
    else:
        if getattr(args, "matrix_args", None):
            raise ValueError("snapshot launches require named options, not matrix-arg")
        baseline = getattr(args, "release_manifest", None) or os.environ.get("CODEACTION_RELEASE_MANIFEST")
        configuration_files = {}
        provider_file = None
        needs_accounts = False
        needs_rate_limits = False
        if command == "eval":
            from codeaction.benchmark.agents import RELEASE_AGENTS, TEST_AGENTS, CANDIDATE_AGENTS, expand_agent_selection
            from codeaction.extensions import read_declarations
            from codeaction.config_paths import (resolve_provider_file as _resolve_provider_file, PROVIDER_ENV_VAR, PROVIDER_ENV_CANDIDATES,
                                                     PROVIDER_RATE_LIMIT_VAR, PROVIDER_RATE_LIMIT_CANDIDATES)
            args.models = list(expand_agent_selection(args.models))
            known = {a.label: a for a in RELEASE_AGENTS + TEST_AGENTS + CANDIDATE_AGENTS}
            local = {e["name"]: e for e in read_declarations(args.extensions) if e["kind"] == "agent"}
            needs_rate_limits = any(name not in local and (known[name].has_provider_profile if name in known else True) for name in args.models)
            needs_accounts = any(name not in local and name in known and known[name].uses_subscription for name in args.models)
            needs_provider = any(bool(local[name].get("credential")) if name in local else
                                 known[name].has_provider_profile if name in known else True for name in args.models)
            if needs_provider:
                args.provider_env_file = args.provider_env_file or _resolve_provider_file(PROVIDER_ENV_VAR, PROVIDER_ENV_CANDIDATES)
                args.provider_rate_limit_file = args.provider_rate_limit_file or _resolve_provider_file(PROVIDER_RATE_LIMIT_VAR, PROVIDER_RATE_LIMIT_CANDIDATES)
                provider_file = args.provider_env_file
        elif command == "run" and args.agent_mode == "reference":
            from codeaction.extensions import read_declarations
            local = {e["name"]: e for e in read_declarations(args.extensions) if e["kind"] == "agent"}
            if args.reference_model_mode == "local":
                needs_provider = bool(local.get(args.agent_label, {}).get("credential"))
            else:
                needs_provider = args.reference_model_mode != "scripted" and args.model not in {"scripted", "scripted-model"}
            if needs_provider:
                provider_file = args.provider_env_file
                needs_rate_limits = args.reference_model_mode != "local"
        if provider_file is not None and not provider_file.expanduser().is_file():
            raise ValueError("provider credential file is missing; configure authentication before starting a new run")
        rate_path = getattr(args, "provider_rate_limit_file", None)
        if needs_rate_limits and rate_path is None:
            from codeaction.config_paths import resolve_provider_file, PROVIDER_RATE_LIMIT_VAR, PROVIDER_RATE_LIMIT_CANDIDATES
            rate_path = resolve_provider_file(PROVIDER_RATE_LIMIT_VAR, PROVIDER_RATE_LIMIT_CANDIDATES)
            args.provider_rate_limit_file = rate_path
        if needs_rate_limits and (rate_path is None or not rate_path.expanduser().is_file()):
            raise ValueError("configure provider rate limits before starting a new run")
        if needs_rate_limits and rate_path is not None and rate_path.expanduser().is_file():
            configuration_files["rate-limits.json"] = rate_path.expanduser()
        accounts = Path(os.environ.get("CODEACTION_AGENTS_CONFIG", "~/.config/codeaction/agents.json")).expanduser()
        if needs_accounts:
            from codeaction.benchmark.agent_config import load_accounts
            configured_accounts = load_accounts(accounts)
            for name in args.models:
                if name in known and known[name].uses_subscription and name not in local:
                    account = configured_accounts.get(known[name].account)
                    if account is None or not account.token_path.exists():
                        raise ValueError(f"configure the selected subscription account before starting: {known[name].account}")
            configuration_files["agents.json"] = accounts
        snapshot, info = prepare_snapshot(
            root, Path.home() / ".cache/codeaction/executions",
            task_pack=getattr(args, "task_pack", None),
            extension_paths=getattr(args, "extensions", []),
            model_registry=getattr(args, "model_registry", None),
            baseline=Path(baseline) if baseline else None,
            require_match=getattr(args, "require_release_match", False), configuration_files=configuration_files, provider_env_file=provider_file)
        if "rate-limits.json" in configuration_files:
            args.provider_rate_limit_file = snapshot / "config/rate-limits.json"
        # All filesystem values are resolved before handing execution to its copy.
        for name, value in vars(args).items():
            if isinstance(value, Path):
                setattr(args, name, value.expanduser().resolve())
        if command in {"run", "smoke", "mcp-health"}:
            if command == "run":
                args.run_dir = output
            args.source_root = snapshot
        else:
            args.out_dir = output
        if command in {"run", "eval", "smoke", "mcp-health"}:
            args.task_pack = snapshot / "benchmark/tasks"
        args.assets_root = getattr(args, "assets_root", None) or Path(os.environ.get("CODEACTION_ASSETS_ROOT", root / "assets"))
        # One compatibility path internally; submission qualification is a result, not a launch gate.
        if hasattr(args, "profile"):
            args.profile = "dev"
        if hasattr(args, "run_profile"):
            args.run_profile = "dev"
        if baseline:
            images = json.loads(Path(baseline).read_text()).get("images", {})
            from codeaction.release import IMAGE_ARGS
            for role, ref in images.items():
                if hasattr(args, IMAGE_ARGS[role]):
                    setattr(args, IMAGE_ARGS[role], ref)
        # Resolve only images that can be used by this launch; never pull/build as a side effect.
        if command == "replay":
            roles = ["sim_image"]
        elif command in {"run", "smoke", "mcp-health"}:
            agent = execution_driver(args.agent_mode).image_option
            roles = ["sim_image", "gateway_image", agent]
            if args.interface_profile == "vendor-mcp-gateway":
                roles += ["scratch_image", "launcher_image"]
        else:
            # Agent selection is resolved using the frozen registry in the worker. Image
            # tags are pinned here for all selected driver families, not for unrelated seats.
            from codeaction.benchmark.agents import resolve_agents
            from codeaction.benchmark.matrix import roster_with_models
            from codeaction.providers.model_registry import load_registry
            old_registry = os.environ.get("CODEACTION_MODEL_REGISTRY_FILE")
            old_extensions = os.environ.get("CODEACTION_EXTENSIONS_FILE")
            os.environ["CODEACTION_MODEL_REGISTRY_FILE"] = str(snapshot / "config/models.json")
            os.environ["CODEACTION_EXTENSIONS_FILE"] = str(snapshot / "config/extensions.json")
            load_registry.cache_clear()
            try:
                agents = resolve_agents(args.models, roster_with_models(args.models))
            finally:
                for key, value in (("CODEACTION_MODEL_REGISTRY_FILE", old_registry), ("CODEACTION_EXTENSIONS_FILE", old_extensions)):
                    if value is None: os.environ.pop(key, None)
                    else: os.environ[key] = value
                load_registry.cache_clear()
            drivers = {agent.driver for agent in agents}
            roles = ["sim_image", "gateway_image"] + [
                execution_driver(driver).image_option
                for driver in sorted(drivers)]
        from codeaction.environments import validate_environment
        for name in roles:
            ref = getattr(args, name, None) or "codeaction-" + name.removesuffix("_image").replace("_", "-") + ":dev"
            image = _inspect_image(ref)
            validate_environment(image.labels, snapshot)
            setattr(args, name, image.image_id)
        record = {"schema_version": "codeaction-execution.v2", "command": command,
                  "source_root": str(snapshot), "output": str(output), "argv": _argv(parser, args)}
        save_launch_record(record_path, record)
        print(json.dumps({"output": str(output), "baseline": info["baseline"], "changes": info["changes"]}), flush=True)
    env = dict(os.environ)
    for name in ("CODEACTION_RELEASE_MANIFEST", "CODEACTION_MODEL_REGISTRY_EXTRA", "CODEACTION_EXTENSIONS_FILE"):
        env.pop(name, None)
    env.update(CODEACTION_ROOT=str(snapshot), CODEACTION_RUNS_ROOT=str(output), PYTHONPATH=str(snapshot / "src"),
               PYTHONDONTWRITEBYTECODE="1", CODEACTION_RUN_CONTEXT=str(snapshot / "config/context.json"),
               CODEACTION_MODEL_REGISTRY_FILE=str(snapshot / "config/models.json"),
               CODEACTION_EXTENSIONS_FILE=str(snapshot / "config/extensions.json"))
    if (snapshot / "config/agents.json").is_file():
        env["CODEACTION_AGENTS_CONFIG"] = str(snapshot / "config/agents.json")
    argv = record["argv"]
    if getattr(args, "dry_run", False) and "--dry-run" not in argv:
        argv = [*argv, "--dry-run"]
    return subprocess.run([sys.executable, "-m", "codeaction.cli.main", *argv], cwd=snapshot,
                          env=env, check=False).returncode
