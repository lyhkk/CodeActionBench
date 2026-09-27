#!/usr/bin/env python3
"""Isolated container entry point for the single Python ``codeaction-reference`` loop."""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path


REFERENCE_EVENT_MARK = "[reference-agent-event] "
REFERENCE_SUMMARY_MARK = "[reference-agent-summary] "


def _positive_int(name: str) -> int:
    try:
        value = int(os.environ[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str) -> float:
    try:
        value = float(os.environ[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _load_provider_env(path: str, *, alias: str | None = None) -> None:
    """Load the alias-keyed credential file into the environment.

    Parsing and the allowlist live in ``codeaction.providers.model_registry`` so the controller-side check and
    the in-container load can never drift apart.  Only ``<ALIAS>_KEY``/``<ALIAS>_BASE_URL`` are
    accepted, which is what keeps a secret file from carrying model identity.
    """
    from codeaction.providers.model_registry import read_credential_file
    provider_path = Path(path)
    if not provider_path.is_file():
        raise ValueError("provider environment file is unavailable")
    values = read_credential_file(provider_path)
    frozen_registry = os.environ.get("CODEACTION_MODEL_REGISTRY_FILE")
    endpoint_file = Path(frozen_registry).with_name("endpoints.json") if frozen_registry else None
    if endpoint_file is not None and endpoint_file.is_file():
        values = {key: value for key, value in values.items() if not key.endswith("_BASE_URL")}
        values.update(json.loads(endpoint_file.read_text()))
    if alias is not None:
        prefix = alias.replace("-", "_").upper()
        values = {key: value for key, value in values.items() if key in {prefix + "_KEY", prefix + "_BASE_URL"}}
        if prefix + "_KEY" not in values:
            raise ValueError(f"credential file does not contain the selected alias {alias!r}")
    os.environ.update(values)


def main() -> int:
    config = Path("/run/codeaction/code/config/models.json")
    if config.is_file():
        os.environ["CODEACTION_MODEL_REGISTRY_FILE"] = str(config)
        os.environ["CODEACTION_EXTENSIONS_FILE"] = str(config.with_name("extensions.json"))
    from codeaction.providers.model_adapter import ScriptedModel, build_provider
    from codeaction.agents.reference.reference_agent import run_episode, scaffold_card
    from codeaction.agents.reference.reference_mcp import RemoteEpisodeRuntime
    from codeaction.interface.tool_surface import (INTERFACE_REFERENCE, resolve_interface_profile,
                                      surface_identity)

    mode = os.environ.get("CODEACTION_REFERENCE_MODEL_MODE", "scripted")
    if mode not in ("scripted", "provider", "local"):
        raise ValueError("CODEACTION_REFERENCE_MODEL_MODE must be scripted or provider")
    task_text = os.environ.get("CODEACTION_TASK_TEXT", "").strip()
    if not task_text:
        raise ValueError("CODEACTION_TASK_TEXT is required")
    max_calls = _positive_int("CODEACTION_MAX_TOOL_CALLS")
    wall_budget = _positive_float("CODEACTION_WALL_BUDGET_S")
    physical_time_budget = _positive_float("CODEACTION_PHYSICAL_TIME_BUDGET_S")
    run_code_limit = _positive_int("CODEACTION_RUN_CODE_MAX_INTERNAL_CALLS")
    try:
        harness_parameters = json.loads(base64.b64decode(
            os.environ["CODEACTION_HARNESS_PARAMETERS_B64"], validate=True).decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CODEACTION_HARNESS_PARAMETERS_B64 must encode a JSON object") from exc
    if not isinstance(harness_parameters, dict):
        raise ValueError("CODEACTION_HARNESS_PARAMETERS_B64 must encode a JSON object")
    # The container serves both reference profiles; which one is a declared run parameter, and
    # the delivered-hash check below is what makes the declaration binding.
    profile_id = os.environ.get("CODEACTION_INTERFACE_PROFILE", INTERFACE_REFERENCE)
    resolve_interface_profile(profile_id)
    expected_surface = surface_identity(profile_id, hybrid=True)
    expected_delivered = os.environ.get("CODEACTION_EXPECTED_DELIVERED_SHA256", "")
    if expected_surface["delivered_sha256"] != expected_delivered:
        raise ValueError("reference container expected tool-surface hash mismatch")

    try:
        provider_rate_limit_policy = json.loads(base64.b64decode(
            os.environ["CODEACTION_PROVIDER_RATE_LIMIT_POLICY_B64"],
            validate=True).decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "CODEACTION_PROVIDER_RATE_LIMIT_POLICY_B64 must encode a JSON object") from exc
    if not isinstance(provider_rate_limit_policy, dict):
        raise ValueError("CODEACTION_PROVIDER_RATE_LIMIT_POLICY_B64 must encode a JSON object")

    if mode == "local":
        from codeaction.extensions import declarations
        credential = declarations("agent")[os.environ["CODEACTION_LOCAL_AGENT"]].get("credential")
        if credential:
            path = os.environ.get("CODEACTION_PROVIDER_ENV_FILE", "/run/secrets/codeaction/provider.env")
            # Explicit aliases only. An unrelated provider key never enters the process environment.
            _load_provider_env(path, alias=credential)
    if mode in {"scripted", "local"}:
        # The offline lifecycle script must only call tools this profile actually delivers:
        # code-first hands over the primitives through run_code, so calling get_robot_state
        # directly there is a tool the model was never given.
        delivered = set(expected_surface["ordered_names"])
        probe = ([("capture_head", {}),
                  ("get_robot_state", {"arms": ["left", "right"]})]
                 if "get_robot_state" in delivered else
                 [("run_code", {"code": "capture_head()\n"
                                        "result = get_robot_state(arms=['left', 'right'])\n"})])
        model = ScriptedModel([
            probe,
            ("done", {
                "report": "offline reference-container lifecycle completed",
                "success_claim": False,
            }),
        ])
    else:
        _load_provider_env(os.environ.get(
            "CODEACTION_PROVIDER_ENV_FILE", "/run/secrets/codeaction/provider.env"))
        model = build_provider(
            model=os.environ.get("CODEACTION_MODEL"),
            rate_limit_policy=provider_rate_limit_policy,
            rate_limit_state_dir=os.environ.get("CODEACTION_PROVIDER_STATE_DIR"),
        )

    scaffold = scaffold_card(
        model.capabilities(), request_profile=model.request_profile(),
        transport_profile=(model.transport_profile()
                           if callable(getattr(model, "transport_profile", None)) else None),
        rate_limit_policy=(model.rate_limit_policy()
                           if callable(getattr(model, "rate_limit_policy", None)) else None),
        implementation=os.environ.get("CODEACTION_LOCAL_AGENT") if mode == "local" else None)
    expected_scaffold = os.environ.get("CODEACTION_EXPECTED_SCAFFOLD_SHA256", "")
    if scaffold["config_sha256"] != expected_scaffold:
        raise ValueError("reference container scaffold config hash mismatch")

    work = Path("/tmp/reference-attempt")
    runtime = RemoteEpisodeRuntime(
        task_text,
        expected_surface=expected_surface,
        max_tool_calls=max_calls,
        wall_budget_s=wall_budget,
        physical_time_budget_s=physical_time_budget,
        image_dir=work / "images",
    )

    def emit(record):
        print(
            REFERENCE_EVENT_MARK
            + json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    try:
        if mode == "local":
            from codeaction.agents.reference.local_agent import run_local_agent
            work.mkdir(parents=True, exist_ok=True)
            stats = run_local_agent(os.environ["CODEACTION_LOCAL_AGENT"], runtime, work, scaffold, emit,
                                    wall_budget_s=wall_budget)
        else:
            stats = run_episode(
                None,
                task_text,
                model,
                work,
                max_steps=max_calls,
                wall_budget_s=wall_budget,
                physical_time_budget_s=physical_time_budget,
                expected_tool_surface=expected_surface,
                runtime_override=runtime,
                tool_definitions=runtime.tools,
                hybrid=True,
                run_code_max_internal_calls=run_code_limit,
                transcript_sink=emit,
                filesystem_policy="structurally_denied",
                expected_scaffold_config_sha256=expected_scaffold,
                harness_parameters=harness_parameters,
            )
        print(
            REFERENCE_SUMMARY_MARK
            + json.dumps({
                "schema_version": "1.0",
                "status": stats["status"],
                "budget_used": stats["budget_used"],
                "total_calls": stats["total_calls"],
                "tool_calls_used": stats["tool_calls_used"],
                "tool_call_budget": stats["tool_call_budget"],
                "total_tool_dispatches": stats["total_tool_dispatches"],
                "model_turns": stats["model_turns"],
                "scaffold_config_sha256": stats["scaffold"]["config_sha256"],
                "delivered_tool_sha256": expected_surface["delivered_sha256"],
            }, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    finally:
        try:
            close_model = getattr(model, "close", None)
            if callable(close_model):
                close_model()
        finally:
            runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
