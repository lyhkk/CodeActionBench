"""Where a subscription-backed agent's account comes from: local configuration, never the repo.

The released roster is a published fact -- eight agents, one of them a vendor CLI -- and it stays
committed, because a leaderboard whose membership depended on who checked it out would not be one
roster. WHO PAYS for the vendor agent is the opposite kind of fact: an account alias, a lane count
and a path to a token file are local to whoever runs the benchmark, and committing them published
one operator's account name into everyone's checkout.

So the roster declares a subscription alias and this module resolves it, exactly the way a model
entry names a credential alias that is resolved from a credential file at run time. An alias with
no configuration is not an error at import: the agent stays in the roster, stays comparable, and
fails with a message naming the alias and this file only when someone actually tries to run it.

The file never holds a token. It holds the PATH to one, so a token can be rotated, mounted or
replaced without the benchmark ever reading it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "codeaction-agents.v1"
ENV_VAR = "CODEACTION_AGENTS_CONFIG"
DEFAULT_PATH = Path("~/.config/codeaction/agents.json")


class AgentConfigError(ValueError):
    """The configuration file exists but does not say something usable."""


@dataclass(frozen=True)
class WindowPolicy:
    """How a rolling-window subscription is governed. Absent means: no governance.

    A vendor CLI that has no rate limit worth modelling is expressed by omitting `window`
    entirely, not by inventing a threshold that never fires.
    """

    hours: float
    stop_dispatch_above: float
    resume_grace_s: float = 60.0
    # Below this much remaining access-token life, a cell is not dispatched: an episode that
    # outlives its token dies mid-run and spends the window for nothing.
    min_access_lifetime_s: float | None = None
    # How to READ the token's expiry. Omitted means the lifetime check is skipped rather than
    # guessed. `kind: json_field` reads one numeric field from a local JSON file.
    expiry_probe: Mapping[str, Any] | None = None
    # How to renew it. Omitted means never renew: the account is paused until its window resets
    # rather than a command being invented on the operator's behalf.
    refresh_command: tuple[str, ...] | None = None
    refresh_max_failures: int = 3
    refresh_retry_s: float = 60.0

    def __post_init__(self) -> None:
        if self.hours <= 0:
            raise AgentConfigError("window.hours must be positive")
        if not 0.0 < self.stop_dispatch_above <= 1.0:
            raise AgentConfigError(
                f"window.stop_dispatch_above must be a fraction in (0, 1], got "
                f"{self.stop_dispatch_above!r}")
        if self.resume_grace_s < 0:
            raise AgentConfigError("window.resume_grace_s must not be negative")
        if self.min_access_lifetime_s is not None and self.min_access_lifetime_s < 0:
            raise AgentConfigError("window.min_access_lifetime_s must not be negative")
        if self.refresh_max_failures < 1:
            raise AgentConfigError("window.refresh.max_failures must be positive")


@dataclass(frozen=True)
class ConfiguredAccount:
    """One subscription the operator has declared locally."""

    alias: str
    plan: str
    max_concurrency: int
    token_file: str
    window: WindowPolicy | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise AgentConfigError(f"{self.alias}: max_concurrency must be positive")
        if not self.token_file:
            raise AgentConfigError(f"{self.alias}: token_file must name a path")

    @property
    def token_path(self) -> Path:
        return Path(self.token_file).expanduser()


def config_path(path: str | os.PathLike | None = None) -> Path:
    """Where the configuration is read from: argument, then env var, then the default."""
    if path is not None:
        return Path(path).expanduser()
    from_env = os.environ.get(ENV_VAR)
    return Path(from_env).expanduser() if from_env else DEFAULT_PATH.expanduser()


def _window(raw: Any, alias: str) -> WindowPolicy | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise AgentConfigError(f"{alias}: window must be an object or null")
    refresh = raw.get("refresh")
    command: tuple[str, ...] | None = None
    max_failures, retry_s = 3, 60.0
    if refresh is not None:
        if not isinstance(refresh, Mapping):
            raise AgentConfigError(f"{alias}: window.refresh must be an object or null")
        argv = refresh.get("command")
        if argv is not None:
            if not isinstance(argv, (list, tuple)) or not argv \
                    or not all(isinstance(item, str) for item in argv):
                raise AgentConfigError(
                    f"{alias}: window.refresh.command must be a non-empty list of strings")
            command = tuple(str(item) for item in argv)
        max_failures = int(refresh.get("max_failures", 3))
        retry_s = float(refresh.get("retry_s", 60.0))
    probe = raw.get("expiry_probe")
    if probe is not None and not isinstance(probe, Mapping):
        raise AgentConfigError(f"{alias}: window.expiry_probe must be an object or null")
    lifetime = raw.get("min_access_lifetime_s")
    return WindowPolicy(
        hours=float(raw.get("hours", 5.0)),
        stop_dispatch_above=float(raw.get("stop_dispatch_above", 0.90)),
        resume_grace_s=float(raw.get("resume_grace_s", 60.0)),
        min_access_lifetime_s=None if lifetime is None else float(lifetime),
        expiry_probe=dict(probe) if probe else None,
        refresh_command=command,
        refresh_max_failures=max_failures,
        refresh_retry_s=retry_s,
    )


def load_accounts(path: str | os.PathLike | None = None) -> dict[str, ConfiguredAccount]:
    """Every locally declared subscription account, keyed by alias.

    A missing file is not an error -- it is the normal state of a checkout that only runs
    reference agents -- and returns an empty mapping.
    """
    resolved = config_path(path)
    if not resolved.is_file():
        return {}
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentConfigError(f"{resolved}: cannot be read as JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise AgentConfigError(f"{resolved}: top level must be an object")
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise AgentConfigError(
            f"{resolved}: schema_version must be {SCHEMA_VERSION!r}, got {version!r}")
    raw_accounts = document.get("accounts") or {}
    if not isinstance(raw_accounts, Mapping):
        raise AgentConfigError(f"{resolved}: accounts must be an object")
    accounts: dict[str, ConfiguredAccount] = {}
    for alias, entry in raw_accounts.items():
        if not isinstance(entry, Mapping):
            raise AgentConfigError(f"{resolved}: account {alias!r} must be an object")
        missing = [key for key in ("plan", "max_concurrency", "token_file") if key not in entry]
        if missing:
            raise AgentConfigError(f"{resolved}: account {alias!r} is missing {missing}")
        accounts[str(alias)] = ConfiguredAccount(
            alias=str(alias),
            plan=str(entry["plan"]),
            max_concurrency=int(entry["max_concurrency"]),
            token_file=str(entry["token_file"]),
            window=_window(entry.get("window"), str(alias)),
        )
    return accounts


def account_for(alias: str, path: str | os.PathLike | None = None) -> ConfiguredAccount | None:
    return load_accounts(path).get(alias)


def unconfigured_message(alias: str, path: str | os.PathLike | None = None) -> str:
    """What to tell someone whose selection needs an account they have not declared."""
    return (f"subscription alias {alias!r} is not configured: write it into "
            f"{config_path(path)} (see configs/agents.example.json). "
            f"The file names a token PATH; it never holds a token.")


__all__ = [
    "AgentConfigError", "ConfiguredAccount", "DEFAULT_PATH", "ENV_VAR", "SCHEMA_VERSION",
    "WindowPolicy", "account_for", "config_path", "load_accounts", "unconfigured_message",
]
