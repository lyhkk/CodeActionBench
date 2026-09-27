"""Portable credential and rate-policy locations shared by public entry points."""
import os
from pathlib import Path

PROVIDER_ENV_VAR = "CODEACTION_PROVIDER_ENV_FILE"
PROVIDER_RATE_LIMIT_VAR = "CODEACTION_PROVIDER_RATE_LIMIT_FILE"
PROVIDER_ENV_CANDIDATES = (
    Path.home() / ".config/codeaction/provider.env",
    Path.home() / ".codeaction_provider.env",
)
PROVIDER_RATE_LIMIT_CANDIDATES = (
    Path.home() / ".config/codeaction/rate-limits.json",
    Path.home() / ".config/codeaction/provider_rate_limits.json",
    Path.home() / ".codeaction/provider_rate_limits.json",
)


def resolve_provider_file(env_var: str, candidates) -> Path:
    """The override, else the first candidate that exists, else the canonical path.

    Returning the canonical path when nothing exists keeps the failure message pointing at where
    the file BELONGS rather than at the last thing tried.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]

