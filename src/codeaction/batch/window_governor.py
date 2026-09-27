"""Governing a rolling-window subscription: read what the window says, decide what to stop.

A provider key answers a request that exceeds its rate with an error, so the adaptive throttle in
`providers/provider_runtime` is enough for it. A subscription is different: the account carries a
rolling multi-hour window shared by every session on it, the vendor CLI reports how much of that
window is spent, and spending the last of it mid-episode wastes the episode AND the window. So a
subscription needs a governor that reads the report and stops dispatching BEFORE the wall.

Everything here is a pure decision over recorded numbers -- what the stream said, what the clock
says, what the policy declares -- so the whole governor is testable without a subscription, a
container or a GPU. Applying a decision is the caller's job.

Absent policy means absent governance. An agent whose plan has no window worth modelling declares
no `window`, and every function here returns "nothing to do" rather than inventing a threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# What the vendor CLI calls the window this governor is about. Other rate-limit kinds in the same
# stream (per-minute, per-account-burst) are not this one and are ignored rather than conflated.
FIVE_HOUR = "five_hour"


@dataclass(frozen=True)
class WindowReading:
    """The last thing an episode's stream said about the account's window."""

    utilization: float | None
    reset_at: float | None
    rejected: bool = False

    @property
    def known(self) -> bool:
        return self.utilization is not None


@dataclass(frozen=True)
class Decision:
    """What the caller should do. `action` is the only thing a caller needs to branch on."""

    action: str                  # "proceed" | "pause_credential" | "refresh_token" | "attention"
    reason: str = ""
    resume_at: float | None = None
    utilization: float | None = None

    @property
    def blocks_dispatch(self) -> bool:
        return self.action != "proceed"


PROCEED = Decision("proceed")


def read_window(stream_path: str | Path) -> WindowReading:
    """The window state an episode's vendor stream reports, or an unknown reading.

    The stream is append-only and a window fact is restated on every event that carries one, so
    the LAST such event is the current truth. A stream that never mentions the window -- a run
    that failed before its first call, or a CLI version that does not report it -- yields an
    unknown reading, which no policy treats as zero.
    """
    path = Path(stream_path)
    utilization: float | None = None
    reset_at: float | None = None
    rejected = False
    if not path.is_file():
        return WindowReading(None, None, False)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "rate_limit" not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping) or row.get("type") != "rate_limit_event":
                continue
            info = row.get("rate_limit_info")
            if not isinstance(info, Mapping) or info.get("rateLimitType") != FIVE_HOUR:
                continue
            value = info.get("utilization")
            if isinstance(value, (int, float)):
                # Reported either as a percentage or as a fraction; both mean the same thing and
                # a fraction above 1.0 would be nonsense, so the scale is read from the value.
                utilization = float(value) / 100.0 if float(value) > 1.0 else float(value)
            reset = info.get("resetsAt", info.get("reset_at"))
            if isinstance(reset, (int, float)):
                reset_at = float(reset) / 1000.0 if float(reset) > 1e11 else float(reset)
            if info.get("status") == "rejected" or row.get("rejected") is True:
                rejected = True
    return WindowReading(utilization, reset_at, rejected)


def after_episode(policy, reading: WindowReading, *, now: float) -> Decision:
    """Whether the account may keep dispatching after an episode reported this reading."""
    if policy is None or not reading.known:
        return PROCEED
    if reading.utilization < policy.stop_dispatch_above:
        return PROCEED
    # A reset time is what makes the pause self-clearing. Without one the window's own length is
    # the only bound available, and using it is more honest than pausing forever.
    resume_at = ((reading.reset_at + policy.resume_grace_s) if reading.reset_at is not None
                 else now + policy.hours * 3600.0)
    return Decision(
        "pause_credential",
        reason=(f"window utilization {reading.utilization:.2f} is at or above "
                f"{policy.stop_dispatch_above:.2f}; not dispatching until the window resets"),
        resume_at=resume_at,
        utilization=reading.utilization,
    )


def token_expiry(policy) -> float | None:
    """When the access token stops working, in epoch seconds, or None if unknowable.

    Only the token's EXPIRY is read. The token itself is never opened by this code path.
    """
    if policy is None or not policy.expiry_probe:
        return None
    probe = policy.expiry_probe
    if probe.get("kind") != "json_field":
        raise ValueError(f"unsupported expiry_probe kind {probe.get('kind')!r}")
    path = Path(str(probe.get("path", ""))).expanduser()
    field = str(probe.get("field", "expiresAt"))
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    found = _find_field(document, field)
    if not isinstance(found, (int, float)):
        return None
    units = str(probe.get("units", "s"))
    return float(found) / 1000.0 if units == "ms" else float(found)


def _find_field(document: Any, field: str) -> Any:
    if isinstance(document, Mapping):
        for key, value in document.items():
            if key == field:
                return value
            found = _find_field(value, field)
            if found is not None:
                return found
    elif isinstance(document, (list, tuple)):
        for value in document:
            found = _find_field(value, field)
            if found is not None:
                return found
    return None


def before_dispatch(policy, *, now: float, expires_at: float | None,
                    consecutive_refresh_failures: int = 0) -> Decision:
    """Whether a cell may start on this account right now.

    A token that will not outlive an episode is not a reason to start one: the episode dies
    mid-run and the window pays for nothing. With a refresh command the answer is "renew first";
    without one it is a stop that names what a person has to do.
    """
    if policy is None or policy.min_access_lifetime_s is None or expires_at is None:
        return PROCEED
    remaining = expires_at - now
    if remaining >= policy.min_access_lifetime_s:
        return PROCEED
    if policy.refresh_command is None:
        return Decision(
            "attention",
            reason=(f"access token has {max(remaining, 0):.0f}s left, below the "
                    f"{policy.min_access_lifetime_s:.0f}s an episode needs, and no refresh "
                    f"command is configured"),
        )
    if consecutive_refresh_failures >= policy.refresh_max_failures:
        return Decision(
            "attention",
            reason=(f"token refresh failed {consecutive_refresh_failures} times in a row; "
                    f"the account is held until a person looks at it"),
        )
    return Decision(
        "refresh_token",
        reason=f"access token has {max(remaining, 0):.0f}s left; renewing before dispatch",
        resume_at=now + policy.refresh_retry_s,
    )


def window_pause_is_due(resume_at: float | None, *, now: float) -> bool:
    """Whether a window pause opened earlier has served its time."""
    return resume_at is not None and now >= resume_at


def summarize(readings: Iterable[WindowReading]) -> dict[str, Any]:
    """What a report shows about an account's window across a batch."""
    known = [r for r in readings if r.known]
    return {
        "episodes_reporting_window": len(known),
        "max_utilization": max((r.utilization for r in known), default=None),
        "last_utilization": known[-1].utilization if known else None,
        "any_rejected": any(r.rejected for r in known),
    }


__all__ = [
    "Decision", "FIVE_HOUR", "PROCEED", "WindowReading", "after_episode", "before_dispatch",
    "read_window", "summarize", "token_expiry", "window_pause_is_due",
]
