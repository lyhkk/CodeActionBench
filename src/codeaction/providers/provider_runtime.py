"""Provider request pacing, retry policy, and safe HTTP telemetry.

The reference-scaffold matrix launches one reference-agent container per benchmark cell.  An in-memory rate
limiter would therefore serialize nothing across GPUs, even when several models share one account.
This module keeps the quota window in a small host-mounted directory and protects it with
``fcntl.flock``.  The state contains only credential aliases, timestamps, token estimates, and
bounded response-header values; API keys and response bodies never enter it.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Mapping, Optional
import uuid


RATE_LIMIT_CONFIG_SCHEMA_VERSION = "1.1"
RATE_LIMIT_POLICY_ID = "shared-smoothed-window-v3"
TRANSPORT_POLICY_ID = "provider-transport-v2"

DEFAULT_TRANSPORT_PROFILE = {
    "id": TRANSPORT_POLICY_ID,
    "connect_timeout_s": 20.0,
    "first_response_timeout_s": 90.0,
    "stream_idle_timeout_s": 90.0,
    "response_hard_timeout_s": 600.0,
    "streaming": False,
    "attempts": 3,
    "retry": {
        "provider_rate_limited": {"delays_s": [60.0, 120.0], "jitter_ratio": 0.0},
        "provider_timeout": {"delays_s": [2.0, 4.0], "jitter_ratio": 0.0},
        "provider_server_error": {"delays_s": [2.0, 4.0], "jitter_ratio": 0.0},
    },
    "server_error_cooldown": {
        "after_consecutive": 0,
        "min_s": 0.0,
        "max_s": 0.0,
    },
}

_FAILURE_CODES = frozenset(DEFAULT_TRANSPORT_PROFILE["retry"])
_SAFE_HEADER = re.compile(
    r"(?i)^(retry-after|"
    r"x-ratelimit-(limit|remaining|reset)-(requests|tokens)|"
    r"request-id|x-[a-z0-9-]*request-id)$")
_DURATION_PART = re.compile(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)")


class ProviderRuntimeConfigError(ValueError):
    """A non-secret provider-control file or registry transport declaration is invalid."""


class ProviderPhaseTimeout(TimeoutError):
    """Safe timeout carrying the exact provider phase without response-body text."""

    def __init__(self, phase: str, limit_s: float):
        self.timeout_phase = str(phase)
        self.limit_s = float(limit_s)
        super().__init__(f"provider {self.timeout_phase} timeout after {self.limit_s:g}s")


def _positive_number(value: Any, what: str) -> float:
    if isinstance(value, bool):
        raise ProviderRuntimeConfigError(f"{what} must be positive")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderRuntimeConfigError(f"{what} must be positive") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ProviderRuntimeConfigError(f"{what} must be positive")
    return parsed


def normalize_transport_profile(raw: Optional[Mapping[str, Any]] = None) -> dict:
    """Overlay and validate a model's transport policy on the benchmark defaults."""
    value = deepcopy(DEFAULT_TRANSPORT_PROFILE)
    if raw is None:
        return value
    if not isinstance(raw, Mapping):
        raise ProviderRuntimeConfigError("transport must be an object")
    unknown = set(raw) - {
        "id", "timeout_s", "connect_timeout_s", "first_response_timeout_s",
        "stream_idle_timeout_s", "response_hard_timeout_s", "streaming",
        "attempts", "retry", "server_error_cooldown",
    }
    if unknown:
        raise ProviderRuntimeConfigError(f"transport has unknown fields: {sorted(unknown)}")
    if "id" in raw:
        value["id"] = str(raw["id"] or "").strip()
        if not value["id"]:
            raise ProviderRuntimeConfigError("transport.id must be non-empty")
    if "timeout_s" in raw:
        legacy = _positive_number(raw["timeout_s"], "transport.timeout_s")
        value.update({
            "first_response_timeout_s": legacy,
            "stream_idle_timeout_s": legacy,
            "response_hard_timeout_s": legacy,
        })
    for field in (
            "connect_timeout_s", "first_response_timeout_s", "stream_idle_timeout_s",
            "response_hard_timeout_s"):
        if field in raw:
            value[field] = _positive_number(raw[field], f"transport.{field}")
    if "streaming" in raw:
        if not isinstance(raw["streaming"], bool):
            raise ProviderRuntimeConfigError("transport.streaming must be boolean")
        value["streaming"] = raw["streaming"]
    if "attempts" in raw:
        attempts = raw["attempts"]
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
            raise ProviderRuntimeConfigError("transport.attempts must be a positive integer")
        value["attempts"] = attempts
    if "retry" in raw:
        if not isinstance(raw["retry"], Mapping):
            raise ProviderRuntimeConfigError("transport.retry must be an object")
        for code, declaration in raw["retry"].items():
            if code not in _FAILURE_CODES:
                raise ProviderRuntimeConfigError(
                    f"transport.retry has unsupported failure code {code!r}")
            if not isinstance(declaration, Mapping):
                raise ProviderRuntimeConfigError(f"transport.retry.{code} must be an object")
            unknown_retry = set(declaration) - {"delays_s", "jitter_ratio", "max_attempts"}
            if unknown_retry:
                raise ProviderRuntimeConfigError(
                    f"transport.retry.{code} has unknown fields: {sorted(unknown_retry)}")
            delays = declaration.get("delays_s")
            if not isinstance(delays, list):
                raise ProviderRuntimeConfigError(
                    f"transport.retry.{code}.delays_s must be a list")
            parsed_delays = []
            for index, delay in enumerate(delays):
                if isinstance(delay, bool):
                    raise ProviderRuntimeConfigError(
                        f"transport.retry.{code}.delays_s[{index}] must be non-negative")
                try:
                    parsed = float(delay)
                except (TypeError, ValueError) as exc:
                    raise ProviderRuntimeConfigError(
                        f"transport.retry.{code}.delays_s[{index}] must be non-negative") from exc
                if not math.isfinite(parsed) or parsed < 0:
                    raise ProviderRuntimeConfigError(
                        f"transport.retry.{code}.delays_s[{index}] must be non-negative")
                parsed_delays.append(parsed)
            ratio = declaration.get("jitter_ratio", 0.0)
            try:
                ratio = float(ratio)
            except (TypeError, ValueError) as exc:
                raise ProviderRuntimeConfigError(
                    f"transport.retry.{code}.jitter_ratio must be in [0, 1]") from exc
            if not math.isfinite(ratio) or ratio < 0 or ratio > 1:
                raise ProviderRuntimeConfigError(
                    f"transport.retry.{code}.jitter_ratio must be in [0, 1]")
            value["retry"][code] = {
                "delays_s": parsed_delays,
                "jitter_ratio": ratio,
            }
            if "max_attempts" in declaration:
                maximum = declaration["max_attempts"]
                if not isinstance(maximum, int) or isinstance(maximum, bool) \
                        or maximum < 1 or maximum > value["attempts"]:
                    raise ProviderRuntimeConfigError(
                        f"transport.retry.{code}.max_attempts must be in "
                        f"[1, transport.attempts]")
                value["retry"][code]["max_attempts"] = maximum
    if "server_error_cooldown" in raw:
        declaration = raw["server_error_cooldown"]
        if not isinstance(declaration, Mapping):
            raise ProviderRuntimeConfigError("transport.server_error_cooldown must be an object")
        unknown_cooldown = set(declaration) - {"after_consecutive", "min_s", "max_s"}
        if unknown_cooldown:
            raise ProviderRuntimeConfigError(
                "transport.server_error_cooldown has unknown fields: "
                f"{sorted(unknown_cooldown)}")
        after = declaration.get("after_consecutive", 0)
        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            raise ProviderRuntimeConfigError(
                "transport.server_error_cooldown.after_consecutive must be a non-negative integer")
        minimum = float(declaration.get("min_s", 0.0))
        maximum = float(declaration.get("max_s", 0.0))
        if not all(math.isfinite(item) and item >= 0 for item in (minimum, maximum)) \
                or maximum < minimum:
            raise ProviderRuntimeConfigError(
                "transport.server_error_cooldown requires 0 <= min_s <= max_s")
        if after and maximum <= 0:
            raise ProviderRuntimeConfigError(
                "transport.server_error_cooldown needs a positive range when enabled")
        value["server_error_cooldown"] = {
            "after_consecutive": after,
            "min_s": minimum,
            "max_s": maximum,
        }
    return value


def provider_sdk_timeout(profile: Mapping[str, Any], override: Any = None) -> Any:
    """Return an SDK timeout with bounded connect/read/write/pool phases.

    HTTPX exposes read-idle rather than separate first-byte and post-first-byte settings. Using
    their minimum is fail-safe; streaming adapters additionally label the observed phase and
    enforce the whole-response deadline while consuming chunks.
    """
    if override is not None:
        return _positive_number(override, "provider timeout override")
    normalized = normalize_transport_profile(profile)
    read_timeout = min(
        normalized["first_response_timeout_s"],
        normalized["stream_idle_timeout_s"],
        normalized["response_hard_timeout_s"],
    )
    try:
        import httpx
        connect_timeout = min(
            normalized["connect_timeout_s"],
            normalized["response_hard_timeout_s"],
        )
        return httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=normalized["response_hard_timeout_s"],
            pool=connect_timeout,
        )
    except ImportError:
        return read_timeout


def provider_attempt_limit(profile: Mapping[str, Any], failure_code: str) -> int:
    """Return the bounded request-attempt count for one structured provider failure.

    ``transport.attempts`` remains the adapter-wide allocation.  A failure declaration may only
    lower that number, which lets an expensive ambiguous read timeout stop immediately while a
    response-free 503 retains its ordinary bounded recovery path.
    """
    normalized = normalize_transport_profile(profile)
    declaration = normalized["retry"].get(str(failure_code)) or {}
    return int(declaration.get("max_attempts", normalized["attempts"]))


def load_rate_limit_config(path: str | Path) -> dict:
    """Load the non-secret, account-specific quota ceilings used by the matrix."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderRuntimeConfigError(f"provider rate-limit config is unreadable: {exc}") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") not in (
            "1.0", RATE_LIMIT_CONFIG_SCHEMA_VERSION):
        raise ProviderRuntimeConfigError(
            "provider rate-limit config schema_version must be '1.0' or "
            f"{RATE_LIMIT_CONFIG_SCHEMA_VERSION!r}")
    groups = raw.get("quota_groups")
    if not isinstance(groups, Mapping):
        raise ProviderRuntimeConfigError("provider rate-limit config quota_groups must be an object")
    normalized_groups = {}
    for name, value in groups.items():
        if not isinstance(value, Mapping):
            raise ProviderRuntimeConfigError(
                f"provider rate-limit config quota_groups.{name} must be an object")
        normalized_groups[str(name)] = dict(value)
    return {"schema_version": RATE_LIMIT_CONFIG_SCHEMA_VERSION,
            "quota_groups": normalized_groups}


def normalize_rate_limit_policy(
    raw: Optional[Mapping[str, Any]], *, quota_group: Optional[str],
    model: Optional[str] = None,
) -> dict:
    """Validate one model's exact rolling windows and deterministic AIMD defaults.

    ``rpm``/``tpm`` remain accepted as legacy 60-second declarations. New account snapshots use
    explicit ``limit`` + ``period_s`` windows so a 1-second RPS ceiling, a 6-second Qwen ceiling,
    and a daily request ceiling cannot be made indistinguishable by unit conversion.
    """
    declaration = dict(raw or {})
    unknown = set(declaration) - {
        "id", "quota_group", "model",
        "rpm", "tpm", "request_windows", "token_windows",
        "initial_utilization", "max_utilization", "increase_step",
        "successes_before_increase", "decrease_factor", "rate_limit_default_cooldown_s",
    }
    if unknown:
        raise ProviderRuntimeConfigError(
            f"rate-limit policy has unknown fields: {sorted(unknown)}")

    def optional_positive_int(name: str) -> Optional[int]:
        value = declaration.get(name)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ProviderRuntimeConfigError(f"rate-limit policy {name} must be a positive integer")
        return value

    def ratio(name: str, default: float) -> float:
        value = float(declaration.get(name, default))
        if not math.isfinite(value) or value <= 0 or value > 1:
            raise ProviderRuntimeConfigError(f"rate-limit policy {name} must be in (0, 1]")
        return value

    successes = declaration.get("successes_before_increase", 100)
    if not isinstance(successes, int) or isinstance(successes, bool) or successes < 1:
        raise ProviderRuntimeConfigError(
            "rate-limit policy successes_before_increase must be a positive integer")
    initial = ratio("initial_utilization", 0.60)
    maximum = ratio("max_utilization", 0.95)
    if maximum < initial:
        raise ProviderRuntimeConfigError(
            "rate-limit policy max_utilization must be >= initial_utilization")
    default_cooldown = float(declaration.get("rate_limit_default_cooldown_s", 60.0))
    if not math.isfinite(default_cooldown) or default_cooldown < 0:
        raise ProviderRuntimeConfigError(
            "rate-limit policy rate_limit_default_cooldown_s must be non-negative")
    legacy_rpm = optional_positive_int("rpm")
    legacy_tpm = optional_positive_int("tpm")

    def windows(name: str, *, token: bool) -> list[dict]:
        raw_windows = declaration.get(name, ())
        if not isinstance(raw_windows, (list, tuple)):
            raise ProviderRuntimeConfigError(f"rate-limit policy {name} must be a list")
        normalized = []
        allowed_fields = {
            "total_tokens", "input_tokens", "input_tokens_excluding_cache_reads",
            "output_tokens",
        }
        for index, item in enumerate(raw_windows):
            if not isinstance(item, Mapping):
                raise ProviderRuntimeConfigError(
                    f"rate-limit policy {name}[{index}] must be an object")
            allowed = {"limit", "period_s", "scope"}
            if token:
                allowed.add("field")
            extra = set(item) - allowed
            if extra:
                raise ProviderRuntimeConfigError(
                    f"rate-limit policy {name}[{index}] has unknown fields: {sorted(extra)}")
            limit = item.get("limit")
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                raise ProviderRuntimeConfigError(
                    f"rate-limit policy {name}[{index}].limit must be a positive integer")
            period = _positive_number(
                item.get("period_s"), f"rate-limit policy {name}[{index}].period_s")
            scope = str(item.get("scope") or "model")
            if scope not in ("model", "credential"):
                raise ProviderRuntimeConfigError(
                    f"rate-limit policy {name}[{index}].scope must be model or credential")
            value = {"limit": int(limit), "period_s": float(period), "scope": scope}
            if token:
                field = str(item.get("field") or "")
                if field not in allowed_fields:
                    raise ProviderRuntimeConfigError(
                        f"rate-limit policy {name}[{index}].field must be one of "
                        f"{sorted(allowed_fields)}")
                value["field"] = field
            normalized.append(value)
        return normalized

    request_windows = windows("request_windows", token=False)
    token_windows = windows("token_windows", token=True)
    if legacy_rpm is not None:
        request_windows.append({
            "limit": legacy_rpm, "period_s": 60.0, "scope": "credential"})
    if legacy_tpm is not None:
        token_windows.append({
            "limit": legacy_tpm,
            "period_s": 60.0,
            "scope": "credential",
            "field": "total_tokens",
        })
    request_windows.sort(key=lambda item: (item["period_s"], item["limit"]))
    token_windows.sort(key=lambda item: (
        item["field"], item["period_s"], item["limit"]))

    return {
        "id": RATE_LIMIT_POLICY_ID,
        "quota_group": str(quota_group) if quota_group is not None else None,
        "model": str(model if model is not None else declaration.get("model"))
        if model is not None or declaration.get("model") is not None else None,
        "rpm": legacy_rpm,
        "tpm": legacy_tpm,
        "request_windows": request_windows,
        "token_windows": token_windows,
        "initial_utilization": initial,
        "max_utilization": maximum,
        "increase_step": ratio("increase_step", 0.05),
        "successes_before_increase": successes,
        "decrease_factor": ratio("decrease_factor", 0.70),
        "rate_limit_default_cooldown_s": default_cooldown,
    }


def rate_limit_policy_for(
    config: Optional[Mapping[str, Any]], quota_group: Optional[str], model: Optional[str] = None,
) -> dict:
    groups = (config or {}).get("quota_groups") if isinstance(config, Mapping) else None
    declaration = groups.get(quota_group) if isinstance(groups, Mapping) and quota_group else None
    if isinstance(declaration, Mapping) and "models" in declaration:
        models = declaration.get("models")
        if not isinstance(models, Mapping):
            raise ProviderRuntimeConfigError(
                f"rate-limit quota group {quota_group!r}.models must be an object")
        model_declaration = models.get(model) if model is not None else None
        if model_declaration is not None and not isinstance(model_declaration, Mapping):
            raise ProviderRuntimeConfigError(
                f"rate-limit quota group {quota_group!r}.models.{model} must be an object")
        shared = {key: value for key, value in declaration.items() if key != "models"}
        shared.update(dict(model_declaration or {}))
        declaration = shared
    return normalize_rate_limit_policy(
        declaration, quota_group=quota_group, model=model)


def strict_rate_limit_coverage(policy: Mapping[str, Any]) -> bool:
    """A paid matrix needs requests plus total tokens or separate input/output windows."""
    request_windows = policy.get("request_windows") or ()
    token_fields = {item.get("field") for item in (policy.get("token_windows") or ())}
    has_tokens = "total_tokens" in token_fields or (
        bool(token_fields & {"input_tokens", "input_tokens_excluding_cache_reads"})
        and "output_tokens" in token_fields)
    return bool(request_windows) and has_tokens


def _headers_of(value: Any) -> Mapping[str, Any]:
    for candidate in (
        getattr(value, "headers", None),
        getattr(getattr(value, "response", None), "headers", None),
    ):
        if isinstance(candidate, Mapping) or callable(getattr(candidate, "items", None)):
            return candidate
    return {}


def safe_response_headers(value: Any) -> dict[str, str]:
    """Return only bounded request/rate metadata; never persist arbitrary provider headers."""
    out = {}
    for key, raw in _headers_of(value).items():
        name = str(key).strip().lower()
        if not _SAFE_HEADER.match(name):
            continue
        text = " ".join(str(raw).split())[:160]
        if text:
            out[name] = text
    return dict(sorted(out.items()))


def _duration_seconds(value: Any, *, now: Optional[float] = None) -> Optional[float]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        parsed = float(text)
        return max(parsed, 0.0) if math.isfinite(parsed) else None
    except ValueError:
        pass
    parts = _DURATION_PART.findall(text)
    if parts and "".join(number + unit for number, unit in parts) == text:
        scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        return sum(float(number) * scale[unit] for number, unit in parts)
    try:
        date_value = parsedate_to_datetime(str(value))
        return max(date_value.timestamp() - float(now if now is not None else time.time()), 0.0)
    except (TypeError, ValueError, OverflowError):
        return None


def retry_after_seconds(headers: Mapping[str, str], *, now: Optional[float] = None) -> Optional[float]:
    return _duration_seconds(headers.get("retry-after"), now=now)


def _exhausted_header_cooldown(headers: Mapping[str, str], now: float) -> float:
    waits = []
    for resource in ("requests", "tokens"):
        remaining = headers.get(f"x-ratelimit-remaining-{resource}")
        try:
            exhausted = float(remaining) <= 0
        except (TypeError, ValueError):
            exhausted = False
        if not exhausted:
            continue
        raw_reset = headers.get(f"x-ratelimit-reset-{resource}")
        parsed = _duration_seconds(raw_reset, now=now)
        try:
            numeric = float(raw_reset)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and numeric > now - 60:
            parsed = max(numeric - now, 0.0)
        if parsed is not None:
            waits.append(parsed)
    return max(waits, default=0.0)


def retry_delay_seconds(
    profile: Mapping[str, Any], failure_code: str, failed_attempt: int,
    headers: Optional[Mapping[str, str]] = None,
    *, random_fn: Callable[[], float] = random.random,
) -> float:
    """Delay before the next attempt, honoring Retry-After when it is longer."""
    rule = (profile.get("retry") or {}).get(str(failure_code)) or {}
    delays = list(rule.get("delays_s") or ())
    index = max(int(failed_attempt) - 1, 0)
    base = float(delays[index]) if index < len(delays) else 0.0
    ratio = float(rule.get("jitter_ratio") or 0.0)
    jittered = base * (1.0 + ratio * max(min(float(random_fn()), 1.0), 0.0))
    retry_after = retry_after_seconds(headers or {})
    return max(jittered, float(retry_after or 0.0))


def call_with_raw_headers(endpoint: Any, request: Mapping[str, Any]) -> tuple[Any, dict[str, str]]:
    """Use SDK raw-response mode when available, preserving fake-client and older-SDK seams."""
    raw_endpoint = getattr(endpoint, "with_raw_response", None)
    raw_create = getattr(raw_endpoint, "create", None)
    if callable(raw_create):
        raw = raw_create(**dict(request))
        headers = safe_response_headers(raw)
        parse = getattr(raw, "parse", None)
        response = parse() if callable(parse) else raw
        return response, headers
    response = endpoint.create(**dict(request))
    return response, safe_response_headers(response)


@dataclass(frozen=True)
class RateLimitReservation:
    ident: str
    estimated_tokens: int
    estimated_input_tokens: int
    estimated_output_tokens: int


class SharedRateLimiter:
    """Cross-process rolling-window limiter and credential/model cooldown state."""

    def __init__(
        self,
        policy: Mapping[str, Any],
        model: str,
        state_dir: str | Path | None,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.time,
        uniform_fn: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.policy = normalize_rate_limit_policy(
            policy, quota_group=policy.get("quota_group"))
        self.model = str(model)
        self._sleep = sleep_fn
        self._now = now_fn
        self._uniform = uniform_fn
        self._state_dir = Path(state_dir) if state_dir else None
        self._shared = self._state_dir is not None
        self._memory_state: Optional[dict] = None
        scope = str(self.policy.get("quota_group") or "unscoped")
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24]
        self._state_path = self._state_dir / f"{digest}.json" if self._state_dir else None
        self._lock_path = self._state_dir / f"{digest}.lock" if self._state_dir else None

    def _initial_state(self) -> dict:
        return {
            "schema_version": RATE_LIMIT_CONFIG_SCHEMA_VERSION,
            "quota_group": self.policy.get("quota_group"),
            "requests": [],
            "utilization": self.policy["initial_utilization"],
            "successes_since_rate_limit": 0,
            "scope_cooldown_until": 0.0,
            "models": {},
        }

    @contextmanager
    def _locked_state(self):
        if self._state_dir is None:
            if self._memory_state is None:
                self._memory_state = self._initial_state()
            yield self._memory_state
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    state = json.loads(self._state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    state = self._initial_state()
                if state.get("schema_version") != RATE_LIMIT_CONFIG_SCHEMA_VERSION \
                        or state.get("quota_group") != self.policy.get("quota_group"):
                    state = self._initial_state()
                state.setdefault("policy_fingerprints", {})[self.model] = hashlib.sha256(
                    json.dumps(
                        self.policy, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                yield state
                temporary = self._state_path.with_name(
                    f".{self._state_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
                temporary.write_text(
                    json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
                os.replace(temporary, self._state_path)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _prune(self, state: dict, now: float) -> None:
        periods = [float(item["period_s"]) for item in (
            list(self.policy.get("request_windows") or ())
            + list(self.policy.get("token_windows") or ()))]
        longest = max(periods, default=60.0)
        state["requests"] = [
            item for item in state.get("requests", [])
            if float(item.get(
                "retain_until", float(item.get("at", 0.0)) + longest)) > now
        ]

    @staticmethod
    def _effective_ceiling(
        state: dict, configured: Optional[int], resource: str, model: str,
    ) -> Optional[int]:
        raw = ((state.get("observed_headers") or {}).get(model) or {}).get(
            f"x-ratelimit-limit-{resource}")
        try:
            observed = int(float(raw))
        except (TypeError, ValueError):
            observed = None
        if observed is not None and observed < 1:
            observed = None
        if configured is None:
            return observed
        return min(int(configured), observed) if observed is not None else int(configured)

    @staticmethod
    def _token_value(item: Mapping[str, Any], field: str) -> int:
        if field == "total_tokens":
            return max(int(item.get("tokens", 0)), 0)
        if field == "input_tokens_excluding_cache_reads":
            # Anthropic reports cache reads separately from ``input_tokens``; the latter already
            # is the quota-bearing value for a limit declared as excluding cache reads.
            return max(int(item.get("input_tokens", 0)), 0)
        return max(int(item.get(field, 0)), 0)

    @staticmethod
    def _window_events(
        events: list[dict], window: Mapping[str, Any], model: str, now: float,
    ) -> list[dict]:
        period = float(window["period_s"])
        return [
            item for item in events
            if now - float(item["at"]) < period
            and (window.get("scope") != "model" or item.get("model") == model)
        ]

    @classmethod
    def _pacing_wait(
        cls,
        events: list[dict],
        now: float,
        model: str,
        estimates: Mapping[str, int],
        request_windows: list[dict],
        token_windows: list[dict],
        utilization: float,
    ) -> float:
        """Spread reservations at the strictest declared rate instead of boundary bursting."""
        if not events:
            return 0.0
        waits = []
        for window in request_windows:
            if float(window["period_s"]) > 60.0:
                continue
            active = cls._window_events(events, window, model, now)
            if active:
                interval = float(window["period_s"]) / max(
                    float(window["limit"]) * utilization, 0.01)
                waits.append(float(active[-1]["at"]) + interval - now)
        for window in token_windows:
            if float(window["period_s"]) > 60.0:
                continue
            active = cls._window_events(events, window, model, now)
            if not active:
                continue
            estimate = max(int(estimates.get(str(window["field"]), 0)), 1)
            interval = float(window["period_s"]) * estimate / max(
                float(window["limit"]) * utilization, 0.01)
            waits.append(float(active[-1]["at"]) + interval - now)
        if not waits:
            return 0.0
        return max(max(waits), 0.0)

    def acquire(
        self,
        estimated_tokens: int,
        *,
        estimated_input_tokens: Optional[int] = None,
        estimated_output_tokens: Optional[int] = None,
    ) -> tuple[RateLimitReservation, float]:
        estimate = max(int(estimated_tokens), 1)
        if estimated_input_tokens is None and estimated_output_tokens is None:
            input_estimate, output_estimate = estimate, 0
        else:
            input_estimate = max(int(estimated_input_tokens or 0), 0)
            output_estimate = max(int(estimated_output_tokens or 0), 0)
            estimate = max(input_estimate + output_estimate, 1)
        estimates = {
            "total_tokens": estimate,
            "input_tokens": input_estimate,
            # Cache hits cannot be known before the response, so reservation is conservative.
            "input_tokens_excluding_cache_reads": input_estimate,
            "output_tokens": output_estimate,
        }
        waited = 0.0
        while True:
            now = self._now()
            wait = 0.0
            with self._locked_state() as state:
                self._prune(state, now)
                model_state = state.setdefault("models", {}).setdefault(
                    self.model, {"cooldown_until": 0.0, "consecutive_server_errors": 0})
                wait = max(
                    float(state.get("scope_cooldown_until", 0.0)) - now,
                    float(model_state.get("cooldown_until", 0.0)) - now,
                    0.0,
                )
                utilization = max(min(float(state.get("utilization", 0.0)),
                                      self.policy["max_utilization"]), 0.01)
                events = state["requests"]
                request_windows = list(self.policy.get("request_windows") or ())
                token_windows = list(self.policy.get("token_windows") or ())
                observed_rpm = self._effective_ceiling(
                    state, None, "requests", self.model)
                observed_tpm = self._effective_ceiling(
                    state, None, "tokens", self.model)
                if observed_rpm is not None:
                    request_windows.append({
                        "limit": observed_rpm, "period_s": 60.0, "scope": "credential"})
                if observed_tpm is not None:
                    token_windows.append({
                        "limit": observed_tpm,
                        "period_s": 60.0,
                        "scope": "credential",
                        "field": "total_tokens",
                    })
                wait = max(
                    wait,
                    self._pacing_wait(
                        events, now, self.model, estimates,
                        request_windows, token_windows, utilization),
                )
                for window in request_windows:
                    period = float(window["period_s"])
                    active = self._window_events(events, window, self.model, now)
                    ceiling = max(int(int(window["limit"]) * utilization), 1)
                    if len(active) >= ceiling:
                        wait = max(wait, period - (now - float(active[0]["at"])))
                for window in token_windows:
                    period = float(window["period_s"])
                    field = str(window["field"])
                    active = self._window_events(events, window, self.model, now)
                    used = sum(self._token_value(item, field) for item in active)
                    ceiling = max(int(int(window["limit"]) * utilization), 1)
                    # A single request may legitimately exceed the utilization target while still
                    # fitting the provider's hard ceiling. Admit it once; subsequent reservations
                    # wait out the debt instead of deadlocking forever.
                    if active and used + estimates[field] > ceiling:
                        wait = max(wait, period - (now - float(active[0]["at"])))
                if wait <= 0:
                    reservation = RateLimitReservation(
                        uuid.uuid4().hex, estimate, input_estimate, output_estimate)
                    events.append({
                        "id": reservation.ident,
                        "model": self.model,
                        "at": now,
                        "retain_until": now + max(
                            [float(item["period_s"]) for item in (
                                request_windows + token_windows)], default=60.0),
                        "tokens": estimate,
                        "input_tokens": input_estimate,
                        "output_tokens": output_estimate,
                        "cached_tokens": 0,
                    })
                    return reservation, round(waited, 3)
            delay = max(float(wait), 0.001)
            self._sleep(delay)
            waited += delay

    @staticmethod
    def _apply_observed_limits(
        state: dict, headers: Mapping[str, str], model: str,
    ) -> None:
        observed = state.setdefault("observed_headers", {}).setdefault(model, {})
        for key in (
            "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
        ):
            if key in headers:
                observed[key] = headers[key]

    def record_success(
        self,
        reservation: RateLimitReservation,
        actual_tokens: Optional[int],
        headers: Mapping[str, str],
        *,
        actual_input_tokens: Optional[int] = None,
        actual_output_tokens: Optional[int] = None,
        cached_input_tokens: Optional[int] = None,
    ) -> None:
        with self._locked_state() as state:
            self._apply_observed_limits(state, headers, self.model)
            header_wait = _exhausted_header_cooldown(headers, self._now())
            if self._shared and header_wait > 0:
                state["scope_cooldown_until"] = max(
                    float(state.get("scope_cooldown_until", 0.0)), self._now() + header_wait)
            if actual_tokens is not None and int(actual_tokens) > 0:
                for item in state.get("requests", []):
                    if item.get("id") == reservation.ident:
                        item["tokens"] = int(actual_tokens)
                        if actual_input_tokens is not None:
                            item["input_tokens"] = max(int(actual_input_tokens), 0)
                        if actual_output_tokens is not None:
                            item["output_tokens"] = max(int(actual_output_tokens), 0)
                        if cached_input_tokens is not None:
                            item["cached_tokens"] = max(int(cached_input_tokens), 0)
                        break
            model_state = state.setdefault("models", {}).setdefault(self.model, {})
            model_state["consecutive_server_errors"] = 0
            successes = int(state.get("successes_since_rate_limit", 0)) + 1
            threshold = self.policy["successes_before_increase"]
            if successes >= threshold:
                state["utilization"] = min(
                    float(state.get("utilization", self.policy["initial_utilization"]))
                    + self.policy["increase_step"],
                    self.policy["max_utilization"],
                )
                successes = 0
            state["successes_since_rate_limit"] = successes

    def record_failure(
        self,
        reservation: RateLimitReservation,
        failure_code: str,
        headers: Mapping[str, str],
        *,
        minimum_cooldown_s: float,
        transport_profile: Mapping[str, Any],
    ) -> float:
        del reservation  # Failed requests remain in the rolling window; providers may count them.
        now = self._now()
        header_cooldown = retry_after_seconds(headers, now=now) or 0.0
        header_cooldown = max(header_cooldown, _exhausted_header_cooldown(headers, now))
        effective_minimum = max(float(minimum_cooldown_s), header_cooldown, 0.0)
        if failure_code == "provider_rate_limited" and effective_minimum <= 0:
            effective_minimum = float(self.policy["rate_limit_default_cooldown_s"])
        cooldown_until = now + effective_minimum
        with self._locked_state() as state:
            self._apply_observed_limits(state, headers, self.model)
            model_state = state.setdefault("models", {}).setdefault(
                self.model, {"cooldown_until": 0.0, "consecutive_server_errors": 0})
            if failure_code == "provider_rate_limited":
                state["utilization"] = max(
                    float(state.get("utilization", self.policy["initial_utilization"]))
                    * self.policy["decrease_factor"],
                    0.05,
                )
                state["successes_since_rate_limit"] = 0
                if self._shared:
                    state["scope_cooldown_until"] = max(
                        float(state.get("scope_cooldown_until", 0.0)), cooldown_until)
            elif failure_code == "provider_server_error":
                consecutive = int(model_state.get("consecutive_server_errors", 0)) + 1
                model_state["consecutive_server_errors"] = consecutive
                declaration = transport_profile.get("server_error_cooldown") or {}
                after = int(declaration.get("after_consecutive") or 0)
                if after and consecutive >= after:
                    minimum = float(declaration.get("min_s") or 0.0)
                    maximum = float(declaration.get("max_s") or minimum)
                    cooldown_until = now + self._uniform(minimum, maximum)
                    if self._shared:
                        model_state["cooldown_until"] = max(
                            float(model_state.get("cooldown_until", 0.0)), cooldown_until)
            else:
                model_state["consecutive_server_errors"] = 0
            return round(max(cooldown_until - now, 0.0), 3) if self._shared \
                else round(effective_minimum, 3)


__all__ = [
    "DEFAULT_TRANSPORT_PROFILE",
    "ProviderRuntimeConfigError",
    "RATE_LIMIT_CONFIG_SCHEMA_VERSION",
    "RateLimitReservation",
    "SharedRateLimiter",
    "call_with_raw_headers",
    "load_rate_limit_config",
    "normalize_rate_limit_policy",
    "normalize_transport_profile",
    "provider_attempt_limit",
    "rate_limit_policy_for",
    "retry_after_seconds",
    "retry_delay_seconds",
    "safe_response_headers",
    "strict_rate_limit_coverage",
]
