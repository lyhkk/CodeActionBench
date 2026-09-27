"""Normalized provider boundary for the Python ``codeaction-reference`` scaffold.

Provider-specific code ends here.  The reference loop consumes ``ModelTurn`` and recorded
``ModelCapabilities`` only; it owns context preparation, stop-reason policy, and tool execution.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import inspect
import json
import os
import random
import re
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from codeaction.contracts.failures import (
    EmptyProviderCompletion,
    FailureCode,
    ProviderCallError,
    classify_provider_exception,
    ensure_usable_provider_completion,
)
from codeaction.providers.model_registry import (
    DEFAULT_REASONING_FIELDS,
    REASONING_RUNGS,
    RegistryError,
    credential_for,
    find_model,
    resolve_model,
)
from codeaction.providers.provider_runtime import (
    ProviderPhaseTimeout,
    SharedRateLimiter,
    call_with_raw_headers,
    normalize_rate_limit_policy,
    normalize_transport_profile,
    provider_attempt_limit,
    provider_sdk_timeout,
    rate_limit_policy_for,
    retry_delay_seconds,
    safe_response_headers,
)


TOKEN_ESTIMATOR_UTF8_BYTES_V1 = "utf8-bytes-v1"
_SUPPORTED_ESTIMATORS = {TOKEN_ESTIMATOR_UTF8_BYTES_V1}

# Request-only context metadata. Kept as a literal here instead of importing ``context`` because
# that module imports ModelCapabilities from this one; both sides are covered by translation tests.
# The wire-contract field a context manager stamps on a message to request a provider-side
# cache checkpoint. Defined HERE because both sides depend on this layer: the harness context
# manager writes it, every provider adapter strips and honors it.
CACHE_CHECKPOINT_FIELD = "_cache_checkpoint"
_CACHE_CHECKPOINT_FIELD = CACHE_CHECKPOINT_FIELD


@dataclass(frozen=True)
class ModelCapabilities:
    context_window_tokens: int
    max_output_tokens: int
    token_estimator_id: str
    supports_tools: bool
    supports_images: bool
    model_seed_support: str = "unsupported"

    def __post_init__(self):
        if int(self.context_window_tokens) <= 0:
            raise ValueError("context_window_tokens must be positive")
        if int(self.max_output_tokens) <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("max_output_tokens must be smaller than the context window")
        if self.token_estimator_id not in _SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"unsupported token estimator {self.token_estimator_id!r}; "
                f"supported={sorted(_SUPPORTED_ESTIMATORS)}")
        if self.supports_tools is not True or self.supports_images is not True:
            raise ValueError("codeaction-reference requires native tool and image support")
        if self.model_seed_support not in ("unsupported", "supported"):
            raise ValueError("model_seed_support must be 'unsupported' or 'supported'")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelTurn:
    message: Dict[str, object]
    stop_reason: str
    usage: Dict[str, Optional[int]]
    provider_request_id: Optional[str]
    # Provider-native hidden reasoning stays out of canonical history and is recorded separately
    # in the transcript. A provider profile may also replay it on that episode's wire requests.
    reasoning_content: Optional[str] = None
    # Structured detail attached to a terminal stop reason.  Populated only where the provider
    # returns one (Anthropic sets it on ``refusal``); the loop reads it to attribute the
    # termination instead of falling through to a generic contract violation.
    stop_details: Optional[Dict[str, object]] = None


class ProviderAdapter(Protocol):
    def capabilities(self) -> ModelCapabilities:
        ...

    def request_profile(self) -> Dict[str, object]:
        ...

    def transport_profile(self) -> Dict[str, object]:
        ...

    def rate_limit_policy(self) -> Dict[str, object]:
        ...

    def step(
        self,
        messages: List[Dict[str, object]],
        tools: List[Dict[str, object]],
        requested_output_tokens: int,
    ) -> ModelTurn:
        ...


def resolve_output_tokens(capabilities: ModelCapabilities, requested_output_tokens: int) -> int:
    requested = int(requested_output_tokens)
    if requested <= 0:
        raise ValueError("requested_output_tokens must be positive")
    return min(requested, int(capabilities.max_output_tokens))


def requested_output_tokens_for_profile(
    capabilities: ModelCapabilities,
    requested_output_tokens: int,
    request_profile: Mapping[str, Any],
) -> int:
    """Apply a registry-declared per-model output request before resolving its native limit."""
    requested = request_profile.get("requested_output_tokens", requested_output_tokens)
    return resolve_output_tokens(capabilities, int(requested))


def _reasoning_profile_name(value: Optional[str]) -> str:
    """Resolve which rung of the benchmark's reasoning ladder this run asks for.

    Which rungs exist for a given model, and what each one means natively, is declared in the
    model registry; this only picks the name.
    """
    profile = str(
        value
        or os.environ.get("BENCH_REASONING_PROFILE")
        or "disabled"
    ).strip().lower()
    if profile not in REASONING_RUNGS:
        raise ValueError(
            f"reasoning profile must be one of {', '.join(REASONING_RUNGS)}")
    return profile


def provider_request_profile(
    model_id: str,
    *,
    scripted: bool = False,
    reasoning_profile: Optional[str] = None,
) -> dict:
    """Return the non-secret provider settings covered by scaffold identity.

    The benchmark declares one abstract reasoning ladder -- disabled/low/medium/high -- and the
    model registry maps each rung onto that vendor's own native control.  Comparability comes
    from the mapping being declared and hashed, not from the settings being numerically equal:
    Qwen's control is an integer ``thinking_budget``, Claude's is an ``output_config.effort``
    enum.  ``BENCH_REASONING_PROFILE`` selects the rung.

    A profile may also declare ``temperature``.  Absent or ``None`` means the client sends no
    sampling override and the provider default applies.  A numeric value is sent explicitly and
    the scaffold card records that exact wire behavior.

    Model ids outside the registry keep a generic default so unit fixtures stay constructible;
    the paths that actually reach an endpoint go through ``build_provider``, which requires a
    registered model.
    """
    if scripted:
        return {"id": "scripted-v1", "reasoning": "none", "extra_body": {}}
    entry = find_model(model_id)
    if entry is not None:
        return entry.profile(_reasoning_profile_name(reasoning_profile))
    return {
        "id": "openai-compatible-default-v1",
        "reasoning": "provider-default",
        "extra_body": {},
    }


def request_profile_of(provider: Any) -> dict:
    method = getattr(provider, "request_profile", None)
    value = (
        method() if callable(method)
        else provider_request_profile(getattr(provider, "model", ""))
    )
    if not isinstance(value, dict) or not isinstance(value.get("extra_body"), dict):
        raise TypeError("provider request_profile() must return a profile object")
    return json.loads(json.dumps(value, sort_keys=True))


def transport_profile_of(provider: Any) -> dict:
    method = getattr(provider, "transport_profile", None)
    value = method() if callable(method) else normalize_transport_profile()
    return normalize_transport_profile(value)


def rate_limit_policy_of(provider: Any) -> dict:
    method = getattr(provider, "rate_limit_policy", None)
    value = method() if callable(method) else normalize_rate_limit_policy(None, quota_group=None)
    return normalize_rate_limit_policy(value, quota_group=value.get("quota_group"))


def _first_int(*values) -> Optional[int]:
    for value in values:
        if value in (None, ""):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def capabilities_for_model(model_id: Any) -> ModelCapabilities:
    """Build declared capabilities for a registered model.

    Capabilities used to arrive as environment variables, which meant a typo in a shell export
    could silently declare a 1M window for a 200K model and disable context management for the
    whole run.  They are registry facts now, so the only way to change them is an audited edit.
    """
    entry = resolve_model(model_id)
    return ModelCapabilities(
        context_window_tokens=entry.context_window_tokens,
        max_output_tokens=entry.max_output_tokens,
        token_estimator_id=entry.token_estimator_id,
        supports_tools=True,
        supports_images=True,
        model_seed_support="unsupported",
    )


def normalize_stop_reason(value: Any, message: Mapping[str, Any]) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "max_tokens": "length",
        "max_output_tokens": "length",
        "tool_use": "tool_calls",
        "end_turn": "stop",
        "stop_sequence": "stop",
    }
    raw = aliases.get(raw, raw)
    if raw:
        return raw
    return "tool_calls" if message.get("tool_calls") else "stop"


def _vendor_passthrough(value: Any) -> Optional[dict]:
    """The vendor-private envelope a tool call must carry back, or None.

    Rebuilding a tool call from id/type/function alone silently discarded this, and on Gemini 3
    that is fatal rather than cosmetic: the signature Google returns in
    ``extra_content.google.thought_signature`` is REQUIRED on every replayed function call, and
    omitting it fails the whole request -- measured 2026-08-10, HTTP 400 "Function call is missing
    a thought_signature in functionCall parts". It is echoed back opaquely and only ever to the
    vendor that issued it on that same conversation, so no vendor sees another's fields and
    nothing is invented; a provider that sends none still gets a byte-identical request, which
    matters because strict vendors reject unknown keys.
    """
    if isinstance(value, Mapping):
        extra = value.get("extra_content")
    else:
        extra = getattr(value, "extra_content", None)
        if extra is None:
            extra = (getattr(value, "model_extra", None) or {}).get("extra_content")
    if not isinstance(extra, Mapping) or not extra:
        return None
    return json.loads(json.dumps(extra))


def _tool_call_to_dict(value: Any) -> dict:
    if isinstance(value, Mapping):
        function = value.get("function") or {}
        call = {
            "id": str(value.get("id") or ""),
            "type": str(value.get("type") or "function"),
            "function": {
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
            },
        }
    else:
        function = getattr(value, "function", None)
        call = {
            "id": str(getattr(value, "id", "") or ""),
            "type": str(getattr(value, "type", "function") or "function"),
            "function": {
                "name": str(getattr(function, "name", "") or ""),
                "arguments": str(getattr(function, "arguments", "") or ""),
            },
        }
    passthrough = _vendor_passthrough(value)
    if passthrough is not None:
        call["extra_content"] = passthrough
    return call


def normalize_message(value: Any) -> Dict[str, object]:
    if isinstance(value, Mapping):
        content = value.get("content")
        tool_calls = value.get("tool_calls") or []
    else:
        content = getattr(value, "content", None)
        tool_calls = getattr(value, "tool_calls", None) or []
    out: Dict[str, object] = {"content": content}
    if tool_calls:
        out["tool_calls"] = [_tool_call_to_dict(item) for item in tool_calls]
    return out


def normalize_reasoning_content(value: Any, fields=None) -> Optional[str]:
    """Read provider-native reasoning without adding it to canonical model history.

    ``reasoning_content`` is a vendor convention, not part of the OpenAI-chat protocol, so which
    field carries the text is a per-model registry fact. Candidates are tried in declared order and
    the first non-empty one wins; a vendor whose field nobody has confirmed can therefore declare
    both spellings and let the run-level capture check settle it, rather than silently recording
    nothing the way the Anthropic path did.
    """
    for field in (fields or ("reasoning_content",)):
        content = (value.get(field) if isinstance(value, Mapping)
                   else getattr(value, field, None))
        if content is None:
            continue
        text = str(content)
        if text.strip():
            return text
    return None


_INLINE_THOUGHT_CACHE: Dict[str, Any] = {}


def split_inline_thought(content: Any, tag: str):
    """Return ``(answer_text, thought_text)`` for a vendor that inlines reasoning in ``content``.

    Google's OpenAI-compatibility layer has no reasoning side channel: with ``include_thoughts``
    the summary arrives inside ``message.content`` wrapped in ``<thought>…</thought>`` (verified
    live 2026-08-08). Leaving it in place is worse than merely losing the reasoning column — the
    text is replayed into the next prompt as the assistant's own answer and shown to readers as
    the model's response, so the split is a correctness fix, not just bookkeeping.

    An unterminated opening tag is treated as thought to the end of the message: that is what a
    turn truncated by the output limit looks like, and keeping the fragment as the answer would
    put a half-written deliberation into the transcript as the model's reply.
    """
    if not isinstance(content, str) or not content:
        return content, None
    pattern = _INLINE_THOUGHT_CACHE.get(tag)
    if pattern is None:
        escaped = re.escape(tag)
        pattern = re.compile(
            rf"<{escaped}\s*>(?P<body>.*?)(?:</{escaped}\s*>|\Z)", re.DOTALL | re.IGNORECASE)
        _INLINE_THOUGHT_CACHE[tag] = pattern
    thoughts = []

    def take(match):
        thoughts.append(match.group("body"))
        return ""

    answer = pattern.sub(take, content).strip()
    thought = "\n".join(part.strip() for part in thoughts if part.strip()).strip()
    return (answer or None), (thought or None)


def _usage(value: Any) -> Dict[str, Optional[int]]:
    if isinstance(value, Mapping):
        prompt = value.get("prompt_tokens")
        completion = value.get("completion_tokens")
        cached = value.get("cached_tokens")
        details = value.get("prompt_tokens_details") or {}
        completion_details = (
            value.get("completion_tokens_details")
            or value.get("output_tokens_details")
            or {})
        reasoning = value.get("reasoning_tokens")
        if reasoning is None:
            reasoning = (completion_details.get("reasoning_tokens")
                         if isinstance(completion_details, Mapping)
                         else getattr(completion_details, "reasoning_tokens", None))
        if cached is None:
            cached = (details.get("cached_tokens")
                      if isinstance(details, Mapping)
                      else getattr(details, "cached_tokens", None))
    else:
        prompt = getattr(value, "prompt_tokens", None)
        completion = getattr(value, "completion_tokens", None)
        details = getattr(value, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        completion_details = (
            getattr(value, "completion_tokens_details", None)
            or getattr(value, "output_tokens_details", None))
        reasoning = getattr(value, "reasoning_tokens", None)
        if reasoning is None and completion_details is not None:
            reasoning = getattr(completion_details, "reasoning_tokens", None)

    def optional_int(item):
        try:
            return int(item) if item is not None else None
        except (TypeError, ValueError):
            return None

    out = {
        "prompt_tokens": optional_int(prompt),
        "completion_tokens": optional_int(completion),
    }
    if cached is not None:
        out["cached_tokens"] = optional_int(cached)
    created = (
        details.get("cache_creation_input_tokens", details.get("cache_write_tokens"))
        if isinstance(details, Mapping)
        else (
            getattr(details, "cache_creation_input_tokens", None)
            if details is not None else None
        )
    )
    if created is None and details is not None and not isinstance(details, Mapping):
        created = getattr(details, "cache_write_tokens", None)
    if created is not None:
        out["cache_creation_tokens"] = optional_int(created)
    if reasoning is not None:
        out["reasoning_tokens"] = optional_int(reasoning)
    return out


def prepare_openai_compatible_messages(
    messages: Sequence[Mapping[str, Any]], model_id: str,
    reasoning_shadow: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, object]]:
    """Remove request-only metadata and translate it only for confirmed DashScope models.

    DashScope documents the marker as a key inside a content PART, on system/user/assistant/tool
    messages alike, with at most four markers per request. A tool result reaches this function as
    a plain string, so for that credential every tool result is rendered as a single text part --
    marked or not. The uniformity is load-bearing rather than cosmetic: the marker set rolls
    forward each turn, and if part form were applied only to the currently marked messages, then
    the turn a marker rolls off would rewrite that message's bytes and invalidate every cache
    entry whose prefix contains it. Only the marker moves; the transported text does not.
    """
    copied = [deepcopy(dict(message)) for message in messages]
    entry = find_model(model_id)
    explicit_dashscope = entry is not None and entry.credential == "qwen"
    candidates: List[Dict[str, object]] = []
    for message in copied:
        if message.get("role") == "assistant" and reasoning_shadow:
            reasoning = reasoning_shadow.get(_assistant_reasoning_key(message))
            if reasoning:
                message["reasoning_content"] = reasoning
        marked_message = message.pop(_CACHE_CHECKPOINT_FIELD, False) is True
        content = message.get("content")
        if explicit_dashscope and message.get("role") == "tool" and isinstance(content, str):
            content = [{"type": "text", "text": content}]
            message["content"] = content
        if not isinstance(content, list):
            continue
        carrier = None
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.pop(_CACHE_CHECKPOINT_FIELD, False) is True:
                candidates.append(block)
            elif block.get("type") == "text":
                carrier = block
        # A message-level marker means the boundary had no block list of its own. It is realized
        # on that message's last text part; a message with no text part (an image-only bundle)
        # yields no candidate and the next-older boundary is used instead.
        # Identity, not equality: two turns can produce byte-identical payloads, and `in` would
        # then silently drop the newer boundary's marker.
        if marked_message and carrier is not None and not any(
                carrier is block for block in candidates):
            candidates.append(carrier)
    if explicit_dashscope:
        for block in candidates[-4:]:
            block["cache_control"] = {"type": "ephemeral"}
    return copied


def _assistant_reasoning_key(message: Mapping[str, Any]) -> str:
    """Identify one normalized assistant turn without persisting provider reasoning in history."""
    payload = {
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, Mapping) \
        else getattr(value, name, default)


def _is_timeout_exception(exc: BaseException) -> bool:
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError) or "timeout" in type(current).__name__.lower():
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _consume_openai_chat_stream(
    stream, transport, *, monotonic=time.monotonic, started_at=None,
):
    """Translate Chat Completions deltas into the same buffered response fields we normalize.

    No chunk or opaque provider object is persisted. The whole-response ceiling is checked after
    every delivered chunk; socket read timeout bounds both first-byte and inter-chunk idle waits.
    """
    started = monotonic() if started_at is None else float(started_at)
    first_at = None
    response_id = None
    finish_reason = None
    usage = None
    content_parts = []
    reasoning_parts = []
    tool_calls = {}
    hard_s = float(transport["response_hard_timeout_s"])
    first_s = float(transport["first_response_timeout_s"])
    idle_s = float(transport["stream_idle_timeout_s"])
    try:
        for chunk in stream:
            now = monotonic()
            if now - started > hard_s:
                raise ProviderPhaseTimeout("response_hard", hard_s)
            if first_at is None:
                first_at = now
                if first_at - started > first_s:
                    raise ProviderPhaseTimeout("first_response", first_s)
            response_id = response_id or _field(chunk, "id")
            chunk_usage = _field(chunk, "usage")
            if chunk_usage is not None:
                usage = chunk_usage
            choices = _field(chunk, "choices", None) or []
            for choice in choices:
                observed_finish = _field(choice, "finish_reason")
                if observed_finish is not None:
                    finish_reason = observed_finish
                delta = _field(choice, "delta") or {}
                content = _field(delta, "content")
                if content:
                    content_parts.append(str(content))
                reasoning = _field(delta, "reasoning_content")
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                for fragment in _field(delta, "tool_calls", None) or []:
                    index = int(_field(fragment, "index", 0) or 0)
                    item = tool_calls.setdefault(index, {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    item["id"] = _field(fragment, "id") or item["id"]
                    item["type"] = _field(fragment, "type") or item["type"]
                    function = _field(fragment, "function") or {}
                    name = _field(function, "name")
                    arguments = _field(function, "arguments")
                    if name:
                        item["function"]["name"] += str(name)
                    if arguments:
                        item["function"]["arguments"] += str(arguments)
    except ProviderPhaseTimeout:
        raise
    except Exception as exc:
        if _is_timeout_exception(exc):
            phase = "first_response" if first_at is None else "stream_idle"
            limit = first_s if first_at is None else idle_s
            raise ProviderPhaseTimeout(phase, limit) from exc
        raise
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if first_at is None:
        raise EmptyProviderCompletion("stream ended before the first response chunk")
    message = {
        "content": "".join(content_parts) or None,
        "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
    }
    return {
        "id": response_id,
        "message": message,
        "reasoning_content": "".join(reasoning_parts) or None,
        "finish_reason": finish_reason,
        "usage": usage,
    }


class ModelAdapter:
    """OpenAI-compatible provider adapter with bounded, selective infrastructure retries."""

    # Class-level defaults so the attributes exist even on instances built with
    # `__new__` (the unit-test seam). They are the accounting an episode needs to tell a
    # vendor outage apart from the model's own time, so a missing one must never be an
    # AttributeError deep inside a retry.
    infra_retries: int = 0
    infra_wait_s: float = 0.0
    quota_wait_s: float = 0.0

    def __init__(
        self,
        model=None,
        provider_model=None,
        base_url=None,
        api_key=None,
        timeout=None,
        attempts=None,
        capabilities: Optional[ModelCapabilities] = None,
        request_profile: Optional[dict] = None,
        transport_profile: Optional[dict] = None,
        rate_limit_policy: Optional[dict] = None,
        rate_limit_state_dir: Optional[str | Path] = None,
        reasoning_profile: Optional[str] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
        event_sink: Optional[Callable[[Dict[str, object]], None]] = None,
    ):
        from openai import OpenAI
        # Model identity, capabilities, and the credential alias are resolved by `build_provider`
        # from the registry and the credential file.  Direct construction is the unit-test seam
        # and must therefore be given them explicitly.
        self.model = model
        # The benchmark id stays the lookup key for registry, capabilities, pricing and quota; only
        # the wire request carries the vendor's own name for the model.
        # Resolved lazily: a test seam may reassign `.model` after construction, and the
        # wire name must follow it unless a vendor name was declared explicitly.
        self._provider_model_override = str(provider_model) if provider_model else None
        if capabilities is None:
            capabilities = capabilities_for_model(self.model)
        self._capabilities = capabilities
        self._request_profile = request_profile or provider_request_profile(
            self.model, reasoning_profile=reasoning_profile)
        entry = find_model(self.model)
        self._credential_alias = entry.credential if entry is not None else None
        self._reasoning_fields = (
            entry.reasoning_fields() if entry is not None else ("reasoning_content",)
        ) or ("reasoning_content",)
        self._inline_thought_tag = entry.inline_thought_tag() if entry is not None else None
        self._transport_profile = normalize_transport_profile(
            transport_profile or (entry.transport_profile() if entry is not None else None))
        effective_timeout = provider_sdk_timeout(self._transport_profile, timeout)
        effective_attempts = self._transport_profile["attempts"] if attempts is None else int(attempts)
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=effective_timeout, max_retries=0)
        self._attempts = effective_attempts
        self._sleep = sleep_fn
        self._random = random_fn
        self._event_sink = event_sink
        self._rate_limit_policy = normalize_rate_limit_policy(
            rate_limit_policy, quota_group=(rate_limit_policy or {}).get("quota_group"))
        self._rate_limiter = SharedRateLimiter(
            self._rate_limit_policy, self.model, rate_limit_state_dir, sleep_fn=sleep_fn)
        self.infra_retries = 0
        # Seconds spent on FAILED provider attempts and the sleeps between them. Credited
        # back to the episode's wall budget: it is the vendor's outage, not the model's
        # deliberation, and charging it scores an infra fault as a model failure.
        self.infra_wait_s = 0.0
        self.quota_wait_s = 0.0
        self._request_token_estimate = None
        self._request_input_token_estimate = None
        self._request_output_token_estimate = None
        self.last_effective_output_tokens = None
        self._prompt_cache_scope = f"codeaction:{self.model}:{secrets.token_hex(16)}"
        # Episode-local only. A new execution constructs a new adapter, so reasoning can be
        # replayed within an episode without crossing an episode or retry boundary.
        self._reasoning_shadow: Dict[str, str] = {}

    def set_event_sink(
        self, sink: Optional[Callable[[Dict[str, object]], None]],
    ) -> None:
        """Attach the transcript-owned sink for safe provider-attempt telemetry."""
        self._event_sink = sink

    def _emit_attempt_failure(
        self,
        *,
        attempt: int,
        failure,
        elapsed_s: float,
        retry_delay_s: float,
        will_retry: bool,
    ) -> None:
        sink = getattr(self, "_event_sink", None)
        if sink is None:
            return
        sink({
            "event": "provider_attempt_failed",
            "observed_at_unix_s": round(time.time(), 3),
            "provider_attempt": int(attempt),
            "provider_attempts_max": int(self._attempts),
            "will_retry": bool(will_retry),
            "failed_request_elapsed_s": round(max(float(elapsed_s), 0.0), 3),
            "retry_delay_s": round(max(float(retry_delay_s), 0.0), 3),
            "failure": failure.to_dict(),
        })

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def transport_profile(self) -> Dict[str, object]:
        return json.loads(json.dumps(self._transport_profile, sort_keys=True))

    def rate_limit_policy(self) -> Dict[str, object]:
        return json.loads(json.dumps(self._rate_limit_policy, sort_keys=True))

    def set_request_token_estimate(
        self, input_tokens: int, output_tokens: Optional[int] = None,
    ) -> None:
        if output_tokens is None:
            self._request_token_estimate = max(int(input_tokens), 1)
            self._request_input_token_estimate = self._request_token_estimate
            self._request_output_token_estimate = 0
            return
        self._request_input_token_estimate = max(int(input_tokens), 0)
        self._request_output_token_estimate = max(int(output_tokens), 0)
        self._request_token_estimate = max(
            self._request_input_token_estimate + self._request_output_token_estimate, 1)

    def _emit_provider_telemetry(self, event: str, **fields: object) -> None:
        sink = getattr(self, "_event_sink", None)
        if sink is not None:
            sink({"event": event, "observed_at_unix_s": round(time.time(), 3), **fields})

    @property
    def _wire_model(self) -> str:
        """The name the vendor API expects; the benchmark id unless one was declared.

        Read defensively: unit tests build adapters through ``__new__`` and set only the
        attributes under test, so a missing override must fall back rather than raise.
        """
        return getattr(self, "_provider_model_override", None) or self.model

    def request_profile(self) -> Dict[str, object]:
        # Written as a branch, not `getattr(self, name, default)`: Python evaluates that default
        # eagerly on every call, so the fallback ran even when the profile was already set. For a
        # model that declares no `disabled` rung -- Fable, Kimi, Gemini, Grok all reason
        # unconditionally -- the fallback's default rung raised, and the model could not be run at
        # all despite being configured correctly.
        value = self._request_profile if hasattr(self, "_request_profile") \
            else provider_request_profile(self.model)
        return json.loads(json.dumps(value, sort_keys=True))

    def step(self, messages, tools, requested_output_tokens) -> ModelTurn:
        profile = self.request_profile()
        replay_reasoning = bool(
            profile.get("replay_reasoning_content") is True
            or profile["extra_body"].get("preserve_thinking") is True
        )
        effective = requested_output_tokens_for_profile(
            self._capabilities, requested_output_tokens, profile)
        self.last_effective_output_tokens = effective
        transport = getattr(self, "_transport_profile", None) or normalize_transport_profile()
        self._transport_profile = transport
        rate_policy = getattr(self, "_rate_limit_policy", None) or normalize_rate_limit_policy(
            None, quota_group=None)
        self._rate_limit_policy = rate_policy
        for attempt in range(1, self._attempts + 1):
            estimate = int(getattr(self, "_request_token_estimate", None) or effective)
            input_estimate = getattr(self, "_request_input_token_estimate", None)
            output_estimate = getattr(self, "_request_output_token_estimate", None)
            limiter = getattr(self, "_rate_limiter", None)
            if limiter is None:
                policy = normalize_rate_limit_policy(None, quota_group=None)
                limiter = SharedRateLimiter(policy, self.model, None, sleep_fn=self._sleep)
                self._rate_limiter = limiter
            reservation, quota_wait = limiter.acquire(
                estimate,
                estimated_input_tokens=input_estimate,
                estimated_output_tokens=output_estimate,
            )
            if quota_wait:
                self.quota_wait_s = round(
                    float(getattr(self, "quota_wait_s", 0.0)) + quota_wait, 3)
                self.infra_wait_s = round(
                    float(getattr(self, "infra_wait_s", 0.0)) + quota_wait, 3)
                self._emit_provider_telemetry(
                    "provider_quota_wait", quota_wait_s=quota_wait,
                    quota_group=self._rate_limit_policy.get("quota_group"))
            attempt_started = time.monotonic()
            try:
                output_parameter = str(profile.get("output_token_parameter") or "max_tokens")
                if output_parameter not in ("max_tokens", "max_completion_tokens"):
                    raise ValueError(
                        f"unsupported output token parameter {output_parameter!r}")
                request = {
                    "model": self._wire_model,
                    "messages": prepare_openai_compatible_messages(
                        messages,
                        self.model,
                        reasoning_shadow=(
                            getattr(self, "_reasoning_shadow", {})
                            if replay_reasoning
                            else None
                        ),
                    ),
                    "tools": tools,
                    output_parameter: effective,
                }
                entry = find_model(self.model)
                credential_alias = (
                    entry.credential if entry is not None
                    else getattr(self, "_credential_alias", None)
                )
                if credential_alias == "xai":
                    scope = getattr(self, "_prompt_cache_scope", None)
                    if not scope:
                        scope = f"codeaction:{self.model}:{secrets.token_hex(16)}"
                        self._prompt_cache_scope = scope
                    request["extra_headers"] = {"x-grok-conv-id": scope}
                # Sampling defaults are provider-owned.  Never inject the historical scaffold
                # default into a vendor request; only an explicit numeric profile value opts in.
                temperature = profile.get("temperature")
                if temperature is not None:
                    request["temperature"] = float(temperature)
                extra_body = json.loads(json.dumps(profile["extra_body"], sort_keys=True))
                if profile.get("prompt_cache_key_scope") == "episode":
                    scope = getattr(self, "_prompt_cache_scope", None)
                    if not scope:
                        scope = f"codeaction:{self.model}:{secrets.token_hex(16)}"
                        self._prompt_cache_scope = scope
                    # Moonshot exposes this as a top-level Chat Completions field.  ``extra_body``
                    # is how the OpenAI SDK forwards compatible vendor fields into the HTTP body.
                    extra_body["prompt_cache_key"] = scope
                if extra_body:
                    request["extra_body"] = extra_body
                streaming = bool(transport.get("streaming"))
                deadline_clock = getattr(self, "_monotonic", time.monotonic)
                stream_started_at = deadline_clock() if streaming else None
                if streaming:
                    request["stream"] = True
                    request["stream_options"] = {"include_usage": True}
                try:
                    response, response_headers = call_with_raw_headers(
                        self._client.chat.completions, request)
                except Exception as exc:
                    if streaming and _is_timeout_exception(exc):
                        raise ProviderPhaseTimeout(
                            "first_response",
                            float(transport["first_response_timeout_s"]),
                        ) from exc
                    raise
                streamed = (
                    _consume_openai_chat_stream(
                        response,
                        transport,
                        monotonic=deadline_clock,
                        started_at=stream_started_at,
                    )
                    if streaming else None
                )
                # A completion with no choices is a provider-side miss, not a harness bug, and it
                # is transient. Indexing it unguarded raised a bare TypeError that the classifier's
                # catch-all recorded as a NON-RETRYABLE provider_server_error: the 2026-08-13
                # qwen3.7-plus canary lost an otherwise healthy episode on turn 47 that way, after
                # 46 good turns. Naming it here puts it back on the ordinary bounded-retry path.
                choices = getattr(response, "choices", None) or []
                if streamed is None:
                    if not choices:
                        raise EmptyProviderCompletion(
                            f"{self.model} returned no completion choices")
                    choice = choices[0]
                    raw_message = choice.message
                    finish_reason = getattr(choice, "finish_reason", None)
                    raw_usage = getattr(response, "usage", None)
                    provider_request_id = str(getattr(response, "id", "")) or None
                else:
                    raw_message = streamed["message"]
                    finish_reason = streamed["finish_reason"]
                    raw_usage = streamed["usage"]
                    provider_request_id = str(streamed["id"] or "") or None
                message = normalize_message(raw_message)
                stop_reason = normalize_stop_reason(finish_reason, message)
                reasoning = (
                    streamed["reasoning_content"] if streamed is not None
                    else normalize_reasoning_content(
                        raw_message, getattr(self, "_reasoning_fields", None))
                )
                tag = getattr(self, "_inline_thought_tag", None)
                if tag:
                    answer, thought = split_inline_thought(message.get("content"), tag)
                    message["content"] = answer
                    reasoning = thought or reasoning
                ensure_usable_provider_completion(
                    message,
                    stop_reason,
                    provider=self.model,
                    auxiliary_content=reasoning,
                )
                if reasoning and replay_reasoning:
                    shadow = getattr(self, "_reasoning_shadow", None)
                    if shadow is None:
                        shadow = {}
                        self._reasoning_shadow = shadow
                    shadow[_assistant_reasoning_key(message)] = reasoning
                usage = _usage(raw_usage)
                actual_tokens = sum(
                    int(usage.get(key) or 0) for key in ("prompt_tokens", "completion_tokens"))
                limiter.record_success(
                    reservation, actual_tokens if actual_tokens > 0 else None, response_headers,
                    actual_input_tokens=usage.get("prompt_tokens"),
                    actual_output_tokens=usage.get("completion_tokens"),
                    cached_input_tokens=usage.get("cached_tokens"),
                )
                if response_headers:
                    self._emit_provider_telemetry(
                        "provider_response_telemetry", response_headers=response_headers)
                return ModelTurn(
                    message=message,
                    stop_reason=stop_reason,
                    usage=usage,
                    provider_request_id=provider_request_id,
                    reasoning_content=reasoning,
                )
            except Exception as exc:
                failure = classify_provider_exception(exc)
                response_headers = safe_response_headers(exc)
                elapsed_s = max(time.monotonic() - attempt_started, 0.0)
                attempt_limit = min(
                    self._attempts,
                    provider_attempt_limit(self._transport_profile, failure.code.value),
                )
                will_retry = bool(failure.retryable and attempt < attempt_limit)
                delay = retry_delay_seconds(
                    self._transport_profile, failure.code.value, attempt, response_headers,
                    random_fn=getattr(self, "_random", random.random),
                ) if will_retry else 0.0
                shared_cooldown = limiter.record_failure(
                    reservation, failure.code.value, response_headers,
                    minimum_cooldown_s=delay,
                    transport_profile=self._transport_profile,
                )
                delay = max(delay, shared_cooldown) if will_retry else 0.0
                self._emit_attempt_failure(
                    attempt=attempt,
                    failure=failure,
                    elapsed_s=elapsed_s,
                    retry_delay_s=delay,
                    will_retry=will_retry,
                )
                if response_headers:
                    self._emit_provider_telemetry(
                        "provider_error_telemetry", response_headers=response_headers)
                if not will_retry:
                    raise ProviderCallError(failure) from exc
                self.infra_retries += 1
                # The failed attempt's own duration counts too, not just the sleep: a 90 s read
                # timeout against an overloaded endpoint steals exactly as much of the episode as
                # a 90 s backoff would.
                self.infra_wait_s = round(
                    self.infra_wait_s + elapsed_s, 3)
                self.infra_wait_s = round(self.infra_wait_s + float(delay), 3)
                self._sleep(delay)
        raise AssertionError("provider attempt loop ended without a classified result")


def provider_label(model_id) -> str:
    """Transport identity recorded in ``identity.comparison.model``.

    The registry declares the protocol, so a vendor that speaks OpenAI-chat is routed because it
    says so, not because its model id failed to match a prefix rule.
    """
    entry = find_model(model_id)
    return entry.protocol if entry is not None else "openai-compatible"


def build_provider(model=None, *, reasoning_profile=None, **kwargs):
    """Return the adapter that speaks a registered model's native API.

    Model id is the routing key because it is already the recorded identity of the tested unit;
    adding a provider must not add a second, independently-settable knob that could disagree
    with it.  The registry then supplies capabilities and the credential alias, so nothing that
    identity depends on comes from an unaudited environment variable.
    """
    name = str(model or os.environ.get("BENCH_MODEL") or "").strip()
    if not name:
        # Vendor-specific ids are how a per-vendor credential block is written, so they must be
        # able to route.  When several disagree there is no defensible precedence -- picking one
        # silently runs, and bills, a model the caller did not choose -- so require BENCH_MODEL.
        candidates = {
            key: str(os.environ.get(key) or "").strip()
            for key in ("ANTHROPIC_MODEL", "QWEN_MODEL", "MODEL")
        }
        distinct = sorted({value for value in candidates.values() if value})
        if len(distinct) > 1:
            named = ", ".join(
                f"{key}={value}" for key, value in sorted(candidates.items()) if value)
            raise ValueError(
                "ambiguous model routing: the environment names more than one model "
                f"({named}); set BENCH_MODEL to the one this run tests")
        name = distinct[0] if distinct else ""
    entry = resolve_model(name)
    api_key, base_url = credential_for(entry)
    # Every OpenAI-compatible vendor that is not OpenAI needs its own host. Omit it and the SDK
    # silently falls back to api.openai.com, where the key is wrong and the model id is unknown --
    # a confusing 401/404 attributed to the wrong vendor, or worse, a request for a Kimi run
    # leaving for OpenAI. The registry knows which alias is which, so make it an upfront error.
    if (entry.protocol in ("openai-compatible", "openai-responses")
            and entry.credential not in (None, "openai")
            and not (kwargs.get("base_url") or base_url)):
        prefix = str(entry.credential).replace("-", "_").upper()
        raise RegistryError(
            f"model {entry.id!r} speaks {entry.protocol} against a non-OpenAI host, but no "
            f"{prefix}_BASE_URL is set; without it the client would send this run to "
            f"api.openai.com. Add {prefix}_BASE_URL to the credential file.")
    shared = dict(
        model=entry.id,
        provider_model=entry.wire_model(),
        capabilities=kwargs.pop("capabilities", None) or capabilities_for_model(entry.id),
        request_profile=kwargs.pop(
            "request_profile", None)
        or entry.profile(_reasoning_profile_name(reasoning_profile)),
        api_key=kwargs.pop("api_key", None) or api_key,
        base_url=kwargs.pop("base_url", None) or base_url,
        transport_profile=kwargs.pop("transport_profile", None) or entry.transport_profile(),
        rate_limit_policy=kwargs.pop("rate_limit_policy", None)
        or rate_limit_policy_for(None, entry.credential, entry.id),
        **kwargs,
    )
    from codeaction.providers.anthropic_adapter import AnthropicAdapter
    from codeaction.providers.openai_responses_adapter import OpenAIResponsesAdapter
    from codeaction.providers.gemini_adapter import GeminiGenerateContentAdapter
    from codeaction.extensions import entrypoint
    factories = {"anthropic": AnthropicAdapter, "openai-responses": OpenAIResponsesAdapter,
                 "google-generate-content": GeminiGenerateContentAdapter,
                 "openai-compatible": ModelAdapter}
    factory = entrypoint("provider", entry.protocol, builtin=factories.get(entry.protocol))
    if factory is None:
        raise RegistryError(f"no provider factory for {entry.protocol}")
    return factory(**shared)



class ScriptedModel:
    """Zero-cost faux provider returning the same normalized turns as a real adapter."""

    def __init__(self, script, *, context_window_tokens=1_000_000,
                 max_output_tokens=8192):
        self._script = list(script)
        self._index = 0
        self.infra_retries = 0
        # Seconds spent on FAILED provider attempts and the sleeps between them. Credited
        # back to the episode's wall budget: it is the vendor's outage, not the model's
        # deliberation, and charging it scores an infra fault as a model failure.
        self.infra_wait_s = 0.0
        self.model = "scripted-model"
        self._capabilities = ModelCapabilities(
            context_window_tokens=int(context_window_tokens),
            max_output_tokens=int(max_output_tokens),
            token_estimator_id=TOKEN_ESTIMATOR_UTF8_BYTES_V1,
            supports_tools=True,
            supports_images=True,
            model_seed_support="supported",
        )

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def request_profile(self) -> Dict[str, object]:
        return provider_request_profile(self.model, scripted=True)

    def step(self, messages, tools, requested_output_tokens) -> ModelTurn:
        resolve_output_tokens(self._capabilities, requested_output_tokens)
        if self._index >= len(self._script):
            entry = [("done", {"report": "script exhausted"})]
        else:
            entry = self._script[self._index]
            if isinstance(entry, tuple):
                entry = [entry]
        self._index += 1
        tool_calls = [
            {
                "id": f"call_{self._index:03d}_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for index, (name, arguments) in enumerate(entry)
        ]
        return ModelTurn(
            message={"content": None, "tool_calls": tool_calls},
            stop_reason="tool_calls",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            provider_request_id=f"scripted-{self._index:03d}",
            reasoning_content=None,
        )


_LEGACY_TEST_CAPABILITIES = ModelCapabilities(
    context_window_tokens=1_000_000,
    max_output_tokens=8192,
    token_estimator_id=TOKEN_ESTIMATOR_UTF8_BYTES_V1,
    supports_tools=True,
    supports_images=True,
    model_seed_support="unsupported",
)


def capabilities_of(provider: Any) -> ModelCapabilities:
    method = getattr(provider, "capabilities", None)
    if callable(method):
        value = method()
        if not isinstance(value, ModelCapabilities):
            raise TypeError("provider capabilities() must return ModelCapabilities")
        return value
    # Only retained so Batch-0 faux providers keep replaying. Real provider adapters must implement
    # the capability boundary and will otherwise be recorded as this explicit compatibility mode.
    return _LEGACY_TEST_CAPABILITIES


def invoke_provider(
    provider: Any,
    messages: List[Dict[str, object]],
    tools: List[Dict[str, object]],
    requested_output_tokens: int,
) -> ModelTurn:
    step = provider.step
    parameters = inspect.signature(step).parameters.values()
    accepts_output_limit = any(
        item.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for item in parameters
    ) or len(list(parameters)) >= 3
    value = (
        step(messages, tools, requested_output_tokens)
        if accepts_output_limit
        else step(messages, tools)
    )
    if isinstance(value, ModelTurn):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        message, usage = value
        normalized = normalize_message(message)
        return ModelTurn(
            message=normalized,
            stop_reason=normalize_stop_reason(None, normalized),
            usage=_usage(usage),
            provider_request_id=None,
            reasoning_content=normalize_reasoning_content(message),
        )
    raise TypeError("provider step() must return ModelTurn")
