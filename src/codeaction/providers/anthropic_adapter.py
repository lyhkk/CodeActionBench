"""Anthropic Messages-API provider adapter for the ``codeaction-reference`` scaffold.

The reference loop owns one canonical wire format: OpenAI-chat messages (``system``/``user``/
``assistant``/``tool``, ``tool_calls``/``tool_call_id``, ``image_url`` data URIs) plus OpenAI
function-tool definitions.  ``ContextManager``, the transcript schema, and the identity hashes are
all defined over that shape, so this adapter translates at the provider boundary in both
directions rather than changing what the loop stores.

Three Anthropic-specific facts drive the translation:

1. ``tool_result`` blocks live in a *user* turn and every result for one assistant turn must ride
   in a **single** user message.  The reference loop appends one ``role="tool"`` message per call
   and interleaves image ``role="user"`` messages between them, so the translator regroups a whole
   post-assistant run into one user turn (tool results first, then the other blocks in order).
2. Thinking blocks are signed and must be replayed **unmodified** on the assistant turns of a
   continued conversation.  The canonical message list deliberately does not carry provider-native
   reasoning, so this adapter keeps a provider-native shadow keyed by the assistant turn's tool-use
   ids (or, for a text-only turn, a digest of its text) and re-attaches the blocks on translation.
3. Sampling parameters are rejected on current Claude models, so none are sent; the reasoning
   control is ``output_config.effort`` rather than a token budget.  Both are carried in the shared
   ``request_profile()["extra_body"]`` seam, which keeps them recorded in the scaffold hash and
   keeps this adapter working across Anthropic SDK versions that have not typed them yet.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from codeaction.contracts.failures import (
    FailureCode,
    ProviderCallError,
    classify_provider_exception,
    ensure_usable_provider_completion,
)
from codeaction.providers.model_adapter import (
    CACHE_CHECKPOINT_FIELD,
    ModelCapabilities,
    ModelTurn,
    capabilities_for_model,
    normalize_stop_reason,
    provider_request_profile,
    resolve_output_tokens,
)
from codeaction.providers.model_registry import find_model
from codeaction.providers.provider_runtime import (
    SharedRateLimiter,
    call_with_raw_headers,
    normalize_rate_limit_policy,
    normalize_transport_profile,
    provider_attempt_limit,
    provider_sdk_timeout,
    retry_delay_seconds,
    safe_response_headers,
)


ANTHROPIC_ADAPTER_VERSION = "1.1.0"
DEFAULT_MEDIA_TYPE = "image/png"
# Two breakpoints: the frozen tools+system prefix (tools render before system, so a marker on the
# last system block caches both), and up to three stable historical observation records. Four is
# the API maximum. Retaining several rolling checkpoints matters when a later turn appends text:
# the provider can read the older exact prefix and write the newly extended prefix in one call.
SYSTEM_CACHE_BREAKPOINT = True
TAIL_CACHE_BREAKPOINT = True

_THINKING_TYPES = ("thinking", "redacted_thinking")


# --------------------------------------------------------------------------------------------
# canonical (OpenAI-chat) -> Anthropic
# --------------------------------------------------------------------------------------------

def _block_to_dict(value: Any) -> Dict[str, Any]:
    """Provider content blocks arrive as SDK models; replay needs plain JSON-safe dicts."""
    if isinstance(value, Mapping):
        return dict(value)
    for attribute in ("model_dump", "dict"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return {k: v for k, v in method().items() if v is not None}
            except Exception:
                continue
    raise TypeError(f"cannot serialize provider content block: {type(value).__name__}")


def _image_source(url: str) -> Dict[str, object]:
    text = str(url or "")
    if not text.startswith("data:"):
        raise ValueError("reference image blocks must be base64 data URIs")
    header, _, payload = text.partition(",")
    media_type = header[len("data:"):].split(";")[0] or DEFAULT_MEDIA_TYPE
    # Validate here rather than at the endpoint: a malformed payload is a harness bug, and the
    # provider would otherwise report it as an opaque 400 mid-episode.
    base64.b64decode(payload, validate=True)
    return {"type": "base64", "media_type": media_type, "data": payload}


def _content_blocks(content: Any) -> List[Dict[str, object]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: List[Dict[str, object]] = []
    for item in content:
        if not isinstance(item, Mapping):
            blocks.append({"type": "text", "text": str(item)})
            continue
        kind = item.get("type")
        if kind == "text":
            text = str(item.get("text") or "")
            if text:
                block = {"type": "text", "text": text}
                if item.get(CACHE_CHECKPOINT_FIELD) is True:
                    block[CACHE_CHECKPOINT_FIELD] = True
                blocks.append(block)
        elif kind == "image_url":
            url = (item.get("image_url") or {}).get("url")
            blocks.append({"type": "image", "source": _image_source(url)})
        else:
            raise ValueError(f"unsupported canonical content block: {kind!r}")
    return blocks


def assistant_shadow_key(message: Mapping[str, Any]) -> str:
    """Stable identity for an assistant turn, used to re-attach its signed thinking blocks.

    Tool-use ids are provider-assigned and unique, so they identify the turn exactly.  A turn with
    no tool calls (the ``no_tool_call`` recovery path) is keyed by a digest of its text instead.
    """
    calls = message.get("tool_calls") or []
    ids = [str((call or {}).get("id") or "") for call in calls]
    if any(ids):
        return "tools:" + "|".join(ids)
    digest = hashlib.sha256(str(message.get("content") or "").encode("utf-8")).hexdigest()
    return "text:" + digest[:32]


def _assistant_content(
    message: Mapping[str, Any],
    shadow: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, object]]:
    # Thinking blocks must lead the assistant turn and must be byte-identical to what the model
    # returned; anything else invalidates their signature.
    blocks: List[Dict[str, object]] = [
        dict(item) for item in shadow.get(assistant_shadow_key(message), ())
    ]
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        blocks.extend(_content_blocks(text))
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        raw = function.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) and raw else {}
        except json.JSONDecodeError:
            # A truncated tool call is a scoreable model outcome upstream; replaying it as an
            # empty input keeps the turn well-formed instead of failing the whole request.
            arguments = {}
        if not isinstance(arguments, Mapping):
            arguments = {}
        blocks.append({
            "type": "tool_use",
            "id": str(call.get("id") or ""),
            "name": str(function.get("name") or ""),
            "input": dict(arguments),
        })
    return blocks


def translate_messages(
    messages: Sequence[Mapping[str, Any]],
    shadow: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return ``(system_blocks, anthropic_messages)`` for one canonical message list."""
    shadow = shadow or {}
    system_blocks: List[Dict[str, object]] = []
    out: List[Dict[str, object]] = []
    tool_results: List[Dict[str, object]] = []
    other_blocks: List[Dict[str, object]] = []

    def flush_user() -> None:
        nonlocal tool_results, other_blocks
        blocks = tool_results + other_blocks
        tool_results, other_blocks = [], []
        if blocks:
            out.append({"role": "user", "content": blocks})

    def carry_message_checkpoint(message: Mapping[str, Any], blocks: List[Dict[str, object]]):
        """Realize a message-level checkpoint on the last block that message contributed.

        The context layer marks a turn boundary at the message level whenever the boundary has no
        block list to hold the marker -- an ordinary tool result. Anthropic accepts
        `cache_control` on any content block, `tool_result` and `tool_use` included, so the marker
        lands exactly at the end of that turn. An assistant turn that contributes no block at all
        (the `no_tool_call` recovery path) drops the marker with its message; the remaining
        boundaries still cover the request.
        """
        if message.get(CACHE_CHECKPOINT_FIELD) is True and blocks:
            blocks[-1][CACHE_CHECKPOINT_FIELD] = True

    for message in messages:
        role = message.get("role")
        if role == "system":
            blocks = _content_blocks(message.get("content"))
            carry_message_checkpoint(message, blocks)
            system_blocks.extend(blocks)
        elif role == "user":
            blocks = _content_blocks(message.get("content"))
            carry_message_checkpoint(message, blocks)
            other_blocks.extend(blocks)
        elif role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": str(message.get("tool_call_id") or ""),
                "content": [{"type": "text", "text": str(message.get("content") or "")}],
            }
            carry_message_checkpoint(message, [result])
            tool_results.append(result)
        elif role == "assistant":
            blocks = _assistant_content(message, shadow)
            # An empty assistant turn carries no information and is rejected by the API.  Dropping
            # it without flushing merges the surrounding user content into one turn, which is
            # lossless: only a turn that emitted tool calls can be followed by tool results, and
            # such a turn is never empty.
            if blocks:
                carry_message_checkpoint(message, blocks)
                flush_user()
                out.append({"role": "assistant", "content": blocks})
        else:
            raise ValueError(f"unsupported canonical role: {role!r}")
    flush_user()
    return system_blocks, out


def apply_cache_breakpoints(
    system_blocks: List[Dict[str, object]],
    messages: List[Dict[str, object]],
) -> None:
    """Translate internal rolling checkpoints into Anthropic's four-marker budget."""
    if SYSTEM_CACHE_BREAKPOINT and system_blocks:
        system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
    if TAIL_CACHE_BREAKPOINT and messages:
        flattened = []
        candidates = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                flattened.append(block)
                if block.pop(CACHE_CHECKPOINT_FIELD, False) is True:
                    candidates.append(block)
        # One marker is reserved for tools+system, leaving three for the newest turn boundaries.
        # Since policy 2.2.0 the context layer mints a candidate at the end of every turn, so the
        # fallback is reached only before the first assistant turn (and by direct callers that
        # send no candidates). It skips image blocks deliberately: those live in the request-local
        # visual tail, whose bytes change every turn, so a write there could never be read back.
        selected = candidates[-3:]
        if not selected:
            tail_text = [block for block in flattened if block.get("type") != "image"]
            selected = tail_text[-1:]
        for block in selected:
            block["cache_control"] = {"type": "ephemeral"}


def translate_tools(tools: Sequence[Mapping[str, Any]]) -> List[Dict[str, object]]:
    out = []
    for item in tools or ():
        function = (item or {}).get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            raise ValueError("anthropic tool translation requires OpenAI function-tool shape")
        out.append({
            "name": name,
            "description": str(function.get("description") or ""),
            "input_schema": dict(
                function.get("parameters") or {"type": "object", "properties": {}}),
        })
    return out


# --------------------------------------------------------------------------------------------
# Anthropic -> canonical (OpenAI-chat)
# --------------------------------------------------------------------------------------------

def translate_response(response: Any) -> Tuple[Dict[str, object], Optional[str], List[Dict[str, Any]]]:
    """Return ``(canonical_message, reasoning_content, thinking_blocks)``."""
    texts: List[str] = []
    tool_calls: List[Dict[str, object]] = []
    thinking_blocks: List[Dict[str, Any]] = []
    reasoning: List[str] = []
    for block in getattr(response, "content", None) or ():
        kind = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", None)
        if kind in _THINKING_TYPES:
            thinking_blocks.append(_block_to_dict(block))
            text = (block.get("thinking") if isinstance(block, Mapping)
                    else getattr(block, "thinking", None))
            if text:
                reasoning.append(str(text))
        elif kind == "text":
            text = block.get("text") if isinstance(block, Mapping) else getattr(block, "text", "")
            if text:
                texts.append(str(text))
        elif kind == "tool_use":
            raw = _block_to_dict(block)
            tool_calls.append({
                "id": str(raw.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(raw.get("name") or ""),
                    "arguments": json.dumps(raw.get("input") or {}, sort_keys=True),
                },
            })
    message: Dict[str, object] = {"content": "\n".join(texts) if texts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, ("\n".join(reasoning) if reasoning else None), thinking_blocks


def translate_stop_details(details: Any) -> Optional[Dict[str, object]]:
    """Normalize the refusal detail object; ``None`` for every non-refusal stop reason.

    Only the typed fields are kept.  ``explanation`` is provider prose and is bounded here so a
    long refusal cannot dominate a transcript record.
    """
    if details is None:
        return None

    def read(name: str) -> Optional[str]:
        value = (details.get(name) if isinstance(details, Mapping)
                 else getattr(details, name, None))
        return str(value) if value is not None else None

    out: Dict[str, object] = {}
    for field in ("type", "category"):
        value = read(field)
        if value is not None:
            out[field] = value
    explanation = read("explanation")
    if explanation is not None:
        out["explanation"] = explanation[:512]
    return out or None


def translate_usage(usage: Any) -> Dict[str, Optional[int]]:
    def read(name: str) -> Optional[int]:
        value = (usage.get(name) if isinstance(usage, Mapping)
                 else getattr(usage, name, None))
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    out: Dict[str, Optional[int]] = {
        "prompt_tokens": read("input_tokens"),
        "completion_tokens": read("output_tokens"),
    }
    cached = read("cache_read_input_tokens")
    if cached is not None:
        out["cached_tokens"] = cached
    created = read("cache_creation_input_tokens")
    if created is not None:
        out["cache_creation_tokens"] = created
    return out


# --------------------------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------------------------

class AnthropicAdapter:
    """Anthropic Messages-API adapter with the same retry and capability contract as ModelAdapter."""

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
        client=None,
    ):
        # Identity, capabilities, and the credential alias come from `build_provider` via the
        # registry and the credential file; direct construction is the unit-test seam.
        self.model = model
        # Resolved lazily: a test seam may reassign `.model` after construction, and the
        # wire name must follow it unless a vendor name was declared explicitly.
        self._provider_model_override = str(provider_model) if provider_model else None
        if capabilities is None:
            capabilities = capabilities_for_model(self.model)
        self._capabilities = capabilities
        self._request_profile = request_profile or provider_request_profile(
            self.model, reasoning_profile=reasoning_profile)
        entry = find_model(self.model)
        self._transport_profile = normalize_transport_profile(
            transport_profile or (entry.transport_profile() if entry is not None else None))
        effective_timeout = provider_sdk_timeout(self._transport_profile, timeout)
        effective_attempts = self._transport_profile["attempts"] if attempts is None else int(attempts)
        if client is not None:
            self._client = client
        else:
            from anthropic import Anthropic
            options = {"api_key": api_key, "timeout": effective_timeout, "max_retries": 0}
            if base_url:
                options["base_url"] = base_url
            self._client = Anthropic(**options)
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
        # Provider-native reasoning is kept out of the canonical message list, so the signed
        # blocks needed to continue a turn are held here instead, keyed by assistant turn.
        self._thinking_shadow: Dict[str, List[Dict[str, Any]]] = {}

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
        # A branch rather than `getattr(..., default)`: the default is evaluated eagerly on every
        # call, and its fallback rung raises for a model that declares no `disabled` profile
        # (Fable reasons unconditionally), which made such a model unusable even when configured.
        value = self._request_profile if hasattr(self, "_request_profile") \
            else provider_request_profile(self.model)
        return json.loads(json.dumps(value, sort_keys=True))

    def build_request(self, messages, tools, effective_output_tokens) -> Dict[str, object]:
        system_blocks, translated = translate_messages(messages, self._thinking_shadow)
        apply_cache_breakpoints(system_blocks, translated)
        request: Dict[str, object] = {
            "model": self._wire_model,
            "max_tokens": int(effective_output_tokens),
            "messages": translated,
        }
        if system_blocks:
            request["system"] = system_blocks
        if tools:
            request["tools"] = translate_tools(tools)
        extra_body = self.request_profile()["extra_body"]
        if extra_body:
            request["extra_body"] = dict(extra_body)
        return request

    def step(self, messages, tools, requested_output_tokens) -> ModelTurn:
        effective = resolve_output_tokens(self._capabilities, requested_output_tokens)
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
                limiter = SharedRateLimiter(rate_policy, self.model, None, sleep_fn=self._sleep)
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
                    quota_group=rate_policy.get("quota_group"))
            attempt_started = time.monotonic()
            try:
                response, response_headers = call_with_raw_headers(
                    self._client.messages,
                    self.build_request(messages, tools, effective))
                message, reasoning, thinking_blocks = translate_response(response)
                stop_reason = normalize_stop_reason(
                    getattr(response, "stop_reason", None), message)
                ensure_usable_provider_completion(
                    message,
                    stop_reason,
                    provider=self.model,
                    auxiliary_content=reasoning,
                )
                if thinking_blocks:
                    self._thinking_shadow[assistant_shadow_key(message)] = thinking_blocks
                usage = translate_usage(getattr(response, "usage", None))
                actual_tokens = sum(
                    int(usage.get(key) or 0) for key in ("prompt_tokens", "completion_tokens"))
                # Anthropic exposes cache-creation tokens separately. They belong to an input
                # limit declared as excluding cache reads, while cache-read tokens do not.
                quota_input_tokens = int(usage.get("prompt_tokens") or 0) \
                    + int(usage.get("cache_creation_tokens") or 0)
                limiter.record_success(
                    reservation, actual_tokens if actual_tokens > 0 else None, response_headers,
                    actual_input_tokens=quota_input_tokens,
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
                    provider_request_id=(str(getattr(response, "id", "")) or None),
                    reasoning_content=reasoning,
                    stop_details=translate_stop_details(
                        getattr(response, "stop_details", None)),
                )
            except Exception as exc:
                failure = classify_provider_exception(exc)
                response_headers = safe_response_headers(exc)
                elapsed_s = max(time.monotonic() - attempt_started, 0.0)
                attempt_limit = min(
                    self._attempts, provider_attempt_limit(transport, failure.code.value))
                will_retry = bool(failure.retryable and attempt < attempt_limit)
                delay = retry_delay_seconds(
                    transport, failure.code.value, attempt, response_headers,
                    random_fn=getattr(self, "_random", random.random),
                ) if will_retry else 0.0
                shared_cooldown = limiter.record_failure(
                    reservation, failure.code.value, response_headers,
                    minimum_cooldown_s=delay, transport_profile=transport)
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


__all__ = [
    "ANTHROPIC_ADAPTER_VERSION",
    "AnthropicAdapter",
    "apply_cache_breakpoints",
    "assistant_shadow_key",
    "translate_messages",
    "translate_response",
    "translate_stop_details",
    "translate_tools",
    "translate_usage",
]
