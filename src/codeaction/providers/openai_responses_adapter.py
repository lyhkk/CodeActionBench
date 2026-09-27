"""OpenAI Responses-API provider adapter for the ``codeaction-reference`` scaffold.

Why a third protocol exists at all: on Chat Completions the GPT reasoning models think and bill for
it, but return none of the text at any effort — OpenAI exposes reasoning summaries **only** on the
Responses API (verified against OpenAI's reasoning guide, 2026-08-08). A benchmark whose whole
premise is comparing how models reason cannot have one vendor's reasoning column structurally
empty, so the GPT row speaks Responses instead of chat.

The reference loop keeps one canonical wire format — OpenAI-chat messages plus OpenAI function-tool
definitions — and this file translates at the boundary in both directions, exactly as
``anthropic_adapter`` does. Four Responses-specific facts drive the translation:

1. **There is no ``messages`` array.** Input is a flat list of typed items: ``message`` items with
   typed content parts, ``function_call`` items, and ``function_call_output`` items. A tool result
   is a top-level item keyed by ``call_id``, not a ``role="tool"`` message.
2. **Content part names differ by role.** User text is ``input_text`` and user images are
   ``input_image``; assistant text is ``output_text``. Sending the wrong one is a 400.
3. **Reasoning items must be replayed.** A turn's ``reasoning`` item has to ride back with the
   assistant turn that followed it or the model loses its own chain across tool calls. The
   canonical message list deliberately does not carry provider-native reasoning, so — as on the
   Anthropic path — this adapter keeps a shadow keyed by the assistant turn's call ids and
   re-attaches the items on translation.
4. **Tool definitions are flat.** ``{"type": "function", "name", "description", "parameters"}``,
   not the Chat Completions ``{"type": "function", "function": {...}}`` nesting.
"""
from __future__ import annotations

import hashlib
import json
import random
import secrets
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from codeaction.contracts.failures import (
    FailureCode,
    ProviderCallError,
    classify_provider_exception,
    default_failure,
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
    provider_sdk_timeout,
    retry_delay_seconds,
    safe_response_headers,
)


OPENAI_RESPONSES_ADAPTER_VERSION = "1.2.0"


class MissingEncryptedReasoningContent(RuntimeError):
    """A tool-producing response cannot be replayed in stateless mode."""


def _canonical_item_sha256(item: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reasoning_item_hashes(items: Sequence[Mapping[str, Any]]) -> List[str]:
    return [_canonical_item_sha256(item) for item in items]


# --------------------------------------------------------------------------------------------
# canonical (OpenAI-chat) -> Responses input items
# --------------------------------------------------------------------------------------------

def _item_to_dict(value: Any) -> Dict[str, Any]:
    """Response items arrive as SDK models; replay needs plain JSON-safe dicts."""
    if isinstance(value, Mapping):
        return dict(value)
    for attribute in ("model_dump", "dict"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return {k: v for k, v in method(exclude_none=True).items()}
            except TypeError:
                return {k: v for k, v in method().items() if v is not None}
            except Exception:
                continue
    raise TypeError(f"cannot serialize provider output item: {type(value).__name__}")


def _input_parts(content: Any) -> List[Dict[str, object]]:
    """Translate canonical user/system content into Responses *input* parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}] if content else []
    parts: List[Dict[str, object]] = []
    for item in content:
        if not isinstance(item, Mapping):
            parts.append({"type": "input_text", "text": str(item)})
            continue
        kind = item.get("type")
        if kind == "text":
            text = str(item.get("text") or "")
            if text:
                part = {"type": "input_text", "text": text}
                if item.get(CACHE_CHECKPOINT_FIELD) is True:
                    part[CACHE_CHECKPOINT_FIELD] = True
                parts.append(part)
        elif kind == "image_url":
            url = (item.get("image_url") or {}).get("url")
            if not str(url or "").startswith("data:"):
                raise ValueError("reference image blocks must be base64 data URIs")
            parts.append({"type": "input_image", "image_url": str(url)})
        else:
            raise ValueError(f"unsupported canonical content block: {kind!r}")
    return parts


def assistant_shadow_key(message: Mapping[str, Any]) -> str:
    """Stable identity for an assistant turn, used to re-attach its reasoning items.

    Tool-call ids are provider-assigned and unique, so they identify the turn exactly. A turn with
    no tool calls is keyed by a digest of its text instead.
    """
    calls = message.get("tool_calls") or []
    ids = [str((call or {}).get("id") or "") for call in calls]
    if any(ids):
        return "tools:" + "|".join(ids)
    digest = hashlib.sha256(str(message.get("content") or "").encode("utf-8")).hexdigest()
    return "text:" + digest[:32]


def translate_messages(
    messages: Sequence[Mapping[str, Any]],
    shadow: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return ``(instructions_texts, input_items)`` for one canonical message list."""
    shadow = shadow or {}
    instructions: List[Dict[str, object]] = []
    out: List[Dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            instructions.extend(_input_parts(message.get("content")))
        elif role == "user":
            parts = _input_parts(message.get("content"))
            if parts:
                out.append({"role": "user", "content": parts})
        elif role == "tool":
            # A tool result is its own top-level item on this protocol, not a message.
            out.append({
                "type": "function_call_output",
                "call_id": str(message.get("tool_call_id") or ""),
                "output": str(message.get("content") or ""),
            })
        elif role == "assistant":
            # Reasoning items must precede the calls they produced, byte-identical to what the
            # model returned; anything else and the model cannot follow its own chain.
            out.extend(dict(item) for item in shadow.get(assistant_shadow_key(message), ()))
            text = message.get("content")
            if isinstance(text, str) and text:
                out.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
            for call in message.get("tool_calls") or []:
                function = (call or {}).get("function") or {}
                out.append({
                    "type": "function_call",
                    "call_id": str(call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or "{}"),
                })
        else:
            raise ValueError(f"unsupported canonical role: {role!r}")
    return instructions, out


def apply_cache_breakpoints(items: List[Dict[str, object]]) -> None:
    """Translate the latest rolling candidates into GPT-5.6 explicit breakpoints.

    Responses accepts breakpoints only on input content blocks. Historical tool outputs are
    top-level items, so the context layer deliberately places candidates on stable user-visible
    observation records instead of asking this adapter to mark an unsupported item.
    """
    candidates: List[Dict[str, object]] = []
    for item in items:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.pop(CACHE_CHECKPOINT_FIELD, False) is True:
                if block.get("type") != "input_text":
                    raise ValueError("OpenAI cache checkpoint must translate to input_text")
                candidates.append(block)
    for block in candidates[-4:]:
        block["prompt_cache_breakpoint"] = {"mode": "explicit"}


def translate_tools(tools: Sequence[Mapping[str, Any]]) -> List[Dict[str, object]]:
    """Chat Completions nests the schema under ``function``; Responses does not."""
    out = []
    for item in tools or ():
        function = (item or {}).get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            raise ValueError("responses tool translation requires OpenAI function-tool shape")
        out.append({
            "type": "function",
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": dict(
                function.get("parameters") or {"type": "object", "properties": {}}),
        })
    return out


# --------------------------------------------------------------------------------------------
# Responses -> canonical (OpenAI-chat)
# --------------------------------------------------------------------------------------------

def translate_response(
    response: Any,
) -> Tuple[Dict[str, object], Optional[str], List[Dict[str, Any]]]:
    """Return ``(canonical_message, reasoning_content, reasoning_items)``."""
    texts: List[str] = []
    tool_calls: List[Dict[str, object]] = []
    reasoning_items: List[Dict[str, Any]] = []
    summaries: List[str] = []
    for item in getattr(response, "output", None) or ():
        raw = _item_to_dict(item)
        kind = raw.get("type")
        if kind == "reasoning":
            reasoning_items.append(raw)
            # The raw chain of thought is never returned; `summary` is what a caller can record,
            # and it is a list of parts rather than a string.
            for part in raw.get("summary") or ():
                text = (part.get("text") if isinstance(part, Mapping)
                        else getattr(part, "text", None))
                if text:
                    summaries.append(str(text))
        elif kind == "message":
            for part in raw.get("content") or ():
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "output_text" and part.get("text"):
                    texts.append(str(part["text"]))
        elif kind == "function_call":
            tool_calls.append({
                "id": str(raw.get("call_id") or raw.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(raw.get("name") or ""),
                    "arguments": str(raw.get("arguments") or "{}"),
                },
            })
        else:
            raise ValueError(f"unsupported Responses output item: {kind!r}")
    message: Dict[str, object] = {"content": "\n".join(texts) if texts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, ("\n".join(summaries) if summaries else None), reasoning_items


def translate_stop_reason(response: Any, message: Mapping[str, Any]) -> str:
    """Map the Responses status/incomplete reason onto the loop's stop-reason vocabulary."""
    status = str(getattr(response, "status", "") or "")
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = (details.get("reason") if isinstance(details, Mapping)
                  else getattr(details, "reason", None))
        if str(reason or "") == "max_output_tokens":
            return "length"
        return "incomplete"
    if message.get("tool_calls"):
        return "tool_calls"
    return normalize_stop_reason(None, message)


def translate_usage(usage: Any) -> Dict[str, Optional[int]]:
    def read(source: Any, name: str) -> Optional[int]:
        value = (source.get(name) if isinstance(source, Mapping)
                 else getattr(source, name, None))
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    out: Dict[str, Optional[int]] = {
        "prompt_tokens": read(usage, "input_tokens"),
        "completion_tokens": read(usage, "output_tokens"),
    }
    input_details = (usage.get("input_tokens_details") if isinstance(usage, Mapping)
                     else getattr(usage, "input_tokens_details", None))
    cached = read(input_details, "cached_tokens") if input_details is not None else None
    if cached is not None:
        out["cached_tokens"] = cached
    created = read(input_details, "cache_write_tokens") if input_details is not None else None
    if created is not None:
        out["cache_creation_tokens"] = created
    output_details = (usage.get("output_tokens_details") if isinstance(usage, Mapping)
                      else getattr(usage, "output_tokens_details", None))
    reasoning = read(output_details, "reasoning_tokens") if output_details is not None else None
    if reasoning is not None:
        out["reasoning_tokens"] = reasoning
    return out


PROMPT_CACHE_KEY_MAX_CHARS = 64


def _prompt_cache_key(model: str, nonce: str) -> str:
    """Episode-scoped cache key that cannot exceed the provider's string limit.

    The key was `f"codeaction:{model}:{nonce}"` with a 32-char nonce, so any registry id longer
    than 23 characters produced a 65+ character key and the provider rejected the FIRST request
    of the episode with `400 string_above_max_length` -- before a single token was billed, and
    with no hint that the entry's NAME was the problem. `gpt-5.6-out8192-noreplay` is 24. The
    nonce carries the isolation and must stay whole; the model segment is only there to make a
    key readable in provider logs, so that is what gives way.
    """
    prefix, nonce = "codeaction:", str(nonce)
    room = PROMPT_CACHE_KEY_MAX_CHARS - len(prefix) - len(nonce) - 1
    if room < 0:
        raise ValueError("prompt cache nonce leaves no room for the key prefix")
    return f"{prefix}{str(model)[:room]}:{nonce}"


# --------------------------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------------------------

class OpenAIResponsesAdapter:
    """Responses-API adapter with the same retry and capability contract as ModelAdapter."""

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
        rate_limit_state_dir=None,
        reasoning_profile: Optional[str] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
        event_sink: Optional[Callable[[Dict[str, object]], None]] = None,
        client=None,
    ):
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
            from openai import OpenAI
            options = {"api_key": api_key, "timeout": effective_timeout, "max_retries": 0}
            if base_url:
                options["base_url"] = base_url
            self._client = OpenAI(**options)
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
        # Per-adapter means per episode in the reference runner. A random scope prevents one
        # benchmark episode from inheriting another episode's provider-side routing state while
        # remaining stable across every turn and infrastructure retry in this episode.
        self._prompt_cache_key = _prompt_cache_key(self.model, secrets.token_hex(16))
        # Provider-native reasoning is kept out of the canonical message list, so the items needed
        # to continue a turn are held here instead, keyed by assistant turn.
        self._reasoning_shadow: Dict[str, List[Dict[str, Any]]] = {}

    def set_event_sink(
        self, sink: Optional[Callable[[Dict[str, object]], None]],
    ) -> None:
        """Attach the transcript-owned sink for safe provider-attempt telemetry."""
        self._event_sink = sink

    def _emit_attempt_failure(
        self,
        *,
        attempt: int,
        attempts_max: Optional[int] = None,
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
            "provider_attempts_max": int(attempts_max or self._attempts),
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
        # call, and its fallback rung raises for any model that declares no `disabled` profile.
        value = self._request_profile if hasattr(self, "_request_profile") \
            else provider_request_profile(self.model)
        return json.loads(json.dumps(value, sort_keys=True))

    def build_request(self, messages, tools, effective_output_tokens) -> Dict[str, object]:
        instructions, items = translate_messages(messages, self._reasoning_shadow)
        apply_cache_breakpoints(items)
        request: Dict[str, object] = {
            "model": self._wire_model,
            "max_output_tokens": int(effective_output_tokens),
            "input": items,
            # Stateless Responses continuity requires the opaque reasoning payload, not merely
            # the public summary.  Asking explicitly makes its presence auditable across SDK and
            # endpoint revisions.
            "include": ["reasoning.encrypted_content"],
            # Reasoning items are only usable on a later turn if the server keeps no state of its
            # own; this loop resends the whole conversation each turn, so server-side chaining is
            # explicitly off and the shadow is the single source of continuity.
            "store": False,
            "prompt_cache_key": self._prompt_cache_key,
            # The endpoint supports this field before the pinned OpenAI SDK has a typed keyword
            # for it. ``extra_body`` is the SDK's documented forward-compatibility path.
            "extra_body": {
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            },
        }
        if instructions:
            request["instructions"] = "\n".join(
                str(part.get("text") or "") for part in instructions)
        if tools:
            request["tools"] = translate_tools(tools)
        profile = self.request_profile()
        temperature = profile.get("temperature")
        if temperature is not None:
            request["temperature"] = float(temperature)
        extra_body = profile["extra_body"]
        if extra_body:
            request.update(json.loads(json.dumps(extra_body, sort_keys=True)))
        return request

    def step(self, messages, tools, requested_output_tokens) -> ModelTurn:
        effective = resolve_output_tokens(self._capabilities, requested_output_tokens)
        self.last_effective_output_tokens = effective
        transport = getattr(self, "_transport_profile", None) or normalize_transport_profile()
        self._transport_profile = transport
        rate_policy = getattr(self, "_rate_limit_policy", None) or normalize_rate_limit_policy(
            None, quota_group=None)
        self._rate_limit_policy = rate_policy
        attempt = 0
        ordinary_failures = 0
        missing_encrypted_retries = 0
        # A response that omits requested encrypted reasoning gets one dedicated retry.  It does
        # not silently consume the ordinary transport retry budget, but still has a hard ceiling.
        maximum_attempts = self._attempts + 1
        while attempt < maximum_attempts:
            attempt += 1
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
                request = self.build_request(messages, tools, effective)
                # The request may replay several retained turns.  Evidence for continuity is
                # attached to the immediately preceding assistant turn so the next-turn check is
                # exact rather than conflating it with older retained history.
                replayed_items: Sequence[Mapping[str, Any]] = ()
                for prior in reversed(messages):
                    if prior.get("role") != "assistant":
                        continue
                    replayed_items = self._reasoning_shadow.get(
                        assistant_shadow_key(prior), ())
                    break
                replayed_hashes = _reasoning_item_hashes(replayed_items)
                response, response_headers = call_with_raw_headers(
                    self._client.responses, request)
                message, reasoning, reasoning_items = translate_response(response)
                usage = translate_usage(getattr(response, "usage", None))
                stop_reason = translate_stop_reason(response, message)
                ensure_usable_provider_completion(
                    message,
                    stop_reason,
                    provider=self.model,
                    auxiliary_content=reasoning,
                )
                profile = self.request_profile()
                evidence_protocol = profile.get("reasoning_replay_evidence")
                if evidence_protocol == "openai-encrypted-v1" and message.get("tool_calls"):
                    encrypted_complete = all(
                        isinstance(item.get("encrypted_content"), str)
                        and bool(item["encrypted_content"].strip())
                        for item in reasoning_items
                    )
                    reasoning_tokens = usage.get("reasoning_tokens")
                    # A low-effort response may legitimately call a tool without generating any
                    # reasoning tokens or reasoning item.  There is then nothing to replay.  Any
                    # generated item still needs its opaque payload, and non-zero/unknown usage
                    # without an item cannot prove stateless continuity.
                    replay_material_missing = (
                        bool(reasoning_items) and not encrypted_complete
                    ) or (
                        not reasoning_items and reasoning_tokens != 0
                    )
                    if replay_material_missing:
                        raise MissingEncryptedReasoningContent(
                            "tool-producing response omitted encrypted reasoning content")
                if reasoning_items:
                    self._reasoning_shadow[assistant_shadow_key(message)] = reasoning_items
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
                if evidence_protocol == "openai-encrypted-v1":
                    self._emit_provider_telemetry(
                        "provider_reasoning_replay",
                        protocol=evidence_protocol,
                        generated_reasoning_items=len(reasoning_items),
                        encrypted_content_present=(
                            bool(reasoning_items) and all(
                                isinstance(item.get("encrypted_content"), str)
                                and bool(item["encrypted_content"].strip())
                                for item in reasoning_items
                            )
                        ),
                        item_sha256=_reasoning_item_hashes(reasoning_items),
                        replayed_reasoning_items=len(replayed_items),
                        replayed_item_sha256=replayed_hashes,
                    )
                return ModelTurn(
                    message=message,
                    stop_reason=stop_reason,
                    usage=usage,
                    provider_request_id=(str(getattr(response, "id", "")) or None),
                    reasoning_content=reasoning,
                )
            except Exception as exc:
                missing_encrypted = isinstance(exc, MissingEncryptedReasoningContent)
                failure = (
                    default_failure(
                        FailureCode.PROVIDER_SERVER_ERROR,
                        retryable=True,
                        detail_safe="MissingEncryptedReasoningContent",
                    )
                    if missing_encrypted
                    else classify_provider_exception(exc)
                )
                response_headers = safe_response_headers(exc)
                elapsed_s = max(time.monotonic() - attempt_started, 0.0)
                if missing_encrypted:
                    will_retry = missing_encrypted_retries < 1
                    missing_encrypted_retries += 1
                else:
                    ordinary_failures += 1
                    will_retry = bool(failure.retryable and ordinary_failures < self._attempts)
                delay = retry_delay_seconds(
                    transport, failure.code.value, attempt, response_headers,
                    random_fn=getattr(self, "_random", random.random),
                ) if will_retry and not missing_encrypted else 0.0
                shared_cooldown = limiter.record_failure(
                    reservation, failure.code.value, response_headers,
                    minimum_cooldown_s=delay, transport_profile=transport)
                delay = max(delay, shared_cooldown) if will_retry else 0.0
                self._emit_attempt_failure(
                    attempt=attempt,
                    attempts_max=maximum_attempts,
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
    "OPENAI_RESPONSES_ADAPTER_VERSION",
    "MissingEncryptedReasoningContent",
    "OpenAIResponsesAdapter",
    "assistant_shadow_key",
    "translate_messages",
    "translate_response",
    "translate_stop_reason",
    "translate_tools",
    "translate_usage",
]
