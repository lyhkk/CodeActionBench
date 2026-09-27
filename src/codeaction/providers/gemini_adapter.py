"""Native Gemini GenerateContent adapter with provider-managed implicit prefix caching.

Every request replays the complete stateless system, tool, task, conversation, and rolling visual
tail context. Gemini may reuse an identical prefix internally, but the harness never references a
provider-side conversation or creates a fixed ``CachedContent`` resource.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from codeaction.contracts.failures import (
    EmptyProviderCompletion,
    ProviderCallError,
    classify_provider_exception,
    ensure_usable_provider_completion,
)
from codeaction.providers.model_adapter import (
    ModelCapabilities,
    ModelTurn,
    capabilities_for_model,
    provider_request_profile,
    resolve_output_tokens,
)
from codeaction.providers.model_registry import find_model
from codeaction.providers.provider_runtime import (
    SharedRateLimiter,
    normalize_rate_limit_policy,
    normalize_transport_profile,
    provider_attempt_limit,
    provider_sdk_timeout,
    retry_delay_seconds,
    safe_response_headers,
)


GEMINI_NATIVE_ROOT = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_CACHE_MODE = "implicit-provider-managed-v1"


class GeminiProtocolError(ValueError):
    """Local request-translation failure classified as a non-retryable harness error."""

    status_code = 400


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _native_root(base_url: Optional[str]) -> str:
    if not base_url:
        return GEMINI_NATIVE_ROOT
    parsed = urlsplit(str(base_url))
    if not parsed.scheme or not parsed.netloc:
        raise GeminiProtocolError("Gemini base URL must be absolute")
    path = parsed.path.rstrip("/")
    if path.endswith("/openai"):
        path = path[:-len("/openai")]
    if not path:
        path = "/v1beta"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _content_blocks(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if value is None:
        return []
    if not isinstance(value, list):
        raise GeminiProtocolError("canonical content must be text or a block list")
    return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def _parts(value: Any) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for block in _content_blocks(value):
        kind = block.get("type")
        if kind == "text":
            # Request-only cache metadata is never sent to Google.
            out.append({"text": str(block.get("text") or "")})
            continue
        if kind != "image_url":
            raise GeminiProtocolError(f"unsupported Gemini content block {kind!r}")
        image = block.get("image_url")
        uri = image.get("url") if isinstance(image, Mapping) else image
        if not isinstance(uri, str) or not uri.startswith("data:") or ";base64," not in uri:
            raise GeminiProtocolError("Gemini images must be base64 data URIs")
        header, data = uri.split(",", 1)
        mime_type = header[5:].split(";", 1)[0]
        if not mime_type.startswith("image/") or not data:
            raise GeminiProtocolError("Gemini image data URI is malformed")
        out.append({"inlineData": {"mimeType": mime_type, "data": data}})
    return out


class _GeminiSchemaFields:
    """The OpenAPI-3.0 subset `FunctionDeclaration.parameters` accepts, as an ALLOWLIST.

    Google validates this field set by name and returns a hard 400 -- `Invalid JSON payload
    received. Unknown name "additionalProperties" at 'tools[0].function_declarations[0]
    .parameters'` -- for anything outside it, which fails the request before the model sees a
    single token. Our delivered schemas are strict JSON Schema and carry `additionalProperties:
    false` on every tool plus `uniqueItems` on capture_evidence_views, so every Gemini episode
    died on turn 1 with zero tool calls until this translation existed.

    An allowlist rather than a denylist because the cost of missing a key is the whole cell, and
    our own schemas are free to gain keywords later. `dropped_for` exists so the difference is
    reportable instead of silent: Gemini is handed a marginally looser contract than the other
    vendors, which is a protocol fact worth recording rather than hiding.
    """

    ALLOWED = frozenset({
        "type", "format", "title", "description", "nullable", "enum", "items", "properties",
        "required", "minItems", "maxItems", "minProperties", "maxProperties", "minLength",
        "maxLength", "pattern", "example", "anyOf", "propertyOrdering", "default",
        "minimum", "maximum",
    })

    def sanitize(self, schema: Any) -> Any:
        """Recursively keep only accepted keywords. Property NAMES are data and are preserved."""
        if isinstance(schema, Mapping):
            out: Dict[str, object] = {}
            for key, value in schema.items():
                if key not in self.ALLOWED:
                    continue
                if key == "properties" and isinstance(value, Mapping):
                    out[key] = {str(prop): self.sanitize(sub) for prop, sub in value.items()}
                elif key in ("items", "anyOf"):
                    out[key] = ([self.sanitize(entry) for entry in value]
                                if isinstance(value, list) else self.sanitize(value))
                else:
                    out[key] = deepcopy(value)
            return out
        if isinstance(schema, list):
            return [self.sanitize(entry) for entry in schema]
        return deepcopy(schema)

    def dropped_for(self, tools: Sequence[Mapping[str, Any]]) -> tuple:
        """Keyword names this translation removes from a tool surface, for the run record."""
        found: set = set()

        def walk(node: Any, in_properties: bool = False) -> None:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    if not in_properties and key not in self.ALLOWED:
                        found.add(str(key))
                    walk(value, in_properties=(key == "properties"))
            elif isinstance(node, list):
                for entry in node:
                    walk(entry)

        for item in tools or ():
            walk(((item or {}).get("function") or {}).get("parameters"))
        return tuple(sorted(found))


GEMINI_SCHEMA_FIELDS = _GeminiSchemaFields()


def translate_tools(tools: Sequence[Mapping[str, Any]]) -> List[Dict[str, object]]:
    declarations = []
    for item in tools or ():
        function = (item or {}).get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            raise GeminiProtocolError("Gemini tool translation requires a function name")
        declarations.append({
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": GEMINI_SCHEMA_FIELDS.sanitize(dict(
                function.get("parameters")
                or {"type": "object", "properties": {}})),
        })
    return [{"functionDeclarations": declarations}] if declarations else []


def _assistant_key(message: Mapping[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    ids = [str((call or {}).get("id") or "") for call in calls]
    if ids and all(ids):
        return "tools:" + "|".join(ids)
    digest = hashlib.sha256(str(message.get("content") or "").encode()).hexdigest()
    return "text:" + digest[:32]


def _tool_result(value: Any) -> Dict[str, object]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        parsed = {"result": str(value or "")}
    return dict(parsed) if isinstance(parsed, Mapping) else {"result": parsed}


def translate_messages(
    messages: Sequence[Mapping[str, Any]],
    shadow: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[str, List[Dict[str, object]]]:
    """Translate canonical history and preserve native signed model parts exactly."""
    shadow = shadow or {}
    systems = [message for message in messages if message.get("role") == "system"]
    if len(systems) != 1:
        raise GeminiProtocolError("Gemini requires exactly one system message")
    system_parts = _parts(systems[0].get("content"))
    if not system_parts or any(set(part) != {"text"} for part in system_parts):
        raise GeminiProtocolError("Gemini system instruction must be text-only")
    system_text = "\n".join(str(part["text"]) for part in system_parts)

    call_names: Dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            call_names[str((call or {}).get("id") or "")] = str(function.get("name") or "")

    contents: List[Dict[str, object]] = []
    pending_function_responses: List[Dict[str, object]] = []

    def flush_function_responses() -> None:
        nonlocal pending_function_responses
        if pending_function_responses:
            # The native API accepts function-response parts in a user turn. Keeping parallel
            # responses together preserves the order of the preceding function-call parts.
            contents.append({"role": "user", "parts": pending_function_responses})
            pending_function_responses = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            name = call_names.get(call_id)
            if not name:
                raise GeminiProtocolError("tool result has no matching Gemini function call")
            pending_function_responses.append({
                "functionResponse": {
                    "name": name,
                    "response": _tool_result(message.get("content")),
                }
            })
            continue
        flush_function_responses()
        if role == "user":
            parts = _parts(message.get("content"))
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue
        if role != "assistant":
            raise GeminiProtocolError(f"unsupported canonical role {role!r}")
        native = shadow.get(_assistant_key(message))
        if native is not None:
            contents.append(deepcopy(dict(native)))
            continue
        parts = _parts(message.get("content"))
        for call in message.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            try:
                args = json.loads(str(function.get("arguments") or "{}"))
            except json.JSONDecodeError:
                args = {}
            parts.append({
                "functionCall": {
                    "name": str(function.get("name") or ""),
                    "args": args if isinstance(args, Mapping) else {},
                }
            })
        if parts:
            contents.append({"role": "model", "parts": parts})
    flush_function_responses()
    return system_text, contents


def translate_response(body: Mapping[str, Any]) -> Tuple[dict, Optional[str], dict, str]:
    candidates = body.get("candidates") or []
    if not candidates:
        raise EmptyProviderCompletion("Gemini response has no candidates")
    candidate = candidates[0]
    native_content = candidate.get("content") or {}
    native_parts = native_content.get("parts") or []
    texts: List[str] = []
    thoughts: List[str] = []
    calls: List[Dict[str, object]] = []
    nonce = secrets.token_hex(8)
    for index, part in enumerate(native_parts):
        if not isinstance(part, Mapping):
            continue
        if part.get("thought") is True:
            if part.get("text"):
                thoughts.append(str(part["text"]))
            continue
        if part.get("text"):
            texts.append(str(part["text"]))
        function = part.get("functionCall")
        if isinstance(function, Mapping):
            call_id = str(function.get("id") or f"gemini-{nonce}-{index}")
            calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(function.get("name") or ""),
                    "arguments": json.dumps(
                        function.get("args") or {}, ensure_ascii=False,
                        sort_keys=True, separators=(",", ":")),
                },
            })
    message: Dict[str, object] = {"content": "\n".join(texts).strip() or None}
    if calls:
        message["tool_calls"] = calls
    finish = str(candidate.get("finishReason") or "").upper()
    if calls:
        stop_reason = "tool_calls"
    elif finish == "MAX_TOKENS":
        stop_reason = "length"
    elif finish in {
        "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
        "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT",
    }:
        stop_reason = "refusal"
    elif finish in ("", "STOP"):
        stop_reason = "stop"
    else:
        stop_reason = finish.lower()
    return message, ("\n".join(thoughts).strip() or None), deepcopy(dict(native_content)), stop_reason


def translate_usage(value: Any) -> Dict[str, Optional[int]]:
    usage = value if isinstance(value, Mapping) else {}
    prompt = _optional_int(usage.get("promptTokenCount"))
    candidate = _optional_int(usage.get("candidatesTokenCount"))
    thoughts = _optional_int(usage.get("thoughtsTokenCount"))
    completion = None
    if candidate is not None or thoughts is not None:
        completion = int(candidate or 0) + int(thoughts or 0)
    out: Dict[str, Optional[int]] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }
    cached = _optional_int(usage.get("cachedContentTokenCount"))
    if cached is not None:
        out["cached_tokens"] = cached
    if thoughts is not None:
        out["reasoning_tokens"] = thoughts
    return out


class GeminiGenerateContentAdapter:
    """Stateless native adapter that leaves common-prefix reuse to Gemini."""

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
        self.model = str(model or "")
        self._capabilities = capabilities or capabilities_for_model(self.model)
        self._request_profile = request_profile or provider_request_profile(
            self.model, reasoning_profile=reasoning_profile)
        entry = find_model(self.model)
        self._transport_profile = normalize_transport_profile(
            transport_profile or (entry.transport_profile() if entry is not None else None))
        self._attempts = int(attempts or self._transport_profile["attempts"])
        effective_timeout = provider_sdk_timeout(self._transport_profile, timeout)
        self._root = _native_root(base_url)
        if client is None:
            import httpx
            self._client = httpx.Client(
                timeout=effective_timeout,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            )
        else:
            self._client = client
        self._sleep = sleep_fn
        self._random = random_fn
        self._event_sink = event_sink
        self._rate_limit_policy = normalize_rate_limit_policy(
            rate_limit_policy, quota_group=(rate_limit_policy or {}).get("quota_group"))
        self._rate_limiter = SharedRateLimiter(
            self._rate_limit_policy, self.model, rate_limit_state_dir, sleep_fn=sleep_fn)
        self.infra_retries = 0
        self.infra_wait_s = 0.0
        self.quota_wait_s = 0.0
        self._request_token_estimate = None
        self._request_input_token_estimate = None
        self._request_output_token_estimate = None
        self.last_effective_output_tokens = None
        self._native_shadow: Dict[str, Dict[str, Any]] = {}

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def _wire_model(self) -> str:
        """The name the vendor API expects; the benchmark id unless one was declared.

        Read defensively: unit tests build adapters through ``__new__`` and set only the
        attributes under test, so a missing override must fall back rather than raise.
        """
        return getattr(self, "_provider_model_override", None) or self.model

    def request_profile(self) -> Dict[str, object]:
        profile = deepcopy(self._request_profile)
        profile["cache"] = {
            "mode": GEMINI_CACHE_MODE,
            "prefix": "provider_matched_common_prefix",
            "client_breakpoint": False,
        }
        return json.loads(json.dumps(profile, sort_keys=True))

    def transport_profile(self) -> Dict[str, object]:
        return json.loads(json.dumps(self._transport_profile, sort_keys=True))

    def rate_limit_policy(self) -> Dict[str, object]:
        return json.loads(json.dumps(self._rate_limit_policy, sort_keys=True))

    def set_event_sink(self, sink: Optional[Callable[[Dict[str, object]], None]]) -> None:
        self._event_sink = sink

    def set_request_token_estimate(
        self, input_tokens: int, output_tokens: Optional[int] = None,
    ) -> None:
        self._request_input_token_estimate = max(int(input_tokens), 0)
        self._request_output_token_estimate = max(int(output_tokens or 0), 0)
        self._request_token_estimate = max(
            self._request_input_token_estimate + self._request_output_token_estimate, 1)

    def _emit(self, event: str, **fields: object) -> None:
        if self._event_sink is not None:
            self._event_sink({
                "event": event,
                "observed_at_unix_s": round(time.time(), 3),
                **fields,
            })

    def _quota_wait(self, waited: float) -> None:
        if not waited:
            return
        self.quota_wait_s = round(self.quota_wait_s + waited, 3)
        self.infra_wait_s = round(self.infra_wait_s + waited, 3)
        self._emit(
            "provider_quota_wait", quota_wait_s=waited,
            quota_group=self._rate_limit_policy.get("quota_group"))

    def _emit_failure(
        self, attempt: int, failure, elapsed_s: float, delay: float, will_retry: bool,
        *, operation: str,
    ) -> None:
        self._emit(
            "provider_attempt_failed",
            provider_operation=operation,
            provider_attempt=int(attempt),
            provider_attempts_max=int(self._attempts),
            will_retry=bool(will_retry),
            failed_request_elapsed_s=round(max(elapsed_s, 0.0), 3),
            retry_delay_s=round(max(delay, 0.0), 3),
            failure=failure.to_dict(),
        )

    def _generation_config(self, effective: int) -> dict:
        profile = self.request_profile()
        config = {"maxOutputTokens": int(effective)}
        temperature = profile.get("temperature")
        if temperature is not None:
            config["temperature"] = float(temperature)
        extra = profile.get("extra_body") or {}
        unknown = set(extra) - {"thinkingConfig"}
        if unknown:
            raise GeminiProtocolError(
                f"unsupported native Gemini profile fields: {sorted(unknown)}")
        config.update(deepcopy(extra))
        return config

    def step(self, messages, tools, requested_output_tokens) -> ModelTurn:
        effective = resolve_output_tokens(self._capabilities, requested_output_tokens)
        self.last_effective_output_tokens = effective
        try:
            system_text, contents = translate_messages(messages, self._native_shadow)
            request = {
                "systemInstruction": {
                    "role": "system", "parts": [{"text": system_text}]},
                "contents": contents,
                "generationConfig": self._generation_config(effective),
            }
            native_tools = translate_tools(tools)
            if native_tools:
                request["tools"] = native_tools
                request["toolConfig"] = {
                    "functionCallingConfig": {"mode": "AUTO"}}
        except ProviderCallError:
            raise
        except Exception as exc:
            raise ProviderCallError(classify_provider_exception(exc)) from exc
        for attempt in range(1, self._attempts + 1):
            estimate = int(self._request_token_estimate or effective)
            reservation, waited = self._rate_limiter.acquire(
                estimate,
                estimated_input_tokens=self._request_input_token_estimate,
                estimated_output_tokens=self._request_output_token_estimate,
            )
            self._quota_wait(waited)
            started = time.monotonic()
            try:
                response = self._client.post(
                    f"{self._root}/models/{self._wire_model}:generateContent", json=request)
                response.raise_for_status()
                body = response.json()
                message, reasoning, native_content, stop_reason = translate_response(body)
                ensure_usable_provider_completion(
                    message,
                    stop_reason,
                    provider=self.model,
                    auxiliary_content=reasoning,
                )
                if message.get("tool_calls"):
                    self._native_shadow[_assistant_key(message)] = native_content
                elif message.get("content") is not None:
                    self._native_shadow[_assistant_key(message)] = native_content
                usage = translate_usage(body.get("usageMetadata"))
                actual = sum(int(usage.get(key) or 0) for key in (
                    "prompt_tokens", "completion_tokens"))
                headers = safe_response_headers(response)
                self._rate_limiter.record_success(
                    reservation, actual if actual > 0 else None, headers,
                    actual_input_tokens=usage.get("prompt_tokens"),
                    actual_output_tokens=usage.get("completion_tokens"),
                    cached_input_tokens=usage.get("cached_tokens"),
                )
                if headers:
                    self._emit("provider_response_telemetry", response_headers=headers)
                request_id = next((headers[key] for key in (
                    "x-request-id", "request-id") if key in headers), None)
                stop_details = None
                if stop_reason == "refusal":
                    stop_details = {"category": str(
                        (body.get("candidates") or [{}])[0].get("finishReason") or "unspecified")}
                return ModelTurn(
                    message=message,
                    stop_reason=stop_reason,
                    usage=usage,
                    provider_request_id=request_id,
                    reasoning_content=reasoning,
                    stop_details=stop_details,
                )
            except Exception as exc:
                failure = classify_provider_exception(exc)
                headers = safe_response_headers(exc)
                elapsed = max(time.monotonic() - started, 0.0)
                attempt_limit = min(
                    self._attempts,
                    provider_attempt_limit(self._transport_profile, failure.code.value),
                )
                will_retry = bool(failure.retryable and attempt < attempt_limit)
                delay = retry_delay_seconds(
                    self._transport_profile, failure.code.value, attempt, headers,
                    random_fn=self._random,
                ) if will_retry else 0.0
                shared = self._rate_limiter.record_failure(
                    reservation, failure.code.value, headers,
                    minimum_cooldown_s=delay,
                    transport_profile=self._transport_profile,
                )
                delay = max(delay, shared) if will_retry else 0.0
                self._emit_failure(
                    attempt, failure, elapsed, delay, will_retry,
                    operation="generate_content")
                if headers:
                    self._emit("provider_error_telemetry", response_headers=headers)
                if not will_retry:
                    raise ProviderCallError(failure) from exc
                self.infra_retries += 1
                self.infra_wait_s = round(self.infra_wait_s + elapsed + delay, 3)
                self._sleep(delay)
        raise AssertionError("Gemini generation attempt loop ended unexpectedly")

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


__all__ = [
    "GEMINI_CACHE_MODE",
    "GeminiGenerateContentAdapter",
    "GeminiProtocolError",
    "translate_messages",
    "translate_response",
    "translate_tools",
    "translate_usage",
]
