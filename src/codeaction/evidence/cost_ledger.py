"""Cost accounting ledger for multi-vendor model API usage and prompt caching.

Tracks USD cost per episode across uncached prompt, completion, cache creation, and cache read
tokens -- and, because the vendors disagree about what a prompt-token count contains, it resolves
that convention from the registry rather than assuming one.

Two conventions exist among the protocols this benchmark runs:

* `anthropic` reports `input_tokens` as ONLY the tokens billed at the full input rate. Cache reads
  and cache writes are separate counters that are not part of it.
* every other protocol (`openai-compatible`, `openai-responses`, `google-generate-content`)
  reports a TOTAL prompt count with the cached part already inside it.

Pricing one convention's numbers under the other's rule is not a rounding error: it double-bills
the cached prefix, or -- the direction the 2026-08-13 canary hit -- drops it entirely, because the
runner's canonical key for a cache read is `cached_tokens` and the old alias list never looked for
it. Two rules follow, and both are enforced below rather than documented and hoped for: a model
with no verified unit price yields no total instead of a default-shaped one, and usage that
contradicts its own declared convention yields no total instead of a clamped one.
"""

from typing import Any, Dict, Optional

from codeaction.providers.model_registry import find_model

# Pricing per 1M tokens in USD (as of 2026-08)
MODEL_PRICING_TABLE = {
    "claude-opus-5": {
        "input": 5.00,
        "output": 25.00,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
    "claude-sonnet-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "gpt-5.6": {
        "input": 2.50,
        "output": 10.00,
        "cache_write": 2.50,
        "cache_read": 1.25,
    },
    "gemini-3.6-flash": {
        "input": 0.075,
        "output": 0.30,
        "cache_write": 0.075,
        "cache_read": 0.01875,
    },
    "kimi-k3": {
        "input": 1.20,
        "output": 4.80,
        "cache_write": 1.20,
        "cache_read": 0.60,
    },
    "grok-4.5": {
        "input": 2.00,
        "output": 6.00,
        "cache_write": 2.00,
        "cache_read": 0.30,
    },
    "qwen3.8-max": {
        "input": 2.00,
        "output": 6.00,
        "cache_write": 2.50,
        "cache_read": 0.17,
    },
}

# Alias mapping for model IDs
MODEL_ALIASES = {
    "gpt-5-6": "gpt-5.6",
    "opus-5": "claude-opus-5",
    "claude-opus": "claude-opus-5",
    "sonnet-5": "claude-sonnet-5",
    "claude-sonnet": "claude-sonnet-5",
    "gemini-3.6": "gemini-3.6-flash",
    "grok": "grok-4.6",
    "grok-latest": "grok-4.6",
    "qwen-max": "qwen3.8-max",
    "qwen3.8": "qwen3.8-max",
    "qwen3.8-max": "qwen3.8-max",
    # Same model, second endpoint: it bills at the same published rates.
    "kimi-k3-intl": "kimi-k3",
}


def normalize_model_name(model_name: str) -> str:
    cleaned = str(model_name).strip().lower()
    return MODEL_ALIASES.get(cleaned, cleaned)


# Protocols whose prompt-token count already contains the cached prefix. `anthropic` is the one
# exception and is listed explicitly so that a newly supported protocol fails loudly here rather
# than silently inheriting whichever rule happens to be the default.
_PROMPT_INCLUDES_CACHED_BY_PROTOCOL = {
    "openai-compatible": True,
    "openai-responses": True,
    "google-generate-content": True,
    "anthropic": False,
    "scripted": True,
}
INCLUDES_CACHED = "includes_cached"
EXCLUDES_CACHED = "excludes_cached"

# Pricing-table keys whose spelling has drifted from the registry id that declares the protocol.
# Without this bridge a priced model would resolve to no convention and silently stop being
# priceable, which is the failure mode one layer up from the one this module is fixing. Empty
# today: the one entry that needed it, grok-4-reasoning -> grok-4.20-0309-reasoning, went away
# with that model. `test_every_registry_model_that_is_priced_declares_its_convention` is what
# keeps this honest -- a priced model that stops resolving fails there rather than in a batch.
REGISTRY_IDS: Dict[str, str] = {}

# Aliases each protocol may use for the same measured quantity. `cached_tokens` /
# `cache_creation_tokens` are this repo's canonical runner keys; the rest are the raw vendor
# spellings, kept so a hand-assembled usage dict from a transcript prices the same way.
_CACHE_READ_KEYS = (
    "cached_tokens", "cache_read_input_tokens", "cache_read_tokens", "cachedContentTokenCount")
_CACHE_WRITE_KEYS = (
    "cache_creation_tokens", "cache_creation_input_tokens", "cache_write_tokens")
_PROMPT_KEYS = ("prompt_tokens", "input_tokens", "promptTokenCount")
_COMPLETION_KEYS = ("completion_tokens", "output_tokens", "candidatesTokenCount")


def _first_int(usage: Dict[str, Any], keys) -> tuple[Optional[int], Optional[str]]:
    """Return the first present non-negative integer, plus a fail-closed parse error."""
    for key in keys:
        value = usage.get(key)
        if value is not None:
            if isinstance(value, bool):
                return None, f"{key} must be a non-negative integer"
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped.isdigit():
                    return None, f"{key} must be a non-negative integer"
                value = stripped
            try:
                parsed = int(value)
            except (OverflowError, TypeError, ValueError):
                return None, f"{key} must be a non-negative integer"
            if not isinstance(value, str) and value != parsed:
                return None, f"{key} must be a non-negative integer"
            if parsed < 0:
                return None, f"{key} must be a non-negative integer"
            return parsed, None
    return 0, None


def prompt_tokens_convention(model_name: str) -> Optional[str]:
    """Whether this model's `prompt_tokens` already contains the cached prefix.

    None for a model this repo does not declare. Defaulting to either convention would be a guess
    about how somebody else's API counts tokens, and a guess is what this module exists to stop:
    an unknown model produces no derived split rather than a plausible wrong one.
    """
    for candidate in (str(model_name or "").strip(), normalize_model_name(model_name),
                      REGISTRY_IDS.get(normalize_model_name(model_name), "")):
        entry = find_model(candidate) if candidate else None
        if entry is not None:
            return (INCLUDES_CACHED
                    if _PROMPT_INCLUDES_CACHED_BY_PROTOCOL.get(entry.protocol, True)
                    else EXCLUDES_CACHED)
    return None


def billing_split(model_name: str, usage: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve one usage record into the four quantities a bill is actually made of.

    Returns `uncached_input_tokens=None` when the model declares no convention here, or when the
    record contradicts the one it does declare -- the caller must then refuse to produce a total
    rather than pricing a guessed or clamped number.
    """
    convention = prompt_tokens_convention(model_name)
    prompt_tokens, prompt_error = _first_int(usage, _PROMPT_KEYS)
    cache_read, cache_read_error = _first_int(usage, _CACHE_READ_KEYS)
    cache_write, cache_write_error = _first_int(usage, _CACHE_WRITE_KEYS)
    completion_tokens, completion_error = _first_int(usage, _COMPLETION_KEYS)
    errors = [
        error for error in (
            prompt_error, cache_read_error, cache_write_error, completion_error)
        if error is not None
    ]
    input_invalid = any(error is not None for error in (
        prompt_error, cache_read_error, cache_write_error))
    if convention is None or input_invalid:
        uncached = None
    elif convention == EXCLUDES_CACHED:
        uncached = prompt_tokens
    else:
        uncached = prompt_tokens - cache_read - cache_write
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_convention": convention,
        "completion_tokens": completion_tokens,
        "uncached_input_tokens": uncached if (uncached is not None and uncached >= 0) else None,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "input_tokens_total": (
            None if convention is None or input_invalid
            else prompt_tokens if convention == INCLUDES_CACHED
            else prompt_tokens + cache_read + cache_write),
        "usage_error": "; ".join(errors) if errors else None,
    }


def calculate_episode_cost(model_name: str, usage: Dict[str, Any]) -> Dict[str, Any]:
    """Price one episode's usage, or say why it cannot be priced.

    `total_cost_usd` is None exactly when the answer would have to be invented: no verified unit
    price for the model, or a usage record whose cached count is impossible under its protocol's
    convention. The measured token split is reported either way.
    """
    model_key = normalize_model_name(model_name)
    pricing = MODEL_PRICING_TABLE.get(model_key)
    split = billing_split(model_name, usage)
    out = {"model": model_key, **split, "total_cost_usd": None,
           "pricing_snapshot": None, "unpriced_reason": None}

    if split["usage_error"] is not None:
        out["unpriced_reason"] = f"invalid usage counters: {split['usage_error']}"
        return out
    if split["prompt_tokens_convention"] is None:
        out["unpriced_reason"] = (
            f"{model_key} is not in codeaction/providers/models/registry.json, so it declares no prompt-token "
            "convention; the cached share cannot be separated from the prompt total without one")
        return out
    if split["uncached_input_tokens"] is None:
        out["unpriced_reason"] = (
            f"usage contradicts the {split['prompt_tokens_convention']} convention declared for "
            f"{model_key}: prompt_tokens={split['prompt_tokens']} is smaller than "
            f"cache_read={split['cache_read_tokens']} + cache_write={split['cache_write_tokens']}")
        return out
    if not pricing:
        out["unpriced_reason"] = (
            f"no verified unit price for {model_key} in MODEL_PRICING_TABLE; add one from the "
            "vendor's published table before reporting a cost for it")
        return out

    total = (
        split["uncached_input_tokens"] / 1_000_000.0 * pricing["input"]
        + split["completion_tokens"] / 1_000_000.0 * pricing["output"]
        + split["cache_write_tokens"] / 1_000_000.0 * pricing["cache_write"]
        + split["cache_read_tokens"] / 1_000_000.0 * pricing["cache_read"]
    )
    out["total_cost_usd"] = round(total, 6)
    out["pricing_snapshot"] = pricing
    return out
