"""The declared model registry and the single credential-resolution path.

Two files, deliberately separate:

* ``codeaction/providers/models/registry.json`` -- committed, non-secret, hashed into identity.  It declares
  everything *about* a model: which protocol it speaks, which credential alias it uses, its
  context window and output limit, and how the benchmark's abstract reasoning ladder
  (disabled/low/medium/high) maps onto that vendor's native control.
* a 0600 credential file -- never committed, and structurally unable to carry identity.  It maps
  ``<ALIAS>_KEY`` / ``<ALIAS>_BASE_URL`` to secrets and nothing else.

The separation is the point.  If a secret file could also declare a model id or a context window,
the values that enter the comparison hash would come from a file nobody can audit, and a published
result could not be checked against what it claims to have run.  Keeping identity in the committed
registry and secrets in the credential file preserves that check while still giving one place to
put every key.

Adding an OpenAI-compatible vendor (Kimi, MiMo, GLM, DeepSeek, a vLLM deployment) is a registry
edit with no code change; only a genuinely new wire protocol needs a new adapter.
"""
from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from codeaction.contracts.identity import sha256_json
from codeaction.providers.provider_runtime import normalize_transport_profile


REGISTRY_PATH = Path(__file__).resolve().parent / "models" / "registry.json"
REGISTRY_SCHEMA_VERSION = "0.1"
SUPPORTED_PROTOCOLS = (
    "openai-compatible", "openai-responses", "anthropic",
    "google-generate-content", "scripted")
TOKEN_ESTIMATOR_UTF8_BYTES_V1 = "utf8-bytes-v1"
# The rungs a task card / CLI may ask for. A model need not implement every rung, but a rung it
# does implement must mean the same place on the ladder across vendors.
# xhigh is above the released roster's top rung. It exists on the ladder so a seat whose model
# declares it can be run at it deliberately -- a thinking-depth comparison is one (model, rung)
# pair against another -- and it is never a default anywhere.
REASONING_RUNGS = ("disabled", "low", "medium", "high", "xhigh", "none")
# How a vendor hands its reasoning text back, and therefore which adapter field carries it.  A rung
# that only *spends* thinking tokens is not enough: the vendor must also be asked to return the
# text, and that is a second switch on a different part of the request.  Declaring the channel here
# makes the omission a registry error instead of a silently empty transcript column.
REASONING_CHANNELS = (
    "reasoning_content", "inline_thought_tag", "thinking_blocks", "responses_summary", "none")
_CHANNELS_BY_PROTOCOL = {
    "openai-compatible": ("reasoning_content", "inline_thought_tag", "none"),
    "openai-responses": ("responses_summary", "none"),
    "anthropic": ("thinking_blocks", "none"),
    "google-generate-content": ("thinking_blocks", "none"),
    "scripted": ("none",),
}
# `inline_thought_tag` exists because one shipped vendor does not use a side channel at all.
# Measured live 2026-08-08: Google's OpenAI-compatibility layer returns the thought summary INSIDE
# `message.content`, wrapped in a `<thought>...</thought>` element, with `extra_content.google
# .thought = true` alongside it. Left alone, that text is not merely unrecorded -- it is replayed
# into the next prompt as assistant content and rendered as the model's answer.
DEFAULT_INLINE_THOUGHT_TAG = "thought"
# Which message fields the OpenAI-compatible normalizer will read for reasoning text. Most vendors
# use `reasoning_content`, but the field is a vendor convention rather than part of the protocol --
# Google's OpenAI-compatibility layer documents how to *request* thought summaries and never says
# which field returns them. Declaring the candidates per model keeps that uncertainty visible and
# lets the run-level capture check decide it, instead of a guess that fails silently.
DEFAULT_REASONING_FIELDS = ("reasoning_content",)
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_CREDENTIAL_KEY_RE = re.compile(r"^([A-Z][A-Z0-9_]*)_(KEY|BASE_URL)$")


class RegistryError(ValueError):
    """Raised for a malformed registry, an unknown model, or an unusable credential file."""


@dataclass(frozen=True)
class ModelEntry:
    id: str
    protocol: str
    credential: Optional[str]
    context_window_tokens: int
    max_output_tokens: int
    token_estimator_id: str
    capabilities_source: str
    default_reasoning_profile: str
    reasoning_profiles: Mapping[str, Mapping[str, Any]]
    reasoning_capture: Mapping[str, Any]
    transport: Mapping[str, Any]
    # The name the vendor's API expects. It equals `id` for every model whose benchmark name is the
    # vendor's own, and differs when one model is registered twice -- for example the same Kimi
    # reached through a second endpoint, where the benchmark needs two ids but the wire name stays
    # the vendor's.
    provider_model: str = ""
    # Optional explicitly declared alternative accounts for a custom model.
    fallback_credentials: Tuple[str, ...] = ()

    def wire_model(self) -> str:
        return self.provider_model or self.id

    def reasoning_fields(self) -> Tuple[str, ...]:
        """Message fields the OpenAI-compatible adapter should read for reasoning text."""
        return tuple(self.reasoning_capture.get("fields") or ())

    def inline_thought_tag(self) -> Optional[str]:
        """XML element wrapping reasoning inside ``content``, for vendors with no side channel."""
        if self.reasoning_capture.get("channel") != "inline_thought_tag":
            return None
        return str(self.reasoning_capture.get("tag") or DEFAULT_INLINE_THOUGHT_TAG)

    def expects_reasoning(self, rung: Optional[str] = None) -> bool:
        """Whether a run on this rung must record non-empty reasoning text.

        The answer is a declaration, not an observation, which is the point: a run that declares
        reasoning and records none is a defect the harness can name, rather than an absence nobody
        notices until someone goes looking for the model's thinking months later.
        """
        name = str(rung or self.default_reasoning_profile)
        return name in tuple(self.reasoning_capture.get("expected_rungs") or ())

    def profile(self, rung: Optional[str] = None) -> Dict[str, Any]:
        """Return the recorded request profile for one rung of the reasoning ladder."""
        name = str(rung or self.default_reasoning_profile)
        if name not in self.reasoning_profiles:
            raise RegistryError(
                f"model {self.id!r} does not declare reasoning profile {name!r}; "
                f"declared={sorted(self.reasoning_profiles)}")
        return json.loads(json.dumps(self.reasoning_profiles[name], sort_keys=True))

    def transport_profile(self) -> Dict[str, Any]:
        """Return the recorded timeout/retry policy for this model endpoint."""
        return json.loads(json.dumps(self.transport, sort_keys=True))

    def identity(self) -> Dict[str, Any]:
        """The non-secret model facts that belong in ``identity.comparison.model``."""
        return {
            "id": self.id,
            "protocol": self.protocol,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "token_estimator_id": self.token_estimator_id,
        }


def _require(value: Any, kind: type, what: str):
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise RegistryError(f"{what} must be {kind.__name__}")
    return value


def _parse_reasoning_capture(
    where: str,
    protocol: str,
    raw: Any,
    profiles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate the declaration of how this vendor returns its reasoning text.

    Required on every model, because the failure it prevents is invisible in the artifacts that
    normally get read: a thinking rung that spends thinking tokens but returns none of the text
    produces a transcript that looks complete and a report that renders nothing.  Making the
    declaration mandatory means a new vendor cannot be added without someone answering "and where
    does its reasoning come back?" -- which is the question that was never asked for Anthropic.

    ``expected_rungs`` is the enforceable half: those rungs must actually record reasoning at run
    time, and a run that does not is a defect the harness reports.
    """
    if not isinstance(raw, Mapping):
        raise RegistryError(
            f"{where}.reasoning_capture must be an object declaring how the vendor returns "
            f"reasoning text: channel one of {list(REASONING_CHANNELS)} plus expected_rungs")
    channel = _require(raw.get("channel"), str, f"{where}.reasoning_capture.channel")
    allowed = _CHANNELS_BY_PROTOCOL.get(protocol, _CHANNELS_BY_PROTOCOL["openai-compatible"])
    if channel not in allowed:
        raise RegistryError(
            f"{where}.reasoning_capture.channel {channel!r} is not readable over protocol "
            f"{protocol!r}; allowed={list(allowed)}")
    rungs = raw.get("expected_rungs")
    if not isinstance(rungs, (list, tuple)):
        raise RegistryError(f"{where}.reasoning_capture.expected_rungs must be a list")
    expected = []
    for rung in rungs:
        name = _require(rung, str, f"{where}.reasoning_capture.expected_rungs entry")
        if name not in profiles:
            raise RegistryError(
                f"{where}.reasoning_capture.expected_rungs names {name!r}, which is not a "
                f"declared reasoning profile")
        if name in expected:
            raise RegistryError(
                f"{where}.reasoning_capture.expected_rungs repeats {name!r}")
        expected.append(name)
    if channel == "none" and expected:
        raise RegistryError(
            f"{where}.reasoning_capture declares channel 'none' but expects reasoning on "
            f"{expected}; a model with no reasoning channel cannot return any")
    tag = raw.get("tag")
    if tag is not None:
        if channel != "inline_thought_tag":
            raise RegistryError(
                f"{where}.reasoning_capture.tag only applies to the 'inline_thought_tag' "
                f"channel; channel is {channel!r}")
        _require(tag, str, f"{where}.reasoning_capture.tag")
    elif channel == "inline_thought_tag":
        tag = DEFAULT_INLINE_THOUGHT_TAG
    fields = raw.get("fields")
    if fields is None:
        parsed_fields = DEFAULT_REASONING_FIELDS if channel == "reasoning_content" else ()
    else:
        if channel != "reasoning_content":
            raise RegistryError(
                f"{where}.reasoning_capture.fields only applies to the 'reasoning_content' "
                f"channel; channel is {channel!r}")
        if not isinstance(fields, (list, tuple)) or not fields:
            raise RegistryError(
                f"{where}.reasoning_capture.fields must be a non-empty list of message field "
                f"names")
        parsed_fields = tuple(
            _require(item, str, f"{where}.reasoning_capture.fields entry") for item in fields)
    note = raw.get("note")
    if note is not None:
        _require(note, str, f"{where}.reasoning_capture.note")
    # A model that spends thinking tokens but returns none of the text is a real, allowed state --
    # OpenAI's chat-completions protocol is exactly that -- but it makes one leaderboard row
    # unreadable in a way the others are not.  It has to be a written decision, not a blank field.
    thinking_rungs = [name for name in profiles if name in ("low", "medium", "high")]
    if channel == "none" and thinking_rungs and not (note or "").strip():
        raise RegistryError(
            f"{where}.reasoning_capture declares channel 'none' while declaring thinking rungs "
            f"{sorted(thinking_rungs)}; write a `note` saying why this vendor's reasoning text "
            f"cannot be recorded, so the gap is a documented asymmetry rather than an oversight")
    out: Dict[str, Any] = {"channel": channel, "expected_rungs": tuple(expected)}
    if parsed_fields:
        out["fields"] = parsed_fields
    if tag is not None:
        out["tag"] = tag
    if note is not None:
        out["note"] = note
    return out


def _parse_entry(model_id: str, raw: Mapping[str, Any]) -> ModelEntry:
    where = f"model {model_id!r}"
    protocol = _require(raw.get("protocol"), str, f"{where}.protocol")
    from codeaction.extensions import declarations
    if protocol not in SUPPORTED_PROTOCOLS and protocol not in declarations("provider"):
        raise RegistryError(
            f"{where}.protocol {protocol!r} is not one of {list(SUPPORTED_PROTOCOLS)}")
    provider_model = raw.get("provider_model")
    if provider_model is not None:
        _require(provider_model, str, f"{where}.provider_model")
    provider_model = str(provider_model or "")
    fallbacks = raw.get("fallback_credentials") or []
    if not isinstance(fallbacks, list) or any(
            not isinstance(alias, str) or not _ALIAS_RE.match(alias) for alias in fallbacks):
        raise RegistryError(f"{where}.fallback_credentials must be lowercase kebab-case aliases")
    credential = raw.get("credential")
    if credential is not None:
        _require(credential, str, f"{where}.credential")
        if not _ALIAS_RE.match(credential):
            raise RegistryError(
                f"{where}.credential {credential!r} must be lowercase kebab-case")
    elif protocol != "scripted":
        raise RegistryError(f"{where} needs a credential alias unless it is scripted")

    capabilities = raw.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise RegistryError(f"{where}.capabilities must be an object")
    context_window = _require(
        capabilities.get("context_window_tokens"), int, f"{where}.context_window_tokens")
    max_output = _require(
        capabilities.get("max_output_tokens"), int, f"{where}.max_output_tokens")
    if context_window <= 0 or max_output <= 0:
        raise RegistryError(f"{where} token limits must be positive")
    if max_output >= context_window:
        raise RegistryError(f"{where}.max_output_tokens must be under the context window")
    estimator = str(capabilities.get("token_estimator_id") or TOKEN_ESTIMATOR_UTF8_BYTES_V1)
    # An unsourced context window is exactly the error that silently breaks context management,
    # so the registry refuses to carry one.
    source = _require(capabilities.get("source"), str, f"{where}.capabilities.source")
    if not source.strip():
        raise RegistryError(f"{where}.capabilities.source must say where the numbers came from")

    profiles = raw.get("reasoning_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise RegistryError(f"{where}.reasoning_profiles must be a non-empty object")
    parsed: Dict[str, Dict[str, Any]] = {}
    for rung, profile in profiles.items():
        if rung not in REASONING_RUNGS:
            raise RegistryError(
                f"{where} declares reasoning rung {rung!r}; the ladder is {list(REASONING_RUNGS)}")
        if not isinstance(profile, Mapping):
            raise RegistryError(f"{where}.reasoning_profiles.{rung} must be an object")
        for field in ("id", "reasoning"):
            _require(profile.get(field), str, f"{where}.reasoning_profiles.{rung}.{field}")
        if not isinstance(profile.get("extra_body"), Mapping):
            raise RegistryError(
                f"{where}.reasoning_profiles.{rung}.extra_body must be an object")
        replay_reasoning = profile.get("replay_reasoning_content")
        if replay_reasoning is not None and not isinstance(replay_reasoning, bool):
            raise RegistryError(
                f"{where}.reasoning_profiles.{rung}.replay_reasoning_content must be a boolean")
        cache_scope = profile.get("prompt_cache_key_scope")
        if cache_scope is not None and cache_scope != "episode":
            raise RegistryError(
                f"{where}.reasoning_profiles.{rung}.prompt_cache_key_scope must be 'episode'")
        replay_evidence = profile.get("reasoning_replay_evidence")
        if replay_evidence is not None and replay_evidence != "openai-encrypted-v1":
            raise RegistryError(
                f"{where}.reasoning_profiles.{rung}.reasoning_replay_evidence must be "
                "'openai-encrypted-v1'")
        if replay_evidence is not None and protocol != "openai-responses":
            raise RegistryError(
                f"{where}.reasoning_profiles.{rung}.reasoning_replay_evidence requires "
                "protocol 'openai-responses'")
        requested_output = profile.get("requested_output_tokens")
        if requested_output is not None:
            if not isinstance(requested_output, int) or isinstance(requested_output, bool) \
                    or requested_output <= 0 or requested_output > max_output:
                raise RegistryError(
                    f"{where}.reasoning_profiles.{rung}.requested_output_tokens must be a "
                    f"positive integer no greater than {max_output}")
        output_parameter = profile.get("output_token_parameter")
        if output_parameter is not None and output_parameter not in (
                "max_tokens", "max_completion_tokens"):
            raise RegistryError(
                f"{where}.reasoning_profiles.{rung}.output_token_parameter must be "
                "'max_tokens' or 'max_completion_tokens'")
        parsed[rung] = dict(profile)
    default = _require(
        raw.get("default_reasoning_profile"), str, f"{where}.default_reasoning_profile")
    if default not in parsed:
        raise RegistryError(f"{where}.default_reasoning_profile {default!r} is not declared")

    capture = _parse_reasoning_capture(where, protocol, raw.get("reasoning_capture"), parsed)
    try:
        transport = normalize_transport_profile(raw.get("transport"))
    except ValueError as exc:
        raise RegistryError(f"{where}.transport is invalid: {exc}") from exc

    return ModelEntry(
        id=model_id,
        provider_model=provider_model,
        protocol=protocol,
        credential=credential,
        fallback_credentials=tuple(fallbacks),
        context_window_tokens=context_window,
        max_output_tokens=max_output,
        token_estimator_id=estimator,
        capabilities_source=source,
        default_reasoning_profile=default,
        reasoning_profiles=parsed,
        reasoning_capture=capture,
        transport=transport,
    )


# A local, uncommitted overlay so testing a new or self-hosted model never edits the released
# registry. Same schema, usually one entry. Entries REPLACE same-named builtin entries and are
# folded into the registry sha, so an overlaid model self-declares in the identity exactly like
# a builtin one. Point CODEACTION_MODEL_REGISTRY_EXTRA elsewhere, or set it empty to disable.
DEFAULT_OVERLAY_PATH = Path.home() / ".config" / "codeaction" / "models.json"


def _overlay_models() -> Dict[str, Any]:
    env = os.environ.get("CODEACTION_MODEL_REGISTRY_EXTRA")
    if env is not None and not env.strip():
        return {}
    overlay_path = Path(env) if env else DEFAULT_OVERLAY_PATH
    if not overlay_path.is_file():
        if env:
            raise RegistryError(f"model registry overlay is unreadable: {overlay_path}")
        return {}
    try:
        raw = json.loads(overlay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"model registry overlay is unreadable: {exc}") from exc
    if raw.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(
            f"model registry overlay schema {raw.get('schema_version')!r} is not "
            f"{REGISTRY_SCHEMA_VERSION!r}")
    models = raw.get("models")
    if not isinstance(models, Mapping) or not models:
        raise RegistryError("model registry overlay declares no models")
    return dict(models)


@lru_cache(maxsize=4)
def load_registry(path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, ModelEntry]]:
    """Read, validate, and cache the registry (plus any local overlay).

    Returns ``(meta, entries_by_id)``. The cache key deliberately ignores the overlay file's
    content; a long-lived process that edits the overlay must call load_registry.cache_clear().
    """
    registry_path = Path(path or os.environ.get("CODEACTION_MODEL_REGISTRY_FILE") or REGISTRY_PATH)
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"model registry is unreadable: {exc}") from exc
    if raw.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(
            f"model registry schema {raw.get('schema_version')!r} is not "
            f"{REGISTRY_SCHEMA_VERSION!r}")
    models = raw.get("models")
    if not isinstance(models, Mapping) or not models:
        raise RegistryError("model registry declares no models")
    overlay = {} if os.environ.get("CODEACTION_MODEL_REGISTRY_FILE") else _overlay_models()
    if overlay:
        models = {**models, **overlay}
    entries = {name: _parse_entry(name, entry) for name, entry in models.items()}
    meta = {
        "registry_version": str(raw.get("registry_version") or ""),
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "sha256": sha256_json(models),
    }
    if not meta["registry_version"]:
        raise RegistryError("model registry needs a registry_version")
    return meta, entries


def reasoning_capture_error(
    entry: ModelEntry,
    rung: Optional[str],
    model_turns: int,
    reasoning_turns: int,
) -> Optional[str]:
    """Say why a finished run failed its own reasoning-capture declaration, or ``None``.

    Kept out of the adapters on purpose: an adapter sees one turn and cannot tell "this turn had
    nothing to think about" from "this vendor is never going to send the text". Only the finished
    run can, so the check lives where the whole run is in hand and is shared by the agent's own
    summary and the controller's attestation.
    """
    if not entry.expects_reasoning(rung):
        return None
    if int(model_turns) <= 0:
        return None
    if int(reasoning_turns) > 0:
        return None
    name = str(rung or entry.default_reasoning_profile)
    return (
        f"model {entry.id!r} declares reasoning capture on rung {name!r} over channel "
        f"{entry.reasoning_capture.get('channel')!r}, but none of the {int(model_turns)} model "
        f"turns recorded any reasoning text")


def registry_identity(path: Optional[str] = None) -> Dict[str, Any]:
    """Version + content digest of the registry, for the comparison identity."""
    meta, _entries = load_registry(path)
    return dict(meta)


def known_models(path: Optional[str] = None) -> Tuple[str, ...]:
    _meta, entries = load_registry(path)
    return tuple(sorted(entries))


def resolve_model(model_id: Any, path: Optional[str] = None) -> ModelEntry:
    """Look up a declared model. Unknown ids fail loudly rather than defaulting."""
    name = str(model_id or "").strip()
    _meta, entries = load_registry(path)
    if name not in entries:
        raise RegistryError(
            f"model {name!r} is not in the registry; declared={list(known_models(path))}. "
            "Add an entry to codeaction/providers/models/registry.json before running it.")
    return entries[name]


def find_model(model_id: Any, path: Optional[str] = None) -> Optional[ModelEntry]:
    """Non-raising lookup, for helpers that must stay usable with test-fixture model names."""
    try:
        return resolve_model(model_id, path)
    except RegistryError:
        return None


# ------------------------------------------------------------------------------------------
# credentials
# ------------------------------------------------------------------------------------------

def _alias_env_prefix(alias: str) -> str:
    return alias.replace("-", "_").upper()


def read_credential_file(path: str | Path, *, allowed_aliases=None) -> Dict[str, str]:
    """Parse and validate the 0600 credential file into an env mapping.

    Only ``<ALIAS>_KEY`` and ``<ALIAS>_BASE_URL`` are accepted, so the file cannot name a model,
    a context window, or any other value that identity depends on.
    """
    credential_path = Path(path)
    try:
        info = credential_path.stat()
    except OSError as exc:
        raise RegistryError("credential file is not readable") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise RegistryError("credential file must be a regular file with mode 0600")
    try:
        lines = credential_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RegistryError("credential file is not readable text") from exc

    permitted = None
    if allowed_aliases is not None:
        permitted = {_alias_env_prefix(str(alias)) for alias in allowed_aliases}
    out: Dict[str, str] = {}
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RegistryError(f"credential file line {number} is invalid")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        match = _CREDENTIAL_KEY_RE.match(key)
        if not match or not value:
            raise RegistryError(
                f"credential key {key!r} is not allowed; use <ALIAS>_KEY or <ALIAS>_BASE_URL")
        if permitted is not None and match.group(1) not in permitted:
            raise RegistryError(
                f"credential key {key!r} names an alias no registered model uses")
        if key in out:
            raise RegistryError(f"credential key {key!r} is duplicated")
        out[key] = value
    return out


def default_credential_path() -> Path:
    """Where both the bare-metal and container paths look for one credential file."""
    override = os.environ.get("CODEACTION_CREDENTIALS")
    return Path(override) if override else Path.home() / ".codeaction" / "credentials"


def load_credentials_into_env(path: str | Path | None = None, *, required: bool = True) -> int:
    """Load the one credential file into ``os.environ``; returns how many keys were set.

    Bare-metal and container runs share this loader so the two can never diverge on what a
    credential file may contain.  Values already exported win, which keeps a one-off override
    possible without editing the file.
    """
    credential_path = Path(path) if path is not None else default_credential_path()
    if not credential_path.exists():
        if required:
            raise RegistryError(
                f"no credential file at {credential_path}; create it with mode 0600 containing "
                "<ALIAS>_KEY lines, or set CODEACTION_CREDENTIALS")
        return 0
    values = read_credential_file(credential_path)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return len(values)


CREDENTIAL_OVERRIDE_ENV = "CODEACTION_CREDENTIAL_OVERRIDES"


def _credential_override(model_id: str, source: Mapping[str, str]) -> Optional[str]:
    """`CODEACTION_CREDENTIAL_OVERRIDES="my-model=alternate-account"`.

    The operator's explicit way to move a model onto another of its declared accounts -- after a
    quota-exhausted attention, say -- without changing which model ran. An override naming an
    alias the model does not declare is refused rather than silently ignored.
    """
    raw = source.get(CREDENTIAL_OVERRIDE_ENV)
    if not raw:
        return None
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RegistryError(
                f"{CREDENTIAL_OVERRIDE_ENV} entries must be <model>=<credential alias>")
        name, alias = (part.strip() for part in item.split("=", 1))
        if name == model_id:
            return alias
    return None


def resolve_credential_alias(
    entry: ModelEntry,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Which of the model's declared accounts this process will actually use.

    Selection order: an explicit operator override, then the declared alias, then each fallback,
    taking the first whose key is present. Returned separately from the secret so callers can
    RECORD the account without touching it -- a run that silently changed endpoints would be the
    same class of defect as one that silently changed instructions.
    """
    if entry.credential is None:
        return None
    source = os.environ if env is None else env
    declared = (entry.credential,) + tuple(entry.fallback_credentials)
    override = _credential_override(entry.id, source)
    if override is not None:
        if override not in declared:
            raise RegistryError(
                f"{CREDENTIAL_OVERRIDE_ENV} names {override!r} for model {entry.id!r}, which "
                f"declares {list(declared)}")
        return override
    for alias in declared:
        if source.get(f"{_alias_env_prefix(alias)}_KEY"):
            return alias
    return entry.credential


def credential_for(
    entry: ModelEntry,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(api_key, base_url)`` for the account resolve_credential_alias selected."""
    if entry.credential is None:
        return None, None
    source = os.environ if env is None else env
    alias = resolve_credential_alias(entry, source)
    prefix = _alias_env_prefix(alias)
    api_key = source.get(f"{prefix}_KEY")
    base_url = source.get(f"{prefix}_BASE_URL")
    if not api_key:
        declared = [entry.credential, *entry.fallback_credentials]
        raise RegistryError(
            f"no credential for model {entry.id!r}: set {prefix}_KEY "
            f"(alias {alias!r}) in the credential file; the model declares {declared}")
    return api_key, base_url or None


__all__ = [
    "ModelEntry",
    "REASONING_CHANNELS",
    "REASONING_RUNGS",
    "REGISTRY_PATH",
    "REGISTRY_SCHEMA_VERSION",
    "RegistryError",
    "SUPPORTED_PROTOCOLS",
    "CREDENTIAL_OVERRIDE_ENV",
    "credential_for",
    "resolve_credential_alias",
    "default_credential_path",
    "load_credentials_into_env",
    "find_model",
    "known_models",
    "load_registry",
    "read_credential_file",
    "reasoning_capture_error",
    "registry_identity",
    "resolve_model",
]
