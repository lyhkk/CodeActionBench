"""Deterministic context preparation for ``codeaction-reference``.

The persisted transcript is never compacted.  This module prepares only the next provider request:
anchors remain intact, recent complete turn units remain verbatim, and older units become a
structured ledger.  Image-bearing records keep a stable text representation in that history while
the currently retained image bytes are projected into one request-local suffix.  No model-generated
summary is used.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from codeaction.runtime.composition import RUN_CODE_RESULT_MAX_IMAGES
from codeaction.providers.model_adapter import (
    CACHE_CHECKPOINT_FIELD,
    ModelCapabilities,
    TOKEN_ESTIMATOR_UTF8_BYTES_V1,
    resolve_output_tokens,
)


CONTEXT_POLICY_ID = "deterministic-ledger"
# 1.1.0: image retention is counted in ROUNDS, not image-bearing messages (see _image_rounds).
# 1.2.0: image eviction and the image token estimate were retuned so that neither fires inside a
# normal episode (retention 2 -> 512, estimate 4096 -> 1024), on the argument that both mechanisms
# REWRITE ALREADY-SENT HISTORY -- `_evict_old_images` replaces an image block with a placeholder,
# and the ledger path drops head units -- and provider prompt caching only hits on a byte-identical
# prefix, so each rewrite invalidates the cache from that position to the end of the request.
# 1.3.0: retention returns to 2 on measured evidence; the estimate stays at 1024. The 1.2.0
# argument optimised the cache HIT RATE and ignored what happens to the denominator. The
# single-factor comparison on 2026-08-10 (gemini-3.6-flash, stack_blocks_three, same seed and
# budget) reads: retention 2 -> 3.86M prompt tokens at 72.0% cached; retention 512 -> 13.67M at
# 94.0% cached, and a final request of 713k tokens. Un-evicted frames are paid for on every later
# turn, so the better ratio bought a 3.5x larger bill. It also cost the ATTEMPT: that run went
# 1.0 -> 0.0. An operation episode's old frames are EXPIRED WORLD STATE, not accumulated evidence
# -- a block photographed on turn 5 has moved by turn 40 -- and the same change moved gpt-5.6 the
# other way, which is what makes retention a capability-relevant scaffold parameter rather than a
# cost knob. It is therefore declared, fixed for every model, and part of the scaffold identity
# (`reference_agent.scaffold_card` -> `config_sha256`), so runs at different retentions can never
# pool. Eviction now DOES fire mid-episode; that is the deliberate behavior, not an oversight.
# 1.4.0: image retention no longer rewrites earlier provider-visible messages. Every image-bearing
# transcript message becomes a stable text record, and the retained image bytes are synthesized in
# one request-local suffix. This preserves an append-only text prefix between turns while retaining
# only two recent image-bearing rounds. Exact duplicate image payloads in the active suffix keep
# their last occurrence and are sent once.
# 1.5.0: each visual-tail label is placed immediately before its corresponding image instead of
# collecting every label before every image. The tail remains wholly after the stable cache prefix,
# so this improves image/text binding without changing cache invalidation or retention behavior.
# 1.6.0: the adjacent label is now the image's permanent obs_id/camera/role/tick record. Image N is
# explicitly request-local; bundle text is kept once in stable history instead of being repeated as
# the identity of every image in that bundle.
# 2.0.0: text compaction protects anchors and the latest exact turn units. Older assistant prose is
# discarded rather than summarized; only deterministic tool arguments/results enter the ledger.
# Under residual pressure, whole older visual-source groups are removed before the newest group.
# 2.1.0: every stable historical image record carries an internal cache-checkpoint candidate.
# Provider adapters translate only the newest supported candidates into their native fields and
# remove the internal marker. The marker is request-only metadata: it never enters the transcript
# or changes the text/image sequence seen by a model.
# 2.2.0: checkpoint candidates track the newest TURN UNITS instead of the newest image records.
# 2.1.0 minted a candidate only where an image round left a stable text record, so a stretch of
# text-only turns minted none: the newest marker stood still while the history grew past it, and
# every one of those turns re-paid the whole growth at full input price. Measured on the
# 2026-08-13 canary (`reference_rank_canary_medium_20260813_5b06a26`), claude-opus-5 went 23 turns
# between candidates and spent 1,144,127 uncached prompt tokens -- 47% of that episode's bill,
# against 4,646,841 tokens read from cache at a tenth of the price; replaying the same per-turn
# request sizes with a per-turn checkpoint prices the input side at $4.15 instead of $9.29.
# qwen3.7-plus is the same bug at its limit: it captured its last image on turn 5, minted three
# candidates in the whole episode, and reported a frozen `cached_tokens` of 8,727 while the prompt
# grew to 129,088. A turn boundary is the right unit because it is where the request grows: the
# uncached remainder is then one turn of tool output rather than an unbounded stretch of them.
# Candidates are bounded by `cache_checkpoint_turn_units` (every explicit-cache provider here caps
# markers at 4) and never land in the request-local visual tail, whose bytes change every turn and
# whose cache write could therefore never be read back.
CONTEXT_POLICY_VERSION = "2.2.0"
IMAGE_POLICY_ID = "reference-images"
IMAGE_POLICY_VERSION = "2.3.0"


VISUAL_SOURCE_NOTE = (
    "[image bytes are supplied only in the request-local visual tail while retained; "
    "load_image(obs_id) inside run_code/run_program can explicitly replay a stored observation]"
)
VISUAL_TAIL_HEADER = (
    "Active visual context (request-local suffix; only retained image-bearing rounds).\n"
    "Image N is local to this request; use obs_id for cross-turn references. "
    "Each permanent obs_id/camera/role/tick label immediately precedes its image; newest is last:"
)


@dataclass(frozen=True)
class ContextPolicy:
    context_policy_id: str = CONTEXT_POLICY_ID
    context_policy_version: str = CONTEXT_POLICY_VERSION
    protocol_overhead_tokens: int = 1024
    ledger_max_units: int = 128
    # The newest complete assistant + tool-result units are exact protected context. If anchors,
    # these units, and the newest visual group do not fit, fail closed instead of summarizing them.
    recent_exact_turn_units: int = 8
    # How many of the most recent IMAGE-BEARING rounds keep their bytes in the request-local visual
    # suffix; every source retains the same stable text record. A WORKING limit again in policy
    # 1.3.0 (see the version note above): the model is meant to act on the freshest view of a scene
    # it keeps changing, and the measured cost of retaining more is both tokens and accuracy.
    # Raising this above the pack's largest budget would switch byte eviction off for every episode
    # -- a different tested unit, not a tuning.
    reference_context_max_image_rounds: int = 2
    # Per ROUND, not per episode: the two limits compose as "up to this many images from one code
    # block, for the last this-many rounds". Raised 2 -> 6 on 2026-08-07 because 2 is below what a
    # single ordinary block produces -- `capture_evidence_views` alone returns three observations --
    # and the batch-collection value position ("code collects, model reviews") is the reason
    # run_code exists.
    run_code_result_max_images: int = RUN_CODE_RESULT_MAX_IMAGES
    # 4096 -> 1024 in policy 1.2.0. The head camera is 640x480, so a real image is ~410 provider
    # tokens; 4096 was ~10x conservative, and over-estimating is not free here. This value only
    # feeds the compaction trigger (`full_estimate <= available`), so a 10x inflation makes the
    # ledger path fire when there is in fact ample window -- and that path drops head units, which
    # breaks the cache prefix exactly like image eviction does. 1024 keeps ~2.5x headroom over the
    # measured cost while no longer manufacturing compactions.
    image_token_estimate: int = 1024
    # How many of the newest turn units end in a cache-checkpoint candidate. 4 is the marker cap
    # that Anthropic Messages, DashScope, and the OpenAI Responses explicit mode all publish, so a
    # larger value would only be sliced away by the adapters; each of them keeps the NEWEST
    # markers, which is what makes the set roll forward. 0 drops the boundary candidates and
    # leaves only the newest stable block-list record -- exactly policy 2.1.0 -- and is kept as
    # the control arm for the A/B that justified this policy. Either way this is a pure billing
    # switch: it changes nothing the model reads.
    cache_checkpoint_turn_units: int = 4

    def __post_init__(self):
        for name in (
            "protocol_overhead_tokens", "ledger_max_units", "recent_exact_turn_units",
            "reference_context_max_image_rounds",
            "run_code_result_max_images", "image_token_estimate",
            "cache_checkpoint_turn_units",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContextView:
    messages: Tuple[Dict[str, object], ...]
    estimated_input_tokens: int
    available_input_tokens: int
    requested_output_tokens: int
    effective_output_tokens: int
    compacted: bool
    text_compacted: bool
    kept_message_ids: Tuple[str, ...]
    dropped_message_ids: Tuple[str, ...]
    evicted_image_message_ids: Tuple[str, ...]
    protected_context_estimated_tokens: int
    recent_exact_context_estimated_tokens: int
    text_history_estimated_tokens: int
    visual_tail_estimated_tokens: int
    visual_tail_image_count: int
    visual_tail_source_message_ids: Tuple[str, ...]
    visual_tail_duplicate_images_suppressed: int
    visual_tail_trimmed_for_budget: bool
    text_compaction_ledger_entries: int
    protected_units_eroded: int
    force_compact_unreducible: bool
    ledger: Mapping[str, object]

    def event(self) -> dict:
        return {
            "context_policy": f"{CONTEXT_POLICY_ID}@{CONTEXT_POLICY_VERSION}",
            "estimated_input_tokens": self.estimated_input_tokens,
            "available_input_tokens": self.available_input_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "effective_output_tokens": self.effective_output_tokens,
            "compacted": self.compacted,
            "text_compacted": self.text_compacted,
            "kept_message_ids": list(self.kept_message_ids),
            "dropped_message_ids": list(self.dropped_message_ids),
            "evicted_image_message_ids": list(self.evicted_image_message_ids),
            "protected_context_estimated_tokens": (
                self.protected_context_estimated_tokens),
            "recent_exact_context_estimated_tokens": (
                self.recent_exact_context_estimated_tokens),
            "text_history_estimated_tokens": self.text_history_estimated_tokens,
            "visual_tail_estimated_tokens": self.visual_tail_estimated_tokens,
            "visual_tail_image_count": self.visual_tail_image_count,
            "visual_tail_source_message_ids": list(
                self.visual_tail_source_message_ids),
            "visual_tail_duplicate_images_suppressed": (
                self.visual_tail_duplicate_images_suppressed),
            "visual_tail_trimmed_for_budget": self.visual_tail_trimmed_for_budget,
            "text_compaction_ledger_entries": self.text_compaction_ledger_entries,
            "protected_units_eroded": self.protected_units_eroded,
            "force_compact_unreducible": self.force_compact_unreducible,
            "ledger": dict(self.ledger),
        }


class ContextBudgetError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _message_id(message: Mapping[str, Any], index: int) -> str:
    value = message.get("_message_id")
    return str(value) if value else f"m{index:06d}"


def _strip_internal(message: Mapping[str, Any]) -> Dict[str, object]:
    return {
        str(key): deepcopy(value)
        for key, value in message.items()
        if not str(key).startswith("_")
    }


def _image_blocks(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        1 for item in content
        if isinstance(item, Mapping) and item.get("type") == "image_url"
    )


def _image_identity(block: Mapping[str, Any]) -> Any:
    """Exact identity without copying a multi-megabyte data URI into canonical JSON bytes."""
    value = block.get("image_url")
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), str(item)) for key, item in value.items()))
    return str(value)


def _image_rounds(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    """Round index per message. A round opens at each assistant turn and covers everything
    appended in response to it.

    Retention is counted in rounds, not in image-bearing messages, because one round emits a
    different number of messages depending on which tool produced the images: an ObservationSet
    of three views arrives as ONE message, while a run_code that captured two images arrives as
    TWO (`attach_images` appends one message per ImageGroup). Counting messages therefore
    charges the same round a different price by tool, and a single two-capture run_code would
    evict the whole preceding round.
    """
    rounds, current = [], 0
    for message in messages:
        if message.get("role") == "assistant":
            current += 1
        rounds.append(current)
    return rounds


def _visual_tail_projection(
    messages: Sequence[Mapping[str, Any]],
    max_image_rounds: int,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    Tuple[str, ...],
    Tuple[str, ...],
    int,
    Tuple[str, ...],
]:
    """Return stable text history plus one request-local suffix containing retained images.

    The canonical/transcript messages stay untouched.  Every provider request receives the same
    text representation for an already-recorded image message, regardless of whether its bytes are
    still inside the retention window.  This is the property that lets prompt caches retain the
    old prefix when the visual window advances.
    """
    copied = [deepcopy(dict(message)) for message in messages]
    image_indices = [
        index for index, message in enumerate(copied)
        if _image_blocks(message.get("content"))
    ]
    if max_image_rounds:
        rounds = _image_rounds(copied)
        keep = set(sorted({rounds[index] for index in image_indices})[-max_image_rounds:])
        retain = [index for index in image_indices if rounds[index] in keep]
    else:
        retain = []
    retain_set = set(retain)
    evicted_ids = tuple(
        _message_id(copied[index], index)
        for index in image_indices if index not in retain_set
    )

    # Materialize every candidate before deduplication so the last occurrence wins without
    # disturbing chronological order among the surviving images.
    candidates = []
    for index in retain:
        message = copied[index]
        content = message.get("content")
        source_label = ""
        labelled_blocks = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "text" and str(item.get("text") or "").strip():
                source_label = str(item.get("text") or "").strip()
            elif item.get("type") == "image_url":
                labelled_blocks.append((source_label, deepcopy(dict(item))))
        for ordinal, (label, block) in enumerate(labelled_blocks, 1):
            candidates.append({
                "source_index": index,
                "source_id": _message_id(message, index),
                "source_label": label or _message_id(message, index),
                "group_ordinal": ordinal,
                "group_size": len(labelled_blocks),
                "block": block,
                "key": _image_identity(block),
            })

    seen = set()
    retained_reversed = []
    for candidate in reversed(candidates):
        if candidate["key"] in seen:
            continue
        seen.add(candidate["key"])
        retained_reversed.append(candidate)
    selected = list(reversed(retained_reversed))
    duplicates_suppressed = len(candidates) - len(selected)

    # Strip image bytes from every historical message, including currently retained sources.
    # The note is added once and never changes as the retention window advances.
    for index in image_indices:
        message = copied[index]
        content = message.get("content")
        kept = [
            deepcopy(item) for item in content
            if not (isinstance(item, Mapping) and item.get("type") == "image_url")
        ]
        # The note is the last stable block of an observation round, which makes it the natural
        # marker carrier when this message ends a turn unit. Policy 2.2.0 no longer marks it here:
        # candidates are minted per turn boundary by `_mark_cache_checkpoints`, which reuses this
        # block when the boundary happens to be an image record and would otherwise mint none for
        # a text-only turn. Marking every record unconditionally is what let the newest candidate
        # fall arbitrarily far behind the end of the request.
        kept.append({"type": "text", "text": VISUAL_SOURCE_NOTE})
        message["content"] = kept

    tail = []
    source_ids = []
    if selected:
        tail_content = [{"type": "text", "text": VISUAL_TAIL_HEADER}]
        for position, candidate in enumerate(selected, 1):
            source_id = str(candidate["source_id"])
            if source_id not in source_ids:
                source_ids.append(source_id)
            tail_content.extend([
                {"type": "text", "text": (
                    f"Image {position} | {candidate['source_label']}")},
                candidate["block"],
            ])
        tail = [{
            "_message_id": "visual-tail",
            "role": "user",
            "content": tail_content,
        }]
    return (
        copied,
        tail,
        evicted_ids,
        tuple(source_ids),
        duplicates_suppressed,
        tuple(str(candidate["source_id"]) for candidate in selected),
    )


def _estimate_value(value: Any, *, image_token_estimate: int) -> int:
    """Conservative versioned estimator: one UTF-8 byte per text token plus fixed image cost."""
    if isinstance(value, Mapping):
        if value.get("type") == "image_url":
            redacted = dict(value)
            redacted["image_url"] = {"url": "[native-image]"}
            return len(_canonical_bytes(redacted)) + int(image_token_estimate)
        return len(_canonical_bytes({
            str(key): (
                "[nested]" if isinstance(item, (Mapping, list)) else item)
            for key, item in value.items()
        })) + sum(
            _estimate_value(item, image_token_estimate=image_token_estimate)
            for item in value.values()
            if isinstance(item, (Mapping, list))
        )
    if isinstance(value, list):
        return 2 + sum(
            _estimate_value(item, image_token_estimate=image_token_estimate) + 1
            for item in value
        )
    return len(_canonical_bytes(value))


def estimate_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    estimator_id: str,
    image_token_estimate: int,
) -> int:
    if estimator_id != TOKEN_ESTIMATOR_UTF8_BYTES_V1:
        raise ValueError(f"unsupported context estimator {estimator_id!r}")
    return sum(
        _estimate_value(_strip_internal(message), image_token_estimate=image_token_estimate)
        for message in messages
    )


def _anchors(messages: Sequence[Mapping[str, Any]]) -> Tuple[List[int], List[int]]:
    explicit = [
        index for index, message in enumerate(messages)
        if message.get("_context_anchor") is True
    ]
    if explicit:
        anchor_set = set(explicit)
        return explicit, [index for index in range(len(messages)) if index not in anchor_set]

    anchors = []
    task_seen = False
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system" and not task_seen:
            anchors.append(index)
        elif role == "user" and not task_seen:
            anchors.append(index)
            task_seen = True
        else:
            break
    anchor_set = set(anchors)
    return anchors, [index for index in range(len(messages)) if index not in anchor_set]


def _turn_units(messages: Sequence[Mapping[str, Any]], indices: Iterable[int]) -> List[List[int]]:
    units: List[List[int]] = []
    current: List[int] = []
    for index in indices:
        if messages[index].get("role") == "assistant" and current:
            units.append(current)
            current = []
        current.append(index)
    if current:
        units.append(current)
    return units


def _bounded_text(value: Any, max_bytes: int) -> dict:
    text = str(value or "")
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return {"text": text, "original_bytes": len(raw), "omitted_bytes": 0}
    head_budget = max_bytes // 2
    tail_budget = max_bytes - head_budget
    head = raw[:head_budget].decode("utf-8", errors="ignore")
    tail = raw[-tail_budget:].decode("utf-8", errors="ignore")
    kept = len(head.encode("utf-8")) + len(tail.encode("utf-8"))
    return {
        "head": head,
        "tail": tail,
        "original_bytes": len(raw),
        "omitted_bytes": len(raw) - kept,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


_FACT_FIELDS = (
    "ok", "status", "call_status", "error", "failure", "abort_reason",
    "action_id", "tick", "obs_id", "obs_ids", "set_id", "pair_id",
    "tool_calls", "result_assigned", "namespace_reset", "filesystem_policy",
    "commanded", "achieved", "resulting_pose", "observed_after",
    "arm", "arms", "ee_pose", "tcp_pose", "orientation",
    "opening_m", "finger_gap_m", "gripper_width", "gripper_val",
    "drive_commanded_closed", "contact", "contacts", "contact_evidence",
    "distance", "distance_m", "measurement", "measurements", "uncertainty",
    "coarse", "validity", "execution", "planning", "recovery", "transition",
    "interrupted_action", "internal_trace", "index", "tool", "args", "result",
)


def _fact_projection(value: Any) -> Any:
    """Keep exact decision facts from one Harness result; never generate a summary."""
    if isinstance(value, Mapping):
        out = {}
        for key in _FACT_FIELDS:
            if key not in value:
                continue
            item = value[key]
            out[key] = (
                _fact_projection(item)
                if key in ("internal_trace", "result") else deepcopy(item)
            )
        return out
    if isinstance(value, list):
        return [_fact_projection(item) for item in value]
    return deepcopy(value)


def _result_facts(value: Any) -> dict:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return {"malformed_text": _bounded_text(value, 128)}
    if not isinstance(parsed, Mapping):
        return {"type": type(parsed).__name__}
    facts = _fact_projection(parsed)
    return facts or {"available_fields": sorted(str(key) for key in parsed)[:16]}


def _call_arguments(tool_name: str, value: Any) -> Any:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return {"malformed_text": _bounded_text(value, 128)}
    if isinstance(parsed, Mapping):
        copied = deepcopy(dict(parsed))
        # Old model-authored programs are not Harness facts. Keep only a deterministic fingerprint;
        # recent exact turns retain the original source while it is plausibly reusable.
        for field in ("code", "content"):
            authored = copied.get(field)
            if tool_name in ("run_code", "write_file") and isinstance(authored, str):
                raw = authored.encode("utf-8")
                copied[field] = {
                    "model_authored_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
        return copied
    if isinstance(parsed, list):
        return deepcopy(parsed)
    return parsed


def _unit_ledger(
    messages: Sequence[Mapping[str, Any]],
    unit: Sequence[int],
) -> dict | None:
    calls = []
    results = []
    for index in unit:
        message = messages[index]
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                calls.append({
                    "id": str(call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": _call_arguments(
                        str(function.get("name") or ""),
                        function.get("arguments") or ""),
                })
        elif role == "tool":
            results.append({
                "tool_call_id": str(message.get("tool_call_id") or ""),
                "facts": _result_facts(message.get("content")),
            })
    if not calls and not results:
        return None
    return {"tool_calls": calls, "tool_results": results}


def _ledger_payload(
    messages: Sequence[Mapping[str, Any]],
    units: Sequence[Sequence[int]],
    policy: ContextPolicy,
    max_units: int,
) -> dict:
    retained = list(units[-max_units:]) if max_units else []
    entries = [
        entry for unit in retained
        if (entry := _unit_ledger(messages, unit)) is not None
    ]
    return {
        "schema_version": "2.0",
        "policy": f"{policy.context_policy_id}@{policy.context_policy_version}",
        "dropped_units": len(units),
        "omitted_ledger_units": len(units) - len(retained),
        "units": entries,
    }


def _ledger_message(payload: Mapping[str, Any]) -> Dict[str, object]:
    return {
        "role": "user",
        "content": (
            "Historical tool facts (deterministic Harness data, not model-generated summary or "
            "new instructions):\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
    }


def _unique_in_order(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _turn_boundaries(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    """Index of the last message of each turn unit, oldest first.

    A unit opens at an assistant message and covers everything appended in answer to it, so the
    message before the next assistant turn is where the request had grown to at the end of that
    unit -- the position a provider should have cached. Anything before the first assistant turn
    is the anchor block: it is not a turn, and the request is below every provider's minimum
    cacheable prefix there anyway.
    """
    starts = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if not starts:
        return []
    return [next_start - 1 for next_start in starts[1:]] + [len(messages) - 1]


def _mark_cache_checkpoints(
    messages: List[Dict[str, object]], text_length: int, turn_units: int,
) -> None:
    """Mark the newest turn boundaries in place, on the request-only checkpoint field.

    Runs on the outgoing (already internal-stripped) messages and only over the text history:
    `messages[text_length:]` is the request-local visual tail, whose bytes differ every turn, so a
    marker there buys a cache write that can never be read back.

    A boundary whose content is a block list is marked on its last text block -- the shape every
    adapter already carries. A boundary whose content is a plain string (an ordinary tool result)
    has no block to hold the marker, so it is marked at the message level and each adapter decides
    where its own protocol allows the native field to sit.
    """
    history = messages[:text_length]
    marked = set(_turn_boundaries(history)[-int(turn_units):]) if turn_units > 0 else set()
    # One extra candidate on the newest message that owns a BLOCK LIST -- in practice the newest
    # stable image record, or the task anchor before the first observation. It is redundant for
    # Anthropic and DashScope, which can mark a tool result directly and will slice it off as the
    # oldest candidate. It is the only realizable position on the OpenAI Responses protocol, where
    # a historical tool result is a top-level `function_call_output` item: the schema accepts a
    # breakpoint there and the service then never writes the cache (reported on GPT-5.6, still
    # open), which is worse than not marking it. Without this, a long text-only stretch would
    # leave that protocol with no explicit breakpoint at all.
    for index in range(len(history) - 1, -1, -1):
        if isinstance(history[index].get("content"), list):
            marked.add(index)
            break
    for index in sorted(marked):
        message = messages[index]
        content = message.get("content")
        if isinstance(content, list):
            carrier = next(
                (item for item in reversed(content)
                 if isinstance(item, dict) and item.get("type") == "text"),
                None,
            )
            if carrier is not None:
                carrier[CACHE_CHECKPOINT_FIELD] = True
                continue
        message[CACHE_CHECKPOINT_FIELD] = True


def _trim_oldest_visual_source(
    visual_tail: Sequence[Mapping[str, Any]],
    image_source_ids: Sequence[str],
) -> Tuple[List[Dict[str, object]], Tuple[str, ...], bool]:
    """Remove one whole oldest source group, while always retaining the newest source group."""
    ordered_sources = _unique_in_order(image_source_ids)
    if len(ordered_sources) <= 1 or not visual_tail:
        return [deepcopy(dict(message)) for message in visual_tail], tuple(image_source_ids), False
    drop_source = ordered_sources[0]
    content = visual_tail[0].get("content")
    if not isinstance(content, list) or len(content) != 1 + 2 * len(image_source_ids):
        raise ContextBudgetError("visual tail shape does not match image-source telemetry")
    kept_content = [deepcopy(content[0])]
    kept_sources = []
    for offset, source_id in enumerate(image_source_ids):
        pair_at = 1 + 2 * offset
        if source_id == drop_source:
            continue
        kept_content.extend([deepcopy(content[pair_at]), deepcopy(content[pair_at + 1])])
        kept_sources.append(str(source_id))
    return ([{
        **deepcopy(dict(visual_tail[0])),
        "content": kept_content,
    }], tuple(kept_sources), True)


class ContextManager:
    def __init__(self, policy: ContextPolicy = ContextPolicy()):
        self.policy = policy

    def prepare(
        self,
        messages: Sequence[Mapping[str, Any]],
        capabilities: ModelCapabilities,
        requested_output_tokens: int,
        *,
        force_compact: bool = False,
    ) -> ContextView:
        effective = resolve_output_tokens(capabilities, requested_output_tokens)
        available = (
            int(capabilities.context_window_tokens)
            - effective
            - int(self.policy.protocol_overhead_tokens)
        )
        if available <= 0:
            raise ContextBudgetError("output reserve and protocol overhead exhaust context window")

        (
            text_history,
            visual_tail,
            evicted,
            visual_source_ids,
            duplicate_images,
            visual_image_source_ids,
        ) = _visual_tail_projection(
            messages, self.policy.reference_context_max_image_rounds)
        anchor_indices, remaining_indices = _anchors(text_history)
        units = _turn_units(text_history, remaining_indices)
        anchor_messages = [text_history[index] for index in anchor_indices]
        recent_count = min(len(units), int(self.policy.recent_exact_turn_units))
        recent_units = units[-recent_count:] if recent_count else []
        compactable_count = len(units) - recent_count

        def estimate(candidate):
            return estimate_messages(
                candidate,
                estimator_id=capabilities.token_estimator_id,
                image_token_estimate=self.policy.image_token_estimate,
            )

        protected_estimate = estimate(anchor_messages)
        recent_exact_estimate = estimate([
            text_history[index]
            for unit in recent_units
            for index in unit
        ])
        visual_tail_estimate = estimate(visual_tail)
        full_text_estimate = estimate(text_history)
        full_candidate = text_history + visual_tail
        full_estimate = full_text_estimate + visual_tail_estimate
        if full_estimate <= available and not force_compact:
            ids = tuple(
                _message_id(message, index)
                for index, message in enumerate(text_history)
            )
            outgoing = [_strip_internal(message) for message in full_candidate]
            # After the estimate and after stripping: the markers are request-only metadata that
            # every adapter consumes and removes, so they must not enter the budget arithmetic
            # that decides compaction, and `_strip_internal` must not take them back out.
            _mark_cache_checkpoints(
                outgoing, len(text_history), self.policy.cache_checkpoint_turn_units)
            return ContextView(
                messages=tuple(outgoing),
                estimated_input_tokens=full_estimate,
                available_input_tokens=available,
                requested_output_tokens=int(requested_output_tokens),
                effective_output_tokens=effective,
                # `compacted` keeps its historical meaning -- ANY reduction of the recorded
                # conversation, image eviction included. `text_compacted` is the narrower new fact.
                compacted=bool(evicted),
                text_compacted=False,
                kept_message_ids=ids,
                dropped_message_ids=(),
                evicted_image_message_ids=evicted,
                protected_context_estimated_tokens=protected_estimate,
                recent_exact_context_estimated_tokens=recent_exact_estimate,
                text_history_estimated_tokens=full_text_estimate,
                visual_tail_estimated_tokens=visual_tail_estimate,
                visual_tail_image_count=sum(
                    _image_blocks(message.get("content")) for message in visual_tail),
                visual_tail_source_message_ids=visual_source_ids,
                visual_tail_duplicate_images_suppressed=duplicate_images,
                visual_tail_trimmed_for_budget=False,
                text_compaction_ledger_entries=0,
                protected_units_eroded=0,
                force_compact_unreducible=False,
                ledger={},
            )

        drop_count = 1 if force_compact and compactable_count else 0
        max_ledger = min(compactable_count, int(self.policy.ledger_max_units))
        active_visual_tail = [deepcopy(dict(message)) for message in visual_tail]
        active_image_source_ids = tuple(visual_image_source_ids)
        visual_tail_trimmed = False
        unreducible = False
        # The floor this method never goes below: the anchors, the newest turn unit, and the newest
        # visual source group. Everything above the floor is reducible, protected units included.
        max_droppable = max(compactable_count, len(units) - 1)
        while True:
            dropped_units = units[:drop_count]
            kept_units = units[drop_count:]
            ledger = _ledger_payload(
                text_history, dropped_units, self.policy, max_ledger)
            ledger_messages = [_ledger_message(ledger)] if ledger["units"] else []
            candidate_text = (
                anchor_messages
                + ledger_messages
                + [
                    text_history[index]
                    for unit in kept_units
                    for index in unit
                ]
            )
            text_estimate = estimate(candidate_text)
            visual_tail_estimate = estimate(active_visual_tail)
            candidate = candidate_text + active_visual_tail
            estimated = text_estimate + visual_tail_estimate
            request_changed = bool(dropped_units) or visual_tail_trimmed
            # Which ordinary rungs of the ladder below can still make this request smaller.
            # Erosion of the protected units is deliberately NOT counted: it is reserved for a
            # request that genuinely does not fit, never spent to satisfy `force_compact` alone.
            reducible = (
                drop_count < compactable_count
                or max_ledger > 0
                or len(_unique_in_order(active_image_source_ids)) > 1
            )
            if estimated <= available and (
                    not force_compact or request_changed or not reducible):
                # `force_compact` is set only after the provider rejected the previous request.
                # When nothing is left to remove, resending identical bytes would just buy the
                # same rejection, so the caller is told to stop rather than pay for the retry.
                unreducible = force_compact and not request_changed
                break
            if drop_count < compactable_count:
                drop_count += 1
                continue
            if max_ledger > 0:
                max_ledger = max_ledger // 2
                continue
            active_visual_tail, active_image_source_ids, trimmed = (
                _trim_oldest_visual_source(
                    active_visual_tail, active_image_source_ids)
            )
            if trimmed:
                visual_tail_trimmed = True
                continue
            if estimated > available and drop_count < max_droppable:
                # Only reachable once the request does not fit even after every other reduction.
                # Eroding the oldest PROTECTED unit is strictly better than failing the episode:
                # its tool calls and Harness facts still enter the ledger, and the newest unit and
                # the anchors are never touched. Losing an old exact turn beats losing the run.
                drop_count += 1
                # Erosion is a reduction, not a deletion. Budget pressure may already have shrunk
                # the ledger to nothing; grant back exactly the slots the eroded units need, which
                # are the newest entries of `dropped_units` and therefore the ones retained.
                max_ledger = max(max_ledger, drop_count - compactable_count)
                continue
            raise ContextBudgetError(
                f"protected anchors, the newest turn unit, and the newest visual source require "
                f"{estimated} tokens; available input budget is {available}")

        kept_indices = anchor_indices + [
            index for unit in kept_units for index in unit
        ]
        dropped_indices = [
            index for unit in dropped_units for index in unit
        ]
        kept_ids = tuple(
            _message_id(text_history[index], index) for index in kept_indices)
        dropped_ids = tuple(
            _message_id(text_history[index], index) for index in dropped_indices)
        active_visual_source_ids = _unique_in_order(active_image_source_ids)
        text_compacted = bool(dropped_ids)
        protected_units_eroded = max(0, drop_count - compactable_count)
        outgoing = [_strip_internal(message) for message in candidate]
        _mark_cache_checkpoints(
            outgoing, len(candidate_text), self.policy.cache_checkpoint_turn_units)
        return ContextView(
            messages=tuple(outgoing),
            estimated_input_tokens=estimated,
            available_input_tokens=available,
            requested_output_tokens=int(requested_output_tokens),
            effective_output_tokens=effective,
            compacted=bool(evicted) or text_compacted,
            text_compacted=text_compacted,
            kept_message_ids=kept_ids,
            dropped_message_ids=dropped_ids,
            evicted_image_message_ids=evicted,
            protected_context_estimated_tokens=protected_estimate,
            recent_exact_context_estimated_tokens=recent_exact_estimate,
            text_history_estimated_tokens=text_estimate,
            visual_tail_estimated_tokens=visual_tail_estimate,
            visual_tail_image_count=sum(
                _image_blocks(message.get("content")) for message in active_visual_tail),
            visual_tail_source_message_ids=active_visual_source_ids,
            visual_tail_duplicate_images_suppressed=duplicate_images,
            visual_tail_trimmed_for_budget=visual_tail_trimmed,
            text_compaction_ledger_entries=len(ledger["units"]),
            protected_units_eroded=protected_units_eroded,
            force_compact_unreducible=unreducible,
            ledger=ledger,
        )
