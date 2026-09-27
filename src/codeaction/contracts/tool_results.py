"""Deterministic model-visible tool-result projection shared by local and MCP adapters."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Tuple

from codeaction.runtime.composition import (MODEL_VISIBLE_RESULT_MAX_BYTES,
                                 RUN_CODE_RESULT_MAX_IMAGES)
from codeaction.contracts.result_contracts import validate_result
from codeaction.interface.schemas import json_safe, serialize
from codeaction.contracts.types import ActionResult, Observation, ObservationPair, ObservationSet


TOOL_RESULT_POLICY_ID = "structured-json-projection"
# 1.1.0: one run_code/run_program block attaches each referenced observation at most once.
# Geometry helpers carry their source obs_id for provenance, so a block that projects several
# points from one frame used to send the same PNG several times and let those duplicates consume
# the image cap.
# 1.2.0: every projected image carries its own stable obs_id/camera/role/tick label. Group-level
# bundle text remains separate and is no longer repeated as though it described each image.
# 2.0.0: recoverable contact ActionResults carry a standard post-settle Observation image. This
# changes which tool results create image-bearing turns, not the context-retention algorithm.
# 2.1.0: that image's payload marker now goes through the same transport rewrite as every other
# single Observation, so the inline transport stops describing a follow-up message.
TOOL_RESULT_POLICY_VERSION = "2.1.0"
FOLLOWUP_IMAGE_TRANSPORT = "followup_user_message"
INLINE_IMAGE_TRANSPORT = "inline_tool_result"


@dataclass(frozen=True)
class ImageGroup:
    text: str
    refs: Tuple[str, ...]
    labels: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.labels and len(self.labels) != len(self.refs):
            raise ValueError("ImageGroup labels must be empty or match refs one-for-one")


@dataclass(frozen=True)
class ToolProjection:
    payload: dict
    full_payload: dict
    image_groups: Tuple[ImageGroup, ...] = ()
    model_image_refs: Tuple[str, ...] = ()
    all_image_refs: Tuple[str, ...] = ()
    all_observation_ids: Tuple[str, ...] = ()
    full_result: Any = None
    truncated: bool = False
    original_bytes: int = 0
    model_bytes: int = 0
    duplicate_observation_ids_suppressed: int = 0
    policy: str = f"{TOOL_RESULT_POLICY_ID}@{TOOL_RESULT_POLICY_VERSION}"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _utf8_edge(text: str, limit: int, *, tail=False) -> str:
    raw = text.encode("utf-8")
    selected = raw[-limit:] if tail else raw[:limit]
    return selected.decode("utf-8", errors="ignore")


def _project_value(value: Any, *, string_limit: int, array_limit: int,
                   object_limit: int) -> Any:
    if isinstance(value, str):
        raw = value.encode("utf-8")
        if len(raw) <= string_limit:
            return value
        head_limit = max(1, string_limit // 2)
        tail_limit = max(1, string_limit - head_limit)
        head = _utf8_edge(value, head_limit)
        tail = _utf8_edge(value, tail_limit, tail=True)
        kept = len(head.encode("utf-8")) + len(tail.encode("utf-8"))
        return {
            "__codeaction_truncated_string__": {
                "head": head,
                "tail": tail,
                "original_bytes": len(raw),
                "omitted_bytes": len(raw) - kept,
            }
        }
    if isinstance(value, list):
        if len(value) <= array_limit:
            return [
                _project_value(
                    item, string_limit=string_limit, array_limit=array_limit,
                    object_limit=object_limit)
                for item in value
            ]
        prefix_items = max(1, array_limit // 2)
        suffix_items = max(1, array_limit - prefix_items)
        prefix = value[:prefix_items]
        suffix = value[-suffix_items:]
        return {
            "__codeaction_truncated_array__": {
                "prefix": [
                    _project_value(
                        item, string_limit=string_limit, array_limit=array_limit,
                        object_limit=object_limit)
                    for item in prefix
                ],
                "suffix": [
                    _project_value(
                        item, string_limit=string_limit, array_limit=array_limit,
                        object_limit=object_limit)
                    for item in suffix
                ],
                "original_items": len(value),
                "omitted_items": len(value) - len(prefix) - len(suffix),
            }
        }
    if isinstance(value, dict):
        keys = sorted(value, key=str)
        omitted = []
        if len(keys) > object_limit:
            prefix_fields = max(1, object_limit // 2)
            suffix_fields = max(1, object_limit - prefix_fields)
            retained = keys[:prefix_fields] + keys[-suffix_fields:]
            retained_set = set(retained)
            omitted = [key for key in keys if key not in retained_set]
            keys = retained
        out = {
            str(key): _project_value(
                value[key], string_limit=string_limit, array_limit=array_limit,
                object_limit=object_limit)
            for key in keys
        }
        if omitted:
            out["__codeaction_truncated_fields__"] = {
                "original_fields": len(value),
                "omitted_fields": len(omitted),
                "omitted_keys_sha256": hashlib.sha256(
                    _canonical_bytes([str(key) for key in omitted])).hexdigest(),
            }
        return out
    return value


def bounded_json_projection(payload: dict, max_bytes: int = MODEL_VISIBLE_RESULT_MAX_BYTES):
    """Return valid JSON data no larger than ``max_bytes`` and explicit projection metadata."""
    if not isinstance(payload, dict):
        raise TypeError("tool payload must be an object")
    ceiling = int(max_bytes)
    if ceiling < 512:
        raise ValueError("model-visible tool-result ceiling must be at least 512 bytes")
    full = json_safe(deepcopy(payload))
    original = _canonical_bytes(full)
    if len(original) <= ceiling:
        return full, {
            "truncated": False,
            "original_bytes": len(original),
            "model_bytes": len(original),
            "omitted_bytes": 0,
            "policy": f"{TOOL_RESULT_POLICY_ID}@{TOOL_RESULT_POLICY_VERSION}",
            "max_bytes": ceiling,
        }

    candidates = (
        (8192, 256, 256), (4096, 128, 128), (2048, 64, 64), (1024, 32, 32),
        (512, 16, 16), (256, 8, 8), (128, 4, 4), (64, 2, 2), (32, 1, 1),
    )
    projected = None
    encoded = b""
    for string_limit, array_limit, object_limit in candidates:
        projected = _project_value(
            full, string_limit=string_limit, array_limit=array_limit,
            object_limit=object_limit)
        encoded = _canonical_bytes(projected)
        if len(encoded) <= ceiling:
            break
    if len(encoded) > ceiling:
        projected = {
            "__codeaction_truncated_value__": {
                "type": "object",
                "original_bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            }
        }
        encoded = _canonical_bytes(projected)
    if len(encoded) > ceiling:
        raise ValueError("model-visible result ceiling cannot hold truncation metadata")
    return projected, {
        "truncated": True,
        "original_bytes": len(original),
        "model_bytes": len(encoded),
        "omitted_bytes": max(0, len(original) - len(encoded)),
        "policy": f"{TOOL_RESULT_POLICY_ID}@{TOOL_RESULT_POLICY_VERSION}",
        "max_bytes": ceiling,
    }


def payload_projection(payload: dict, *, max_bytes=MODEL_VISIBLE_RESULT_MAX_BYTES) -> ToolProjection:
    """Projection for an error/control payload that has no typed result or images."""
    full = json_safe(deepcopy(payload))
    model_payload, meta = bounded_json_projection(full, max_bytes)
    return ToolProjection(
        payload=model_payload,
        full_payload=full,
        truncated=meta["truncated"],
        original_bytes=meta["original_bytes"],
        model_bytes=meta["model_bytes"],
    )


def _direct_groups(result_obj: Any) -> Tuple[ImageGroup, ...]:
    if isinstance(result_obj, Observation):
        return (ImageGroup(
            "Observation",
            (result_obj.image_ref,),
            (f"{result_obj.obs_id} | {result_obj.camera} | role=observation | "
             f"tick={result_obj.tick}",),
        ),)
    if isinstance(result_obj, ObservationSet):
        inverse_roles = {}
        for role, obs_id in result_obj.roles.items():
            inverse_roles.setdefault(obs_id, []).append(role)
        labels = []
        refs = []
        for obs in result_obj.observations:
            roles = ",".join(sorted(inverse_roles.get(obs.obs_id, []))) or "unlabelled"
            labels.append(
                f"{obs.obs_id} | {obs.camera} | role={roles} | tick={obs.tick}")
            refs.append(obs.image_ref)
        return (ImageGroup(
            f"Observation bundle | set_id={result_obj.set_id} | tick={result_obj.tick}",
            tuple(refs),
            tuple(labels),
        ),)
    if isinstance(result_obj, ObservationPair):
        return (ImageGroup(
            f"Observation pair | pair_id={result_obj.pair_id}",
            (result_obj.before.image_ref, result_obj.after.image_ref),
            (
                f"{result_obj.before.obs_id} | {result_obj.before.camera} | role=before | "
                f"tick={result_obj.before.tick}",
                f"{result_obj.after.obs_id} | {result_obj.after.camera} | role=after | "
                f"tick={result_obj.after.tick}",
            ),
        ),)
    return ()


def _direct_observation_ids(result_obj: Any) -> Tuple[str, ...]:
    if isinstance(result_obj, Observation):
        return (result_obj.obs_id,)
    if isinstance(result_obj, ObservationSet):
        return tuple(obs.obs_id for obs in result_obj.observations)
    if isinstance(result_obj, ObservationPair):
        return (result_obj.before.obs_id, result_obj.after.obs_id)
    return ()


def _inline_payload(payload: dict, result_obj: Any) -> None:
    """Preserve the MCP bridge's pre-extraction image-marker wording byte-for-byte."""
    if isinstance(result_obj, Observation):
        payload["image"] = "attached as an image block in this tool result"
    elif isinstance(result_obj, ObservationSet):
        payload["images"] = "attached as image blocks in this tool result"
        for obs in payload.get("observations", []):
            obs["image"] = "attached as an image block in this tool result"
    elif isinstance(result_obj, ObservationPair):
        payload["images"] = "attached as image blocks in this tool result"
        payload["before"]["image"] = "attached as an image block in this tool result"
        payload["after"]["image"] = "attached as an image block in this tool result"


def _unique_by_last_occurrence(values: Iterable[str]) -> Tuple[str, ...]:
    """Return unique strings ordered by their last occurrence in ``values``.

    The image cap is a tail cap.  If code captures one frame, uses it early, then explicitly uses
    it again after several other frames, that last use keeps it in the recent tail without sending
    the same bytes twice.
    """
    materialized = tuple(str(value) for value in values)
    seen = set()
    reverse_unique = []
    for value in reversed(materialized):
        if value in seen:
            continue
        seen.add(value)
        reverse_unique.append(value)
    return tuple(reversed(reverse_unique))


# internal_trace reaches the model: `model_result` carries it byte-identically. Fields added for
# the archive must therefore be removed here, or the agent's information surface moves and
# episodes recorded on either side of the change stop being the same tested unit.
SERVER_ONLY_TRACE_KEYS = ("args", "args_omitted", "achieved_pose")


def _strip_server_only_trace(payload):
    trace = payload.get("internal_trace") if isinstance(payload, dict) else None
    if not isinstance(trace, list):
        return
    for entry in trace:
        if isinstance(entry, dict):
            for key in SERVER_ONLY_TRACE_KEYS:
                entry.pop(key, None)


def project_tool_result(
    result_obj: Any,
    *,
    toolbox,
    code_observation_ids: Iterable[str] = (),
    image_transport: str = FOLLOWUP_IMAGE_TRANSPORT,
    run_code_result_max_images: int = RUN_CODE_RESULT_MAX_IMAGES,
    model_result_max_bytes: int = MODEL_VISIBLE_RESULT_MAX_BYTES,
    serialize_result: bool = True,
    tool_name: str | None = None,
) -> ToolProjection:
    """Build one model-visible payload while retaining all observation references out of band."""
    if image_transport not in (FOLLOWUP_IMAGE_TRANSPORT, INLINE_IMAGE_TRANSPORT):
        raise ValueError(f"unknown image transport {image_transport!r}")
    limit = int(run_code_result_max_images)
    if limit < 0:
        raise ValueError("run_code_result_max_images must be non-negative")

    full_payload = (serialize(result_obj) if serialize_result
                    else json_safe(deepcopy(dict(result_obj))))
    # Say when code captured more images than the transport attaches. Without this the caller
    # receives every obs_id but only the last `limit` images and has no way to tell the difference
    # -- it reads as "I have seen everything". The history-eviction path already leaves a visible
    # marker; this is its missing counterpart. It goes in the PAYLOAD rather than in an extra text
    # message because the MCP transport forwards only `model_image_refs`, so a text-only image
    # group would reach the reference scaffold and silently vanish on the vendor agent.
    raw_code_ids = tuple(str(obs_id) for obs_id in code_observation_ids)
    code_ids = _unique_by_last_occurrence(raw_code_ids)
    duplicates_suppressed = len(raw_code_ids) - len(code_ids)
    # run_code/run_program payloads expose the observation list to the model.  Keep that list in
    # lockstep with the actual attachments so repeated provenance does not look like repeated
    # evidence.  The internal trace remains untouched and records every primitive call.
    if raw_code_ids and isinstance(full_payload.get("obs_ids"), list):
        full_payload["obs_ids"] = list(code_ids)
    withheld = code_ids[:-limit] if limit else code_ids
    if withheld:
        full_payload["images_withheld"] = {
            "count": len(withheld),
            "obs_ids": list(withheld),
            "note": f"code captured more images than the {limit} most recent ones attached here; "
                    "these observations remain addressable by obs_id",
        }
    if tool_name is not None and getattr(toolbox, "enforce_result_contracts", False):
        validate_result(tool_name, full_payload)
    payload = deepcopy(full_payload)
    _strip_server_only_trace(payload)
    if image_transport == INLINE_IMAGE_TRANSPORT:
        _inline_payload(payload, result_obj)
    model_payload, projection_meta = bounded_json_projection(
        payload, model_result_max_bytes)

    direct_groups = _direct_groups(result_obj)
    direct_refs = tuple(ref for group in direct_groups for ref in group.refs)
    direct_ids = _direct_observation_ids(result_obj)

    all_code_refs = []
    for obs_id in code_ids:
        try:
            all_code_refs.append(str(toolbox.image_path(obs_id)))
        except Exception:
            continue

    visible_ids = code_ids[-limit:] if limit else ()
    code_groups = []
    for obs_id in visible_ids:
        try:
            ref = str(toolbox.image_path(obs_id))
        except Exception:
            continue
        label = f"{obs_id} | role=code_result"
        try:
            observation = toolbox._get_obs(obs_id)
        except Exception:
            observation = None
        if observation is not None:
            label = (f"{observation.obs_id} | {observation.camera} | role=code_result | "
                     f"tick={observation.tick}")
        code_groups.append(ImageGroup(
            "Observation captured or replayed inside code execution",
            (ref,),
            (label,),
        ))

    image_groups = direct_groups + tuple(code_groups)
    model_refs = tuple(ref for group in image_groups for ref in group.refs)
    return ToolProjection(
        payload=model_payload,
        full_payload=full_payload,
        image_groups=image_groups,
        model_image_refs=model_refs,
        all_image_refs=direct_refs + tuple(all_code_refs),
        all_observation_ids=direct_ids + code_ids,
        full_result=result_obj,
        truncated=projection_meta["truncated"],
        original_bytes=projection_meta["original_bytes"],
        model_bytes=projection_meta["model_bytes"],
        duplicate_observation_ids_suppressed=duplicates_suppressed,
    )
