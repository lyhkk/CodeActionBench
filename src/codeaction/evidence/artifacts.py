"""Attempt artifact inventory, integrity verification, and release-evidence checks."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


MANIFEST_NAME = "artifact_manifest.v1.json"
MANIFEST_SCHEMA = "artifact-manifest.v1"
_OBS_ID = re.compile(r"^obs_[0-9]+$")
_IMAGE_DIGEST = re.compile(r"^(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_REPLAY_KEYS = frozenset({
    "encrypted_content", "previous_response_id", "prompt_cache_key",
})


class ArtifactManifestError(ValueError):
    """The sealed inventory no longer matches the attempt directory."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], f"cannot read {path.name}: {type(exc).__name__}"
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return [], f"invalid JSON in {path.name}:{line_number}"
        if not isinstance(value, dict):
            return [], f"non-object record in {path.name}:{line_number}"
        records.append(value)
    return records, None


def _private_replay_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _PRIVATE_REPLAY_KEYS:
                found.add(lowered)
            found.update(_private_replay_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_private_replay_keys(child))
    return found


def validate_reasoning_replay_events(
    events: Iterable[dict[str, Any]], requirement: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Validate adjacent-turn replay using hashes only; provider blobs are forbidden."""
    rows = [dict(event) for event in events]
    errors: list[str] = []
    leaked = sorted(_private_replay_keys(rows))
    if leaked:
        errors.append(f"private provider replay fields leaked into transcript: {leaked}")
    if requirement is None:
        return {"required": False}, errors
    if requirement != "openai-encrypted-v1":
        errors.append(f"unsupported reasoning replay evidence contract: {requirement!r}")
        return {"required": True, "protocol": requirement}, errors

    model_turns = [row for row in rows if row.get("event") == "model_turn"]
    replay_rows = [row for row in rows if row.get("event") == "provider_reasoning_replay"]
    turns = [row.get("turn") for row in model_turns]
    replay_turns = [row.get("turn") for row in replay_rows]
    if any(not isinstance(turn, int) or isinstance(turn, bool) for turn in turns + replay_turns):
        errors.append("reasoning replay evidence has an invalid turn number")
    if len(set(replay_turns)) != len(replay_turns):
        errors.append("reasoning replay evidence has duplicate turn records")
    if sorted(replay_turns) != sorted(turns):
        errors.append("reasoning replay evidence does not cover every successful model turn")

    by_turn = {row.get("turn"): row for row in replay_rows}
    normalized = []
    verified_transitions = 0
    for model_turn in model_turns:
        turn = model_turn.get("turn")
        row = by_turn.get(turn)
        if row is None:
            continue
        generated = row.get("item_sha256")
        replayed = row.get("replayed_item_sha256")
        generated_count = row.get("generated_reasoning_items")
        replayed_count = row.get("replayed_reasoning_items")
        encrypted_present = row.get("encrypted_content_present")
        usage = model_turn.get("usage")
        reasoning_tokens = usage.get("reasoning_tokens") if isinstance(usage, dict) else None
        if row.get("protocol") != requirement:
            errors.append(f"turn {turn} reasoning replay protocol mismatch")
        if not isinstance(generated, list) or not all(
                isinstance(value, str) and _HEX64.fullmatch(value) for value in generated):
            errors.append(f"turn {turn} generated reasoning hashes are invalid")
            generated = []
        if not isinstance(replayed, list) or not all(
                isinstance(value, str) and _HEX64.fullmatch(value) for value in replayed):
            errors.append(f"turn {turn} replayed reasoning hashes are invalid")
            replayed = []
        if isinstance(generated_count, bool) or not isinstance(generated_count, int) \
                or generated_count != len(generated):
            errors.append(f"turn {turn} generated reasoning count mismatch")
        if isinstance(replayed_count, bool) or not isinstance(replayed_count, int) \
                or replayed_count != len(replayed):
            errors.append(f"turn {turn} replayed reasoning count mismatch")
        if not isinstance(encrypted_present, bool):
            errors.append(f"turn {turn} encrypted-content presence is not boolean")
        if isinstance(reasoning_tokens, bool) or not isinstance(reasoning_tokens, int) \
                or reasoning_tokens < 0:
            errors.append(f"turn {turn} reasoning-token usage is missing or invalid")
        message = model_turn.get("message")
        message_calls = message.get("tool_calls") if isinstance(message, dict) else None
        tool_turn = model_turn.get("stop_reason") == "tool_calls" or bool(message_calls)
        if tool_turn:
            if generated and encrypted_present is not True:
                errors.append(f"turn {turn} cannot continue without encrypted reasoning")
            if not generated and reasoning_tokens != 0:
                errors.append(
                    f"turn {turn} has reasoning usage but no encrypted reasoning item")
        normalized.append({
            "turn": turn,
            "generated_item_sha256": generated,
            "replayed_item_sha256": replayed,
            "encrypted_content_present": encrypted_present,
            "reasoning_tokens": reasoning_tokens,
            "tool_turn": tool_turn,
        })

    ordered = sorted(normalized, key=lambda item: item["turn"] if isinstance(
        item["turn"], int) else -1)
    for previous, current in zip(ordered, ordered[1:]):
        if previous["tool_turn"]:
            if current["replayed_item_sha256"] != previous["generated_item_sha256"]:
                errors.append(
                    f"turn {current['turn']} did not replay turn {previous['turn']} reasoning")
            elif previous["generated_item_sha256"]:
                verified_transitions += 1
    canonical = json.dumps(
        ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "required": True,
        "protocol": requirement,
        "turns": len(model_turns),
        "tool_turns": sum(1 for item in ordered if item["tool_turn"]),
        "verified_transitions": verified_transitions,
        "evidence_sha256": hashlib.sha256(canonical).hexdigest(),
    }, errors


def _reasoning_replay_error(attempt: Path, result: dict[str, Any]) -> str | None:
    comparison = ((result.get("identity") or {}).get("comparison") or {})
    request_profile = ((comparison.get("model") or {}).get("request_profile") or {})
    requirement = request_profile.get("reasoning_replay_evidence")
    transcript = attempt / "reference_transcript.jsonl"
    if not transcript.is_file():
        return "reasoning replay evidence requires reference_transcript.jsonl" \
            if requirement is not None else None
    events, read_error = _read_jsonl(transcript)
    if read_error:
        return read_error
    summary, errors = validate_reasoning_replay_events(events, requirement)
    if errors:
        return "; ".join(errors)
    if requirement is not None:
        attestation = _read_json(attempt / "reference_agent_attestation.json")
        if attestation.get("reasoning_replay") != summary:
            return "reference-agent reasoning replay attestation differs from transcript"
    return None


def _purpose(path: str) -> tuple[str, bool]:
    name = Path(path).name
    if path.startswith("tools/observer/") and name.lower().endswith(
            (".png", ".jpg", ".jpeg")):
        return "evaluator_observation", False
    if path.startswith("tools/") and _is_observation_file(name):
        return "model_observation", True
    if name == "result.json":
        return "result", False
    if name == "run_meta.json":
        return "run_metadata", False
    if name == "provenance.json":
        return "provenance", False
    if name == "controller_status.json":
        return "controller_status", False
    if name == "transcript.jsonl":
        return "server_transcript", False
    if name == "reference_transcript.jsonl":
        return "model_transcript", False
    if name in ("full.mp4", "review.mp4"):
        return "episode_video" if name == "full.mp4" else "review_video", False
    if name == "video_meta.json":
        return "video_metadata", False
    if "attestation" in name:
        return "attestation", False
    if "audit" in name:
        return "audit", False
    if name.startswith("contact_") or name.startswith("contact-"):
        return "contact_diagnostic", False
    if name == "compose.log" or name.endswith(".log"):
        return "runtime_log", False
    return "supporting_evidence", False


def _is_observation_file(name: str) -> bool:
    stem = Path(name).stem
    return stem == "obs" or bool(re.match(r"^obs_[0-9]+(?:_|$)", stem))


def _required_paths(attempt: Path) -> set[str]:
    required = {"provenance.json", "controller_status.json"}
    status = _read_json(attempt / "controller_status.json")
    if status.get("state") in (None, "starting", "dry_run"):
        return required
    required.update({
        "filesystem_audit.json",
        "gateway_attestation.json",
        "controller_identity_attestation.json",
    })
    provenance = _read_json(attempt / "provenance.json")
    is_health = status.get("vendor_mcp_health") not in (None, "not_applicable")
    if is_health:
        required.add("vendor_mcp_health.json")
        return required
    required.update({
        "result.json",
        "run_meta.json",
        "transcript.jsonl",
        "full.mp4",
        "review.mp4",
        "video_meta.json",
    })
    if provenance.get("interface_profile") in ("reference-mcp", "reference-code-first"):
        required.update({"reference_transcript.jsonl", "reference_agent_attestation.json"})
    return required


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)
    else:
        yield value


def _referenced_observations(attempt: Path) -> set[str]:
    found: set[str] = set()
    for name in ("transcript.jsonl", "reference_transcript.jsonl"):
        path = attempt / name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            for value in _walk_values(event):
                if isinstance(value, str) and _OBS_ID.fullmatch(value):
                    found.add(value)
    return found


def _missing_observation_files(attempt: Path) -> list[str]:
    tools = attempt / "tools"
    missing = []
    for obs_id in sorted(_referenced_observations(attempt)):
        if not any(tools.glob(f"{obs_id}*.png")):
            missing.append(f"{obs_id} -> tools/{obs_id}*.png")
    return missing


def _executed_sim_actions(attempt: Path) -> int:
    """Tool calls that actually moved the simulator, from the server-side transcript."""
    path = attempt / "transcript.jsonl"
    if not path.is_file():
        return 0
    executed = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return 0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        start, end = event.get("sim_step_start"), event.get("sim_step_end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            executed += 1
    return executed


def _missing_observer_frames(attempt: Path) -> list[str]:
    """Eval-side observation frames are written per sim-mutating action, and their writer
    swallows its own errors by design so a render hiccup cannot kill an episode. Nothing
    references them, so the dangling-reference check cannot see them either: an attempt could
    move the robot for forty minutes, record none of it, and still look complete. This is the
    check that notices. It asserts presence, not an exact count -- the per-tool frame cadence is
    behaviour, and pinning a number here would break on every legitimate change to it.
    """
    executed = _executed_sim_actions(attempt)
    if executed == 0:
        return []
    frames = list((attempt / "tools" / "observer").glob("tick_*.png"))
    if frames:
        return []
    return [f"tools/observer/tick_*.png absent after {executed} executed sim action(s)"]


def _inventory(attempt: Path, required: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    files = []
    unsafe = []
    for path in sorted(attempt.rglob("*")):
        rel = path.relative_to(attempt).as_posix()
        if rel == MANIFEST_NAME or rel.startswith("release/"):
            continue
        if path.is_symlink():
            unsafe.append(rel)
            continue
        if not path.is_file():
            continue
        purpose, model_visible = _purpose(rel)
        files.append({
            "path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "purpose": purpose,
            "model_visible": model_visible,
            "required": rel in required,
        })
    return files, unsafe


def _tested_unit(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        return {}
    comparison = identity.get("comparison")
    if not isinstance(comparison, dict):
        return {}
    tested = comparison.get("tested_unit")
    return tested if isinstance(tested, dict) else {}


def _comparison(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        return {}
    comparison = identity.get("comparison")
    return comparison if isinstance(comparison, dict) else {}


def build_artifact_manifest(attempt_dir: Path) -> dict[str, Any]:
    """Build a complete, deterministic inventory without mutating the attempt."""
    attempt = Path(attempt_dir).resolve()
    if not attempt.is_dir():
        raise ArtifactManifestError(f"attempt directory does not exist: {attempt}")
    required = _required_paths(attempt)
    files, unsafe = _inventory(attempt, required)
    listed = {entry["path"] for entry in files}
    missing_required = sorted(required - listed)
    dangling = _missing_observation_files(attempt) + _missing_observer_frames(attempt)
    result = _read_json(attempt / "result.json")
    reasoning_replay_error = _reasoning_replay_error(attempt, result)
    evidence_complete = not (
        missing_required or dangling or unsafe or reasoning_replay_error)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attempt_id": attempt.name,
        "hash_algorithm": "sha256",
        "files": files,
        "missing_required": missing_required,
        "dangling_references": dangling,
        "observer_frames": len(list((attempt / "tools" / "observer").glob("tick_*.png"))),
        "unsafe_paths": unsafe,
        "reasoning_replay_error": reasoning_replay_error,
        "evidence_complete": evidence_complete,
        "submittable": result.get("submittable") is True and evidence_complete,
    }


def write_artifact_manifest(attempt_dir: Path) -> dict[str, Any]:
    """Atomically seal the current attempt inventory."""
    attempt = Path(attempt_dir).resolve()
    manifest = build_artifact_manifest(attempt)
    fd, raw_path = tempfile.mkstemp(prefix=f".{MANIFEST_NAME}.", dir=attempt)
    tmp = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(attempt / MANIFEST_NAME)
    finally:
        if tmp.exists():
            tmp.unlink()
    return manifest


def write_run_summary(run_dir: Path) -> dict[str, Any]:
    """Write the canonical metric/protocol summary when the run has a result."""
    root = Path(run_dir).resolve()
    run = _read_json(root / "run.json")
    if not run:
        raise ArtifactManifestError(f"invalid run.json in {root}")
    attempts_declared = run.get("protocol_attempts_declared")
    if not isinstance(attempts_declared, int) or isinstance(attempts_declared, bool) \
            or attempts_declared < 1:
        declared_seeds = run.get("declared_scene_seeds")
        attempts_declared = (
            len(declared_seeds)
            if isinstance(declared_seeds, list) and declared_seeds
            else int(run.get("attempts") or 0)
        )
    from codeaction.verification.summarize import summarize_run_dir
    summary = summarize_run_dir(
        root, attempts_declared=attempts_declared, write=True)
    return summary or {}


def verify_artifact_manifest(attempt_dir: Path) -> dict[str, Any]:
    """Verify every sealed file and report evidence completeness separately."""
    attempt = Path(attempt_dir).resolve()
    path = attempt / MANIFEST_NAME
    try:
        sealed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"invalid {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(sealed, dict) or sealed.get("schema_version") != MANIFEST_SCHEMA:
        raise ArtifactManifestError(f"invalid {MANIFEST_NAME} schema")
    current = build_artifact_manifest(attempt)
    sealed_files = {entry.get("path"): entry for entry in sealed.get("files", [])
                    if isinstance(entry, dict) and isinstance(entry.get("path"), str)}
    current_files = {entry["path"]: entry for entry in current["files"]}
    missing = sorted(set(sealed_files) - set(current_files))
    unlisted = sorted(set(current_files) - set(sealed_files))
    modified = sorted(
        rel for rel in set(sealed_files) & set(current_files)
        if sealed_files[rel].get("size_bytes") != current_files[rel]["size_bytes"]
        or sealed_files[rel].get("sha256") != current_files[rel]["sha256"]
    )
    stale_policy = any(
        sealed.get(key) != current.get(key)
        for key in ("missing_required", "dangling_references", "unsafe_paths",
                    "reasoning_replay_error",
                    "evidence_complete", "submittable")
    )
    integrity_ok = not (missing or unlisted or modified or stale_policy)
    return {
        "integrity_ok": integrity_ok,
        "evidence_complete": current["evidence_complete"] and integrity_ok,
        "submittable": current["submittable"] and integrity_ok,
        "missing_files": missing,
        "unlisted_files": unlisted,
        "modified_files": modified,
        "missing_required": current["missing_required"],
        "dangling_references": current["dangling_references"],
        "unsafe_paths": current["unsafe_paths"],
        "reasoning_replay_error": current["reasoning_replay_error"],
        "stale_policy": stale_policy,
    }
