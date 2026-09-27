"""Validate controller-authored container provenance before it reaches run metadata.

The model-facing containers never author this object. The host-side controller writes one file
inside the attempt directory; the sim process validates it before serving any tools and merges it
into run_meta.json only after the episode ends.
"""
import json
import re
from pathlib import Path

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$")
_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_.+-]{0,63}$")
_MODEL = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_.+:/-]{0,255}$")
_MAX_BYTES = 64 * 1024

LEGACY_REQUIRED_KEYS = {
    "runtime", "profile", "task_name", "seed", "attempt_index", "source_commit",
    "source_dirty", "sim_image_id", "sim_image_digest", "sim_image_commit", "lock_sha256",
    "dockerfile_sha256", "task_pack_version", "task_pack_sha256", "cache_state",
    "controller_version",
    "agent_image_id", "agent_image_digest", "agent_cli", "agent_cli_version",
    "model", "agent_label", "interface", "gpu", "tool_schema_sha256",
}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_legacy_provenance(value: dict) -> dict:
    """Return a JSON-safe copy of a complete provenance object or raise ValueError."""
    if not isinstance(value, dict):
        raise ValueError("provenance must be a JSON object")
    keys = set(value)
    missing = sorted(LEGACY_REQUIRED_KEYS - keys)
    unknown = sorted(keys - LEGACY_REQUIRED_KEYS)
    if missing:
        raise ValueError(f"provenance missing required keys: {missing}")
    if unknown:
        raise ValueError(f"provenance contains unknown keys: {unknown}")

    def match(key, pattern):
        v = value[key]
        if not isinstance(v, str) or not pattern.fullmatch(v):
            raise ValueError(f"provenance {key} has invalid format")

    if value["runtime"] != "docker":
        raise ValueError("provenance runtime must be 'docker'")
    if value["profile"] not in {"dev", "eval"}:
        raise ValueError("provenance profile must be 'dev' or 'eval'")
    if value["source_dirty"] is not False:
        raise ValueError("container benchmark provenance requires source_dirty=false")
    if value["cache_state"] not in {"cold", "warm"}:
        raise ValueError("provenance cache_state must be 'cold' or 'warm'")
    if not _is_int(value["seed"]) or value["seed"] < 0:
        raise ValueError("provenance seed must be a non-negative integer")
    if not _is_int(value["attempt_index"]) or value["attempt_index"] < 0:
        raise ValueError("provenance attempt_index must be a non-negative integer")
    if not _is_int(value["gpu"]) or value["gpu"] < 0 or value["gpu"] > 31:
        raise ValueError("provenance gpu must be an integer in [0, 31]")
    if value["interface"] != "mcp-agent":
        raise ValueError("provenance interface must be 'mcp-agent'")

    for key in ("task_name", "task_pack_version", "controller_version", "agent_cli",
                "agent_cli_version", "agent_label"):
        match(key, _NAME if key in {"task_name", "agent_cli"} else _VERSION)
    match("model", _MODEL)
    for key in ("source_commit", "sim_image_commit"):
        match(key, _HEX40)
    for key in ("lock_sha256", "dockerfile_sha256", "task_pack_sha256",
                "tool_schema_sha256"):
        match(key, _HEX64)
    for key in ("sim_image_id", "agent_image_id"):
        match(key, _IMAGE_ID)
    for key in ("sim_image_digest", "agent_image_digest"):
        v = value[key]
        if v is not None and (not isinstance(v, str) or not _IMAGE_DIGEST.fullmatch(v)):
            raise ValueError(
                f"provenance {key} must be null, a local content ID, or a registry digest")
    if value["profile"] == "eval" and (value["sim_image_digest"] is None
                                         or value["agent_image_digest"] is None):
        raise ValueError("eval provenance requires content-addressed identities for both images")

    # Round-trip produces a detached object containing only JSON primitives.
    return json.loads(json.dumps(value, sort_keys=True))


_COMMON_V02 = {
    "schema_version", "runtime", "profile", "task_name", "seed", "attempt_index",
    "source_commit", "source_dirty", "task_pack_version", "task_pack_sha256",
    "cache_state", "controller_version", "model", "agent_label", "interface",
    "interface_profile", "gpu", "expected_identity",
    # Declared harness parameter that does not move base_tool_set_sha256, so provenance is the
    # only place two runs differing in it can be told apart. Required, not optional: an absent
    # value would let an anchor-off run be read as a default-configuration result.
    "orientation_anchor",
}
_DOCKER_V02 = {
    "sim_image_id", "sim_image_digest", "sim_image_commit", "lock_sha256",
    "dockerfile_sha256", "agent_image_id", "agent_image_digest",
    "gateway_image_id", "gateway_image_digest", "gateway_image_commit",
    "agent_cli", "agent_cli_version",
}
_CONDA_V02 = {"environment_lock_sha256", "host_fingerprint_sha256"}


def _validate_v02(value: dict) -> dict:
    from codeaction.contracts.identity import comparison_identity, make_identity

    runtime = value.get("runtime")
    if runtime not in ("docker", "conda"):
        raise ValueError("provenance runtime must be 'docker' or 'conda'")
    required = _COMMON_V02 | (_DOCKER_V02 if runtime == "docker" else _CONDA_V02)
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise ValueError(f"provenance missing required keys: {missing}")
    if unknown:
        raise ValueError(f"provenance contains unknown keys: {unknown}")
    if value["schema_version"] != "0.2":
        raise ValueError("provenance schema_version must be '0.2'")
    if value["profile"] not in {"dev", "eval"}:
        raise ValueError("provenance profile must be 'dev' or 'eval'")
    if runtime == "conda" and value["profile"] != "dev":
        raise ValueError("conda provenance is dev-only")
    if type(value["source_dirty"]) is not bool:
        raise ValueError("provenance source_dirty must be a boolean")
    if value["profile"] == "eval" and value["source_dirty"]:
        raise ValueError("eval provenance requires source_dirty=false")
    if value["cache_state"] not in {"cold", "warm"}:
        raise ValueError("provenance cache_state must be 'cold' or 'warm'")
    if value["orientation_anchor"] not in {"on", "off"}:
        raise ValueError("provenance orientation_anchor must be 'on' or 'off'")
    if not _is_int(value["seed"]) or value["seed"] < 0:
        raise ValueError("provenance seed must be a non-negative integer")
    if not _is_int(value["attempt_index"]) or value["attempt_index"] < 0:
        raise ValueError("provenance attempt_index must be a non-negative integer")
    if not _is_int(value["gpu"]) or value["gpu"] < 0 or value["gpu"] > 31:
        raise ValueError("provenance gpu must be an integer in [0, 31]")
    if value["interface_profile"] not in {
            "reference-mcp", "reference-code-first",
            "vendor-mcp-direct", "vendor-mcp-gateway"}:
        raise ValueError("provenance interface_profile is invalid")
    for key in ("task_name", "task_pack_version", "controller_version", "agent_label"):
        match_value = value[key]
        pattern = _NAME if key == "task_name" else _VERSION
        if not isinstance(match_value, str) or not pattern.fullmatch(match_value):
            raise ValueError(f"provenance {key} has invalid format")
    if not isinstance(value["model"], str) or not _MODEL.fullmatch(value["model"]):
        raise ValueError("provenance model has invalid format")
    if not isinstance(value["interface"], str) or not _NAME.fullmatch(value["interface"]):
        raise ValueError("provenance interface has invalid format")
    for key in ("source_commit",):
        if not isinstance(value[key], str) or not _HEX40.fullmatch(value[key]):
            raise ValueError(f"provenance {key} has invalid format")
    if not isinstance(value["task_pack_sha256"], str) \
            or not _HEX64.fullmatch(value["task_pack_sha256"]):
        raise ValueError("provenance task_pack_sha256 has invalid format")

    expected = value.get("expected_identity")
    if not isinstance(expected, dict):
        raise ValueError("provenance expected_identity must be an object")
    try:
        comparison = comparison_identity(**expected["comparison"])
        normalized_identity = make_identity(comparison, expected["trial"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid provenance expected_identity: {exc}") from exc
    if normalized_identity["trial"]["scene_seed"] != value["seed"] or \
            normalized_identity["trial"]["attempt_index"] != value["attempt_index"]:
        raise ValueError("provenance trial identity disagrees with seed/attempt_index")

    if runtime == "docker":
        # Both reference profiles run the same container; they differ only in which primitives
        # are delivered as schemas, which is recorded in the tool surface, not the interface name.
        expected_interface = (
            "reference-agent"
            if value["interface_profile"] in ("reference-mcp", "reference-code-first")
            else "mcp-agent")
        if value["interface"] != expected_interface:
            raise ValueError(
                f"docker provenance interface must be {expected_interface!r} "
                f"for {value['interface_profile']!r}")
        for key in ("sim_image_commit", "gateway_image_commit"):
            # Source archives have no Git metadata. Image IDs and digests below
            # still identify the exact environment; a commit is only attribution.
            if value[key] != "unknown" and (
                    not isinstance(value[key], str) or not _HEX40.fullmatch(value[key])):
                raise ValueError(f"provenance {key} has invalid format")
        for key in ("lock_sha256", "dockerfile_sha256"):
            if not isinstance(value[key], str) or not _HEX64.fullmatch(value[key]):
                raise ValueError(f"provenance {key} has invalid format")
        for key in ("sim_image_id", "agent_image_id", "gateway_image_id"):
            if not isinstance(value[key], str) or not _IMAGE_ID.fullmatch(value[key]):
                raise ValueError(f"provenance {key} has invalid format")
        for key in ("sim_image_digest", "agent_image_digest", "gateway_image_digest"):
            digest = value[key]
            if digest is not None and (
                    not isinstance(digest, str) or not _IMAGE_DIGEST.fullmatch(digest)):
                raise ValueError(
                    f"provenance {key} must be null, a local content ID, or a registry digest")
        if value["profile"] == "eval" and (
                value["sim_image_digest"] is None
                or value["agent_image_digest"] is None
                or value["gateway_image_digest"] is None):
            raise ValueError(
                "eval provenance requires digest-pinned sim, agent, and gateway images")
        for key in ("agent_cli", "agent_cli_version"):
            pattern = _NAME if key == "agent_cli" else _VERSION
            if not isinstance(value[key], str) or not pattern.fullmatch(value[key]):
                raise ValueError(f"provenance {key} has invalid format")
    else:
        if value["interface"] != "reference-agent":
            raise ValueError("conda provenance interface must be 'reference-agent'")
        for key in _CONDA_V02:
            if not isinstance(value[key], str) or not _HEX64.fullmatch(value[key]):
                raise ValueError(f"provenance {key} has invalid format")

    detached = json.loads(json.dumps(value, sort_keys=True))
    detached["expected_identity"] = normalized_identity
    return detached


def validate_provenance(value: dict) -> dict:
    """Validate schema 0.2 runtime-union attestations; keep implicit 0.1 readable."""
    if not isinstance(value, dict):
        raise ValueError("provenance must be a JSON object")
    if value.get("schema_version") == "0.2":
        return _validate_v02(value)
    return _validate_legacy_provenance(value)


def load_provenance(path, attempt_dir: Path) -> dict:
    """Load provenance only from inside ``attempt_dir``; symlink escapes fail closed."""
    root = Path(attempt_dir).resolve()
    requested = Path(path)
    candidate = (root / requested).resolve() if not requested.is_absolute() else requested.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("provenance file must be inside the attempt directory") from exc
    if not candidate.is_file():
        raise ValueError(f"provenance file does not exist: {candidate.name}")
    if candidate.stat().st_size > _MAX_BYTES:
        raise ValueError(f"provenance file exceeds {_MAX_BYTES} bytes")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid provenance JSON: {exc}") from exc
    return validate_provenance(value)
