#!/usr/bin/env python3
"""Controller for provenance-bound container vendor-agent attempts.

The controller owns Docker lifecycle and host paths but never receives a model token value.  The
agent receives only the token file bind declared by docker/compose.yml.  ``doctor`` and
``run --dry-run`` never start an agent process.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from codeaction.paths import DOCKER_ROOT, PROJECT_ROOT, REPOSITORY_ROOT, TASKS_ROOT

ROOT = REPOSITORY_ROOT
CODEACTION_ROOT = PROJECT_ROOT
COMPOSE_FILE = DOCKER_ROOT / "compose.yml"
CONTROLLER_VERSION = "0.5"

# `${NAME:?message}` in compose.yml. Compose interpolates the WHOLE file before it selects
# services, so one of these being unset OR EMPTY aborts the run even when the only service using
# it belongs to a profile this run never starts.
_COMPOSE_REQUIRED = re.compile(r"\$\{([A-Z_][A-Z0-9_]*):\?")


def compose_required_variables(compose_text):
    """Every variable compose.yml refuses to run without."""
    return set(_COMPOSE_REQUIRED.findall(compose_text))


def missing_required_env(env, compose_text):
    """Required variables that would abort interpolation, empty values included.

    Checked before `up` so the failure names the variable, instead of surfacing as a compose
    interpolation error attributed to a service this profile does not even start.
    """
    return sorted(name for name in compose_required_variables(compose_text) if not env.get(name))
from codeaction.launch import context as execution_context
from codeaction.providers.model_registry import (  # noqa: E402
    REASONING_RUNGS, RegistryError, find_model, reasoning_capture_error, resolve_model)
from codeaction.providers.provider_runtime import (  # noqa: E402
    load_rate_limit_config, normalize_rate_limit_policy, rate_limit_policy_for,
    strict_rate_limit_coverage)
from codeaction.evidence.artifacts import (  # noqa: E402
    MANIFEST_NAME, ArtifactManifestError,
    validate_reasoning_replay_events,
    verify_artifact_manifest,
    write_artifact_manifest, write_run_summary)

# Standing default for real provider runs: thinking ON at the top rung. `reasoning_profile` enters
# the model identity hash, so runs made before this default changed are a DIFFERENT tested unit and
# must not be pooled with later ones -- the recorded profile is what tells them apart.
# How many subscription-backed vendor processes may run on this host AT ONCE, per vendor. The
# name predates the second seat and is deliberately not renamed: it is hashed into every
# published run's driver.config_sha256, so renaming it would fork the identity of results that
# are otherwise identical.
CLAUDE_SUBSCRIPTION_CONCURRENCY_LIMIT = 4
TRANSPORT_CONFORMANCE_PROMPT = """This is a non-scoring transport conformance check.
Do not perform the benchmark task and do not move either robot arm.
Use ToolSearch if needed, call the CodeAction capture_head tool exactly once, inspect the returned
image, then call done with report="IMAGE_OK" and success_claim=false. Do nothing else.
"""
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$")

# The task the dev-only commands (doctor, mcp-health) exercise the machinery with. It has
# to be a REGISTERED task: the loader refuses anything else, so a retired name here is a
# command that cannot run. `click_bell` is the cheapest released card (28-call budget).
DEFAULT_DEV_TASK = "click_bell"
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_AUDIT_REQUEST_MARK = "[filesystem-audit-request] "
_AUDIT_RESULT_MARK = "[filesystem-audit-result] "
_GATEWAY_ATTESTATION_MARK = "[gateway-attestation] "
_REFERENCE_EVENT_MARK = "[reference-agent-event] "
_REFERENCE_SUMMARY_MARK = "[reference-agent-summary] "
_RAW_MCP_SURFACE_MARK = "[raw-mcp-surface] "
_VENDOR_MCP_HEALTH_MARK = "[vendor-mcp-health] "


class _SkipTurnsView(Exception):
    """Internal: this attempt has no episode to project."""


class ControllerError(RuntimeError):
    """A fail-closed controller validation or lifecycle error."""


@dataclass(frozen=True)
class ImageInfo:
    ref: str
    image_id: str
    digest: str | None
    labels: dict[str, str]


@dataclass(frozen=True)
class RuntimeInfo:
    source_commit: str
    sim: ImageInfo
    agent: ImageInfo
    gateway: ImageInfo
    scratch: ImageInfo | None
    launcher: ImageInfo | None
    lock_sha256: str
    dockerfile_sha256: str
    task_pack_version: str
    task_pack_sha256: str
    base_tool_set_sha256: str
    # Integrity findings a dev-profile run records instead of refusing on; eval raises. Each
    # entry becomes a non_submittable_reason, so the run stays honest without being blocked.
    integrity_notes: tuple = ()


def _run_output(argv: list[str], *, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ControllerError(f"command failed ({argv[0]} exit {proc.returncode}){suffix}")
    return proc.stdout


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state(source_root: Path) -> tuple[str, bool]:
    from codeaction.launch import context
    frozen = context()
    if frozen is not None:
        return frozen["source_commit"], False
    # Source archives have no repository metadata. The
    # execution snapshot still records the actual file content independently.
    if not (source_root / ".git").exists():
        return "0" * 40, False
    commit = _run_output(["git", "-C", str(source_root), "rev-parse", "HEAD"]).strip()
    if not _COMMIT.fullmatch(commit):
        raise ControllerError("source HEAD is not a full 40-hex commit")
    dirty = bool(_run_output(["git", "-C", str(source_root), "status", "--porcelain"]).strip())
    return commit, dirty


def _inspect_image(ref: str) -> ImageInfo:
    try:
        values = json.loads(_run_output(["docker", "image", "inspect", ref]))
    except json.JSONDecodeError as exc:
        raise ControllerError(f"docker returned malformed inspection JSON for {ref!r}") from exc
    if not isinstance(values, list) or len(values) != 1:
        raise ControllerError(f"expected one inspected image for {ref!r}")
    value = values[0]
    image_id = value.get("Id")
    if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id):
        raise ControllerError(f"image {ref!r} has an invalid content ID")
    labels = ((value.get("Config") or {}).get("Labels") or {})
    if not isinstance(labels, dict):
        raise ControllerError(f"image {ref!r} has invalid labels")
    digest = ref if _IMAGE_DIGEST.fullmatch(ref) else image_id
    return ImageInfo(ref=ref, image_id=image_id, digest=digest,
                     labels={str(k): str(v) for k, v in labels.items()})


def _validate_image_revisions(source_commit: str, images: list[ImageInfo],
                              source_root: Path = ROOT) -> None:
    from codeaction.release import image_matches_runtime, runtime_identity
    from codeaction.launch import context
    if context() is not None or all("org.codeaction.environment-sha256" in image.labels for image in images):
        from codeaction.environments import validate_environment
        for image in images:
            try:
                validate_environment(image.labels, source_root)
            except ValueError as exc:
                raise ControllerError(str(exc)) from exc
        return
    expected = runtime_identity(source_root)
    mismatched = []
    for image in images:
        if not image_matches_runtime(image.labels, expected):
            mismatched.append(image.ref)
    if mismatched:
        raise ControllerError("images differ from current source; run tools/build_images.sh: "
                              + ", ".join(sorted(mismatched)))


def _validate_codex_home(path: Path) -> None:
    """A Codex seat mounts a CODEX_HOME directory, not a token file.

    Only auth.json is required and only it is copied into the container, but the directory is
    checked as a whole: it is the account's own state, and a world-readable one would hand a
    subscription to anything else on the box.
    """
    try:
        info = path.stat()
    except OSError as exc:
        raise ControllerError("Codex home directory is not readable") from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise ControllerError("Codex home must be a directory with mode 0700")
    auth = path / "auth.json"
    try:
        auth_info = auth.stat()
    except OSError as exc:
        raise ControllerError("Codex home has no readable auth.json") from exc
    if not stat.S_ISREG(auth_info.st_mode) or auth_info.st_mode & 0o077:
        raise ControllerError("Codex auth.json must be a regular file with mode 0600")


def _validate_vendor_credential(path: Path, kind: str) -> None:
    """Dispatch to the checker for the credential SHAPE this seat declares."""
    if kind == "token_file":
        _validate_token_file(path)
    elif kind == "config_home":
        _validate_codex_home(path)
    else:
        raise ControllerError(f"unknown vendor credential kind: {kind!r}")


def _validate_token_file(path: Path) -> None:
    from codeaction.agents.vendor.claude_auth import read_credentials
    try:
        read_credentials(path)
    except ValueError as exc:
        raise ControllerError(str(exc)) from exc


def _validate_provider_env_file(path: Path) -> None:
    """Accept only ``<ALIAS>_KEY`` / ``<ALIAS>_BASE_URL`` for aliases the registry actually uses.

    Model identity and capabilities are registry facts, not credential-file authority: a file
    nobody can audit must never be able to set a value that enters the comparison hash.
    """
    from codeaction.providers.model_registry import RegistryError, known_models, read_credential_file
    from codeaction.providers.model_registry import resolve_model
    # A model's DECLARED credentials are its primary and its fallbacks -- that is how the registry
    # reads them everywhere else. Allowing only the primary here meant a model could declare a
    # fallback alias it was then forbidden to carry a key for, so the fallback could never be
    # reached: the credential file was rejected before anything got to use it.
    aliases = {
        alias
        for entry in (resolve_model(name) for name in known_models())
        for alias in ((entry.credential,) + tuple(entry.fallback_credentials))
        if alias
    }
    try:
        found = read_credential_file(path, allowed_aliases=aliases)
    except RegistryError as exc:
        raise ControllerError(str(exc)) from exc
    if not found:
        raise ControllerError("credential file declares no key")


def _validated_provider_rate_policy(path: Path | None, entry) -> tuple[dict, dict]:
    if path is None:
        raise ControllerError(
            "provider runs require --provider-rate-limit-file with explicit request/token windows")
    try:
        config = load_rate_limit_config(path.resolve())
        policy = rate_limit_policy_for(config, entry.credential, entry.id)
    except ValueError as exc:
        raise ControllerError(str(exc)) from exc
    if not strict_rate_limit_coverage(policy):
        raise ControllerError(
            "provider rate-limit config requires explicit request/token windows for model "
            f"{entry.id!r} under credential alias {entry.credential!r}")
    return config, policy


def _check_host(gpu: int) -> None:
    _run_output(["docker", "compose", "version", "--format", "json"])
    raw = _run_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
    indices = {int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()}
    if gpu not in indices:
        raise ControllerError(f"GPU {gpu} is not visible to nvidia-smi")


def _load_task(task_name: str, task_pack: Path, *,
               strict_pins: bool = True) -> tuple[dict, str, dict]:
    """Read-only card load for preflight (doctor/smoke/inspect).

    ``strict_pins`` mirrors the run path: a diagnostic must not refuse the very configuration
    the run itself accepts, or `doctor` would reject a modified tool surface that `run`
    happily executes and files as non-submittable.
    """
    if not _SAFE_NAME.fullmatch(task_name):
        raise ControllerError("task name contains unsupported characters")
    try:
        from codeaction.benchmark.taskcard import load_task, validate_task_pack
        pack_info = validate_task_pack(task_pack)
        if task_name not in pack_info["tasks"]:
            raise ValueError(f"task {task_name!r} is not registered in the selected task pack")
        card = load_task(task_name, tasks_root=task_pack, strict_pins=strict_pins)
    except Exception as exc:
        raise ControllerError(f"task package rejected: {exc}") from exc
    instruction = card.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ControllerError("task package has no model instruction")
    return card, instruction, pack_info


def collect_runtime(source_root: Path, sim_ref: str, agent_ref: str, gateway_ref: str,
                    scratch_ref: str, launcher_ref: str, profile: str,
                    task_pack: Path | None = None, expected_agent_cli: str = "claude-code",
                    interface_profile: str = "vendor-mcp-direct",
                    release: dict | None = None) -> RuntimeInfo:
    source_root = source_root.resolve()
    commit, dirty = ((release["source_commit"], False) if release is not None
                     else _git_state(source_root))
    notes: list = []

    def gate(condition: bool, message: str, note: str) -> None:
        """eval refuses; dev records the finding and continues (it lands in
        non_submittable_reasons). Runnability failures stay unconditional raises."""
        if not condition:
            return
        if profile == "eval":
            raise ControllerError(message)
        notes.append(note)

    gate(dirty, "container runs require a clean source worktree", "source_dirty")
    sim = _inspect_image(sim_ref)
    agent = _inspect_image(agent_ref)
    gateway = _inspect_image(gateway_ref)
    gateway_dev = interface_profile == "vendor-mcp-gateway"
    scratch = _inspect_image(scratch_ref) if gateway_dev else None
    launcher = _inspect_image(launcher_ref) if gateway_dev else None
    images = [sim, agent, gateway] + (
        [scratch, launcher] if scratch is not None and launcher is not None else [])
    try:
        if release is None:
            _validate_image_revisions(commit, images, source_root)
    except ControllerError:
        if profile == "eval":
            raise
        notes.append("image_revision_mismatch")
    if profile == "eval" and any(image.digest is None for image in images):
        raise ControllerError(
            "eval profile requires every image pinned by a local content ID or registry digest")

    expected_lock = _sha256(source_root / "docker/requirements.lock")
    gate(sim.labels.get("org.codeaction.lock_sha256") != expected_lock,
         "sim image lock label disagrees with docker/requirements.lock",
         "sim_lock_label_mismatch")
    gate(sim.labels.get("org.codeaction.reproducibility") != "pinned-dev",
         "sim image lacks the pinned-dev reproducibility label",
         "sim_reproducibility_label_missing")
    gate(agent.labels.get("org.codeaction.reproducibility") != "pinned-dev",
         "agent image lacks the pinned-dev reproducibility label",
         "agent_reproducibility_label_missing")
    gate(agent.labels.get("org.codeaction.agent-cli") != expected_agent_cli,
         f"agent image CLI label is not {expected_agent_cli}",
         "agent_cli_label_mismatch")
    cli_version = agent.labels.get("org.codeaction.agent-cli-version", "")
    gate(not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z_.+-]{0,63}", cli_version),
         "agent image has an invalid CLI version label",
         "agent_cli_version_label_invalid")
    if expected_agent_cli == "codeaction-reference":
        expected_reference_lock = _sha256(
            source_root / "docker/reference-agent.requirements.lock")
        gate(agent.labels.get(
                "org.codeaction.reference-lock-sha256") != expected_reference_lock,
             "reference agent image lock label disagrees with "
             "docker/reference-agent.requirements.lock",
             "reference_lock_label_mismatch")
    auxiliaries = [(gateway, "codeaction-gateway")]
    if scratch is not None and launcher is not None:
        auxiliaries.extend([
            (scratch, "codeaction-scratch"),
            (launcher, "codeaction-scratch-launcher"),
        ])
    for image, title in auxiliaries:
        gate(image.labels.get("org.opencontainers.image.title") != title,
             f"auxiliary image title label is not {title}", "aux_image_label_mismatch")
        gate(image.labels.get("org.codeaction.reproducibility") != "pinned-dev",
             f"{title} lacks the pinned-dev reproducibility label",
             "aux_image_label_mismatch")

    if task_pack is None:
        task_pack = source_root / "benchmark/tasks"
    try:
        from codeaction.benchmark.taskcard import validate_task_pack
        pack_info = validate_task_pack(task_pack)
    except Exception as exc:
        raise ControllerError(f"task package rejected: {exc}") from exc
    from codeaction.interface.tool_surface import BASE_TOOL_SET_SHA256
    return RuntimeInfo(
        source_commit=commit,
        integrity_notes=tuple(dict.fromkeys(notes)),
        sim=sim,
        agent=agent,
        gateway=gateway,
        scratch=scratch,
        launcher=launcher,
        lock_sha256=expected_lock,
        dockerfile_sha256=_sha256(source_root / "docker/sim.Dockerfile"),
        task_pack_version=pack_info["taskset_version"],
        task_pack_sha256=pack_info["sha256"],
        base_tool_set_sha256=BASE_TOOL_SET_SHA256,
    )


def build_provenance(info: RuntimeInfo, *, profile: str, task_name: str, seed: int,
                     attempt_index: int, cache_state: str, model: str,
                     agent_label: str, gpu: int, interface_profile: str,
                     expected_identity: dict, orientation_anchor: str = "on") -> dict:
    value = {
        "schema_version": "0.2",
        "runtime": "docker",
        "profile": profile,
        "task_name": task_name,
        "seed": seed,
        "attempt_index": attempt_index,
        "source_commit": info.source_commit,
        "source_dirty": "source_dirty" in info.integrity_notes,
        "sim_image_id": info.sim.image_id,
        "sim_image_digest": info.sim.digest,
        "sim_image_commit": info.sim.labels["org.opencontainers.image.revision"],
        "lock_sha256": info.lock_sha256,
        "dockerfile_sha256": info.dockerfile_sha256,
        "task_pack_version": info.task_pack_version,
        "task_pack_sha256": info.task_pack_sha256,
        "cache_state": cache_state,
        # Declared harness parameter. It does not move base_tool_set_sha256, so provenance is where
        # two runs that differ here are told apart; pooling them would be a category error.
        "orientation_anchor": orientation_anchor,
        "controller_version": CONTROLLER_VERSION,
        "model": model,
        "agent_label": agent_label,
        "interface": (
            "reference-agent"
            if interface_profile in ("reference-mcp", "reference-code-first")
            else "mcp-agent"),
        "interface_profile": interface_profile,
        "gpu": gpu,
        "agent_image_id": info.agent.image_id,
        "agent_image_digest": info.agent.digest,
        "gateway_image_id": info.gateway.image_id,
        "gateway_image_digest": info.gateway.digest,
        "gateway_image_commit":
            info.gateway.labels["org.opencontainers.image.revision"],
        "agent_cli": info.agent.labels["org.codeaction.agent-cli"],
        "agent_cli_version": info.agent.labels["org.codeaction.agent-cli-version"],
        "expected_identity": expected_identity,
    }
    from codeaction.evidence.provenance import validate_provenance
    return validate_provenance(value)


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"{label} must be a JSON object")
    return value


def _safe_run_id(task_name: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task_part = task_name.lower().replace("_", "-")[:32].rstrip("-")
    return f"{task_part}-{stamp}-{uuid.uuid4().hex[:8]}"


def _validated_run_id(value: str) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise ControllerError(
            "run ID must contain 1-63 lowercase letters, digits, or hyphens")
    return value


@contextlib.contextmanager
def _gpu_lock(gpu: int):
    lock_path = Path(f"/tmp/codeaction-gpu-{gpu}.lock")
    with lock_path.open("a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerError(f"GPU {gpu} already has an active codeaction run") from exc
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


@contextlib.contextmanager
def _vendor_subscription_slot(agent_mode: str = "claude"):
    """Fail closed before one more subscription-backed process of THIS vendor can start.

    The ceiling is per host and per vendor: two seats authenticate as two different
    subscriptions, so a Claude run must not consume a Codex slot or vice versa. The Claude lock
    path is unchanged from when it was the only one.
    """
    for slot in range(CLAUDE_SUBSCRIPTION_CONCURRENCY_LIMIT):
        lock_path = Path(f"/tmp/codeaction-{agent_mode}-subscription-{slot}.lock")
        fh = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            continue
        try:
            yield slot
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
        return
    raise ControllerError(
        f"the host already has {CLAUDE_SUBSCRIPTION_CONCURRENCY_LIMIT} active "
        f"{agent_mode} subscription runs")


# The name is kept for the controller test that calls it directly; it is the claude-mode alias.
_claude_subscription_slot = _vendor_subscription_slot


def _compose(argv: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("ab") as log:
        return subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                              check=False).returncode


def collect_filesystem_audit(log_path: Path) -> dict:
    """Build the controller-owned audit summary from gateway/launcher service logs.

    The gateway emits one request ID before every launcher call. The launcher emits exactly one
    result with the same ID after tracing the scratch container. Missing/duplicate/malformed pairs
    are incomplete and therefore fail closed; script output is never an audit input.
    """
    requests, results, errors = set(), {}, []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        lines = []
        errors.append(f"compose log unreadable: {type(exc).__name__}")
    for line in lines:
        for marker, kind in ((_AUDIT_REQUEST_MARK, "request"),
                             (_AUDIT_RESULT_MARK, "result")):
            if marker not in line:
                continue
            try:
                value = json.loads(line.split(marker, 1)[1])
                request_id = value["request_id"]
                if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
                    raise ValueError("invalid request_id")
                if kind == "request":
                    if request_id in requests:
                        errors.append(f"duplicate audit request {request_id}")
                    requests.add(request_id)
                else:
                    if request_id in results:
                        errors.append(f"duplicate audit result {request_id}")
                    results[request_id] = value
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"malformed audit {kind}: {type(exc).__name__}")
    missing = sorted(requests - set(results))
    orphaned = sorted(set(results) - requests)
    if missing:
        errors.append(f"missing audit results: {len(missing)}")
    if orphaned:
        errors.append(f"orphaned audit results: {len(orphaned)}")
    call_outcomes = [results[key].get("outcome") for key in sorted(requests & set(results))]
    invalid = [value for value in call_outcomes if value not in ("clean", "violation")]
    if invalid:
        errors.append(f"incomplete audit calls: {len(invalid)}")
    if errors:
        outcome = "incomplete"
    elif "violation" in call_outcomes:
        outcome = "violation"
    else:
        outcome = "clean"
    calls = []
    for request_id in sorted(requests & set(results)):
        value = results[request_id]
        calls.append({key: value.get(key) for key in (
            "request_id", "outcome", "trace_lines", "trace_truncated", "workspace_bytes",
            "workspace_limit_bytes", "workspace_special_files", "workspace_setid_files",
            "violations")})
    return {"schema_version": "0.1", "backend": "docker-strace", "outcome": outcome,
            "request_count": len(requests), "result_count": len(results),
            "errors": errors, "calls": calls}


def collect_gateway_attestation(log_path: Path, *, expected_profile: str,
                                expected_sha256: str) -> dict:
    values = []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if _GATEWAY_ATTESTATION_MARK not in line:
            continue
        try:
            value = json.loads(line.split(_GATEWAY_ATTESTATION_MARK, 1)[1])
        except json.JSONDecodeError:
            value = {"healthy": False, "error": "malformed gateway attestation"}
        values.append(value)
    if len(values) != 1:
        return {
            "schema_version": "0.1", "healthy": False,
            "error": f"expected one gateway attestation, found {len(values)}",
            "interface_profile": expected_profile,
            "expected_delivered_sha256": expected_sha256,
        }
    value = values[0]
    healthy = (
        value.get("healthy") is True
        and value.get("interface_profile") == expected_profile
        and value.get("expected_delivered_sha256") == expected_sha256
        and value.get("observed_delivered_sha256") == expected_sha256
    )
    return {**value, "healthy": healthy,
            **({} if healthy else {"error": "gateway identity mismatch"})}


def _marked_json(log_path: Path, marker: str) -> list[dict]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    values = []
    for line in lines:
        if marker not in line:
            continue
        try:
            value = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            value = {"healthy": False, "error": f"malformed {marker.strip()} record"}
        values.append(value)
    return values


def _failure_identity(failure):
    """The part of a failure record BOTH sides can be expected to hold.

    `detail_safe` is a human-readable diagnostic minted where the failure was detected -- e.g. the
    sim's "run_code: 1 new unintended contact(s) at physics step 455". The agent's own end event
    never receives that string, so requiring the two records to be equal made every
    `unintended_collision` ending fail attestation, which the controller turns into
    `attempt N failed` and aborts the whole run. Measured 2026-08-07: the first collision-ending
    batch lost attempts 1 and 2 of every task that way, even though the outcome was a perfectly
    scoreable model failure both sides agreed on. Compare what carries identity -- code, origin,
    stage, and the scoreable/retryable classification -- and let the diagnostic differ.
    """
    if not isinstance(failure, dict):
        return failure
    return {key: value for key, value in failure.items() if key != "detail_safe"}


def collect_reference_agent_attestation(
    log_path: Path,
    *,
    expected_scaffold_sha256: str,
    expected_delivered_sha256: str,
    require_scripted_sequence: bool,
    result_path: Path | None = None,
    model_id: str | None = None,
    reasoning_profile: str | None = None,
) -> tuple[dict, list[dict]]:
    events = _marked_json(log_path, _REFERENCE_EVENT_MARK)
    summaries = _marked_json(log_path, _REFERENCE_SUMMARY_MARK)
    meta = [item for item in events if item.get("event") == "meta"]
    ends = [item for item in events if item.get("event") == "end"]
    model_turns = [item for item in events if item.get("event") == "model_turn"]
    terminal_turns = [
        item for item in events
        if item.get("event") in ("endpoint_failure", "context_length_exceeded")
    ]
    tool_events = [
        item for item in events if item.get("event") in ("tool", "done", "limit")
    ]
    errors = []
    # Attestation has one hard job: prove that the scored episode used the frozen scaffold and
    # that the simulator did not execute work the reference agent never issued.  Post-result
    # bookkeeping can still be useful operational evidence, but it must not invalidate an
    # otherwise complete physical result.  Keep those observations in one warning channel rather
    # than growing a manual-accept branch for every harmless controller disagreement.
    warnings = []
    if len(meta) != 1 or len(ends) != 1 or len(summaries) != 1:
        errors.append(
            f"reference records meta/end/summary={len(meta)}/{len(ends)}/{len(summaries)}")
    local_agent = bool(meta and meta[0].get("agent_implementation"))
    if not model_turns and not terminal_turns and not local_agent:
        errors.append("reference transcript has no normalized model turn")
    observed_scaffold = (
        ((meta[0].get("scaffold") or {}).get("config_sha256")) if meta else None)
    request_profile = (
        ((meta[0].get("scaffold") or {}).get("provider_request_profile") or {})
        if meta else {})
    replay_summary, replay_errors = validate_reasoning_replay_events(
        events, request_profile.get("reasoning_replay_evidence"))
    errors.extend(replay_errors)
    observed_delivered = (
        summaries[0].get("delivered_tool_sha256") if summaries else None)
    reference_stats = None
    if ends:
        end = ends[0]
        usage = end.get("usage")
        missing_usage = isinstance(usage, dict) and all(usage.get(k) is None for k in ("prompt_tokens", "completion_tokens"))
        if missing_usage and local_agent:
            warnings.append("token_usage_unavailable")
            reference_stats = dict(end)
        elif not isinstance(usage, dict) or any(
                key not in usage or isinstance(usage.get(key), bool)
                or not isinstance(usage.get(key), int) or usage.get(key) < 0
                for key in ("prompt_tokens", "completion_tokens")):
            errors.append("reference transcript has invalid usage totals")
        else:
            reference_stats = {
                key: end.get(key) for key in (
                    "status", "failure", "budget_used", "total_calls", "turns",
                    "tool_calls_used", "tool_call_budget", "total_tool_dispatches",
                    "model_turns",
                    "reasoning_turns", "infra_retries", "usage", "wall_s", "images_in_context",
                    "image_messages_in_context", "image_blocks_in_context",
                    "unique_images_in_context", "image_blocks_sent", "unique_images_sent",
                    "repeated_image_blocks_sent", "duplicate_observation_ids_suppressed",
                    "text_history_estimated_tokens", "visual_tail_estimated_tokens",
                    "visual_tail_image_count", "visual_tail_source_message_ids",
                    "visual_tail_duplicate_images_suppressed",
                    "requested_output_tokens", "effective_output_tokens",
                    "context_compacted", "protected_units_eroded",
                    "finalize_control_error",
                )
            }
        end_turns = end.get("turns")
        matching_terminal_turns = [
            item for item in terminal_turns if item.get("event") == end.get("status")
        ]
        turn_numbers = [item.get("turn") for item in model_turns + matching_terminal_turns]
        turn_sequence_matches = (
            isinstance(end_turns, int)
            and not isinstance(end_turns, bool)
            and end_turns >= 1
            and all(isinstance(turn, int) and not isinstance(turn, bool)
                    for turn in turn_numbers)
            and sorted(turn_numbers) == list(range(1, end_turns + 1))
            and all(_failure_identity(item.get("failure")) ==
                    _failure_identity(end.get("failure"))
                    for item in matching_terminal_turns)
        )
        if not turn_sequence_matches and not local_agent:
            errors.append("reference model-turn counter mismatch")
        # A run whose model was configured to think but returned none of the text is a defect in
        # the request, not in the model: the transcript looks complete, the tokens are billed, and
        # the reasoning column is silently null. The registry declares which rungs must produce it.
        entry = find_model(model_id) if model_id else None
        if entry is not None:
            observed = end.get("reasoning_turns")
            if observed is None:
                observed = sum(
                    1 for item in model_turns
                    if str(item.get("reasoning_content") or "").strip())
            capture_error = reasoning_capture_error(
                entry, reasoning_profile, len(model_turns), int(observed))
            if capture_error:
                errors.append(capture_error)
        if summaries:
            for key in ("status", "budget_used", "total_calls"):
                if summaries[0].get(key) != end.get(key):
                    errors.append(f"reference summary/end {key} mismatch")
    if observed_scaffold != expected_scaffold_sha256:
        errors.append("reference scaffold config hash mismatch")
    if observed_delivered != expected_delivered_sha256:
        errors.append("reference delivered tool hash mismatch")
    if require_scripted_sequence:
        # The offline script is shaped by the delivered surface: direct calls where the primitives
        # are delivered, one run_code block where they are not. Assert the shape that matches, and
        # look for the image on whichever call actually carried it.
        names = [item.get("tool") for item in tool_events]
        expected_head = (["run_code", None] if names[:1] == ["run_code"]
                         else ["capture_head", "get_robot_state", None])
        if names[:len(expected_head)] != expected_head \
                or not any(item.get("event") == "done" for item in tool_events):
            errors.append(f"scripted reference call order is invalid: {names}")
        carriers = {"capture_head", "run_code"}
        images = max(
            ((item.get("result_projection") or {}).get("model_image_count") or 0)
            for item in tool_events if item.get("tool") in carriers) if any(
                item.get("tool") in carriers for item in tool_events) else 0
        if images < 1:
            errors.append("MCP image did not reach the reference model loop")
    observed_server_status = None
    if result_path is not None:
        try:
            result = json.loads(Path(result_path).read_text(encoding="utf-8"))
            server_stats = result["stats"]
            # The bridge must not let a host-side finalize failure crash the tool channel, so it
            # swallows the exception and logs `finalize_error` into the SIM transcript. Nothing
            # read it: a verifier or persistence fault that still produced a result.json left the
            # attestation healthy. This is the only place holding both files.
            sim_transcript = Path(result_path).with_name("transcript.jsonl")
            if sim_transcript.is_file():
                for line in sim_transcript.read_text(encoding="utf-8").splitlines():
                    if '"finalize_error"' not in line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("event") == "finalize_error":
                        warnings.append(f"host finalize callback failed: {item.get('error')}")
            observed_server_status = server_stats.get("status")
            if not ends or observed_server_status != ends[0].get("status"):
                # Name the transport when it is the cause. A failed finalize control leaves the
                # two sides disagreeing by construction, and reporting only the mismatch blames
                # the episode for a harness fault.
                transport = ends[0].get("finalize_control_error") if ends else None
                message = (
                    f"reference finalize control never reached the simulator ({transport})"
                    if transport else "reference/sim termination status mismatch")
                # A declared finalize-control transport failure after result.json exists is a
                # lifecycle diagnostic.  An unexplained status disagreement remains a hard
                # transcript-integrity error.
                (warnings if transport else errors).append(message)
            # The simulator adopts the agent's counters at finalize, so these two comparisons
            # now verify the wiring rather than the counts. The counts themselves are checked
            # through the signed difference the adoption recorded: it may never be negative,
            # which would mean the simulator executed calls the agent never made.
            signed_excess = {}
            for field in ("reference_only_calls", "reference_only_budget"):
                excess = server_stats.get(field)
                signed_excess[field] = (
                    isinstance(excess, int) and not isinstance(excess, bool) and excess >= 0)
                if excess is not None and not signed_excess[field]:
                    errors.append(f"simulator ran ahead of the reference agent ({field}={excess})")
            if ends and _failure_identity(server_stats.get("failure")) != _failure_identity(
                    ends[0].get("failure")):
                errors.append("reference/sim runtime failure mismatch")
            if ends and server_stats.get("budget_used") != ends[0].get("budget_used"):
                (warnings if signed_excess["reference_only_budget"] else errors).append(
                    "reference/sim budget counter mismatch")
            if ends and server_stats.get("total_calls") != ends[0].get("total_calls"):
                (warnings if signed_excess["reference_only_calls"] else errors).append(
                    "reference/sim total-call counter mismatch")
        except (KeyError, TypeError, OSError, json.JSONDecodeError):
            errors.append("reference/sim result contract is unreadable")
    return {
        "schema_version": "1.2",
        "healthy": not errors,
        "expected_scaffold_config_sha256": expected_scaffold_sha256,
        "observed_scaffold_config_sha256": observed_scaffold,
        "expected_delivered_sha256": expected_delivered_sha256,
        "observed_delivered_sha256": observed_delivered,
        "observed_server_status": observed_server_status,
        "events": len(events),
        "model_turns": None if local_agent else len(model_turns),
        "stats": reference_stats,
        "reasoning_replay": replay_summary,
        "errors": errors,
        "warnings": warnings,
    }, events


def collect_raw_mcp_surface(log_path: Path, *, expected_sha256: str) -> dict:
    values = _marked_json(log_path, _RAW_MCP_SURFACE_MARK)
    if len(values) != 1:
        return {
            "schema_version": "1.0", "healthy": False,
            "error": f"expected one raw MCP surface record, found {len(values)}",
            "expected_delivered_sha256": expected_sha256,
        }
    value = values[0]
    healthy = (
        value.get("healthy") is True
        and value.get("expected_delivered_sha256") == expected_sha256
        and value.get("observed_delivered_sha256") == expected_sha256
    )
    return {**value, "healthy": healthy}


def collect_vendor_mcp_health(log_path: Path) -> dict:
    values = _marked_json(log_path, _VENDOR_MCP_HEALTH_MARK)
    if len(values) != 1:
        return {
            "schema_version": "1.0", "healthy": False,
            "error": f"expected one vendor MCP health record, found {len(values)}",
        }
    value = values[0]
    healthy = (
        value.get("healthy") is True
        and value.get("strict_mcp_config") is True
        and value.get("credential_supplied") is False
        and value.get("server") == "codeaction"
    )
    return {**value, "healthy": healthy}


def _record_gateway_attestation(attempt: Path, attestation: dict) -> None:
    """Bind the controller-observed gateway surface to host-authored run artifacts."""
    for name in ("result.json", "run_meta.json"):
        path = attempt / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControllerError(f"cannot update gateway attestation in {name}: {exc}") from exc
        if not isinstance(value, dict):
            raise ControllerError(f"cannot update gateway attestation in {name}: not an object")
        identity_attestation = value.get("identity_attestation")
        if identity_attestation is None:
            identity_attestation = {}
            value["identity_attestation"] = identity_attestation
        if not isinstance(identity_attestation, dict):
            raise ControllerError(
                f"cannot update gateway attestation in {name}: invalid identity_attestation")
        identity_attestation["gateway_observed"] = dict(attestation)
        identity_attestation["gateway_match"] = attestation.get("healthy") is True
        _atomic_json(path, value)


def _record_reference_attestation(
    attempt: Path,
    attestation: dict,
    events: list[dict],
) -> None:
    transcript_name = "reference_transcript.jsonl"
    transcript_path = attempt / transcript_name
    transcript_path.write_text(
        "".join(json.dumps(
            item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for item in events),
        encoding="utf-8",
    )
    for name in ("result.json", "run_meta.json"):
        path = attempt / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControllerError(
                f"cannot update reference attestation in {name}: {exc}") from exc
        identity_attestation = value.setdefault("identity_attestation", {})
        if not isinstance(identity_attestation, dict):
            raise ControllerError(
                f"cannot update reference attestation in {name}: invalid identity_attestation")
        identity_attestation["reference_agent"] = dict(attestation)
        identity_attestation["reference_agent_match"] = attestation.get("healthy") is True
        value["transcript_bundle"] = {
            "episode_server": "transcript.jsonl",
            "reference_agent": transcript_name,
            "complete_model_and_tool_evidence": attestation.get("healthy") is True,
        }
        _atomic_json(path, value)


def _rebuild_run_report() -> None:
    """Rebuild the HTML report from the HOST, where the tree is writable.

    The sim container mounts the worktree read-only, so the in-episode `finalize_run` could
    stamp `run_meta.json` into the attempt (that path is the writable output mount) but its
    report rebuild always failed with `Read-only file system` and was logged as skipped. The
    index silently stopped advancing on 2026-08-07 while nine days of runs accumulated. The
    metadata half already worked, so only the rebuild moves here, and it stays best effort: a
    report problem must never fail a completed run.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from codeaction.reporting.reports import backfill_legacy, rebuild
        backfill_legacy()
        rebuild()
    except Exception as exc:                     # noqa: BLE001 - reporting is never fatal
        print(f"codeaction: run report not rebuilt ({type(exc).__name__}: {exc})", file=sys.stderr)


def _record_release_status(
    attempt: Path,
    *,
    stage: str,
    eligible: bool,
    blockers: list[str],
) -> None:
    """Write the controller-selected release state into every reportable artifact."""
    for name in ("result.json", "run_meta.json"):
        path = attempt / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControllerError(f"cannot update release status in {name}: {exc}") from exc
        if not isinstance(value, dict):
            raise ControllerError(f"cannot update release status in {name}: not an object")
        value["submission_stage"] = str(stage)
        value["submittable"] = bool(eligible)
        value["non_submittable_reasons"] = list(blockers)
        _atomic_json(path, value)


from codeaction.interface.tool_surface import surface_is_submittable  # noqa: E402

from codeaction.contracts.identity import (  # noqa: E402
    FIXTURE_DRIVER_KIND, REFERENCE_DRIVER_KIND, VENDOR_DRIVER_KIND,
    driver_kind_for_agent_mode as _identity_driver_kind)
from codeaction.agents.vendor.clis import (  # noqa: E402
    VENDOR_AGENT_MODES, is_vendor_mode, vendor_cli)


def _driver_kind_for_agent_mode(agent_mode: str) -> str:
    """One agent stack per mode; raised as a ControllerError to keep the CLI's error surface."""
    try:
        return _identity_driver_kind(agent_mode)
    except ValueError as exc:
        raise ControllerError(str(exc)) from None


def _build_tested_unit(*, interface_profile: str, driver) -> dict:
    """What was tested: the tool interface and the agent stack that drove it.

    Deliberately excludes any verdict ABOUT the run. Two attempts identical in every tested
    respect must share one comparison identity regardless of whether either one is submittable.
    """
    return {"interface_profile": interface_profile, "driver": dict(driver)}


def is_reference_driver(tested_unit) -> bool:
    """True for an agent using the normalized reference MCP transport."""
    return ((tested_unit.get("driver") or {}).get("kind")) in {REFERENCE_DRIVER_KIND, "local_agent"}


def _release_decision(
    args, info=None, extra_reasons=(),
) -> tuple[str, bool, list[str], str | None]:
    """Whether this run counts toward a benchmark submission, and if not, why.

    A verdict ABOUT a run, never a property of what was tested, so it stays out of the comparison
    identity. Every agent stack is submittable on a submittable interface; a submission is exactly a
    set of submittable attempts.
    """
    reasons: list[str] = list(extra_reasons or ())
    from codeaction.launch import context
    execution = context()
    if execution and execution["baseline"] == "matching" and execution.get("contracts_from_runtime"):
        reasons = [r for r in reasons if r not in {"modified_tool_surface", "modified_instruction_surface"}]
    if execution is not None and execution["baseline"] != "matching":
        reasons.append("custom_experiment")
    if info is not None:
        reasons.extend(info.integrity_notes)
    if getattr(args, "transport_conformance", False):
        reasons.append("transport_conformance")
    if bool(getattr(args, "dry_run", False)):
        reasons.append("dry_run")
    if args.agent_mode == "fixture":
        reasons.append("offline_fixture")
    elif args.agent_mode == "reference" and getattr(args, "reference_model_mode", None) == "local":
        reasons.append("local_agent_unqualified")
    elif args.agent_mode == "reference" and \
            getattr(args, "reference_model_mode", None) != "provider":
        reasons.append("scripted_model")
    if not surface_is_submittable(args.interface_profile):
        reasons.append("experimental_interface")
    reasons = sorted(set(reasons))
    stage = "submittable" if not reasons else "non-submittable"
    return stage, (not reasons), reasons, None


def _release_status(args, info=None) -> tuple[str, bool, list[str], str | None]:
    """Compatibility projection used by read-only callers and existing status tests."""
    return _release_decision(args, info)


def _validate_attempt_identity(attempt: Path, expected_identity: dict) -> dict:
    """Require the sim-authored result to echo the controller identity exactly."""
    from codeaction.contracts.identity import sha256_json

    path = attempt / "result.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot validate result identity: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerError("cannot validate result identity: result is not an object")
    attestation = value.get("identity_attestation")
    healthy = (
        value.get("identity") == expected_identity
        and isinstance(attestation, dict)
        and attestation.get("episode_server_match") is True
    )
    return {
        "schema_version": "1.0",
        "healthy": healthy,
        "expected_identity_sha256":
            sha256_json(expected_identity),
        "observed_identity_sha256":
            sha256_json(value.get("identity")) if isinstance(value.get("identity"), dict) else None,
        "episode_server_match": (
            attestation.get("episode_server_match")
            if isinstance(attestation, dict) else None),
    }


def execute_run(args, info: RuntimeInfo, instruction: str,
                compose: Callable[[list[str], dict[str, str], Path], int] = _compose) -> Path:
    from codeaction.benchmark.taskcard import (declared_scene_seeds, instruction_for_scene_seed,
                                  load_task)
    # eval refuses pin drift; dev records it and runs. The comparison identity always
    # carries the surface computed from the code that ran, so a modified capability layer
    # self-declares regardless of what the card pins.
    pin_drift: dict = {}
    card = load_task(args.task, tasks_root=args.task_pack.resolve(),
                     strict_pins=(args.profile == "eval"), drift_sink=pin_drift)
    pin_drift_reasons = sorted(
        {"tool_set": "modified_tool_surface",
         "instructions": "modified_instruction_surface"}[key] for key in pin_drift)
    try:
        declared_seeds = declared_scene_seeds(card, start_seed=args.start_seed)
    except ValueError as exc:
        raise ControllerError(str(exc)) from exc
    if args.attempts < 1 or args.attempts > len(declared_seeds):
        raise ControllerError(
            f"attempts must be in [1, {len(declared_seeds)}] for {args.task}")
    explicit_attempt_index = getattr(args, "attempt_index", None)
    if explicit_attempt_index is None:
        attempt_indices = list(range(args.attempts))
    else:
        if args.attempts != 1:
            raise ControllerError("--attempt-index requires --attempts 1")
        if explicit_attempt_index < 0 or explicit_attempt_index >= len(declared_seeds):
            raise ControllerError(
                f"attempt index must be in [0, {len(declared_seeds) - 1}] for {args.task}")
        attempt_indices = [explicit_attempt_index]
    try:
        attempt_instructions = [
            instruction_for_scene_seed(card, declared_seeds[index])
            for index in attempt_indices
        ]
    except ValueError as exc:
        raise ControllerError(str(exc)) from exc
    unique_instructions = set(attempt_instructions)
    if len(unique_instructions) != 1:
        raise ControllerError(
            "one run cannot aggregate attempts with different model instructions")
    # The task card plus scene seed is authoritative.  ``instruction`` remains in this internal
    # signature only for compatibility with existing callers; never let a second text source enter
    # the rendered prompt or its identity.
    instruction = attempt_instructions[0]

    is_reference = args.agent_mode == "reference"
    from codeaction.extensions import declarations
    local_declaration = declarations("agent").get(args.agent_label, {})
    local_credential = local_declaration.get("credential")
    reported_model = (local_declaration.get("config", {}).get("model", "agent-managed")
                      if is_reference and args.reference_model_mode == "local" else args.model)
    if not isinstance(reported_model, str) or not reported_model:
        raise ControllerError("local agent config.model must be a non-empty declaration when provided")
    provider_rate_config = None
    if is_reference and args.reference_model_mode == "provider" \
            and args.provider_rate_limit_file is not None:
        provider_rate_config = load_rate_limit_config(args.provider_rate_limit_file.resolve())
    model_entry = (
        resolve_model(args.model)
        if is_reference and args.reference_model_mode == "provider" else None)
    provider_rate_policy = (
        rate_limit_policy_for(provider_rate_config, model_entry.credential, model_entry.id)
        if model_entry is not None
        else normalize_rate_limit_policy(None, quota_group=None))
    provider_transport_profile = (
        model_entry.transport_profile() if model_entry is not None else None)

    requested_run_id = getattr(args, "run_id", None)
    run_id = (
        _validated_run_id(requested_run_id)
        if requested_run_id is not None else _safe_run_id(args.task)
    )
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise ControllerError("run directory already exists; results are immutable")
    run_dir.mkdir(parents=True, exist_ok=False)
    submission_stage, submittable, non_submittable_reasons, _release_digest = _release_decision(
        args, info, extra_reasons=pin_drift_reasons)
    _atomic_json(run_dir / "run.json", {
        "schema_version": "0.2",
        "command": getattr(args, "command", "run"),
        "run_id": run_id,
        "runtime": "docker",
        "profile": args.profile,
        "agent_mode": args.agent_mode,
        "interface_profile": args.interface_profile,
        "task_name": args.task,
        "attempts": args.attempts,
        "attempt_indices": attempt_indices,
        "protocol_attempts_declared": len(declared_seeds),
        "start_seed": args.start_seed,
        "declared_scene_seeds": declared_seeds,
        "source_commit": info.source_commit,
        "task_pack_version": info.task_pack_version,
        "task_pack_sha256": info.task_pack_sha256,
        "base_tool_set_sha256": info.base_tool_set_sha256,
        "model": reported_model,
        "reasoning_profile": getattr(args, "reasoning_profile", None) or (
            "high" if is_vendor_mode(args.agent_mode) else "none"),
        "execution_context": execution_context(),
        "claude_subscription_concurrency_limit": (
            CLAUDE_SUBSCRIPTION_CONCURRENCY_LIMIT
            if is_vendor_mode(args.agent_mode) else None),
        "agent_label": args.agent_label,
        "gpu": args.gpu,
        "controller_version": CONTROLLER_VERSION,
        "provider_transport_profile": provider_transport_profile,
        "provider_rate_limit_policy": provider_rate_policy,
        "submission_stage": (
            "credential-free-health"
            if getattr(args, "command", None) == "mcp-health" else submission_stage),
        "submittable": submittable,
        "non_submittable_reasons": non_submittable_reasons,
        "dry_run": bool(args.dry_run),
    })
    mcp_config = {
        "mcpServers": {"codeaction": {
            "command": "python3",
            "args": ["/usr/local/bin/mcp_tcp_client.py", "--host", "gateway", "--port", "8765"],
        }}
    }
    mcp_path = run_dir / "mcp.json"
    _atomic_json(mcp_path, mcp_config)
    mcp_path.chmod(0o600)

    from codeaction.contracts.identity import build_identity_from_card, sha256_json
    from codeaction.contracts.harness_parameters import (EPISODE_TABLE_WORLD_DISABLED,
                                            declared_harness_parameters)
    from codeaction.interface.instructions import reference_instruction_surface, vendor_instruction_surface
    from codeaction.providers.model_adapter import (
        capabilities_for_model, provider_label, provider_request_profile)
    from codeaction.providers.model_registry import registry_identity
    from codeaction.agents.reference.reference_agent import scaffold_card
    from codeaction.interface.tool_surface import surface_identity
    from codeaction.contracts.tool_results import (
        MODEL_VISIBLE_RESULT_MAX_BYTES, RUN_CODE_RESULT_MAX_IMAGES,
        TOOL_RESULT_POLICY_ID, TOOL_RESULT_POLICY_VERSION)
    expected_surface = surface_identity(args.interface_profile, hybrid=True)
    budgets = card["budgets"]
    instruction_builder = reference_instruction_surface if is_reference else vendor_instruction_surface
    # The controller is a DIFFERENT process from the episode host and never assigns
    # CODEACTION_CUROBO_TABLE_WORLD, so it must declare the value the SIM will run under rather than read its
    # own shell. Reading the ambient environment here produced a declared block that disagreed with
    # the host's, and the host's identity preflight then refused every attempt on
    # `instruction_surface`.
    harness_parameters = declared_harness_parameters(
        orientation_anchor=getattr(args, "orientation_anchor", "on") == "on",
        table_world_disabled=EPISODE_TABLE_WORLD_DISABLED)
    instruction_kwargs = {}
    if is_vendor_mode(args.agent_mode):
        # The discovery sentence is a fact about the seat's CLI, so it comes from the seat table.
        # The offline fixture is neither reference nor vendor and keeps the default sentence --
        # the gate's own smoke is what runs it.
        instruction_kwargs["tool_discovery"] = vendor_cli(args.agent_mode).tool_discovery
    rendered_instructions = instruction_builder(
        task_text=instruction,
        max_tool_calls=int(budgets["max_tool_calls"]),
        physical_time_budget_s=float(budgets["physical_time_budget_s"]),
        run_code_max_internal_calls=int(budgets["run_code_max_internal_calls"]),
        harness_parameters=harness_parameters,
        **instruction_kwargs,
    )
    instruction_identity = {
        key: rendered_instructions[key] for key in (
            "instruction_contract_sha256", "instruction_surface_sha256",
            "fragment_ids", "fragment_manifest")
    }
    environment_identity = {
        "runtime": "docker",
        "source_commit": info.source_commit,
        "sim_image_digest": info.sim.digest or info.sim.image_id,
        "environment_lock_sha256": info.lock_sha256,
        "asset_manifest_sha256": sha256_json(card["scene"].get("asset_pins") or {}),
        "embodiment": card["scene"].get("embodiment"),
        "task_config_sha256": sha256_json(card["scene"]),
    }
    from codeaction.launch import context
    execution = context()
    if execution is not None:
        environment_identity["code_components"] = execution["components"]
    reference_capabilities = None
    expected_scaffold_sha256 = None
    reasoning_profile = getattr(args, "reasoning_profile", None) or (
        "high" if is_vendor_mode(args.agent_mode) else "none")
    transport_conformance = bool(getattr(args, "transport_conformance", False))
    if is_reference:
        reference_capabilities = capabilities_for_model(args.model)
        if args.reference_model_mode in {"scripted", "local"}:
            reference_capabilities = replace(
                reference_capabilities, model_seed_support="supported")
        reference_request_profile = provider_request_profile(
            args.model,
            scripted=args.reference_model_mode in {"scripted", "local"},
            reasoning_profile=reasoning_profile)
        reference_scaffold = scaffold_card(
            reference_capabilities, request_profile=reference_request_profile,
            transport_profile=provider_transport_profile,
            rate_limit_policy=provider_rate_policy,
            implementation=args.agent_label if args.reference_model_mode == "local" else None)
        expected_scaffold_sha256 = reference_scaffold["config_sha256"]
        driver = {
            "kind": "local_agent" if args.reference_model_mode == "local" else "benchmark_reference_scaffold",
            "id": info.agent.labels["org.codeaction.agent-cli"],
            "version": info.agent.labels["org.codeaction.agent-cli-version"],
            "image_digest": info.agent.digest or info.agent.image_id,
            "config_sha256": expected_scaffold_sha256,
            "mcp_profile": args.interface_profile,
            "mcp_transport": "reference-mcp-client@1.0.0",
        }
        model_identity = {
            "provider": (
                "scripted" if args.reference_model_mode == "scripted"
                else provider_label(args.model)),
            "id": args.model,
            "reasoning": reference_request_profile["reasoning"],
            "temperature": reference_scaffold["temperature"],
            "requested_output_tokens": reference_scaffold["requested_output_tokens"],
            "effective_output_tokens": reference_scaffold["effective_output_tokens"],
            "capabilities": reference_capabilities.to_dict(),
            "request_profile": reference_request_profile,
            "transport_profile": reference_scaffold["provider_transport_profile"],
            "rate_limit_policy": reference_scaffold["provider_rate_limit_policy"],
        }
        if args.reference_model_mode == "local":
            driver.update(id=args.agent_label, version=local_declaration.get("version", "local"))
            model_identity = {"id": reported_model, "provider": "agent-managed",
                              "evidence": "declaration" if reported_model != "agent-managed" else "unavailable",
                              "config": local_declaration.get("config", {})}
    else:
        driver = {
            "kind": _driver_kind_for_agent_mode(args.agent_mode),
            "id": info.agent.labels["org.codeaction.agent-cli"],
            "version": info.agent.labels["org.codeaction.agent-cli-version"],
            "image_digest": info.agent.digest or info.agent.image_id,
            "config_sha256": sha256_json({
                "agent_mode": args.agent_mode,
                "agent_label": args.agent_label,
                "interface_profile": args.interface_profile,
                "model": args.model,
                "effort": reasoning_profile,
                "subscription_concurrency_limit": CLAUDE_SUBSCRIPTION_CONCURRENCY_LIMIT,
                "transport_conformance": transport_conformance,
                "conformance_prompt_sha256": (
                    sha256_json(TRANSPORT_CONFORMANCE_PROMPT)
                    if transport_conformance else None),
                "tool_result_policy": {
                    "id": TOOL_RESULT_POLICY_ID,
                    "version": TOOL_RESULT_POLICY_VERSION,
                    "max_bytes": MODEL_VISIBLE_RESULT_MAX_BYTES,
                },
                "image_policy": {
                    "run_code_result_max_images": RUN_CODE_RESULT_MAX_IMAGES,
                    "reference_context_max_image_rounds": "agent-managed",
                },
            }),
            "mcp_profile": args.interface_profile,
        }
        model_identity = {
            "provider": "vendor-agent",
            "id": args.model,
            "reasoning": f"effort-{reasoning_profile}",
            "temperature": "agent-managed",
            "requested_output_tokens": "agent-managed",
            "effective_output_tokens": "agent-managed",
        }
    tested_unit = _build_tested_unit(
        interface_profile=args.interface_profile, driver=driver)

    for index in attempt_indices:
        seed = declared_seeds[index]
        attempt = run_dir / f"attempt-{index:03d}-seed-{seed:06d}"
        attempt.mkdir(mode=0o700)
        vendor_dir = attempt / "vendor"
        if is_vendor_mode(args.agent_mode) and not args.dry_run:
            vendor_dir.mkdir(mode=0o700)
        expected_identity = build_identity_from_card(
            card,
            {"taskset_version": info.task_pack_version,
             "sha256": info.task_pack_sha256},
            environment=environment_identity,
            tool_surface=expected_surface,
            instruction_surface=instruction_identity,
            tested_unit=tested_unit,
            model=model_identity,
            source_commit=info.source_commit,
            declared_scene_seeds=declared_seeds,
            scene_seed=seed,
            attempt_index=index,
        )
        provenance = build_provenance(
            info, profile=args.profile, task_name=args.task, seed=seed,
            attempt_index=index, cache_state=args.cache_state,
            model=reported_model, agent_label=args.agent_label, gpu=args.gpu,
            interface_profile=args.interface_profile,
            expected_identity=expected_identity,
            orientation_anchor=getattr(args, "orientation_anchor", "on"))
        _atomic_json(attempt / "provenance.json", provenance)
        project_id = f"{run_id}-a{index:03d}"
        status = {"attempt_index": index, "seed": seed,
                  "state": "dry_run" if args.dry_run else "starting"}
        _atomic_json(attempt / "controller_status.json", status)
        if args.dry_run:
            write_artifact_manifest(attempt)
            continue

        provider_state_dir = args.cache_dir.resolve() / "provider-rate-limit-state"
        provider_state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "CODEACTION_GPU": str(args.gpu),
            "CODEACTION_RUN_ID": project_id,
            "CODEACTION_SIM_IMAGE": info.sim.ref,
            "CODEACTION_CLAUDE_AGENT_IMAGE": args.claude_agent_image,
            "CODEACTION_REFERENCE_AGENT_IMAGE": args.reference_agent_image,
            "CODEACTION_FIXTURE_AGENT_IMAGE": args.fixture_agent_image,
            "CODEACTION_GATEWAY_IMAGE": info.gateway.ref,
            # Compose interpolates inactive profiles too. These fallbacks are only placeholders:
            # direct/reference profiles never start scratch-launcher and never inspect these images.
            "CODEACTION_LAUNCHER_IMAGE": (
                info.launcher.ref if info.launcher is not None else args.launcher_image),
            "CODEACTION_SCRATCH_IMAGE_ID": (
                info.scratch.image_id if info.scratch is not None else args.scratch_image),
            "CODEACTION_UID": str(os.getuid()),
            "CODEACTION_GID": str(os.getgid()),
            "CODEACTION_CACHE_DIR": str(args.cache_dir.resolve()),
            "CODEACTION_PROVIDER_STATE_DIR": str(provider_state_dir),
            "CODEACTION_SOURCE_ROOT": str(args.source_root.resolve()),
            "CODEACTION_TASK_PACK_ROOT": str(args.task_pack.resolve()),
            "CODEACTION_ATTEMPT_DIR": str(attempt),
            "CODEACTION_VENDOR_DIR": (
                str(vendor_dir) if is_vendor_mode(args.agent_mode) else "/tmp"),
            "CODEACTION_CODEX_AGENT_IMAGE": args.codex_agent_image,
            "CODEACTION_ASSETS_ROOT": str(args.assets_root.resolve()),
            "CODEACTION_TASK": args.task,
            "CODEACTION_SEED": str(seed),
            "CODEACTION_ORIENTATION_ANCHOR": ("1" if getattr(args, "orientation_anchor", "on") == "on" else "0"),
            "CODEACTION_AGENT_LABEL": args.agent_label,
            "CODEACTION_TOKEN_FILE": (
                str(args.token_file.resolve())
                if args.agent_mode == "claude"
                and getattr(args, "command", None) != "mcp-health"
                else "/dev/null"),
            # Codex authenticates from a DIRECTORY, not a file: it needs auth.json plus the
            # sqlite state it opens at startup. The mount is read-only and the container copies
            # the one credential out of it, so a run can neither mutate nor lock the account.
            "CODEACTION_CODEX_HOME": (
                str(args.token_file.resolve())
                if args.agent_mode == "codex" else "/dev/null"),
            "CODEACTION_PROVIDER_ENV_FILE": (
                str(args.provider_env_file.resolve())
                if is_reference and (args.reference_model_mode == "provider" or
                                     (args.reference_model_mode == "local" and local_credential))
                else "/dev/null"),
            "CODEACTION_MCP_CONFIG": str(mcp_path),
            # Vendor agents receive the complete, task-bound episode configuration before their
            # first turn. Reference agents build their native system/user messages from the same
            # rendered instruction surface inside the isolated agent container, so the reference-scaffold
            # surface carries no controller prompt at all. It still needs a placeholder, for the
            # same reason as the inactive-profile image fallbacks above: compose interpolates the
            # vendor `agent` service even on a reference run that never starts it, and its
            # `${CODEACTION_PROMPT:?}` aborts the whole file on an EMPTY value, not just an unset
            # one. Vendor mode keeps the empty default so that guard stays armed where it matters.
            "CODEACTION_PROMPT": (
                TRANSPORT_CONFORMANCE_PROMPT if transport_conformance else
                rendered_instructions.get(
                    "controller_prompt",
                    "unused: the reference agent renders its own messages"
                    if is_reference else "")),
            "CODEACTION_MODEL": args.model,
            "CODEACTION_CLAUDE_EFFORT": (
                reasoning_profile if args.agent_mode == "claude" else "high"),
            "CODEACTION_CODEX_EFFORT": (
                reasoning_profile if args.agent_mode == "codex" else "high"),
            "CODEACTION_REFERENCE_MODEL_MODE": args.reference_model_mode,
            "CODEACTION_LOCAL_AGENT": args.agent_label,
            "CODEACTION_REASONING_PROFILE": reasoning_profile,
            "CODEACTION_PROVIDER_RATE_LIMIT_POLICY_B64": base64.b64encode(json.dumps(
                provider_rate_policy, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).decode("ascii"),
            "CODEACTION_TASK_TEXT": instruction,
            "CODEACTION_MAX_TOOL_CALLS": str(budgets["max_tool_calls"]),
            "CODEACTION_WALL_BUDGET_S": str(budgets["wall_budget_s"]),
            "CODEACTION_PHYSICAL_TIME_BUDGET_S": str(
                budgets["physical_time_budget_s"]),
            "CODEACTION_RUN_CODE_MAX_INTERNAL_CALLS": str(
                budgets["run_code_max_internal_calls"]),
            "CODEACTION_HARNESS_PARAMETERS_B64": base64.b64encode(json.dumps(
                harness_parameters, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).decode("ascii"),
            "CODEACTION_EXPECTED_SCAFFOLD_SHA256": (
                expected_scaffold_sha256 or "0" * 64),
            "CODEACTION_EXPECTED_ORDERED_NAMES": json.dumps(
                expected_surface["ordered_names"], separators=(",", ":")),
            "CODEACTION_SERVER_INTERFACE_PROFILE": (
                args.interface_profile if is_reference else "vendor-mcp-direct"),
            "CODEACTION_INTERFACE_PROFILE": args.interface_profile,
            "CODEACTION_TOOL_DISCOVERY": (
                vendor_cli(args.agent_mode).tool_discovery
                if is_vendor_mode(args.agent_mode) else "deferred_toolsearch"),
            "CODEACTION_EXPECTED_DELIVERED_SHA256": expected_surface["delivered_sha256"],
        })
        from codeaction.launch import context
        from codeaction.components import compose_mounts
        env.update(compose_mounts(args.source_root.resolve()))
        if context() is not None:
            env["CODEACTION_RUN_CONTEXT"] = str(args.source_root / "config/context.json")
        # Fail here, naming the variable, instead of letting compose abort interpolation with an
        # error attributed to a service this profile never starts. The two are indistinguishable
        # in compose's own output, which is what made an empty CODEACTION_PROMPT read like a broken
        # vendor agent on a run that has no vendor agent.
        blank_required = missing_required_env(env, COMPOSE_FILE.read_text(encoding="utf-8"))
        if blank_required:
            raise ControllerError(
                "compose.yml requires these variables to be set and non-empty: "
                + ", ".join(blank_required))

        log_path = attempt / "compose.log"
        if getattr(args, "command", None) == "mcp-health":
            profile, exit_service = "mcp-health", "vendor-mcp-health"
        else:
            from codeaction.agents.runtime_registry import execution_driver
            descriptor = execution_driver(args.agent_mode)
            profile, exit_service = descriptor.compose_profile, descriptor.compose_service
        profiles = [profile]
        if args.interface_profile == "vendor-mcp-gateway":
            profiles.append("gateway-dev")
        profile_args = [
            value for selected in profiles for value in ("--profile", selected)
        ]
        up = ["docker", "compose", "-f", str(COMPOSE_FILE), *profile_args, "up",
              "--abort-on-container-exit", "--exit-code-from", exit_service]
        down = ["docker", "compose", "-f", str(COMPOSE_FILE), *profile_args, "down",
                "--remove-orphans", "--volumes", "--timeout", "30"]
        up_rc = down_rc = None
        lifecycle_error = None
        try:
            up_rc = compose(up, env, log_path)
        except Exception as exc:
            lifecycle_error = f"compose up raised {type(exc).__name__}"
        try:
            down_rc = compose(down, env, log_path)
        except Exception as exc:
            down_error = f"compose down raised {type(exc).__name__}"
            lifecycle_error = f"{lifecycle_error}; {down_error}" if lifecycle_error else down_error
        audit = collect_filesystem_audit(log_path)
        is_mcp_health = getattr(args, "command", None) == "mcp-health"
        gateway_attestation = (
            {
                "schema_version": "0.1",
                "healthy": True,
                "not_applicable": True,
                "reason": "CLI health proves config/initialize; raw MCP surface is a separate gate",
                "interface_profile": args.interface_profile,
            }
            if is_mcp_health else collect_gateway_attestation(
                log_path,
                expected_profile=args.interface_profile,
                expected_sha256=expected_surface["delivered_sha256"],
            )
        )
        _atomic_json(attempt / "gateway_attestation.json", gateway_attestation)
        if not is_mcp_health:
            _record_gateway_attestation(attempt, gateway_attestation)
        reference_attestation = {"healthy": True, "not_applicable": True}
        if is_reference:
            reference_attestation, reference_events = collect_reference_agent_attestation(
                log_path,
                expected_scaffold_sha256=expected_scaffold_sha256,
                expected_delivered_sha256=expected_surface["delivered_sha256"],
                require_scripted_sequence=args.reference_model_mode == "scripted",
                result_path=attempt / "result.json",
                model_id=args.model,
                reasoning_profile=reasoning_profile,
            )
            _atomic_json(
                attempt / "reference_agent_attestation.json",
                reference_attestation)
            _record_reference_attestation(
                attempt, reference_attestation, reference_events)
        _record_release_status(
            attempt,
            stage=submission_stage,
            eligible=submittable,
            blockers=non_submittable_reasons,
        )
        identity_attestation = (
            {"healthy": True, "not_applicable": True}
            if is_mcp_health else (
                _validate_attempt_identity(attempt, expected_identity)
                if (attempt / "result.json").is_file()
                else {"healthy": False, "error": "result.json missing"}))
        _atomic_json(
            attempt / "controller_identity_attestation.json",
            identity_attestation)
        raw_mcp_surface = {"healthy": True, "not_applicable": True}
        if args.agent_mode == "fixture":
            raw_mcp_surface = collect_raw_mcp_surface(
                log_path, expected_sha256=expected_surface["delivered_sha256"])
            _atomic_json(attempt / "raw_mcp_surface.json", raw_mcp_surface)
        vendor_mcp_health = {"healthy": True, "not_applicable": True}
        if is_mcp_health:
            vendor_mcp_health = collect_vendor_mcp_health(log_path)
            _atomic_json(attempt / "vendor_mcp_health.json", vendor_mcp_health)
        vendor_runtime = {"healthy": True, "not_applicable": True}
        if is_vendor_mode(args.agent_mode) and not is_mcp_health:
            stream_path = vendor_dir / "vendor_stream.jsonl"
            if stream_path.is_file():
                # Each seat declares its own normalizer; both write the same three artifacts with
                # the same field names, so nothing downstream of here knows which CLI ran.
                normalize_stream = importlib.import_module(
                    vendor_cli(args.agent_mode).sidecar_module).normalize_stream
                vendor_runtime = normalize_stream(
                    stream_path,
                    attempt,
                    expected_model=args.model,
                    expected_effort=reasoning_profile,
                    require_conformance=transport_conformance,
                )
                cli_exit_path = vendor_dir / "vendor_cli_exit.json"
                try:
                    cli_exit = json.loads(cli_exit_path.read_text(encoding="utf-8"))
                    cli_exit_code = int(cli_exit["cli_exit_code"])
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    cli_exit_code = up_rc if isinstance(up_rc, int) else 1
                from codeaction.agents.vendor.normalize_exit import normalize
                _atomic_json(attempt / "agent_exit.json", normalize(
                    stream_path,
                    cli_exit_code,
                    server_result=attempt / "result.json",
                ))
            else:
                vendor_runtime = {
                    "schema_version": "1.0",
                    "healthy": False,
                    "errors": ["vendor_stream.jsonl missing"],
                }
                _atomic_json(
                    attempt / "vendor_runtime_attestation.json", vendor_runtime)
        restricted_audit_path = attempt / "filesystem_audit.json"
        if restricted_audit_path.is_file():
            try:
                restricted_audit = json.loads(restricted_audit_path.read_text(encoding="utf-8"))
                if restricted_audit.get("outcome") != "structurally_denied" or \
                        restricted_audit.get("host_filesystem_exposed") is not False:
                    raise ValueError("restricted backend did not deny host filesystem")
                audit["restricted_program_backend"] = restricted_audit
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                audit["outcome"] = "incomplete"
                audit["errors"].append(
                    f"invalid restricted-program audit: {type(exc).__name__}")
        _atomic_json(attempt / "filesystem_audit.json", audit)
        has_result = (attempt / "result.json").is_file()
        audit_clean = audit["outcome"] == "clean"
        gateway_healthy = gateway_attestation.get("healthy") is True
        reference_healthy = reference_attestation.get("healthy") is True
        raw_surface_healthy = raw_mcp_surface.get("healthy") is True
        vendor_health_healthy = vendor_mcp_health.get("healthy") is True
        vendor_runtime_healthy = vendor_runtime.get("healthy") is True
        identity_healthy = identity_attestation.get("healthy") is True
        complete = (up_rc == 0 and down_rc == 0
                    and (has_result or is_mcp_health) and audit_clean
                    and gateway_healthy
                    and reference_healthy and raw_surface_healthy
                    and vendor_health_healthy and vendor_runtime_healthy and identity_healthy
                    and lifecycle_error is None)
        status.update({"state": "complete" if complete else "failed",
                       "compose_up_exit": up_rc, "compose_down_exit": down_rc,
                       "result_present": has_result,
                       "filesystem_audit": audit["outcome"],
                       "filesystem_audit_calls": audit["request_count"],
                       "gateway_attestation": (
                           "not_applicable" if is_mcp_health
                           else ("healthy" if gateway_healthy else "mismatch")),
                       "reference_agent_attestation": (
                           "healthy" if reference_healthy else "mismatch")
                           if is_reference else "not_applicable",
                       "raw_mcp_surface": (
                           "healthy" if raw_surface_healthy else "mismatch")
                           if args.agent_mode == "fixture" else "not_applicable",
                       "vendor_mcp_health": (
                           "healthy" if vendor_health_healthy else "mismatch")
                           if is_mcp_health
                           else "not_applicable",
                       "vendor_runtime_attestation": (
                           "healthy" if vendor_runtime_healthy else "mismatch")
                           if is_vendor_mode(args.agent_mode) and not is_mcp_health
                           else "not_applicable",
                       "controller_identity_attestation": (
                           "not_applicable" if is_mcp_health else
                           ("healthy" if identity_healthy else "mismatch"))})
        if lifecycle_error:
            status["lifecycle_error"] = lifecycle_error
        elif up_rc == 0 and down_rc == 0 and not has_result and not is_mcp_health:
            status["lifecycle_error"] = "compose exited without result.json"
        elif up_rc == 0 and down_rc == 0 and not audit_clean:
            status["lifecycle_error"] = f"filesystem audit is {audit['outcome']}"
        elif up_rc == 0 and down_rc == 0 and not gateway_healthy:
            status["lifecycle_error"] = "gateway tool-surface attestation failed"
        elif up_rc == 0 and down_rc == 0 and not reference_healthy:
            status["lifecycle_error"] = "reference-agent transcript attestation failed"
        elif up_rc == 0 and down_rc == 0 and not raw_surface_healthy:
            status["lifecycle_error"] = "raw MCP tool-surface attestation failed"
        elif up_rc == 0 and down_rc == 0 and not vendor_health_healthy:
            status["lifecycle_error"] = "credential-free vendor MCP health failed"
        elif up_rc == 0 and down_rc == 0 and not vendor_runtime_healthy:
            status["lifecycle_error"] = "vendor runtime attestation failed"
        elif up_rc == 0 and down_rc == 0 and not identity_healthy:
            status["lifecycle_error"] = "controller/sim identity attestation failed"
        try:
            # Regenerate (or produce, for vendor/fixture attempts whose agent-side artifacts
            # only exist after normalization above) the turn-by-turn projection. Must run
            # BEFORE write_artifact_manifest: the verifier reports unlisted files as an
            # integrity failure, so the view has to be part of the sealed inventory.
            # A credential-free MCP health check runs no episode, so it has no transcript to
            # project and must not seal an error file saying so.
            if is_mcp_health:
                raise _SkipTurnsView
            from codeaction.reporting.turns_view import write_turns_view
            write_turns_view(attempt)
            status["turns_view"] = "written"
        except _SkipTurnsView:
            status["turns_view"] = "not_applicable"
        except Exception as exc:
            # Never fatal -- a projection bug must not destroy a finished episode -- but the
            # outcome is recorded in a REQUIRED file so a missing view is visible without
            # anyone thinking to look for its error sidecar.
            status["turns_view"] = f"error: {type(exc).__name__}: {exc}"
            (attempt / "turns.v1.error.json").write_text(
                json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n",
                encoding="utf-8")
        _atomic_json(attempt / "controller_status.json", status)
        write_artifact_manifest(attempt)
        if not complete:
            write_run_summary(run_dir)
            raise ControllerError(f"attempt {index} failed; diagnostics preserved in {log_path}")
    write_run_summary(run_dir)
    _rebuild_run_report()
    return run_dir


def inspect_run(run_dir: Path, *, require_sealed_evidence: bool = False) -> dict:
    run_dir = run_dir.resolve()
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError(f"invalid run.json: {exc}") from exc
    is_mcp_health = (
        run.get("command") == "mcp-health"
        or run.get("submission_stage") == "credential-free-health")
    from codeaction.evidence.provenance import load_provenance
    attempts = []
    for attempt in sorted(run_dir.glob("attempt-*-seed-*")):
        provenance = load_provenance("provenance.json", attempt)
        try:
            status = json.loads((attempt / "controller_status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControllerError(f"invalid controller status in {attempt.name}: {exc}") from exc
        has_result = (attempt / "result.json").is_file()
        has_transcript = (attempt / "transcript.jsonl").is_file()
        has_reference_transcript = (attempt / "reference_transcript.jsonl").is_file()
        audit_path = attempt / "filesystem_audit.json"
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if status.get("state") == "dry_run":
                audit = None
            else:
                raise ControllerError(f"invalid filesystem audit in {attempt.name}: {exc}") from exc
        if status.get("state") == "complete" and not is_mcp_health \
                and not (has_result and has_transcript):
            raise ControllerError(f"complete attempt {attempt.name} lacks result or transcript")
        if status.get("state") == "complete" \
                and provenance["interface_profile"] in (
                    "reference-mcp", "reference-code-first") \
                and not has_reference_transcript:
            raise ControllerError(
                f"complete reference attempt {attempt.name} lacks reference transcript")
        if status.get("state") == "complete" and (not isinstance(audit, dict)
                                                   or audit.get("outcome") != "clean"):
            raise ControllerError(f"complete attempt {attempt.name} lacks a clean filesystem audit")
        manifest_path = attempt / MANIFEST_NAME
        if manifest_path.is_file():
            try:
                artifact_check = verify_artifact_manifest(attempt)
            except ArtifactManifestError as exc:
                raise ControllerError(f"invalid artifact manifest in {attempt.name}: {exc}") from exc
            if not artifact_check["integrity_ok"]:
                details = []
                for key in ("missing_files", "unlisted_files", "modified_files",
                            "missing_required", "dangling_references", "unsafe_paths"):
                    if artifact_check.get(key):
                        details.append(f"{key}={artifact_check[key]}")
                if artifact_check.get("stale_policy"):
                    details.append("stale_policy=true")
                raise ControllerError(
                    f"artifact manifest mismatch in {attempt.name}: " + "; ".join(details))
        else:
            artifact_check = {
                "integrity_ok": False,
                "evidence_complete": False,
                "submittable": False,
            }
        if require_sealed_evidence and status.get("state") == "complete" \
                and not artifact_check["submittable"]:
            raise ControllerError(
                f"attempt {attempt.name} has no sealed, complete evidence")
        attempts.append({"name": attempt.name, "seed": provenance["seed"],
                         "attempt_index": provenance["attempt_index"],
                         "state": status.get("state"), "result": has_result,
                         "transcript": has_transcript,
                         "reference_transcript": has_reference_transcript,
                         "filesystem_audit": audit.get("outcome") if audit else None,
                         "artifact_manifest": manifest_path.is_file(),
                         "artifact_integrity": artifact_check["integrity_ok"],
                         "evidence_complete": artifact_check["evidence_complete"],
                         "submittable": artifact_check["submittable"]})
    if len(attempts) != run.get("attempts"):
        raise ControllerError("attempt directory count disagrees with run.json")
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControllerError(f"invalid summary.json: {exc}") from exc
    else:
        summary = None
    if require_sealed_evidence and (
            not isinstance(summary, dict) or summary.get("protocol_valid") is not True):
        blockers = summary.get("validity_errors") if isinstance(summary, dict) else [
            "summary.json missing or invalid"]
        raise ControllerError(f"run summary is not release-ready: {blockers}")
    return {"run_id": run.get("run_id"), "task_name": run.get("task_name"),
            "profile": run.get("profile"), "dry_run": run.get("dry_run"),
            "summary": {
                "present": isinstance(summary, dict),
                "protocol_complete": summary.get("protocol_complete")
                if isinstance(summary, dict) else None,
                "submittable": summary.get("protocol_valid")
                if isinstance(summary, dict) else None,
                "non_submittable_reasons": summary.get("validity_errors")
                if isinstance(summary, dict) else None,
            },
            "attempts": attempts}


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    # --tier is the readable alias. Three unrelated things are called "profile" in this
    # CLI (run tier, interface profile, reasoning profile) and a fourth in Compose.
    parser.add_argument("--profile", "--tier", dest="profile",
                        choices=("dev", "eval"), default="dev")
    parser.add_argument(
        "--agent-mode",
        choices=(*VENDOR_AGENT_MODES, "fixture", "reference"), default="claude")
    parser.add_argument(
        "--interface-profile", "--surface", "--mcp-profile", dest="interface_profile",
        choices=("reference-mcp", "reference-code-first", "vendor-mcp-direct",
                 "vendor-mcp-gateway"),
        default=None)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--release-manifest", type=Path,
                        default=Path(os.environ["CODEACTION_RELEASE_MANIFEST"])
                        if os.environ.get("CODEACTION_RELEASE_MANIFEST") else None,
                        help="verify release source/assets and use pinned images")
    parser.add_argument("--task-pack", type=Path, default=None,
                        help="task pack root (default: <source-root>/benchmark/tasks)")
    parser.add_argument("--sim-image", default="codeaction-sim:dev")
    parser.add_argument("--claude-agent-image", "--agent-image", dest="claude_agent_image", default="codeaction-claude-agent:dev")
    parser.add_argument("--codex-agent-image", default="codeaction-codex-agent:dev")
    parser.add_argument(
        "--reference-agent-image", default="codeaction-reference-agent:dev")
    parser.add_argument("--fixture-agent-image", default="codeaction-fixture-agent:dev")
    parser.add_argument("--gateway-image", default="codeaction-gateway:dev")
    parser.add_argument("--scratch-image", default="codeaction-scratch:dev")
    parser.add_argument("--launcher-image", default="codeaction-scratch-launcher:dev")
    parser.add_argument("--token-file", type=Path,
                        default=None)
    parser.add_argument("--provider-env-file", type=Path,
                        default=Path.home() / ".config/codeaction/provider.env")
    parser.add_argument(
        "--provider-rate-limit-file", type=Path, default=None,
        help="non-secret JSON RPM/TPM ceilings keyed by credential alias")
    parser.add_argument(
        "--reference-model-mode", choices=("scripted", "provider", "local"), default=None)
    parser.add_argument(
        "--reasoning-profile", choices=REASONING_RUNGS, default=None,
        help="rung of the benchmark reasoning ladder; the registry maps it onto the vendor's "
             "native control and it is recorded in the model/scaffold identity",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "codeaction")
    parser.add_argument("--assets-root", type=Path,
                        default=Path(os.environ.get("CODEACTION_ASSETS_ROOT", ROOT / "assets")))
    parser.add_argument("--gpu", type=int, default=0)


def _validate_common(args) -> tuple[RuntimeInfo, str]:
    if getattr(args, "attempts", 1) < 1 or getattr(args, "attempts", 1) > 100:
        raise ControllerError("attempt count must be in [1, 100]")
    if getattr(args, "start_seed", 0) < 0:
        raise ControllerError("start seed must be non-negative")
    attempt_index = getattr(args, "attempt_index", None)
    if attempt_index is not None:
        if attempt_index < 0:
            raise ControllerError("attempt index must be non-negative")
        if getattr(args, "attempts", 1) != 1:
            raise ControllerError("--attempt-index requires --attempts 1")
    if getattr(args, "gpu", 0) < 0 or getattr(args, "gpu", 0) > 31:
        raise ControllerError("GPU index must be in [0, 31]")
    if args.task_pack is None:
        args.task_pack = args.source_root.resolve() / "benchmark/tasks"
    if args.interface_profile is None:
        args.interface_profile = (
            "reference-mcp"
            if args.agent_mode == "reference" else "vendor-mcp-direct")
    reference_profiles = ("reference-mcp", "reference-code-first")
    if args.agent_mode == "reference" and args.interface_profile not in reference_profiles:
        raise ControllerError(
            f"reference agent mode requires one of {reference_profiles}")
    if args.agent_mode != "reference" and args.interface_profile in reference_profiles:
        raise ControllerError(
            "the reference profiles are available only in reference agent mode")
    # The code-first interface delivers its primitives through run_code, so a non-hybrid run has no way to reach them.
    if args.interface_profile == "reference-code-first" and args.profile != "dev":
        raise ControllerError("reference-code-first is an experimental arm and is dev-only")
    if args.agent_mode == "fixture" and args.interface_profile != "vendor-mcp-direct":
        raise ControllerError("offline fixture mode is pinned to vendor-mcp-direct")
    if args.interface_profile == "vendor-mcp-gateway" and args.profile != "dev":
        raise ControllerError("vendor-mcp-gateway is dev-only")
    if getattr(args, "transport_conformance", False) and (
            not is_vendor_mode(args.agent_mode) or args.profile != "dev"
            or getattr(args, "attempts", 1) != 1):
        raise ControllerError(
            "--transport-conformance requires dev profile, a vendor agent mode, and one attempt")
    if args.agent_mode == "fixture":
        # Fixture mode has no model. Never let caller labels imply a tested model/agent pair.
        args.agent_label = "offline-fixture"
        args.model = "offline-fixture"
    if args.agent_mode == "reference":
        if getattr(args, "agent_label", None) == "claude-code":
            args.agent_label = "codeaction-reference"
        if args.reference_model_mode is None:
            args.reference_model_mode = (
                "scripted" if args.model in ("scripted", "scripted-model") else "provider")
        if args.reference_model_mode in {"scripted", "local"}:
            args.model = "scripted-model"
            # scripted-model declares only 'none'; asserting a thinking rung for a faux provider
            # would put a value in the identity hash that never described the run.
            if args.reasoning_profile is None:
                args.reasoning_profile = "none"
        else:
            if args.reasoning_profile is None:
                from codeaction.benchmark.agents import default_reasoning_for_model
                args.reasoning_profile = default_reasoning_for_model(args.model)
            # Capabilities and the reasoning ladder are registry facts; an unregistered model
            # fails here rather than after a scene has booted.
            try:
                entry = resolve_model(args.model)
            except RegistryError as exc:
                raise ControllerError(str(exc)) from exc
            if args.reasoning_profile not in entry.reasoning_profiles:
                raise ControllerError(
                    f"model {entry.id!r} does not declare reasoning profile "
                    f"{args.reasoning_profile!r}; declared="
                    f"{sorted(entry.reasoning_profiles)}")
            _validate_provider_env_file(args.provider_env_file.resolve())
            _validated_provider_rate_policy(args.provider_rate_limit_file, entry)
    elif is_vendor_mode(args.agent_mode):
        args.reference_model_mode = args.reference_model_mode or "scripted"
        args.reasoning_profile = args.reasoning_profile or "high"
        # The vendor CLI selects the model itself, but the id we stamp into the identity hash
        # must still name a model this benchmark declares. An unregistered id produces a result
        # labelled with a model that never existed. The effort must likewise be a rung that
        # model declares WITH an effort value: the registry is where "this model has xhigh" is
        # written down, and a fixed trio here would have quietly capped every seat at high.
        if args.model not in ("mcp-health-no-model",):
            try:
                entry = resolve_model(args.model)
            except RegistryError as exc:
                raise ControllerError(str(exc)) from exc
            efforts = sorted(rung for rung, profile in entry.reasoning_profiles.items()
                             if profile.get("effort"))
            if args.reasoning_profile not in efforts:
                raise ControllerError(
                    f"vendor-agent effort must be one of {efforts} for {entry.id}")
        elif args.reasoning_profile not in {"low", "medium", "high"}:
            raise ControllerError("vendor-agent effort must be one of low, medium, or high")
    else:
        args.reference_model_mode = args.reference_model_mode or "scripted"
        args.reasoning_profile = args.reasoning_profile or "none"
    _check_host(args.gpu)
    card, _, pack_info = _load_task(args.task, args.task_pack.resolve(),
                                   strict_pins=(getattr(args, 'profile', 'dev') == 'eval'))
    try:
        from codeaction.benchmark.taskcard import declared_scene_seeds, instruction_for_scene_seed
        validation_seeds = declared_scene_seeds(card, start_seed=args.start_seed)
        selected_index = attempt_index if attempt_index is not None else 0
        if selected_index >= len(validation_seeds):
            raise ControllerError(
                f"attempt index must be in [0, {len(validation_seeds) - 1}] for {args.task}")
        instruction = instruction_for_scene_seed(card, validation_seeds[selected_index])
    except ValueError as exc:
        raise ControllerError(str(exc)) from exc
    if is_vendor_mode(args.agent_mode) and getattr(args, "command", None) != "mcp-health":
        # There is no default any more: a vendor run is told which path to mount, by the batch
        # (from the account's declaration) or by hand. Saying so beats mounting a guessed path.
        if args.token_file is None:
            raise ControllerError(
                "a vendor run needs --token-file; the batch supplies it from the subscription "
                "account declared in agents.json")
        _validate_vendor_credential(
            args.token_file.resolve(), vendor_cli(args.agent_mode).credential.kind)
    if not args.cache_dir.is_dir() or not os.access(args.cache_dir, os.W_OK):
        raise ControllerError("cache directory must already exist and be writable")
    if not args.assets_root.is_dir():
        raise ControllerError("assets root does not exist")
    from codeaction.agents.runtime_registry import execution_driver
    descriptor = execution_driver(args.agent_mode)
    expected_cli = descriptor.cli_label
    claude_agent_image = getattr(args, descriptor.image_option)
    runtime_extra = {}
    from codeaction.launch import context
    if context() is not None:
        baseline = args.source_root / "config/baseline.json"
        if baseline.is_file():
            from codeaction.release import verify_assets
            verify_assets(args.assets_root.resolve(), json.loads(baseline.read_text())["assets"])
    if getattr(args, "release_manifest", None) is not None:
        from codeaction.release import apply_release
        try:
            runtime_extra["release"] = apply_release(args)
        except (ValueError, OSError) as exc:
            raise ControllerError(f"release validation failed: {exc}") from exc
        claude_agent_image = getattr(args, descriptor.image_option)
    info = collect_runtime(args.source_root, args.sim_image, claude_agent_image, args.gateway_image,
                           args.scratch_image, args.launcher_image, args.profile,
                           args.task_pack.resolve(), expected_cli,
                           args.interface_profile, **runtime_extra)
    if (info.task_pack_version, info.task_pack_sha256) != (
            pack_info["taskset_version"], pack_info["sha256"]):
        raise ControllerError("task pack changed during runtime validation")
    return info, instruction


def visible_gpus() -> list[int]:
    """Every GPU this machine reports, or [0] when it reports none.

    A front door that defaults to one GPU quietly serialises a batch on a four-GPU box, and the
    default is the setting nobody passes.
    """
    import subprocess
    override = os.environ.get("CUDA_VISIBLE_DEVICES")
    if override:
        indices = [part.strip() for part in override.split(",") if part.strip()]
        return [int(index) for index in indices if index.isdigit()] or [0]
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return [0]
    found = [int(line.strip()) for line in listing.splitlines() if line.strip().isdigit()]
    return found or [0]


# Named batch launch options exposed by the public eval entry point. None preserves
# the matrix runner's defaults, including provider environment configuration.
_EVAL_LAUNCH_OPTIONS = {
    "assets_root": (Path, "installed simulator resource directory"),
    "release_manifest": (Path, "validated release manifest with immutable image references"),
    "provider_env_file": (Path, "local provider credential file"),
    "provider_rate_limit_file": (Path, "provider request/token limits JSON"),
    "token_file": (Path, "local vendor subscription token file"),
    "sim_image": (str, "simulator image reference"),
    "claude_agent_image": (str, "Claude agent image reference"),
    "codex_agent_image": (str, "Codex agent image reference"),
    "reference_agent_image": (str, "reference agent image reference"),
    "gateway_image": (str, "gateway image reference"),
    "default_credential_limit": (int, "maximum concurrent episodes per credential"),
}


def eval_matrix_argv(args) -> list[str]:
    """The batch-runner argv one `codeaction eval` means.

    `eval` is a front door, not a second scheduler: it names the four things someone asks for --
    which models, how many attempts, which tasks, which GPUs -- and hands the rest to the batch
    runner unchanged, so there is one queue, one directory layout and one manifest."""
    from codeaction.benchmark.agents import expand_agent_selection
    models = list(expand_agent_selection(args.models))
    gpus = args.gpus if args.gpus else visible_gpus()
    argv = ["--models", *models, "--attempts", str(args.attempts),
            "--gpus", *[str(gpu) for gpu in gpus],
            "--run-profile", args.run_profile]
    argv += ["--tasks", *args.tasks] if args.tasks else ["--all-tasks"]
    if args.task_pack is not None:
        argv += ["--task-pack", str(args.task_pack)]
    if args.out_dir is not None:
        argv += ["--out-dir", str(args.out_dir)]
    if args.dry_run:
        argv.append("--dry-run")
    for name in _EVAL_LAUNCH_OPTIONS:
        value = getattr(args, name, None)
        if value is not None:
            argv.extend(["--" + name.replace("_", "-"), str(value)])
    return argv + list(args.matrix_args)


def run_eval(args) -> int:
    from codeaction.benchmark.matrix import main as matrix_main
    from codeaction.launch import context
    if context() is not None:
        return matrix_main(eval_matrix_argv(args))
    manifest = getattr(args, "release_manifest", None)
    if manifest is None and os.environ.get("CODEACTION_RELEASE_MANIFEST"):
        manifest = Path(os.environ["CODEACTION_RELEASE_MANIFEST"])
    output = args.out_dir.expanduser().resolve() if args.out_dir is not None else None
    record_path = output.parent / f".{output.name}.execution.json" if output is not None else None
    resuming = record_path is not None and record_path.exists()
    if not resuming and (manifest is None or args.dry_run):
        return matrix_main(eval_matrix_argv(args))
    from codeaction.execution_snapshot import prepare_release_snapshot, save_launch_record
    from codeaction.release import verify_release_source
    if args.out_dir is None:
        raise ControllerError("release evaluation requires out_dir")
    # Existing batches use saved settings, even if the current recipe has changed.
    if resuming:
        record = json.loads(record_path.read_text())
        if record.get("schema_version") != "codeaction-execution.v1":
            raise ControllerError("unsupported saved execution record")
        snapshot = Path(record["source_root"])
        pinned = snapshot / "release-manifest.json"
        # The saved runtime validates its own schema after integrity is checked here.
        verify_release_source(pinned, snapshot)
        argv = record["argv"]
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ControllerError("invalid saved execution arguments")
        if "--out-dir" not in argv or Path(argv[argv.index("--out-dir") + 1]) != output:
            raise ControllerError("saved execution output differs from the selected batch")
    else:
        if args.matrix_args:
            raise ControllerError("release configuration requires named options, not matrix_arg forwarding")
        argv = eval_matrix_argv(args)
        snapshot, pinned = prepare_release_snapshot(
            ROOT, Path(manifest), Path.home() / ".cache/codeaction/executions")
        # Resolve caller paths before changing cwd. Tasks come from the release.
        for flag in ("--out-dir", "--task-pack", "--provider-env-file",
                     "--provider-rate-limit-file", "--token-file", "--assets-root"):
            if flag in argv:
                index = argv.index(flag) + 1
                argv[index] = str(Path(argv[index]).expanduser().resolve())
        if args.task_pack is not None:
            raise ControllerError("release evaluation uses its pinned task pack; select tasks with --tasks")
        if "--release-manifest" in argv:
            argv[argv.index("--release-manifest") + 1] = str(pinned)
        else:
            argv.extend(["--release-manifest", str(pinned)])
        if "--assets-root" not in argv:
            assets = Path(os.environ.get("CODEACTION_ASSETS_ROOT", ROOT / "assets"))
            argv.extend(["--assets-root", str(assets.expanduser().resolve())])
        record = {"schema_version": "codeaction-execution.v1", "source_root": str(snapshot),
                  "argv": argv}
        try:
            save_launch_record(record_path, record)
        except FileExistsError:
            raise ControllerError("another launch prepared this output; rerun to resume its saved configuration")
    if args.dry_run:
        argv = [*argv, "--dry-run"]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(snapshot / "src")
    environment["CODEACTION_ROOT"] = str(snapshot)
    environment["CODEACTION_RELEASE_MANIFEST"] = str(pinned)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "codeaction.benchmark.matrix", *argv],
                          cwd=snapshot, env=environment, check=False).returncode


def regrade_summary(records) -> dict:
    """The rollup a report reads: how many runs moved, in which direction, and what was refused.

    A refused run has no regraded verdict, so it is counted apart and never averaged into the
    rate -- an unknown is not a failure, and a mean that quietly treats it as one would be the
    same error the refusal exists to prevent.
    """
    graded = [r for r in records if r.get("regradable")]
    up = sum(1 for r in graded if r["success"]["changed"] and r["success"]["regraded"])
    down = sum(1 for r in graded if r["success"]["changed"] and not r["success"]["regraded"])
    rate = lambda key: (round(sum(1 for r in graded if r["success"][key]) / len(graded), 4)
                        if graded else None)
    return {
        "runs": len(records), "regradable": len(graded),
        "refused": len(records) - len(graded),
        "refusals": sorted({reason for r in records if not r.get("regradable")
                            for reason in (r.get("blocked_by") or [])})[:20],
        "verdict_changed": up + down, "verdict_up": up, "verdict_down": down,
        "success_rate": {"recorded": rate("recorded"), "regraded": rate("regraded")},
        "cards_changed": sum(1 for r in graded if r.get("card_changed")),
        "milestones_changed": sorted({name for r in graded for name in r["milestones_changed"]}),
        "not_offline_evaluable": sorted({name for r in graded
                                         for name in r["milestones_not_offline_evaluable"]}),
    }


def run_regrade(args) -> int:
    """Re-score every recorded attempt under a path and report what the current cards say."""
    from codeaction.verification.regrade import find_runs, regrade_run
    runs = find_runs(args.path)
    if not runs:
        raise ControllerError(f"no result.json found under {args.path}")
    records, changed = [], []
    for run_dir in runs:
        try:
            record = regrade_run(run_dir, tasks_root=args.task_pack, write=not args.dry_run)
        except Exception as exc:                      # one unreadable run is not a failed sweep
            records.append({"source_run": str(run_dir), "regradable": False,
                            "blocked_by": [f"{type(exc).__name__}: {exc}"]})
            continue
        records.append(record)
        if record.get("regradable") and (record["success"]["changed"]
                                         or record["milestones_changed"]):
            changed.append(record)
    summary = regrade_summary(records)
    summary["written"] = (None if args.dry_run
                          else "regrade.json beside each result.json, regrade_summary.json at the "
                               "root of the tree")
    summary["changed_runs"] = [{"task": r.get("task"), "run": r.get("source_run"),
                                "success": r["success"], "milestones": r["milestones_changed"]}
                               for r in changed][:50]
    if not args.dry_run and not (Path(args.path) / "result.json").is_file():
        (Path(args.path) / "regrade_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    # The line a reader quotes, in the shape the delta actually has.
    delta = summary["success_rate"]
    print(f"regrade delta over {summary['regradable']} regradable run(s): "
          f"{summary['verdict_changed']} changed "
          f"({summary['verdict_up']} up, {summary['verdict_down']} down), "
          f"success rate {delta['recorded']} -> {delta['regraded']}"
          f"; {summary['refused']} refused")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeaction")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="read-only runtime validation")
    _add_runtime_args(doctor)
    # A default task must be one the loader accepts: an unregistered name is refused, so a
    # retired card as the default made the first command in the README fail.
    doctor.add_argument("--task", default=DEFAULT_DEV_TASK)
    doctor.add_argument("--model", default="claude-opus-5")
    doctor.add_argument("--agent-label", default="claude-code")
    doctor.set_defaults(start_seed=0)

    run = sub.add_parser("run", help="run provenance-bound container attempts")
    _add_runtime_args(run)
    run.add_argument("--config", type=Path, help="versioned YAML options; see docs/configuration.md")
    run.add_argument("--task", required=True)
    run.add_argument("--attempts", type=int, default=1)
    run.add_argument(
        "--attempt-index", type=int, default=None,
        help="run exactly one declared protocol attempt index; requires --attempts 1")
    run.add_argument("--start-seed", type=int, default=0)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument(
        "--run-id", default=None,
        help="stable controller-owned ID used for exact compose cleanup on batch recovery")
    run.add_argument("--cache-state", choices=("cold", "warm"), default="warm")
    # Declared harness parameter, not a tool-surface change: it does not move
    # base_tool_set_sha256, so runs that differ here are separated by provenance and by the
    # transcript `meta` record instead. Default is the shipped behaviour.
    run.add_argument("--orientation-anchor", choices=("on", "off"), default="on",
                     help="hold the call-start orientation on every straight leg (default on); "
                          "off reproduces pre-D0-4.2.0 leg semantics for A/B")
    run.add_argument("--model", default="claude-opus-5")
    run.add_argument("--agent-label", default="claude-code")
    run.add_argument(
        "--transport-conformance", action="store_true",
        help="run one non-scoring Claude image-MCP conformance attempt")
    run.add_argument("--dry-run", action="store_true")

    smoke = sub.add_parser(
        "smoke", help="credential-free one-attempt container lifecycle smoke")
    _add_runtime_args(smoke)
    smoke.add_argument("--task", required=True)
    smoke.add_argument("--start-seed", type=int, default=0)
    smoke.add_argument("--model", default="scripted")
    smoke.add_argument("--agent-label", default="claude-code")

    health = sub.add_parser(
        "mcp-health", help="credential-free real vendor-CLI MCP config/health check")
    _add_runtime_args(health)
    health.add_argument("--task", default=DEFAULT_DEV_TASK)
    health.add_argument("--model", default="mcp-health-no-model")
    health.add_argument("--agent-label", default="claude-code")
    health.set_defaults(start_seed=0)

    ev = sub.add_parser(
        "eval", help="measure one or more models over the released task pack")
    ev.add_argument("--config", type=Path, help="versioned YAML options; see docs/configuration.md")
    ev.add_argument("--model", nargs="+", action="extend", dest="models", metavar="MODEL",
                    required=True,
                    help="agent label or model id (registry or your overlay); several per flag, "
                         "and the flag repeats. Also accepts the group selectors "
                         "all-agents / reference-agents / vendor-agents")
    ev.add_argument("--attempts", type=int, default=3,
                    help="attempts per (model, task) cell; the released protocol is 3")
    ev.add_argument("--tasks", nargs="+", default=None,
                    help="task subset; default is every task the pack registers")
    ev.add_argument("--gpus", nargs="+", type=int, default=None,
                    help="GPU indices to run on; defaults to every GPU this machine reports")
    ev.add_argument("--out-dir", type=Path, default=None)
    ev.add_argument("--task-pack", type=Path, default=None)
    ev.add_argument("--profile", "--tier", dest="run_profile", choices=("dev", "eval"),
                    default="dev")
    ev.add_argument("--dry-run", action="store_true",
                    help="print the plan and the per-cell commands, create nothing")
    for name, (kind, description) in _EVAL_LAUNCH_OPTIONS.items():
        flags = ["--" + name.replace("_", "-")]
        if name == "claude_agent_image":
            flags.append("--agent-image")
        ev.add_argument(*flags, dest=name, type=kind, default=None, help=description)
    ev.add_argument("--matrix-arg", action="append", dest="matrix_args", default=[],
                    metavar="ARG",
                    help="pass one argument straight through to the batch runner, for the "
                         "scheduling knobs this front door does not name")

    replay = sub.add_parser("replay", help="optionally replay selected fixed reference sequences")
    replay.add_argument("--config", type=Path, help="YAML options; see configs/replay.example.yaml")
    replay.add_argument("--tasks", nargs="+", required=True)
    replay.add_argument("--gpu", type=int, default=0)
    replay.add_argument("--out-dir", type=Path, default=None)
    replay.add_argument("--assets-root", type=Path,
                        default=Path(os.environ.get("CODEACTION_ASSETS_ROOT", str(ROOT / "assets"))))
    replay.add_argument("--sim-image", default=os.environ.get("CODEACTION_SIM_IMAGE", "codeaction-sim:dev"))
    replay.add_argument("--release-manifest", type=Path,
                        default=os.environ.get("CODEACTION_RELEASE_MANIFEST"))
    replay.add_argument("--dry-run", action="store_true")

    rg = sub.add_parser(
        "regrade", help="re-score recorded runs against the current cards, with no simulator")
    rg.add_argument("path", type=Path,
                    help="a run directory, or any tree containing result.json files")
    rg.add_argument("--task-pack", type=Path, default=None)
    rg.add_argument("--dry-run", action="store_true",
                    help="report without writing regrade.json beside each result")

    inspect = sub.add_parser("inspect", help="validate an existing run directory")
    inspect.add_argument("run_dir", type=Path)
    inspect.add_argument(
        "--require-sealed-evidence", "--require-release-ready",
        dest="require_sealed_evidence", action="store_true",
        help="fail unless every complete attempt has sealed, complete evidence")
    from codeaction.cli.maintenance import add_parsers
    add_parsers(sub)
    for entry in (run, ev, replay):
        entry.add_argument("--extensions", nargs="*", type=Path, default=[])
        entry.add_argument("--model-registry", type=Path, default=None)
        entry.add_argument("--require-release-match", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from codeaction.cli.config import expand_config
    parser = build_parser()
    args = parser.parse_args(expand_config(parser, list(sys.argv[1:] if argv is None else argv)))
    try:
        if args.command in {"changes", "release"}:
            from codeaction.cli.maintenance import main as maintenance_main
            return maintenance_main(args, ROOT)
        from codeaction.launch import launch, context, preview
        if args.command in {"run", "eval", "replay", "smoke", "mcp-health"} and context() is None:
            output = getattr(args, "out_dir", None) or getattr(args, "run_dir", None)
            saved = output and (output.expanduser().resolve().parent / f".{output.name}.execution.json").is_file()
            if not getattr(args, "dry_run", False) or saved:
                return launch(parser, args, ROOT)
            return preview(parser, args, ROOT)
        if args.command == "eval":
            return run_eval(args)
        if args.command == "replay":
            from codeaction.cli.replay import run_replay
            return run_replay(args)
        if args.command == "regrade":
            return run_regrade(args)
        if args.command == "inspect":
            print(json.dumps(inspect_run(
                args.run_dir, require_sealed_evidence=args.require_sealed_evidence),
                indent=2, sort_keys=True))
            return 0
        if args.command in ("smoke", "mcp-health"):
            args.attempts = 1
            args.dry_run = False
            args.cache_state = "warm"
            if args.command == "smoke" and not (
                    args.agent_mode == "fixture"
                    or (args.agent_mode == "reference"
                        and args.model in ("scripted", "scripted-model")
                        and args.reference_model_mode in (None, "scripted"))):
                raise ControllerError(
                    "credential-free smoke accepts only fixture or scripted reference mode")
            if args.command == "mcp-health" and args.agent_mode != "claude":
                # Pinned deliberately: mcp-health runs mcp_health.sh, which lives in the Claude
                # image. It is a transport check for that image, not a seat-neutral one.
                raise ControllerError("mcp-health requires the real claude agent image")
            tmp_root = Path(tempfile.mkdtemp(prefix=f"codeaction-{args.command}-"))
            args.run_dir = tmp_root / "run"
            info, instruction = _validate_common(args)
            if args.command == "mcp-health" \
                    and args.interface_profile != "vendor-mcp-direct":
                raise ControllerError("mcp-health is pinned to vendor-mcp-direct")
            lock = _gpu_lock(args.gpu)
            with lock:
                result = execute_run(args, info, instruction)
            report = inspect_run(result)
            try:
                shutil.rmtree(tmp_root)
            except OSError as exc:
                raise ControllerError(
                    f"cannot remove successful {args.command} artifacts: {exc}") from exc
            print(json.dumps({
                "ok": True,
                "command": args.command,
                "temporary_artifacts_removed": True,
                "report": report,
            }, indent=2, sort_keys=True))
            return 0
        info, instruction = _validate_common(args)
        if args.command == "doctor":
            print(json.dumps({"ok": True, "profile": args.profile, "task": args.task,
                              "source_commit": info.source_commit,
                              "sim_image_id": info.sim.image_id,
                              "agent_image_id": info.agent.image_id,
                              "gateway_image_id": info.gateway.image_id,
                              "scratch_image_id": (
                                  info.scratch.image_id if info.scratch is not None else None),
                              "launcher_image_id": (
                                  info.launcher.image_id if info.launcher is not None else None),
                              "task_pack_version": info.task_pack_version,
                              "task_pack_sha256": info.task_pack_sha256,
                              "base_tool_set_sha256": info.base_tool_set_sha256},
                             indent=2, sort_keys=True))
            return 0
        gpu_lock = contextlib.nullcontext() if args.dry_run else _gpu_lock(args.gpu)
        subscription_lock = (
            _vendor_subscription_slot(args.agent_mode)
            if is_vendor_mode(args.agent_mode) and not args.dry_run
            else contextlib.nullcontext())
        with gpu_lock:
            with subscription_lock:
                result = execute_run(args, info, instruction)
        print(json.dumps({"ok": True, "run_dir": str(result),
                          "dry_run": bool(args.dry_run)}, sort_keys=True))
        return 0
    except (ControllerError, ValueError, OSError) as exc:
        print(f"codeaction: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
