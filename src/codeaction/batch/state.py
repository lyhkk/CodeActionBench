"""Durable state for one manually expanded benchmark batch.

The atomic snapshot is the source of truth.  The JSONL journal is an audit trail, not a replay log.
Every real execution is immutable; retrying a logical episode creates the next execution record.
"""
from __future__ import annotations

import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from codeaction.batch.spec import StageTarget, target_cell_rows


BATCH_SPEC_SCHEMA_VERSION = "1.0"
BATCH_STATE_SCHEMA_VERSION = "2.0"
BATCH_EVENT_SCHEMA_VERSION = "1.0"
# How many times one cell may finish in `retry_wait` before the batch stops re-running it and
# asks a human instead. A model+harness fault is usually deterministic, so retrying it is a
# straight transfer from the user's balance to the provider with no new evidence bought.
MAX_CELL_RETRY_WAITS = 3
# `abandoned` is a terminal human decision: the cell will not be run again in this batch and
# stops holding the stage open. It exists because a deterministic model+harness failure can
# never reach `accepted`, and before it the only options were to keep paying for the same
# failure or to discard the whole batch.
CELL_STATUSES = frozenset({
    "queued", "running", "retry_wait", "accepted", "needs_attention", "abandoned"})
EXECUTION_OUTCOMES = frozenset({"accepted", "retry_wait", "needs_attention"})
ACCEPTED_ATTEMPT_EVIDENCE_SCHEMA = "accepted-attempt-evidence.v1"
ACCEPTED_ATTEMPT_EVIDENCE_FIELDS = frozenset({
    "schema_version",
    "run_dir",
    "attempt_dir",
    "artifact_manifest",
    "artifact_manifest_sha256",
    "run_sha256",
    "result_sha256",
    "identity_sha256",
    "comparison_sha256",
})
ACCEPTED_ATTEMPT_HASH_FIELDS = frozenset({
    "artifact_manifest_sha256",
    "run_sha256",
    "result_sha256",
    "identity_sha256",
    "comparison_sha256",
})


class BatchStateError(RuntimeError):
    """The durable batch state is incompatible, corrupt, or used through an invalid transition."""


class ControllerLockError(BatchStateError):
    """Another controller already owns this batch directory."""


class CellUnavailableError(BatchStateError):
    """A competing worker changed a candidate before this worker acquired it."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BatchStateError(f"batch state value is not JSON serializable: {exc}") from exc


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchStateError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchStateError(f"batch document must be a JSON object: {path}")
    return value


def _safe_segment(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} \
            or any(character in value for character in ("/", "\\", "\0")):
        raise BatchStateError(f"{field} is not a safe path/identity segment: {value!r}")
    return value


def _canonical_relative_path(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise BatchStateError(f"{field} must be a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") \
            or any(part in {"", ".", ".."} for part in path.parts) \
            or path.as_posix() != value:
        raise BatchStateError(f"{field} must be a canonical relative path")
    return path


def _is_strict_descendant(path: PurePosixPath, parent: PurePosixPath) -> bool:
    return len(path.parts) > len(parent.parts) \
        and path.parts[:len(parent.parts)] == parent.parts


def _validate_accepted_attempt_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != ACCEPTED_ATTEMPT_EVIDENCE_FIELDS:
        raise BatchStateError("accepted_attempt evidence fields are incomplete or unknown")
    if value.get("schema_version") != ACCEPTED_ATTEMPT_EVIDENCE_SCHEMA:
        raise BatchStateError("unsupported accepted_attempt evidence schema")

    run_dir = _canonical_relative_path(value.get("run_dir"), "accepted_attempt.run_dir")
    attempt_dir = _canonical_relative_path(
        value.get("attempt_dir"), "accepted_attempt.attempt_dir")
    artifact_manifest = _canonical_relative_path(
        value.get("artifact_manifest"), "accepted_attempt.artifact_manifest")
    if not _is_strict_descendant(attempt_dir, run_dir):
        raise BatchStateError("accepted_attempt.attempt_dir must be under run_dir")
    expected_manifest = attempt_dir / "artifact_manifest.v1.json"
    if artifact_manifest != expected_manifest:
        raise BatchStateError(
            "accepted_attempt.artifact_manifest must name the attempt artifact manifest")

    for field in ACCEPTED_ATTEMPT_HASH_FIELDS:
        digest = value.get(field)
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(character not in "0123456789abcdef" for character in digest):
            raise BatchStateError(
                f"accepted_attempt.{field} must be a 64-character lowercase hex digest")
    return copy.deepcopy(dict(value))


def cell_id(model: str, task: str, attempt_index: int) -> str:
    _safe_segment(model, "model")
    _safe_segment(task, "task")
    if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) or attempt_index < 0:
        raise BatchStateError("attempt_index must be a non-negative integer")
    return f"{model}/{task}/attempt-{attempt_index:03d}"


class _ControllerLock:
    def __init__(self, batch_dir: Path) -> None:
        self.path = batch_dir / ".controller.lock"
        self._stream = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise ControllerLockError(
                f"another controller owns batch directory {self.path.parent}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "acquired_at": _utc_now()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        self._stream = stream

    def release(self) -> None:
        if self._stream is None:
            return
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None


class BatchStateStore:
    """Single-controller owner of a durable batch snapshot and immutable execution history."""

    def __init__(
        self,
        batch_dir: Path,
        specification: dict[str, Any],
        state: dict[str, Any],
        lock: _ControllerLock,
        clock: Callable[[], str],
    ) -> None:
        self.batch_dir = batch_dir
        self.specification = specification
        self._state = state
        self._lock = lock
        self._clock = clock
        self._thread_lock = threading.RLock()
        self._closed = False

    @classmethod
    def open_or_create(
        cls,
        batch_dir: Path,
        *,
        target: StageTarget,
        models: Iterable[str],
        comparison_identity: Mapping[str, Any],
        batch_id: str | None = None,
        model_lanes: Mapping[str, int] | None = None,
        clock: Callable[[], str] = _utc_now,
    ) -> "BatchStateStore":
        root = Path(batch_dir).resolve()
        lock = _ControllerLock(root)
        lock.acquire()
        try:
            store = cls._open_locked(
                root,
                target=target,
                models=models,
                comparison_identity=comparison_identity,
                batch_id=batch_id,
                model_lanes=model_lanes,
                clock=clock,
                lock=lock,
            )
        except BaseException:
            lock.release()
            raise
        return store

    @classmethod
    def open_existing(
        cls,
        batch_dir: Path,
        *,
        clock: Callable[[], str] = _utc_now,
    ) -> "BatchStateStore":
        """Open one existing batch as its sole durable state writer.

        This is deliberately separate from ``open_or_create``: an operator action must never
        initialize a directory or expand a stage merely to resolve already-recorded evidence.
        """
        root = Path(batch_dir).resolve()
        lock = _ControllerLock(root)
        lock.acquire()
        try:
            specification = _read_object(root / "batch_spec.json")
            state = _read_object(root / "batch_state.json")
            cls._assert_invariants(specification, state)
            return cls(root, specification, state, lock, clock)
        except BaseException:
            lock.release()
            raise

    @classmethod
    def _open_locked(
        cls,
        root: Path,
        *,
        target: StageTarget,
        models: Iterable[str],
        comparison_identity: Mapping[str, Any],
        batch_id: str | None,
        clock: Callable[[], str],
        lock: _ControllerLock,
        model_lanes: Mapping[str, int] | None = None,
    ) -> "BatchStateStore":
        selected_models = tuple(models)
        if not selected_models or len(selected_models) != len(set(selected_models)):
            raise BatchStateError("models must be non-empty and contain no duplicates")
        for model in selected_models:
            _safe_segment(model, "model")
        if not isinstance(comparison_identity, Mapping) or not comparison_identity:
            raise BatchStateError("comparison_identity must be a non-empty JSON object")
        # How many episodes of ONE model may hold a lease at the same time. Frozen with the
        # batch rather than passed per call, because it is an invariant the durable state is
        # checked against on every reopen, not a scheduling preference. The default of 1 is what
        # every batch enforced before it was expressible.
        lanes = dict(model_lanes or {})
        unknown_lanes = sorted(set(lanes) - set(selected_models))
        if unknown_lanes:
            raise BatchStateError(f"model_lanes names models outside the batch: {unknown_lanes}")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in lanes.values()):
            raise BatchStateError("model_lanes values must be positive integers")
        resolved_lanes = {model: int(lanes.get(model, 1)) for model in selected_models}
        identity = copy.deepcopy(dict(comparison_identity))
        identity_sha256 = _object_sha256(identity)

        spec_path = root / "batch_spec.json"
        state_path = root / "batch_state.json"
        if spec_path.exists() != state_path.exists():
            raise BatchStateError("batch_spec.json and batch_state.json must either both exist or neither")

        if not spec_path.exists():
            unexpected = sorted(
                path.name for path in root.iterdir()
                if path.name != ".controller.lock"
            )
            if unexpected:
                raise BatchStateError(
                    f"refusing to initialize a non-empty batch directory: {unexpected}")
            now = clock()
            identifier = batch_id or uuid.uuid4().hex
            _safe_segment(identifier, "batch_id")
            specification = {
                "schema_version": BATCH_SPEC_SCHEMA_VERSION,
                "batch_id": identifier,
                "created_at": now,
                "comparison_identity": identity,
                "comparison_sha256": identity_sha256,
                "models": list(selected_models),
                "model_lanes": resolved_lanes,
                "task_pack": {
                    "taskset_version": target.taskset_version,
                    "sha256": target.task_pack_sha256,
                },
            }
            state = cls._initial_state(specification, target, now)
            cls._assert_invariants(specification, state)
            _atomic_json(spec_path, specification)
            _atomic_json(state_path, state)
            cls._append_event_file(root, {
                "schema_version": BATCH_EVENT_SCHEMA_VERSION,
                "batch_id": identifier,
                "revision": state["revision"],
                "observed_at": now,
                "event": "batch_created",
                "detail": {"stage": target.stage, "requested_cells": len(state["requested_cells"])},
            })
            return cls(root, specification, state, lock, clock)

        specification = _read_object(spec_path)
        state = _read_object(state_path)
        cls._validate_resume_identity(
            specification, state, target, selected_models, identity, identity_sha256,
            model_lanes=resolved_lanes if model_lanes is not None else None)
        cls._assert_invariants(specification, state)
        store = cls(root, specification, state, lock, clock)
        store.expand_stage(target)
        return store

    @staticmethod
    def _target_cells(models: Iterable[str], target: StageTarget, now: str) -> dict[str, Any]:
        cells = {}
        for row in target_cell_rows(target, tuple(models)):
            identifier = cell_id(row.model, row.task, row.attempt_index)
            cells[identifier] = {
                "model": row.model,
                "task": row.task,
                "attempt_index": row.attempt_index,
                "scene_seed": row.scene_seed,
                "status": "queued",
                "execution_count": 0,
                "current_execution_id": None,
                "accepted_execution_id": None,
                "lease_id": None,
                "attention_id": None,
                "retry_wait_count": 0,
                "created_at": now,
                "updated_at": now,
            }
        return cells

    @staticmethod
    def _target_dispatch_order(models: Iterable[str], target: StageTarget) -> list[str]:
        selected = tuple(models)
        if target.exact_cells is not None:
            return [row.key for row in target_cell_rows(target, selected)]
        # Keep the historical task-major policy, but freeze it as mutable operator-owned state
        # instead of recomputing it from the score matrix on every scheduler pass.
        return [
            cell_id(model, episode.task, episode.attempt_index)
            for episode in target.episodes for model in selected
        ]

    @classmethod
    def _initial_state(
        cls, specification: Mapping[str, Any], target: StageTarget, now: str,
    ) -> dict[str, Any]:
        cells = cls._target_cells(specification["models"], target, now)
        return {
            "schema_version": BATCH_STATE_SCHEMA_VERSION,
            "batch_id": specification["batch_id"],
            "revision": 1,
            "status": "ready",
            "active_stage": target.stage,
            "updated_at": now,
            "requested_cells": list(cells),
            "dispatch_queue": cls._target_dispatch_order(
                specification["models"], target),
            "stage_history": [{
                "stage": target.stage,
                "activated_at": now,
                "target": target.as_dict(),
            }],
            "cells": cells,
            "executions": {},
            "leases": {},
            "attentions": {},
            # Durable so a controller restart -- which the watchdog performs on its own --
            # cannot silently resume a queue a human deliberately stopped.
            "queue_pause": None,
            # A rolling-window subscription that has spent its budget is held here rather than
            # through an attention: an attention waits for a person, and this waits for a clock.
            # Absent in states written before the governor existed, so every read tolerates that.
            "credential_pauses": {},
        }

    @staticmethod
    def _validate_resume_identity(
        specification: Mapping[str, Any],
        state: Mapping[str, Any],
        target: StageTarget,
        models: tuple[str, ...],
        comparison_identity: Mapping[str, Any],
        comparison_sha256: str,
        model_lanes: Mapping[str, int] | None = None,
    ) -> None:
        if specification.get("schema_version") != BATCH_SPEC_SCHEMA_VERSION:
            raise BatchStateError("unsupported batch_spec.json schema_version")
        if state.get("schema_version") != BATCH_STATE_SCHEMA_VERSION:
            raise BatchStateError("unsupported batch_state.json schema_version")
        if specification.get("batch_id") != state.get("batch_id"):
            raise BatchStateError("batch spec/state IDs disagree")
        if tuple(specification.get("models") or ()) != models:
            raise BatchStateError("selected models differ from the frozen batch specification")
        if model_lanes is not None and dict(specification.get("model_lanes") or {}) != dict(
                model_lanes):
            raise BatchStateError(
                "model_lanes differ from the frozen batch specification")
        if specification.get("comparison_sha256") != comparison_sha256 \
                or specification.get("comparison_identity") != comparison_identity:
            raise BatchStateError("comparison identity differs from the frozen batch specification")
        expected_pack = {
            "taskset_version": target.taskset_version,
            "sha256": target.task_pack_sha256,
        }
        if specification.get("task_pack") != expected_pack:
            raise BatchStateError("task pack differs from the frozen batch specification")

    @staticmethod
    def _append_event_file(root: Path, event: Mapping[str, Any]) -> None:
        path = root / "batch_events.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _ensure_open(self) -> None:
        if self._closed:
            raise BatchStateError("batch state store is closed")

    def _commit(self, event: str | None, detail: Mapping[str, Any] | None = None) -> None:
        self._state["revision"] += 1
        self._state["updated_at"] = self._clock()
        self._refresh_status()
        self._assert_invariants(self.specification, self._state)
        _atomic_json(self.batch_dir / "batch_state.json", self._state)
        if event is not None:
            self._append_event_file(self.batch_dir, {
                "schema_version": BATCH_EVENT_SCHEMA_VERSION,
                "batch_id": self._state["batch_id"],
                "revision": self._state["revision"],
                "observed_at": self._state["updated_at"],
                "event": event,
                "detail": dict(detail or {}),
            })

    def _refresh_status(self) -> None:
        counts = self.stage_counts()
        settled = counts["accepted"] + counts["abandoned"]
        if counts["total"] and settled == counts["total"]:
            # Abandoned cells settle the stage without being accepted; the release set records
            # them as explicit gaps rather than pretending the cell was never requested.
            status = "stage_complete"
        elif counts["running"]:
            status = "running"
        elif counts["queued"] or counts["retry_wait"]:
            status = "ready"
        elif counts["needs_attention"]:
            status = "needs_attention"
        else:
            raise BatchStateError(f"cannot derive batch status from counts: {counts}")
        self._state["status"] = status

    def expand_stage(self, target: StageTarget) -> bool:
        """Add one manually requested superset target; never advance to another stage."""
        with self._thread_lock:
            self._ensure_open()
            expected_pack = self.specification["task_pack"]
            if expected_pack != {
                    "taskset_version": target.taskset_version,
                    "sha256": target.task_pack_sha256}:
                raise BatchStateError("cannot expand a batch with a different task pack")
            now = self._clock()
            desired = self._target_cells(self.specification["models"], target, now)
            existing_requested = set(self._state["requested_cells"])
            desired_ids = set(desired)
            if not existing_requested <= desired_ids:
                raise BatchStateError(
                    "manual stage targets may only expand; the requested target would remove cells")
            existing_cells = self._state["cells"]
            for identifier in existing_requested:
                expected = desired[identifier]
                actual = existing_cells[identifier]
                for field in ("model", "task", "attempt_index", "scene_seed"):
                    if actual.get(field) != expected[field]:
                        raise BatchStateError(
                            f"cell identity drift for {identifier}: field {field}")
            new_ids = [identifier for identifier in desired if identifier not in existing_cells]
            same_target = (
                self._state["active_stage"] == target.stage
                and self._state["requested_cells"] == list(desired)
            )
            if same_target:
                return False
            for identifier in new_ids:
                existing_cells[identifier] = desired[identifier]
            self._state["active_stage"] = target.stage
            self._state["requested_cells"] = list(desired)
            self._state["dispatch_queue"] = self._target_dispatch_order(
                self.specification["models"], target)
            self._state["stage_history"].append({
                "stage": target.stage,
                "activated_at": now,
                "target": target.as_dict(),
            })
            self._commit("stage_expanded", {
                "stage": target.stage,
                "added_cells": len(new_ids),
                "requested_cells": len(desired),
            })
            return True

    def prioritize_cells(self, identifiers: Iterable[str], *, note: str) -> None:
        """Move exact requested cells to the front without changing score coverage."""
        ordered = tuple(identifiers)
        if not ordered or len(ordered) != len(set(ordered)):
            raise BatchStateError("priority cells must be a non-empty unique list")
        if not isinstance(note, str) or not note.strip():
            raise BatchStateError("priority note must be non-empty")
        with self._thread_lock:
            self._ensure_open()
            requested = list(self._state["requested_cells"])
            unknown = [identifier for identifier in ordered if identifier not in requested]
            if unknown:
                raise BatchStateError(f"priority cells are not requested: {unknown}")
            current = list(self._state.get("dispatch_queue") or requested)
            self._state["dispatch_queue"] = [
                *ordered, *(identifier for identifier in current if identifier not in ordered),
            ]
            self._commit("dispatch_queue_prioritized", {
                "cells": list(ordered), "note": note.strip(),
            })

    def acquire_cell(
        self,
        identifier: str,
        *,
        worker_id: str,
        gpu: int,
        deadline_at: str | None = None,
        lease_id: str | None = None,
    ) -> dict[str, Any]:
        with self._thread_lock:
            self._ensure_open()
            _safe_segment(worker_id, "worker_id")
            if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0:
                raise BatchStateError("gpu must be a non-negative integer")
            try:
                cell = self._state["cells"][identifier]
            except KeyError as exc:
                raise BatchStateError(f"unknown cell {identifier!r}") from exc
            if identifier not in self._state["requested_cells"]:
                raise BatchStateError(f"cell is not requested by the active stage: {identifier}")
            if cell["status"] != "queued":
                raise CellUnavailableError(
                    f"cell {identifier} must be queued before acquire, got {cell['status']}")
            lanes = self.model_lanes(cell.get("model"))
            in_flight = 0
            for active in self._state["leases"].values():
                active_cell = self._state["cells"][active["cell_id"]]
                if active.get("gpu") == gpu:
                    raise CellUnavailableError(f"GPU {gpu} already has an active batch lease")
                if active_cell.get("model") == cell.get("model"):
                    in_flight += 1
            if in_flight >= lanes:
                raise CellUnavailableError(
                    f"model {cell['model']} already holds {in_flight} of its {lanes} "
                    f"concurrent batch lease(s)")
            now = self._clock()
            execution_number = int(cell["execution_count"]) + 1
            execution_id = f"execution-{execution_number:03d}"
            lease_identifier = lease_id or uuid.uuid4().hex
            _safe_segment(lease_identifier, "lease_id")
            if lease_identifier in self._state["leases"]:
                raise BatchStateError(f"duplicate lease_id {lease_identifier}")
            execution_key = f"{identifier}/{execution_id}"
            if execution_key in self._state["executions"]:
                raise BatchStateError(f"duplicate execution {execution_key}")
            lease = {
                "lease_id": lease_identifier,
                "cell_id": identifier,
                "execution_id": execution_id,
                "worker_id": worker_id,
                "gpu": gpu,
                "acquired_at": now,
                "heartbeat_at": now,
                "deadline_at": deadline_at,
                "launch_state": "reserved",
                "launch_protocol": None,
                "command_sha256": None,
                "target_command_sha256": None,
                "process_start_id": None,
                "pid": None,
                "pgid": None,
                "run_dir": None,
                "compose_project": None,
            }
            execution = {
                "execution_id": execution_id,
                "cell_id": identifier,
                "status": "running",
                "started_at": now,
                "finished_at": None,
                "lease_id": lease_identifier,
                "detail": {},
            }
            cell.update({
                "status": "running",
                "execution_count": execution_number,
                "current_execution_id": execution_id,
                "lease_id": lease_identifier,
                "attention_id": None,
                "updated_at": now,
            })
            self._state["executions"][execution_key] = execution
            self._state["leases"][lease_identifier] = lease
            self._commit("cell_acquired", {
                "cell_id": identifier,
                "execution_id": execution_id,
                "lease_id": lease_identifier,
                "worker_id": worker_id,
                "gpu": gpu,
            })
            return copy.deepcopy(lease)

    def update_lease(self, lease_id: str, **values: Any) -> None:
        allowed = {
            "heartbeat_at", "deadline_at", "launch_state", "command_sha256",
            "target_command_sha256", "launch_protocol", "process_start_id", "pid", "pgid",
            "run_dir", "compose_project",
        }
        unknown = set(values) - allowed
        if unknown:
            raise BatchStateError(f"unsupported lease fields: {sorted(unknown)}")
        with self._thread_lock:
            self._ensure_open()
            try:
                lease = self._state["leases"][lease_id]
            except KeyError as exc:
                raise BatchStateError(f"unknown active lease {lease_id!r}") from exc
            lease.update(values)
            lease["heartbeat_at"] = values.get("heartbeat_at", self._clock())
            self._commit(None)

    def finish_execution(
        self,
        lease_id: str,
        *,
        outcome: str,
        detail: Mapping[str, Any] | None = None,
        next_retry_at: str | None = None,
    ) -> str:
        if outcome != "retry_wait":
            raise BatchStateError(f"unsupported execution outcome {outcome!r}")
        with self._thread_lock:
            self._ensure_open()
            identifier = self._finish_execution_locked(
                lease_id,
                outcome=outcome,
                detail=detail,
                attention_id=None,
                next_retry_at=next_retry_at,
            )
            return identifier

    def finish_accepted_execution(
        self,
        lease_id: str,
        accepted_attempt: Mapping[str, Any],
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> str:
        evidence = _validate_accepted_attempt_evidence(accepted_attempt)
        if detail is not None and not isinstance(detail, Mapping):
            raise BatchStateError("accepted execution detail must be an object")
        execution_detail = copy.deepcopy(dict(detail or {}))
        if "accepted_attempt" in execution_detail:
            raise BatchStateError(
                "accepted execution detail must not override accepted_attempt evidence")
        execution_detail["accepted_attempt"] = evidence
        with self._thread_lock:
            self._ensure_open()
            return self._finish_execution_locked(
                lease_id,
                outcome="accepted",
                detail=execution_detail,
                attention_id=None,
                next_retry_at=None,
            )

    def finish_with_attention(
        self,
        lease_id: str,
        *,
        attention: Mapping[str, Any],
        detail: Mapping[str, Any] | None = None,
    ) -> str:
        """Atomically finish a running execution and open its human-attention record."""
        with self._thread_lock:
            self._ensure_open()
            try:
                lease = self._state["leases"][lease_id]
            except KeyError as exc:
                raise BatchStateError(f"unknown active lease {lease_id!r}") from exc
            identifier = lease["cell_id"]
            execution_id = lease["execution_id"]
            attention_id = self._open_attention_locked(
                attention, cell_id=identifier, execution_id=execution_id)
            self._finish_execution_locked(
                lease_id,
                outcome="needs_attention",
                detail=detail,
                attention_id=attention_id,
                next_retry_at=None,
            )
            return attention_id

    def _open_attention_locked(
        self, attention: Mapping[str, Any], *, cell_id: str, execution_id: str,
    ) -> str:
        """Record one open attention. Caller holds the thread lock and commits afterwards."""
        attention_id = f"A-{len(self._state['attentions']) + 1:04d}"
        if attention_id in self._state["attentions"]:
            raise BatchStateError(f"duplicate attention ID {attention_id}")
        now = self._clock()
        record = copy.deepcopy(dict(attention))
        record.update({
            "attention_id": attention_id,
            "status": "open",
            "cell_id": cell_id,
            "execution_id": execution_id,
            "opened_at": now,
            "updated_at": now,
            "resolution": None,
        })
        self._state["attentions"][attention_id] = record
        return attention_id

    def _finish_execution_locked(
        self,
        lease_id: str,
        *,
        outcome: str,
        detail: Mapping[str, Any] | None,
        attention_id: str | None,
        next_retry_at: str | None,
    ) -> str:
        try:
            lease = self._state["leases"][lease_id]
        except KeyError as exc:
            raise BatchStateError(f"unknown active lease {lease_id!r}") from exc
        identifier = lease["cell_id"]
        cell = self._state["cells"][identifier]
        execution_id = lease["execution_id"]
        execution_key = f"{identifier}/{execution_id}"
        execution = self._state["executions"][execution_key]
        if cell.get("status") != "running" or cell.get("lease_id") != lease_id \
                or execution.get("status") != "running":
            raise BatchStateError(f"running execution invariant failed for lease {lease_id}")
        now = self._clock()
        execution.update({
            "status": outcome,
            "finished_at": now,
            "detail": copy.deepcopy(dict(detail or {})),
        })
        retries = int(cell.get("retry_wait_count") or 0)
        if outcome == "retry_wait":
            retries += 1
            if retries >= MAX_CELL_RETRY_WAITS and attention_id is None:
                # Stop paying for a failure that has already repeated. The cell keeps its
                # evidence and waits for a person, who can fix the cause and requeue it, or
                # abandon it. `needs_attention` is a durable pause of THIS cell only.
                outcome = "needs_attention"
                attention_id = self._open_attention_locked({
                    "kind": "repeated_failure",
                    "scope": {"kind": "cell", "id": identifier},
                    "summary": (f"cell finished in retry_wait {retries} times; "
                                "the batch stopped re-running it"),
                    "model": cell.get("model"),
                }, cell_id=identifier, execution_id=execution_id)
        elif outcome == "accepted":
            retries = 0
        execution["status"] = outcome
        cell.update({
            "status": outcome,
            "current_execution_id": None,
            "lease_id": None,
            "attention_id": attention_id,
            "next_retry_at": next_retry_at if outcome == "retry_wait" else None,
            "retry_wait_count": retries,
            "updated_at": now,
        })
        if outcome == "accepted":
            if cell.get("accepted_execution_id") is not None:
                raise BatchStateError(f"cell {identifier} already has an accepted execution")
            cell["accepted_execution_id"] = execution_id
        del self._state["leases"][lease_id]
        self._commit("execution_finished", {
            "cell_id": identifier,
            "execution_id": execution_id,
            "outcome": outcome,
            "attention_id": attention_id,
        })
        return identifier

    def abandon_cell(self, identifier: str, *, note: str) -> None:
        """Retire a cell this batch will not run again. Requires a written reason.

        Only a cell that is not currently executing may be abandoned: a running episode is left
        to finish so its evidence is never truncated mid-flight, and an accepted cell is already
        settled. The note is the record of why a leaderboard row has a gap.
        """
        if not isinstance(note, str) or not note.strip():
            raise BatchStateError("abandon note must be non-empty")
        with self._thread_lock:
            self._ensure_open()
            try:
                cell = self._state["cells"][identifier]
            except KeyError as exc:
                raise BatchStateError(f"unknown cell {identifier!r}") from exc
            previous = cell["status"]
            if previous not in {"queued", "retry_wait", "needs_attention"}:
                raise BatchStateError(
                    f"cell {identifier} cannot be abandoned from {previous}")
            cell.update({
                "status": "abandoned",
                "next_retry_at": None,
                "abandon_note": note.strip(),
                "updated_at": self._clock(),
            })
            self._commit("cell_abandoned", {
                "cell_id": identifier,
                "previous_status": previous,
                "attention_id": cell.get("attention_id"),
                "note": note.strip(),
            })


    def credential_pauses(self) -> dict[str, dict[str, Any]]:
        """Window pauses as recorded. Missing key means a state written before the governor."""
        return dict((self._state or {}).get("credential_pauses") or {})

    def open_credential_pause(self, alias: str, *, resume_at: float, note: str,
                              utilization: float | None = None) -> None:
        """Hold one credential until `resume_at`.

        Not an attention: an attention is a durable stop that waits for a person to look at it,
        and a spent window waits for a clock. Re-opening extends the hold rather than shortening
        it -- two episodes reporting the same window must not let the later, smaller reading move
        the resume time earlier.
        """
        if not isinstance(alias, str) or not alias:
            raise BatchStateError("credential pause needs an alias")
        if not isinstance(note, str) or not note.strip():
            raise BatchStateError("credential pause needs a note")
        with self._thread_lock:
            self._ensure_open()
            pauses = self._state.setdefault("credential_pauses", {})
            existing = pauses.get(alias) or {}
            previous = float(existing.get("resume_at") or 0.0)
            pauses[alias] = {
                "resume_at": max(float(resume_at), previous),
                "note": note,
                "utilization": utilization,
                "opened_at": self._clock(),
            }
            self._commit("credential_paused", {"credential": alias})

    def clear_due_credential_pauses(self, now: float) -> tuple[str, ...]:
        """Release every window pause whose time has come. Returns the aliases released."""
        with self._thread_lock:
            self._ensure_open()
            pauses = self._state.get("credential_pauses") or {}
            due = tuple(sorted(
                alias for alias, row in pauses.items()
                if float((row or {}).get("resume_at") or 0.0) <= float(now)))
            if not due:
                return ()
            for alias in due:
                pauses.pop(alias, None)
            self._commit("credential_pause_cleared", {"credentials": list(due)})
            return due

    def set_queue_pause(self, *, paused: bool, note: str) -> None:
        """Stop or resume handing out queued cells. Running episodes are never interrupted.

        This is the breakpoint a human needs when a fault is noticed before any automatic cap
        fires: the batch stops spending, whatever is in flight still completes and is recorded,
        and resuming continues from exactly the same queue.
        """
        if not isinstance(note, str) or not note.strip():
            raise BatchStateError("queue pause note must be non-empty")
        with self._thread_lock:
            self._ensure_open()
            now = self._clock()
            self._state["queue_pause"] = (
                {"note": note.strip(), "paused_at": now} if paused else None)
            self._commit("queue_paused" if paused else "queue_resumed",
                         {"note": note.strip()})

    @property
    def queue_paused(self) -> bool:
        return bool((self._state or {}).get("queue_pause"))

    def request_stop(self, *, request_id: str, note: str) -> None:
        """Pause dispatch and durably cancel exactly the currently active leases."""
        with self._thread_lock:
            self._ensure_open()
            now = self._clock()
            request = {"request_id": request_id, "note": note, "requested_at": now}
            self._state["queue_pause"] = {"note": note, "paused_at": now, "stop_request_id": request_id}
            for lease in self._state["leases"].values():
                lease.setdefault("operator_stop", request)
            self._commit("operator_stop_requested", request)

    def lease_stop_request(self, lease_id: str) -> dict | None:
        with self._thread_lock:
            self._ensure_open()
            return copy.deepcopy(self._state["leases"].get(lease_id, {}).get("operator_stop"))

    def requeue_cell(self, identifier: str, *, note: str) -> None:
        if not isinstance(note, str) or not note.strip():
            raise BatchStateError("requeue note must be non-empty")
        with self._thread_lock:
            self._ensure_open()
            try:
                cell = self._state["cells"][identifier]
            except KeyError as exc:
                raise BatchStateError(f"unknown cell {identifier!r}") from exc
            previous = cell["status"]
            if previous != "retry_wait":
                raise BatchStateError(
                    f"cell {identifier} cannot be requeued from {previous}")
            cell.update({
                "status": "queued",
                "attention_id": None,
                "next_retry_at": None,
                "updated_at": self._clock(),
            })
            self._commit("cell_requeued", {
                "cell_id": identifier,
                "previous_status": previous,
                "attention_id": None,
                "note": note.strip(),
            })

    def resolve_attention_and_requeue(
        self,
        attention_id: str,
        *,
        note: str,
        control_request_id: str | None = None,
    ) -> str:
        """Resolve one open attention and requeue its exact logical episode atomically."""
        if not isinstance(note, str) or not note.strip():
            raise BatchStateError("attention resolution note must be non-empty")
        with self._thread_lock:
            self._ensure_open()
            try:
                attention = self._state["attentions"][attention_id]
            except KeyError as exc:
                raise BatchStateError(f"unknown attention {attention_id!r}") from exc
            if attention.get("status") != "open":
                raise BatchStateError(f"attention {attention_id} is not open")
            identifier = attention["cell_id"]
            cell = self._state["cells"][identifier]
            if cell.get("status") != "needs_attention" \
                    or cell.get("attention_id") != attention_id:
                raise BatchStateError(
                    f"attention {attention_id} does not own a needs_attention cell")
            now = self._clock()
            if control_request_id is not None:
                _safe_segment(control_request_id, "control_request_id")
            attention.update({
                "status": "resolved",
                "updated_at": now,
                "resolution": {
                    "resolved_at": now,
                    "note": note.strip(),
                    "action": "requeue",
                    "control_request_id": control_request_id,
                },
            })
            cell.update({
                "status": "queued",
                "attention_id": None,
                "next_retry_at": None,
                "updated_at": now,
            })
            self._commit("attention_resolved_and_requeued", {
                "attention_id": attention_id,
                "cell_id": identifier,
                "note": note.strip(),
                "control_request_id": control_request_id,
            })
            return identifier

    def resolve_attention_and_accept(
        self,
        attention_id: str,
        *,
        accepted_attempt: Mapping[str, Any],
        note: str,
        control_request_id: str | None = None,
    ) -> str:
        """Accept one sealed execution after an explicitly recorded operator decision."""
        if not isinstance(note, str) or not note.strip():
            raise BatchStateError("attention acceptance note must be non-empty")
        evidence = _validate_accepted_attempt_evidence(accepted_attempt)
        with self._thread_lock:
            self._ensure_open()
            try:
                attention = self._state["attentions"][attention_id]
            except KeyError as exc:
                raise BatchStateError(f"unknown attention {attention_id!r}") from exc
            if attention.get("status") != "open":
                raise BatchStateError(f"attention {attention_id} is not open")
            identifier = attention["cell_id"]
            cell = self._state["cells"][identifier]
            execution_id = attention["execution_id"]
            execution_key = f"{identifier}/{execution_id}"
            execution = self._state["executions"].get(execution_key)
            if cell.get("status") != "needs_attention" \
                    or cell.get("attention_id") != attention_id \
                    or not isinstance(execution, dict) \
                    or execution.get("status") != "needs_attention":
                raise BatchStateError(
                    f"attention {attention_id} does not own an accept-ready execution")
            now = self._clock()
            if control_request_id is not None:
                _safe_segment(control_request_id, "control_request_id")
            resolution = {
                "resolved_at": now,
                "note": note.strip(),
                "action": "accept",
                "control_request_id": control_request_id,
            }
            attention.update({
                "status": "resolved",
                "updated_at": now,
                "resolution": resolution,
            })
            detail = copy.deepcopy(dict(execution.get("detail") or {}))
            detail["accepted_attempt"] = evidence
            detail["manual_acceptance"] = {
                "attention_id": attention_id,
                "category": attention.get("category"),
                "note": note.strip(),
            }
            execution.update({"status": "accepted", "detail": detail})
            cell.update({
                "status": "accepted",
                "accepted_execution_id": execution_id,
                "attention_id": None,
                "next_retry_at": None,
                "retry_wait_count": 0,
                "updated_at": now,
            })
            self._commit("attention_resolved_and_accepted", {
                "attention_id": attention_id,
                "cell_id": identifier,
                "execution_id": execution_id,
                "note": note.strip(),
                "control_request_id": control_request_id,
            })
            return identifier

    def model_lanes(self, model: Any) -> int:
        """How many episodes of `model` may hold a lease at once (frozen with the batch).

        Defaults to 1 for a batch created before lanes were expressible, which is exactly what
        those batches enforced."""
        lanes = self.specification.get("model_lanes") or {}
        return int(lanes.get(model, 1))

    def open_attentions(self) -> list[dict[str, Any]]:
        with self._thread_lock:
            self._ensure_open()
            return [
                copy.deepcopy(attention)
                for attention in self._state["attentions"].values()
                if attention.get("status") == "open"
            ]

    def stage_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in sorted(CELL_STATUSES)}
        for identifier in self._state["requested_cells"]:
            counts[self._state["cells"][identifier]["status"]] += 1
        counts["total"] = len(self._state["requested_cells"])
        return counts

    def active_leases(self) -> list[dict[str, Any]]:
        with self._thread_lock:
            self._ensure_open()
            return [copy.deepcopy(value) for value in self._state["leases"].values()]

    def snapshot(self) -> dict[str, Any]:
        with self._thread_lock:
            self._ensure_open()
            return copy.deepcopy(self._state)

    @staticmethod
    def read_snapshot(batch_dir: Path) -> dict[str, Any]:
        return _read_object(Path(batch_dir).resolve() / "batch_state.json")

    @classmethod
    def _assert_invariants(
        cls, specification: Mapping[str, Any], state: Mapping[str, Any],
    ) -> None:
        if state.get("schema_version") != BATCH_STATE_SCHEMA_VERSION:
            raise BatchStateError("unsupported batch state schema")
        if state.get("batch_id") != specification.get("batch_id"):
            raise BatchStateError("batch state/spec IDs disagree")
        revision = state.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise BatchStateError("batch revision must be a positive integer")
        cells = state.get("cells")
        executions = state.get("executions")
        leases = state.get("leases")
        attentions = state.get("attentions")
        requested = state.get("requested_cells")
        if not isinstance(cells, dict) or not isinstance(executions, dict) \
                or not isinstance(leases, dict) or not isinstance(attentions, dict) \
                or not isinstance(requested, list):
            raise BatchStateError("batch state collections have invalid types")
        if len(requested) != len(set(requested)) or not set(requested) <= set(cells):
            raise BatchStateError("requested_cells must be unique and reference known cells")
        dispatch = state.get("dispatch_queue")
        if dispatch is not None and (
                not isinstance(dispatch, list) or len(dispatch) != len(set(dispatch))
                or set(dispatch) != set(requested)):
            raise BatchStateError(
                "dispatch_queue must be an exact permutation of requested_cells")

        accepted_execution_keys = set()
        for identifier, cell in cells.items():
            expected = cell_id(cell.get("model"), cell.get("task"), cell.get("attempt_index"))
            if identifier != expected:
                raise BatchStateError(f"cell key disagrees with identity: {identifier}")
            status = cell.get("status")
            if status not in CELL_STATUSES:
                raise BatchStateError(f"invalid status for {identifier}: {status}")
            lease_id = cell.get("lease_id")
            current_execution_id = cell.get("current_execution_id")
            if status == "running":
                if not lease_id or lease_id not in leases or not current_execution_id:
                    raise BatchStateError(f"running cell lacks active lease/execution: {identifier}")
            elif lease_id is not None or current_execution_id is not None:
                raise BatchStateError(f"non-running cell retains active lease/execution: {identifier}")
            accepted_execution_id = cell.get("accepted_execution_id")
            if status == "accepted":
                if not accepted_execution_id:
                    raise BatchStateError(f"accepted cell lacks accepted execution: {identifier}")
                execution_key = f"{identifier}/{accepted_execution_id}"
                if execution_key in accepted_execution_keys:
                    raise BatchStateError(f"accepted execution is reused: {execution_key}")
                accepted_execution_keys.add(execution_key)
                if executions.get(execution_key, {}).get("status") != "accepted":
                    raise BatchStateError(f"accepted execution record is invalid: {execution_key}")
            elif accepted_execution_id is not None:
                raise BatchStateError(f"non-accepted cell retains accepted execution: {identifier}")
            attention_id = cell.get("attention_id")
            if status == "needs_attention":
                attention = attentions.get(attention_id)
                if not attention or attention.get("status") != "open" \
                        or attention.get("cell_id") != identifier:
                    raise BatchStateError(f"needs_attention cell lacks open attention: {identifier}")
            elif attention_id is not None:
                raise BatchStateError(f"non-attention cell retains attention ID: {identifier}")

        active_gpus = set()
        active_models: dict[str, int] = {}
        declared_lanes = specification.get("model_lanes") or {}
        for lease_id, lease in leases.items():
            if lease.get("lease_id") != lease_id:
                raise BatchStateError(f"lease key disagrees with lease_id: {lease_id}")
            identifier = lease.get("cell_id")
            cell = cells.get(identifier)
            if not cell or cell.get("status") != "running" or cell.get("lease_id") != lease_id:
                raise BatchStateError(f"lease does not match a running cell: {lease_id}")
            execution_key = f"{identifier}/{lease.get('execution_id')}"
            execution = executions.get(execution_key)
            if not execution or execution.get("status") != "running" \
                    or execution.get("lease_id") != lease_id:
                raise BatchStateError(f"lease does not match a running execution: {lease_id}")
            launch_state = lease.get("launch_state")
            if launch_state not in {"reserved", "launching", "launched"}:
                raise BatchStateError(f"lease has invalid launch_state: {lease_id}")
            if launch_state == "reserved" and any(
                    lease.get(field) is not None
                    for field in (
                        "launch_protocol", "command_sha256", "target_command_sha256",
                        "process_start_id", "pid", "pgid")):
                raise BatchStateError(f"reserved lease has process identity: {lease_id}")
            launch_protocol = lease.get("launch_protocol")
            if launch_protocol not in {None, "pipe-gate-v1"}:
                raise BatchStateError(f"lease has unsupported launch protocol: {lease_id}")
            if launch_protocol == "pipe-gate-v1" \
                    and not lease.get("target_command_sha256"):
                raise BatchStateError(f"gated lease lacks target command identity: {lease_id}")
            if launch_state == "launched" and (
                    not lease.get("command_sha256") or lease.get("pid") is None
                    or lease.get("pgid") is None):
                raise BatchStateError(f"launched lease lacks process identity: {lease_id}")
            gpu = lease.get("gpu")
            model = cell.get("model")
            if gpu in active_gpus:
                raise BatchStateError(f"GPU has multiple active leases: {gpu}")
            allowed = declared_lanes.get(model, 1)
            if active_models.get(model, 0) + 1 > allowed:
                raise BatchStateError(f"model has multiple active leases: {model}")
            active_gpus.add(gpu)
            active_models[model] = active_models.get(model, 0) + 1

        for execution_key, execution in executions.items():
            identifier = execution.get("cell_id")
            expected_key = f"{identifier}/{execution.get('execution_id')}"
            if execution_key != expected_key or identifier not in cells:
                raise BatchStateError(f"execution key/identity is invalid: {execution_key}")
            if execution.get("status") not in {"running", *EXECUTION_OUTCOMES}:
                raise BatchStateError(f"execution status is invalid: {execution_key}")
            if execution.get("status") == "accepted":
                detail = execution.get("detail")
                if not isinstance(detail, Mapping) or "accepted_attempt" not in detail:
                    raise BatchStateError(
                        f"accepted execution lacks accepted_attempt evidence: {execution_key}")
                try:
                    _validate_accepted_attempt_evidence(detail["accepted_attempt"])
                except BatchStateError as exc:
                    raise BatchStateError(
                        f"accepted execution evidence is invalid: {execution_key}: {exc}") from exc

        for attention_id, attention in attentions.items():
            if attention.get("attention_id") != attention_id:
                raise BatchStateError(f"attention key disagrees with ID: {attention_id}")
            if attention.get("status") not in {"open", "resolved"}:
                raise BatchStateError(f"attention status is invalid: {attention_id}")
            identifier = attention.get("cell_id")
            if identifier not in cells:
                raise BatchStateError(f"attention references unknown cell: {attention_id}")
            if attention.get("status") == "open" and (
                    cells[identifier].get("status") != "needs_attention"
                    or cells[identifier].get("attention_id") != attention_id):
                raise BatchStateError(f"open attention is not linked from its cell: {attention_id}")

    def close(self) -> None:
        with self._thread_lock:
            if self._closed:
                return
            self._closed = True
            self._lock.release()

    def __enter__(self) -> "BatchStateStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
