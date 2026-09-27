"""Lock-free operator requests consumed by the single durable batch-state writer."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from codeaction.batch.attention import AttentionReporter
from codeaction.batch.results import (
    BatchResultsError,
    validate_execution_attempt,
    write_batch_result_views,
)
from codeaction.batch.state import BatchStateError, BatchStateStore
from codeaction.contracts.failures import bounded_detail


CONTROL_REQUEST_SCHEMA_VERSION = "1.1"
# 1.0 carried exactly one action and therefore one field set. 1.1 keeps that shape for
# `requeue_attention` and gives each new action the fields it actually needs, so a request can
# never carry a target that its action ignores. A 1.0 document stays valid: an operator's queued
# requeue must not be rejected because the controller was upgraded underneath it.
_ACCEPTED_REQUEST_SCHEMA_VERSIONS = frozenset({"1.0", "1.1"})
_BASE_REQUEST_FIELDS = frozenset({
    "schema_version", "request_id", "batch_id", "requested_at", "action", "note"})
_ACTION_FIELDS = {
    "requeue_attention": frozenset({"attention_id"}),
    "accept_attention": frozenset({"attention_id"}),
    "abandon_cell": frozenset({"cell_id"}),
    "prioritize_cells": frozenset({"cell_ids"}),
    "pause_queue": frozenset(),
    "resume_queue": frozenset(),
    "stop_queue": frozenset(),
}
CONTROL_RESULT_SCHEMA_VERSION = "1.0"
_ATTENTION_ID = re.compile(r"^A-\d{4,}$")
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_CELL_ID = re.compile(r"^[^/\s]+/[^/\s]+/attempt-\d{3,}$")


class ControlRequestError(RuntimeError):
    """An operator request is malformed, unsafe, or cannot be queued."""


@dataclass(frozen=True)
class ControlApplyReport:
    applied: tuple[str, ...]
    rejected: tuple[str, ...]
    already_applied: tuple[str, ...]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ControlRequestError(f"control document must not be a symlink: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlRequestError(
            f"cannot read valid control document {path.name}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ControlRequestError(f"control document must be an object: {path.name}")
    return value


def _safe_note(note: str) -> str:
    value = bounded_detail(note)
    if not value:
        raise ControlRequestError("requeue note must be non-empty")
    return value


def enqueue_attention_requeue(
    batch_dir: Path,
    attention_id: str,
    *,
    note: str,
    request_id: str | None = None,
    clock=_utc_now,
) -> dict[str, Any]:
    """Create one immutable request without taking the controller's state lock."""
    if not isinstance(attention_id, str) or not _ATTENTION_ID.fullmatch(attention_id):
        raise ControlRequestError("attention_id must use A-NNNN format")
    identifier = request_id or uuid.uuid4().hex
    if not isinstance(identifier, str) or not _REQUEST_ID.fullmatch(identifier):
        raise ControlRequestError("request_id must be 32 lowercase hexadecimal characters")
    root = Path(batch_dir).resolve()
    state = BatchStateStore.read_snapshot(root)
    attention = (state.get("attentions") or {}).get(attention_id)
    if not isinstance(attention, Mapping) or attention.get("status") != "open":
        raise ControlRequestError(f"attention is not open: {attention_id}")
    request = {
        "schema_version": CONTROL_REQUEST_SCHEMA_VERSION,
        "request_id": identifier,
        "batch_id": state.get("batch_id"),
        "requested_at": clock(),
        "action": "requeue_attention",
        "attention_id": attention_id,
        "note": _safe_note(note),
    }
    path = root / "control_requests" / f"{identifier}.json"
    if path.exists() or (root / "control_results" / f"{identifier}.json").exists():
        raise ControlRequestError(f"control request already exists: {identifier}")
    _atomic_json(path, request)
    return request


def enqueue_attention_accept(
    batch_dir: Path,
    attention_id: str,
    *,
    note: str,
    request_id: str | None = None,
    clock=_utc_now,
) -> dict[str, Any]:
    """Queue operator acceptance for the active controller, the only durable state writer."""
    if not isinstance(attention_id, str) or not _ATTENTION_ID.fullmatch(attention_id):
        raise ControlRequestError("attention_id must use A-NNNN format")
    identifier = request_id or uuid.uuid4().hex
    if not isinstance(identifier, str) or not _REQUEST_ID.fullmatch(identifier):
        raise ControlRequestError("request_id must be 32 lowercase hexadecimal characters")
    root = Path(batch_dir).resolve()
    state = BatchStateStore.read_snapshot(root)
    attention = (state.get("attentions") or {}).get(attention_id)
    if not isinstance(attention, Mapping) or attention.get("status") != "open":
        raise ControlRequestError(f"attention is not open: {attention_id}")
    if attention.get("category") != "harness_attestation_failed":
        raise ControlRequestError(
            "only harness_attestation_failed may be manually accepted")
    request = {
        "schema_version": CONTROL_REQUEST_SCHEMA_VERSION,
        "request_id": identifier,
        "batch_id": state.get("batch_id"),
        "requested_at": clock(),
        "action": "accept_attention",
        "attention_id": attention_id,
        "note": _safe_note(note),
    }
    path = root / "control_requests" / f"{identifier}.json"
    if path.exists() or (root / "control_results" / f"{identifier}.json").exists():
        raise ControlRequestError(f"control request already exists: {identifier}")
    _atomic_json(path, request)
    return request


def enqueue_queue_control(
    batch_dir: Path,
    action: str,
    *,
    note: str,
    cell_id: str | None = None,
    cell_ids: list[str] | tuple[str, ...] | None = None,
    request_id: str | None = None,
    clock=_utc_now,
) -> dict[str, Any]:
    """Queue a pause, resume, or abandon without taking the controller's state lock.

    Written as a file for the same reason the requeue is: the single durable writer is the
    controller, and an operator must be able to act while it holds the lock.
    """
    if action not in {"abandon_cell", "prioritize_cells", "pause_queue", "resume_queue", "stop_queue"}:
        raise ControlRequestError(f"unsupported control action: {action}")
    identifier = request_id or uuid.uuid4().hex
    if not _REQUEST_ID.fullmatch(str(identifier)):
        raise ControlRequestError("request_id must be 32 lowercase hexadecimal characters")
    root = Path(batch_dir).resolve()
    state = BatchStateStore.read_snapshot(root)
    request: dict[str, Any] = {
        "schema_version": CONTROL_REQUEST_SCHEMA_VERSION,
        "request_id": identifier,
        "batch_id": state.get("batch_id"),
        "requested_at": clock(),
        "action": action,
        "note": _safe_note(note),
    }
    if action == "abandon_cell":
        if cell_id not in (state.get("cells") or {}):
            raise ControlRequestError(f"unknown cell: {cell_id}")
        request["cell_id"] = cell_id
    elif action == "prioritize_cells":
        ordered = list(cell_ids or ())
        if not ordered or len(ordered) != len(set(ordered)) \
                or any(identifier not in (state.get("cells") or {}) for identifier in ordered):
            raise ControlRequestError(
                "priority cells must be a non-empty unique list of requested cells")
        request["cell_ids"] = ordered
    elif cell_id is not None:
        raise ControlRequestError(f"{action} takes no cell")
    path = root / "control_requests" / f"{identifier}.json"
    if path.exists() or (root / "control_results" / f"{identifier}.json").exists():
        raise ControlRequestError(f"control request already exists: {identifier}")
    _atomic_json(path, request)
    return request


def _validate_request(request: Mapping[str, Any], *, batch_id: str) -> None:
    action = request.get("action")
    if action not in _ACTION_FIELDS:
        raise ControlRequestError("unsupported control action")
    if set(request) != _BASE_REQUEST_FIELDS | _ACTION_FIELDS[action]:
        raise ControlRequestError("control request has unknown or missing fields")
    if request.get("schema_version") not in _ACCEPTED_REQUEST_SCHEMA_VERSIONS:
        raise ControlRequestError("unsupported control request schema")
    if request.get("batch_id") != batch_id:
        raise ControlRequestError("control request targets a different batch")
    if not _REQUEST_ID.fullmatch(str(request.get("request_id") or "")):
        raise ControlRequestError("invalid control request ID")
    if action in {"requeue_attention", "accept_attention"} \
            and not _ATTENTION_ID.fullmatch(str(request.get("attention_id") or "")):
        raise ControlRequestError("invalid attention ID")
    if action == "abandon_cell" and not _CELL_ID.fullmatch(str(request.get("cell_id") or "")):
        raise ControlRequestError("invalid cell ID")
    if action == "prioritize_cells":
        identifiers = request.get("cell_ids")
        if not isinstance(identifiers, list) or not identifiers \
                or len(identifiers) != len(set(identifiers)) \
                or any(not _CELL_ID.fullmatch(str(identifier)) for identifier in identifiers):
            raise ControlRequestError("invalid priority cell list")
    if _safe_note(request.get("note")) != request.get("note"):
        raise ControlRequestError("control request note is not normalized or safe")


def _result_document(
    request: Mapping[str, Any],
    *,
    status: str,
    observed_at: str,
    cell_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": CONTROL_RESULT_SCHEMA_VERSION,
        "request_id": request.get("request_id"),
        "batch_id": request.get("batch_id"),
        "observed_at": observed_at,
        "action": request.get("action"),
        "attention_id": request.get("attention_id"),
        "target_cell_id": request.get("cell_id"),
        "status": status,
        "cell_id": cell_id,
    }
    if error:
        result["error"] = str(error)[:500]
    return result


def apply_control_requests(
    store: BatchStateStore,
    reporter: AttentionReporter,
    *,
    clock=_utc_now,
    attention_accepter=None,
) -> ControlApplyReport:
    """Apply each durable request once; recover the state-commit/result-write crash window."""
    request_dir = store.batch_dir / "control_requests"
    result_dir = store.batch_dir / "control_results"
    if not request_dir.exists():
        return ControlApplyReport((), (), ())
    applied: list[str] = []
    rejected: list[str] = []
    already: list[str] = []
    for path in sorted(request_dir.glob("*.json")):
        request: Mapping[str, Any] | None = None
        result_path = result_dir / path.name
        if result_path.exists():
            already.append(path.stem)
            continue
        try:
            request = _read_object(path)
            _validate_request(request, batch_id=store.specification["batch_id"])
            request_id = request["request_id"]
            if path.stem != request_id:
                raise ControlRequestError("control filename disagrees with request_id")
            snapshot = store.snapshot()
            action = request["action"]
            # Every branch decides "already applied" from durable state, never from the result
            # file alone, so the crash window between committing state and writing the result
            # recovers into the same answer.
            if action == "requeue_attention":
                attention = snapshot["attentions"].get(request["attention_id"])
                resolution = (attention.get("resolution")
                              if isinstance(attention, Mapping) else None)
                settled = (isinstance(resolution, Mapping)
                           and resolution.get("control_request_id") == request_id)
                cell_id = attention.get("cell_id") if isinstance(attention, Mapping) else None
                if not settled:
                    cell_id = reporter.resolve_and_requeue(
                        request["attention_id"],
                        note=request["note"],
                        control_request_id=request_id,
                    )
            elif action == "accept_attention":
                attention = snapshot["attentions"].get(request["attention_id"])
                resolution = (attention.get("resolution")
                              if isinstance(attention, Mapping) else None)
                settled = (isinstance(resolution, Mapping)
                           and resolution.get("control_request_id") == request_id)
                cell_id = attention.get("cell_id") if isinstance(attention, Mapping) else None
                if not settled:
                    if attention_accepter is not None:
                        cell_id = attention_accepter(
                            store, reporter, request["attention_id"], request["note"], request_id)
                    else:
                        if not isinstance(attention, Mapping):
                            raise ControlRequestError("unknown attention")
                        run_dir = Path(str(attention.get("location") or "")).parent
                        cell = snapshot["cells"][attention["cell_id"]]
                        evidence = validate_execution_attempt(
                            store.batch_dir,
                            store.specification,
                            cell,
                            run_dir,
                            allow_manual_attestation_acceptance=True,
                        )
                        cell_id = store.resolve_attention_and_accept(
                            request["attention_id"], accepted_attempt=evidence,
                            note=request["note"], control_request_id=request_id)
                        reporter.sync_outputs()
                        write_batch_result_views(
                            store.batch_dir, store.specification, store.snapshot())
            elif action == "abandon_cell":
                cell_id = request["cell_id"]
                settled = snapshot["cells"][cell_id]["status"] == "abandoned"
                if not settled:
                    store.abandon_cell(cell_id, note=request["note"])
            elif action == "prioritize_cells":
                requested_order = list(request["cell_ids"])
                current_queue = list(snapshot.get("dispatch_queue") or snapshot["requested_cells"])
                settled = current_queue[:len(requested_order)] == requested_order
                if not settled:
                    store.prioritize_cells(requested_order, note=request["note"])
                cell_id = None
            elif action == "stop_queue":
                cell_id = None
                settled = (snapshot.get("queue_pause") or {}).get("stop_request_id") == request_id
                if not settled:
                    store.request_stop(request_id=request_id, note=request["note"])
            else:
                cell_id = None
                paused = action == "pause_queue"
                settled = bool(snapshot.get("queue_pause")) == paused
                if not settled:
                    store.set_queue_pause(paused=paused, note=request["note"])
            if settled:
                status = "already_applied"
                already.append(request_id)
            else:
                status = "applied"
                applied.append(request_id)
            result = _result_document(
                request, status=status, observed_at=clock(), cell_id=cell_id)
        except (BatchResultsError, BatchStateError, ControlRequestError, KeyError, TypeError) as exc:
            if not isinstance(request, Mapping):
                request = {
                    "request_id": path.stem,
                    "batch_id": store.specification["batch_id"],
                    "action": None,
                    "attention_id": None,
                    "cell_id": None,
                }
            rejected.append(str(request.get("request_id") or path.stem))
            result = _result_document(
                request,
                status="rejected",
                observed_at=clock(),
                error=f"{type(exc).__name__}: {exc}",
            )
        _atomic_json(result_path, result)
    return ControlApplyReport(tuple(applied), tuple(rejected), tuple(already))


# --------------------------------------------------------------------------------------
# Operator command line
#
# The three enqueue functions above have existed since the control plane was written, and until
# now nothing called them but their own test: an operator could not list an attention, let alone
# resolve one. Everything here is deliberately thin. Reads go through `BatchStateStore.
# read_snapshot`, which takes no lock, and writes only drop a request file for the controller --
# the single durable writer -- to apply. Both properties are what make this safe to run against a
# batch that is mid-episode.
# --------------------------------------------------------------------------------------

_SCOPE_EFFECT = {
    "cell": "holds only this run; the same agent's other tasks keep going",
    "model": "holds every queued run of this agent until you resolve it",
    "credential": "holds every queued run on this account until you resolve it",
}


def _load_state(batch_dir: Path) -> dict[str, Any]:
    try:
        return BatchStateStore.read_snapshot(batch_dir)
    except (BatchStateError, OSError) as exc:
        raise ControlRequestError(f"cannot read batch state in {batch_dir}: {exc}") from exc


def _cell_label(state: Mapping[str, Any], cell_id: Any) -> str:
    cell = (state.get("cells") or {}).get(cell_id) or {}
    if not cell:
        return str(cell_id)
    attempt = cell.get("attempt_index")
    attempt_text = f"{attempt:03d}" if isinstance(attempt, int) else str(attempt)
    return (f"{cell.get('model')}/{cell.get('task')} attempt {attempt_text} "
            f"seed {cell.get('scene_seed')}")


def _render_attention(state: Mapping[str, Any], attention: Mapping[str, Any],
                      *, verbose: bool) -> list[str]:
    scope = attention.get("scope") or {}
    kind = str(scope.get("kind"))
    category = str(attention.get("category"))
    acceptable = category == "harness_attestation_failed"
    lines = [
        f"{attention.get('attention_id')}  category={category}",
        f"    scope   {kind}:{scope.get('id')}  -- {_SCOPE_EFFECT.get(kind, 'unknown scope')}",
        f"    cell    {_cell_label(state, attention.get('cell_id'))}",
        f"    opened  {attention.get('opened_at')}   execution={attention.get('execution_id')}",
        f"    actions requeue" + ("" if acceptable
                                  else "   (accept is refused: only harness_attestation_failed "
                                       "may be accepted by hand)"),
    ]
    summary = attention.get("summary") or attention.get("note")
    if summary:
        lines.append(f"    summary {str(summary)[:400]}")
    if verbose:
        lines.append("    " + json.dumps(attention, indent=2, sort_keys=True).replace(
            "\n", "\n    "))
    return lines


def _cmd_list(args) -> int:
    state = _load_state(args.batch_dir)
    counts: dict[str, int] = {}
    for identifier in state.get("requested_cells") or []:
        status = ((state.get("cells") or {}).get(identifier) or {}).get("status", "?")
        counts[status] = counts.get(status, 0) + 1
    print(f"batch {state.get('batch_id')}  dir={Path(args.batch_dir).resolve()}")
    print("queue: " + " | ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f" | total {len(state.get('requested_cells') or [])}")
    pause = state.get("queue_pause")
    print(f"paused: {pause.get('note') if pause else 'no'}")
    leases = list((state.get("leases") or {}).values())
    if leases:
        print("running:")
        for lease in sorted(leases, key=lambda value: value.get("gpu", -1)):
            print(f"  gpu {lease.get('gpu')}  {_cell_label(state, lease.get('cell_id'))}"
                  f"  lease={lease.get('lease_id')} execution={lease.get('execution_id')}")
    attentions = [value for value in (state.get("attentions") or {}).values()
                  if value.get("status") == "open"]
    if not attentions:
        print("open attentions: none")
        return 0
    print(f"open attentions: {len(attentions)}")
    for attention in sorted(attentions, key=lambda value: str(value.get("attention_id"))):
        for line in _render_attention(state, attention, verbose=False):
            print("  " + line)
    return 0


def _cmd_show(args) -> int:
    state = _load_state(args.batch_dir)
    attention = (state.get("attentions") or {}).get(args.attention_id)
    if not isinstance(attention, Mapping):
        raise ControlRequestError(f"unknown attention {args.attention_id}")
    for line in _render_attention(state, attention, verbose=True):
        print(line)
    return 0


def _cmd_status(args) -> int:
    root = Path(args.batch_dir).resolve()
    result = root / "control_results" / f"{args.request_id}.json"
    pending = root / "control_requests" / f"{args.request_id}.json"
    if result.exists():
        print(json.dumps(_read_object(result), indent=2, sort_keys=True))
        return 0
    if pending.exists():
        print(f"pending: the controller has not applied {args.request_id} yet")
        return 0
    raise ControlRequestError(f"unknown control request {args.request_id}")


def _report(request: Mapping[str, Any]) -> int:
    print(f"queued {request['action']} request {request['request_id']}")
    print("the controller applies it on its next poll; check with: "
          f"control ... status {request['request_id']}")
    return 0


def build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m codeaction.batch.control",
        description="Inspect a running batch and resolve the attentions it raises.")
    parser.add_argument("batch_dir", type=Path, help="the batch directory (holds batch_state.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="queue counts, running leases and open attentions")
    show = sub.add_parser("show", help="one attention in full")
    show.add_argument("attention_id")
    status = sub.add_parser("status", help="whether the controller applied a request")
    status.add_argument("request_id")

    for name, help_text in (
            ("requeue", "resolve an attention and re-run its episode from scratch"),
            ("accept", "resolve an attention by accepting its sealed execution as the result")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("attention_id")
        command.add_argument("--note", required=True,
                             help="why -- recorded immutably next to the decision")
    abandon = sub.add_parser("abandon", help="give up on one cell: not re-run, not accepted")
    abandon.add_argument("cell_id")
    abandon.add_argument("--note", required=True)
    prioritize = sub.add_parser("prioritize", help="move cells to the front of the dispatch queue")
    prioritize.add_argument("cell_ids", nargs="+")
    prioritize.add_argument("--note", required=True)
    for name, help_text in (("pause", "stop dispatching NEW episodes; running ones finish"),
                            ("stop", "pause dispatch and interrupt active attempts; retain all executions"),
                            ("resume", "undo a pause")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--note", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "show":
            return _cmd_show(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "requeue":
            return _report(enqueue_attention_requeue(
                args.batch_dir, args.attention_id, note=args.note))
        if args.command == "accept":
            return _report(enqueue_attention_accept(
                args.batch_dir, args.attention_id, note=args.note))
        if args.command == "abandon":
            return _report(enqueue_queue_control(
                args.batch_dir, "abandon_cell", cell_id=args.cell_id, note=args.note))
        if args.command == "prioritize":
            return _report(enqueue_queue_control(
                args.batch_dir, "prioritize_cells", cell_ids=args.cell_ids, note=args.note))
        if args.command in ("pause", "resume", "stop"):
            return _report(enqueue_queue_control(
                args.batch_dir, f"{args.command}_queue", note=args.note))
    except (ControlRequestError, BatchStateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
