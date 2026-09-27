"""Human-attention records derived from and atomically linked to durable batch state."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping, TextIO

from codeaction.batch.state import BatchStateError, BatchStateStore


ATTENTION_SCHEMA_VERSION = "1.0"
_SENSITIVE_KEYS = frozenset({
    "authorization", "api_key", "access_token", "refresh_token", "password", "secret",
})


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
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


def _bounded_text(value: Any, field: str, *, limit: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatchStateError(f"attention {field} must be a non-empty string")
    return value.strip()[:limit]


def _validate_safe(value: Any, path: str = "attention") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SENSITIVE_KEYS:
                raise BatchStateError(f"sensitive field is forbidden in {path}: {key}")
            _validate_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_safe(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise BatchStateError(f"attention value is not JSON-safe at {path}")


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_attention(record: Mapping[str, Any], *, stream: TextIO, bell: bool = True) -> None:
    """Render one concise, actionable alert; ring only for an attached terminal."""
    if bell and getattr(stream, "isatty", lambda: False)():
        stream.write("\a")
    scope = record.get("scope") or {}
    lines = [
        f"[ATTENTION {record.get('attention_id', '?')}]",
        f"task={record.get('task')} model={record.get('model')} "
        f"attempt={record.get('attempt_index')} seed={record.get('scene_seed')}",
        f"execution={record.get('execution_id')} category={record.get('category')}",
        f"scope={scope.get('kind')}:{scope.get('id')}",
        f"reason={record.get('reason')}",
        f"location={record.get('location')}",
        f"next={record.get('suggested_action')}",
    ]
    stream.write("\n".join(lines) + "\n")
    stream.flush()


class AttentionReporter:
    """Create/resolve attention in state, then refresh its human-facing projections."""

    def __init__(self, store: BatchStateStore, *, stream: TextIO | None = None) -> None:
        self.store = store
        self.stream = stream if stream is not None else sys.stderr

    def open_for_lease(
        self,
        lease_id: str,
        *,
        category: str,
        reason: str,
        scope: Mapping[str, str],
        location: str,
        suggested_action: str,
        evidence: Mapping[str, Any] | None = None,
        execution_detail: Mapping[str, Any] | None = None,
    ) -> str:
        snapshot = self.store.snapshot()
        lease = snapshot["leases"].get(lease_id)
        if lease is None:
            raise BatchStateError(f"unknown active lease {lease_id!r}")
        cell = snapshot["cells"][lease["cell_id"]]
        scope_value = dict(scope)
        if set(scope_value) != {"kind", "id"}:
            raise BatchStateError("attention scope must contain exactly kind and id")
        if scope_value["kind"] not in {"cell", "model", "credential"}:
            raise BatchStateError("attention scope kind must be cell, model, or credential")
        if scope_value["kind"] == "cell" and scope_value["id"] != lease["cell_id"]:
            raise BatchStateError("cell-scoped attention must name the running cell")
        if scope_value["kind"] == "model" and scope_value["id"] != cell["model"]:
            raise BatchStateError("model-scoped attention must name the running model")
        record = {
            "category": _bounded_text(category, "category", limit=128),
            "reason": _bounded_text(reason, "reason"),
            "scope": {
                "kind": scope_value["kind"],
                "id": _bounded_text(scope_value["id"], "scope.id", limit=256),
            },
            "location": _bounded_text(location, "location"),
            "suggested_action": _bounded_text(suggested_action, "suggested_action"),
            "evidence": dict(evidence or {}),
            "model": cell["model"],
            "task": cell["task"],
            "attempt_index": cell["attempt_index"],
            "scene_seed": cell["scene_seed"],
        }
        record["fingerprint"] = _fingerprint({
            "cell_id": lease["cell_id"],
            "category": record["category"],
            "scope": record["scope"],
        })
        _validate_safe(record)
        attention_id = self.store.finish_with_attention(
            lease_id, attention=record, detail=execution_detail)
        self.sync_outputs()
        created = next(
            item for item in self.store.open_attentions()
            if item["attention_id"] == attention_id)
        render_attention(created, stream=self.stream)
        return attention_id

    def resolve_and_requeue(
        self,
        attention_id: str,
        *,
        note: str,
        control_request_id: str | None = None,
    ) -> str:
        identifier = self.store.resolve_attention_and_requeue(
            attention_id, note=note, control_request_id=control_request_id)
        self.sync_outputs()
        self.stream.write(f"[RESOLVED {attention_id}] requeued {identifier}\n")
        self.stream.flush()
        return identifier

    def sync_outputs(self) -> None:
        snapshot = self.store.snapshot()
        all_records = sorted(
            snapshot["attentions"].values(), key=lambda record: record["attention_id"])
        open_records = [record for record in all_records if record["status"] == "open"]
        open_document = {
            "schema_version": ATTENTION_SCHEMA_VERSION,
            "batch_id": snapshot["batch_id"],
            "batch_revision": snapshot["revision"],
            "open_count": len(open_records),
            "attentions": open_records,
        }
        _atomic_text(
            self.store.batch_dir / "attention_open.json",
            json.dumps(open_document, indent=2, ensure_ascii=False) + "\n",
        )
        history = "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in all_records
        )
        _atomic_text(self.store.batch_dir / "attention.jsonl", history)
        lines = ["# Attention", "", f"Open: {len(open_records)} · Total: {len(all_records)}", ""]
        if not open_records:
            lines.append("No open attention.")
        for record in open_records:
            scope = record["scope"]
            lines.extend([
                f"## {record['attention_id']} · {record['category']}",
                "",
                f"- Episode: `{record['model']}/{record['task']}/attempt-"
                f"{record['attempt_index']:03d}` (seed {record['scene_seed']})",
                f"- Execution: `{record['execution_id']}`",
                f"- Scope: `{scope['kind']}:{scope['id']}`",
                f"- Reason: {record['reason']}",
                f"- Location: `{record['location']}`",
                f"- Next action: {record['suggested_action']}",
                "",
            ])
        _atomic_text(self.store.batch_dir / "ATTENTION.md", "\n".join(lines).rstrip() + "\n")
