"""Fail-closed recovery of stale reference-scaffold process and compose leases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from codeaction.batch.attention import AttentionReporter
from codeaction.batch.launch_gate import LAUNCH_PROTOCOL
from codeaction.batch.state import BatchStateError, BatchStateStore


_PROJECT = re.compile(r"^codeaction-[a-z0-9][a-z0-9-]*-a\d{3}$")
_DOCKER_ID = re.compile(r"^[0-9a-f]{12,64}$")
_VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


class RecoveryError(RuntimeError):
    """A stale lease cannot be inspected or cleaned safely."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_id: str
    command_sha256: str


@dataclass(frozen=True)
class ReconciliationReport:
    requeued_cells: tuple[str, ...]
    attention_ids: tuple[str, ...]
    accepted_cells: tuple[str, ...] = ()


def _command_sha256(argv: Sequence[str]) -> str:
    payload = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_linux_process(pid: int) -> ProcessIdentity | None:
    """Return the identity needed to prove PID ownership, or None when the PID is gone."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        raise RecoveryError("process PID must be a positive integer")
    root = Path(f"/proc/{pid}")
    if not root.exists():
        return None
    try:
        stat_text = (root / "stat").read_text(encoding="utf-8")
        closing = stat_text.rfind(")")
        if closing < 0:
            raise ValueError("missing comm terminator")
        tail = stat_text[closing + 2:].split()
        start_id = tail[19]
        raw_argv = (root / "cmdline").read_bytes()
        argv = [
            item.decode("utf-8", errors="surrogateescape")
            for item in raw_argv.split(b"\0") if item
        ]
        if not argv:
            raise ValueError("empty command line")
        pgid = os.getpgid(pid)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, IndexError, ValueError) as exc:
        raise RecoveryError(f"cannot inspect process {pid}: {type(exc).__name__}") from exc
    return ProcessIdentity(
        pid=pid,
        pgid=pgid,
        start_id=start_id,
        command_sha256=_command_sha256(argv),
    )


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RecoveryError(f"cannot inspect process group {pgid}: permission denied") from exc


def terminate_owned_process_group(
    lease: Mapping[str, Any],
    *,
    inspector: Callable[[int], ProcessIdentity | None] = inspect_linux_process,
    term_grace_s: float = 15.0,
    kill_grace_s: float = 5.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, dict[str, Any]]:
    """Terminate only when all persisted identity fields still match the live process."""
    pid = lease.get("pid")
    pgid = lease.get("pgid")
    if pid is None:
        return False, {"status": "identity_incomplete", "field": "pid"}
    try:
        observed = inspector(int(pid))
    except (TypeError, ValueError, RecoveryError) as exc:
        return False, {"status": "inspection_failed", "error": type(exc).__name__}
    if observed is None:
        return True, {"status": "not_running"}
    expected = {
        "pgid": pgid,
        "start_id": lease.get("process_start_id"),
        "command_sha256": lease.get("command_sha256"),
    }
    actual = {
        "pgid": observed.pgid,
        "start_id": observed.start_id,
        "command_sha256": observed.command_sha256,
    }
    if not expected["start_id"] or expected != actual:
        return False, {
            "status": "identity_mismatch",
            "pid": int(pid),
            "pgid_matches": expected["pgid"] == actual["pgid"],
            "start_id_matches": expected["start_id"] == actual["start_id"],
            "command_matches": expected["command_sha256"] == actual["command_sha256"],
        }
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except ProcessLookupError:
        return True, {"status": "not_running"}
    except (OSError, TypeError, ValueError) as exc:
        return False, {"status": "term_failed", "error": type(exc).__name__}
    deadline = monotonic() + term_grace_s
    while monotonic() < deadline:
        if not _group_exists(int(pgid)):
            return True, {"status": "terminated", "signal": "SIGTERM"}
        sleep(min(0.1, max(0.0, deadline - monotonic())))
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except ProcessLookupError:
        return True, {"status": "terminated", "signal": "SIGTERM"}
    except (OSError, TypeError, ValueError) as exc:
        return False, {"status": "kill_failed", "error": type(exc).__name__}
    deadline = monotonic() + kill_grace_s
    while monotonic() < deadline:
        if not _group_exists(int(pgid)):
            return True, {"status": "terminated", "signal": "SIGKILL"}
        sleep(min(0.1, max(0.0, deadline - monotonic())))
    return False, {"status": "process_group_survived_sigkill", "pgid": int(pgid)}


def _run_docker(
    argv: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess],
    timeout_s: float,
) -> subprocess.CompletedProcess:
    try:
        return runner(
            list(argv), check=False, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RecoveryError(f"Docker cleanup command failed: {type(exc).__name__}") from exc


def _listed_resources(
    project: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess],
    timeout_s: float,
) -> dict[str, list[str]]:
    queries = {
        "container": ["docker", "container", "ls", "--all", "--quiet"],
        "network": ["docker", "network", "ls", "--quiet"],
        "volume": ["docker", "volume", "ls", "--quiet"],
    }
    resources: dict[str, list[str]] = {}
    label = f"label=com.docker.compose.project={project}"
    for kind, command in queries.items():
        completed = _run_docker(
            [*command, "--filter", label], runner=runner, timeout_s=timeout_s)
        if completed.returncode != 0:
            raise RecoveryError(f"cannot list {kind} resources for exact compose project")
        values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        pattern = _VOLUME_NAME if kind == "volume" else _DOCKER_ID
        if any(not pattern.fullmatch(value) for value in values):
            raise RecoveryError(f"Docker returned an unsafe {kind} identifier")
        resources[kind] = values
    return resources


def cleanup_compose_project(
    project: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout_s: float = 60.0,
) -> tuple[bool, dict[str, Any]]:
    """Remove only resources bearing one validated exact Compose project label."""
    if not isinstance(project, str) or not _PROJECT.fullmatch(project):
        return False, {"status": "unsafe_project_name"}
    try:
        before = _listed_resources(project, runner=runner, timeout_s=timeout_s)
        removals = {
            "container": ["docker", "container", "rm", "--force"],
            "network": ["docker", "network", "rm"],
            "volume": ["docker", "volume", "rm"],
        }
        for kind in ("container", "network", "volume"):
            identifiers = before[kind]
            if not identifiers:
                continue
            completed = _run_docker(
                [*removals[kind], *identifiers], runner=runner, timeout_s=timeout_s)
            if completed.returncode != 0:
                return False, {
                    "status": "remove_failed", "resource_kind": kind,
                    "return_code": completed.returncode,
                }
        after = _listed_resources(project, runner=runner, timeout_s=timeout_s)
    except RecoveryError as exc:
        return False, {"status": "docker_error", "error": str(exc)}
    remaining = {kind: len(values) for kind, values in after.items() if values}
    if remaining:
        return False, {"status": "resources_remain", "remaining": remaining}
    return True, {
        "status": "clean",
        "removed": {kind: len(values) for kind, values in before.items()},
    }


def reconcile_active_leases(
    store: BatchStateStore,
    reporter: AttentionReporter,
    *,
    terminate: Callable[[Mapping[str, Any]], tuple[bool, Mapping[str, Any]]] = (
        terminate_owned_process_group),
    cleanup: Callable[[str], tuple[bool, Mapping[str, Any]]] = cleanup_compose_project,
    settle_preserved: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None,
    ] | None = None,
) -> ReconciliationReport:
    """Resolve every lease inherited from a dead controller before scheduling new work."""
    requeued: list[str] = []
    attentions: list[str] = []
    accepted: list[str] = []
    for lease in store.active_leases():
        lease_id = lease["lease_id"]
        cell_id = lease["cell_id"]
        launch_state = lease.get("launch_state")
        never_authorized = launch_state == "reserved" or (
            launch_state == "launching" and lease.get("launch_protocol") == LAUNCH_PROTOCOL
            and lease.get("pid") is None and lease.get("pgid") is None
            and lease.get("command_sha256") is None)
        if lease.get("operator_stop") and never_authorized:
            attention_id = reporter.open_for_lease(
                lease_id, category="operator_stopped", reason="operator stopped this attempt before launch",
                scope={"kind": "cell", "id": cell_id}, location=str(store.batch_dir),
                suggested_action="requeue explicitly when ready",
                evidence={"stop_request": dict(lease["operator_stop"])})
            attentions.append(attention_id)
            continue
        if launch_state == "reserved":
            store.finish_execution(
                lease_id,
                outcome="retry_wait",
                detail={"recovery": {"status": "reserved_lease_requeued"}},
                next_retry_at=None,
            )
            requeued.append(cell_id)
            continue
        if launch_state == "launching" \
                and lease.get("launch_protocol") == LAUNCH_PROTOCOL \
                and lease.get("pid") is None and lease.get("pgid") is None \
                and lease.get("command_sha256") is None:
            store.finish_execution(
                lease_id,
                outcome="retry_wait",
                detail={"recovery": {
                    "status": "gated_process_was_never_authorized",
                    "launch_protocol": LAUNCH_PROTOCOL,
                }},
                next_retry_at=None,
            )
            requeued.append(cell_id)
            continue
        if launch_state != "launched":
            attention_id = reporter.open_for_lease(
                lease_id,
                category="recovery_launch_uncertain",
                reason="controller stopped while child launch identity was incomplete",
                scope={"kind": "cell", "id": cell_id},
                location=str(lease.get("run_dir") or store.batch_dir),
                suggested_action="verify no process or compose project remains, then requeue",
                evidence={"launch_state": launch_state},
                execution_detail={"recovery": {"status": "launch_uncertain"}},
            )
            attentions.append(attention_id)
            continue

        process_ok, process_detail = terminate(lease)
        project = lease.get("compose_project")
        if not process_ok or not isinstance(project, str):
            cleanup_ok, cleanup_detail = (False, {"status": "project_missing"})
        else:
            cleanup_ok, cleanup_detail = cleanup(project)
        recovery_detail = {
            "process": dict(process_detail),
            "compose": dict(cleanup_detail),
        }
        if process_ok and cleanup_ok:
            if settle_preserved is not None:
                cell = store.snapshot()["cells"].get(cell_id)
                try:
                    if not isinstance(cell, Mapping):
                        raise RecoveryError("stale lease has no durable cell")
                    settled = settle_preserved(lease, cell, recovery_detail)
                    if settled is not None:
                        if settled not in {"accepted", "retry_wait", "needs_attention"}:
                            raise RecoveryError(
                                f"preserved settlement returned unsupported status {settled!r}")
                        snapshot = store.snapshot()
                        current = snapshot["cells"].get(cell_id, {})
                        if current.get("status") != settled or any(
                                active.get("lease_id") == lease_id
                                for active in store.active_leases()):
                            raise RecoveryError(
                                "preserved settlement did not durably finish the stale lease")
                        if settled == "accepted":
                            accepted.append(cell_id)
                        elif settled == "retry_wait":
                            requeued.append(cell_id)
                        else:
                            attention_id = current.get("attention_id")
                            if not isinstance(attention_id, str) or not attention_id:
                                raise RecoveryError(
                                    "preserved attention settlement lacks an attention ID")
                            attentions.append(attention_id)
                        continue
                except Exception as exc:
                    if not any(
                            active.get("lease_id") == lease_id
                            for active in store.active_leases()):
                        raise RecoveryError(
                            "preserved settlement failed after releasing its lease") from exc
                    attention_id = reporter.open_for_lease(
                        lease_id,
                        category="recovery_preserved_artifact_invalid",
                        reason=(
                            "preserved reference-scaffold evidence could not be settled safely after "
                            "cleanup"),
                        scope={"kind": "cell", "id": cell_id},
                        location=str(lease.get("run_dir") or store.batch_dir),
                        suggested_action=(
                            "inspect the preserved sealed artifact before requeueing this "
                            "episode"),
                        evidence={"exception_type": type(exc).__name__},
                        execution_detail={"recovery": recovery_detail},
                    )
                    attentions.append(attention_id)
                    continue
            store.finish_execution(
                lease_id,
                outcome="retry_wait",
                detail={"recovery": recovery_detail},
                next_retry_at=None,
            )
            requeued.append(cell_id)
            continue
        attention_id = reporter.open_for_lease(
            lease_id,
            category="recovery_cleanup_uncertain",
            reason="stale reference-scaffold process or compose cleanup could not be proven",
            scope={"kind": "cell", "id": cell_id},
            location=str(lease.get("run_dir") or store.batch_dir),
            suggested_action="inspect the recorded PID/PGID and compose project before requeue",
            evidence={
                "process_status": process_detail.get("status"),
                "compose_status": cleanup_detail.get("status"),
            },
            execution_detail={"recovery": recovery_detail},
        )
        attentions.append(attention_id)
    return ReconciliationReport(
        requeued_cells=tuple(requeued),
        attention_ids=tuple(attentions),
        accepted_cells=tuple(accepted),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean one exact stale codeaction Compose project.")
    parser.add_argument("command", choices=("cleanup-project",))
    parser.add_argument("--project", required=True)
    args = parser.parse_args(argv)
    clean, detail = cleanup_compose_project(args.project)
    print(json.dumps(detail, sort_keys=True))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
