"""Hard-deadline subprocess supervision for one durable batch lease."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from codeaction.batch.state import BatchStateError, BatchStateStore
from codeaction.batch.launch_gate import LAUNCH_PROTOCOL


class SupervisorError(RuntimeError):
    """The execution could not be launched or supervised safely."""


@dataclass(frozen=True)
class ProcessResult:
    return_code: int | None
    timed_out: bool
    terminated: bool
    killed: bool
    wall_s: float
    cleanup_return_code: int | None
    cleanup_timed_out: bool
    error: str | None
    operator_stop: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _command_sha256(argv: Sequence[str]) -> str:
    payload = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gate_command(descriptor: int, command: Sequence[str]) -> list[str]:
    gate = Path(__file__).with_name("launch_gate.py").resolve()
    return [sys.executable, str(gate), "--fd", str(descriptor), "--", *command]


def _linux_process_start_id(pid: int) -> str | None:
    """Linux PID-reuse guard; unavailable platforms deliberately return unknown."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = stat_text.rfind(")")
        if closing < 0:
            return None
        fields = stat_text[closing + 2:].split()
    except (OSError, UnicodeError):
        return None
    return fields[19] if len(fields) > 19 else None


def _terminate_group(
    process: subprocess.Popen,
    *,
    term_grace_s: float,
    kill_grace_s: float,
) -> tuple[bool, bool]:
    if process.poll() is not None:
        return False, False
    terminated = True
    killed = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return terminated, killed
    try:
        process.wait(timeout=term_grace_s)
        return terminated, killed
    except subprocess.TimeoutExpired:
        killed = True
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=kill_grace_s)
    except subprocess.TimeoutExpired as exc:
        raise SupervisorError(
            f"process group {process.pid} survived SIGKILL for {kill_grace_s}s") from exc
    return terminated, killed


class ProcessSupervisor:
    """Run an owned process group while heartbeating its durable GPU lease."""

    def __init__(
        self,
        store: BatchStateStore,
        *,
        heartbeat_interval_s: float = 10.0,
        term_grace_s: float = 15.0,
        kill_grace_s: float = 5.0,
        monotonic=time.monotonic,
    ) -> None:
        for value, field in (
                (heartbeat_interval_s, "heartbeat_interval_s"),
                (term_grace_s, "term_grace_s"),
                (kill_grace_s, "kill_grace_s")):
            if value <= 0:
                raise ValueError(f"{field} must be positive")
        self.store = store
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.term_grace_s = float(term_grace_s)
        self.kill_grace_s = float(kill_grace_s)
        self.monotonic = monotonic

    def _stop_agent_containers(self, project: str, log) -> bool:
        """Stop the selected project's agents first so the controller can collect evidence."""
        result = subprocess.run([
            "docker", "ps", "--filter", f"label=com.docker.compose.project={project}",
            "--format", '{{.ID}} {{.Label "com.docker.compose.service"}}'],
            capture_output=True, text=True, check=True, timeout=10)
        agents = {"reference-agent", "fixture-agent", "agent", "codex-agent", "claude-agent"}
        ids = [line.split()[0] for line in result.stdout.splitlines()
               if len(line.split()) == 2 and line.split()[1] in agents]
        if ids:
            subprocess.run(["docker", "stop", "--time", "5", *ids],
                           stdout=log, stderr=log, check=True, timeout=15)
        return bool(ids)

    def run(
        self,
        lease_id: str,
        command: Sequence[str],
        *,
        log_path: Path,
        cwd: Path,
        env: Mapping[str, str],
        hard_timeout_s: float,
        run_dir: Path,
        compose_project: str | None = None,
        cleanup_command: Sequence[str] | None = None,
        cleanup_timeout_s: float = 60.0,
    ) -> ProcessResult:
        if hard_timeout_s <= 0:
            raise ValueError("hard_timeout_s must be positive")
        if cleanup_timeout_s <= 0:
            raise ValueError("cleanup_timeout_s must be positive")
        argv = [str(value) for value in command]
        if not argv:
            raise ValueError("command must be non-empty")
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = self.monotonic()
        process = None
        timed_out = False
        terminated = False
        killed = False
        error = None
        cleanup_return_code = None
        cleanup_timed_out = False
        reraised: BaseException | None = None
        operator_stop = self.store.lease_stop_request(lease_id)
        if operator_stop:
            return ProcessResult(None, False, False, False, 0.0, None, False, None, operator_stop)
        last_heartbeat = started
        with log_path.open("ab") as log:
            read_fd = None
            write_fd = None
            try:
                self.store.update_lease(
                    lease_id,
                    launch_state="launching",
                    launch_protocol=LAUNCH_PROTOCOL,
                    target_command_sha256=_command_sha256(argv),
                    run_dir=str(Path(run_dir).resolve()),
                    compose_project=compose_project,
                    heartbeat_at=_utc_now(),
                )
                read_fd, write_fd = os.pipe()
                gated_argv = _gate_command(read_fd, argv)
                process = subprocess.Popen(
                    gated_argv,
                    cwd=Path(cwd),
                    env=dict(env),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(read_fd,),
                )
                os.close(read_fd)
                read_fd = None
                self.store.update_lease(
                    lease_id,
                    launch_state="launched",
                    command_sha256=_command_sha256(gated_argv),
                    pid=process.pid,
                    pgid=process.pid,
                    process_start_id=_linux_process_start_id(process.pid),
                    heartbeat_at=_utc_now(),
                )
                os.write(write_fd, b"1")
                os.close(write_fd)
                write_fd = None
                while True:
                    operator_stop = self.store.lease_stop_request(lease_id)
                    if operator_stop:
                        if compose_project:
                            try:
                                if self._stop_agent_containers(compose_project, log):
                                    process.wait(timeout=30)
                            except (subprocess.SubprocessError, OSError):
                                pass
                        if process.poll() is None:
                            terminated, killed = _terminate_group(
                                process, term_grace_s=self.term_grace_s, kill_grace_s=self.kill_grace_s)
                        break
                    elapsed = self.monotonic() - started
                    remaining = hard_timeout_s - elapsed
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        process.wait(timeout=min(self.heartbeat_interval_s, remaining, 1.0))
                        break
                    except subprocess.TimeoutExpired:
                        if self.monotonic() - last_heartbeat >= self.heartbeat_interval_s:
                            self.store.update_lease(lease_id, heartbeat_at=_utc_now())
                            last_heartbeat = self.monotonic()
                if timed_out:
                    terminated, killed = _terminate_group(
                        process,
                        term_grace_s=self.term_grace_s,
                        kill_grace_s=self.kill_grace_s,
                    )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, Exception):
                    reraised = exc
                if process is not None and process.poll() is None:
                    try:
                        terminated, killed = _terminate_group(
                            process,
                            term_grace_s=self.term_grace_s,
                            kill_grace_s=self.kill_grace_s,
                        )
                    except BaseException as terminate_exc:
                        error += f"; cleanup {type(terminate_exc).__name__}: {terminate_exc}"
            finally:
                for descriptor in (read_fd, write_fd):
                    if descriptor is not None:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
            nonzero_exit = process is not None and process.returncode not in (None, 0)
            if (operator_stop or timed_out or error is not None or nonzero_exit) and cleanup_command:
                cleanup_return_code, cleanup_timed_out, cleanup_error = self._run_cleanup(
                    cleanup_command,
                    log=log,
                    cwd=Path(cwd),
                    env=env,
                    timeout_s=cleanup_timeout_s,
                )
                if cleanup_error:
                    error = f"{error}; {cleanup_error}" if error else cleanup_error
        if reraised is not None:
            raise reraised
        return ProcessResult(
            return_code=process.returncode if process is not None else None,
            timed_out=timed_out,
            terminated=terminated,
            killed=killed,
            wall_s=round(self.monotonic() - started, 6),
            cleanup_return_code=cleanup_return_code,
            cleanup_timed_out=cleanup_timed_out,
            error=error,
            operator_stop=operator_stop,
        )

    def _run_cleanup(
        self,
        command: Sequence[str],
        *,
        log,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: float,
    ) -> tuple[int | None, bool, str | None]:
        argv = [str(value) for value in command]
        if not argv:
            return None, False, "cleanup command is empty"
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=dict(env),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return None, False, f"cleanup launch {type(exc).__name__}: {exc}"
        try:
            return process.wait(timeout=timeout_s), False, None
        except subprocess.TimeoutExpired:
            try:
                _terminate_group(
                    process,
                    term_grace_s=min(self.term_grace_s, timeout_s),
                    kill_grace_s=self.kill_grace_s,
                )
            except SupervisorError as exc:
                return process.returncode, True, str(exc)
            return process.returncode, True, None
