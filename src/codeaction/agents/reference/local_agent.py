"""Local Python/command agents use the same isolated MCP client and episode lifecycle."""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
import subprocess
import threading
import time

from codeaction.extensions import declarations, entrypoint
from codeaction.contracts.failures import FailureCode, default_failure


def run_local_agent(name, runtime, work: Path, scaffold: dict, emit, *, wall_budget_s: float):
    scaffold = {"config_sha256": scaffold.get("config_sha256"), "implementation": name}
    start = time.monotonic()
    calls = []
    from codeaction.contracts.version import TRANSCRIPT_SCHEMA_VERSION
    emit({"event": "meta", "schema_version": TRANSCRIPT_SCHEMA_VERSION, "agent_implementation": name, "scaffold": scaffold})

    class Client:
        tools = runtime.tools

        def call(self, tool: str, arguments: dict):
            event = runtime.dispatch(tool, arguments)
            record = {"event": "done" if tool == "done" else "tool", "tool": tool,
                      "args": arguments, "result": event.payload, "step": event.step}
            calls.append(record)
            emit(record)
            # Inline image bytes are returned by the MCP transport, never simulator paths.
            return {"result": event.payload, "images": [
                {"mime_type": "image/png", "base64": __import__("base64").b64encode(Path(path).read_bytes()).decode()}
                for path in event.projection.model_image_refs]}

    client = Client()
    context = {"task": runtime.task_text, "output_dir": str(work), "tools": runtime.tools,
               "config": declarations("agent")[name].get("config", {})}
    factory = entrypoint("agent", name)
    if factory is None:
        raise ValueError(f"unknown local agent: {name}")
    # The provider image is isolated; this function never runs in the host controller.
    if declarations("agent")[name].get("mode", "python") == "command":
        argv = factory(context)
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("command agent factory must return an argv list")
        with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=work,
                              start_new_session=True) as process:
            timed_out = threading.Event()
            def terminate():
                timed_out.set()
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            timer = threading.Timer(wall_budget_s, terminate)
            timer.start()
            try:
                process.stdin.write(json.dumps(context) + "\n")
                process.stdin.flush()
                for line in process.stdout:
                    request = json.loads(line)
                    response = client.call(request["tool"], request["arguments"])
                    process.stdin.write(json.dumps(response) + "\n")
                    process.stdin.flush()
                code = process.wait()
                if timed_out.is_set():
                    runtime.finalize("wall_budget", failure=default_failure(FailureCode.WALL_BUDGET_EXHAUSTED))
                elif code != 0:
                    raise RuntimeError("command agent exited unsuccessfully")
            finally:
                timer.cancel()
    else:
        def timeout(_signal, _frame):
            raise TimeoutError("local agent exceeded its wall budget")
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, wall_budget_s)
        try:
            factory(context, client)
        except TimeoutError:
            runtime.finalize("wall_budget", failure=default_failure(FailureCode.WALL_BUDGET_EXHAUSTED))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    if not runtime._finalized:
        runtime.finalize("no_done")
    stats = {"event": "end", "status": runtime.status, "failure": runtime.failure.to_dict() if runtime.failure else None,
             "budget_used": runtime.steps, "total_calls": runtime.total_calls,
             "tool_calls_used": runtime.steps, "tool_call_budget": runtime.max_tool_calls,
             "total_tool_dispatches": len(calls), "model_turns": None, "turns": None,
             "usage": {"prompt_tokens": None, "completion_tokens": None},
             "wall_s": time.monotonic() - start, "scaffold": scaffold,
             "diagnostics_unavailable": ["token_usage", "reasoning_summary", "model_turns"]}
    emit(stats)
    return stats
