"""MCP transport runtime for the isolated ``codeaction-reference`` container.

The model loop remains in :mod:`codeaction.agents.reference.reference_agent`.  This module only maps its sequential
runtime calls to one MCP connection, validates the complete server surface, and converts MCP image
blocks into temporary native-image files consumed by that same loop.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from codeaction.runtime.argcheck import ArgumentContractError, validate_arguments
from codeaction.runtime.episode import RuntimeCall
from codeaction.contracts.failures import (
    FailureCode,
    FailureOrigin,
    default_failure,
    failure_from_legacy_status,
    is_recoverable_action_abort,
)
from codeaction.contracts.identity import sha256_json
from codeaction.runtime.mcp_control import (
    MALFORMED_ARGUMENTS_MARKER,
    REFERENCE_CANCELLED_TOOL,
    REFERENCE_FINALIZE_TOOL,
    REFERENCE_MALFORMED_TOOL,
    REFERENCE_READY_TOOL,
    REFERENCE_WALL_CREDIT_TOOL,
)
from codeaction.interface.schemas import PROGRAM_TOOL_NAMES
from codeaction.contracts.tool_results import ImageGroup, ToolProjection, payload_projection


REFERENCE_MCP_CLIENT_VERSION = "1.0.0"
DEFAULT_CONNECT_TIMEOUT_S = 180.0
DEFAULT_CALL_TIMEOUT_S = 610.0
MAX_LINE_BYTES = 20 * 1024 * 1024


class McpErrorResult(Exception):
    """A ``tools/call`` result flagged ``isError``: the server refused or the handler raised.

    Kept distinct from a transport failure because the two need opposite verdicts. The MCP SDK
    validates arguments against the declared ``inputSchema`` before our handler runs and returns
    ``isError: true`` with a plain-text message on failure (mcp/server/lowlevel/server.py,
    ``_make_error_result``). Parsing that text as the tool payload raised ``JSONDecodeError``, which
    the dispatcher recorded as ``origin=environment, scoreable=false`` -- so a model's own
    out-of-range argument ended the attempt as an environment fault and scored nothing.
    """

    def __init__(self, text: str):
        super().__init__(text or "MCP error result carried no text")
        self.text = str(text or "")

    @property
    def is_input_validation(self) -> bool:
        """Whether the server rejected the ARGUMENTS rather than failing while executing."""
        return self.text.startswith(MCP_INPUT_VALIDATION_PREFIX)


# The MCP SDK's own literal. Used only as a corroborating signal: the authority is re-running the
# declared argument schema locally, because that schema is the same one the server published.
MCP_INPUT_VALIDATION_PREFIX = "Input validation error:"
# Bound on vendor/server text copied into a model-visible payload or a recorded detail.
MCP_ERROR_TEXT_MAX_CHARS = 400


def _first_text(contents: list) -> str:
    for item in contents:
        if isinstance(item, Mapping) and item.get("type") == "text" \
                and isinstance(item.get("text"), str):
            return item["text"]
    return ""


def openai_tools_from_mcp(raw_tools: list[dict]) -> list[dict]:
    """Convert the exact MCP ``tools/list`` order into native function-tool definitions."""
    out = []
    for index, item in enumerate(raw_tools):
        if not isinstance(item, Mapping):
            raise ValueError(f"MCP tool {index} is not an object")
        name = item.get("name")
        description = item.get("description")
        input_schema = item.get("inputSchema")
        if not isinstance(name, str) or not name:
            raise ValueError(f"MCP tool {index} has no name")
        if not isinstance(description, str) or not isinstance(input_schema, Mapping):
            raise ValueError(f"MCP tool {name!r} has an invalid schema")
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": dict(input_schema),
            },
        })
    return out


class McpJsonRpcClient:
    """Minimal newline-delimited MCP client; no SDK or ambient config."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ):
        deadline = time.monotonic() + float(connect_timeout_s)
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.socket = socket.create_connection(
                    (str(host), int(port)), timeout=min(2.0, connect_timeout_s))
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.1)
        else:
            raise OSError(f"reference MCP connection failed: {type(last_error).__name__}")
        self.socket.settimeout(float(call_timeout_s))
        self.stream = self.socket.makefile("rwb")
        self.next_id = 1
        self._lock = threading.Lock()

    def call(self, method: str, params=None) -> dict:
        with self._lock:
            request_id = self.next_id
            self.next_id += 1
            request = {"jsonrpc": "2.0", "id": request_id, "method": str(method)}
            if params is not None:
                request["params"] = params
            self.stream.write((
                json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8"))
            self.stream.flush()
            while True:
                raw = self.stream.readline(MAX_LINE_BYTES + 1)
                if len(raw) > MAX_LINE_BYTES:
                    raise ValueError("MCP response exceeds reference-client line limit")
                if not raw:
                    raise EOFError("MCP connection closed before response")
                response = json.loads(raw)
                if not isinstance(response, dict) or response.get("id") != request_id:
                    continue
                if "error" in response:
                    raise RuntimeError(f"MCP method {method!r} failed")
                result = response.get("result")
                if not isinstance(result, dict):
                    raise ValueError(f"MCP method {method!r} returned a non-object")
                return result

    def notify(self, method: str, params=None) -> None:
        with self._lock:
            request = {"jsonrpc": "2.0", "method": str(method)}
            if params is not None:
                request["params"] = params
            self.stream.write((
                json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8"))
            self.stream.flush()

    def tool(self, name: str, arguments=None) -> dict:
        return self.call(
            "tools/call",
            {"name": str(name), "arguments": arguments if arguments is not None else {}},
        )

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            self.socket.close()


class RemoteEpisodeRuntime:
    """EpisodeRuntime-shaped adapter over the reference MCP profile."""

    def __init__(
        self,
        task_text: str,
        *,
        expected_surface: Mapping[str, Any],
        max_tool_calls: int,
        wall_budget_s: float,
        image_dir: str | Path,
        physical_time_budget_s: float = 900.0,
        host: str | None = None,
        port: int | None = None,
        client: McpJsonRpcClient | None = None,
    ):
        self.task_text = str(task_text)
        self.max_tool_calls = int(max_tool_calls)
        self.max_steps = self.max_tool_calls
        self.wall_budget_s = float(wall_budget_s)
        self.physical_time_budget_s = float(physical_time_budget_s)
        self.steps = 0
        self.total_calls = 0
        self.status = "running"
        self.done_report = None
        self.failure = None
        self.finalize_control_error = None
        self._finalized = False
        self._image_index = 0
        self._image_dir = Path(image_dir)
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or McpJsonRpcClient(
            host or os.environ.get("CODEACTION_SIM_HOST", "gateway"),
            int(port or os.environ.get("CODEACTION_SIM_PORT", "8765")),
        )
        self.client.call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "codeaction-reference",
                    "version": REFERENCE_MCP_CLIENT_VERSION,
                },
            },
        )
        self.client.notify("notifications/initialized")
        listed = self.client.call("tools/list").get("tools")
        if not isinstance(listed, list):
            raise ValueError("MCP tools/list did not return a tool list")
        self.tools = openai_tools_from_mcp(listed)
        names = [
            str((item.get("function") or {}).get("name") or "")
            for item in self.tools
        ]
        if names != list(expected_surface.get("ordered_names") or []):
            raise ValueError("reference MCP ordered tool surface mismatch")
        if sha256_json(self.tools) != expected_surface.get("delivered_sha256"):
            raise ValueError("reference MCP delivered tool hash mismatch")
        composition_tools = {"run_code", *PROGRAM_TOOL_NAMES}
        self.registry = {
            name: None for name in names
            if name != "done" and name not in composition_tools
        }
        # The control is deliberately absent from tools/list and queues behind scene boot.
        ready = self.client.tool(REFERENCE_READY_TOOL, {})
        if ready.get("isError"):
            raise RuntimeError("reference MCP scene readiness failed")
        self._t0 = time.monotonic()
        self._t0_true = self._t0
        self.wall_credit_s = 0.0

    @property
    def budget_used(self) -> int:
        return self.steps

    @property
    def over(self) -> bool:
        return self.status != "running"

    @property
    def wall_s(self) -> float:
        """True elapsed time, credits included: what a stopwatch on the attempt would read."""
        return round(time.monotonic() - self._t0_true, 1)

    def check_wall_budget(self) -> bool:
        if self.over:
            return False
        if time.monotonic() - self._t0 <= self.wall_budget_s:
            return True
        self.finalize("wall_budget")
        return False

    def credit_wall_budget(self, seconds: float) -> float:
        """Return provider-outage time to BOTH wall clocks that can end this attempt.

        A reference-scaffold attempt is bounded twice over: here, before each model turn, and in the sim
        host's own `EpisodeRuntime` on each tool dispatch. Crediting only this side would just move the failure --
        the agent would keep going and the host would end the attempt on the next tool call, with
        `wall_budget` attributed to the model exactly as before. So the credit is applied locally
        and mirrored to the host over a transport-only control that the model never sees.
        """
        amount = float(seconds)
        if not amount > 0.0 or self.over:
            return 0.0
        headroom = float(self.wall_budget_s) - self.wall_credit_s
        applied = min(amount, max(headroom, 0.0))
        if applied <= 0.0:
            return 0.0
        try:
            self.client.tool(REFERENCE_WALL_CREDIT_TOOL, {"seconds": applied})
        except Exception:
            # Best effort by design: a credit that does not reach the host must not end the
            # attempt. The two clocks then disagree, and the host's stricter one wins -- the same
            # behaviour as before this control existed.
            return 0.0
        self.wall_credit_s = round(self.wall_credit_s + applied, 3)
        self._t0 += applied
        return applied

    def _image_projection(self, payload: dict, contents: list[dict]) -> ToolProjection:
        refs = []
        for content in contents:
            if not isinstance(content, Mapping) or content.get("type") != "image":
                continue
            if content.get("mimeType") != "image/png" or not isinstance(content.get("data"), str):
                raise ValueError("reference MCP supports only base64 image/png blocks")
            self._image_index += 1
            path = self._image_dir / f"mcp-image-{self._image_index:06d}.png"
            path.write_bytes(base64.b64decode(content["data"], validate=True))
            refs.append(str(path))
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        groups = (
            (ImageGroup("image block(s) returned by the preceding MCP tool:", tuple(refs)),)
            if refs else ()
        )
        return ToolProjection(
            payload=dict(payload),
            full_payload=dict(payload),
            image_groups=groups,
            model_image_refs=tuple(refs),
            all_image_refs=tuple(refs),
            truncated=False,
            original_bytes=len(encoded),
            model_bytes=len(encoded),
        )

    def _decode_tool_result(self, result: Mapping[str, Any]) -> ToolProjection:
        contents = result.get("content")
        if not isinstance(contents, list):
            raise ValueError("MCP tools/call result has no content list")
        # An `isError` result is a declared protocol outcome carrying PLAIN TEXT, not JSON, and not
        # a broken transport. Parsing it as a payload raised JSONDecodeError and killed the episode
        # as `origin=environment` -- see McpErrorResult.
        if result.get("isError"):
            raise McpErrorResult(_first_text(contents))
        texts = [
            item.get("text") for item in contents
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        if len(texts) != 1 or not isinstance(texts[0], str):
            raise ValueError("reference MCP tool result must contain exactly one JSON text block")
        payload = json.loads(texts[0])
        if not isinstance(payload, dict):
            raise ValueError("reference MCP model-visible payload must be an object")
        return self._image_projection(payload, contents)

    def dispatch(self, name, args, *, malformed_arguments=False) -> RuntimeCall:
        name = str(name)
        self.total_calls += 1
        if self.over:
            return RuntimeCall(
                name, args,
                payload_projection({"error": f"EPISODE_OVER ({self.status})"}),
                self.steps, False, kind="over")
        if name != "done" and self.steps >= self.max_tool_calls:
            self.finalize("budget_exhausted")
            return RuntimeCall(
                name, args,
                payload_projection({
                    "error": "EPISODE_OVER (budget_exhausted): "
                             "the tool-call budget is exhausted."
                }),
                self.steps, False, kind="limit", failure=self.failure)

        wire_name = REFERENCE_MALFORMED_TOOL if malformed_arguments else name
        wire_args = (
            {"tool": name}
            if malformed_arguments else (args if isinstance(args, dict) else {})
        )
        stage = "call"
        try:
            raw_result = self.client.tool(wire_name, wire_args)
            stage = "decode"
            projection = self._decode_tool_result(raw_result)
        except McpErrorResult as exc:
            return self._error_result_call(name, args, wire_name, wire_args, exc)
        except Exception as exc:
            failure = default_failure(
                FailureCode.TOOL_RUNTIME_ERROR,
                # Naming the stage matters: the framing parse in `McpJsonRpcClient.call` and the
                # payload parse in `_decode_tool_result` both raise JSONDecodeError, and the old
                # detail could not tell them apart without a container round-trip.
                detail_safe=f"reference MCP transport ({stage}): {type(exc).__name__}",
            )
            # If the call reached the server but its response was malformed, the hidden control can
            # still preserve the harness-origin failure in the authoritative sim-side result.
            self.finalize("episode_fatal", failure=failure)
            return RuntimeCall(
                name, args, payload_projection({"error": "episode aborted"}),
                self.steps, False, kind="fatal", failure=failure,
                error_detail=f"{name}: reference MCP transport failed")

        error = str(projection.payload.get("error") or "")
        if error.startswith("EPISODE_OVER"):
            if "unintended_collision" in error:
                status = "unintended_collision"
                kind = "terminal"
                self.steps += 1
                charged = True
            elif "physical_time_budget_exhausted" in error:
                status = "physical_time_budget_exhausted"
                kind = "terminal"
                self.steps += 1
                charged = True
            else:
                status = "wall_budget" if "wall_budget" in error else "budget_exhausted"
                kind = "limit"
                charged = False
            self.status = status
            self.failure = failure_from_legacy_status(status, tested_origin="model")
            self._finalized = True
            return RuntimeCall(
                name, args, projection, self.steps, charged,
                kind=kind, failure=self.failure)

        if name == "done":
            if error:
                self.steps += 1
                failure = default_failure(FailureCode.INVALID_TOOL_ARGUMENTS)
                return RuntimeCall(
                    name, args, projection, self.steps, True, failure=failure)
            self.done_report = dict(args)
            self.status = "done"
            self._finalized = True
            return RuntimeCall(name, args, projection, self.steps, False, kind="done")

        self.steps += 1
        failure = (
            default_failure(FailureCode.INVALID_TOOL_ARGUMENTS)
            if malformed_arguments else None
        )
        kind = ("recoverable_abort"
                if is_recoverable_action_abort(projection.full_payload) else "tool")
        return RuntimeCall(
            name, args, projection, self.steps, True, kind=kind, failure=failure)

    def cancel_after_recoverable_abort(self, name, args) -> RuntimeCall:
        """Mirror local runtime accounting without sending the cancelled call to the host."""
        name = str(name)
        mirrored = self.client.tool(REFERENCE_CANCELLED_TOOL, {"tool": name})
        if mirrored.get("isError"):
            raise RuntimeError("reference MCP cancelled-call accounting failed")
        self.total_calls += 1
        charged = name != "done" and self.steps < self.max_tool_calls
        if charged:
            self.steps += 1
        return RuntimeCall(
            name,
            args,
            payload_projection({
                # Same wording as the local runtime: all aborted actions use one barrier.
                "error": (
                    "not executed: an earlier call in this assistant turn returned a "
                    "structured action abort; inspect that result and replan in the next "
                    "turn"),
                "executed": False,
                "cancelled_by": "action_abort",
            }),
            self.steps,
            charged,
            kind="cancelled",
        )

    def _error_result_call(self, name, args, wire_name, wire_args,
                           exc: McpErrorResult) -> RuntimeCall:
        """Split an ``isError`` result into a model argument error and a harness fault.

        Attribution is decided by re-running the DECLARED argument schema locally rather than by
        trusting the server's wording: the server published that same schema, so agreement is
        evidence the model's arguments were the problem, and it survives a change in vendor text.
        An argument rejection is recoverable and scoreable, exactly like the sim-side
        ``INVALID_TOOL_ARGUMENTS`` path -- the model is charged, told what was wrong, and continues.
        Anything else means a handler raised server-side: the sim's state is no longer trustworthy,
        so the episode still ends, but as a HARNESS fault that nobody may score.
        """
        text = exc.text[:MCP_ERROR_TEXT_MAX_CHARS]
        argument_error = None
        if not isinstance(wire_args, Mapping):
            argument_error = "arguments were not an object"
        else:
            try:
                validate_arguments(wire_name, wire_args)
            except ArgumentContractError as contract_exc:
                argument_error = str(contract_exc)
        if argument_error is not None or exc.is_input_validation:
            self.steps += 1
            return RuntimeCall(
                name, args,
                payload_projection({"error": f"bad arguments: {text}"}),
                self.steps, True, kind="tool",
                failure=default_failure(
                    FailureCode.INVALID_TOOL_ARGUMENTS,
                    detail_safe=f"mcp_error_result(input_validation): {text}"),
            )
        failure = default_failure(
            FailureCode.TOOL_RUNTIME_ERROR,
            origin=FailureOrigin.HARNESS,
            scoreable=False,
            detail_safe=f"mcp_error_result(server_side): {text}",
        )
        self.finalize("episode_fatal", failure=failure)
        return RuntimeCall(
            name, args, payload_projection({"error": "episode aborted"}),
            self.steps, False, kind="fatal", failure=failure,
            error_detail=f"{name}: MCP server returned an error result")

    def finalize(self, status, *, failure=None, context=None) -> bool:
        del context
        if self._finalized:
            return False
        payload = {
            "status": str(status),
            "failure": failure.to_dict() if failure is not None else None,
            "total_calls": self.total_calls,
            "budget_used": self.steps,
        }
        try:
            result = self.client.tool(REFERENCE_FINALIZE_TOOL, payload)
            if result.get("isError"):
                raise RuntimeError("reference finalize control failed")
        except Exception as exc:
            # Always leave a trace. The common terminal statuses (endpoint_failure,
            # output_length_exceeded, a classified budget_exhausted) all arrive WITH a failure, so
            # the old `if failure is None` guard dropped the fact that the sim never learned the
            # verdict. Downstream that surfaced only as "reference/sim termination status
            # mismatch", which reads as an episode defect rather than the harness transport fault
            # it is. The recorded reason is what the attestation names.
            self.finalize_control_error = (
                f"{type(exc).__name__}: {exc}"[:MCP_ERROR_TEXT_MAX_CHARS])
            if failure is None:
                failure = default_failure(
                    FailureCode.HARNESS_CONTRACT_VIOLATION,
                    detail_safe="reference finalize control transport failed",
                )
        self.status = str(status)
        self.failure = failure or failure_from_legacy_status(self.status, tested_origin="model")
        self._finalized = True
        return True

    def close(self) -> None:
        self.client.close()
