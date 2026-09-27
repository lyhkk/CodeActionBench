#!/usr/bin/env python3
"""Normalize a vendor-agent process exit into bounded ``agent_exit.json`` evidence.

Classification uses structured stream fields, the CLI return code, the MCP server result, and an
optional container-status JSON object. Codex 0.154.0 omits an error code for usage exhaustion;
its exact terminal-error prefix is recognized separately. Vendor prose is never persisted here.
"""
import argparse
import json
import sys
from pathlib import Path

_AP = Path(__file__).resolve().parent.parent
if str(_AP) not in sys.path:
    sys.path.insert(0, str(_AP))

from codeaction.contracts.failures import classify_agent_exit  # noqa: E402


AGENT_EXIT_SCHEMA_VERSION = "1.0"


def _read_json(path):
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_stream(path):
    source = sys.stdin if str(path) == "-" else open(path, "r", encoding="utf-8")
    last_result = {}
    event_count = 0
    invalid_lines = 0
    try:
        for line in source:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(event, dict):
                continue
            event_count += 1
            if event.get("type") == "result":
                last_result = event
            elif event.get('type') in ('turn.failed', 'thread.failed'):
                error = event.get('error') if isinstance(event.get('error'), dict) else {}
                code = error.get('code') or error.get('type')
                # Only a CLI terminal event counts. Model text and tool results cannot pause
                # an account by mentioning a limit. Unknown messages retain crash semantics.
                quota_message = not code and str(error.get('message') or '').startswith(
                    "You've hit your usage limit.")
                last_result = {
                    'subtype': 'usage_limit_reached' if quota_message else 'error_during_execution',
                    'api_error_status': code, 'is_error': True,
                    'error_source': 'codex_terminal_message' if quota_message else 'codex_terminal_code',
                }
    finally:
        if source is not sys.stdin:
            source.close()
    return last_result, event_count, invalid_lines


def normalize(
    stream_path,
    cli_exit_code,
    *,
    server_result=None,
    container_status=None,
    timed_out=False,
    transport_lost=False,
    controller_killed=False,
) -> dict:
    last_result, event_count, invalid_lines = _read_stream(stream_path)
    result = _read_json(server_result)
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    server_status = result.get("status") or stats.get("status")
    container = _read_json(container_status)
    evidence = {
        "cli_exit_code": int(cli_exit_code),
        "api_error_status": last_result.get("api_error_status"),
        "client_subtype": last_result.get("subtype"),
        "error_source": last_result.get("error_source"),
        "server_status": server_status,
        "timed_out": bool(timed_out),
        "transport_lost": bool(transport_lost),
        "controller_killed": bool(controller_killed),
        "container": {
            key: container[key]
            for key in ("exit_code", "oom_killed", "killed", "transport_lost")
            if key in container
        },
    }
    failure = classify_agent_exit(evidence)
    return {
        "schema_version": AGENT_EXIT_SCHEMA_VERSION,
        **evidence,
        "stream": {
            "source": "stdin" if str(stream_path) == "-" else Path(stream_path).name,
            "event_count": event_count,
            "invalid_json_lines": invalid_lines,
            "result_event_seen": bool(last_result),
            "result_subtype": last_result.get("subtype"),
            "is_error": last_result.get("is_error"),
            "vendor_result_text_retained": False,
        },
        "failure": failure.to_dict() if failure is not None else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-json", required=True,
                    help="vendor stream-json path, or '-' to read it from stdin")
    ap.add_argument("--cli-exit-code", required=True, type=int)
    ap.add_argument("--server-result", default=None,
                    help="attempt result.json written by the MCP episode server")
    ap.add_argument("--container-status", default=None,
                    help="optional structured container/controller status JSON")
    ap.add_argument("--timed-out", action="store_true")
    ap.add_argument("--transport-lost", action="store_true")
    ap.add_argument("--controller-killed", action="store_true")
    ap.add_argument("--output", default="-", help="agent_exit.json path, or '-' for stdout")
    args = ap.parse_args(argv)
    out = normalize(
        args.stream_json,
        args.cli_exit_code,
        server_result=args.server_result,
        container_status=args.container_status,
        timed_out=args.timed_out,
        transport_lost=args.transport_lost,
        controller_killed=args.controller_killed,
    )
    payload = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
