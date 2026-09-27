"""Definitions for declared interface-only tools outside the pinned D0 base set."""


BASH_EXEC_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "exit_code": {"type": "integer"},
        "timed_out": {"type": "boolean"},
        "truncated": {"type": "boolean"},
        "output": {"type": "string"},
        "error": {"type": "string"},
    },
    "required": ["ok"],
    "additionalProperties": False,
}

BASH_EXEC_DEFINITION = {
    "name": "bash_exec",
    "description": ("Execute a shell script in an isolated scratch container. Files under "
                    "/workspace persist for this episode. The script may call public robot tools "
                    "with: codeaction-tool TOOL JSON_OBJECT. Returns "
                    "{ok,exit_code,timed_out,truncated,output}; gateway failure returns {ok:false,error}."),
    "inputSchema": {
        "type": "object",
        "properties": {"script": {"type": "string"}},
        "required": ["script"],
        "additionalProperties": False,
    },
    "outputSchema": BASH_EXEC_OUTPUT_SCHEMA,
}


def validate_bash_exec_result(payload):
    """Fail closed if the gateway or launcher drifts from the advertised extra-tool result."""
    if not isinstance(payload, dict):
        raise ValueError("bash_exec result must be an object")
    unknown = sorted(set(payload) - set(BASH_EXEC_OUTPUT_SCHEMA["properties"]))
    if unknown:
        raise ValueError(f"bash_exec result contains unknown keys: {unknown}")
    if not isinstance(payload.get("ok"), bool):
        raise ValueError("bash_exec result.ok must be boolean")
    if payload["ok"] or "error" not in payload:
        expected = {
            "exit_code": int, "timed_out": bool, "truncated": bool, "output": str,
        }
        for name, expected_type in expected.items():
            value = payload.get(name)
            if (not isinstance(value, expected_type)
                    or expected_type is int and isinstance(value, bool)):
                raise ValueError(f"bash_exec result.{name} has the wrong type")
    elif not isinstance(payload.get("error"), str):
        raise ValueError("bash_exec failure result.error must be a string")
