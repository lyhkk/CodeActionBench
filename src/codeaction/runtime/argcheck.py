"""Runtime enforcement of the declared tool argument contract (spec §4).

`TOOL_SPECS` was only ever a provider-side hint: a native tool schema constrains what a model
*generates*, so on the default surface the model never sent a malformed call — but nothing
enforced the contract at execution time. Any path that reaches the ToolBox without provider
schemas in front of it (a code-first surface, a `run_code` block, a vendor CLI) fell through to
Python's own `TypeError`, which catches an unknown keyword but not a wrong enum, a two-element
`target_xyz`, or a string where a number belongs. The code-first arm produced 52 such calls.

Validation therefore belongs at the ToolBox boundary, where direct calls and in-sandbox calls
pass through the same gate. It deliberately does NOT live in a static analysis of submitted code:
a check that only sees literal call sites would enforce the contract on plainly-written code and
skip it wherever the model used a variable, a dict, or a wrapper — making "was your code
statically analysable" part of what the benchmark measures.

The vocabulary matched here is exactly the vocabulary `TOOL_SPECS` uses: type, enum, items,
minItems/maxItems, minimum. Anything a schema does not state is not enforced.
"""

import math
from typing import Any, Mapping

from codeaction.interface.schemas import TOOL_SPECS

__all__ = ["ArgumentContractError", "validate_arguments", "checked_registry"]

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(float(v)),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, (list, tuple)),
    "object": lambda v: isinstance(v, Mapping),
}


class ArgumentContractError(ValueError):
    """A call whose arguments contradict the tool's declared schema."""


def _type_name(value: Any) -> str:
    for name, check in _TYPE_CHECKS.items():
        if check(value):
            return name
    return type(value).__name__


def _check_value(where: str, value: Any, schema: Mapping[str, Any]) -> list:
    problems = []
    expected = schema.get("type")
    if expected and not _TYPE_CHECKS.get(expected, lambda _v: True)(value):
        return [f"{where} must be {expected}, got {_type_name(value)}"]
    choices = schema.get("enum")
    if choices is not None and value not in choices:
        return [f"{where} must be one of {list(choices)}, got {value!r}"]
    if expected == "array":
        low, high = schema.get("minItems"), schema.get("maxItems")
        if low is not None and len(value) < int(low):
            problems.append(f"{where} needs at least {low} items, got {len(value)}")
        if high is not None and len(value) > int(high):
            problems.append(f"{where} allows at most {high} items, got {len(value)}")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                problems.extend(_check_value(f"{where}[{index}]", item, item_schema))
    minimum = schema.get("minimum")
    if minimum is not None and _TYPE_CHECKS["number"](value) and value < minimum:
        problems.append(f"{where} must be >= {minimum}, got {value}")
    maximum = schema.get("maximum")
    if maximum is not None and _TYPE_CHECKS["number"](value) and value > maximum:
        problems.append(f"{where} must be <= {maximum}, got {value}")
    return problems


def validate_arguments(tool: str, arguments: Mapping[str, Any]) -> None:
    """Raise `ArgumentContractError` when a call contradicts the tool's declared schema."""
    spec = TOOL_SPECS.get(str(tool))
    if spec is None:
        return
    schema = spec[1] or {}
    properties = schema.get("properties") or {}
    given = dict(arguments or {})
    problems = []
    missing = [name for name in (schema.get("required") or []) if name not in given]
    if missing:
        problems.append(f"missing required argument(s): {missing}")
    unknown = sorted(set(given) - set(properties))
    if unknown:
        problems.append(f"unknown argument(s): {unknown}; accepted: {sorted(properties)}")
    for name, value in given.items():
        prop = properties.get(name)
        if isinstance(prop, Mapping):
            problems.extend(_check_value(name, value, prop))
    if problems:
        raise ArgumentContractError(f"{tool}: " + "; ".join(problems))


def checked_registry(registry: Mapping[str, Any]) -> dict:
    """Wrap a name→callable registry so every call is validated before it reaches the tool."""
    def wrap(name, fn):
        def call(**kwargs):
            validate_arguments(name, kwargs)
            return fn(**kwargs)
        call.__name__ = str(name)
        call.__doc__ = getattr(fn, "__doc__", None)
        call.__wrapped__ = fn
        return call
    return {name: wrap(name, fn) for name, fn in registry.items()}
