"""
Primitive program schema and variable resolution.

A primitive program is a `list[dict]`. Each entry is normalized into the
canonical shape:

    {
        "op":      <op_name: str>,
        "args":    <kwargs: dict>,
        "save_as": <varname: str | None>,
    }

Variable references — runtime substitution
==========================================
Any string in `args` of the form ``"$name"`` or ``"$name.path.into.result"``
is resolved against the running program state. State is a dict keyed by
`save_as` names; each value is the full PrimitiveResult of that step.

Resolution rules:
- ``$name``                 → state["name"]
- ``$name.field``           → state["name"]["field"]
- ``$name.field.subfield``  → ditto, walking the dict tree
- list-indexing notation ``[N]`` is supported on intermediate or terminal
  segments, e.g. ``$pose.data.position[2]``.

No Python ``eval`` or ``exec`` is ever used. Resolution is pure dict/list
traversal. Unknown references return an explicit error tuple rather than
raising — callers are expected to convert that into a FAILED ResultDict.
"""

from typing import Any, Iterable, List, Optional, Tuple


# ── Public constants ──────────────────────────────────────────────────────

REF_PREFIX = "$"


# ── Normalization ─────────────────────────────────────────────────────────

def normalize_program_entry(entry: dict) -> dict:
    """
    Normalize a single program entry.

    Accepts both:
        {"op": "move_delta", "arm": "right", "delta": [0,0,0.05], "save_as": "x"}
    and:
        {"op": "move_delta", "args": {"arm": "right", "delta": [0,0,0.05]},
         "save_as": "x"}

    Returns: canonical form
        {"op": <str>, "args": <dict>, "save_as": <str|None>}

    Raises:
        TypeError / ValueError for structurally invalid entries — the caller
        should catch these and report them as validation errors.
    """
    if not isinstance(entry, dict):
        raise TypeError(f"program entry must be a dict, got {type(entry).__name__}")
    if "op" not in entry:
        raise ValueError(f"program entry missing 'op' key: {entry!r}")

    op = entry["op"]
    if not isinstance(op, str) or not op:
        raise ValueError(f"'op' must be a non-empty string, got {op!r}")

    # Args resolution: explicit "args" wins; otherwise, inline keys.
    if "args" in entry:
        args = entry["args"]
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise TypeError(
                f"'args' must be a dict, got {type(args).__name__}: {args!r}"
            )
        args = dict(args)
    else:
        args = {k: v for k, v in entry.items() if k not in ("op", "save_as", "args")}

    save_as = entry.get("save_as")
    if save_as is not None:
        if not isinstance(save_as, str) or not save_as:
            raise ValueError(f"'save_as' must be a non-empty string, got {save_as!r}")
        if save_as.startswith(REF_PREFIX):
            raise ValueError(
                f"'save_as' name must not start with '{REF_PREFIX}': {save_as!r}"
            )

    return {"op": op, "args": args, "save_as": save_as}


def normalize_program(program: Iterable[dict]) -> List[dict]:
    """
    Normalize every entry in a program list.

    Each entry must be a dict; otherwise the offending index is reported.
    Returns a fresh list of canonical dicts.
    """
    if not isinstance(program, list):
        raise TypeError(
            f"program must be a list, got {type(program).__name__}"
        )
    out: List[dict] = []
    for i, entry in enumerate(program):
        try:
            out.append(normalize_program_entry(entry))
        except (TypeError, ValueError) as e:
            raise ValueError(f"program[{i}]: {e}") from e
    return out


# ── Variable reference parsing & resolution ───────────────────────────────

def is_reference(value: Any) -> bool:
    """A string is a variable reference iff it starts with ``$``."""
    return isinstance(value, str) and value.startswith(REF_PREFIX) and len(value) > 1


def _parse_segments(ref: str) -> List[str]:
    """
    Break ``"$grasp.data.position[2]"`` into
        ["grasp", "data", "position", "[2]"]
    Bracket segments are kept as-is so the resolver can detect indexing.
    """
    body = ref[len(REF_PREFIX):]
    # Split on '.' first, then split off bracket suffixes.
    raw = body.split(".")
    segs: List[str] = []
    for chunk in raw:
        # Pull off any number of trailing [N] groups.
        head, *brackets = _split_brackets(chunk)
        if head:
            segs.append(head)
        for b in brackets:
            segs.append(b)
    return segs


def _split_brackets(token: str) -> List[str]:
    """Split 'position[2][1]' → ['position', '[2]', '[1]']."""
    out: List[str] = []
    cur = ""
    i = 0
    n = len(token)
    while i < n:
        c = token[i]
        if c == "[":
            if cur:
                out.append(cur)
                cur = ""
            j = token.find("]", i)
            if j == -1:
                raise ValueError(f"unbalanced '[' in reference segment: {token!r}")
            out.append(token[i:j + 1])  # keep brackets to flag indexing
            i = j + 1
        else:
            cur += c
            i += 1
    if cur:
        out.append(cur)
    return out


def _step(value: Any, segment: str) -> Tuple[Any, Optional[str]]:
    """One step of traversal. Returns (next_value, error_msg_or_None)."""
    if segment.startswith("[") and segment.endswith("]"):
        idx_str = segment[1:-1]
        try:
            idx = int(idx_str)
        except ValueError:
            return None, f"non-integer list index {segment!r}"
        if not isinstance(value, (list, tuple)):
            return None, f"cannot index non-sequence with {segment} (got {type(value).__name__})"
        if idx < -len(value) or idx >= len(value):
            return None, f"index {idx} out of range for sequence of length {len(value)}"
        return value[idx], None
    # Dict field
    if not isinstance(value, dict):
        return None, f"cannot read field {segment!r} from non-dict {type(value).__name__}"
    if segment not in value:
        return None, f"field {segment!r} not found (available: {sorted(value.keys())[:10]})"
    return value[segment], None


def resolve_program_value(value: Any, state: dict) -> Tuple[Any, Optional[str]]:
    """
    Recursively resolve any string references inside ``value`` against ``state``.

    Returns ``(resolved_value, error)``. On success, ``error`` is ``None``.
    On failure, the resolved value is undefined and ``error`` is a human-
    readable string suitable for an error ResultDict.

    Containers (list, tuple, dict) are walked recursively; strings that are
    not references are returned as-is. Non-string scalars pass through.
    """
    # String — maybe a reference.
    if isinstance(value, str):
        if not is_reference(value):
            return value, None
        try:
            segs = _parse_segments(value)
        except ValueError as e:
            return None, f"cannot parse reference {value!r}: {e}"
        if not segs:
            return None, f"empty reference {value!r}"
        root = segs[0]
        if root not in state:
            return None, (
                f"unknown variable {REF_PREFIX}{root} "
                f"(known: {sorted(state.keys())})"
            )
        cur: Any = state[root]
        for seg in segs[1:]:
            cur, err = _step(cur, seg)
            if err:
                return None, f"resolving {value!r}: {err}"
        return cur, None

    # Containers — walk and resolve children, preserving shape.
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            sub, err = resolve_program_value(v, state)
            if err:
                return None, err
            out[k] = sub
        return out, None

    if isinstance(value, list):
        out_list: list = []
        for i, v in enumerate(value):
            sub, err = resolve_program_value(v, state)
            if err:
                return None, f"list[{i}]: {err}"
            out_list.append(sub)
        return out_list, None

    if isinstance(value, tuple):
        out_tuple = []
        for i, v in enumerate(value):
            sub, err = resolve_program_value(v, state)
            if err:
                return None, f"tuple[{i}]: {err}"
            out_tuple.append(sub)
        return tuple(out_tuple), None

    # Scalar (int, float, bool, None, etc.)
    return value, None


# ── Static reference collection (used by the validator) ───────────────────

def collect_references(value: Any) -> List[str]:
    """
    Return all variable-reference strings found anywhere inside ``value``.
    Used by the validator to detect forward / unknown references statically.
    """
    refs: List[str] = []
    if isinstance(value, str):
        if is_reference(value):
            refs.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            refs.extend(collect_references(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            refs.extend(collect_references(v))
    return refs


def reference_root(ref: str) -> Optional[str]:
    """Return the root variable name of a reference, or None if not a ref."""
    if not is_reference(ref):
        return None
    body = ref[len(REF_PREFIX):]
    return body.split(".")[0].split("[")[0]
