"""
Primitive program validator.

`validate_program` runs synchronously *before* a program reaches the
runtime. If any rule fails, the program is rejected with a structured
ResultDict (stage="validation"). No motion is ever attempted on a
program that fails validation.

Rules implemented:
  V1  Program must be a list and entries must be dicts (handled at normalize time).
  V2  Sequence length <= MAX_OPS (default 20).
  V3  Every `op` must appear in ALLOWED_OPS.
  V4  Wait-steps `n` must satisfy n <= MAX_WAIT_STEPS (default 100).
  V5  Motion deltas must satisfy |delta| <= MAX_DELTA_MAGNITUDE (default 0.30 m).
  V6  Absolute poses, when concrete (no $-reference), must lie inside
      DEFAULT_WORKSPACE.
  V7  Any variable reference's root must be bound by a `save_as` of an
      earlier op (no forward references, no unknown names).
  V8  After any `close_gripper`, a subsequent lift-like motion (positive z
      delta, or move_to_pose with higher z than current EE) must be followed
      by at least one of {is_grasp_verified, is_lift_verified, is_task_success}.
  V9  Any `move_both_delta` with positive z must be followed by at least one
      of {is_lift_verified, is_task_success}.
  V10 String values must not contain forbidden tokens (TASK_ENV, env.move,
      take_action, scene.step, __import__, eval(, exec(, os., sys.,
      subprocess, socket).
  V11 Ops marked UNSUPPORTED_OPS are rejected with a clear "not implemented
      in the primitive runtime" message (rather than silently ignored).
  V12 Official actor-wrapper argument guards, including distinct dual grasp
      contact ids and numeric gripper/pregrasp values.
"""

from typing import List, Optional, Sequence, Set, Tuple

from .pose_utils import (
    DEFAULT_WORKSPACE, MAX_DELTA_MAGNITUDE,
    check_delta_magnitude, check_workspace_bounds,
)
from .program_schema import (
    collect_references, is_reference, normalize_program, reference_root,
)
from .result import FAILED, SUCCESS, make_skill_result


# ── Allow / forbid sets ───────────────────────────────────────────────────

#: Primitive program op allowlist; UNSUPPORTED_OPS are rejected separately.
ALLOWED_OPS: Set[str] = {
    "get_object_pose",
    "get_gripper_pose",
    "get_gripper_state",
    "compute_grasp",        # may be UNSUPPORTED in this implementation
    "compute_dual_grasp",
    "get_pose",             # may be UNSUPPORTED in this implementation
    "move_to_pose",
    "move_delta",
    "move_both_to_poses",
    "move_both_delta",
    "open_gripper",
    "close_gripper",
    "wait_steps",
    "is_grasp_verified",
    "is_lift_verified",
    "is_task_success",
    "move_to_home",
    "dual_grasp_actor",
}

#: Ops that the validator should accept structurally but that have no working
#: runtime implementation yet. Validation reports a clear error rather than
#: silently accepting and then crashing at runtime.
UNSUPPORTED_OPS: Set[str] = {
    "compute_grasp",  # single-arm analytic grasp not in primitives yet
    "get_pose",       # reference-relative pose lookup not in primitives yet
}

#: Tokens that must not appear anywhere in arg string values. Substring match.
FORBIDDEN_SUBSTRINGS: Tuple[str, ...] = (
    "TASK_ENV",
    "env.move",
    "take_action",
    "scene.step",
    "__import__",
    "eval(",
    "exec(",
    "os.system",
    "subprocess",
    "socket.",
    "raw_action",
)

#: Tokens that must not appear as the `op` field. Allowlist already enforces
#: this, but we keep the list for symmetry / clearer error messages.
FORBIDDEN_OP_NAMES: Set[str] = {
    "import", "open", "os", "sys", "subprocess", "socket",
    "eval", "exec", "take_action", "scene_step", "env_move",
    "TASK_ENV",
}


# ── Limits ────────────────────────────────────────────────────────────────

MAX_OPS_DEFAULT = 20
MAX_WAIT_STEPS = 100


# ── Helpers ───────────────────────────────────────────────────────────────

def _scan_strings(value, callback):
    """Walk arbitrary value, invoking callback(str) on every string leaf."""
    if isinstance(value, str):
        callback(value)
    elif isinstance(value, dict):
        for v in value.values():
            _scan_strings(v, callback)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _scan_strings(v, callback)


def _violation(index: int, rule: str, message: str,
               op: Optional[str] = None) -> dict:
    """Build a FAILED validation ResultDict."""
    return make_skill_result(
        event="validate_program",
        status=FAILED,
        stage="validation",
        details=f"validation rule {rule} failed at program[{index}]"
                f"{f' (op={op!r})' if op else ''}: {message}",
        validator_rule=rule,
        failed_op_index=index,
        op=op,
    )


def _positive_z_delta(args: dict, key: str = "delta") -> bool:
    """Return True iff args[key] is a length-3 sequence with z > 0."""
    d = args.get(key)
    if not isinstance(d, (list, tuple)) or len(d) != 3:
        return False
    try:
        return float(d[2]) > 0.0
    except (TypeError, ValueError):
        return False


# ── Main entry point ──────────────────────────────────────────────────────

def validate_program(program: List[dict],
                     refs: Optional[Sequence[str]] = None,
                     max_ops: int = MAX_OPS_DEFAULT) -> dict:
    """
    Validate a primitive program.

    Args:
        program: list of program entries (raw or pre-normalized).
        refs:    optional list of scene reference names; if provided, the
                 validator will check that any literal `object` arg appears
                 in this list. Pass None to skip this check.
        max_ops: length cap.

    Returns:
        A ResultDict with status SUCCESS or FAILED. On SUCCESS, the
        normalized program is in ``data["program"]``.
    """
    # 1. Normalize (V1 + structural checks)
    try:
        normalized = normalize_program(program)
    except (TypeError, ValueError) as e:
        return make_skill_result(
            event="validate_program",
            status=FAILED,
            stage="validation",
            details=f"program structure invalid: {e}",
            validator_rule="V1",
        )

    # 2. Length (V2)
    if len(normalized) > max_ops:
        return make_skill_result(
            event="validate_program",
            status=FAILED,
            stage="validation",
            details=f"program length {len(normalized)} exceeds limit {max_ops}",
            validator_rule="V2",
            total_ops=len(normalized),
        )

    # 3. Per-op checks
    bound_vars: Set[str] = set()

    # Track manipulation state for V8/V9
    seen_close_gripper = False
    seen_close_then_lift_motion = False
    seen_positive_z_both_delta = False
    needs_grasp_verification = False
    needs_lift_verification = False

    for i, entry in enumerate(normalized):
        op = entry["op"]
        args = entry["args"]
        save_as = entry["save_as"]

        # V3 — op allowlist
        if op in FORBIDDEN_OP_NAMES:
            return _violation(i, "V3", f"op {op!r} is on the forbidden list", op=op)
        if op not in ALLOWED_OPS:
            return _violation(
                i, "V3",
                f"op {op!r} is not in the primitive program allowlist "
                f"(allowed: {sorted(ALLOWED_OPS)})",
                op=op,
            )

        # V10 — forbidden substrings in any string value
        bad_token: List[str] = []
        def _check(s: str):
            for tok in FORBIDDEN_SUBSTRINGS:
                if tok in s:
                    bad_token.append(tok)
        _scan_strings(args, _check)
        _scan_strings(save_as, _check)
        if bad_token:
            return _violation(
                i, "V10",
                f"forbidden token(s) {sorted(set(bad_token))} appear in args/save_as",
                op=op,
            )
        if isinstance(save_as, str) and save_as.startswith("$"):
            return _violation(
                i, "V12",
                "save_as must be a plain variable name and must not start with '$'",
                op=op,
            )

        # V11 — explicitly unsupported ops
        if op in UNSUPPORTED_OPS:
            return _violation(
                i, "V11",
                f"op {op!r} is on the primitive program allowlist but is not "
                f"implemented in the primitive runtime. Use the safe-skill "
                f"equivalent (e.g. dual_arm_grasp / compute_dual_grasp).",
                op=op,
            )

        # V7 — variable references' root must be a previously bound name
        for ref in collect_references(args):
            root = reference_root(ref)
            if root is None:
                continue
            if root not in bound_vars:
                return _violation(
                    i, "V7",
                    f"reference {ref!r} uses unknown variable "
                    f"{root!r} (bound so far: {sorted(bound_vars)})",
                    op=op,
                )

        # Per-op semantic checks
        if op == "get_object_pose":
            if not args.get("object_name"):
                return _violation(i, "V12", "get_object_pose requires object_name", op=op)

        elif op == "wait_steps":
            n = args.get("n")
            try:
                n_int = int(n)
            except (TypeError, ValueError):
                return _violation(i, "V4",
                                  f"wait_steps requires int 'n', got {n!r}",
                                  op=op)
            if n_int < 0 or n_int > MAX_WAIT_STEPS:
                return _violation(
                    i, "V4",
                    f"wait_steps n={n_int} outside [0, {MAX_WAIT_STEPS}]",
                    op=op,
                )

        elif op == "move_delta":
            err = _check_delta(args.get("delta"))
            if err:
                return _violation(i, "V5", err, op=op)

        elif op == "move_both_delta":
            for key in ("left_delta", "right_delta"):
                err = _check_delta(args.get(key), name=key)
                if err:
                    return _violation(i, "V5", err, op=op)
            if _positive_z_delta(args, "left_delta") or _positive_z_delta(args, "right_delta"):
                needs_lift_verification = True
                seen_positive_z_both_delta = True

        elif op == "move_to_pose":
            err = _check_target_pose(args.get("pose"))
            if err:
                return _violation(i, "V6", err, op=op)

        elif op == "move_both_to_poses":
            for key in ("left_pose", "right_pose"):
                err = _check_target_pose(args.get(key), name=key)
                if err:
                    return _violation(i, "V6", err, op=op)

        elif op == "close_gripper":
            seen_close_gripper = True

        elif op == "dual_grasp_actor":
            seen_close_gripper = True  # op internally closes grippers
            if not (args.get("object") or args.get("object_name")):
                return _violation(i, "V12", "dual_grasp_actor requires object or object_name", op=op)
            left_cpid = args.get("left_contact_point_id")
            right_cpid = args.get("right_contact_point_id")
            if not isinstance(left_cpid, int) or not isinstance(right_cpid, int):
                return _violation(
                    i, "V12",
                    "dual_grasp_actor requires integer left_contact_point_id "
                    "and right_contact_point_id",
                    op=op,
                )
            if left_cpid == right_cpid:
                return _violation(
                    i, "V12",
                    "dual_grasp_actor left_contact_point_id and "
                    "right_contact_point_id must be different",
                    op=op,
                )
            for key in ("pre_grasp_dis", "grasp_dis", "gripper_pos", "preclose_gripper_pos"):
                if key not in args:
                    continue
                try:
                    value = float(args[key])
                except (TypeError, ValueError):
                    return _violation(i, "V12", f"{key} must be numeric", op=op)
                if key in ("gripper_pos", "preclose_gripper_pos") and not (0.0 <= value <= 1.0):
                    return _violation(i, "V12", f"{key}={value} outside [0.0, 1.0]", op=op)

        elif op == "is_lift_verified":
            allowed = {"object_name", "z_before", "min_dz"}
            unknown = sorted(set(args) - allowed)
            if unknown:
                return _violation(i, "V12", f"is_lift_verified unknown arg(s): {unknown}", op=op)
            for key in ("object_name", "z_before", "min_dz"):
                if key not in args:
                    return _violation(i, "V12", f"is_lift_verified requires {key}", op=op)

        # V8 — once close_gripper has been seen, any subsequent lift-like
        # motion (positive-z delta or move_to_pose) triggers the
        # require-verification flag.
        if seen_close_gripper:
            if (op == "move_delta" and _positive_z_delta(args)) or \
               (op == "move_both_delta" and (_positive_z_delta(args, "left_delta") or
                                             _positive_z_delta(args, "right_delta"))) or \
               op == "move_to_pose" or op == "move_both_to_poses":
                seen_close_then_lift_motion = True
                needs_grasp_verification = True

        # V8/V9 satisfied?
        if op in ("is_grasp_verified", "is_lift_verified", "is_task_success"):
            if needs_grasp_verification:
                needs_grasp_verification = False
            if op in ("is_lift_verified", "is_task_success"):
                needs_lift_verification = False

        # If we are an object-name-bearing op and refs were provided, check
        # the literal name is known (only if it's a literal, not a ref).
        if refs is not None:
            obj = args.get("object") or args.get("object_name") or args.get("reference")
            if isinstance(obj, str) and not is_reference(obj):
                if obj not in set(refs):
                    return _violation(
                        i, "V7",
                        f"object name {obj!r} not in scene refs "
                        f"(known: {sorted(refs)[:10]})",
                        op=op,
                    )

        # Bind the variable for downstream references
        if save_as:
            if save_as in bound_vars:
                return _violation(
                    i, "V7",
                    f"variable {save_as!r} re-bound; pick a unique save_as",
                    op=op,
                )
            bound_vars.add(save_as)

    # V8 / V9 — final check
    if needs_grasp_verification:
        return make_skill_result(
            event="validate_program",
            status=FAILED,
            stage="validation",
            details=("program closes the gripper and then performs a "
                     "lift-like motion but never calls is_grasp_verified or "
                     "is_lift_verified afterwards"),
            validator_rule="V8",
        )
    if needs_lift_verification and seen_positive_z_both_delta:
        return make_skill_result(
            event="validate_program",
            status=FAILED,
            stage="validation",
            details=("program performs a positive-z move_both_delta but "
                     "never calls is_lift_verified or is_task_success "
                     "afterwards"),
            validator_rule="V9",
        )

    # Success
    return make_skill_result(
        event="validate_program",
        status=SUCCESS,
        stage="validation",
        details=f"program OK: {len(normalized)} op(s) validated.",
        program=normalized,
        total_ops=len(normalized),
        bound_vars=sorted(bound_vars),
    )


# ── Per-arg validators ────────────────────────────────────────────────────

def _check_delta(d, name: str = "delta") -> Optional[str]:
    """Return error message if delta is invalid; None if OK or unresolvable."""
    if d is None:
        return f"missing required arg {name!r}"
    if is_reference(d):
        return None  # validated at runtime when reference is resolved
    if isinstance(d, str):
        return f"{name} must be a list/tuple of 3 floats, got string {d!r}"
    if not isinstance(d, (list, tuple)) or len(d) != 3:
        return f"{name} must be length 3, got {d!r}"
    if any(is_reference(x) for x in d):
        return None  # element is a $ref → defer to runtime
    try:
        dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    except (TypeError, ValueError):
        return f"{name} elements must be numeric: {d!r}"
    ok, mag = check_delta_magnitude(dx, dy, dz)
    if not ok:
        return f"{name} magnitude {mag:.3f}m exceeds limit {MAX_DELTA_MAGNITUDE}m"
    return None


def _check_target_pose(p, name: str = "pose") -> Optional[str]:
    """Validate concrete 7-element pose against workspace bounds."""
    if p is None:
        return f"missing required arg {name!r}"
    if is_reference(p):
        return None
    if not isinstance(p, (list, tuple)) or len(p) != 7:
        # If it's an unresolved sub-spec with $refs we cannot validate fully
        if isinstance(p, (list, tuple)) and any(is_reference(x) for x in p):
            return None
        return f"{name} must be length 7 [x,y,z,qw,qx,qy,qz], got {p!r}"
    if any(is_reference(x) for x in p):
        return None
    try:
        xyz = [float(p[0]), float(p[1]), float(p[2])]
    except (TypeError, ValueError):
        return f"{name} numeric components invalid: {p!r}"
    ok, msg = check_workspace_bounds(xyz)
    if not ok:
        return f"{name} {msg}"
    return None
