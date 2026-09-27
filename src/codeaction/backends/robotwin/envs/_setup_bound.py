"""Assign at scene setup the attributes an upstream success predicate reads.

Three upstream tasks (`open_laptop`, `place_object_scale`, `put_object_cabinet`) compute the arm
they will use — and one of them the object's start height — inside `play_once`, then read those
attributes back in `check_success`. That is sound for expert data collection and unusable for a
benchmark: an agent episode never runs `play_once`, so the predicate raises AttributeError and the
task can never be scored. `check_success_audit` lists them under `play_only_attr_deps`.

The fix is NOT to change what the predicate means. Each rule is derivable from the scene alone
(the arm follows the object's side, or the laptop's facing; the start height is the object's height
before anything moves), so the subclasses here recompute exactly that rule once the scene is built
and settled. The expert then recomputes the identical value at the top of `play_once` and
overwrites it, which is why an expert replay is bit-identical to upstream and why admission
evidence gathered here is evidence about the upstream task.

Each subclass also carries the upstream predicate VERBATIM rather than inheriting it, so the static
audit analyses the real body against this class's own setup assignments (an inherited predicate is
invisible to a per-class AST scan, and a `return super().check_success()` stub would report clean
without evidence). `UPSTREAM_PREDICATE_SHA256` pins the copy: the test recomputes the digest from
the live upstream source, so an upstream edit fails the suite instead of silently splitting the two
predicates apart.
"""

import ast
import hashlib
import inspect
from pathlib import Path

from codeaction.paths import ROBOTWIN_ROOT


def upstream_predicate_digest(task_name: str) -> str:
    """SHA-256 of the upstream `check_success` source for `task_name` (whitespace-normalised).

    Normalising line-leading whitespace keeps the pin from tripping on a reindent while still
    catching any change to what the predicate reads, compares or returns.
    """
    source = (ROBOTWIN_ROOT / "envs" / f"{task_name}.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "check_success":
            body = ast.get_source_segment(source, node) or ""
            normalised = "\n".join(line.strip() for line in body.splitlines() if line.strip())
            return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    raise ValueError(f"envs/{task_name}.py defines no check_success")
def predicate_digest(path, class_name: str = None) -> str:
    """SHA-256 of one `check_success` body, whitespace-normalised.

    The same normalisation `upstream_predicate_digest` uses, addressed by FILE instead of by
    upstream task name, so a card can pin the predicate it actually derived from even when that
    predicate lives in this package rather than in `envs/`. Normalising line-leading whitespace
    keeps the pin from tripping on a reindent while still catching any change to what the
    predicate reads, compares or returns.
    """
    source = Path(path).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if class_name is not None and node.name != class_name:
            continue
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef) and sub.name == "check_success":
                body = ast.get_source_segment(source, sub) or ""
                normalised = "\n".join(line.strip() for line in body.splitlines() if line.strip())
                return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    raise ValueError(f"{path} defines no check_success"
                     + (f" on class {class_name}" if class_name else ""))
