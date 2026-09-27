"""Generic RoboTwin env loader for codeaction benchmark episodes
(spec docs/2026-07-08-dualarm-longhorizon-baseline-design.md §2).

Mirrors the standard eval bring-up — script/eval_policy.py's class_decorator + setup_demo with
eval_mode/is_test/skip_expert_check — but PARAMETERIZED by task_name and WITHOUT ever calling
play_once() (the agent replaces the expert). _build_task_args is a task_name-parameterized copy of
depth_grounding_diagnostic._build_lift_pot_args; its correctness (and load_task_scene's) is proven by
the remote smoke data/dualarm_smoke.py — the robot config.yml + assets live only on the sim box.

Sim imports are function-local so `import codeaction.backends.robotwin.scene` stays sim-free: the pure helper
_eval_setup_seed is unit-testable on the Mac (tests/test_codeaction_scene.py); load_task_scene runs only
on the sim box.

GT wall (spec §7): setup-side / out-of-band. Hands env+vp to the ToolBox via closure (the agent never
gets the handle) and env.check_success() to the verifier. Reads NO object ground truth —
env_check_success tasks need none. Introduce no deny-listed GT accessor here (see the maintainer note
below); the source-leak self-audit globs codeaction/*.py.
"""
# GT-wall reminder — NEVER call a deny-listed GT accessor in this file (actor_center, get_object_pose,
# get_scene_objects, get_segmentation, get_depth, get_point_cloud, .actor, env.pot, get_contact_point);
# tests/test_codeaction_leakaudit.py globs codeaction/*.py and will fail. env_check_success needs none.
import importlib
import os
from pathlib import Path

import yaml

from codeaction.paths import CODEACTION_CONFIG_ROOT, ROBOTWIN_ROOT, ROBOTWIN_CONFIG_ROOT


def _eval_setup_seed(seed_index: int) -> int:
    """CLI seed index -> setup_demo seed, exactly as script/eval_policy.py maps it."""
    return 100000 * (1 + int(seed_index))


_ENV_PACKAGES = {"envs": "envs", "envs_ext": "codeaction.backends.robotwin.envs"}


def _resolve_env_class(task_name: str, declared_source: str | None = None):
    """Resolve the environment source. Returns (env_class, env_source).

    A card's `scene.env_source` DECIDES which package the env comes from; it is not merely
    checked afterwards. That matters when both packages define the same task name, which is
    exactly what happens when a released predicate is corrected in place: the corrected env
    subclasses the upstream one and keeps its name, and search-order-only resolution would
    always return the upstream class and silently score the task with the predicate we replaced.
    Undeclared cards keep the historical order — upstream `envs.<task>` first, this package as
    the fallback for new task envs — so every frozen card resolves exactly as before.

    A ModuleNotFoundError raised from INSIDE an env module (a real missing dependency, or the
    `envs` package itself absent from sys.path) propagates instead of being misread as
    'task not in envs/'."""
    if declared_source is not None:
        if declared_source not in _ENV_PACKAGES:
            raise ValueError(f"unknown scene.env_source {declared_source!r}")
        mod = importlib.import_module(f"{_ENV_PACKAGES[declared_source]}.{task_name}")
        source = declared_source
    else:
        try:
            mod, source = importlib.import_module(f"envs.{task_name}"), "envs"
        except ModuleNotFoundError as e:
            if e.name != f"envs.{task_name}":
                raise
            mod = importlib.import_module(
                f"codeaction.backends.robotwin.envs.{task_name}")
            source = "envs_ext"
    from codeaction.backends.robotwin.task_adapter import adapt_task_class

    return adapt_task_class(getattr(mod, task_name)), source


def _validate_env_source(expected, actual):
    """Belt-and-braces on the resolution above: when a card declares a source, that is what was
    imported, so this can only fire on an undeclared card — which is why it still exists."""
    if expected is not None and expected != actual:
        raise ValueError(f"task card declares env_source={expected!r} "
                         f"but the env resolved from {actual!r}")


def _yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.load(fh.read(), Loader=yaml.FullLoader)


def _task_config_path(config: str) -> Path:
    owned = CODEACTION_CONFIG_ROOT / f"{config}.yml"
    return owned if owned.is_file() else ROBOTWIN_CONFIG_ROOT / f"{config}.yml"


def _build_task_args(task_name: str, seed: int, config: str, eval_mode: bool = True) -> dict:
    """Assemble setup_demo(**args) the same way script/eval_policy.py does — a task_name-parameterized
    copy of depth_grounding_diagnostic._build_lift_pot_args (the ONLY task-specific value is
    task_name). Robot/camera config paths resolve against the repo root so CWD does not matter.
    Not locally unit-tested (robot assets live on the sim box); the remote smoke proves it."""
    args = _yaml(_task_config_path(config))
    args.update(seed=seed, task_name=task_name, task_config=config,
                skip_expert_check=True, eval_video_log=False)
    if eval_mode:
        args["eval_mode"] = True

    emb = args.get("embodiment")
    et = _yaml(ROBOTWIN_CONFIG_ROOT / "_embodiment_config.yml")

    def _robot_dir(name):
        return str(ROBOTWIN_ROOT / et[name]["file_path"])

    if len(emb) == 1:
        args["left_robot_file"] = _robot_dir(emb[0])
        args["right_robot_file"] = _robot_dir(emb[0])
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = _robot_dir(emb[0])
        args["right_robot_file"] = _robot_dir(emb[1])
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items must be 1 or 3")
    args["left_embodiment_config"] = _yaml(os.path.join(args["left_robot_file"], "config.yml"))
    args["right_embodiment_config"] = _yaml(os.path.join(args["right_robot_file"], "config.yml"))

    cc = _yaml(ROBOTWIN_CONFIG_ROOT / "_camera_config.yml")
    hct = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cc[hct]["h"]
    args["head_camera_w"] = cc[hct]["w"]
    return args


def _codeaction_curobo_planner_class():
    """Resolve the planner class injected by CodeActionTaskMixin."""
    from codeaction.backends.robotwin.robot import CuroboPlanner

    return CuroboPlanner


def assert_curobo_planner(env):
    """Fail the episode rather than let it plan against object ground truth (GT wall, spec §7).

    The task adapter constructs `codeaction.backends.robotwin.robot.Robot`, so the type check must
    use the CodeAction planner class rather than upstream's same-named class. The CodeAction robot
    fails closed when cuRobo is unavailable; accepting an upstream MplibPlanner here would plan
    against scene geometry and ground-truth object poses.

    Deliberately has no environment-variable escape hatch: an opt-out would restore exactly the
    silent-degradation shape this removes. Sim-only (function-local sim import)."""
    curobo_cls = _codeaction_curobo_planner_class()
    if curobo_cls is None:
        raise RuntimeError(
            "cuRobo is not importable in the CodeAction backend; refusing to construct a "
            "scene-aware fallback planner")

    robot = getattr(env, "robot", None)
    if robot is None:
        raise RuntimeError("env exposes no .robot, so the planner actually in use cannot be checked")

    if getattr(robot, "communication_flag", False):
        # Per-arm yml differ -> planners live in subprocesses; planner_process_worker constructs
        # CuroboPlanner unconditionally, so a live worker is the strongest available evidence.
        for side in ("left", "right"):
            proc = getattr(robot, f"{side}_proc", None)
            if proc is None or not proc.is_alive():
                raise RuntimeError(
                    f"{side}-arm cuRobo planner subprocess is missing or dead; the episode would "
                    "plan through an unverified path")
        return

    for side in ("left", "right"):
        in_use = getattr(robot, f"{side}_planner", None)
        if not isinstance(in_use, curobo_cls):
            raise RuntimeError(
                f"{side} arm is planning with {type(in_use).__name__}, not CuroboPlanner. The "
                "MplibPlanner fallback is scene-aware (object geometry + GT poses) and must never "
                "run a benchmark episode.")


def load_task_scene(task_name, *, seed, config="demo_clean_aloha", expected_env_source=None):
    """Boot a RoboTwin task env for a codeaction episode — generic over task_name, NEVER runs the expert
    play_once(). Resolves the env class from upstream `envs/` or the policy-side `envs_ext/`
    package (pass the card's optional `scene.env_source` as expected_env_source to fail fast on
    a declaration mismatch). Returns {env, vp, table_z, env_source}. Sim-only (function-local sim
    imports)."""
    from codeaction.backends.robotwin.perception import VisionPerception, SimPerceptionBackend

    setup_seed = _eval_setup_seed(seed)
    args = _build_task_args(task_name, setup_seed, config, eval_mode=True)
    args.pop("seed", None)          # setup_demo takes seed explicitly, not inside **args
    env_class, env_source = _resolve_env_class(task_name, expected_env_source)
    _validate_env_source(expected_env_source, env_source)
    env = env_class()
    env.setup_demo(now_ep_num=0, seed=setup_seed, is_test=True, **args)
    assert_curobo_planner(env)
    vp = VisionPerception(env, backend=SimPerceptionBackend(env))
    table_z = 0.74 + float(getattr(env, "table_z_bias", 0.0))
    return dict(env=env, vp=vp, table_z=table_z, env_source=env_source,
                task_name=task_name, config=config)


def reset_task_scene(ctx, *, seed, clear_cache=False):
    """In-process episode reset (D10): close the current scene and re-run setup_demo on the SAME
    env instance for a new seed — exactly script/eval_policy.py's per-episode loop (setup_demo →
    close_env → setup_demo), amortizing the cold boot across seeds/episodes. Returns an updated
    ctx dict (same env object; fresh VisionPerception — cameras are rebuilt by setup_demo, so
    every per-scene handle must be re-created; the same applies to any StepObserver, which must be
    re-attached to the NEW scene object). `clear_cache=True` additionally drops the SAPIEN mesh
    cache, mirroring eval_policy's periodic clear_cache_freq. Determinism/equivalence vs a fresh
    process is gated by data/reset_equivalence_probe.py, not assumed here."""
    from codeaction.backends.robotwin.perception import VisionPerception, SimPerceptionBackend

    env, task_name, config = ctx["env"], ctx["task_name"], ctx["config"]
    env.close_env(clear_cache=clear_cache)
    setup_seed = _eval_setup_seed(seed)
    args = _build_task_args(task_name, setup_seed, config, eval_mode=True)
    args.pop("seed", None)
    env.setup_demo(now_ep_num=0, seed=setup_seed, is_test=True, **args)
    assert_curobo_planner(env)   # Robot.reset can rebuild planners; re-check every episode
    vp = VisionPerception(env, backend=SimPerceptionBackend(env))
    table_z = 0.74 + float(getattr(env, "table_z_bias", 0.0))
    return dict(env=env, vp=vp, table_z=table_z, env_source=ctx["env_source"],
                task_name=task_name, config=config)
