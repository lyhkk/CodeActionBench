"""Run only the requested reference sequences; no model credentials or automatic retries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

from codeaction.paths import PROJECT_ROOT


def run_replay(args) -> int:
    from codeaction.benchmark.taskcard import validate_task_pack
    from codeaction.cli.main import ControllerError, _inspect_image
    from codeaction.release import load_release, verify_assets, image_matches_runtime, runtime_identity
    from codeaction.execution_snapshot import prepare_release_snapshot

    available = validate_task_pack(PROJECT_ROOT / "benchmark/tasks")["tasks"]
    if len(args.tasks) != len(set(args.tasks)) or any(task not in available for task in args.tasks):
        raise ControllerError("replay tasks must be distinct names from the released task pack")
    if args.gpu < 0:
        raise ControllerError("gpu must be non-negative")
    image_ref = args.sim_image
    assets = args.assets_root.expanduser().resolve()
    try:
        release = load_release(args.release_manifest, PROJECT_ROOT) if args.release_manifest else None
        if release is not None:
            image_ref = release["images"]["sim"]
        if args.dry_run:
            print(json.dumps({"tasks": args.tasks, "gpu": args.gpu, "sim_image": image_ref,
                              "out_dir": str(args.out_dir) if args.out_dir else "new runs/replay directory"}))
            return 0
        if release is not None:
            verify_assets(assets, release["assets"])
        if any(not (assets / tree).is_dir() for tree in ("objects", "embodiments", "background_texture")):
            raise ValueError("set assets_root or CODEACTION_ASSETS_ROOT to the installed resources")
        # A tag moved by a later build must not change the remaining selected tasks.
        image = _inspect_image(image_ref)
        from codeaction.launch import context
        if context() is not None:
            from codeaction.environments import validate_environment
            validate_environment(image.labels, PROJECT_ROOT, "sim")
        elif release is None and not image_matches_runtime(image.labels, runtime_identity(PROJECT_ROOT)):
            raise ValueError("sim image differs from runtime; run tools/build_images.sh")
        image_ref = image.image_id
        task_pack = PROJECT_ROOT / "benchmark/tasks"
        if release is not None:
            snapshot, _ = prepare_release_snapshot(
                PROJECT_ROOT, args.release_manifest, Path.home() / ".cache/codeaction/executions")
            task_pack = snapshot / "benchmark/tasks"
        if args.out_dir is None:
            parent = PROJECT_ROOT / "runs/replay"
            parent.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix="selected-", dir=parent))
        else:
            output = args.out_dir.expanduser().resolve()
            output.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, CODEACTION_ASSETS_ROOT=str(assets), CODEACTION_SIM_IMAGE=image_ref,
                   CODEACTION_GPU=str(args.gpu), CODEACTION_TASK_PACK_ROOT=str(task_pack),
                   CODEACTION_SIM_CODE=str(PROJECT_ROOT))
        results = []
        print(f"Replay output: {output}", flush=True)
        for task in args.tasks:
            process = subprocess.run(["bash", str(PROJECT_ROOT / "tools/replay_container.sh"),
                                      task, str(output / task)], env=env, check=False)
            results.append({"task": task, "exit_code": process.returncode})
            (output / "selection.json").write_text(json.dumps(
                {"sim_image": image_ref, "results": results}, indent=2) + "\n")
        failed = [item["task"] for item in results if item["exit_code"]]
        print("Failed tasks: " + ", ".join(failed) if failed else "All selected replays passed.")
        return 1 if failed else 0
    except (OSError, ValueError, KeyError) as exc:
        raise ControllerError(f"replay: {exc}") from exc
