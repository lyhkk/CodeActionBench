"""Policy-side env package for benchmark task environments.

New long-horizon task envs live HERE, not in upstream `envs/` — import-and-subclass, the same
doctrine that keeps `CodeAction/` byte-untouched. Every module in this package:

- defines exactly ONE task class whose name equals the module name (the generic loader in
  `codeaction.backends.robotwin.scene` resolves `envs.<task_name>` first, then `envs_ext.<task_name>`);
- subclasses `envs._base_task.Base_Task` (directly or via an existing upstream task class) and
  reuses upstream machinery (`rand_create_sapien_urdf_obj`, `together_move_to_pose`, ...);
- treats `play_once()` as the HOST-SIDE scripted reference: it may read ground truth, but it
  must not set sticky success flags, call `check_success()` mid-episode, or leave attrs that
  `check_success` depends on (the AST audit `codeaction.check_success_audit` scans this package —
  see tests/test_codeaction_check_success_audit.py);
- is never imported by the agent-facing surface (`codeaction/tools.py`, `runner.py`, `sandbox.py`,
  `mcp_bridge.py` must not reference this package — enforced in tests/test_codeaction_envs_ext.py).

Importing a task module requires the sim runtime (the upstream `envs` package pulls SAPIEN);
keep THIS file import-free so AST audits and package discovery stay Mac-runnable.
"""
