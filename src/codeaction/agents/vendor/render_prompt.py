#!/usr/bin/env python3
"""Render one task-bound vendor-agent initial prompt for the development shell launcher."""
from __future__ import annotations

import argparse
from codeaction.interface.instructions import controller_prompt  # noqa: E402
from codeaction.benchmark.taskcard import (declared_scene_seeds, instruction_for_scene_seed,
                              load_task)  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--scene-seed", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--run-code-max-internal-calls", type=int)
    parser.add_argument("--agent-mode", default="claude",
                        help="which seat's discovery sentence to render")
    args = parser.parse_args(argv)
    from codeaction.agents.vendor.clis import vendor_cli
    tool_discovery = vendor_cli(args.agent_mode).tool_discovery
    card = load_task(args.task_name)
    scene_seed = (
        int(args.scene_seed)
        if args.scene_seed is not None else declared_scene_seeds(card)[0]
    )
    task_instruction = instruction_for_scene_seed(card, scene_seed)
    budgets = card["budgets"]
    max_tool_calls = (
        int(args.max_tool_calls)
        if args.max_tool_calls is not None else int(budgets["max_tool_calls"])
    )
    run_code_max_internal_calls = (
        int(args.run_code_max_internal_calls)
        if args.run_code_max_internal_calls is not None
        else int(budgets["run_code_max_internal_calls"])
    )
    print(controller_prompt(
        task_text=task_instruction,
        max_tool_calls=max_tool_calls,
        run_code_max_internal_calls=run_code_max_internal_calls,
        tool_discovery=tool_discovery,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
