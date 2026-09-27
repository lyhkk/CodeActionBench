"""A credential-free command agent using the JSON-lines tool protocol."""
import json
from pathlib import Path
import sys


def create(context):
    return [sys.executable, "-u", str(Path(__file__).resolve())]


def call(tool, arguments):
    print(json.dumps({"tool": tool, "arguments": arguments}), flush=True)
    response = sys.stdin.readline()
    if not response:
        raise RuntimeError("tool channel closed before its response")
    return json.loads(response)


def main():
    context = json.loads(sys.stdin.readline())
    if not context["task"] or not context["output_dir"]:
        raise ValueError("task text and output directory are required")
    observation = call("capture_head", {})
    if not observation["images"]:
        raise RuntimeError("image tool result did not reach the command agent")
    call("get_robot_state", {"arms": ["left", "right"]})
    call("done", {"report": "Command agent lifecycle check completed.", "success_claim": False})


if __name__ == "__main__":
    main()
