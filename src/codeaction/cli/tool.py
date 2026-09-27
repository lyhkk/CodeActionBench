#!/usr/bin/env python3
"""Scratch-container CLI for one allowlisted benchmark tool call."""
import argparse
import json
import os
import sys

from codeaction.runtime.public_tool_socket import call_public_tool


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="codeaction-tool")
    parser.add_argument("tool")
    parser.add_argument("arguments", help="one JSON object")
    parser.add_argument("--socket", default=os.environ.get(
        "CODEACTION_TOOL_SOCKET", "/run/codeaction/public-tools.sock"))
    args = parser.parse_args(argv)
    try:
        arguments = json.loads(args.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("arguments JSON must be an object")
        result = call_public_tool(args.socket, args.tool, arguments)
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, separators=(",", ":")),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
