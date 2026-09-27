#!/usr/bin/env python3
"""Hold a supervised command until its parent durably records process identity."""
from __future__ import annotations

import os
import signal
import subprocess
import sys


LAUNCH_PROTOCOL = "pipe-gate-v1"
_RELEASE_BYTE = b"1"
_NOT_RELEASED_EXIT = 125


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) < 4 or values[0] != "--fd" or values[2] != "--":
        return _NOT_RELEASED_EXIT
    try:
        descriptor = int(values[1])
    except ValueError:
        return _NOT_RELEASED_EXIT
    command = values[3:]
    if descriptor < 0 or not command:
        return _NOT_RELEASED_EXIT
    try:
        released = os.read(descriptor, 1)
    except OSError:
        return _NOT_RELEASED_EXIT
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    if released != _RELEASE_BYTE:
        return _NOT_RELEASED_EXIT
    # The supervisor signals the whole process group.  Keeping this wrapper alive until the
    # protected command exits lets the parent prove whether SIGTERM drained the group or whether
    # it must escalate to SIGKILL.  A caught handler is reset to the default in the exec'd child.
    signal.signal(signal.SIGTERM, lambda _signum, _frame: None)
    try:
        process = subprocess.Popen(command)
    except OSError:
        return 126
    return int(process.wait())


if __name__ == "__main__":
    raise SystemExit(main())
