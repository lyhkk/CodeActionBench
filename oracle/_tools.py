"""Dispatch a recorded call through the agent tool surface and log its result."""
from pathlib import Path

from codeaction.interface.schemas import serialize, to_json


class ToolSurfaceCaller:
    def __init__(self, toolbox, log_path=None):
        self._reg = toolbox.registry_map()
        self._log_path = Path(log_path) if log_path else None
        self.n_calls = 0
        self.calls = []

    def call(self, name, **kwargs):
        if name not in self._reg:
            raise KeyError(f"{name!r} is not on the agent tool surface")
        self.n_calls += 1
        payload = serialize(self._reg[name](**kwargs))
        status = payload.get("status") if isinstance(payload, dict) else None
        self.calls.append({"i": self.n_calls, "tool": name, "status": status})
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(to_json({"i": self.n_calls, "tool": name,
                                  "args": kwargs, "result": payload}) + "\n")
        return payload
