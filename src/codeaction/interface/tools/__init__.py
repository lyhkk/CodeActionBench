"""The model-facing tool surface, assembled from four orthogonal capability families."""
from codeaction.interface.tools._base import (ORIENTATION_ANCHOR_ENV, ToolBoxBase,
                                              orientation_anchor_enabled)
from codeaction.interface.tools._base import (
    D0_TOOLS,
    checked_registry)
from codeaction.interface.tools.motion import MotionTools
from codeaction.interface.tools.poses import PoseTools
from codeaction.interface.tools.sensing import SensingTools
from codeaction.interface.tools.vision import VisionTools

__all__ = ["ToolBox", "ORIENTATION_ANCHOR_ENV", "orientation_anchor_enabled"]


class ToolBox(SensingTools, VisionTools, PoseTools, MotionTools, ToolBoxBase):
    """One episode's tool surface. Construct per attempt; expose to the runner via registry_map()."""
    def registry_map(self):
        """name → bound callable for every D0 registry tool implemented here (`done` and the two
        deferred draw variants are runner-level / not in v1).

        Every callable is wrapped in the declared argument contract, so the schema is enforced at
        execution rather than merely advertised to a provider. Direct dispatch and `run_code` both
        bind from here, which is what makes the two paths agree.
        """
        def guard(name, fn):
            def call(**kwargs):
                # A prior call that reached the threshold may not be followed by another tool,
                # even for a direct registry caller outside EpisodeRuntime.
                self._raise_if_episode_terminal()
                result = fn(**kwargs)
                self._raise_if_episode_terminal()
                return result
            call.__name__ = str(name)
            call.__doc__ = getattr(fn, "__doc__", None)
            call.__wrapped__ = fn
            return call

        guarded = {
            name: guard(name, getattr(self, name))
            for name in D0_TOOLS if callable(getattr(self, name, None))
        }
        from codeaction.extensions import declarations, entrypoint
        for name, entry in declarations("tool").items():
            implementation = entrypoint("tool", name, builtin=getattr(self, name, None))
            def bind(function):
                def invoke(**kwargs):
                    return function(self, **kwargs)
                return invoke
            guarded[name] = guard(name, bind(implementation))
        return checked_registry(guarded)
