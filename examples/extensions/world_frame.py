"""Explicit replacement that preserves the existing world-frame tool contract."""


def read(toolbox):
    return toolbox.get_world_frame()
