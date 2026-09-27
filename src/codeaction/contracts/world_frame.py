"""THE single canonical world-frame source (spec §8). Prompt text, overlays, and eval all derive from
here — no axis strings hardcoded elsewhere. Sim-verified: +x=right, +y=forward, +z=up, meters.
NOTE: axis_bench is a SEPARATE VLM-test artifact with its own convention — unrelated, not sourced here."""

_AXES = {"+x": "right", "-x": "left", "+y": "forward", "-y": "backward", "+z": "up", "-z": "down"}


def get_world_frame() -> dict:
    prompt = ("World frame (meters): +x = robot's right, +y = robot forward, +z = up. "
              "In the head camera, +x is image-right. Coordinate-axis definitions do not "
              "determine the position or orientation of any scene surface.")
    return {"axes": dict(_AXES), "units": "meters", "prompt_text": prompt}
