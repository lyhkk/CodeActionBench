"""Single source for code-composition limits and hard-stop effects."""

from copy import deepcopy

STDOUT_MAX_CHARS = 8000
MAX_PROGRAM_FILES = 16
MAX_PROGRAM_FILE_BYTES = 64 * 1024
MAX_PROGRAM_TOTAL_BYTES = 256 * 1024
MAX_PROGRAM_PATH_BYTES = 160
MAX_PROGRAM_PATH_PARTS = 4
DEFAULT_READ_CHUNK_BYTES = 8 * 1024
MAX_READ_CHUNK_BYTES = 16 * 1024
MODEL_VISIBLE_RESULT_MAX_BYTES = 32 * 1024
# Owned here (the composition contract delivers it to the model); the harness context
# policy reads this value as its default, not the other way around.
RUN_CODE_RESULT_MAX_IMAGES = 6


def declared_composition_contract(*, max_internal_tool_calls=500, timeout_s=180.0):
    """Return the effective limits/effects delivered before the first composition call."""
    return {
        "timeout_s": float(timeout_s),
        "stdout_max_chars": STDOUT_MAX_CHARS,
        "max_internal_tool_calls": int(max_internal_tool_calls),
        "model_visible_result_max_bytes": MODEL_VISIBLE_RESULT_MAX_BYTES,
        "run_code_result_max_images": RUN_CODE_RESULT_MAX_IMAGES,
        "program_workspace": {
            "max_files": MAX_PROGRAM_FILES,
            "max_file_bytes": MAX_PROGRAM_FILE_BYTES,
            "max_total_bytes": MAX_PROGRAM_TOTAL_BYTES,
            "max_path_bytes": MAX_PROGRAM_PATH_BYTES,
            "max_path_parts": MAX_PROGRAM_PATH_PARTS,
            "suffixes": [".py", ".md"],
            "default_read_chunk_bytes": DEFAULT_READ_CHUNK_BYTES,
            "max_read_chunk_bytes": MAX_READ_CHUNK_BYTES,
        },
        "reset_effects": {
            "ordinary_exception": {
                "namespace_reset": False, "stdout": "returned",
                "result": "returned_if_assigned", "virtual_files_kept": True,
            },
            "hard_stop": {
                "events": ["timeout", "child_death", "action_aborted"],
                "namespace_reset": True, "stdout": "empty", "result": "null_unassigned",
                "virtual_files_kept": True, "episode_active_after_action_abort": True,
                "interrupted_action": "returned_when_available",
            },
        },
    }


def model_composition_contract(contract, *, include_program_workspace=False):
    """Project implementation limits onto the composition tools actually delivered."""
    value = deepcopy(dict(contract))
    if include_program_workspace:
        return value
    value.pop("program_workspace", None)
    for effect in (value.get("reset_effects") or {}).values():
        if isinstance(effect, dict):
            effect.pop("virtual_files_kept", None)
    return value
