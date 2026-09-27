"""Load CLI options from a strict, versioned YAML file using the CLI's own schema."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    pairs = loader.construct_pairs(node)
    result = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("configuration keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _option_actions(parser, command_name):
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {a.option_strings[0][2:].replace("-", "_"): a
            for a in commands.choices[command_name]._actions
            if a.option_strings and a.option_strings[0].startswith("--")}


def resolve_options(parser, command_name: str, options: dict, base_dir: Path) -> dict:
    """Validate explicit options and resolve paths before a recipe is copied elsewhere."""
    options = dict(options)
    if 'agent_image' in options:
        if 'claude_agent_image' in options:
            raise ValueError('use only claude_agent_image, not both image option names')
        options['claude_agent_image'] = options.pop('agent_image')
    actions = _option_actions(parser, command_name)
    resolved = {}
    for key, value in options.items():
        if key not in actions or key in {"help", "config", "matrix_arg"}:
            raise ValueError(f"unsupported option: {key}")
        action = actions[key]
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f"{key} must be a boolean")
            resolved[key] = value
            continue
        multiple = action.nargs in ("+", "*")
        values = value if multiple else [value]
        if not isinstance(values, list) or (not values and action.nargs != "*"):
            raise ValueError(f"{key} must be a non-empty list")
        items = []
        for item in values:
            expected = int if action.type is int else str
            if type(item) is not expected:
                raise ValueError(f"{key} requires {expected.__name__} values")
            if action.choices is not None and item not in action.choices:
                raise ValueError(f"{key} must be one of {list(action.choices)}")
            if action.type is Path:
                item = str((base_dir / Path(item).expanduser()).resolve())
            items.append(item)
        resolved[key] = items if multiple else items[0]
    return resolved


def load_options(parser, command_name: str, path: Path) -> dict:
    """Read the same strict YAML format for the CLI and reproduction wrapper."""
    path = path.expanduser().resolve()
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    if not isinstance(data, dict) or set(data) != {"schema_version", "options"}:
        raise ValueError("configuration requires exactly schema_version and options")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(data["options"], dict):
        raise ValueError("options must be a mapping")
    return resolve_options(parser, command_name, data["options"], path.parent)


def options_argv(parser, command_name: str, options: dict) -> list[str]:
    """Encode already validated options; argparse still checks required arguments."""
    actions = _option_actions(parser, command_name)
    result = [command_name]
    for key, value in options.items():
        flag = actions[key].option_strings[0]
        if isinstance(actions[key], argparse._StoreTrueAction):
            if value:
                result.append(flag)
        elif isinstance(value, list):
            if value:
                result.extend([flag, *map(str, value)])
        else:
            result.append(f"{flag}={value}")
    return result


def expand_config(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    """Translate explicit config values to arguments; argparse owns defaults and choices."""
    if not any(arg == "--config" or arg.startswith("--config=") for arg in argv):
        return argv
    try:
        if not argv or argv[0] not in ("run", "eval", "replay"):
            raise ValueError("--config is supported by run, eval and replay")
        probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        probe.add_argument("--config", type=Path, action="append", required=True)
        probe.add_argument("--dry-run", action="store_true")
        options, extra = probe.parse_known_args(argv[1:])
        if extra or len(options.config) != 1:
            raise ValueError("use one --config and optionally --dry-run; put other options in YAML")
        values = load_options(parser, argv[0], options.config[0])
        result = options_argv(parser, argv[0], values)
        if options.dry_run:
            result.append("--dry-run")
        return result
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"configuration: {exc}")
