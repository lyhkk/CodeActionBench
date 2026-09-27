"""Small reproduction shortcuts; evaluation, scheduling and recovery stay in codeaction eval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUEUES = ('reference-all', 'claude-code-opus-5', 'codex-astra')


def installed_images(root: Path = ROOT) -> dict[str, str]:
    path = root / 'configs/local/images.json'
    if not path.exists():
        return {}
    from codeaction.release import IMAGE_ARGS, PINNED_IMAGE
    images = json.loads(path.read_text()).get('images')
    if isinstance(images, dict) and 'agent' in images:
        raise ValueError('Installed images use the retired Claude role; rerun tools/setup.sh')
    if not isinstance(images, dict) or any(
            role not in IMAGE_ARGS or not isinstance(ref, str) or not PINNED_IMAGE.fullmatch(ref)
            for role, ref in images.items()):
        raise ValueError('Invalid installed image selection; rerun tools/setup.sh')
    return images


def auth_image(role: str) -> str:
    return installed_images().get(role, f'codeaction-{role}:dev')


def environment(root: Path) -> dict[str, str]:
    # A reproducible shortcut uses the declared files, not another checkout's shell overrides.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('CODEACTION_', 'BENCH_'))
           and key not in {'PYTHONPATH', 'PYTHONHOME', 'CUDA_VISIBLE_DEVICES'}}
    env.update(PYTHONPATH=str(root / 'src'), CODEACTION_ROOT=str(root),
               CODEACTION_MODEL_REGISTRY_EXTRA='', PYTHONDONTWRITEBYTECODE='1',
               CODEACTION_AGENTS_CONFIG=str(Path.home() / '.config/codeaction/agents.json'))
    return env


def gpu_list(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('use GPU indices such as 0 or 0,1,2,3') from exc
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('GPU indices must be distinct non-negative integers')
    return result


def prepare(args, root: Path = ROOT) -> tuple[Path, Path]:
    from codeaction.cli.config import load_options, options_argv, resolve_options
    from codeaction.cli.main import build_parser, _EVAL_LAUNCH_OPTIONS
    from codeaction.release import IMAGE_ARGS
    parser = build_parser()
    custom = args.command == 'run' and getattr(args, 'config', None) is not None
    if custom:
        options = load_options(parser, 'eval', args.config)
        label = 'custom'
    elif args.command == 'run':
        options = load_options(parser, 'eval', root / f'configs/reproduce/{args.queue}.yaml')
        # Queue files also serve the direct CLI. The shortcut supplies fresh output and
        # installed paths; only --config treats every recipe field as a user override.
        for key in ('out_dir', 'assets_root', 'model_registry',
                    'provider_env_file', 'provider_rate_limit_file'):
            options.pop(key, None)
        label = args.queue
    elif args.command == 'demo':
        label = args.model
        if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in label):
            raise ValueError('model label must be a path-safe name')
        options = {'model': [label], 'tasks': [args.task], 'attempts': 1}
    else:
        label = 'parallel-check'
        options = {'model': ['scripted'], 'tasks': [
            'click_bell', 'press_stapler', 'place_bread_skillet', 'lift_pot'], 'attempts': 1}
    defaults = dict(assets_root=str(Path.home() / '.cache/codeaction/assets'),
                    model_registry=str(root / 'src/codeaction/providers/models/registry.json'),
                    provider_env_file=str(Path.home() / '.config/codeaction/provider.env'),
                    provider_rate_limit_file=str(Path.home() / '.config/codeaction/rate-limits.json'))
    settings_path = root / 'configs/local/reproduction.json'
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        if not isinstance(settings, dict):
            raise ValueError('installation settings must be an object; rerun tools/setup.sh')
        saved = {key: settings[key] for key in defaults if key in settings}
        defaults.update(resolve_options(parser, 'eval', saved, settings_path.parent))
    # An explicit baseline owns its image selection. Otherwise use installed roles as defaults.
    if 'release_manifest' not in options:
        defaults.update({IMAGE_ARGS[role]: ref for role, ref in installed_images(root).items()
                         if IMAGE_ARGS[role] in _EVAL_LAUNCH_OPTIONS})
    defaults['gpus'] = [0] if custom or args.command == 'demo' else [0, 1, 2, 3]
    defaults['attempts'] = 3
    options = {**defaults, **options}
    if not custom and getattr(args, 'gpus', None) is not None:
        options['gpus'] = args.gpus
    options = resolve_options(parser, 'eval', options, root)
    if not Path(options['assets_root']).is_dir():
        raise ValueError('configured assets directory is missing; set assets_root or rerun tools/setup.sh')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(options.get('out_dir', root / ('runs' if args.command == 'run' else 'demos') / f'{label}-{stamp}'))
    record = output.parent / f'.{output.name}.execution.json'
    if output.exists() or record.exists():
        raise ValueError('output already exists; use resume with the saved config or choose a new out_dir')
    options['out_dir'] = str(output)
    parser.parse_args(options_argv(parser, 'eval', options))
    config = root / 'configs/local' / f'{label}-{stamp}.yaml'
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open('x') as stream:
        yaml.safe_dump({'schema_version': 1, 'options': options}, stream, sort_keys=False)
    return config, output


def run_eval(config: Path, *, background: bool, dry_run: bool, root: Path = ROOT) -> int:
    from codeaction.cli.config import load_options
    from codeaction.cli.main import build_parser
    config = config.expanduser().resolve()
    options = load_options(build_parser(), 'eval', config)
    output = (config.parent / Path(options['out_dir']).expanduser()).resolve()
    command = [sys.executable, '-m', 'codeaction.cli.main', 'eval', '--config', str(config)]
    env = environment(root)
    # Actual launch validates again; the preview spends no model quota and starts no containers.
    record = output.parent / f'.{output.name}.execution.json'
    if not record.exists():
        checked = subprocess.run([*command, '--dry-run'], cwd=root, env=env, check=False)
        if checked.returncode:
            return checked.returncode
    elif dry_run:
        print('Existing batch: launch restores and validates its saved snapshot and images.', flush=True)
    print(f'Config: {config.relative_to(root) if config.is_relative_to(root) else config}', flush=True)
    print(f'Results: {output.relative_to(root) if output.is_relative_to(root) else output}', flush=True)
    print('Resume: bash tools/reproduce.sh resume ' + shlex.quote(str(config)) + ' --background', flush=True)
    if dry_run:
        return 0
    if background:
        output.parent.mkdir(parents=True, exist_ok=True)
        console = output.with_name(output.name + '.console.log')
        # Append on resume; the controller gives every execution its own immutable directory.
        with console.open('a') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Controller PID: {process.pid}; console: {console}', flush=True)
        print('Watch: bash tools/reproduce.sh watch ' + shlex.quote(str(output)), flush=True)
        return 0
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


def configure_claude_api(directory: Path) -> None:
    from codeaction.agents.vendor.claude_auth import read_credentials
    from codeaction.providers.model_registry import read_credential_file
    values = read_credential_file(directory / 'provider.env')
    if not values.get('ANTHROPIC_KEY'):
        raise ValueError('Fill ANTHROPIC_KEY in ~/.config/codeaction/provider.env first')
    accounts_path = directory / 'agents.json'
    accounts = json.loads(accounts_path.read_text())
    credential = directory / 'claude_api_key.sh'
    selected = {'ANTHROPIC_API_KEY': values['ANTHROPIC_KEY']}
    if values.get('ANTHROPIC_BASE_URL'):
        selected['ANTHROPIC_BASE_URL'] = values['ANTHROPIC_BASE_URL']
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as temp:
        temporary = Path(temp.name)
        temp.write(''.join(f'{key}={shlex.quote(value)}\n' for key, value in selected.items()))
    try:
        read_credentials(temporary)
        temporary.replace(credential)
    finally:
        temporary.unlink(missing_ok=True)
    accounts['accounts']['claude-subscription'] = {
        'plan': 'api', 'max_concurrency': 1, 'token_file': str(credential)}
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as temp:
        temp.write(json.dumps(accounts, indent=2) + '\n')
        temporary = Path(temp.name)
    try:
        temporary.replace(accounts_path)
    finally:
        temporary.unlink(missing_ok=True)
    print('Claude Code now uses the selected Anthropic API credential, with one concurrent task. '
          'API calls are billed separately from subscriptions.', flush=True)


def select_codex_account(directory: Path, credential: Path, *, plan: str) -> None:
    accounts_path = directory / 'agents.json'
    accounts = json.loads(accounts_path.read_text())
    previous = accounts['accounts'].get('codex-subscription', {})
    accounts['accounts']['codex-subscription'] = {
        'plan': plan, 'token_file': str(credential),
        'max_concurrency': 1 if plan == 'api' else previous.get('max_concurrency', 1)}
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as temp:
        temp.write(json.dumps(accounts, indent=2) + '\n')
        temporary = Path(temp.name)
    try:
        temporary.replace(accounts_path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_codex_api(directory: Path) -> None:
    from codeaction.providers.model_registry import read_credential_file
    values = read_credential_file(directory / 'provider.env')
    key = values.get('OPENAI_KEY')
    if not key:
        raise ValueError('Fill OPENAI_KEY in ~/.config/codeaction/provider.env first')
    endpoint = values.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    if endpoint.rstrip('/') != 'https://api.openai.com/v1':
        raise ValueError('Codex API login requires the default OpenAI endpoint; '
                         'a custom OPENAI_BASE_URL is not supported by this shortcut')
    # Validate the account declaration before creating or replacing any credential.
    accounts = json.loads((directory / 'agents.json').read_text())
    if not isinstance(accounts.get('accounts'), dict):
        raise ValueError('agents.json must contain an accounts object')
    credential = directory / 'codex-api'
    with tempfile.TemporaryDirectory(prefix='.codex-api-', dir=directory) as temp:
        staging = Path(temp)
        result = subprocess.run(['docker', 'run', '--rm', '-i', '--network', 'none',
            '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'CODEX_HOME=/auth',
            '-v', f'{staging}:/auth', '--entrypoint', 'codex', auth_image('codex-agent'),
            'login', '--with-api-key', '-c', 'cli_auth_credentials_store="file"'],
            input=key + '\n', text=True, capture_output=True, check=False)
        if result.returncode:
            # Do not relay CLI output from a credential-handling command.
            raise ValueError('Codex API login failed; existing account settings were preserved')
        auth = staging / 'auth.json'
        try:
            saved = json.loads(auth.read_text())
        except (OSError, ValueError):
            raise ValueError('Codex API login did not create a valid auth.json') from None
        if not isinstance(saved, dict) or saved.get('OPENAI_API_KEY') != key or saved.get('tokens'):
            raise ValueError('Codex API login did not save exclusive API authentication')
        credential.mkdir(mode=0o700, exist_ok=True)
        credential.chmod(0o700)
        auth.chmod(0o600)
        auth.replace(credential / 'auth.json')
    select_codex_account(directory, credential, plan='api')
    print('Codex now uses the selected OpenAI API credential, with one concurrent task. '
          'The subscription login is preserved. API calls are billed separately.', flush=True)


def check_vendor(gpu: int, root: Path = ROOT) -> int:
    settings = json.loads((root / 'configs/local/reproduction.json').read_text())
    accounts = json.loads((Path.home() / '.config/codeaction/agents.json').read_text())
    credential = Path(accounts['accounts']['claude-subscription']['token_file']).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'demos' / f'claude-transport-{stamp}'
    config = root / 'configs/local' / f'claude-transport-{stamp}.yaml'
    options = dict(agent_mode='claude', interface_profile='vendor-mcp-direct',
        model='claude-opus-5', agent_label='claude-code-opus-5', reasoning_profile='high',
        task='click_bell', attempts=1, transport_conformance=True, gpu=gpu,
        assets_root=settings['assets_root'], token_file=str(credential), run_dir=str(output))
    from codeaction.release import IMAGE_ARGS
    options.update({IMAGE_ARGS[role]: ref for role, ref in installed_images(root).items()})
    with config.open('x') as stream:
        yaml.safe_dump({'schema_version': 1, 'options': options}, stream, sort_keys=False)
    print(f'Config: {config.relative_to(root)}\nResults: {output.relative_to(root)}', flush=True)
    print('One paid Claude Code image/tool check; it does not perform or score the task.', flush=True)
    return subprocess.run([sys.executable, '-m', 'codeaction.cli.main', 'run', '--config', str(config)],
                          cwd=root, env=environment(root), check=False).returncode


def authenticate(vendor: str, *, api: bool = False) -> int:
    if api:
        configure = configure_claude_api if vendor == 'claude' else configure_codex_api
        configure(Path.home() / '.config/codeaction')
        return 0
    if vendor == 'claude':
        print('After login, save export CLAUDE_CODE_OAUTH_TOKEN=... in '
              '~/.config/codeaction/claude_oauth_token.sh with mode 600. See QUICKSTART 5C.', flush=True)
        return subprocess.run(['docker', 'run', '--rm', '-it', '--entrypoint', 'claude',
                               auth_image('claude-agent'), 'setup-token'], check=False).returncode
    directory = Path.home() / '.config/codeaction/codex'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    result = subprocess.run(['docker', 'run', '--rm', '-it', '--user', f'{os.getuid()}:{os.getgid()}',
        '-e', 'CODEX_HOME=/auth', '-v', f'{directory}:/auth', '--entrypoint', 'codex',
        auth_image('codex-agent'), 'login', '--device-auth', '-c', 'cli_auth_credentials_store="file"'],
        check=False)
    if result.returncode == 0:
        (directory / 'auth.json').chmod(0o600)
        select_codex_account(directory.parent, directory, plan='subscription')
    return result.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('demo', 'run', 'check', 'resume'):
        command = sub.add_parser(name)
        command.add_argument('--background', action='store_true')
        command.add_argument('--dry-run', action='store_true')
        if name == 'resume':
            command.add_argument('config', type=Path)
        else:
            command.add_argument('--gpus', type=gpu_list)
        if name == 'demo':
            command.add_argument('--model', default='gemini-3.6-flash')
            command.add_argument('--task', default='place_bread_skillet')
        if name == 'run':
            command.add_argument('queue', choices=QUEUES, nargs='?')
            command.add_argument('--config', type=Path, help='start a new experiment from an eval YAML recipe')
    auth = sub.add_parser('auth')
    auth.add_argument('vendor', choices=('codex', 'claude'))
    auth.add_argument('--api', action='store_true',
                      help='use ANTHROPIC_KEY (Claude) or OPENAI_KEY (Codex) from private provider.env')
    vendor_check = sub.add_parser('check-vendor', help='one paid Claude Code image/tool transport check')
    vendor_check.add_argument('vendor', choices=('claude',))
    vendor_check.add_argument('--gpu', type=int, default=0)
    watch = sub.add_parser('watch')
    watch.add_argument('batch', type=Path)
    control = sub.add_parser('control')
    control.add_argument('arguments', nargs=argparse.REMAINDER)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('run', type=Path)
    report = sub.add_parser('report')
    from codeaction.reporting.comparison import add_arguments
    add_arguments(report)
    args = parser.parse_args(argv)
    if args.command == 'run':
        if bool(args.queue) == bool(args.config):
            parser.error('run requires either a queue or --config')
        if args.config is not None and args.gpus is not None:
            parser.error('put gpus in the custom YAML; --gpus is only for built-in queues')
    try:
        if args.command == 'auth':
            return authenticate(args.vendor, api=args.api)
        if args.command == 'check-vendor':
            return check_vendor(args.gpu)
        if args.command in ('watch', 'control', 'report', 'inspect'):
            module, arguments = {
                'watch': ('codeaction.reporting.batch_watch', [str(getattr(args, 'batch', '')), '--watch', '10']),
                'control': ('codeaction.batch.control', getattr(args, 'arguments', [])),
                'inspect': ('codeaction.cli.main', ['inspect', str(getattr(args, 'run', ''))]),
                'report': ('codeaction.reporting.comparison', [*map(str, getattr(args, 'inputs', []))]),
            }[args.command]
            if args.command == 'report':
                for name in ('out', 'pricing', 'match_map', 'inventory_out'):
                    value = getattr(args, name, None)
                    if value is not None:
                        arguments += ['--' + name.replace('_', '-'), str(value)]
                for reference in args.reference or []:
                    arguments += ['--reference', str(reference)]
            return subprocess.run([sys.executable, '-m', module, *arguments], env=environment(ROOT), check=False).returncode
        config = args.config if args.command == 'resume' else prepare(args)[0]
        return run_eval(config, background=args.background, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.exit(1, f'Reproduction: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
