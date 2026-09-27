"""Read the selected Claude credential as data, then exec the isolated CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
from urllib.parse import urlsplit


def read_credentials(path: Path) -> tuple[str, dict[str, str]]:
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError('Claude credential file must be a regular file with mode 0600')
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError('Claude credential file is not readable text') from exc
    values: dict[str, str] = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(
            r'\s*(?:export\s+)?(CLAUDE_CODE_OAUTH_TOKEN|ANTHROPIC_API_KEY|ANTHROPIC_BASE_URL)=(.*)', line)
        if match is None or match[1] in values:
            raise ValueError('Claude credential file contains an unsupported or duplicate assignment')
        try:
            parts = shlex.split(match[2], posix=True)
        except ValueError as exc:
            raise ValueError('Claude credential file contains an invalid quoted value') from exc
        if len(parts) != 1 or not parts[0] or any(c.isspace() or ord(c) < 32 for c in parts[0]):
            raise ValueError('Claude credential values must be nonempty, without whitespace')
        values[match[1]] = parts[0]
    if set(values) == {'CLAUDE_CODE_OAUTH_TOKEN'}:
        return 'subscription_oauth', values
    if 'ANTHROPIC_API_KEY' in values and set(values) <= {'ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL'}:
        if 'ANTHROPIC_BASE_URL' in values:
            url = urlsplit(values['ANTHROPIC_BASE_URL'])
            if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('Anthropic endpoint must be an HTTP(S) URL without credentials, query or fragment')
        return 'anthropic_api', values
    raise ValueError('Choose exactly one Claude OAuth token or Anthropic API key; do not mix them')


def main() -> int:
    try:
        method, credentials = read_credentials(Path(os.environ['CLAUDE_OAUTH_TOKEN_FILE']))
        if len(sys.argv) < 2:
            raise ValueError('Claude executable is required')
    except (KeyError, ValueError) as exc:
        print(f'Claude authentication: {exc}', file=sys.stderr)
        return 2
    env = {key: value for key, value in os.environ.items() if key not in {
        'CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL'}}
    env.update(credentials)
    output = os.environ.get('CODEACTION_VENDOR_OUTPUT_DIR')
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'vendor_auth.json').write_text(json.dumps({'method': method}) + '\n')
    os.execvpe(sys.argv[1], sys.argv[1:], env)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
