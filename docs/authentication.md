# Credentials and authentication

Start with the [Gemini demo](../QUICKSTART.md#3-run-one-gemini-attempt-and-watch-it),
or choose [a queue and its authentication steps](../QUICKSTART.md#5-full-reproduction).
This page is a reference for credential formats and existing host CLI installations.

Keep shareable experiment settings in `configs/reproduce/`. Keep credentials and local account
settings in `~/.config/codeaction/`. The reference agent uses provider API keys; Claude Code
supports either an Anthropic API key or subscription OAuth. The choice of agent remains
separate from the credential used to pay for its requests.

Create the local directory:

```bash
install -d -m 700 "$HOME/.config/codeaction"
```

Copy the templates once, without overwriting any existing configuration:

```bash
cp -n secrets/provider.env.example "$HOME/.config/codeaction/provider.env"
cp -n configs/provider-rate-limits.example.json "$HOME/.config/codeaction/rate-limits.json"
cp -n configs/agents.example.json "$HOME/.config/codeaction/agents.json"
chmod 600 "$HOME/.config/codeaction/provider.env" "$HOME/.config/codeaction/rate-limits.json" "$HOME/.config/codeaction/agents.json"
```

## Reference agent: API keys

Open `~/.config/codeaction/provider.env` in your editor and uncomment the key for the selected
provider. Values are literal, without shell quotes or `export`. Do not put keys into YAML or
command arguments. The credential parser accepts the names below, not the vendors' SDK variable
names such as `OPENAI_API_KEY`.

| Reproduction config | Key entry | Rate-limit alias |
|---|---|---|
| `claude-opus-5`, `claude-sonnet-5` | `ANTHROPIC_KEY` | `anthropic` |
| `gpt-5.6` | `OPENAI_KEY` | `openai` |
| `gemini-3.6-flash` | `GOOGLE_GEMINI_KEY` | `google-gemini` |
| `grok-4.6` | `XAI_KEY` | `xai` |
| `kimi-k3` | `KIMI_KEY` | `kimi` |
| `qwen3.8-max` | `QWEN_KEY` | `qwen` |

The last column names the credential aliases. Kimi, Qwen and xAI also require
`KIMI_BASE_URL`, `QWEN_BASE_URL` and `XAI_BASE_URL`, respectively, using the
OpenAI-compatible endpoint supplied with the key. Other providers use built-in endpoints
unless overridden. Model access depends on your provider account.

Edit `rate-limits.json` to match your account. The supplied values demonstrate the format and
are not a claim about provider quotas. Limits control dispatch and are recorded in batch identity.

## Claude Code: Anthropic API or subscription OAuth

With `ANTHROPIC_KEY` already filled in `provider.env`, configure API authentication:

```bash
bash tools/reproduce.sh auth claude --api
bash tools/reproduce.sh check-vendor claude --gpu 0
```

The first command copies only the selected Anthropic credential to the private
`claude_api_key.sh` and updates the local Claude account to API billing with concurrency
one. The second command spends model quota on one image/tool transport check; it is not
a benchmark score. Full task runs still use the ordinary `demo` and `run` commands.
API credentials are read as data, never executed as shell code. Each attempt records
`vendor/vendor_auth.json` with its authentication method and no secret values.

For subscription OAuth, use the pinned CLI:

In a normal terminal, run:

```bash
bash tools/reproduce.sh auth claude
```

Create `~/.config/codeaction/claude_oauth_token.sh` in your editor with one assignment:

```bash
export CLAUDE_CODE_OAUTH_TOKEN=
```

Then restrict access:

```bash
chmod 600 "$HOME/.config/codeaction/claude_oauth_token.sh"
```

The `claude-subscription` entry in `agents.json` points to this file. Adjust its
`max_concurrency` to the concurrency you intend to allow. Leave `ANTHROPIC_API_KEY` unset when
using subscription OAuth in your terminal; the benchmark container uses its isolated credentials.
Do not paste the token into logs or commit it. The benchmark does not perform the interactive login.

## Codex: isolated login directory

Codex uses a private directory containing `auth.json`. Choose one method with the pinned CLI:

```bash
bash tools/reproduce.sh auth codex
```

Or fill `OPENAI_KEY` in private `provider.env`, then select API billing:

```bash
bash tools/reproduce.sh auth codex --api
```

Subscription login uses `~/.config/codeaction/codex/`; API login uses
`~/.config/codeaction/codex-api/`. Each command selects that directory in the existing
`codex-subscription` account declaration. API setup sets `plan: api` and concurrency one,
and preserves the subscription credential. The key is passed through stdin to a network-disabled
login container; it is never put in argv or logs. The API shortcut supports the default
OpenAI endpoint. Existing batches retain their frozen account selection.

## Renew credentials or change accounts

When a Claude subscription token expires or is revoked, run `bash tools/reproduce.sh auth claude`
and replace the value in `~/.config/codeaction/claude_oauth_token.sh`; keep the file mode at `0600`.
For Codex, renew the subscription login with `bash tools/reproduce.sh auth codex`. Each attempt
uses an isolated copy of `auth.json`, so credentials refreshed inside an attempt are not saved
back to the host login directory. For API credentials, update the private provider file and rerun
the corresponding `auth --api` command.

Finish or stop active attempts before changing accounts. Start a new batch for a different
account; existing batches retain their configured credential paths and concurrency settings.

## Local settings versus reproduction settings

- YAML: agent selection, tasks, attempts, GPUs, output directory, image references and file paths.
- `provider.env`: API keys and optional endpoint URLs only.
- `rate-limits.json`: account-specific provider limits.
- `agents.json`: vendor account aliases, billing plan, concurrency and credential paths.
- Claude credential file / Codex directory: vendor authentication material only.

Do not share local credential files with results. Preserve the experiment YAML and the recorded
execution identities. A rerun reproduces the protocol and configuration; model sampling and
simulator execution can still change outcomes.
