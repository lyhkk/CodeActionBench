# Released evaluation configurations

Use these configurations to reproduce the paper's 675-attempt evaluation in three batches:
seven reference models together, Claude Code, and Codex CLI. Each configuration selects
25 tasks × 3 attempts on their fixed primary scene seeds and preserves the published model,
reasoning and interface settings. Code and configuration are frozen automatically;
source, task and image identities are recorded by the runner. Select a release manifest and set
`require_release_match: true` to enforce a strict release match before launch.

| Config | Agent | Credentials |
|---|---|---|
| `claude-opus-5.yaml` | Reference / Claude Opus | Anthropic API |
| `claude-sonnet-5.yaml` | Reference / Claude Sonnet | Anthropic API |
| `gpt-5.6.yaml` | Reference / GPT | OpenAI API |
| `gemini-3.6-flash.yaml` | Reference / Gemini | Gemini API |
| `grok-4.6.yaml` | Reference / Grok | xAI API |
| `kimi-k3.yaml` | Reference / Kimi | `KIMI_KEY` and `KIMI_BASE_URL` |
| `qwen3.8-max.yaml` | Reference / Qwen | Qwen API |
| `claude-code-opus-5.yaml` | Claude Code / Opus | Claude subscription OAuth or Anthropic API |
| `codex-astra.yaml` | Codex CLI / Astra | Codex subscription login or OpenAI API |

After following [Quickstart](../../QUICKSTART.md#5-full-reproduction), select one queue.
Wait for it to finish before using the same GPUs for another queue.

Reference:

```bash
bash tools/reproduce.sh run reference-all --background
```

Claude Code:

```bash
bash tools/reproduce.sh run claude-code-opus-5 --background
```

Codex Astra:

```bash
bash tools/reproduce.sh run codex-astra --background
```

Each uses four GPUs by default. Use disjoint `--gpus` lists for simultaneous queues,
or wait for one to finish. The API queue serializes each model and shared credential;
vendors use the concurrency declared in `agents.json`, up to four lanes.

The wrapper copies a recipe to `configs/local/`, assigns a unique output directory and
prints exact watch/resume commands. Interrupted attempts can be requeued through
`bash tools/reproduce.sh control`; each execution is retained and reports select only
accepted executions. See [recovery](../../QUICKSTART.md#7-attention-and-recovery)
for pause, attention and resume commands.

For custom experiments, edit a recipe copy and use `codeaction eval --config` in your
installed Python environment. Relative paths resolve from that YAML's directory.
Dependency images resolve to immutable IDs at launch; ordinary code edits use new
source snapshots without rebuilding compatible environments.
