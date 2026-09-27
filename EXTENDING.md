# Extending CodeActionBench

After setup, use `bash tools/reproduce.sh run --config experiment.yaml` for a new local
experiment. Explicit YAML values override installed resource/image/credential defaults.
Omit `out_dir` for a fresh timestamped directory on every invocation; an explicit directory
must be new. The wrapper prints the saved resolved YAML and its resume command. Relative paths
retain the original YAML's directory as their base. See [configuration](docs/configuration.md).

Each new output directory freezes code, tasks, selected extension files and model
configuration. Editing the checkout afterward does not change that experiment. No commit,
full test suite or reference replay is required to try a local change.

## Complete example recipes

Run any row with `bash tools/reproduce.sh run --config <configuration>`. Add `--dry-run`
to inspect its CPU launch plan first. The free examples need installed simulator resources
and GPU containers, but no model credentials. Each uses one task and one attempt.

| Configuration | Files under `examples/` | Expected behavior |
|---|---|---|
| `configs/local-agent.example.yaml` | `extensions/agent.yaml`, `local_agent.py` | Python agent receives RGB, reads robot state and calls `done`. |
| `configs/command-agent.example.yaml` | `extensions/command_agent.yaml`, `command_agent.py` | Child process exchanges context, calls and image responses over JSON lines. |
| `configs/local-tool.example.yaml` | `extensions/tool_agent.yaml`, `tool_agent.py`, `tool.yaml`, `tool.py` | Agent calls `echo_value` and checks its returned text. |
| `configs/replace-tool.example.yaml` | Tool example plus `extensions/replace-tool.yaml`, `world_frame.py` | Explicitly replaces `get_world_frame` while retaining its schema and behavior. |
| `configs/custom-task.example.yaml` | `tasks/`, `extensions/verifier.yaml`, `verifier.py`, Python agent | Runs only `click_bell_custom` with simulator-side `local_success` scoring. |
| `configs/local-model.example.yaml` | `extensions/models.json` | **Paid API:** alternate registry label for `qwen3.8-max` on the existing protocol. |
| `configs/local-provider.example.yaml` | `extensions/provider-models.json`, `provider.yaml`, `provider.py` | **Paid API:** selects the declared `local_provider` factory. |

Free agents check observations and lifecycle; they do not solve the manipulation task.
Expect accepted execution artifacts with tool/image records and a verifier verdict, which
may be false. Missing token/model-turn counts remain unavailable. The wrapper prints output
and resume paths; use its watch/inspect/report commands to read the results. CPU declaration
checks alone do not establish GPU/container execution.

## Add a model

Copy a compatible registry entry and explicitly set `model_registry` in YAML to the file
with its `models` mapping. Choose the model ID in `model`. The wrapper disables implicit
home-directory registry overlays. The two paid example recipes use the `qwen` credential
alias; configure `QWEN_KEY`, `QWEN_BASE_URL` and private account rate limits before running.
The resolved registry is frozen and delivered to the isolated agent container. API keys stay in
`provider_env_file`; never put them in model definitions. Set provider limits for your account.
Adding a model on an existing API protocol needs no image rebuild. New models use their declared
`default_reasoning_profile`; published agent recipes retain their existing reasoning settings.

For a new API protocol, use the provider declaration in `examples/extensions/provider.yaml`.
Its factory implements the existing `ProviderAdapter` interface and receives the same resolved
capabilities, request profile, credentials and rate policy as a built-in adapter. Set the model's
`protocol` to the declaration's name. New SDK dependencies require an updated agent environment.
The example delegates to the existing OpenAI-compatible adapter; running it consumes API quota.

## Add an agent

The smallest credential-free example is:

```bash
bash tools/reproduce.sh run --config configs/local-agent.example.yaml --dry-run
bash tools/reproduce.sh run --config configs/local-agent.example.yaml
```

A Python entry point receives `(context, client)`. `context` contains task text, public tool
schemas and an output directory. `client.call(name, arguments)` returns `result` and `images`
(the image MIME type and base64 bytes). Calls use the same server, validation, budgets and
termination protocol as built-in agents. Call `done` explicitly; returning without it records
`no_done`. The example checks image delivery and ends without claiming task success.

For a command agent, set `mode: command`; its Python factory receives `context` and returns an
argv list. The child receives one JSON context line on stdin, writes requests of the form
`{"tool": "capture_head", "arguments": {}}` to stdout, and reads one JSON response per request.
Use stderr for diagnostics. The command runs in the agent container, with a wall timeout.
Flush each request; non-JSON stdout is a protocol error. See the complete `command_agent.py`
example and its `mode: command` declaration.
Neither form receives the repository, scoring implementation, task solutions or simulator state.

Local agents are identified separately from the reference scaffold. Missing usage, model-turn
counts and reasoning summaries stay unavailable, not zero. To use an API credential, declare
`credential: <alias>` in the agent declaration and set `provider_env_file` in YAML. The
shared provider file is supported; only that alias is exported to the agent process environment. Its values are loaded only inside the agent process; local agent-specific
non-secret settings can be supplied under the declaration's `config` mapping. Task verdicts and artifact integrity
remain checked. Official submission qualification is a separate result. Vendor-specific native
CLI integrations keep their existing authentication and isolation adapters.

## Add tools or verifiers

Copy `examples/extensions/tool.yaml` or `verifier.yaml` and its Python file, then add the
manifest to YAML's `extensions` list. Paths inside a declaration are relative to that declaration.
Only explicitly listed `.py`/`.json` files are copied. Give each extension a distinct name;
replacing an existing name requires `replace: true` and is recorded as a changed experiment.
The replacement example preserves the original schema; changing implementation bytes alone
still changes experiment identity.

A tool handler receives `(toolbox, **arguments)`. It runs in the simulator and must declare
input and output schemas plus its return description. Direct MCP calls and `run_code` use the
same checked registry. Tool code is trusted simulator-side implementation, not an agent plugin;
keep it object-independent and expose only the intended public observations.

A verifier receives `(spec, *, env, **context)` and returns the usual verdict with boolean
`success`. Select its name in the task card's `verifier.kind`. It is never added to the tool list.
Missing evidence must raise an error rather than guess a verdict. Existing recorded evidence
can be regraded only when it contains the readings the new verifier needs.
The custom-task example wires the declaration into `task.json` through `verifier.kind` and
raises if the environment predicate fails or supplies no evidence.

Python implementation and task changes reuse the installed dependency environment. Change the
relevant environment only when system libraries, Python dependencies or a vendor CLI binary
change. Optional `requirements` entries are checked on extension load; nothing is auto-installed.
Additional host-file/network permissions are not silently granted by local declarations.

## Copy or add a task

Keep the official default 25-task registry unchanged by using a separate task pack. These
CPU-only commands copy the current card, assign a new ID, register it and validate its files:

```bash
.codeaction-env/bin/python examples/task_pack.py create \
  --source benchmark/tasks/click_bell --name my_task --out experiments/my-tasks
.codeaction-env/bin/python examples/task_pack.py validate experiments/my-tasks
```

The new directory contains `registry.json` and `my_task/{task.json,instruction.md}`; oracle
solutions are not copied. Validation prints the registered tasks and content digest for this
pack only. Keep directory names, registration and `task.name` consistent, retain schema
versions and canary markers, and use `instruction.md` as the only task wording source.

Save this as `experiments/my-task.yaml`:

```yaml
schema_version: 1
options:
  model: [local_example]
  tasks: [my_task]
  task_pack: ./my-tasks
  extensions: [../examples/extensions/agent.yaml]
  attempts: 1
  gpus: [0]
```

```bash
bash tools/reproduce.sh run --config experiments/my-task.yaml
```

The copy preserves scene, budgets and the three-attempt task protocol; `attempts: 1` selects
a demonstration. To add a different task, edit its instruction, scene and verifier, then
validate again. `scene.task_name` must name an implemented simulator environment; a new
physical scene needs that implementation as well as its task card. Add further card IDs to
the separate pack's registry and select them in YAML; no scheduler branch or fixed task
count is required. For a new predicate, copy the verifier declaration/Python file, list it
in `extensions`, and select its name in `verifier.kind` as the checked-in custom task does.

## Core changes and comparisons

Ordinary tool, budget and workflow Python edits enter new snapshots without rebuilding
dependency images. Existing batches retain their saved source/configuration/images. Budget
values are derived from expert-time metadata; arbitrary card budget edits are rejected.
If deliberately changing the budget policy, update its shared implementation and card values
together and record it as a changed protocol experiment. Start with a small task selection
and inspect tool outcomes, termination reasons and budget use before scaling up.

```bash
.codeaction-env/bin/python -m codeaction.cli.main changes --base HEAD
```

This reports affected components, tasks, dependency environments and public CPU checks. Use a new
output directory to try the change. A batch uses one execution-driver family; run local agents,
reference providers and different native vendor CLIs in separate output batches. To reproduce an exact published baseline, set
`release_manifest` and `require_release_match: true`; otherwise differences are recorded and the
experiment can run. See [development](docs/development.md) and [release packaging](docs/release-packaging.md).
