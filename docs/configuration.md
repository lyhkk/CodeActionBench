# Run from a configuration file

After [installation](installation.md), start a new custom batch through the wrapper:

```bash
bash tools/reproduce.sh run --config configs/local-agent.example.yaml --dry-run
bash tools/reproduce.sh run --config configs/local-agent.example.yaml
```

This free example checks Python agent image/tool delivery without model credentials; it
does not solve the task. Install images and assets first. See [extensions](../EXTENDING.md)
for complete command-agent, model, provider, tool, verifier and task-pack examples.

## Wrapper defaults and resume

`run --config` is mutually exclusive with a built-in queue. Put GPU selection in YAML;
`--gpus` is reserved for built-in shortcuts. Both forms accept `--dry-run` and `--background`.
Precedence is explicit YAML > installation settings > defaults:

- `configs/local/reproduction.json` stores `assets_root`, `provider_env_file` and
  `provider_rate_limit_file`. Setup writes standard credential locations. To reuse another
  installation, edit these path strings manually; an optional `model_registry` path is also
  supported. Relative saved paths resolve from this JSON's directory. Store no credentials
  in this file.
- `configs/local/images.json` is the existing installed role-to-image selection. Only omitted
  image options inherit it. An explicit `release_manifest` selects its own images instead.
- Without saved values, resources default to `~/.cache/codeaction/assets`, credential paths
  to `~/.config/codeaction/`, and the registry to the built-in file. A custom batch defaults
  to GPU 0 and three attempts. Omitted `tasks` selects the chosen pack's registered tasks;
  the default official pack contains 25.

Explicit registry, image, GPU and task choices are preserved. The wrapper disables implicit
shell/home model overlays. It writes the resolved recipe to `configs/local/` and prints its
location, output path and resume command. Omit `out_dir` for a new timestamped directory
on every invocation. An explicit output must be fresh; existing directories or execution
records are rejected. `--dry-run` saves the resolved YAML and checks a CPU launch plan without
containers or model calls.

Use the printed saved recipe to launch or resume those settings:

```bash
bash tools/reproduce.sh resume configs/local/custom-<timestamp>.yaml --background
```

Replace `<timestamp>` with the printed filename; do not edit that saved recipe. Resume does
not reapply installation defaults or reread the original recipe. Once an execution record
exists, the runtime restores its original snapshot and images. Use `run --config` again
for a new experiment from edited source or configuration.

## Format and paths

`schema_version: 1` and `options` are the only top-level fields. Options use the CLI's primary
long flag with underscores: `--agent-mode` becomes `agent_mode`. `eval` uses `model` for its list
of agent labels, `profile` for the run tier, and `attempts` for attempts per task. `run` uses
`model` as a single string and `attempts` for the number of attempts.

Unknown fields, duplicate keys, wrong types, missing required options, and invalid choices fail
before execution. Relative paths, including `extensions`, resolve from the original YAML's
directory even after its resolved recipe is saved elsewhere. There is no shell expansion or
YAML inheritance. The wrapper's installation defaults are the only added merge layer.

The lower-level `.codeaction-env/bin/python -m codeaction.cli.main eval --config FILE`
remains available, as do `run` and `replay`. Direct commands retain CLI defaults and do not
merge setup settings. Do not mix their YAML with overrides other than `--dry-run`.

Use `.codeaction-env/bin/python -m codeaction.cli.main eval --help` for option descriptions. `matrix_arg`
forwarding is deliberately unavailable in YAML: configuration values must have named fields.
Provider files, vendor token files, image references and `default_credential_limit` are named
`eval` options. `assets_root` selects external resources and `release_manifest` selects the published image lock.
Advanced per-credential overrides currently remain in the matrix CLI.

## Model evaluation

Copy the run example and change `reference_model_mode` to `provider`, `model` to a registered
model ID, and `agent_label` to the name for this evaluated agent. Add `provider_env_file` and
`provider_rate_limit_file` pointing to your local credential and rate-limit files. Select a
`reasoning_profile` supported by that model. Keep credentials out of YAML.

For exact reproduction, set `require_release_match: true` and supply `release_manifest` with the
manifest for the baseline you want to reproduce. It pins source, assets and images. Setup saves
installed image digests in `configs/local/images.json`; the wrapper uses these as defaults.
Explicit YAML image options override them. Local development tags are resolved to content IDs
when a new experiment starts.

The single-run and batch artifacts already record execution identities, model configuration and
protocol settings. Preserve the YAML alongside the results as the user-authored launch recipe;
it does not replace the recorded evidence. Reading historical results does not require enabling
historical execution rules.

A reference model batch can use this wrapper configuration after filling the private provider
credentials and rate limits selected by setup:

```yaml
schema_version: 1
options:
  model: [gpt-5.6]
  attempts: 3
  gpus: [0]
  default_credential_limit: 1
```

Omitting `tasks` selects the whole chosen pack (25 official tasks, 75 episodes per agent).
This uses a paid provider; inspect it with
`bash tools/reproduce.sh run --config experiment.yaml --dry-run` first.
For exact reproduction set `require_release_match: true` and use a matching release manifest. The batch
runner freezes their identities and rejects incompatible resume attempts.

## Execution snapshots and resume

Every actual `run`, `eval` and `replay` freezes the current code and task inputs automatically,
with or without a release manifest. `extensions` is a list of local declaration files;
`model_registry` selects local non-secret model definitions. See [extensions](../EXTENDING.md).
Credentials remain outside snapshots. Resolved model definitions, endpoint URLs and account/rate
settings are frozen; credential contents can be rotated at their existing paths. Changing a
provider endpoint requires a new output directory. URLs must not contain authentication material.

`release_manifest` provides the comparison baseline and dependency image selection. Modified
code/tasks run as a distinct experiment by default. Add `require_release_match: true` only for
exact reproduction; it rejects differences before container launch. Both paths retain protocol,
isolation, budget and artifact checks. Submission eligibility is separate from task success.

The launch record beside an output directory points to its saved runtime and parameters.
For direct `eval`, reusing that output resumes the original batch, even after editing YAML.
The wrapper requires the explicit `resume` command instead.
Single `run` and `replay` outputs remain immutable; use a fresh output directory to retry them. New experiments
use a new output directory. Keep the source cache and pinned image IDs for future resume.
Missing/corrupt copies fail rather than silently using current code. Older saved batches delegate
to their own runtime. No clean Git checkout is required for experiments.

Direct CLI `--dry-run` prepares a temporary plan without Docker or persistent output; the
wrapper also saves its resolved YAML. Runtime packages are
read-only inside containers. Agent packages exclude simulator code, scoring and task solutions.
External mutable assets still receive full validation when using a release baseline.

## Optional reference replay

```bash
.codeaction-env/bin/python -m codeaction.cli.main replay --config configs/replay.example.yaml
```

Edit `options.tasks` to select one or several tasks. A new directory is created for each
invocation, so the same config can retry a task without overwriting evidence. Optional fields
are `gpu`, `out_dir` (must be new), `assets_root`, `sim_image`, and `release_manifest`.
Replay uses no model credentials and is not required before evaluation. A failed task is
reported and the remaining selected tasks still run. Occasional failures may be retried.

## Attempt count

Use `attempts` in YAML and `--attempts` on the command line for both `run` and `eval`.
The released evaluation default remains three attempts per task. A saved batch resumes its
original arguments and runtime; choose a new output directory to change its attempt count.
