# Local checks

Use the CPU checks for source changes and the free installation check for the container setup.

## Public CPU checks

After setup, from the source root:

```bash
# Check changes against a Git commit.
.codeaction-env/bin/python tools/check_changes.py --base HEAD

# Check the complete tree, including an extracted source archive without Git.
.codeaction-env/bin/python tools/check_changes.py --all
```

Replace `HEAD` with the intended comparison commit when reviewing committed changes.
The checks cover Python/shell syntax, local documentation links, component declarations and task
validity. They run without containers or model calls. Use a small task selection to evaluate
changes to runtime behavior.

## Free installation check

After [setup](../QUICKSTART.md), on an idle GPU:

```bash
bash tools/reproduce.sh check --gpus 0 --background
```

The command runs scripted tool calls through real containers without model credentials. Use
`--gpus 0,1,2,3` for four idle GPUs and the printed Watch command to follow completion. Inspect
its artifacts as described in [Quickstart](../QUICKSTART.md#4-read-the-result). Accepted execution
means valid evidence; scripted task outcomes are infrastructure checks, not model scores.

## Check an experiment

Use `--dry-run` to inspect the selected tasks, models and configuration before launching:

```bash
bash tools/reproduce.sh run --config experiment.yaml --dry-run
```

After the run, use the inspection and reporting commands in
[Quickstart](../QUICKSTART.md#4-read-the-result) to check its artifacts and verifier verdict.
A configuration that selects a provider model or vendor agent consumes that account's quota.
See [Development](development.md) for source changes and frozen run configuration.
