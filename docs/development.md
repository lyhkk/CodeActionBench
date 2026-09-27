# Developing changes

A new experiment freezes source, task packages, extensions and non-secret model configuration.
Containers read their component-specific copies. Ordinary Python, task and verifier changes
reuse compatible dependency environments. Use a new output directory to evaluate each change;
no commit or manual hash editing is needed. Follow [Extensions](../EXTENDING.md) for custom
configuration and runnable examples.

## Check the changed components

After setup, from the source root:

```bash
.codeaction-env/bin/python tools/check_changes.py --base HEAD
```

Use another commit as the comparison base when reviewing a committed change. For an extracted
source archive, or to check the entire public tree:

```bash
.codeaction-env/bin/python tools/check_changes.py --all
```

These public CPU checks validate syntax, local documentation links, component declarations and
task packages without starting Docker or model calls. The component impact report identifies
affected areas. Exercise behavior changes with a small task selection and inspect its output
before expanding the experiment. See [Local checks](testing.md).

## Dependency changes and container checks

Rebuild only when system libraries, Python dependencies, vendor CLI binaries or the environment
launch contract change. Follow [image management](release-packaging.md) to build the affected
environments and refresh the installed selection. Existing batches retain their original images.
Dockerfile bytes contribute to environment identity, including comments.

After setup, a free installation check exercises real containers and tools on an idle GPU:

```bash
bash tools/reproduce.sh check --gpus 0 --background
```

Use its printed Watch command to follow completion and inspect accepted evidence. For behavior
changes, select the affected tasks or relevant container checks. [Reference replay](../oracle/README.md)
is optional and can select individual tasks. Model API or subscription runs require a separate,
explicit choice; the free check uses scripted calls.

## Existing runs and baseline comparison

Resuming a batch restores its saved snapshot, resolved configuration and pinned images. Later
edits do not change that batch. Single-run and replay outputs are immutable; new experiments
use new directories. Missing or damaged snapshots fail explicitly.

`release_manifest` selects a comparison baseline and its environments. By default, source or
task differences produce a custom experiment with recorded differences and normal task scoring.
`require_release_match: true` requires exact agreement before containers launch. Preserve the
baseline file and distinguish protocol or scoring changes in comparisons, including changes to
implementation that leave schemas unchanged. Without a selected registered baseline, a run
remains unregistered.

Historical results retain their recorded identities. Supported older batches resume through
their saved runtime and original images.

## Version an experiment

The image manifest identifies dependency environments; a benchmark manifest also records
source, tasks, scoring, configuration and resources. Use the [versioning guide](release-packaging.md)
to create a baseline from a committed source tree. Keep earlier manifests and existing run
snapshots so previous experiments remain traceable.
