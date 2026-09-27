# Reference-sequence replay

Each released task provides `solution/oracle_calls.jsonl`, a fixed sequence of tool names,
literal arguments and recorded results for comparison. Replay executes the names and arguments
at the task card's primary scene seed; the host-side verifier scores the resulting state.
Recorded results are comparison data, not inputs to action selection.

After installing the CLI, select tasks in `options.tasks` in
[`configs/replay.example.yaml`](../configs/replay.example.yaml) and configure its resource path
and simulator image for your installation:

```bash
.codeaction-env/bin/codeaction replay --config configs/replay.example.yaml
```

Replay requires the simulator image, a GPU and installed resources, and uses no model API.
It is optional. Each invocation creates a new output directory under `runs/replay/`; an explicit
`out_dir` must not exist. `--dry-run` previews the selection. Other selected tasks continue after
a failure, and `selection.json` records their exit codes.

## Public source layout

```text
oracle/
  __init__.py          package marker
  replay.py            ledger loading, ordered dispatch, result comparison and CLI
  _host.py             recording, latch monitoring and out-of-band verification
  _tools.py            tool dispatch and serialized call logging
  transcript.py        render recorded calls and observations
  README.md            replay documentation

benchmark/tasks/<task>/solution/
  oracle_calls.jsonl   fixed calls and recorded results for comparison
  solve.sh             direct simulator-host replay entry
  README.md            task sequence instructions
```

Task data carries a canary marking it as excluded from training data.

## New replay outputs

Each replay writes fresh evidence outside the source solution directory:

- `replay_summary.json`: verifier verdict, ledger and task identities, call count and differences
  from recorded results.
- `result.json`: verifier, latch, safety and recording results.
- `oracle_calls.jsonl`: executed calls with the replay's actual results.
- `oracle_transcript.md` and `.jsonl`: readable calls and observations.
- `run_meta.json`, videos and captured images: runtime provenance and evidence.

The sequence is open-loop and specific to its scene. Planner variation can change results;
each replay records its own verifier verdict and deviations from the recorded results.

For a host with simulator dependencies and assets installed directly,
`bash tools/run_oracle.sh replay click_bell` provides the lower-level entry. The task's `solve.sh`
invokes that same entry; it requires the simulator Python environment, not just the host CLI.
