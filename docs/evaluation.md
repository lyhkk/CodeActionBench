# Evaluation protocol and results

## Attempts and scenes

The released protocol uses **three attempts per agent and task**, on one fixed primary
scene seed. `codeaction eval` defaults to three attempts. Task cards declare the same count;
backup seeds are excluded from scored evaluation. A smaller development run is incomplete.

The task-pack version and digest identify this protocol.

## Runtime and comparison identity

All experiments record the code/task snapshot and immutable dependency image identities.
With `release_manifest`, the result records agreement or differences against that baseline.
`require_release_match: true` makes differences a pre-launch error; otherwise custom experiments
run and score normally. Runtime integrity, task verdict and official submission eligibility are
separate conclusions. Missing local-agent diagnostics remain unavailable, not zero.

Comparison identity includes the task, task pack, environment, tool and instruction surfaces,
verifier, agent/model configuration, budgets, and randomness protocol. Aggregate only compatible
identities. The reference scaffold and a vendor CLI are separate evaluated agents even when
they use the same model.

The task verifier computes success out of band. Budgets, safety findings, milestones, token
usage, latency, and tool-call counts provide additional measurements. `done` records an agent's
claim; the verifier determines the outcome.

## Artifacts

Each attempt records provenance, results, tool interactions, observations, and a sealed artifact
manifest. `turns.v1.json` presents the interaction by turn. Available reasoning text and provider
usage evidence retain their distinct fields; a token count does not imply recorded reasoning text.
The review video contains simulator-time camera frames. Agent reasoning and provider waiting
are not recorded in the video; available text remains in the trajectory record.

```bash
codeaction inspect /path/to/run
codeaction regrade /path/to/run-or-batch
```

`inspect` verifies the artifact manifest and protocol status. `regrade` reads existing records
without a simulator. Preserve original evidence and identify the verifier used for any new grade.

## Reference evidence

Each task publishes a fixed tool-call ledger and recorded verification evidence. Replay checks
that the sequence still solves the task on the selected runtime. It uses literal arguments from
the ledger and has no model or perception-query step. Motion-planner variation can change tool
results, so the replay records both result differences and the final verifier verdict.

A new replay carries the task-file and pack digests and the exact ledger hash. Admission
requires verification evidence when a task's card or instruction changes.

Task pack 0.34.0 uses `attempts_per_seed` and randomness protocol 3.0. Released cards
contain the task instructions, scenes, tool budgets, verifier predicates and seeds.
Each trial carries one `attempt_index`.

## Compare a run with a reference collection

Reference comparison is offline: it reads selected evidence and generates a local HTML report.
Choose the reference explicitly and inspect the available task, agent/model and configuration
groups first. From an installed environment:

```bash
python -m codeaction.reporting.comparison /path/to/new-batch \
  --reference /path/to/reference-collection --inventory-out /path/to/inventory.json
```

Inputs can be accepted batch directories, their `submission_manifest.v1.json` files, or a
supplied collection containing `measurements.jsonl` and matching `trajectories/` verdicts.
Repeat `--reference` for multiple reference sources; list multiple positional inputs for new
batches. Keep input order unchanged between inventory and report: source IDs are `candidate-1`,
`candidate-2`, and `reference-1`, `reference-2` in that order. The inventory lists selected and
requested task/model coverage, each recorded identity, and measurement availability.

Create a JSON match map by copying the exact `selector` objects of the groups you intend to
compare. This example illustrates the format; replace the task, agent and candidate key with
values from your inventory:

```json
{
  "schema": "codeaction-reference-matches.v1",
  "pairs": [
    {
      "candidate": {
        "source": "candidate-1",
        "task": "click_bell",
        "agent": "gemini-3.6-flash",
        "comparison_key": "COPY_THE_FULL_RECORDED_KEY_FROM_INVENTORY"
      },
      "reference": {
        "source": "reference-1",
        "task": "click_bell",
        "agent": "gemini-3.6-flash",
        "comparison_key": null
      }
    }
  ]
}
```

`null` selects an unknown execution identity in a supplied projection. A recorded batch
requires its exact key. The mapping cannot supply or override identity evidence. No task or
agent aliases are inferred. Unknown selectors, duplicate accepted trial selections, and reuse
of a group in more than one pair are rejected. Unmapped groups remain visible in the report.

```bash
python -m codeaction.reporting.comparison /path/to/new-batch \
  --reference /path/to/reference-collection --match-map /path/to/matches.json \
  --out /path/to/reference-comparison.html
```

Open the HTML locally; its download button saves the embedded
`codeaction-reference-comparison.v1` JSON. Place outputs outside the input evidence directories.
The ordinary summary command without `--reference` remains available with `--out`.

All differences are **candidate minus reference**. Success-rate differences use percentage
points. The report shows both full-selection observed differences and differences on common
attempt numbers, alongside unmatched attempts and requested attempts without accepted executions.
Resource/token differences use only common attempt numbers with that measurement on both sides;
their paired coverage is shown. Missing tokens or unknown requested/protocol coverage stay
unavailable. An absent attempt is never treated as a failed attempt.

Observed differences remain descriptive when identities differ or are unknown. A separate
verified-pair success difference requires equal complete recorded comparison and trial
identities, including scene, model and agent configuration. Conditions, model, configuration,
and trial identity each report `matched`, `mismatched`, or `unknown`, with differences and
missing fields listed. A known difference takes precedence over unknown evidence in the
overall status; no common attempt numbers leaves trial agreement unknown. Task and agent labels in the supplied
collection do not reconstruct its original execution identity, so mapping it to a new run
does not establish strict reproducibility. Pricing remains an API-equivalent estimate using
`--pricing` or the bundled price snapshot; token estimation and cost assumptions are retained
in the report's measurement details.
