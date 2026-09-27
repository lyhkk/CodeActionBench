# Exporting public results

A selected collection such as `release_675/` contains a `MANIFEST.json` with its episode membership
and both original and corrected verdicts. Export it to a separate directory:

```bash
.codeaction-env/bin/python -m codeaction.reporting.public_results /path/to/release_675 /path/to/release_675_public
```

Open the output's `index.html` directly in a browser. It provides filtering by task, agent,
outcome, and scoring version, and links each attempt to its decisions, calls, observations, and
review video. `summary.csv` and `index.json` provide the same selection for analysis.

Each episode contains `turns.v1.json`, `episode.json`, and `source.json`. The exporter regenerates
the turn projection from raw transcripts, checks coverage, verifies copied media, and records
input hashes. It refuses existing output directories and paths inside the sealed source tree.
It never adds files to historical attempts.

Original and corrected verdicts remain separate. Existing task-pack versions are retained.
Vendor stream events and reference model inferences have different turn boundaries; their turn
counts should not be interpreted as directly comparable inference counts.

The public projection replaces machine-specific absolute paths with `[local-path]`. Images and
review videos are copied with verified hashes. Ground-truth snapshots, credentials, and internal
recovery files are outside the export's file selection.

The public result format is `codeaction-public-results.v2`: the episode number is `attempt`
in JSON/CSV, the browser labels it “Attempt”, and directories use `attempt-NNN`. The exporter
reads historical membership manifests without changing them. `source.json` retains their
original episode locations and hashes so the exported view remains traceable to sealed evidence.

## Reference comparison inputs and reports

The offline [reference comparison workflow](evaluation.md#compare-a-run-with-a-reference-collection)
accepts validated batch selections or the supplied measurement format:

```text
reference-collection/
  measurements.jsonl
  trajectories/<task>/<agent>/attempt-NNN.json
```

Each measurement's task, agent, positive attempt number and boolean verifier verdict must agree
with its trajectory. These numeric projections do not carry a validated execution identity;
their comparison key, requested coverage and protocol coverage remain unknown. A match map
selects corresponding groups without filling in missing provenance. The `codeaction-public-results.v2`
browser export described above has a different layout and is not a comparison input: retain an
accepted batch selection or the supplied measurement collection for this workflow.

Generate `codeaction-comparison-inventory.v1` with `--reference` and `--inventory-out`, select
groups explicitly in a `codeaction-reference-matches.v1` map, then pass `--match-map` and `--out`.
The HTML contains a downloadable `codeaction-reference-comparison.v1` JSON document with source
coverage, selected identities, mapped and unmapped groups, condition mismatches, observed
differences and measurement availability. It does not combine anonymous projections with new
executions into one evaluation group. Batch inputs still use only the sealed execution accepted
by the controller; retries are not discovered by scanning run directories.
