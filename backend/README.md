# RoboTwin backend

`robotwin/` contains the simulator source snapshot used by CodeActionBench. The
upstream repository, exact commit, and SHA-256 of every included file are recorded
in `robotwin.lock.json`. The snapshot preserves the upstream MIT license.

The included `envs`, `task_config`, `script`, and `description` files are unchanged
from that commit. CodeAction-specific robot behavior and task adaptations live in
`src/codeaction/backends/robotwin/`. The backend's upstream expert methods are not
used to generate the published reference sequences or during agent evaluation.

Large resources are distributed separately. Containers mount them read-only at
`/opt/robotwin/assets`. Host-side backend development requires an `assets` directory
(or symlink) inside the backend checkout, because upstream asset loaders use
backend-relative paths. Normal container evaluation needs no host Python simulator
environment and no additional RoboTwin checkout.

To update the backend, select an upstream commit, replace the snapshot, regenerate
its lock, then run source-contract tests, the full container gate and task replays.
Do not overwrite the pinned source with an arbitrary installed RoboTwin version.
