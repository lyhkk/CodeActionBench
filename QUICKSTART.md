# Quickstart

Use a Linux x86-64 GPU machine and a Bash terminal. Follow **1 → 2 → 3 → 4** for
one Gemini attempt and its results. Then choose a [full queue](#5-full-reproduction)
or use [recovery](#7-attention-and-recovery) when needed. Vendor-only users can go
from setup directly to their queue's authentication instructions.

Reserve **120 GB free for setup and demos**, **150 GB for the full evaluation and
a result export**, or **200 GB or more for several experiments**. Resources occupy
31.5 GB including ZIPs; allow about 30 GB for images plus build space, and 20–30 GB
for full-run outputs and an export. Allow additional space for source builds and retained
experiments. Browsing the supplied results alone needs about 230 MB after extraction.

## 1. Prepare the Linux host

Start in the root of the source checkout or extracted source archive, containing
`README.md` and `pyproject.toml`. All later commands run from that directory.

```bash
# Confirm this is the source root.
test -f pyproject.toml

# Confirm the NVIDIA driver is working; GPU indices are used in the YAML configs.
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

Use a working NVIDIA driver compatible with CUDA 12.1.1, which the images provide.
The reference environment uses driver 535.216.03. Host CUDA and SAPIEN installations
are not needed. Provision at least one GPU exclusively for this evaluation.

If Docker and NVIDIA Container Toolkit are already working, continue to the disk check
below. Otherwise, expand the first-time host instructions:

<details>
<summary>First-time host setup: Docker and GPU support on Ubuntu 22.04/24.04</summary>

Install once:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg unzip nano python3-venv

# Docker Engine and Compose, from Docker's package repository.
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

# Allow Docker containers to use the installed NVIDIA driver.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# The benchmark invokes Docker as your current user.
sudo usermod -aG docker "$(id -un)"
# Refresh group membership in a new shell; continue the guide in that shell.
newgrp docker
```

Docker group membership grants control of Docker on this host. The installation commands
follow the [Docker](https://docs.docker.com/engine/install/ubuntu/) and
[NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
package instructions. Verify access before downloading the benchmark resources:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If a utility is missing on Ubuntu, install it with
`sudo apt-get install -y curl unzip coreutils perl nano`.

</details>

Check free space before starting setup:

```bash
df -h
docker info --format 'Docker data directory: {{.DockerRootDir}}'
docker system df
```

Check the filesystem containing Docker's data directory, your resource directory and the
source checkout. They may be different disks. For a full run with separate disks, allow
90 GB free for Docker/builds and 60 GB for resources/results. `docker system df` reports
used Docker space across the whole host, including other projects; `df -h` reports free
space. Setup checks the required utilities and Docker access before downloading.

## 2. Download resources and install the images

This release includes `docker/images.json`; setup pulls the verified `v1.0.0` images
from Docker Hub's `codeactionbench` namespace using their fixed digests.
Use `--build` to request source construction even when a manifest is available.

Choose the resource destination **before setup**. Press Enter for
`~/.cache/codeaction/assets`, or enter an absolute directory on a larger disk.
This changes only resources; Docker uses its own data directory and results go
under this checkout's `demos/` or `runs/`.

```bash
read -r -p 'Resource directory (Enter for default): ' CODEACTION_ASSET_DIR
CODEACTION_ASSET_DIR=${CODEACTION_ASSET_DIR:-"$HOME/.cache/codeaction/assets"}
```

Now choose **one** environment method, in the same terminal.

**A — venv, with Python 3.10 or newer:**

```bash
bash tools/setup.sh venv --assets-root "$CODEACTION_ASSET_DIR"
```

**B — an existing Conda/Miniforge installation:**

```bash
if ! command -v conda >/dev/null; then
  read -r -p 'Conda installation directory: ' CODEACTION_CONDA_DIR
  source "$CODEACTION_CONDA_DIR/etc/profile.d/conda.sh"
fi
bash tools/setup.sh conda --assets-root "$CODEACTION_ASSET_DIR"
```

If venv reports missing `ensurepip`, use B or install `python3-venv`.
Setup creates `.codeaction-env/` and private credential templates, downloads and verifies
three resource ZIPs, and pulls the six runtime images or builds the seven environments
including their base. Wait for **Setup complete**. Installed image references are saved in
`configs/local/images.json` for runs and authentication. Source-build logs are in
`runs/_logs/build_<image>.log`; compatible images are reused. A failed pull reports the
failure and does not silently build instead.
No environment activation or exported run variables are needed afterward.

To choose the source-build alternative explicitly:

```bash
bash tools/setup.sh venv --build --assets-root "$CODEACTION_ASSET_DIR"
# With Conda instead:
# bash tools/setup.sh conda --build --assets-root "$CODEACTION_ASSET_DIR"
```

The downloads use public assets from [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).
Upstream licenses and attribution are retained; see [backend attribution](backend/README.md).
The fixed dataset revision and archive checksums are in `tools/download_assets.sh`;
downloads resume.

## 3. Run one Gemini attempt and watch it

Setup created private templates in `~/.config/codeaction/`. Fill the Gemini key locally:

```bash
nano "$HOME/.config/codeaction/provider.env"
```

Uncomment this line and fill its empty value. Use literal `NAME=value`, without
quotes or `export`; leave unused providers commented out.

```dotenv
GOOGLE_GEMINI_KEY=
```

Set the `google-gemini` entry to your account's request/token limits:

```bash
nano "$HOME/.config/codeaction/rate-limits.json"
```

`request_windows.limit` is requests per `period_s`; `token_windows.limit` is tokens
per window. The template's 10 requests and 100,000 tokens per minute are examples.
Overly low limits cause long **Quota Pacing Wait** periods. Keys stay in the private
file, outside YAML, command arguments and shared logs.

Start **Gemini × place_bread_skillet × one attempt**, on GPU 0. This uses API quota:

```bash
bash tools/reproduce.sh demo --background
```

The command prints **Config**, **Results**, **Resume** and **Watch** paths/commands.
Paste the Results path below, such as the printed `demos/gemini-3.6-flash-<timestamp>`:

```bash
read -r -p 'Batch directory: ' CODEACTION_BATCH_DIR
bash tools/reproduce.sh watch "$CODEACTION_BATCH_DIR"
```

For a completed demo, the watcher shows `accepted=1/1` and `running=0`. Acceptance
means the evidence is valid; the verifier may report success or failure.
Ctrl+C exits the watcher while leaving the batch running. If an attention appears,
use [section 7](#7-attention-and-recovery). Gemini attempts took a median 13 minutes
in the supplied records; account pacing can add time.

## 4. Read the result

After the attempt is accepted, generate its comparison page:

```bash
bash tools/reproduce.sh report "$CODEACTION_BATCH_DIR" \
  --out "$CODEACTION_BATCH_DIR/comparison.html"
.codeaction-env/bin/python -m json.tool \
  "$CODEACTION_BATCH_DIR/results/submission_results.v1.json" | less
```

Open the printed batch directory's `comparison.html` in a browser. On a remote GPU
host, copy the HTML file to your viewing machine; the report is self-contained.

```text
<batch>/
  batch_state.json                Queue state and accepted execution selection
  ATTENTION.md                    Problems needing intervention
  _monitor/                       Progress summaries
  results/submission_results.v1.json
  comparison.html                 Generated comparison report
  runs/<agent>/<task>/attempt-000/execution-001/run/
    summary.json
    attempt-000-seed-NNNNNN/
      result.json                 Verifier verdict and stop reason
      turns.v1.json               Text/thinking, tool calls and results
      tools/                      Observations and tool artifacts
      review.mp4                  Review video
      full.mp4                    Recorded video
      provenance.json             Code, task, configuration and image identities
      controller_status.json      Execution status
      filesystem_audit.json       Isolation checks
      artifact_manifest.v1.json   Evidence inventory
```

Videos contain simulator camera frames; available agent thinking is in `turns.v1.json`.
A requeue preserves `execution-001` and adds `execution-002`; reports select accepted
executions rather than counting every directory. Live attempts start at `000`;
the supplied trajectory collection uses `001`–`003`.

To check one run's evidence:

```bash
CODEACTION_RUN_DIR=$(find "$CODEACTION_BATCH_DIR/runs" -type d -name run -print -quit)
bash tools/reproduce.sh inspect "$CODEACTION_RUN_DIR"
```

## 5. Full reproduction

Choose **one queue at a time on the same GPUs**. Each queue defaults to GPUs
`0,1,2,3`; append `--gpus 0` for one GPU. Four selected GPUs only enable four vendor
workers when that account's `max_concurrency` is also four.

| Queue | Attempts | Four-GPU planning estimate |
|---|---:|---|
| [Reference](#5a-reference-api-models) | 525 | About 2–3 days |
| [Codex Astra](#5b-codex-astra) | 75 | About 3–5 active hours, plus quota waits |
| [Claude Code](#5c-claude-code) | 75 | About 8–12 active hours, plus quota waits |

These estimates use recorded episode durations and exclude setup. One-GPU serial totals
were about 204, 10.7 and 28.5 hours respectively. Astra had the highest supplied success
rate (55/75); queues can run in any order. Actual duration depends on task behavior,
concurrency, provider latency and account limits.

### 5A. Reference API models

Fill all six provider keys and the three required endpoint URLs:

```bash
nano "$HOME/.config/codeaction/provider.env"
```

```dotenv
ANTHROPIC_KEY=
OPENAI_KEY=
GOOGLE_GEMINI_KEY=
XAI_KEY=
XAI_BASE_URL=
KIMI_KEY=
KIMI_BASE_URL=
QWEN_KEY=
QWEN_BASE_URL=
```

Use the OpenAI-compatible endpoints supplied with the xAI, Kimi and Qwen keys.
The other providers use built-in endpoints. Kimi has one credential entry.

| Configurations | Rate-limit entry |
|---|---|
| `claude-opus-5`, `claude-sonnet-5` | `anthropic` |
| `gpt-5.6` | `openai` |
| `gemini-3.6-flash` | `google-gemini` |
| `grok-4.6` | `xai` |
| `kimi-k3` | `kimi` |
| `qwen3.8-max` | `qwen` |

Set each account's limits, then launch:

```bash
nano "$HOME/.config/codeaction/rate-limits.json"
bash tools/reproduce.sh run reference-all --background
```

The scheduler runs at most one active task per API model and credential alias.
Opus and Sonnet share a credential and serialize. Up to four different eligible
credential groups can run on four GPUs. To watch, use the command printed at launch.

### 5B. Codex Astra

Choose subscription login or OpenAI API authentication. Both use the pinned CLI.

**Subscription:**

```bash
bash tools/reproduce.sh auth codex
```

Open the printed URL and enter the one-time code. Device-code login must be enabled
for your account. The command writes `~/.config/codeaction/codex/auth.json`; keep it private.
[Codex authentication](https://developers.openai.com/codex/auth/).

Set the existing `codex-subscription` entry's `max_concurrency` to `4` if your account
supports four concurrent sessions; keep `1` for a single worker. Its `token_file` must
remain `~/.config/codeaction/codex`.

```bash
nano "$HOME/.config/codeaction/agents.json"
```

The recorded batch used about **80% of a Pro 5× weekly allowance**. Current allowance
accounting can differ, so a new batch may need more than one weekly window.

**OpenAI API instead of a subscription:** fill `OPENAI_KEY` in the private file,
then select API authentication:

```bash
nano "$HOME/.config/codeaction/provider.env"
bash tools/reproduce.sh auth codex --api
```

This saves API authentication in `~/.config/codeaction/codex-api/auth.json`, preserves
the subscription login, and selects API billing with one worker in `agents.json`.
The shortcut uses the default OpenAI endpoint and requires API access to `gpt-6-astra`.
Rerun it after rotating the key. To select the subscription again, use
`bash tools/reproduce.sh auth codex`. Authentication changes apply to new batches;
existing batches retain their saved account selection.

For the full 75-attempt queue:

```bash
bash tools/reproduce.sh run codex-astra --background
```

<details>
<summary>Optional: try one Astra attempt first</summary>

Run this instead of the full queue, on an idle GPU:

```bash
bash tools/reproduce.sh demo --model codex-astra --gpus 0 --background
```

Use the printed Watch command and [section 4](#4-read-the-result) to inspect it.

</details>

### 5C. Claude Code

Choose subscription OAuth or Anthropic API authentication, then launch the queue below.

**Subscription OAuth:** log in with the pinned CLI and save the resulting token locally:

```bash
bash tools/reproduce.sh auth claude
umask 077
touch "$HOME/.config/codeaction/claude_oauth_token.sh"
chmod 600 "$HOME/.config/codeaction/claude_oauth_token.sh"
nano "$HOME/.config/codeaction/claude_oauth_token.sh"
```

Follow the login URL in your browser. The token file must contain this assignment,
with the token filled after `=`:

```bash
export CLAUDE_CODE_OAUTH_TOKEN=
```

Edit only the `claude-subscription` entry in `agents.json`; preserve the Codex entry.
For a four-worker subscription batch, the Claude entry is:

```bash
nano "$HOME/.config/codeaction/agents.json"
```

```json
{
  "plan": "subscription",
  "max_concurrency": 4,
  "token_file": "~/.config/codeaction/claude_oauth_token.sh",
  "window": {
    "hours": 5,
    "stop_dispatch_above": 0.90,
    "resume_grace_s": 60
  }
}
```

Use `max_concurrency: 1` for one worker. The optional `window` policy pauses new dispatch
when the CLI reports at least 90% five-hour utilization, then resumes at the reported
reset plus 60 seconds (or waits five hours if no reset is reported). Weekly quota
rejections still need [recovery](#7-attention-and-recovery). Omit `window` to disable
this advance pause.

**Anthropic API instead of a subscription:** fill `ANTHROPIC_KEY` in the private file,
then select API authentication:

```bash
nano "$HOME/.config/codeaction/provider.env"
bash tools/reproduce.sh auth claude --api
```

This creates private `claude_api_key.sh` and updates only the Claude account to API
billing, concurrency one and no subscription window. Rerun after rotating the key.
An explicit `ANTHROPIC_BASE_URL` is preserved. Avoid using this queue and reference
Anthropic models concurrently with the same key; separate queues do not share a rate limiter.

**Launch after completing one authentication method:**

```bash
bash tools/reproduce.sh run claude-code-opus-5 --background
```

A subscription's **five-hour window is shared account allowance**, not a task time limit.
Four workers consume it faster; the full queue can span several windows or days, including
weekly waits. [Claude usage rules](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work).

<details>
<summary>Optional: small paid image/tool connection check</summary>

Run before the full queue, on an idle GPU:

```bash
bash tools/reproduce.sh check-vendor claude --gpu 0
```

This captures one image and calls `done` without manipulating the scene. Outputs go to
`demos/claude-transport-<timestamp>/`, including the CLI stream and image/tool checks.
It checks authentication and transport, not the task score.

</details>

### Queue outputs and concurrent experiments

Each launch prints a saved `configs/local/<queue>-<timestamp>.yaml`, a
`runs/<queue>-<timestamp>/` output directory, and Watch/Resume commands.
Use the printed Watch command, then [read results](#4-read-the-result).
Demo commands use `demos/`. Repeating `run` or `demo` creates a new experiment;
`resume` with the saved YAML continues the original one.

To run queues concurrently, assign **disjoint GPU lists** with `--gpus`:
for example reference `0,1`, Claude `2`, Astra `3`. Keep one simulator per GPU and
coordinate shared API accounts yourself. Batches do not share GPU or credential locks.
Each new batch freezes code, tasks, model settings and rate limits; credentials stay private.

## 6. Compare all results

Once your queues have accepted results, generate a single comparison file:

```bash
read -r -p 'Reference batch directory: ' CODEACTION_REFERENCE_DIR
read -r -p 'Claude Code batch directory: ' CODEACTION_CLAUDE_DIR
read -r -p 'Astra batch directory: ' CODEACTION_ASTRA_DIR
bash tools/reproduce.sh report \
  "$CODEACTION_REFERENCE_DIR" "$CODEACTION_CLAUDE_DIR" "$CODEACTION_ASTRA_DIR" \
  --out runs/comparison.html
```

The report shows success rates, task coverage, per-task scores, paired agent outcomes,
execution and physical time, tool counts, token coverage and API-equivalent cost estimates.
It validates accepted executions, avoids counting retries twice and separates different
evaluation conditions. Incomplete queues show their actual coverage.

For a separate result collection, regenerate its report after installing the source:

```bash
read -r -p 'Extracted result collection directory: ' CODEACTION_RESULTS_DIR
bash tools/reproduce.sh report "$CODEACTION_RESULTS_DIR" --out "$CODEACTION_RESULTS_DIR/comparison.html"
```

Missing diagnostics stay unavailable. Prices in `configs/analysis-pricing.json` are
dated analysis inputs, not subscription charges; pass `--pricing YOUR_PRICING.json`
to use another snapshot. Estimated vendor output is labeled. The report describes its
estimates, including zero cache usage assumed only for cost when cache counters are missing.
Missing input/output usage prevents pricing.

## 7. Attention and recovery

The controller monitors workers and timeouts. The Watch command is read-only: it shows
progress and attentions, and does not restart a stopped controller. Select a batch:

```bash
read -r -p 'Batch directory: ' CODEACTION_BATCH_DIR
bash tools/reproduce.sh control "$CODEACTION_BATCH_DIR" list
cat "$CODEACTION_BATCH_DIR/ATTENTION.md"
read -r -p 'Attention ID: ' CODEACTION_ATTENTION_ID
bash tools/reproduce.sh control "$CODEACTION_BATCH_DIR" show "$CODEACTION_ATTENTION_ID"
```

An attention can block an attempt, model or shared credential while other eligible work
continues. Restore quota or fix the reported cause, then requeue the interrupted attempt:

```bash
bash tools/reproduce.sh control "$CODEACTION_BATCH_DIR" requeue "$CODEACTION_ATTENTION_ID" \
  --note "cause resolved; retry interrupted attempt"
```

This submits a request for the controller's next poll. **If the controller stopped**,
restart it with the original Config path printed at launch:

```bash
read -r -p 'Saved YAML config path: ' CODEACTION_EVAL_CONFIG
bash tools/reproduce.sh resume "$CODEACTION_EVAL_CONFIG" --background
```

Requeue retains earlier executions and may spend model quota again. A verifier failure
with valid evidence is a result; use requeue for interrupted or invalid executions.

To pause new dispatch while allowing active attempts to finish:

```bash
bash tools/reproduce.sh control "$CODEACTION_BATCH_DIR" pause --note "operator pause"
```

To allow dispatch again:

```bash
bash tools/reproduce.sh control "$CODEACTION_BATCH_DIR" resume --note "continue evaluation"
```

To stop active attempts as well as new dispatch:

```bash
bash tools/reproduce.sh control "$CODEACTION_BATCH_DIR" stop --note "stop API spending"
```

The controller records which active attempts to interrupt, stops their agent containers,
and allows bounded evidence collection before cleanup. Stopped attempts require attention;
partial verifier values are not accepted as scores and there is no automatic retry.
Use `control ... list` to confirm the stop was applied and there are no running leases.
To run them again, resolve each `operator_stopped` attention with `requeue`, then resume
the queue. Prior execution directories are retained. These controls apply to batches
created with this runtime; saved older batches retain their original control capabilities.

These requests apply while the controller is running. Resume restores the original
snapshot and image IDs. Keep the batch, its adjacent hidden execution record and
snapshots together when backing up. New code, task or rate-limit changes require a new
experiment; credential renewal at the existing path works with resume.

Monitor storage during long runs:

```bash
du -sh "$CODEACTION_BATCH_DIR"
df -h .
docker system df
```

The original 675 episode directories occupy 7.2 GB, including 1.85 GB of videos.
Batch snapshots, retries and exports add space. Preserve images needed by unfinished
batches; a readable video/trajectory export alone is not a resumable batch.

## 8. Optional checks and image builds

**Free installation check:** run scripted attempts with real containers and tools on an idle
GPU, without model credentials:

```bash
bash tools/reproduce.sh check --gpus 0 --background
```

Use `--gpus 0,1,2,3` to exercise four idle GPUs. These scripted attempts check the
execution pipeline and artifact generation.
Use the printed Watch command and section 4 to inspect results.
For any `demo`, `run` or `check`, add `--dry-run` to create and preview a plan without
launching containers; the printed Resume command then starts that exact plan.

**Build images separately after setup:**

```bash
CODEACTION_PYTHON=.codeaction-env/bin/python bash tools/build_images.sh all
bash tools/images.sh capture
```

This reuses compatible images. Ordinary code/task edits use fresh source snapshots
without rebuilding dependency images. An API-only setup can use `--profile reference-mcp`;
the default setup pulls the six runtime roles, while `--build` builds all seven environments.

**CPU checks for source changes:** use `python tools/check_changes.py --base <commit>`
in a Git checkout or `python tools/check_changes.py --all` for a source archive.
See [testing](docs/testing.md) for scope and [extensions](EXTENDING.md) for custom
models, agents, tools, verifiers and tasks.
