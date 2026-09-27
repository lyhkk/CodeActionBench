# Installation

Follow [QUICKSTART](../QUICKSTART.md) for the complete host, authentication, run and
recovery instructions. CodeActionBench includes its pinned RoboTwin backend source;
no second checkout or private image registry is needed.

The simulation host needs Linux x86-64, an NVIDIA GPU and driver, Docker Compose and
NVIDIA Container Toolkit. The reference hardware is four NVIDIA A10 GPUs. The containers provide CUDA
12.1.1 and simulator dependencies; the host CLI requires Python 3.10 or newer.

## Prepare the environment

From the source directory, choose one setup command:

```bash
# Option A: Python with venv support.
bash tools/setup.sh venv
```

```bash
# Option B: an existing Conda installation; also the fallback if ensurepip is missing.
bash tools/setup.sh conda
```

Setup creates `.codeaction-env/`, installs the CLI, creates missing private credential
templates, downloads and verifies the three pinned public resource archives, and installs
the dependency environments. Releases with `docker/images.json` pull fixed registry digests.
Source checkouts without that manifest build explicitly; `--build` selects this alternative
for any release. Builds reuse compatible images. Resource URLs and checksums
are in `tools/download_assets.sh`; image definitions and dependency locks are in `docker/`.
No shell activation or run-related exports are needed afterward.

`configs/local/images.json` records the installed image selection for new runs and vendor
authentication. Changing source, tasks or model settings does not require updating this file.
Pull failure is reported directly; it never triggers an unrequested build.

Reserve **120 GB free for setup and a few demos**, or **150 GB for a full evaluation and
a separate export**. ZIPs plus extracted resources occupy 31.5 GB; images, build cache,
temporary files and results need additional space. Check the Docker data disk as well
as the resource/output disk, and leave extra space when building images or retaining multiple runs.
The Quickstart covers alternate resource disks and disk-usage commands.

## Authenticate and run

Follow the [Gemini credentials and demo](../QUICKSTART.md#3-run-one-gemini-attempt-and-watch-it),
then select any of the [three reproduction queues](../QUICKSTART.md#5-full-reproduction):
reference, Claude Code or Codex. Setup, demos and full queues use the same environment
definitions. Ordinary launches do not run a full test suite or rebuild images.
Each queue includes its own authentication instructions. API keys and vendor login
files stay in `~/.config/codeaction/`, outside the source directory and experiment YAML.

For source and installation diagnostics, see [local checks](testing.md). Credential-free reference
replay is optional; see [reference replay](../oracle/README.md).
