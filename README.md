# <img src="docs/images/codeactionbench.svg" width="36" height="36" alt=""> CodeActionBench

[Project website](https://codeactionbench.org/) ·
[Full results & trajectories](https://huggingface.co/datasets/Yiheng-Lyu/CodeActionBench) ·
Paper (arXiv coming soon) · [Quickstart](QUICKSTART.md) · [Documentation](docs/README.md)

CodeActionBench evaluates how well general-purpose multimodal models turn visual
understanding and reasoning into embodied manipulation via executable code. The
benchmark contains 25 manipulation tasks that evaluate this capability through
agentic Code-as-Policy.

Without task-specific fine-tuning, demonstrations, external specialist perception
or grasp modules, privileged scene state, or predefined task policies, agents
should select visual evidence, form task-relevant 3D estimates, construct
manipulation targets, and iteratively execute and revise their policies.

A shared robot API provides RGB observations, calibrated geometric operations,
robot feedback, and bounded motion, leaving task-dependent decisions to the
evaluated agent. Fixed task instances, resource budgets, and a hidden
physical-outcome verifier support controlled comparisons across models and
harness configurations.

The included reference harness runs API models. Codex CLI and Claude Code connect
to the robot tools through MCP.

## Results and trajectories

All 675 attempts are on [Hugging Face](https://huggingface.co/datasets/Yiheng-Lyu/CodeActionBench):
25 tasks × 9 agent configurations × 3 attempts. The collection includes successes
and failures, with recorded model text, available reasoning, tool calls and results,
observation images, and a video for each attempt.

Download the complete dataset, extract it, and open `index.html` to explore the
trajectories offline. The smaller trajectories and results archive contains JSON
files and `results.csv` for analysis. You can also watch example runs on the
[project website](https://codeactionbench.org/).

## Getting started

You need Linux x86-64, an NVIDIA GPU, Docker Compose, NVIDIA Container Toolkit,
and Python 3.10+. Reserve 120 GB for setup and demos, or 150 GB for
a full evaluation and export. See [host setup](QUICKSTART.md#1-prepare-the-linux-host)
for prerequisites.

From the repository root, install the environment and check that the simulator
and robot tools work:

```bash
bash tools/setup.sh venv
bash tools/reproduce.sh check --gpus 0
```

Setup downloads the resources and pinned container images. If you use Conda,
replace the first command with `bash tools/setup.sh conda`. The check uses a GPU
but needs no model credentials or API quota.

To try a model on one task, follow the [Gemini demo](QUICKSTART.md#3-run-one-gemini-attempt-and-watch-it)
and [result viewing instructions](QUICKSTART.md#4-read-the-result). Model runs use
API or subscription quota. For all nine published configurations, see the
[full evaluation guide](QUICKSTART.md#5-full-reproduction).

## Documentation

| Guide | Contents |
|---|---|
| [Quickstart](QUICKSTART.md) | Installation, first run, full evaluation, reports, and recovery |
| [Evaluation configurations](configs/reproduce/README.md) | The nine published model and agent settings |
| [Evaluation protocol](docs/evaluation.md) | Attempts, scoring, recorded evidence, and result comparison |
| [Extensions](EXTENDING.md) | Add your own models, agents, tools, verifiers, and tasks |
| [Development](docs/development.md) | Source changes, experiment snapshots, and checks |
| [All documentation](docs/README.md) | Configuration, authentication, result export, and image management |

Task definitions are in [`benchmark/tasks/`](benchmark/tasks/), the runtime is in
[`src/codeaction/`](src/codeaction/), and runnable extension examples are in
[`examples/`](examples/).

## Citation

We'll add the paper link and BibTeX citation when the arXiv preprint is available.

## License and acknowledgments

The code uses the [MIT License](LICENSE). The
[released evaluation dataset](https://huggingface.co/datasets/Yiheng-Lyu/CodeActionBench)
is licensed under CC BY 4.0.

We thank the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) team for
open-sourcing the simulation platform and robot assets that CodeActionBench builds
on. See the [backend documentation](backend/README.md) for upstream licenses and
attribution.
