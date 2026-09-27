# CodeAction containers

Dependency images provide the simulator, Python libraries and vendor CLI binaries. Each new
experiment mounts frozen application code through role-specific read-only snapshots.

| Role | Purpose |
|---|---|
| `sim` | RoboTwin backend, task environment and verifier; resources and task files are read-only mounts. |
| `reference-agent` | Reference scaffold and provider adapters, with only its declared source components. |
| `claude-agent` | Pinned Claude Code CLI with an isolated session and the selected credential file. |
| `codex-agent` | Pinned Codex CLI with an isolated session and a copy of the selected login directory. |
| `gateway` | Relays the declared tool interface between the agent and simulator. |
| `fixture-agent` | Exercises image and tool transport without model credentials. |
| `sim-base` | Base simulator dependencies used when building the simulator image. |
| `scratch`, `scratch-launcher` | Optional code-first execution environment and launcher. |

Agents receive their role-specific components; simulator state, verifier implementation,
resources and task solutions remain outside their mounts. `compose.yml` defines these mounts.
`compose.dev.yml` provides an explicit source override for development.

After setup, run the free installation check on an idle GPU:

```bash
bash tools/reproduce.sh check --gpus 0 --background
```

Inspect its artifacts using the Watch and inspection commands printed at launch. To check
that an image manifest matches the dependency declarations:

```bash
bash tools/images.sh check docker/images.json --profile all
```

See [Versions and dependency images](../docs/release-packaging.md) for installation, rebuilding
changed dependencies and updating selected image roles.
