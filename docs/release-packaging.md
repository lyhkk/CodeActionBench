# Versions and dependency images

Publish the source repository, dependency images on Docker Hub, and a versioned release
manifest. Results are exported as a separate collection. Resource installation uses the
fixed public RoboTwin archives in `tools/download_assets.sh`.

## Build dependency environments

From the repository root on a configured GPU host:

```bash
bash tools/setup.sh conda --build
```

To bypass Docker build cache:

```bash
bash tools/setup.sh conda --build --no-cache
```

The source-build route obtains dependencies from the upstream package sources declared in
the Dockerfiles and lock files. Published image digests identify exact environments for
reproduction; online package indexes can change the bytes produced by a later source build.

## Publish images

Create the repositories in your chosen Docker Hub namespace and enable immutable version
tags where available. The publication set is `codeaction-sim`, `codeaction-gateway`,
`codeaction-reference-agent`, `codeaction-fixture-agent`, `codeaction-claude-agent`,
`codeaction-codex-agent` and `codeaction-sim-base`. The base image supports downstream
builds; normal evaluation installation pulls the six runtime roles.

Authenticate on the GPU host. Keep credentials in Docker's credential store; never put
them in this repository or commands shared with users.

```bash
docker login
read -r -p 'Docker Hub namespace: ' CODEACTION_DOCKER_NAMESPACE
read -r -p 'New image version: ' CODEACTION_IMAGE_VERSION

# Inspect local images and print the intended uploads; no registry writes.
bash tools/images.sh publish --namespace "$CODEACTION_DOCKER_NAMESPACE" \
  --version "$CODEACTION_IMAGE_VERSION" --out docker/images.json

# Upload, resolve registry digests, pull back and verify IDs, then write the manifest.
bash tools/images.sh publish --namespace "$CODEACTION_DOCKER_NAMESPACE" \
  --version "$CODEACTION_IMAGE_VERSION" --out docker/images.json --push
```

The uploader rejects existing target tags and an existing output manifest. For the next
environment version, use a new output path such as `docker/images-v2.json`; review and
select the new manifest explicitly. A failed partial upload is not an accepted release.
To continue an interrupted upload, add `--resume`: existing tags are reused only when their
content IDs exactly match the local images; different content is refused. The final manifest
is created atomically after every selected image is verified and never replaces an existing
file.

For a dependency update affecting only selected roles, supply the previous manifest and
the roles to publish. For example, after updating and building the Claude environment:

```bash
bash tools/images.sh publish --namespace "$CODEACTION_DOCKER_NAMESPACE" \
  --version "$CODEACTION_IMAGE_VERSION" --base-manifest docker/images.json \
  --roles claude-agent --out docker/images-v2.json
```

Review the plan, then repeat it with `--push`; add `--resume` after an interrupted upload.
Use multiple names after `--roles` to update several environments. Only those roles need
local images or new tags. All other entries retain their original registry digest, image ID
and environment identity. The combined manifest must cover all seven published roles.
Both `--base-manifest` and `--roles` are required for an incremental publication; omitting
both retains full publication.

Every reused role must still match the current dependency declarations, including
`sim-base`. A simulator base or shared simulator lock change also affects `sim`; include
both roles when their declarations change. Selected local images must carry matching
environment labels. Dockerfile comments and ignore files are part of that identity, so
changing them can require a rebuild even when dependency versions stay the same.

Images stay on the GPU host and go directly to Docker Hub. They do not contain credentials
or experiment outputs. Application code is loaded through process-specific read-only
snapshots. Task, tool or verifier changes reuse the same environments when dependencies and
the launch interface remain compatible.

## Installation from published images

Commit the verified `docker/images.json` alongside matching dependency declarations.
Installation then selects registry images automatically:

```bash
bash tools/setup.sh venv
# Or:
bash tools/setup.sh conda
```

An explicit alternative manifest is supported:

```bash
bash tools/setup.sh venv --images-manifest docker/images-v2.json
```

The manifest is checked before the large resource download. Pulls verify environment
labels and image IDs; `configs/local/images.json` saves the exact selection for new
experiments and vendor authentication. Pull failure does not trigger a source build.
Use `--build` explicitly for that alternative. Existing batches keep their original IDs.

Profile installation is incremental. For example:

```bash
bash tools/images.sh pull docker/images.json --profile reference-mcp
bash tools/images.sh pull docker/images.json --profile vendor-mcp-direct
```

The second command keeps the installed reference agent while updating the roles in its
profile. `capture --profile` uses the same merge rule for locally built images. The selected
roles replace their previous entries only after the operation succeeds; other entries remain
unchanged. A failed pull, image check or selection write leaves the previous selection intact.
Already downloaded images and local Docker aliases may remain after a failure. Invalid
selection files are rejected before Docker operations rather than silently discarded.

## Create a benchmark manifest

A benchmark manifest records the committed source, tasks, model configuration, dependency
environments and resource identities. Generate the resource manifest for the installed assets,
then create a new benchmark version from the source checkout:

```bash
read -r -p 'Installed resources directory: ' CODEACTION_ASSET_DIR
.codeaction-env/bin/python -m codeaction.release assets "$CODEACTION_ASSET_DIR" .release/assets.json
.codeaction-env/bin/codeaction changes --base HEAD~1
.codeaction-env/bin/codeaction release prepare --version v1.0.0 --base HEAD~1
```

Preparation includes all six runtime roles by default. It uses `docker/images.json`
when present; `--images-manifest` selects another verified image set. Local development
without a registry manifest records installed content IDs. Individual `--image ROLE=REF`
selections take precedence over the automatically discovered manifest and replace the default
image set for specialized releases; include at least `sim` and `gateway`. Supplying both
`--image` and an explicit `--images-manifest` is an error.

The immutable benchmark manifest records source/task identities, environments, model
settings and resources. It neither builds nor uploads, refuses an existing version,
and requires committed source. Keep prior manifests and image digests available for
old runs. A new benchmark release does not require a new environment release unless
its dependency declarations or launch interface change.
