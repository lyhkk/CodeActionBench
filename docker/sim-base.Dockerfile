# syntax=docker/dockerfile:1
# CodeActionBench simulator dependencies — PINNED DEVELOPMENT IMAGE (not yet a fully reproducible build: apt
# packages are unpinned and the wheel index is live; offline wheelhouse rebuild = Phase 5).
# Deps come from docker/requirements.lock (generated from the ACCEPTED bare-metal env's
# pip freeze); pytorch3d/curobo build from
# audited commits, mirroring script/bootstrap_a100_python.sh. Paths mirror bare-metal
# /Robotwin/* so mounts and recorded commands stay identical.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG MINIFORGE_VERSION=26.3.2-2
ARG MINIFORGE_SHA256=42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94
RUN curl -fL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
      -o /tmp/miniforge.sh \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum -c - \
    && bash /tmp/miniforge.sh -b -p /Robotwin/conda && rm /tmp/miniforge.sh
RUN /Robotwin/conda/bin/conda create -y -p /Robotwin/conda/envs/robotwin python=3.10.20 pip

ENV PY=/Robotwin/conda/envs/robotwin/bin/python
COPY docker/requirements.lock /tmp/requirements.lock
RUN $PY -m pip install --no-cache-dir -r /tmp/requirements.lock \
      --extra-index-url https://download.pytorch.org/whl/cu121

# No GPU at build time: build natively for A10 (sm_86) while retaining the audited sm_80
# compatibility artifact used by older images.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" FORCE_CUDA=1 MAX_JOBS=8
RUN git clone --filter=blob:none --no-checkout \
      https://github.com/facebookresearch/pytorch3d.git /Robotwin/builds/src/pytorch3d-0.7.8 \
    && git -C /Robotwin/builds/src/pytorch3d-0.7.8 -c advice.detachedHead=false \
      checkout --detach 75ebeeaea0908c5527e7b1e305fbc7681382db47 \
    && $PY -m pip install --no-build-isolation /Robotwin/builds/src/pytorch3d-0.7.8
RUN git clone --filter=blob:none --no-checkout \
      https://github.com/NVlabs/curobo.git /Robotwin/builds/src/curobo-0.7.8 \
    && git -C /Robotwin/builds/src/curobo-0.7.8 -c advice.detachedHead=false \
      checkout --detach d64c4b005459db10c5dd867d8b30a87d5bda9bdb \
    && $PY -m pip install -e /Robotwin/builds/src/curobo-0.7.8 --no-build-isolation
RUN $PY -m pip check

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04@sha256:8bbc6e304b193e84327fa30d93eea70ec0213b808239a46602a919a479a73b12
ENV DEBIAN_FRONTEND=noninteractive
# git: curobo is an editable install whose import runs setuptools_scm (git describe) against
# /Robotwin/builds/src/curobo-0.7.8 — same behavior as bare metal, which also has git.
# python3: system interpreter for stdlib helper scripts and the vendor launcher tests
# (bare metal has one; the conda env python stays the sim interpreter).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvulkan1 vulkan-tools libegl1 libgl1 libglvnd0 libglib2.0-0 \
      libxext6 libx11-6 libsm6 libxrender1 libgomp1 ca-certificates git python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /Robotwin/conda /Robotwin/conda
COPY --from=build /Robotwin/builds/src /Robotwin/builds/src

RUN chmod -R a+rX /Robotwin
