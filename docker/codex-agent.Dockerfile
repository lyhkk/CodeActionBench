# syntax=docker/dockerfile:1
# Codex vendor agent image. Like the first vendor seat, it deliberately contains no RoboTwin
# source, assets, task pack, or results: the ground-truth wall is the mount table, and this image
# is the half of it that is baked in.
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# The official release package for this exact tag, verified byte-for-byte before it is unpacked.
# Codex ships no apt or npm channel, so the pin IS the digest: an upstream retag would change the
# hash and fail the build rather than silently reseat the agent on a different CLI.
ARG CODEX_VERSION=0.154.0
ARG CODEX_SHA256=fc6e3e3b85f2cf7d664520ee5c66a7fe4aa12bae7d46834f47e2f165fd0d6f78
RUN set -eux; \
    url="https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/codex-package-x86_64-unknown-linux-musl.tar.gz"; \
    curl -fsSL -o /tmp/codex.tgz "$url"; \
    echo "${CODEX_SHA256}  /tmp/codex.tgz" | sha256sum -c -; \
    mkdir -p /opt/codex; \
    tar -xzf /tmp/codex.tgz -C /opt/codex; \
    rm -f /tmp/codex.tgz; \
    test -x /opt/codex/bin/codex; \
    # The code-mode host stays. gpt-6-astra declares tool_mode="code_mode_only" and the CLI
    # enforces it: with the host disabled the tool router fails closed and the model is handed NO
    # tools at all -- measured on the first live attempt ("Code Mode is unavailable because
    # code-mode host is disabled"). For this model the host is the only transport through which
    # ANY tool, including the benchmark's own run_code, can be called. The capability is
    # declared residual in clis.py and bounded by this container, not removed.
    test -x /opt/codex/bin/codex-code-mode-host; \
    ln -s /opt/codex/bin/codex /usr/local/bin/codex; \
    codex --version | grep -F "${CODEX_VERSION}"

RUN mkdir -p /workspace /usr/local/share/codeaction && chmod 0777 /workspace

ARG SOURCE_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="codeaction-codex-agent" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.codeaction.agent-cli="codex" \
      org.codeaction.agent-cli-version="${CODEX_VERSION}" \
      org.codeaction.reproducibility="pinned-dev"

ENV CODEACTION_CODEX_AUTH_DIR=/run/secrets/codeaction/codex-home \
    CODEACTION_CODEX_CONFIG=/usr/local/share/codeaction/codex_config.toml \
    CODEACTION_CODEX_STATE_DIR=/run/codeaction-codex-home \
    CODEACTION_SIM_HOST=sim \
    CODEACTION_SIM_PORT=8766
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint_codex_agent.sh"]
