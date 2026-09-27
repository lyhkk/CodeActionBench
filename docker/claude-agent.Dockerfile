# syntax=docker/dockerfile:1
# Vendor agent image. It deliberately contains no RoboTwin source, assets, or results.
FROM node:22.17.0-bookworm-slim@sha256:b04ce4ae4e95b522112c2e5c52f781471a5cbc3b594527bcddedee9bc48c03a0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3 \
    && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_CODE_VERSION=2.1.212
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && npm cache clean --force \
    && claude --version | grep -F "${CLAUDE_CODE_VERSION}"

RUN mkdir -p /workspace /usr/local/share/codeaction && chmod 0777 /workspace

ARG SOURCE_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="codeaction-claude-agent" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.codeaction.agent-cli="claude-code" \
      org.codeaction.agent-cli-version="${CLAUDE_CODE_VERSION}" \
      org.codeaction.reproducibility="pinned-dev"

ENV DISABLE_AUTOUPDATER=1 \
    CLAUDE_OAUTH_TOKEN_FILE=/run/secrets/codeaction/claude_oauth_token.sh \
    CODEACTION_SIM_HOST=sim \
    CODEACTION_SIM_PORT=8766
WORKDIR /workspace
USER node
ENTRYPOINT ["/usr/local/bin/entrypoint_agent.sh"]
