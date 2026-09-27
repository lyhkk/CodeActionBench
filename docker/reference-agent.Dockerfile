# syntax=docker/dockerfile:1
# Dependency environment; application code is supplied as a verified read-only snapshot.
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
COPY docker/reference-agent.requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock packaging
RUN mkdir -p /workspace /run/codeaction-public /run/codeaction-workspace && chmod 0777 /workspace /run/codeaction-public /run/codeaction-workspace
ARG SOURCE_COMMIT=unknown
ARG REFERENCE_LOCK_SHA=unknown
LABEL org.opencontainers.image.title="codeaction-reference-agent" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.codeaction.reproducibility="pinned-dev" \
      org.codeaction.agent-cli="codeaction-reference" \
      org.codeaction.agent-cli-version="0.16.0" \
      org.codeaction.reference-lock-sha256="${REFERENCE_LOCK_SHA}"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/run/codeaction/code/src HOME=/tmp \
    CODEACTION_SIM_HOST=gateway CODEACTION_SIM_PORT=8765
WORKDIR /workspace
USER 65532:65532
ENTRYPOINT ["python3", "/run/codeaction/code/src/codeaction/agents/reference/container_main.py"]
