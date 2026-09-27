# syntax=docker/dockerfile:1
# Dependency environment; application code is supplied as a verified read-only snapshot.
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
RUN apt-get update && apt-get install -y --no-install-recommends strace && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /workspace /run/codeaction-public /run/codeaction-workspace && chmod 0777 /workspace /run/codeaction-public /run/codeaction-workspace
RUN ln -s /opt/codeaction/src/codeaction/cli/tool.py /usr/local/bin/codeaction-tool
ARG SOURCE_COMMIT=unknown
ARG REFERENCE_LOCK_SHA=unknown
LABEL org.opencontainers.image.title="codeaction-scratch" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.codeaction.reproducibility="pinned-dev"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/codeaction/src HOME=/tmp \
    CODEACTION_SIM_HOST=gateway CODEACTION_SIM_PORT=8765
WORKDIR /workspace
USER 65532:65532
ENTRYPOINT ["/bin/sh"]
