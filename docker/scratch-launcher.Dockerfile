# syntax=docker/dockerfile:1
# Dependency environment; application code is supplied as a verified read-only snapshot.
FROM docker:27.5.1-cli@sha256:851f91d241214e7c6db86513b270d58776379aacc5eb9c4a87e5b47115e3065c
RUN apk add --no-cache python3
RUN mkdir -p /workspace /run/codeaction-public /run/codeaction-workspace && chmod 0777 /workspace /run/codeaction-public /run/codeaction-workspace
ARG SOURCE_COMMIT=unknown
ARG REFERENCE_LOCK_SHA=unknown
LABEL org.opencontainers.image.title="codeaction-scratch-launcher" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.codeaction.reproducibility="pinned-dev"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/run/codeaction/code/src HOME=/tmp \
    CODEACTION_SIM_HOST=gateway CODEACTION_SIM_PORT=8765
WORKDIR /workspace
USER 65532:65532
ENTRYPOINT ["python3", "/run/codeaction/code/src/codeaction/runtime/scratch_launcher.py"]
