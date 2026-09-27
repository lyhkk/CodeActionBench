# syntax=docker/dockerfile:1
ARG SIM_BASE_IMAGE=codeaction-sim-base:dev
FROM ${SIM_BASE_IMAGE}
ARG SOURCE_COMMIT=unknown
ARG LOCK_SHA=unknown
LABEL org.opencontainers.image.title="codeaction-sim" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.codeaction.lock_sha256="${LOCK_SHA}" \
      org.codeaction.reproducibility="pinned-dev"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics PYOPENGL_PLATFORM=egl \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/codeaction/src:/opt/codeaction \
    ROBOTWIN_ROOT=/opt/robotwin CODEACTION_CUROBO_BOUNDED_PLAN=1 CODEACTION_CUROBO_TABLE_WORLD=0
ENTRYPOINT ["bash", "/opt/codeaction/docker/entrypoint_sim.sh"]
