# Scoring API image.
#
# Multi-stage so the runtime layer carries no build toolchain. The application
# runs as a non-root user: a service that accepts order payloads from the
# internet should not be root inside its own container.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1     PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update  && apt-get install -y --no-install-recommends build-essential libgomp1  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv  && /opt/venv/bin/pip install --upgrade pip  && /opt/venv/bin/pip install .

# -----------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

# libgomp is required by LightGBM at runtime.
RUN apt-get update  && apt-get install -y --no-install-recommends libgomp1  && rm -rf /var/lib/apt/lists/*  && useradd --create-home --uid 10001 rto

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"     PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=rto:rto config ./config
COPY --chown=rto:rto migrations ./migrations
COPY --chown=rto:rto alembic.ini ./alembic.ini

USER rto
EXPOSE 8000

# The container is healthy when the process is up. Readiness - which requires a
# loaded model - is a separate endpoint, so an instance with no model is
# restarted by nobody but also receives no traffic from a readiness-aware proxy.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3     CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "rto_sentinel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
