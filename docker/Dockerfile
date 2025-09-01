# ---- Base: slim, fast, deterministic ----
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    # allow switching extras at build time: --build-arg EXTRAS="dev,fast,dist"
    GCFL_EXTRAS=""

# OS deps (build tools + common libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project (two-step copy for better layer caching if deps change rarely)
COPY pyproject.toml README.md ./ 
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY examples ./examples
COPY accel ./accel

# Results dirs (kept empty; bind-mounted usually)
RUN mkdir -p results/logs results/figures results/cache

# Install: core or with extras (fast/dist/dev) if provided
RUN python -m pip install -U pip wheel \
 && if [ -n "$GCFL_EXTRAS" ]; then \
        python -m pip install -e ".[${GCFL_EXTRAS}]"; \
    else \
        python -m pip install -e .; \
    fi

# Default command: quick sanity run (override in `docker run ...`)
CMD ["python", "-m", "gcfl.run", "-c", "configs/base.yaml", "-o", "results/logs/run.parquet"]
