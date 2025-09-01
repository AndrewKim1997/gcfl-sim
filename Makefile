# ====== gcfl-sim Makefile ======
PY ?= python
PIP ?= pip
PKG ?= gcfl
LOGS ?= results/logs
FIGS ?= results/figures

.PHONY: help setup dev lint format test run sweep bench clean clean-all docker-build-cpu docker-build-cuda docker-run-cpu docker-run-cuda

help:
	@echo "Targets:"
	@echo "  setup            Create venv and install (core)"
	@echo "  dev              Install with [dev] (and optionally FAST/DIST env flags)"
	@echo "  lint             Ruff lint"
	@echo "  format           Ruff format"
	@echo "  test             Run pytest"
	@echo "  run              Run a single experiment"
	@echo "  sweep            Run a parameter sweep"
	@echo "  bench            Run benchmark script (if present)"
	@echo "  docker-build-*   Build Docker images (cpu/cuda)"
	@echo "  docker-run-*     Run demo inside Docker"
	@echo "  clean            Remove caches and build"
	@echo "  clean-all        Also purge results/"

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && $(PIP) install -U pip && $(PIP) install -e .

# FAST=1 or DIST=1 to include extras
dev:
	. .venv/bin/activate || true; \
	extras="[dev]"; \
	if [ "$(FAST)" = "1" ] && [ "$(DIST)" = "1" ]; then extras="[dev,fast,dist]"; \
	elif [ "$(FAST)" = "1" ]; then extras="[dev,fast]"; \
	elif [ "$(DIST)" = "1" ]; then extras="[dev,dist]"; fi; \
	$(PIP) install -U pip && $(PIP) install -e .$$extras

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest -q

run:
	mkdir -p $(LOGS)
	$(PY) -m $(PKG).run --config configs/base.yaml --out $(LOGS)/demo.parquet || \
	$(PY) -m $(PKG).run --config configs/base.yaml --out $(LOGS)/demo.csv

sweep:
	mkdir -p $(LOGS)
	$(PY) -m $(PKG).sweep --config configs/sweeps/alpha_pi.yaml --out $(LOGS)/alpha_pi.parquet || \
	$(PY) -m $(PKG).sweep --config configs/sweeps/alpha_pi.yaml --out $(LOGS)/alpha_pi.csv

bench:
	@if [ -f scripts/benchmark.py ]; then \
		$(PY) scripts/benchmark.py; \
	else \
		echo "[bench] scripts/benchmark.py not found (skipping)"; \
	fi

docker-build-cpu:
	docker build -f docker/Dockerfile -t gcfl-sim:cpu .

docker-build-cuda:
	docker build -f docker/Dockerfile.cuda -t gcfl-sim:cuda .

docker-run-cpu: docker-build-cpu
	docker run --rm -v "$$PWD:/workspace" gcfl-sim:cpu \
		bash -lc 'python -m $(PKG).run --config configs/base.yaml --out results/logs/docker_demo.parquet'

docker-run-cuda: docker-build-cuda
	docker run --gpus all --rm -v "$$PWD:/workspace" gcfl-sim:cuda \
		bash -lc 'python -m $(PKG).run --config configs/base.yaml --out results/logs/docker_demo.parquet'

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info

clean-all: clean
	rm -rf results/logs/* results/figures/* results/cache/* 2>/dev/null || true
