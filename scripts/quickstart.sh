#!/usr/bin/env bash
# Hello-world run & sweeps for gcfl-sim
# Usage:
#   bash scripts/quickstart.sh
# Env:
#   GCFL_BACKEND=reference|scale|ray|dask  (default: reference)
#   GCFL_WORKERS=<int>                     (used by 'scale' backend; default: 0)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND="${GCFL_BACKEND:-reference}"
WORKERS="${GCFL_WORKERS:-0}"

mkdir -p results/logs results/figures

echo "[1/4] Single run via gcfl-run (backend=${BACKEND}, workers=${WORKERS})"
python -m gcfl.run \
  -c configs/base.yaml \
  -o results/logs/run.parquet \
  --backend "${BACKEND}" \
  --workers "${WORKERS}"

echo "[2/4] Alpha–Pi sweep"
python -m gcfl.sweep \
  -c configs/sweeps/alpha_pi.yaml \
  -o results/logs/alpha_pi.parquet \
  --backend "${BACKEND}" \
  --workers "${WORKERS}"

echo "[3/4] Alpha–Phi boundary sweep"
python -m gcfl.sweep \
  -c configs/sweeps/boundary.yaml \
  -o results/logs/boundary.parquet \
  --backend "${BACKEND}" \
  --workers "${WORKERS}"

echo "[4/4] Making demo figures (if matplotlib is available)"
python scripts/make_figs.py \
  --glob "results/logs/*.parquet" \
  --outdir results/figures

echo
echo "Done."
echo "  Logs   → results/logs/"
echo "  Figures→ results/figures/"
