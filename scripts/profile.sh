#!/usr/bin/env bash
# Simple profiling helpers for gcfl-sim.
# Requirements:
#   - cProfile: stdlib (always available)
#   - py-spy (optional): pip install py-spy

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTDIR="results/profile"
mkdir -p "$OUTDIR"

CFG="${1:-configs/base.yaml}"
OUT="${OUTDIR}/cprofile_run.pstats"

echo "[cProfile] Profiling gcfl.run → ${OUT}"
python -m cProfile -o "${OUT}" -m gcfl.run -c "${CFG}" -o results/logs/profile_run.parquet

echo "[tip] View with: snakeviz ${OUT}  # if installed"
echo

if command -v py-spy >/dev/null 2>&1; then
  echo "[py-spy] Recording top functions (10s)"
  py-spy top -- python -m gcfl.run -c "${CFG}" -o /dev/null || true
  echo "[py-spy] Saving a flame graph → ${OUTDIR}/flame.svg (10s)"
  py-spy record -o "${OUTDIR}/flame.svg" --duration 10 -- python -m gcfl.run -c "${CFG}" -o /dev/null || true
else
  echo "[py-spy] Not installed; skipping. Install with: pip install py-spy"
fi
