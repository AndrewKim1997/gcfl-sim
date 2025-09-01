# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Add more aggregator/mechanism plugins
- Ray/Dask docs polish
- Performance benchmarks and regression guards

## [0.1.0] - 2025-09-01
### Added
- Initial public release of `gcfl-sim`:
  - Deterministic reference backend and scalable backend skeleton
  - Modular plugins: aggregators (mean/median/trimmed/sorted_weighted), mechanisms (`u_orth_penalty`), signals (affine)
  - CLI entry points: `gcfl-run`, `gcfl-sweep`
  - Optional accelerations (`numba`, `pybind11` scaffolding) and distributed runners (Ray/Dask)
  - Minimal Dockerfiles (CPU / CUDA) and compose template
