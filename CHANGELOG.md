# Changelog
All notable changes to this project will be documented in this file.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning uses [SemVer](https://semver.org/).

## [Unreleased]
### Added
- Core project scaffolding (package layout, configs, scripts).
- Reference and scale backends (skeletons).
- Aggregates/mechanisms/signals plugin registries.
- Logging to CSV/Parquet with run metadata.
- CLI entry points: `gcfl-run`, `gcfl-sweep`.

### Changed
- N/A

### Fixed
- N/A

## [0.1.0] - 2025-09-01
### Added
- Initial public release of `gcfl-sim` with a lightweight simulator engine
  and optional extras: `fast` (Numba/pybind11) and `dist` (Ray/Dask).
