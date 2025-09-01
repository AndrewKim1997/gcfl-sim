# Contributing to gcfl-sim

Thank you for your interest in improving **gcfl-sim** — a lightweight, extensible simulator for gaming & cooperation in federated learning.  
This document explains how to set up a dev environment, make changes, and submit high-quality PRs.

---

## Quick start (development)

### 1) Environment
You can use either `venv + pip` or Conda. Python 3.10–3.12 is supported.

**venv + pip**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
# core + dev tools
pip install -e .[dev]
# optional: acceleration / distributed
# pip install -e .[fast]
# pip install -e .[dist]
# pip install -e .[fast,dist,dev]
````

**Conda**

```bash
conda env create -f environment.yml
conda activate gcfl-sim
```

### 2) Useful commands

With the provided `Makefile`:

```bash
make dev          # install with [dev] (use FAST=1 and/or DIST=1 to include extras)
make lint         # ruff lint
make format       # ruff format
make test         # pytest
make run          # single run → results/logs/demo.*
make sweep        # parameter sweep → results/logs/alpha_pi.*
make clean-all    # cleanup + remove results/*
```

> Results (logs, figures, caches) are **not** committed. Keep artifacts under `results/`.

---

## Project layout (high level)

```
src/gcfl/
  backends/            # reference (deterministic), scale (vectorized), dist backends
  aggregates/          # aggregator plugins (mean, median, trimmed, sorted_weighted, …)
  mechanisms/          # mechanism plugins (e.g., u_orth_penalty)
  signals/             # signal/noise models (e.g., affine)
  engine.py            # round loop: signals → aggregate → mechanism → update → log
  metrics.py           # M, PoG, PoC, DeltaU, …
  params.py            # config schema / validation / merging
  rng.py               # deterministic substreams (run/repeat/round/client)
  io.py                # logging to CSV/Parquet + metadata
  registry.py          # plugin registration / discovery
  run.py               # CLI: single experiment
  sweep.py             # CLI: parameter sweeps (local or distributed)
```

---

## Coding standards

* **Style & lint**: [ruff] is the source of truth (it also formats).
  Run `make lint` and `make format` before pushing.
* **Typing**: add type hints where practical; keep public APIs typed.
  Prefer `numpy.typing.NDArray` for arrays.
* **Docs & comments**: prefer short, precise docstrings (Google or NumPy style).
* **Imports**: standard lib → third-party → local; no relative imports across top-level packages.
* **No heavy globals**: pass configuration/contexts explicitly.

[ruff]: https://github.com/astral-sh/ruff

---

## Tests

* Use **pytest**. Put unit tests under `tests/unit/` and invariants/boundary tests under `tests/property/`.
* Keep tests **deterministic**:

  * Use `gcfl.rng` substreams for seeds (run/repeat/round/client).
  * The **reference backend** must be fully deterministic.
* If you add a new backend or a fast path, provide:

  * numerical **agreement tests**: `reference ↔ scale` within documented tolerances,
  * minimal **performance** checks (e.g., “no regression worse than X%” if a benchmark exists).
* CI runs lint + tests + a **small sweep** for smoke coverage.

---

## Plugins (aggregators / mechanisms / signals)

We welcome new plugins! Follow these rules:

1. Place your implementation in the right namespace (e.g., `src/gcfl/aggregates/my_method.py`).
2. Expose a thin, typed interface and register it in `gcfl.registry` or module-level `__all__`.
3. Provide unit tests for:

   * invariants (e.g., symmetry, monotonicity),
   * edge cases (ties, trimming boundaries, empty/NaN handling as applicable).

**Minimal aggregator example**

```python
# src/gcfl/aggregates/weighted_mean.py
from __future__ import annotations
import numpy as np
from typing import Optional

def aggregate(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Weighted mean with safe fallbacks."""
    v = np.asarray(values, dtype=float)
    if weights is None:
        return float(v.mean())
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    denom = w.sum()
    return float((v * w).sum() / denom) if denom != 0 else float(v.mean())
```

Add it to the registry (e.g., key `"weighted_mean"`) so it’s discoverable from configs/CLI.

---

## Configuration & schema

* Configuration files are YAML. Validation and defaults live in `gcfl.params`.
* When changing the schema:

  * keep backward compatibility or provide a clear migration path,
  * update `docs/CONFIGS.md`,
  * add a test that loads a representative config and exercises the new fields.

---

## Determinism policy

* The **reference backend** is the canonical source for numerical behavior.
* The **scale backend** may reorder operations but must match reference within documented tolerances (e.g., `rtol=1e-7`, `atol=1e-9` for double; specify in tests).
* Randomness is controlled by substreamed seeds (see `gcfl.rng`). Never use `np.random.*` globally.

---

## Performance policy

* Fast paths (Numba/pybind11) must be optional and **feature-equivalent** to Python paths.
* Provide graceful fallbacks if an accelerator is unavailable.
* Consider adding a simple benchmark to `scripts/benchmark.py` and a perf guard in CI when relevant.

---

## Submitting a PR

1. **Open an issue** first for larger changes to discuss design/fit.
2. Create a feature branch: `feat/<short-name>` or `fix/<short-name>`.
3. Follow **Conventional Commits** for messages:

   * `feat(aggregates): add weighted_mean`
   * `fix(engine): correct round update off-by-one`
   * `docs: clarify config schema`
4. Ensure:

   * `make lint` & `make test` pass locally,
   * docs updated if behavior changes (`docs/*.md`, README examples),
   * `CHANGELOG.md` updated under **\[Unreleased]**.
5. Open a PR with a clear description, motivation, and link to the issue.

**PR checklist**

* [ ] Tests cover the change (unit/invariant/perf if applicable)
* [ ] Lint & format pass
* [ ] Docs/config examples updated
* [ ] Changelog updated
* [ ] No large artifacts or `results/*` files committed

---

## Versioning & releases

* We use **Semantic Versioning** (MAJOR.MINOR.PATCH).
* Maintainers cut releases by updating `CHANGELOG.md`, tagging, and (optionally) building wheels and Docker images.

---

## Code of Conduct

We follow the principles of the Contributor Covenant. Be respectful and constructive.
(If a formal `CODE_OF_CONDUCT.md` is later added, this section will link to it.)

---

## Security

If you discover a security or integrity issue (e.g., a way to bypass determinism/constraints in unintended ways), please report privately to **[security@yourdomain.example](mailto:security@yourdomain.example)**. We will coordinate a fix before public disclosure.

---

## Licensing

By contributing, you agree that your code is licensed under the repository’s **MIT License** and that you have the right to submit it (no incompatible third-party code).

---

## Questions?

Open a GitHub Discussion or an Issue with the **“question”** label.
Thanks again for helping make **gcfl-sim** better!

```

If you’d like, I can tailor the email address, supported Python versions, tolerance defaults, or add a tiny PR template section as well.
```
