```
gcfl-sim/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ CHANGELOG.md
├─ CONTRIBUTING.md
├─ .gitignore
├─ .dockerignore
├─ pyproject.toml                 # extras: fast(numba/pybind11), dist(ray/dask)
├─ environment.yml                # conda alternative
├─ Makefile                       # build/run/test/bench shortcuts (optional)
│
├─ src/
│  └─ gcfl/                       # import name: gcfl  (independent from gcfl repro repo)
│     ├─ __init__.py
│     ├─ types.py                 # dataclasses / TypedDict schemas
│     ├─ rng.py                   # deterministic substreams (run/repeat/round/client)
│     ├─ params.py                # config validation & merge
│     ├─ engine.py                # round loop: signals→aggregate→mechanism→update→metrics
│     ├─ dynamics.py              # state updates / maps / trajectories (if used)
│     ├─ metrics.py               # M, PoG, PoC, DeltaU, etc.
│     ├─ io.py                    # CSV/Parquet logging + metadata
│     ├─ registry.py              # plugin discovery/registration
│     ├─ backends/                # execution backends
│     │  ├─ __init__.py
│     │  ├─ reference.py          # single-thread, deterministic
│     │  ├─ scale.py              # vectorized / multi-thread
│     │  ├─ ray_backend.py        # optional distributed
│     │  └─ dask_backend.py       # optional distributed
│     ├─ aggregates/              # aggregator plugins
│     │  ├─ __init__.py
│     │  ├─ mean.py
│     │  ├─ median.py
│     │  ├─ trimmed.py
│     │  └─ sorted_weighted.py
│     ├─ mechanisms/              # mechanism plugins (penalty/reward rules)
│     │  ├─ __init__.py
│     │  └─ u_orth_penalty.py
│     ├─ signals/                 # signal/noise models
│     │  ├─ __init__.py
│     │  └─ affine.py
│     ├─ utils/
│     │  ├─ profiling.py
│     │  ├─ logging.py
│     │  └─ version.py
│     ├─ run.py                   # CLI: single experiment
│     └─ sweep.py                 # CLI: parameter sweeps (local or distributed)
│
├─ accel/                         # optional acceleration
│  ├─ numba_kernels.py            # JIT kernels (fallback-safe)
│  └─ cpp/
│     ├─ CMakeLists.txt
│     ├─ fast_kernels.cpp         # hot loops (sorting/trim/loops)
│     └─ pybind_module.cpp        # pybind11 bindings
│
├─ configs/                       # reference configs (kept minimal)
│  ├─ base.yaml
│  └─ sweeps/
│     ├─ alpha_pi.yaml
│     └─ boundary.yaml
│
├─ scripts/
│  ├─ quickstart.sh               # hello world run/sweep
│  ├─ benchmark.py                # perf/scale bench
│  ├─ profile.sh                  # cProfile/py-spy helpers
│  └─ make_figs.py                # demo plotting from logs (optional)
│
├─ examples/
│  ├─ python_api.py               # minimal API usage
│  └─ plugins/                    # how-to write a plugin
│     ├─ my_aggregator.py
│     └─ README.md
│
├─ tests/
│  ├─ unit/
│  │  ├─ test_aggregates.py
│  │  ├─ test_mechanisms.py
│  │  ├─ test_engine.py
│  │  └─ test_rng.py
│  ├─ property/
│  │  └─ test_invariants.py       # invariants / boundary properties
│  └─ perf/
│     └─ test_ref_vs_scale.py     # tolerance between backends
│
├─ results/
│  ├─ logs/.gitkeep               # not committed; kept for local runs
│  ├─ figures/.gitkeep
│  └─ cache/.gitkeep
│
├─ docker/
│  ├─ Dockerfile                  # CPU base image
│  ├─ Dockerfile.cuda             # optional CUDA image
│  └─ docker-compose.yml          # optional Ray/Dask cluster
│
├─ docs/                          # lightweight developer/user docs (optional)
│  ├─ API.md
│  ├─ CONFIGS.md
│  ├─ PLUGINS.md
│  ├─ DISTRIBUTED.md
│  └─ REPRODUCIBILITY.md
│
└─ .github/
   └─ workflows/
      ├─ ci.yml                   # lint + unit/property + small sweep
      ├─ docker.yml               # build & push images (optional)
      └─ build_wheels.yml         # wheel build (optional)
```
