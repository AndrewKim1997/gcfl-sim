# Example plugins for gcfl-sim

This folder demonstrates how to add **external plugins** (e.g., custom aggregators) without modifying the core package.

## How it works

- `gcfl-sim` discovers plugins through **registration decorators** in `gcfl.registry`.
- When Python imports your plugin module, its `@register_*(name)` decorator executes and **adds** the callable to the global registry under the chosen name.
- The engine/backends then resolve names (e.g., `aggregator.kind`) via the registry and call your code.

## Files

- `my_aggregator.py` — implements a simple `topk_mean` aggregator and registers it as `"topk_mean"`.

## Try it

1. Make this folder importable (choose one):
   - Run from repo root and prepend to `PYTHONPATH`:
     ```bash
     PYTHONPATH=examples python -m gcfl.run -c configs/base.yaml \
       --aggregator topk_mean --out results/logs/topk.parquet
     ```
   - Or add `examples/` to your IDE's source roots.
   - Or copy the plugin into `src/gcfl/aggregates/` and rename if you want to ship it as built-in.

2. Import the plugin **once** before running:
   ```python
   import examples.plugins.my_aggregator  # registers "topk_mean"

With the CLI, you can wrap the call:

```bash
python -c "import examples.plugins.my_aggregator; import gcfl.run as r; r.main(['-c','configs/base.yaml','--aggregator','topk_mean','-o','results/logs/topk.parquet'])"
```

3. Inspect results and compare:

   ```bash
   python scripts/make_figs.py --glob "results/logs/topk.*" --outdir results/figures
   ```

## Config knobs

Your plugin can accept arbitrary keyword arguments through the config:

```yaml
aggregator:
  kind: topk_mean
  k_frac: 0.25
```

They’ll be passed as `**kwargs` to your `aggregate(...)` function.

## Tips

* Keep pure-Python logic and be **robust to NaN/inf** (omit policy recommended).
* If you add heavy numeric kernels, make them optional and provide a NumPy fallback.
* Write a unit test under `tests/unit/` so CI validates your plugin’s behavior.

```
