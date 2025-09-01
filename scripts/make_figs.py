#!/usr/bin/env python
"""
Make quick demo figures from gcfl-sim logs.
- Works with single-run logs and sweep logs (alpha–pi, alpha–phi).
- If matplotlib is not installed, exits gracefully with a notice.

Usage:
  python scripts/make_figs.py --glob "results/logs/*.parquet" --outdir results/figures
"""
from __future__ import annotations
import argparse, glob, os
import pandas as pd
import numpy as np

def _load_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)

def _ensure_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa
        return True
    except Exception:
        print("[figs] matplotlib not installed; skipping figure generation.")
        return False

def _plot_time_series(df: pd.DataFrame, out_png: str) -> bool:
    """Plot per-round mean±std of M across repeats."""
    if not {"round", "M"}.issubset(df.columns):
        return False
    import matplotlib.pyplot as plt
    g = df.groupby("round")["M"].agg(["mean", "std", "count"]).reset_index()
    x = g["round"].to_numpy()
    mu = g["mean"].to_numpy()
    sd = g["std"].fillna(0.0).to_numpy()
    lo, hi = mu - sd, mu + sd

    plt.figure(figsize=(7.0, 4.0))
    plt.plot(x, mu, label="M (mean)")
    plt.fill_between(x, lo, hi, alpha=0.2, label="±1 std")
    plt.xlabel("Round")
    plt.ylabel("M")
    plt.title("Monitoring signal over rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True

def _col(df: pd.DataFrame, name_alt1: str, name_alt2: str | None = None):
    for c in (name_alt1, name_alt2):
        if c is not None and c in df.columns:
            return c
    return None

def _plot_alpha_pi_heatmap(df: pd.DataFrame, out_png: str) -> bool:
    """Heatmap of M (or PoG) over α×π if present."""
    a_col = _col(df, "mechanism.alpha", "alpha")
    p_col = _col(df, "mechanism.pi", "pi")
    if a_col is None or p_col is None:
        return False
    import matplotlib.pyplot as plt
    # aggregate over repeats/rounds
    P = df.groupby([p_col, a_col])["M"].mean().reset_index()
    pivot = P.pivot(index=p_col, columns=a_col, values="M").sort_index().sort_index(axis=1)
    plt.figure(figsize=(6.4, 5.2))
    plt.imshow(pivot.values, aspect="auto", origin="lower",
               extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    plt.colorbar(label="M (mean)")
    plt.xlabel("alpha")
    plt.ylabel("pi")
    plt.title("M over (alpha, pi)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True

def _plot_boundary_frontier(df: pd.DataFrame, out_png: str) -> bool:
    """Sign(DeltaU) heatmap and φ*(α) frontier if (alpha, phi, DeltaU) grid exists."""
    a_col = _col(df, "mechanism.alpha", "alpha")
    f_col = _col(df, "mechanism.phi", "phi")
    if a_col is None or f_col is None or "DeltaU" not in df.columns:
        return False
    import matplotlib.pyplot as plt
    from gcfl.metrics import first_zero_crossing_frontier

    # Average DeltaU over repeats/rounds per grid cell
    G = df.groupby([f_col, a_col])["DeltaU"].mean().reset_index()
    A = np.sort(G[a_col].unique())
    F = np.sort(G[f_col].unique())
    mat = G.pivot(index=f_col, columns=a_col, values="DeltaU").reindex(index=F, columns=A).values

    # Frontier
    frontier = first_zero_crossing_frontier(A, F, mat)

    # Plot heatmap + frontier
    plt.figure(figsize=(6.8, 5.2))
    plt.imshow(mat, aspect="auto", origin="lower", extent=[A.min(), A.max(), F.min(), F.max()])
    plt.colorbar(label="DeltaU (mean)")
    plt.plot(frontier["alpha"], frontier["phi_star"], linewidth=2.0)
    plt.xlabel("alpha")
    plt.ylabel("phi")
    plt.title("sign(DeltaU) and φ*(α) frontier")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/logs/*.parquet")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args(argv)

    if not _ensure_mpl():
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[figs] no files match: {args.glob}")
        return 0

    for path in files:
        try:
            df = _load_any(path)
        except Exception as e:
            print(f"[figs] failed to load {path}: {e}")
            continue

        base = os.path.splitext(os.path.basename(path))[0]
        made_any = False
        # 1) Time series (single-run or any run with rounds)
        made_any |= _plot_time_series(df, os.path.join(args.outdir, f"{base}__M_over_rounds.png"))
        # 2) Alpha–Pi heatmap
        made_any |= _plot_alpha_pi_heatmap(df, os.path.join(args.outdir, f"{base}__alpha_pi_M.png"))
        # 3) Boundary frontier (alpha–phi, DeltaU)
        made_any |= _plot_boundary_frontier(df, os.path.join(args.outdir, f"{base}__boundary_frontier.png"))

        if made_any:
            print(f"[figs] wrote figures for {os.path.basename(path)}")
        else:
            print(f"[figs] no suitable columns in {os.path.basename(path)}; skipped.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
