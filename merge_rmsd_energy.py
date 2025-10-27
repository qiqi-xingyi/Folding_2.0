# --*-- conding:utf-8 --*--
# @time:10/27/25 12:35
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:merge_rmsd_energy.py

# -*- coding: utf-8 -*-
"""
Merge RMSD table with per-conformation energies (matched by id==bitstring),
then compute correlations (Pearson, Spearman) and a simple linear fit.

Outputs under: e_results/1m7y/analysis_out/
  - merged_rmsd_energy.csv
  - energy_rmsd_correlations.csv
  - energy_rmsd_correlations.txt
  - plots/*.png (if matplotlib is available)

You can double-click to run (uses default paths), or run with CLI args, e.g.:
  python merge_rmsd_energy.py \
    --rmsd_tsv e_results/1m7y/best_from_csv/rmsd_table.tsv \
    --energies_jsonl e_results/1m7y/energies.jsonl \
    --out_dir e_results/1m7y/analysis_out
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ---------- defaults (edit if needed) ----------
DEFAULT_RMSD_PATH = os.path.join("e_results", "1m7y", "best_from_csv", "rmsd_table.tsv")
DEFAULT_ENE_PATH  = os.path.join("e_results", "1m7y", "energies.jsonl")
DEFAULT_OUT_DIR   = os.path.join("e_results", "1m7y", "analysis_out")


# ---------- utils ----------
def ensure_outdir(d: str):
    os.makedirs(d, exist_ok=True)


def read_rmsd_table(path: str) -> pd.DataFrame:
    """
    Expected cols: id, length, rmsd
    Auto-detects CSV vs TSV by extension; falls back to trying both.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    suf = os.path.splitext(path)[1].lower()
    if suf in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t")
    elif suf in [".csv"]:
        df = pd.read_csv(path)
    else:
        # try TSV then CSV
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path)
    # Normalize column names
    cols = {c.strip().lower(): c for c in df.columns}
    # Try to find id/rmsd
    id_col = None
    for cand in ["id", "bitstring"]:
        if cand in cols:
            id_col = cols[cand]
            break
    if id_col is None:
        raise ValueError("RMSD table must contain 'id' or 'bitstring' column.")
    if "rmsd" not in cols:
        raise ValueError("RMSD table must contain 'rmsd' column.")
    # Make light copy & type cast
    out = df.copy()
    out.rename(columns={id_col: "id"}, inplace=True)
    out["id"] = out["id"].astype(str)
    out["rmsd"] = pd.to_numeric(out["rmsd"], errors="coerce")
    return out


def read_energies_jsonl(path: str) -> pd.DataFrame:
    """
    Reads jsonl file with at least 'bitstring' and energy terms.
    Returns DataFrame with columns: bitstring, <energies...>
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
    if not rows:
        raise RuntimeError(f"No rows parsed from {path}")

    df = pd.DataFrame(rows)
    if "bitstring" not in df.columns:
        # try 'id'
        if "id" in df.columns:
            df.rename(columns={"id": "bitstring"}, inplace=True)
        else:
            raise ValueError("energies.jsonl must contain 'bitstring' key (or 'id').")

    # Keep only bitstring + energy columns
    energy_cols = [c for c in df.columns if c.startswith("E_") or c in ("E_total",)]
    keep = ["bitstring"] + energy_cols
    df2 = df[keep].copy()
    df2["bitstring"] = df2["bitstring"].astype(str)
    # cast energies to float when possible
    for c in energy_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(c)


def rankdata(a: np.ndarray) -> np.ndarray:
    """
    Return average ranks (1..N) for array a, ties receive the mean rank.
    Equivalent to scipy.stats.rankdata(method="average") but without SciPy.
    """
    a = np.asarray(a)
    n = a.size
    if n == 0:
        return np.array([], dtype=float)

    # stable sort so equal values keep relative order (not required but nice)
    sorter = np.argsort(a, kind="mergesort")
    a_sorted = a[sorter]

    # boundaries of runs of equal values
    # diff[k] == True at the start of a new value and at the end sentinel
    diff = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1], [True]))
    idx = np.flatnonzero(diff)  # e.g., [0, ..., n]

    # ranks in sorted order (1-based)
    ranks_sorted = np.empty(n, dtype=float)
    for i in range(len(idx) - 1):
        start = idx[i]
        end = idx[i + 1]  # slice is [start:end]
        # ranks for this run are start+1 .. end
        first = start + 1
        last = end
        avg = 0.5 * (first + last)
        ranks_sorted[start:end] = avg

    # invert sorting permutation
    inv_sorter = np.empty(n, dtype=int)
    inv_sorter[sorter] = np.arange(n)
    return ranks_sorted[inv_sorter]



def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return safe_pearson(rx, ry)


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    y ~ a*x + b
    """
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)


def maybe_plot_scatter(out_dir: str, x: np.ndarray, y: np.ndarray, name: str):
    if not _HAS_MPL:
        return
    a, b = linear_fit(x, y)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 120)
    ys = a * xs + b

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.plot(xs, ys)
    plt.xlabel(name)
    plt.ylabel("RMSD (Å)")
    plt.title(f"{name} vs RMSD\nslope={a:.3f}")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig_path = os.path.join(plot_dir, f"scatter_{name}_vs_RMSD.png")
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmsd_tsv", default=DEFAULT_RMSD_PATH, help="Path to rmsd table (tsv/csv)")
    parser.add_argument("--energies_jsonl", default=DEFAULT_ENE_PATH, help="Path to energies.jsonl")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--make_plots", action="store_true", help="Save scatter plots if matplotlib available")
    args = parser.parse_args()

    ensure_outdir(args.out_dir)

    # Read inputs
    rmsd_df = read_rmsd_table(args.rmsd_tsv)
    ene_df  = read_energies_jsonl(args.energies_jsonl)

    # Merge
    merged = rmsd_df.merge(ene_df, left_on="id", right_on="bitstring", how="left", validate="m:1")
    missing = merged["bitstring"].isna().sum()
    if missing > 0:
        print(f"[warn] {missing} rows in RMSD table were not matched in energies.jsonl")

    # Save merged CSV
    merged_csv = os.path.join(args.out_dir, "merged_rmsd_energy.csv")
    merged.to_csv(merged_csv, index=False)
    print(f"[ok] merged table written -> {merged_csv}")

    # Determine energy columns
    energy_cols = [c for c in merged.columns if c.startswith("E_") or c == "E_total"]
    if not energy_cols:
        raise RuntimeError("No energy columns found in merged table.")

    # Prepare arrays
    y = merged["rmsd"].to_numpy(dtype=float)
    mask = np.isfinite(y)
    results = []
    for col in energy_cols:
        x = merged[col].to_numpy(dtype=float)
        m = mask & np.isfinite(x)
        if m.sum() < 3:
            pear = float("nan"); spear = float("nan"); a = float("nan"); b = float("nan")
        else:
            pear = safe_pearson(x[m], y[m])
            spear = safe_spearman(x[m], y[m])
            a, b = linear_fit(x[m], y[m])

        results.append({
            "metric": col,
            "n": int(m.sum()),
            "pearson_r": pear,
            "spearman_rho": spear,
            "lin_slope": a,
            "lin_intercept": b,
            "missing": int(len(y) - m.sum()),
        })

        if args.make_plots and m.sum() >= 3:
            maybe_plot_scatter(args.out_dir, x[m], y[m], col)

    # Save correlations (CSV)
    corr_csv = os.path.join(args.out_dir, "energy_rmsd_correlations.csv")
    corr_df = pd.DataFrame(results)
    # Sort by absolute Pearson (desc) then absolute Spearman
    corr_df["abs_pearson"] = corr_df["pearson_r"].abs()
    corr_df["abs_spearman"] = corr_df["spearman_rho"].abs()
    corr_df.sort_values(["abs_pearson", "abs_spearman"], ascending=[False, False], inplace=True)
    corr_df.drop(columns=["abs_pearson", "abs_spearman"], inplace=True)
    corr_df.to_csv(corr_csv, index=False)
    print(f"[ok] correlations written -> {corr_csv}")

    # Write a human-readable TXT summary
    txt_path = os.path.join(args.out_dir, "energy_rmsd_correlations.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Correlation of energies vs RMSD (higher |r| => stronger relationship)\n")
        f.write(f"Rows in merged table: {len(merged)} (unmatched energies: {int(missing)})\n\n")
        for _, r in corr_df.iterrows():
            f.write(f"[{r['metric']}] n={int(r['n'])}, pearson={r['pearson_r']:.4f}, "
                    f"spearman={r['spearman_rho']:.4f}, slope={r['lin_slope']:.4f}, "
                    f"intercept={r['lin_intercept']:.4f}, missing={int(r['missing'])}\n")
        # Top-3 by |pearson|
        top = corr_df.head(3)["metric"].tolist()
        f.write("\nTop-3 by |Pearson r|: " + ", ".join(top) + "\n")
    print(f"[ok] summary written -> {txt_path}")

    print("\nDone.")

if __name__ == "__main__":
    # If launched by double-click in some IDEs, argv might be empty: we rely on defaults.
    main()
