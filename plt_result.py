# --*-- coding: utf-8 --*--
# @time: 2025/11/06
# @Author : Yuqi Zhang
# @File : plot_rmsd_results.py
#
# Description:
#   Read merged RMSD CSV and generate multiple publication-ready figures:
#     1) Grouped bar per pdb_id (QSAD/AF3/ColabFold/VQE)
#     2) Boxplots by method
#     3) ECDF curves by method
#     4) Scatter vs QSAD (with y=x reference)
#     5) Delta bar (method - QSAD) per pdb_id
#   Also writes summary statistics (mean/median/std/count/win_rate_vs_qsad).
#
# Notes:
#   - Uses matplotlib only, no seaborn.
#   - Does not force specific colors; lets matplotlib choose defaults.
#   - Handles missing values gracefully.
#   - If too many pdb_ids, use --top_n to limit the grouped bar plot.

import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_COLUMNS = ["qsad_rmsd", "af3_rmsd", "colabfold_rmsd", "vqe_rmsd"]
METHOD_LABELS = {
    "qsad_rmsd": "QSAD",
    "af3_rmsd": "AF3",
    "colabfold_rmsd": "ColabFold",
    "vqe_rmsd": "VQE",
}


def ecdf_values(x: np.ndarray):
    """Return ECDF x_sorted, y for a 1D array (drop NaNs)."""
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/median/std/count & win_rate_vs_qsad (≤ QSAD) per method."""
    stats = []
    for col in METHOD_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        s = {
            "method": METHOD_LABELS[col],
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(ddof=1),
            "count": series.count(),
        }
        if col != "qsad_rmsd":
            paired = df[["qsad_rmsd", col]].copy()
            paired = paired.apply(pd.to_numeric, errors="coerce")
            paired = paired.dropna()
            if len(paired) > 0:
                win_rate = (paired[col] <= paired["qsad_rmsd"]).mean()
            else:
                win_rate = np.nan
            s["win_rate_vs_qsad"] = win_rate
        else:
            s["win_rate_vs_qsad"] = np.nan
        stats.append(s)
    return pd.DataFrame(stats)


def plot_grouped_bar(df: pd.DataFrame, out_dir: Path, top_n: int = 0):
    """Grouped bar for per-pdb_id comparison across methods."""
    data = df.copy()
    # Optionally limit number of examples for readability
    if top_n and top_n > 0:
        data = data.head(top_n)

    pdb_ids = data["pdb_id"].astype(str).tolist()
    n = len(pdb_ids)
    if n == 0:
        print("[WARN] No rows to plot for grouped bar.")
        return

    # Prepare matrix values aligned to METHOD_COLUMNS
    vals = []
    for col in METHOD_COLUMNS:
        vals.append(pd.to_numeric(data[col], errors="coerce").to_numpy())
    vals = np.vstack(vals)  # shape: (4, n)

    # Bar layout
    x = np.arange(n)
    m = len(METHOD_COLUMNS)
    width = min(0.8 / m, 0.2)

    fig = plt.figure(figsize=(max(10, n * 0.5), 6))
    for i, col in enumerate(METHOD_COLUMNS):
        shift = (i - (m - 1) / 2) * (width + 0.02)
        plt.bar(x + shift, vals[i], width=width, label=METHOD_LABELS[col])

    plt.xticks(x, pdb_ids, rotation=90)
    plt.ylabel("RMSD (Å)")
    plt.title("RMSD per PDB (Grouped Bar)")
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / "bar_grouped_per_pdb.png"
    out_pdf = out_dir / "bar_grouped_per_pdb.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def plot_boxplots(df: pd.DataFrame, out_dir: Path):
    """Boxplots per method."""
    data = [pd.to_numeric(df[col], errors="coerce").dropna().values for col in METHOD_COLUMNS]
    labels = [METHOD_LABELS[c] for c in METHOD_COLUMNS]

    if all(len(d) == 0 for d in data):
        print("[WARN] No data for boxplots.")
        return

    fig = plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("RMSD (Å)")
    plt.title("RMSD Distribution by Method (Boxplot)")
    plt.tight_layout()

    out_png = out_dir / "boxplot_by_method.png"
    out_pdf = out_dir / "boxplot_by_method.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def plot_ecdf(df: pd.DataFrame, out_dir: Path):
    """ECDF per method (lower-left is better)."""
    fig = plt.figure(figsize=(7, 5))

    plotted_any = False
    for col in METHOD_COLUMNS:
        xs, ys = ecdf_values(pd.to_numeric(df[col], errors="coerce").values)
        if xs.size == 0:
            continue
        plt.plot(xs, ys, label=METHOD_LABELS[col])
        plotted_any = True

    if not plotted_any:
        print("[WARN] No data for ECDF.")
        plt.close(fig)
        return

    plt.xlabel("RMSD (Å)")
    plt.ylabel("ECDF")
    plt.title("ECDF of RMSD by Method")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / "ecdf_by_method.png"
    out_pdf = out_dir / "ecdf_by_method.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def plot_scatter_vs_qsad(df: pd.DataFrame, out_dir: Path):
    """Scatter of each method vs QSAD with y=x reference."""
    base = pd.to_numeric(df["qsad_rmsd"], errors="coerce")
    methods = [c for c in METHOD_COLUMNS if c != "qsad_rmsd"]

    for col in methods:
        comp = pd.to_numeric(df[col], errors="coerce")
        valid = ~(base.isna() | comp.isna())
        x = base[valid].values
        y = comp[valid].values

        fig = plt.figure(figsize=(5.5, 5.5))
        if x.size == 0:
            print(f"[WARN] No paired data for scatter {METHOD_LABELS[col]} vs QSAD.")
            plt.close(fig)
            continue

        plt.scatter(x, y, s=25, alpha=0.8, edgecolors="none")
        lim_max = np.nanmax([x.max(), y.max()])
        lim_min = np.nanmin([x.min(), y.min()])
        pad = 0.1 * (lim_max - lim_min if lim_max > lim_min else 1.0)
        lo = max(0.0, lim_min - pad)
        hi = lim_max + pad
        plt.plot([lo, hi], [lo, hi])  # y = x

        plt.xlabel("QSAD RMSD (Å)")
        plt.ylabel(f"{METHOD_LABELS[col]} RMSD (Å)")
        plt.title(f"{METHOD_LABELS[col]} vs QSAD")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()

        out_png = out_dir / f"scatter_{METHOD_LABELS[col].lower()}_vs_qsad.png"
        out_pdf = out_dir / f"scatter_{METHOD_LABELS[col].lower()}_vs_qsad.pdf"
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[Saved] {out_png}")
        print(f"[Saved] {out_pdf}")


def plot_delta_bar(df: pd.DataFrame, out_dir: Path, top_n: int = 0):
    """
    Bar plot of (method - QSAD) per pdb_id.
    Positive: method worse than QSAD; Negative: method better than QSAD.
    """
    data = df.copy()
    if top_n and top_n > 0:
        data = data.head(top_n)

    base = pd.to_numeric(data["qsad_rmsd"], errors="coerce")
    pdb_ids = data["pdb_id"].astype(str).tolist()
    n = len(pdb_ids)
    if n == 0:
        print("[WARN] No rows to plot for delta bar.")
        return

    deltas = []
    labels = []
    for col in ["af3_rmsd", "colabfold_rmsd", "vqe_rmsd"]:
        comp = pd.to_numeric(data[col], errors="coerce")
        delta = comp - base
        deltas.append(delta.to_numpy())
        labels.append(METHOD_LABELS[col])

    x = np.arange(n)
    m = len(deltas)
    width = min(0.8 / m, 0.25)

    fig = plt.figure(figsize=(max(10, n * 0.5), 6))
    for i in range(m):
        shift = (i - (m - 1) / 2) * (width + 0.02)
        plt.bar(x + shift, deltas[i], width=width, label=labels[i])

    plt.axhline(0.0)
    plt.xticks(x, pdb_ids, rotation=90)
    plt.ylabel("ΔRMSD (method − QSAD) (Å)")
    plt.title("Per-PDB ΔRMSD relative to QSAD")
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / "delta_bar_vs_qsad.png"
    out_pdf = out_dir / "delta_bar_vs_qsad.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot RMSD comparisons from merged CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("result_summary/result_rmsd_merged.csv"),
        help="Path to merged RMSD CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("result_summary/figs"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=0,
        help="Limit number of PDBs for per-PDB bar plots (0 means all).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Merged CSV not found: {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Basic validations
    for col in ["pdb_id"] + METHOD_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    # Summary stats
    stats_df = summary_stats(df)
    stats_out = args.out_dir / "summary_stats.csv"
    stats_df.to_csv(stats_out, index=False)
    print(f"[Saved] {stats_out}")

    # Plots
    plot_grouped_bar(df, args.out_dir, top_n=args.top_n)
    plot_boxplots(df, args.out_dir)
    plot_ecdf(df, args.out_dir)
    plot_scatter_vs_qsad(df, args.out_dir)
    plot_delta_bar(df, args.out_dir, top_n=args.top_n)

    print("[Done] All figures saved.")


if __name__ == "__main__":
    main()
