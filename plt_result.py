# --*-- coding: utf-8 --*--
# @time: 2025/11/06
# @Author : Yuqi Zhang
# @File : plot_rmsd_results.py
#
# Description:
#   Generate RMSD comparison figures with unified colors and enlarged Arial font.
#   Mean marker in boxplots changed to color #D71B5D (pink-red).

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== Global settings =====================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "legend.fontsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
})

METHOD_COLUMNS = ["qsad_rmsd", "af3_rmsd", "colabfold_rmsd", "vqe_rmsd"]
METHOD_LABELS = {
    "qsad_rmsd": "QSAD",
    "vqe_rmsd": "VQE",
    "af3_rmsd": "AF3",
    "colabfold_rmsd": "ColabFold",
}
METHOD_COLORS = {
    "qsad_rmsd": "#00A59B",
    "vqe_rmsd": "#A7BCDF",
    "af3_rmsd": "#A5A5A5",
    "colabfold_rmsd": "#8499BB",
}
MEAN_MARKER_COLOR = "#D71B5D"  # boxplot mean triangle color


def ecdf_values(x: np.ndarray):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
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
            paired = df[["qsad_rmsd", col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(paired) > 0:
                win_rate = (paired[col] <= paired["qsad_rmsd"]).mean()
            else:
                win_rate = np.nan
            s["win_rate_vs_qsad"] = win_rate
        else:
            s["win_rate_vs_qsad"] = np.nan
        stats.append(s)
    return pd.DataFrame(stats)


def plot_grouped_bar(df, out_dir: Path, top_n=0):
    data = df.copy()
    if top_n > 0:
        data = data.head(top_n)

    pdb_ids = data["pdb_id"].astype(str).tolist()
    n = len(pdb_ids)
    if n == 0:
        return

    vals = [pd.to_numeric(data[col], errors="coerce").to_numpy() for col in METHOD_COLUMNS]
    x = np.arange(n)
    m = len(METHOD_COLUMNS)
    width = min(0.8 / m, 0.2)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 6))
    for i, col in enumerate(METHOD_COLUMNS):
        shift = (i - (m - 1) / 2) * (width + 0.02)
        ax.bar(
            x + shift,
            vals[i],
            width=width,
            label=METHOD_LABELS[col],
            color=METHOD_COLORS[col],
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pdb_ids, rotation=90)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD Comparison per PDB")
    ax.legend()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"bar_grouped_per_pdb.{ext}", dpi=300)
    plt.close(fig)


def plot_boxplots(df, out_dir: Path):
    data = [pd.to_numeric(df[c], errors="coerce").dropna() for c in METHOD_COLUMNS]
    colors = [METHOD_COLORS[c] for c in METHOD_COLUMNS]
    labels = [METHOD_LABELS[c] for c in METHOD_COLUMNS]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        showmeans=True,
        meanprops=dict(
            marker="^", markerfacecolor=MEAN_MARKER_COLOR,
            markeredgecolor="black", markersize=10
        ),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD Distribution by Method")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"boxplot_by_method.{ext}", dpi=300)
    plt.close(fig)


def plot_ecdf(df, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for col in METHOD_COLUMNS:
        xs, ys = ecdf_values(pd.to_numeric(df[col], errors="coerce").values)
        if xs.size == 0:
            continue
        ax.plot(xs, ys, label=METHOD_LABELS[col], color=METHOD_COLORS[col], linewidth=2)

    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of RMSD")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ecdf_by_method.{ext}", dpi=300)
    plt.close(fig)


def plot_scatter_vs_qsad(df, out_dir: Path):
    base = pd.to_numeric(df["qsad_rmsd"], errors="coerce")
    for col in ["af3_rmsd", "colabfold_rmsd", "vqe_rmsd"]:
        comp = pd.to_numeric(df[col], errors="coerce")
        valid = ~(base.isna() | comp.isna())
        x, y = base[valid], comp[valid]
        if len(x) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, s=40, alpha=0.8, color=METHOD_COLORS[col], edgecolors="none")
        lo, hi = 0, max(x.max(), y.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("QSAD RMSD (Å)")
        ax.set_ylabel(f"{METHOD_LABELS[col]} RMSD (Å)")
        ax.set_title(f"{METHOD_LABELS[col]} vs QSAD")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"scatter_{METHOD_LABELS[col].lower()}_vs_qsad.{ext}", dpi=300)
        plt.close(fig)


def plot_delta_bar(df, out_dir: Path, top_n=0):
    data = df.copy()
    if top_n > 0:
        data = data.head(top_n)
    base = pd.to_numeric(data["qsad_rmsd"], errors="coerce")

    pdb_ids = data["pdb_id"].astype(str).tolist()
    n = len(pdb_ids)
    if n == 0:
        return

    comps = ["af3_rmsd", "colabfold_rmsd", "vqe_rmsd"]
    deltas = [pd.to_numeric(data[c], errors="coerce") - base for c in comps]
    colors = [METHOD_COLORS[c] for c in comps]
    labels = [METHOD_LABELS[c] for c in comps]

    x = np.arange(n)
    width = min(0.8 / len(comps), 0.25)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 6))
    for i, (delta, label, color) in enumerate(zip(deltas, labels, colors)):
        shift = (i - (len(comps) - 1) / 2) * (width + 0.02)
        ax.bar(x + shift, delta, width=width, label=label,
               color=color, edgecolor="black", linewidth=0.3)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pdb_ids, rotation=90)
    ax.set_ylabel("ΔRMSD (method − QSAD) (Å)")
    ax.set_title("Per-PDB ΔRMSD relative to QSAD")
    ax.legend()
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"delta_bar_vs_qsad.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot RMSD comparison figures with unified style.")
    parser.add_argument(
        "--csv", type=Path,
        default=Path("result_summary/result_rmsd_merged.csv"),
        help="Merged RMSD CSV path"
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path("result_summary/figs"),
        help="Output directory for figures"
    )
    parser.add_argument(
        "--top_n", type=int, default=0,
        help="Limit number of pdb_ids for grouped bars (0 means all)"
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Merged CSV not found: {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]

    stats_df = summary_stats(df)
    stats_out = args.out_dir / "summary_stats.csv"
    stats_df.to_csv(stats_out, index=False)
    print(f"[Saved] {stats_out}")

    plot_grouped_bar(df, args.out_dir, top_n=args.top_n)
    plot_boxplots(df, args.out_dir)
    plot_ecdf(df, args.out_dir)
    plot_scatter_vs_qsad(df, args.out_dir)
    plot_delta_bar(df, args.out_dir, top_n=args.top_n)

    print("[Done] All figures saved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
