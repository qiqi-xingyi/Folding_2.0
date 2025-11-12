# --*-- conding:utf-8 --*--
# @time:11/11/25 19:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_time.py

# -*- coding: utf-8 -*-
"""
Plot QSAD vs VQE runtimes from result_summary/runtime_comparison.csv
Style:
  - QSAD color: #E47159
  - VQE  color: #3D5C6F
  - Narrow bars, large fonts
  - Y-axis clipped at 30000 s
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


QSAD_COLOR = "#E47159"
VQE_COLOR  = "#3D5C6F"
Y_CAP = 30000.0

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})


def human_seconds(x, _pos):
    # 12,345 style ticks
    return f"{int(x):,}"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    need = {"pdb_id", "qsad_time_total_s", "vqe_time_s"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    # Keep original CSV order (no sort), clip values for plotting
    df["qsad_plot_s"] = pd.to_numeric(df["qsad_time_total_s"], errors="coerce").clip(upper=Y_CAP)
    df["vqe_plot_s"]  = pd.to_numeric(df["vqe_time_s"], errors="coerce").clip(upper=Y_CAP)
    return df


def plot_runtime(df: pd.DataFrame, out_path: Path):
    # Only rows where VQE runtime is available
    sub = df.dropna(subset=["vqe_time_s"]).copy()
    if sub.empty:
        raise RuntimeError("No rows with VQE runtimes available to plot.")

    x = range(len(sub))
    labels = list(sub["pdb_id"])

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    width = 0.28
    ax.bar([i - width/2 for i in x], sub["qsad_plot_s"],
           width=width, color=QSAD_COLOR, edgecolor="none", label="QSAD total (s)")
    ax.bar([i + width/2 for i in x], sub["vqe_plot_s"],
           width=width, color=VQE_COLOR, edgecolor="none", label="VQE total (s)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("QSAD vs VQE total runtime per fragment")
    ax.set_ylim(0, Y_CAP)
    ax.yaxis.set_major_formatter(FuncFormatter(human_seconds))

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="Path to result_summary/runtime_comparison.csv")
    ap.add_argument("--out", type=Path, default=Path("figs/runtime_qsad_vs_vqe.png"),
                    help="Output figure path (PNG)")
    return ap.parse_args()


def main():
    args = parse_args()
    df = load_data(args.csv)
    plot_runtime(df, args.out)
    print(f"[OK] Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
