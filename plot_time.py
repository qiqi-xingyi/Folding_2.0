# --*-- conding:utf-8 --*--
# @time:11/11/25 19:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_time.py

# -*- coding: utf-8 -*-
"""
IDE-ready plotting script for QSAD vs VQE runtime comparison.

Features:
- Reads result_summary/runtime_comparison.csv
- Plots QSAD (#E47159) vs VQE (#3D5C6F) total runtimes
- Narrow bars, large fonts, publication style
- Y-axis clipped at 30,000 seconds
- Output: figs/runtime_qsad_vs_vqe.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ====== File paths ======
CSV_PATH = "result_summary/runtime_comparison.csv"
OUT_PATH = "figs/runtime_qsad_vs_vqe.png"

# ====== Style parameters ======
QSAD_COLOR = "#E47159"
VQE_COLOR = "#3D5C6F"
Y_CAP = 30000.0

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 600,
})


def human_seconds(x, _pos):
    """Format y-axis ticks with comma separators."""
    return f"{int(x):,}"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"pdb_id", "qsad_time_total_s", "vqe_time_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["qsad_plot_s"] = pd.to_numeric(df["qsad_time_total_s"], errors="coerce").clip(upper=Y_CAP)
    df["vqe_plot_s"] = pd.to_numeric(df["vqe_time_s"], errors="coerce").clip(upper=Y_CAP)
    return df


def plot_runtime(df: pd.DataFrame, out_path: str):
    df = df.dropna(subset=["vqe_time_s"]).copy()
    if df.empty:
        raise RuntimeError("No rows with VQE runtime data available.")

    x = range(len(df))
    labels = list(df["pdb_id"])

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.26

    ax.bar([i - width / 2 for i in x], df["qsad_plot_s"],
           width=width, color=QSAD_COLOR, label="QSAD total (s)", edgecolor="black", linewidth=0.7)
    ax.bar([i + width / 2 for i in x], df["vqe_plot_s"],
           width=width, color=VQE_COLOR, label="VQE total (s)", edgecolor="black", linewidth=0.7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("QSAD vs VQE total runtime per fragment")
    ax.set_ylim(0, Y_CAP)
    ax.yaxis.set_major_formatter(FuncFormatter(human_seconds))
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Figure saved to: {out_path}")


if __name__ == "__main__":
    df = load_data(CSV_PATH)
    plot_runtime(df, OUT_PATH)
