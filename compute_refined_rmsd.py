# --*-- coding:utf-8 --*--
# @time:11/2/25 22:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:compute_refined_rmsd.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot RMSD comparisons across QSAD, VQE, AF3, and ColabFold.
Only bar charts use custom fixed colors with outlines.
"""

import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ----------------------------
# File paths
# ----------------------------
AF3_TXT = "output_final/af3_rmsd_summary.txt"
COLABFOLD_TXT = "output_final/colabfold_rmsd_summary.txt"
VQE_TXT = "output_final/vqe_rmsd_summary.txt"
QSAD_CSV = "output_final/qsad_rmsd_summary.csv"

OUT_DIR = "final_fig"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True

# ----------------------------
# Custom bar colors
# ----------------------------
BAR_COLORS = {
    "QSAD": "#CCFF00",      # fluorescent yellow-green
    "AF3": "#5B3A29",       # dark brown
    "VQE": "#003366",       # deep blue
    "ColabFold": "#4A4A4A", # dark gray
}
EDGE_KW = dict(edgecolor="black", linewidth=0.6)


# ----------------------------
# Helper loaders
# ----------------------------
def load_tab_txt(path: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "\t" not in line:
                continue
            pid, val = line.split("\t", 1)
            try:
                data[pid.strip()] = float(val.strip())
            except Exception:
                pass
    return data


def load_qsad_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["pdb_id", "sequence", "final_rmsd"]].copy()


def savefig(out_path_base: str):
    png = f"{out_path_base}.png"
    pdf = f"{out_path_base}.pdf"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"[saved] {png}\n[saved] {pdf}")


# ----------------------------
# Bar charts (custom colors)
# ----------------------------
def grouped_bar_all(df: pd.DataFrame, methods: List[str]):
    plot_df = df.copy()
    if "QSAD" in plot_df.columns:
        plot_df = plot_df.sort_values("QSAD", na_position="last")
    pdbs = plot_df["pdb_id"].tolist()

    x = np.arange(len(pdbs))
    width = max(0.1, min(0.8 / max(1, len(methods)), 0.22))

    fig = plt.figure(figsize=(max(12, 0.35 * len(pdbs)), 6))
    ax = plt.gca()

    for i, meth in enumerate(methods):
        heights = plot_df[meth].to_numpy() if meth in plot_df.columns else np.full(len(pdbs), np.nan)
        xpos = x + (i - (len(methods) - 1) / 2) * width
        for xi, yi in zip(xpos, heights):
            if pd.notna(yi):
                ax.bar(
                    xi, yi, width=width,
                    color=BAR_COLORS.get(meth, "#999999"),
                    **EDGE_KW
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pdbs, rotation=90)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD across methods (QSAD set)")

    handles = [Patch(facecolor=BAR_COLORS[m], **EDGE_KW) for m in methods]
    ax.legend(handles, methods, ncols=min(4, len(methods)))

    savefig(os.path.join(OUT_DIR, "bar_all_methods"))
    plt.close()


def improvement_bar(df: pd.DataFrame, other: str, base: str = "QSAD"):
    if other not in df.columns or base not in df.columns:
        return
    sub = df[["pdb_id", base, other]].dropna()
    if sub.empty:
        return

    sub = sub.sort_values(base)
    diff = sub[other] - sub[base]

    fig = plt.figure(figsize=(max(10, 0.25 * len(sub)), 4))
    ax = plt.gca()
    ax.bar(
        sub["pdb_id"],
        diff,
        color=BAR_COLORS.get(other, "#999999"),
        **EDGE_KW
    )
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xticklabels(sub["pdb_id"], rotation=90)
    ax.set_ylabel(f"{other} - {base} (Å)")
    ax.set_title(f"Improvement vs {other} (positive means {base} is better)")

    handles = [Patch(facecolor=BAR_COLORS[other], **EDGE_KW)]
    ax.legend(handles, [other])

    savefig(os.path.join(OUT_DIR, f"improvement_{other.lower()}_vs_{base.lower()}"))
    plt.close()


# ----------------------------
# Default-style plots
# ----------------------------
def scatter_vs_qsad(df: pd.DataFrame, other: str, base: str = "QSAD"):
    if other not in df.columns or base not in df.columns:
        return
    sub = df[[base, other]].dropna()
    if sub.empty:
        return

    xmin = float(min(sub[base].min(), sub[other].min()))
    xmax = float(max(sub[base].max(), sub[other].max()))
    pad = 0.1 * (xmax - xmin) if xmax > xmin else 0.5
    lo, hi = xmin - pad, xmax + pad

    plt.figure(figsize=(5, 5))
    plt.scatter(sub[base], sub[other], s=20)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(f"{base} RMSD (Å)")
    plt.ylabel(f"{other} RMSD (Å)")
    plt.title(f"{base} vs {other} RMSD")
    savefig(os.path.join(OUT_DIR, f"scatter_{base.lower()}_vs_{other.lower()}"))
    plt.close()


def boxplot_by_method(df: pd.DataFrame, methods: List[str]):
    vals, labels = [], []
    for m in methods:
        if m in df.columns and df[m].notna().sum() > 0:
            vals.append(df[m].dropna().values)
            labels.append(m)
    if not vals:
        return
    plt.figure(figsize=(1.8 * len(labels), 5))
    plt.boxplot(vals, labels=labels, showfliers=True, patch_artist=False)
    plt.ylabel("RMSD (Å)")
    plt.title("RMSD distribution by method")
    savefig(os.path.join(OUT_DIR, "boxplot_rmsd_by_method"))
    plt.close()


def ecdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def ecdf_plot(df: pd.DataFrame, methods: List[str]):
    available = [m for m in methods if m in df.columns and df[m].notna().sum() > 0]
    if not available:
        return
    plt.figure(figsize=(6, 5))
    for m in available:
        x, y = ecdf(df[m].dropna().values)
        plt.plot(x, y, label=m)
    plt.xlabel("RMSD (Å)")
    plt.ylabel("ECDF")
    plt.title("ECDF of RMSD by method")
    plt.legend()
    savefig(os.path.join(OUT_DIR, "ecdf_rmsd_by_method"))
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    af3 = load_tab_txt(AF3_TXT)
    colab = load_tab_txt(COLABFOLD_TXT)
    vqe = load_tab_txt(VQE_TXT)
    qsad_df = load_qsad_csv(QSAD_CSV)

    merged = qsad_df.rename(columns={"final_rmsd": "QSAD"}).copy()
    merged["AF3"] = merged["pdb_id"].map(af3).astype(float)
    merged["ColabFold"] = merged["pdb_id"].map(colab).astype(float)
    merged["VQE"] = merged["pdb_id"].map(vqe).astype(float)

    merged_out = os.path.join(OUT_DIR, "merged_rmsd.csv")
    merged.to_csv(merged_out, index=False)
    print(f"[saved] {merged_out}")

    methods = ["QSAD", "VQE", "AF3", "ColabFold"]

    # custom-colored bar charts
    grouped_bar_all(merged, methods)
    for other in ["AF3", "ColabFold", "VQE"]:
        improvement_bar(merged, other, base="QSAD")

    # other plots keep default style
    for other in ["AF3", "ColabFold", "VQE"]:
        scatter_vs_qsad(merged, other, base="QSAD")

    boxplot_by_method(merged, methods)
    ecdf_plot(merged, methods)

    print("[done] All figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
