#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RMSD plots with:
- Robust ID normalization to reduce NaNs (strip + lowercase)
- Custom colors ONLY for bar charts (QSAD neon yellow-green; AF3 dark brown; VQE deep blue; ColabFold dark gray)
- Proper tick handling (no warnings) and improved grid aesthetics
- Optional hatch on missing bars (can toggle with SHOW_MISSING_HATCH)
"""

import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------
# Files
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
# Bar colors (bars only)
# ----------------------------
BAR_COLORS = {
    "QSAD": "#CCFF00",      # fluorescent yellow-green
    "AF3": "#5B3A29",       # dark brown
    "VQE": "#003366",       # deep blue
    "ColabFold": "#4A4A4A", # dark gray
}
EDGE_KW = dict(edgecolor="black", linewidth=0.7)
SHOW_MISSING_HATCH = True  # set False to hide the hatch rectangles for missing values

# ----------------------------
# Helpers
# ----------------------------
def _norm_pid(s: str) -> str:
    return s.strip().lower()

def load_tab_txt(path: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    if not os.path.exists(path):
        return d
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "\t" not in line:
                continue
            pid, val = line.split("\t", 1)
            try:
                d[_norm_pid(pid)] = float(val.strip())
            except Exception:
                pass
    return d

def load_qsad_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize columns
    df = df.rename(columns={"final_rmsd": "QSAD"})
    df["pdb_id_norm"] = df["pdb_id"].astype(str).map(_norm_pid)
    return df[["pdb_id", "pdb_id_norm", "sequence", "QSAD"]].copy()

def savefig(out_path_base: str):
    png = f"{out_path_base}.png"
    pdf = f"{out_path_base}.pdf"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"[saved] {png}\n[saved] {pdf}")

# ----------------------------
# Plots
# ----------------------------
def grouped_bar_all(df: pd.DataFrame, methods: List[str]):
    # sort by QSAD
    plot_df = df.sort_values("QSAD", na_position="last").reset_index(drop=True)
    pdbs = plot_df["pdb_id"].tolist()
    n = len(pdbs)

    x = np.arange(n)
    width = max(0.08, min(0.8 / max(1, len(methods)), 0.2))

    fig = plt.figure(figsize=(max(12, 0.35 * n), 6))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # draw bars (with consistent colors) and optional hatch rectangles for missing
    for i, meth in enumerate(methods):
        vals = plot_df[meth].to_numpy() if meth in plot_df.columns else np.full(n, np.nan)
        xpos = x + (i - (len(methods) - 1) / 2) * width

        # real bars
        mask = ~np.isnan(vals)
        if mask.any():
            ax.bar(
                xpos[mask], vals[mask], width=width,
                color=BAR_COLORS.get(meth, "#999999"),
                **EDGE_KW, label=meth if i == 0 else None
            )
        # hatch rectangles for missing (to keep group visual consistency)
        if SHOW_MISSING_HATCH:
            m2 = np.isnan(vals)
            if m2.any():
                ax.bar(
                    xpos[m2], np.zeros(m2.sum()), width=width,
                    facecolor="none", hatch="////", **EDGE_KW
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pdbs, rotation=90)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD across methods (QSAD set)")

    handles = [Patch(facecolor=BAR_COLORS[m], **EDGE_KW) for m in methods]
    labels = methods
    if SHOW_MISSING_HATCH:
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="////"))
        labels = labels + ["missing"]
    ax.legend(handles, labels, ncols=min(5, len(labels)))

    savefig(os.path.join(OUT_DIR, "bar_all_methods"))
    plt.close()

def improvement_bar(df: pd.DataFrame, other: str, base: str = "QSAD"):
    if other not in df.columns or base not in df.columns:
        return
    sub = df[["pdb_id", base, other]].dropna().sort_values(base).reset_index(drop=True)
    if sub.empty:
        return

    diff = sub[other].to_numpy() - sub[base].to_numpy()
    x = np.arange(len(sub))

    fig = plt.figure(figsize=(max(10, 0.25 * len(sub)), 4))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    ax.bar(
        x, diff,
        color=BAR_COLORS.get(other, "#999999"),
        **EDGE_KW
    )
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["pdb_id"].tolist(), rotation=90)
    ax.set_ylabel(f"{other} - {base} (Å)")
    ax.set_title(f"Improvement vs {other} (positive → {base} better)")

    handles = [Patch(facecolor=BAR_COLORS[other], **EDGE_KW)]
    ax.legend(handles, [other])

    savefig(os.path.join(OUT_DIR, f"improvement_{other.lower()}_vs_{base.lower()}"))
    plt.close()

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
    plt.boxplot(vals, tick_labels=labels, showfliers=True, patch_artist=False)  # modern arg
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

    # join on normalized ids
    merged = qsad_df.copy()
    merged["AF3"] = merged["pdb_id_norm"].map(af3).astype(float)
    merged["ColabFold"] = merged["pdb_id_norm"].map(colab).astype(float)
    merged["VQE"] = merged["pdb_id_norm"].map(vqe).astype(float)

    merged_out = os.path.join(OUT_DIR, "merged_rmsd.csv")
    merged.drop(columns=["pdb_id_norm"]).to_csv(merged_out, index=False)
    print(f"[saved] {merged_out}")

    methods = ["QSAD", "VQE", "AF3", "ColabFold"]

    grouped_bar_all(merged.drop(columns=["pdb_id_norm"]), methods)
    for other in ["AF3", "ColabFold", "VQE"]:
        improvement_bar(merged.drop(columns=["pdb_id_norm"]), other, base="QSAD")

    for other in ["AF3", "ColabFold", "VQE"]:
        scatter_vs_qsad(merged, other, base="QSAD")

    boxplot_by_method(merged, methods)
    ecdf_plot(merged, methods)

    print("[done] All figures saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
