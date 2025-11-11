# --*-- conding:utf-8 --*--
# @time:11/11/25 17:48
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Time_summary.py

# -*- coding: utf-8 -*-
"""
QSAD vs VQE runtime summarizer with custom figure styling.
Color palette:
  QSAD -> #E47159
  VQE  -> #3D5C6F
Y-limit capped at 30000 s.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def load_qsad_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["pdb_id"] = df["pdb_id"].astype(str).str.strip()
    for c in ["L", "n_qubits", "groups"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["depth_mean", "depth_max", "avg_time_per_shot_sec",
              "seconds_total", "effective_samples"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_vqe_time(meta_path: Path) -> Optional[float]:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        v = meta.get("quantum_metadata", {}).get("execution_time_s", None)
        return float(v) if v and float(v) > 0 else None
    except Exception:
        return None


def attach_vqe_times(df: pd.DataFrame, qdock_root: Path) -> pd.DataFrame:
    vqe_times = []
    for pdb_id in df["pdb_id"]:
        meta_path = qdock_root / pdb_id / f"{pdb_id}_metadata.json"
        vqe_times.append(read_vqe_time(meta_path))
    df = df.copy()
    df["vqe_time_s"] = vqe_times
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["qsad_time_total_s"] = df["seconds_total"]
    df["qsad_time_per_group_s"] = df["seconds_total"] / df["groups"]
    df["speedup_vqe_over_qsad"] = df["vqe_time_s"] / df["qsad_time_total_s"]
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_bar_compare(df: pd.DataFrame, out_png: Path, cap: float = 30000.0) -> None:
    sub = df.dropna(subset=["vqe_time_s"]).copy()
    if sub.empty:
        return
    # clip large values
    sub["qsad_plot_s"] = sub["qsad_time_total_s"].clip(upper=cap)
    sub["vqe_plot_s"] = sub["vqe_time_s"].clip(upper=cap)

    x = range(len(sub))
    labels = list(sub["pdb_id"])

    plt.figure(figsize=(12, 6))
    width = 0.28
    plt.bar([i - width/2 for i in x], sub["qsad_plot_s"],
            width=width, color="#E47159", label="QSAD total (s)", edgecolor="none")
    plt.bar([i + width/2 for i in x], sub["vqe_plot_s"],
            width=width, color="#3D5C6F", label="VQE total (s)", edgecolor="none")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Runtime (seconds)")
    plt.title("QSAD vs VQE total runtime per fragment")
    plt.ylim(0, cap)
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()


def make_scaling_plot(df: pd.DataFrame, out_png: Path, cap: float = 30000.0) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["L"], df["qsad_time_total_s"].clip(upper=cap),
                c="#E47159", edgecolors="none", s=60)
    plt.xlabel("Sequence length (L)")
    plt.ylabel("QSAD total runtime (seconds)")
    plt.title("QSAD runtime scaling with sequence length (L)")
    plt.ylim(0, cap)
    plt.grid(alpha=0.4, linestyle="--")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("quantum_data/quantum_data_summary.csv"))
    p.add_argument("--qdock", type=Path, default=Path("QDockBank"))
    p.add_argument("--out", type=Path, default=Path("result_summary"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_qsad_csv(args.csv)
    df = attach_vqe_times(df, args.qdock)
    df = compute_metrics(df)
    save_csv(df, args.out / "runtime_comparison.csv")

    make_bar_compare(df, args.out / "runtime_qsad_vs_vqe.png", cap=30000.0)
    make_scaling_plot(df, args.out / "runtime_qsad_scaling.png", cap=30000.0)
    print(f"[OK] Figures and CSV written to {args.out}")


if __name__ == "__main__":
    main()
