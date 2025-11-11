# --*-- conding:utf-8 --*--
# @time:11/11/25 17:48
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Time.py

# -*- coding: utf-8 -*-
"""
QSAD vs VQE runtime summarizer
Paths (defaults match your repo layout):
  - quantum_data/quantum_data_summary.csv
  - QDockBank/<pdb_id>/<pdb_id>_metadata.json
Outputs to:
  - result_summary/runtime_comparison.csv
  - result_summary/summary_stats.csv
  - result_summary/runtime_qsad_vs_vqe.png
  - result_summary/runtime_qsad_scaling.png
Usage:
  python runtime_summary.py
  # or override paths:
  python runtime_summary.py --csv quantum_data/quantum_data_summary.csv \
                            --qdock QDockBank \
                            --out result_summary \
                            --logy
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


def load_qsad_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"QSAD CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    required = {
        "pdb_id", "sequence", "L", "n_qubits", "depth_mean", "depth_max",
        "avg_time_per_shot_sec", "groups", "seconds_total", "effective_samples"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    # Clean types
    df["pdb_id"] = df["pdb_id"].astype(str).str.strip()
    for col in ["L", "n_qubits", "groups"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["depth_mean", "depth_max", "avg_time_per_shot_sec",
                "seconds_total", "effective_samples"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_vqe_time(meta_path: Path) -> Optional[float]:
    """Return execution_time_s from metadata json, or None if missing/invalid."""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta: Dict[str, Any] = json.load(f)
        v = meta.get("quantum_metadata", {}).get("execution_time_s", None)
        if v is None:
            return None
        v = float(v)
        if v <= 0:
            return None
        return v
    except Exception:
        return None


def attach_vqe_times(df: pd.DataFrame, qdock_root: Path) -> pd.DataFrame:
    vqe_times = []
    for pdb_id in df["pdb_id"]:
        meta_path = qdock_root / pdb_id / f"{pdb_id}_metadata.json"
        vqe_times.append(read_vqe_time(meta_path))
    out = df.copy()
    out["vqe_time_s"] = vqe_times
    return out


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["qsad_time_total_s"] = out["seconds_total"]
    out["qsad_time_per_group_s"] = out["seconds_total"] / out["groups"]
    # Speedup: VQE_time / QSAD_time (only when both are present)
    out["speedup_vqe_over_qsad"] = out["vqe_time_s"] / out["qsad_time_total_s"]
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_bar_compare(df: pd.DataFrame, out_png: Path, logy: bool = False) -> None:
    sub = df.dropna(subset=["vqe_time_s"]).copy()
    if sub.empty:
        return
    x = range(len(sub))
    labels = list(sub["pdb_id"])

    plt.figure(figsize=(12, 6))
    plt.bar([i for i in x], sub["qsad_time_total_s"], width=0.4, label="QSAD total (s)")
    plt.bar([i + 0.4 for i in x], sub["vqe_time_s"], width=0.4, label="VQE total (s)")
    plt.xticks([i + 0.2 for i in x], labels, rotation=45, ha="right")
    plt.ylabel("Runtime (seconds)")
    plt.title("QSAD vs VQE total runtime per fragment")
    if logy:
        plt.yscale("log")
        plt.ylabel("Runtime (seconds, log scale)")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def make_scaling_plot(df: pd.DataFrame, out_png: Path, logy: bool = False) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["L"], df["qsad_time_total_s"])
    plt.xlabel("Sequence length (L)")
    plt.ylabel("QSAD total runtime (seconds)")
    if logy:
        plt.yscale("log")
        plt.ylabel("QSAD total runtime (seconds, log)")
    plt.title("QSAD runtime scaling w.r.t. sequence length (L)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def summarize_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate dataset-level summary: mean/median/IQR for key metrics."""
    def series_stats(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {"count": 0}
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        return {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
            "min": float(s.min()),
            "q1": float(q1),
            "median": float(s.median()),
            "q3": float(q3),
            "max": float(s.max()),
            "iqr": float(q3 - q1),
        }

    rows = {
        "qsad_total_s": series_stats(df["qsad_time_total_s"]),
        "qsad_per_group_s": series_stats(df["qsad_time_per_group_s"]),
        "vqe_total_s (available_only)": series_stats(df["vqe_time_s"].dropna()),
        "speedup (VQE/QSAD, available_only)": series_stats(df["speedup_vqe_over_qsad"].dropna()),
        "effective_samples": series_stats(df["effective_samples"]),
        "depth_mean": series_stats(df["depth_mean"]),
        "n_qubits": series_stats(df["n_qubits"]),
        "L": series_stats(df["L"]),
    }
    stat_df = (
        pd.DataFrame(rows)
        .T.reset_index()
        .rename(columns={"index": "metric"})
        .fillna("")
    )
    return stat_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize QSAD vs VQE runtimes.")
    p.add_argument("--csv", type=Path, default=Path("quantum_data/quantum_data_summary.csv"),
                   help="Path to QSAD summary CSV")
    p.add_argument("--qdock", type=Path, default=Path("QDockBank"),
                   help="Path to QDockBank root")
    p.add_argument("--out", type=Path, default=Path("result_summary"),
                   help="Output directory")
    p.add_argument("--logy", action="store_true", help="Use log scale on y-axis for plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_qsad_csv(args.csv)
    df = attach_vqe_times(df, args.qdock)
    df = compute_metrics(df)

    # Save per-fragment table
    out_csv = args.out / "runtime_comparison.csv"
    save_csv(df, out_csv)

    # Save dataset-level stats
    stats_df = summarize_stats(df)
    save_csv(stats_df, args.out / "summary_stats.csv")

    # Plots
    make_bar_compare(df, args.out / "runtime_qsad_vs_vqe.png", logy=args.logy)
    make_scaling_plot(df, args.out / "runtime_qsad_scaling.png", logy=args.logy)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {args.out / 'summary_stats.csv'}")
    print(f"[OK] Saved figures to: {args.out}")


if __name__ == "__main__":
    main()
