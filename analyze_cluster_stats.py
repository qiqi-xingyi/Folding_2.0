# --*-- conding:utf-8 --*--
# @time:10/31/25 18:31
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:analyze_cluster_stats.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze QSAD clustering stats JSON and report diagnostics.

- Validates basic fields
- Summarizes cluster size distribution
- Flags pathologies (giant cluster, too many micro-clusters, no energy separation)
- Writes a CSV summary next to the JSON
- Exits with non-zero code if severe issues are detected (optional via --strict)

Usage:
  python analyze_cluster_stats.py /path/to/cluster_stats.json
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def load_stats(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "stats" not in data or not isinstance(data["stats"], list):
        raise ValueError("Invalid JSON: missing 'stats' list.")
    return data


def gini_coefficient(x: np.ndarray) -> float:
    # Standard Gini (requires non-negative values)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if np.any(x < 0):
        raise ValueError("Gini requires non-negative values.")
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x_sorted = np.sort(x)
    n = x.size
    cumx = np.cumsum(x_sorted)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


def summarize(data: Dict[str, Any], strict: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], int]:
    # Build DataFrame
    df = pd.DataFrame(data["stats"])
    required_cols = {"cluster", "size", "energy_median", "energy_q05"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in stats: {missing}")

    # Sort by size desc, then energy_median asc
    df = df.sort_values(["size", "energy_median"], ascending=[False, True]).reset_index(drop=True)

    # Totals and sanity checks
    n_rows = int(data.get("n_rows", df["size"].sum()))
    total_size = int(df["size"].sum())
    best_cid = data.get("best_cluster_id", None)

    # Distribution metrics
    sizes = df["size"].to_numpy()
    top1 = int(sizes[0]) if len(sizes) > 0 else 0
    top1_frac = top1 / max(1, n_rows)
    k_nontrivial = int((sizes >= 50).sum())
    k_micro = int((sizes <= 5).sum())

    # Energy separation diagnostics
    e_median = df["energy_median"].to_numpy(dtype=float)
    e_unique = np.unique(np.round(e_median, decimals=6))
    energy_all_same = (e_unique.size == 1)

    # Robust dispersion of energy across clusters (IQR)
    q25, q75 = np.percentile(e_median, [25, 75]) if len(e_median) > 0 else (np.nan, np.nan)
    iqr = float(q75 - q25) if (np.isfinite(q25) and np.isfinite(q75)) else np.nan

    # Gini of cluster sizes (how imbalanced the partition is)
    gini = gini_coefficient(sizes) if sizes.size > 0 else 0.0

    # Heuristics and warnings
    warnings: List[str] = []
    errors: List[str] = []

    if total_size != n_rows:
        warnings.append(f"Sum of cluster sizes ({total_size}) != n_rows ({n_rows}).")

    if top1_frac > 0.8:
        warnings.append(f"Giant cluster detected: top cluster holds {top1_frac:.2%} of all samples.")

    if k_nontrivial <= 1:
        warnings.append(f"Only {k_nontrivial} clusters have size >= 50; partition may be too coarse or disconnected.")

    if k_micro > len(df) * 0.5:
        warnings.append(f"More than half clusters are micro-clusters (size <= 5): {k_micro}/{len(df)}.")

    if energy_all_same:
        warnings.append("All clusters have the same energy_median (to 1e-6). No energy separation observed.")

    if not np.isnan(iqr) and iqr < 1e-6:
        warnings.append(f"Energy median IQR across clusters is near zero ({iqr:.3g}).")

    if gini > 0.8:
        warnings.append(f"Cluster size Gini is very high ({gini:.3f}). Partition is highly imbalanced.")

    # Prepare summary dict
    summary = dict(
        n_rows=n_rows,
        n_clusters=len(df),
        best_cluster_id=best_cid,
        top1_size=top1,
        top1_frac=top1_frac,
        k_nontrivial=k_nontrivial,
        k_micro=k_micro,
        energy_all_same=bool(energy_all_same),
        energy_median_iqr=iqr,
        gini=gini,
        warnings=warnings,
    )

    # Severity to exit code
    exit_code = 0
    if strict:
        # Treat certain conditions as hard failures in --strict mode
        if top1_frac > 0.9 or energy_all_same or k_nontrivial == 0:
            errors.append("Severe pathology detected (disconnected or no separation).")
        if errors:
            exit_code = 2
        elif warnings:
            exit_code = 1

    return df, summary, exit_code


def save_csv(df: pd.DataFrame, json_path: str) -> str:
    out_dir = os.path.dirname(os.path.abspath(json_path))
    out_csv = os.path.join(out_dir, "cluster_stats_table.csv")
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=str, help="Path to cluster_stats.json")
    ap.add_argument("--strict", action="store_true", help="Return non-zero exit code on severe issues.")
    args = ap.parse_args()

    try:
        data = load_stats(args.json_path)
        df, summary, exit_code = summarize(data, strict=args.strict)
        out_csv = save_csv(df, args.json_path)

        print("=== Cluster Stats Summary ===")
        print(f"File               : {args.json_path}")
        print(f"Rows (n_rows)      : {summary['n_rows']}")
        print(f"#Clusters          : {summary['n_clusters']}")
        print(f"Best Cluster ID    : {summary['best_cluster_id']}")
        print(f"Top1 size          : {summary['top1_size']} ({summary['top1_frac']:.2%})")
        print(f"Gini(size)         : {summary['gini']:.4f}")
        print(f"Non-trivial (>=50) : {summary['k_nontrivial']}")
        print(f"Micro (<=5)        : {summary['k_micro']}")
        print(f"Energy all same    : {summary['energy_all_same']}")
        print(f"Energy median IQR  : {summary['energy_median_iqr']:.6f}")
        if summary["warnings"]:
            print("\n--- Warnings ---")
            for w in summary["warnings"]:
                print(f"- {w}")
        print(f"\nCSV saved to       : {out_csv}")

        sys.exit(exit_code)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
