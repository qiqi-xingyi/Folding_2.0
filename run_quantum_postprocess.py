# --*-- coding:utf-8 --*--
# @time: 10/22/25
# @Author: Yuqi Zhang
# @File: run_quantum_postprocess.py

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from qsadpp.io import (
    load_all_samples,
    normalize_counts_to_prob_within,
    expand_bitstrings,
    ensure_standard_columns,
)
from qsadpp.cluster import (
    ClusterConfig,
    cluster_group,
    select_topK_per_group,
)


# ----------------------------
# Small analytics helpers
# ----------------------------

def _default_group_keys(df: pd.DataFrame) -> List[str]:
    """Heuristic: choose stable experiment keys that do not include 'bitstring'."""
    prefer = [
        "L", "n_qubits", "shots", "beta", "seed", "label",
        "backend", "ibm_backend", "circuit_hash",
        "protein", "sequence", "source_file",
    ]
    keys = [c for c in prefer if c in df.columns]
    return keys


def compute_per_experiment(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    """
    Aggregate rows per experiment and bitstring.
    Emits: counts (if present), q_prob (sum normalized within group), and frequency rank.
    """
    if "bitstring" not in df.columns:
        return pd.DataFrame()

    # weight for frequency ranking: prefer counts; otherwise q_prob
    weight = None
    if "count" in df.columns:
        weight = df["count"].fillna(0.0).astype(float)
    elif "counts" in df.columns:
        weight = df["counts"].fillna(0.0).astype(float)
    else:
        weight = df["q_prob"].fillna(0.0).astype(float)

    gcols = group_keys + ["bitstring"]
    agg = (
        df.assign(_w=weight)
          .groupby(gcols, dropna=False, sort=False, as_index=False)
          .agg(
              q_prob=("q_prob", "sum"),
              w=("_w", "sum"),
          )
    )
    # Rank within experiment by descending weight
    agg["rank_in_group"] = (
        agg.groupby(group_keys, dropna=False)["w"].rank(ascending=False, method="dense").astype(int)
    )
    return agg.sort_values(group_keys + ["rank_in_group", "w"], ascending=[True]*len(group_keys)+[True, False])


def compute_per_group_stats(df: pd.DataFrame, group_keys: List[str], how: str = "mean") -> pd.DataFrame:
    """
    Compute mean/median statistics of numeric columns within each experiment group.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return pd.DataFrame()
    if how == "mean":
        out = df.groupby(group_keys, dropna=False, as_index=False)[num_cols].mean()
    else:
        out = df.groupby(group_keys, dropna=False, as_index=False)[num_cols].median()
    return out


def compute_bit_marginals(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    """
    Compute bit marginals E[b_j] within each experiment group.
    Requires expanded b0..b{L-1}.
    """
    bcols = [c for c in df.columns if isinstance(c, str) and c.startswith("b") and c[1:].isdigit()]
    if not bcols:
        return pd.DataFrame()
    out = df.groupby(group_keys, dropna=False, as_index=False)[bcols].mean()
    return out


def compute_mode_hamming(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    """
    For each experiment group, find the modal bitstring (by counts if available, else by q_prob),
    then compute average Hamming distance to that mode within the group.
    """
    if "bitstring" not in df.columns:
        return pd.DataFrame()

    # weight selector
    if "count" in df.columns:
        w = df["count"].fillna(0.0).astype(float)
    elif "counts" in df.columns:
        w = df["counts"].fillna(0.0).astype(float)
    else:
        w = df["q_prob"].fillna(0.0).astype(float)

    # find mode per group
    gcols = group_keys + ["bitstring"]
    agg = (
        df.assign(_w=w)
          .groupby(gcols, dropna=False, as_index=False)
          .agg(w=("_w", "sum"))
    )
    # idx of max per group
    idx = agg.groupby(group_keys, dropna=False)["w"].idxmax()
    modes = agg.loc[idx, group_keys + ["bitstring"]].rename(columns={"bitstring": "mode_bitstring"})

    # merge mode back and compute Hamming
    merged = df.merge(modes, on=group_keys, how="left", validate="many_to_one")
    bs = merged["bitstring"].astype(str).to_numpy()
    ms = merged["mode_bitstring"].astype(str).to_numpy()

    # fast hamming
    def hamming(a: str, b: str) -> int:
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    ham = np.fromiter((hamming(a, b) for a, b in zip(bs, ms)), dtype=np.int64, count=len(bs))
    merged["hamming_to_mode"] = ham

    out = (
        merged.groupby(group_keys, dropna=False, as_index=False)
              .agg(
                  mode_bitstring=("mode_bitstring", "first"),
                  avg_hamming=("hamming_to_mode", "mean"),
                  max_hamming=("hamming_to_mode", "max"),
              )
    )
    return out


def per_experiment_clustering_and_reps(
    df: pd.DataFrame,
    cfg: ClusterConfig,
    group_keys: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster within each experiment group independently, then select top-K representatives.

    Returns
    -------
    clustered_all : pd.DataFrame
        Original rows with an extra 'cluster' column.
    reps_all : pd.DataFrame
        Concatenated representatives across groups.
    """
    if df.empty:
        return df.assign(cluster=pd.Series(dtype=int)), df

    clustered_list: List[pd.DataFrame] = []
    reps_list: List[pd.DataFrame] = []

    # iterate groups deterministically
    for _, g in df.groupby(group_keys, dropna=False, sort=False):
        if len(g) == 0:
            continue
        # cluster this group
        g_clustered = cluster_group(g, cfg)
        clustered_list.append(g_clustered)

        # select representatives
        reps = select_topK_per_group(
            g_clustered,
            per_cluster_max=cfg.per_cluster_max,
            beta_logq=cfg.beta_logq,
        )
        reps_list.append(reps)

    clustered_all = pd.concat(clustered_list, ignore_index=True) if clustered_list else df.assign(cluster=pd.NA)
    reps_all = pd.concat(reps_list, ignore_index=True) if reps_list else pd.DataFrame()
    return clustered_all, reps_all


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QSADPP Quantum Post-Processing")
    ap.add_argument("--input", required=True, help="Input directory containing sampling CSV files")
    ap.add_argument("--output", required=True, help="Output directory for analysis CSVs")
    ap.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    ap.add_argument("--method", default="kmedoids", choices=["kmedoids", "kmeans"], help="Clustering method")
    ap.add_argument("--k", type=int, default=8, help="Number of clusters")
    ap.add_argument("--per-cluster-max", type=int, default=3, help="Max representatives per cluster")
    ap.add_argument("--beta-logq", type=float, default=0.2, help="Weight for -log(q) in representative scoring")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    in_dir = Path(args.input).expanduser()
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== QSADPP Quantum Post-Processing ===")
    print(f"Input dir : {in_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Recursive : {args.recursive}")
    print(f"Method/k  : {args.method}/{args.k}")
    print(f"Seed      : {args.seed}")

    # 1) Load all CSVs
    df = load_all_samples(in_dir, recursive=args.recursive)
    if df.empty:
        print("[!] No data found. Exiting.")
        return
    print(f"[+] Loaded {len(df)} rows, columns = {list(df.columns)}")

    # 2) Ensure standard columns and normalize probabilities
    df = ensure_standard_columns(df)
    df = normalize_counts_to_prob_within(df)  # adds/overwrites q_prob

    # 3) Expand bitstrings to b0..b{L-1} (keeps original bitstring)
    df = expand_bitstrings(df, bit_col="bitstring", drop_original=False)

    # 4) Define grouping keys per "experiment"
    group_keys = _default_group_keys(df)
    print(f"[+] Group-by keys: {group_keys if group_keys else '(none)'}")

    # 5) Analytics tables
    per_exp = compute_per_experiment(df, group_keys)
    per_mean = compute_per_group_stats(df, group_keys, how="mean")
    per_median = compute_per_group_stats(df, group_keys, how="median")
    bit_marg = compute_bit_marginals(df, group_keys)
    mode_ham = compute_mode_hamming(df, group_keys)

    # 6) Clustering and representatives (per experiment)
    cfg = ClusterConfig(
        method=args.method,
        k=int(args.k),
        seed=int(args.seed),
        beta_logq=float(args.beta_logq),
        per_cluster_max=int(args.per_cluster_max),
    )
    clustered, reps = per_experiment_clustering_and_reps(df, cfg, group_keys)

    # 7) Write outputs
    def _write(name: str, frame: pd.DataFrame) -> None:
        p = out_dir / f"{name}.csv"
        frame.to_csv(p, index=False)
        print(f"  [✓] {name:<22} -> {p}")

    print("[*] Writing analysis CSV files...")
    if not per_exp.empty:   _write("per_experiment", per_exp)
    if not per_mean.empty:  _write("per_group_mean", per_mean)
    if not per_median.empty:_write("per_group_median", per_median)
    if not bit_marg.empty:  _write("bit_marginals", bit_marg)
    if not mode_ham.empty:  _write("mode_hamming", mode_ham)
    if not clustered.empty: _write("clustered", clustered)
    if not reps.empty:      _write("representatives", reps)

    print("[✓] Done.")


if __name__ == "__main__":
    main()
