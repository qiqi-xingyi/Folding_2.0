# --*-- conding:utf-8 --*--
# @time:10/21/25 14:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pipeline.py

"""
End-to-end pipeline to:
- Load CSVs
- Group samples
- Full reverse decoding (ALL bitstrings): bitstring -> ProteinFoldingResult -> Cα
- Compute Tier-A features (E_A, etc.)
- Cluster per group; select representatives and final top-K (K=5 by default)
- Save analysis tables and plots
- Export top-K XYZ files per group (Cα-only)

You must provide a `build_problem` callable:
    build_problem(group_meta: dict) -> ProteinFoldingProblem
and two light helpers giving side-chain info:
    get_side_chain_hot_vector(group_meta: dict) -> List[bool]
    get_fifth_bit_flag(group_meta: dict) -> bool

These are kept explicit to avoid hidden coupling with external code.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .io import read_samples, aggregate_counts_to_prob, iter_groups, save_parquet, save_csv, GROUP_KEYS
from .data import get_mj_table
from .reverse_decoder import ReverseDecoder, batch_decode_coords_via_problem
from .features import compute_features_for_group, TierAWeights
from .cluster import ClusterConfig, cluster_group, select_topK_per_group
from .utils import write_xyz_ca
from .visuals import plot_energy_hist, plot_S_vs_logq, plot_energy_box_by_cluster


def run_full_pipeline(
    csv_paths: Sequence[str | Path],
    output_dir: str | Path,
    build_problem: Callable[[Mapping[str, object]], object],
    get_side_chain_hot_vector: Callable[[Mapping[str, object]], List[bool]],
    get_fifth_bit_flag: Callable[[Mapping[str, object]], bool],
    cluster_cfg: Optional[ClusterConfig] = None,
    tierA_weights: Optional[TierAWeights] = None,
    topK_per_group: int = 5,
    per_cluster_max: int = 2,
    mj_path: Optional[str | Path] = None,
) -> None:
    """
    Execute the full analysis and save outputs under `output_dir`.

    Saved artifacts (per group):
      analysis/
        clean.parquet
        features.parquet
        clusters.parquet
        cluster_stats.csv
        representatives.csv
        ranking.csv
        reports/
          energy_hist_{gid}.pdf
          S_vs_logq_{gid}.pdf
          energy_box_by_cluster_{gid}.pdf
      rep_structures/
        {group_id}/top5_{rank}_{bitstring}.xyz   # Cα-only XYZ

    Notes
    -----
    - 'group_id' is a stable string constructed from GROUP_KEYS.
    - Features/Ranks include ALL bitstrings (no top-K truncation before clustering).
    """
    out_root = Path(output_dir)
    (out_root / "analysis" / "reports").mkdir(parents=True, exist_ok=True)
    (out_root / "rep_structures").mkdir(parents=True, exist_ok=True)

    # 1) Load and aggregate to probs
    raw = read_samples(csv_paths)
    clean = aggregate_counts_to_prob(raw, group_keys=GROUP_KEYS, validate_shots=False)
    save_parquet(clean, out_root / "analysis" / "clean.parquet")

    # 2) Load MJ table
    mj_table = get_mj_table(mj_path)

    # 3) Setup configs
    cfg = cluster_cfg or ClusterConfig()
    wts = tierA_weights or TierAWeights()
    decoder = ReverseDecoder()  # we will use problem.interpret(...) to get coordinates

    # 4) Per-group processing
    all_features_rows: List[pd.DataFrame] = []
    all_clusters_rows: List[pd.DataFrame] = []
    all_reps_rows: List[pd.DataFrame] = []
    all_stats_rows: List[pd.DataFrame] = []
    all_rank_rows: List[pd.DataFrame] = []

    for meta, gdf in iter_groups(clean, group_keys=GROUP_KEYS):
        # Build a stable group id (no spaces/slashes)
        gid = "_".join(str(meta[k]) for k in GROUP_KEYS).replace("/", "_")
        seq = str(meta["sequence"])
        out_group_dir = out_root / "rep_structures" / gid
        out_group_dir.mkdir(parents=True, exist_ok=True)

        # 4.1 full reverse decode to Cα coords for ALL bitstrings
        items = list(zip(gdf["bitstring"].tolist(), gdf["prob"].tolist()))
        problem = build_problem(meta)
        side_hot = get_side_chain_hot_vector(meta)
        fifth_bit = get_fifth_bit_flag(meta)

        decoded = batch_decode_coords_via_problem(
            problem=problem,
            items=items,
            side_chain_hot_vector=side_hot,
            fifth_bit=fifth_bit,
            calpha_getter=None,  # use default safe_get_calpha_coords
        )

        # 4.2 compute features for ALL bitstrings
        feats = compute_features_for_group(
            sequence=seq,
            mj_table=mj_table,
            items=((row["bitstring"], row["q_prob"], row["ca_coords"]) for row in decoded),
        )
        feat_df = pd.DataFrame(feats)
        # attach metadata
        for k, v in meta.items():
            feat_df[k] = v
        all_features_rows.append(feat_df)

        # 4.3 cluster this group
        clusters_df, reps_df, stats_df = cluster_group(feat_df, cfg)
        all_clusters_rows.append(clusters_df.assign(group_id=gid))
        all_reps_rows.append(reps_df.assign(group_id=gid))
        all_stats_rows.append(stats_df.assign(group_id=gid))

        # 4.4 ranking and top-K selection
        top_df = select_topK_per_group(clusters_df, K=topK_per_group, per_cluster_max=per_cluster_max)
        all_rank_rows.append(top_df.assign(group_id=gid))

        # 4.5 save per-group plots
        reports = out_root / "analysis" / "reports"
        plot_energy_hist(clusters_df, reports / f"energy_hist_{gid}.pdf", title=f"E_A Distribution [{gid}]")
        plot_S_vs_logq(clusters_df, reports / f"S_vs_logq_{gid}.pdf", title=f"S vs -log q [{gid}]")
        plot_energy_box_by_cluster(clusters_df, reports / f"energy_box_by_cluster_{gid}.pdf", title=f"E_A by Cluster [{gid}]")

        # 4.6 export top-K XYZ (Cα-only)
        # Need to look up coordinates for each selected bitstring
        # Build a dict for quick lookup
        ca_map = {row["bitstring"]: row["ca_coords"] for row in decoded}
        for rank_i, row in top_df.sort_values("S").reset_index(drop=True).iterrows():
            bits = row["bitstring"]
            ca = ca_map[bits]
            write_xyz_ca(
                out_group_dir / f"top{topK_per_group}_{rank_i+1}_{bits}.xyz",
                coords=ca,
                sequence=seq,
            )

    # 5) save global tables
    features_all = pd.concat(all_features_rows, ignore_index=True)
    clusters_all = pd.concat(all_clusters_rows, ignore_index=True)
    reps_all = pd.concat(all_reps_rows, ignore_index=True)
    stats_all = pd.concat(all_stats_rows, ignore_index=True)
    ranking_all = pd.concat(all_rank_rows, ignore_index=True)

    save_parquet(features_all, out_root / "analysis" / "features.parquet")
    save_parquet(clusters_all, out_root / "analysis" / "clusters.parquet")
    save_csv(reps_all, out_root / "analysis" / "representatives.csv")
    save_csv(stats_all, out_root / "analysis" / "cluster_stats.csv")
    save_csv(ranking_all, out_root / "analysis" / "ranking.csv")
