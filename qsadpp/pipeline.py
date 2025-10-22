# --*-- conding:utf-8 --*--
# @time:10/21/25 14:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .io import (
    GROUP_KEYS,
    read_samples,
    aggregate_counts_to_prob,
    iter_groups,
    save_parquet,
    save_csv,
)
from .data import get_mj_table
from .features import compute_tierA_features, TierAWeights
from .cluster import ClusterConfig, cluster_group, select_topK_per_group
from .utils import write_xyz

# Optional imports for visuals; wrapped in try/except below
try:
    from .visuals import (
        plot_energy_histogram,
        plot_energy_vs_logq,
        plot_energy_box_by_cluster,
    )
    _HAS_VISUALS = True
except Exception:
    _HAS_VISUALS = False


@dataclass
class PipelinePaths:
    analysis_dir: Path
    reports_dir: Path
    reps_root: Path


def _ensure_dirs(out_dir: Path) -> PipelinePaths:
    analysis = out_dir / "analysis"
    reports = analysis / "reports"
    reps = out_dir / "rep_structures"
    analysis.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    reps.mkdir(parents=True, exist_ok=True)
    return PipelinePaths(analysis, reports, reps)


def _decode_one_coords(
    problem_builder: Callable[[Mapping[str, object]], object],
    meta: Mapping[str, object],
    bitstring: str,
) -> np.ndarray:
    """
    Minimal and robust decoding:
    - Build a problem object for the meta.
    - Call interpret with a distribution that concentrates on this bitstring.
    - Expect the returned object to provide get_calpha_coords().
    """
    problem = problem_builder(meta)
    binary_probs = {bitstring: 1.0}
    result = problem.interpret(binary_probs)
    coords = np.asarray(result.get_calpha_coords(), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Decoded coordinates must be (L, 3).")
    return coords


def _features_for_group(
    meta: Mapping[str, object],
    group_df: pd.DataFrame,
    build_problem: Callable[[Mapping[str, object]], object],
    mj_table: Mapping[str, Mapping[str, float]],
    weights: TierAWeights,
) -> pd.DataFrame:
    """
    Compute per-bitstring features for a single group.
    """
    rows: List[Dict[str, object]] = []
    sequence = str(meta.get("sequence", ""))
    protein = str(meta.get("protein", ""))
    label = meta.get("label")
    backend = meta.get("backend")
    ibm_backend = meta.get("ibm_backend")
    beta = meta.get("beta")
    seed = meta.get("seed")
    circuit_hash = meta.get("circuit_hash")

    for _, r in group_df.iterrows():
        bitstring = str(r["bitstring"])
        q_prob = float(r["prob"])
        try:
            ca = _decode_one_coords(build_problem, meta, bitstring)
        except Exception:
            # Fail-safe: skip this item if decoding fails
            continue

        feats = compute_tierA_features(sequence, ca, mj_table, weights=weights)
        feats.update(
            dict(
                bitstring=bitstring,
                q_prob=q_prob,
                protein=protein,
                sequence=sequence,
                label=label,
                backend=backend,
                ibm_backend=ibm_backend,
                beta=beta,
                seed=seed,
                circuit_hash=circuit_hash,
            )
        )
        rows.append(feats)

    if not rows:
        return pd.DataFrame(
            columns=[
                "E_A",
                "E_clash",
                "E_mj",
                "R_g",
                "clash_cnt",
                "contact_cnt",
                "bitstring",
                "q_prob",
                "protein",
                "sequence",
                "label",
                "backend",
                "ibm_backend",
                "beta",
                "seed",
                "circuit_hash",
            ]
        )

    return pd.DataFrame(rows)


def _save_topK_xyz(
    reps_df: pd.DataFrame,
    reps_root: Path,
    group_meta: Mapping[str, object],
    build_problem: Callable[[Mapping[str, object]], object],
) -> None:
    """
    Export XYZ for representative rows by re-decoding each bitstring.
    """
    protein = str(group_meta.get("protein", "unknown"))
    sequence = str(group_meta.get("sequence", ""))
    subdir_name = f"{protein}_{sequence}"
    out_dir = reps_root / subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, (_, row) in enumerate(reps_df.iterrows(), start=1):
        bitstring = str(row["bitstring"])
        coords = _decode_one_coords(build_problem, group_meta, bitstring)
        xyz_path = out_dir / f"top5_{rank}_{bitstring}.xyz"
        write_xyz(xyz_path, coords, title=f"{protein} {sequence} {bitstring}")


def run_full_pipeline(
    csv_paths: Sequence[str | Path],
    output_dir: str | Path,
    build_problem: Callable[[Mapping[str, object]], object],
    get_side_chain_hot_vector: Callable[[Mapping[str, object]], List[bool]],
    get_fifth_bit_flag: Callable[[Mapping[str, object]], bool],
    cluster_cfg: ClusterConfig,
    tierA_weights: Optional[TierAWeights] = None,
    topK_per_group: int = 5,
    per_cluster_max: int = 2,
    mj_path: Optional[str | Path] = None,
) -> None:
    """
    End-to-end post-processing:
      1) Read raw CSVs and normalize.
      2) Aggregate counts to probabilities per group.
      3) Reverse decode each bitstring to coordinates and compute features.
      4) Cluster per group and pick representatives.
      5) Save analysis tables, reports, and Top-K XYZ files.
    """
    out_dir = Path(output_dir)
    paths = _ensure_dirs(out_dir)

    # 1) Read
    raw = read_samples(csv_paths)

    # 2) Aggregate
    clean = aggregate_counts_to_prob(raw, group_keys=GROUP_KEYS, validate_shots=False)
    save_parquet(clean, paths.analysis_dir / "clean.parquet")

    # 3) MJ table
    mj_table = get_mj_table(mj_path)

    weights = tierA_weights or TierAWeights()

    # Containers
    all_features_rows: List[pd.DataFrame] = []
    all_assignments_rows: List[pd.DataFrame] = []
    all_reps_rows: List[pd.DataFrame] = []

    # Iterate groups
    for gid, (meta, gdf) in enumerate(iter_groups(clean, group_keys=GROUP_KEYS), start=1):
        # Compute features for this group
        feat_df = _features_for_group(
            meta=meta,
            group_df=gdf,
            build_problem=build_problem,
            mj_table=mj_table,
            weights=weights,
        )

        if feat_df.empty:
            continue

        # Save per-group features intermediate if desired (optional)
        all_features_rows.append(feat_df.assign(group_id=gid))

        # Cluster
        labels, centers, used = cluster_group(feat_df, cluster_cfg)

        # Assignments dataframe
        assignments_df = feat_df.copy()
        assignments_df["cluster"] = labels.values
        all_assignments_rows.append(assignments_df.assign(group_id=gid))

        # Representatives
        reps_df = select_topK_per_group(
            feat_df,
            labels,
            per_cluster_max=per_cluster_max,
            beta_logq=cluster_cfg.beta_logq,
        )
        # Limit to topK_per_group globally per group
        if len(reps_df) > topK_per_group:
            reps_df = reps_df.sort_values("E_A").head(topK_per_group)
        all_reps_rows.append(reps_df.assign(group_id=gid))

        # Save XYZ for representatives of this group
        try:
            _save_topK_xyz(reps_df, paths.reps_root, meta, build_problem)
        except Exception:
            # Continue even if XYZ export fails for some representatives
            pass

    # Concatenate across groups
    if all_features_rows:
        features = pd.concat(all_features_rows, ignore_index=True)
        save_parquet(features, paths.analysis_dir / "features.parquet")
        # Also write CSV for human inspection
        save_csv(features, paths.analysis_dir / "features.csv")

    if all_assignments_rows:
        clusters = pd.concat(all_assignments_rows, ignore_index=True)
        save_parquet(clusters, paths.analysis_dir / "clusters.parquet")

    if all_reps_rows:
        reps_all = pd.concat(all_reps_rows, ignore_index=True)
        # Save representatives and a ranking view
        reps_all = reps_all.copy()
        # A simple ranking by E_A then -q_prob
        ranking = reps_all.sort_values(["E_A", "q_prob"], ascending=[True, False])
        save_csv(reps_all, paths.analysis_dir / "representatives.csv")
        save_csv(ranking, paths.analysis_dir / "ranking.csv")

        # Per-cluster stats (optional)
        if all_assignments_rows:
            try:
                stats = (
                    clusters.groupby(["group_id", "cluster"])["E_A"]
                    .agg(["count", "mean", "min", "max"])
                    .reset_index()
                )
                save_csv(stats, paths.analysis_dir / "cluster_stats.csv")
            except Exception:
                pass

    # Reports
    if _HAS_VISUALS and all_features_rows:
        try:
            features = pd.concat(all_features_rows, ignore_index=True)
            plot_energy_histogram(features, paths.reports_dir / "energy_hist.pdf")
            plot_energy_vs_logq(features, paths.reports_dir / "S_vs_logq.pdf")
            if all_assignments_rows:
                clusters = pd.concat(all_assignments_rows, ignore_index=True)
                plot_energy_box_by_cluster(clusters, paths.reports_dir / "energy_box_by_cluster.pdf")
        except Exception:
            pass

