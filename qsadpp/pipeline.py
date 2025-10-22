# --*-- coding:utf-8 --*--
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
from .utils import write_xyz_ca

# Optional imports for visuals; wrapped in try/except below
try:
    from .visuals import (
        plot_energy_hist,
        plot_S_vs_logq,
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

    @staticmethod
    def make(root: str | Path) -> "PipelinePaths":
        rootp = Path(root)
        analysis = rootp / "analysis"
        reports = analysis / "reports"
        reps = rootp / "rep_structures"
        analysis.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        reps.mkdir(parents=True, exist_ok=True)
        return PipelinePaths(analysis, reports, reps)


def _decode_one_coords(problem_obj, bitstring: str) -> np.ndarray:
    """
    Decode one bitstring to Cα coordinates using the provided problem object.

    The problem object must provide a method:
        interpret({"bitstring": ..., "prob": ...}) -> {"coords": np.ndarray(L,3), ...}
    """
    res = problem_obj.interpret({"bitstring": bitstring, "prob": 1.0})
    coords = np.asarray(res["coords"], dtype=float)
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be (L,3)"
    return coords


def _features_for_group(
    df_group: pd.DataFrame,
    meta: Mapping[str, object],
    mj_table: Dict[str, Dict[str, float]],
    weights: TierAWeights,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Compute features for all bitstrings within one metadata group.
    Returns a (features_df, extra_info) tuple.
    """
    bitstrings = df_group["bitstring"].astype(str).tolist()
    probs = pd.to_numeric(df_group["prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # -- Decode all to coords (Cα only) --
    problem = meta["__problem__"]
    coords_list: List[np.ndarray] = []
    for b in bitstrings:
        try:
            coords = _decode_one_coords(problem, b)
        except Exception:
            # fall back to NaN entry if decoding fails; will be dropped later
            coords = np.full((int(meta["L"]), 3), np.nan, dtype=float)
        coords_list.append(coords)

    # -- Compute Tier-A features --
    features = compute_tierA_features(
        coords_list=coords_list,
        sequence=str(meta["sequence"]),
        probs=probs,
        mj_table=mj_table,
        weights=weights,
    )
    # attach metadata columns
    for k in ["protein", "sequence", "label", "backend", "ibm_backend", "beta", "seed", "circuit_hash", "L"]:
        features[k] = meta.get(k, None)
    return features, {"decoded_ok": True}


def _save_topK_xyz(
    reps_df: pd.DataFrame,
    group_meta: Mapping[str, object],
    out_root: Path,
) -> None:
    """
    Save Top-K representative Cα-only XYZ files for quick visualization/docking.
    """
    protein = str(group_meta.get("protein", "unknown"))
    sequence = str(group_meta.get("sequence", ""))
    L = int(group_meta.get("L", len(sequence)))
    group_dir = out_root / protein
    group_dir.mkdir(parents=True, exist_ok=True)

    for _, row in reps_df.iterrows():
        bitstring = str(row["bitstring"])
        try:
            coords = np.asarray(row["coords"], dtype=float)
            if coords.shape != (L, 3):
                continue
            xyz_path = group_dir / f"{protein}_{bitstring[:16]}_L{L}.xyz"
            # Use the sequence as the element labels
            write_xyz_ca(xyz_path, coords, sequence)
        except Exception:
            # skip any malformed row
            pass


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
      2) Aggregate counts -> probabilities per experiment group.
      3) Load MJ table.
      4) For each group:
         a) decode all bitstrings -> coords
         b) compute Tier-A features (E_clash, E_mj, R_g, E_A)
         c) cluster + select representatives
         d) save analysis tables and representative XYZs
      5) Generate report plots (if matplotlib available).
    """
    paths = PipelinePaths.make(output_dir)

    # 1) Read
    raw = read_samples(csv_paths)
    save_parquet(raw, paths.analysis_dir / "raw.parquet")

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

    # 4) Iterate over groups
    for gid, (meta, df_group) in enumerate(iter_groups(clean)):
        meta = dict(meta)
        meta["__problem__"] = build_problem(meta)
        meta["group_id"] = gid

        try:
            feat_df, _ = _features_for_group(df_group, meta, mj_table, weights)
        except Exception:
            # skip group on fatal errors
            continue

        # store group id
        feat_df["group_id"] = gid
        all_features_rows.append(feat_df)
        save_parquet(feat_df, paths.analysis_dir / f"group_{gid:04d}_features.parquet")

        # clustering per group
        try:
            labels, centers, used = cluster_group(feat_df, cluster_cfg)
            assign = feat_df.loc[used, ["bitstring", "E_A", "q_prob", "protein", "sequence"]].copy()
            assign["group_id"] = gid
            assign["cluster"] = labels
            all_assignments_rows.append(assign)
            save_parquet(assign, paths.analysis_dir / f"group_{gid:04d}_clusters.parquet")

            reps = select_topK_per_group(
                assign.merge(feat_df[["bitstring", "coords"]], on="bitstring", how="left"),
                topK=topK_per_group,
                per_cluster_max=per_cluster_max,
                beta_logq=cluster_cfg.beta_logq,
            )
            reps["group_id"] = gid
            all_reps_rows.append(reps)
            save_csv(reps, paths.analysis_dir / f"group_{gid:04d}_representatives.csv")

            # export XYZ
            _save_topK_xyz(reps, meta, paths.reps_root)

        except Exception:
            # clustering is optional; continue pipeline even if it fails
            pass

    # Roll-up tables
    if all_features_rows:
        features_all = pd.concat(all_features_rows, ignore_index=True)
        save_parquet(features_all, paths.analysis_dir / "features.parquet")
        save_csv(features_all, paths.analysis_dir / "features.csv")
    if all_assignments_rows:
        clusters_all = pd.concat(all_assignments_rows, ignore_index=True)
        save_parquet(clusters_all, paths.analysis_dir / "clusters.parquet")
        save_csv(clusters_all, paths.analysis_dir / "clusters.csv")
        # basic stats
        try:
            stats = (
                clusters_all.groupby(["group_id", "cluster"])["E_A"]
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
            # Add S = E_A - beta_logq * log(q_prob) for plotting
            _q = np.clip(
                pd.to_numeric(features["q_prob"], errors="coerce")
                .fillna(1e-300)
                .to_numpy(dtype=float),
                1e-300,
                None,
            )
            _EA = pd.to_numeric(features["E_A"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            features["S"] = (_EA - cluster_cfg.beta_logq * np.log(_q)).astype(float)

            plot_energy_hist(features, paths.reports_dir / "energy_hist.pdf")
            plot_S_vs_logq(features, paths.reports_dir / "S_vs_logq.pdf")
            if all_assignments_rows:
                clusters = pd.concat(all_assignments_rows, ignore_index=True)
                plot_energy_box_by_cluster(clusters, paths.reports_dir / "energy_box_by_cluster.pdf")
        except Exception:
            pass
