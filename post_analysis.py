# --*-- conding:utf-8 --*--
# @time:10/23/25 18:25
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:analysis.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import pandas as pd

from analysis_reconstruction.cluster_analysis import ClusterConfig, ClusterAnalyzer
from analysis_reconstruction.structure_refine import RefineConfig, StructureRefiner


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    setup_logging()

    ap = argparse.ArgumentParser(description="QSAD: cluster -> best cluster -> refine CA structure")
    ap.add_argument("--input", default="e_results/1m7y/energies.jsonl", help="Path to energies JSONL/CSV/Parquet")
    ap.add_argument("--cluster-out", default=None, help="Directory for cluster reports (default: <input_dir>/cluster_out)")
    ap.add_argument("--mode", default="standard", choices=["fast", "standard", "premium"], help="Refine mode")
    ap.add_argument("--subsample-max", type=int, default=64, help="Max samples from best cluster for refinement")
    ap.add_argument("--top-energy-pct", type=float, default=0.30, help="Top energy percentile used before random fill")
    ap.add_argument("--target-ca-distance", type=float, default=3.8, help="Target Cα-Cα neighbor distance (Å)")
    ap.add_argument("--geom-thresh", type=float, default=7.5, help="Prefilter threshold for E_geom (<=)")
    ap.add_argument("--steric-ok", action="store_true", help="Require E_steric <= 0 in prefilter")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    input_path = args.input
    input_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    cluster_out = args.cluster_out or os.path.join(input_dir, "cluster_out")

    # ---------------------------
    # 1) Cluster analysis
    # ---------------------------
    pre_rules = {}
    if args.geom_thresh is not None:
        pre_rules[f"E_geom<={args.geom_thresh}"] = True
    if args.steric_ok:
        pre_rules["E_steric<=0"] = True

    c_cfg = ClusterConfig(
        method="kmedoids",
        k_candidates=[2, 3, 4, 5, 6, 7, 8],
        energy_key="E_total",
        prefilter_rules=pre_rules,
        random_seed=args.seed,
        output_dir=cluster_out,
        main_vectors_col="main_vectors",
        strict_same_length=True,
    )

    analyzer = ClusterAnalyzer(c_cfg)
    analyzer.load_file(input_path)
    analyzer.fit()
    analyzer.save_reports()

    best_idx = analyzer.get_best_cluster_indices()
    if not best_idx:
        raise SystemExit("No best cluster selected. Check prefilter thresholds or data columns.")

    # Build best-cluster DataFrame for refinement
    df_raw: pd.DataFrame = analyzer.df_raw  # type: ignore
    best_cluster_df = df_raw.iloc[best_idx].reset_index(drop=True)

    # ---------------------------
    # 2) Structure refinement (Cα only)
    # ---------------------------
    r_cfg = RefineConfig(
        refine_mode=args.mode,
        subsample_max=args.subsample_max,
        top_energy_pct=args.top_energy_pct,
        positions_col="main_positions",
        vectors_col="main_vectors",
        energy_key="E_total",
        sequence_col="sequence",
        target_ca_distance=args.target_ca_distance,
        proj_smooth_strength=0.10,
        proj_iters=10,
        do_local_polish=False,          # set True and pass energy_fn if you have a callable energy
        output_dir=input_dir,           # write final files into the same folder as input
        random_seed=args.seed,
    )

    # If you have a callable energy on CA-only coordinates, pass it here; else keep None.
    energy_fn = None

    refiner = StructureRefiner(r_cfg, energy_fn=energy_fn)
    refiner.load_cluster_dataframe(best_cluster_df)
    refiner.run()
    refiner.save_outputs()

    logging.info("Done. Refined Cα structure written to: %s", input_dir)


if __name__ == "__main__":
    main()
