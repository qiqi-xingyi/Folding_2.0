# --*-- coding:utf-8 --*--
# @time:10/31/25 19:55
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:analysis.py
"""
Top-level entry for QSAD analysis and structure reconstruction.
Steps:
  1) Load energy and feature JSONL files (quantum sampling results)
  2) Perform multi-view clustering (geometry/feature/bitstring) with energy bias
  3) Select the best-energy cluster
  4) Reconstruct and refine the Cα-only structure
Outputs:
  - <input_dir>/cluster_out/: clustering reports
  - refined_ca.pdb / refined_ca.csv / refine_report.json in <input_dir>
"""

import os
import logging
import pandas as pd

from analysis_reconstruction.cluster_analysis import ClusterConfig, ClusterAnalyzer
from analysis_reconstruction.structure_refine import RefineConfig, StructureRefiner


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    setup_logging()

    # === User parameters ===
    energies_path = "e_results/1m7y/energies.jsonl"
    features_path = "e_results/1m7y/features.jsonl"
    refine_mode = "premium"  # "fast" | "standard" | "premium"
    subsample_max = 5
    top_energy_pct = 0.2
    geom_thresh = 5.5  # E_geom ≤ geom_thresh
    require_no_steric = True  # E_steric == 0
    target_ca_distance = 3.7
    random_seed = 24
    do_local_polish = False

    # === Paths ===
    input_dir = os.path.dirname(os.path.abspath(energies_path)) or "."
    cluster_out = os.path.join(input_dir, "cluster_out")
    os.makedirs(cluster_out, exist_ok=True)

    logging.info("=== QSAD Reconstruction ===")
    logging.info("Energies : %s", energies_path)
    logging.info("Features : %s", features_path)
    logging.info("Out dir  : %s", input_dir)
    logging.info("Clusters : %s", cluster_out)

    # ---------------------------
    # 1) Cluster analysis
    # ---------------------------
    c_cfg = ClusterConfig(
        use_geom=True, use_feat=True, use_ham=True,
        w_geom=0.5, w_feat=0.3, w_ham=0.2,
        knn=40, diff_time=2,
        n_runs=5, min_cluster_size=25,
        seed=random_seed,
        bitstring_col="bitstring",
        positions_col="main_positions",
        energy_key="E_total",
    )

    analyzer = ClusterAnalyzer(c_cfg)
    analyzer.load_files(energies_path, features_path)

    # Prefilter data before clustering
    if geom_thresh is not None and "E_geom" in analyzer.df.columns:
        before = len(analyzer.df)
        analyzer.df = analyzer.df[analyzer.df["E_geom"] <= float(geom_thresh)].reset_index(drop=True)
        logging.info("Prefilter E_geom ≤ %.3f: %d → %d", geom_thresh, before, len(analyzer.df))
    if require_no_steric and "E_steric" in analyzer.df.columns:
        before = len(analyzer.df)
        analyzer.df = analyzer.df[analyzer.df["E_steric"] <= 0.0].reset_index(drop=True)
        logging.info("Prefilter E_steric == 0: %d → %d", before, len(analyzer.df))

    if len(analyzer.df) < 2:
        logging.error("Not enough samples after prefilter. Abort.")
        return

    analyzer.fit()
    analyzer.save_reports(cluster_out)

    best_idx = analyzer.get_best_cluster_indices()
    if not best_idx:
        logging.error("No best cluster found. Check thresholds or data integrity.")
        return

    best_cluster_df: pd.DataFrame = analyzer.df.iloc[best_idx].reset_index(drop=True)
    logging.info("Best cluster size: %d", len(best_cluster_df))

    # ---------------------------
    # 2) Structure refinement
    # ---------------------------
    r_cfg = RefineConfig(
        refine_mode=refine_mode,
        subsample_max=subsample_max,
        top_energy_pct=top_energy_pct,
        positions_col="main_positions",
        vectors_col="main_vectors",
        energy_key="E_total",
        sequence_col="sequence",
        target_ca_distance=target_ca_distance,
        proj_smooth_strength=0.10,
        proj_iters=10,
        do_local_polish=do_local_polish,
        output_dir=input_dir,
        random_seed=random_seed,
    )

    energy_fn = None

    refiner = StructureRefiner(r_cfg, energy_fn=energy_fn)
    refiner.load_cluster_dataframe(best_cluster_df)
    refiner.run()
    refiner.save_outputs()

    logging.info("Refinement complete.")
    logging.info("Output files:")
    logging.info("  - %s/refined_ca.pdb", input_dir)
    logging.info("  - %s/refined_ca.csv", input_dir)
    logging.info("  - %s/refine_report.json", input_dir)
    logging.info("  - %s/cluster_out/", input_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
