# --*-- conding:utf-8 --*--
# @time:10/23/25 18:25
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:analysis.py


"""
Top-level entry for QSAD analysis and structure reconstruction.
Steps:
  1. Load energy JSONL file (quantum sampling results)
  2. Perform clustering on main_vectors
  3. Select best-energy cluster
  4. Reconstruct & refine the Cα-only structure
Outputs:
  - cluster_out/: clustering reports
  - refined_ca.pdb / refined_ca.csv / refine_report.json in same directory as input
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
    input_path = "e_results/1m7y/energies.jsonl"  # path to your energy file
    refine_mode = "standard"                      # "fast", "standard", or "premium"
    subsample_max = 64
    top_energy_pct = 0.30
    geom_thresh = 7.5
    steric_ok = True
    target_ca_distance = 3.8
    random_seed = 42
    do_local_polish = False                       # True if you have energy_fn

    # === Derived paths ===
    input_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    cluster_out = os.path.join(input_dir, "cluster_out")
    os.makedirs(cluster_out, exist_ok=True)

    logging.info("=== QSAD Reconstruction ===")
    logging.info("Input file : %s", input_path)
    logging.info("Output dir : %s", input_dir)
    logging.info("Cluster dir: %s", cluster_out)

    # ---------------------------
    # 1) Cluster analysis
    # ---------------------------
    pre_rules = {}
    if geom_thresh is not None:
        pre_rules[f"E_geom<={geom_thresh}"] = True
    if steric_ok:
        pre_rules["E_steric<=0"] = True

    c_cfg = ClusterConfig(
        method="kmedoids",
        k_candidates=[2, 3, 4, 5, 6, 7, 8],
        energy_key="E_total",
        prefilter_rules=pre_rules,
        random_seed=random_seed,
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
        logging.error("No best cluster selected. Check thresholds or data integrity.")
        return

    # Extract the best cluster data
    df_raw: pd.DataFrame = analyzer.df_raw  # type: ignore
    best_cluster_df = df_raw.iloc[best_idx].reset_index(drop=True)
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
        output_dir=input_dir,      # write refined structure here
        random_seed=random_seed,
    )

    # Optional energy callback (set to None by default)
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
