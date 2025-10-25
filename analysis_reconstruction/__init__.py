# --*-- conding:utf-8 --*--
# @time:10/23/25 17:42
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py.py

"""
analysis_reconstruction package

This package provides two main modules for QSAD post-processing:

1. Cluster analysis (`cluster_analysis`)
   - Perform energy-guided clustering on quantum sampling results.
   - Key classes:
       * ClusterConfig
       * ClusterAnalyzer

2. Structure refinement (`structure_refine`)
   - Refine and reconstruct a protein active-site conformation
     from clustered Cα-only coordinates.
   - Key classes:
       * RefineConfig
       * StructureRefiner

Typical usage example:
-----------------------
from analysis_reconstruction import ClusterConfig, ClusterAnalyzer, RefineConfig, StructureRefiner

# --- Clustering ---
c_cfg = ClusterConfig(method="kmedoids", energy_key="E_total")
analyzer = ClusterAnalyzer(c_cfg)
analyzer.load_file("e_results/1m7y/energies.jsonl")
analyzer.fit()
analyzer.save_reports()

best_idx = analyzer.get_best_cluster_indices()

# --- Refinement ---
r_cfg = RefineConfig(refine_mode="standard")
refiner = StructureRefiner(r_cfg)
refiner.load_cluster_dataframe(analyzer.df_raw.iloc[best_idx])
refiner.run()
refiner.save_outputs()
"""

from .cluster_analysis import ClusterConfig, ClusterAnalyzer
from .structure_refine import RefineConfig, StructureRefiner

__all__ = [
    "ClusterConfig",
    "ClusterAnalyzer",
    "RefineConfig",
    "StructureRefiner",
]