# --*-- coding:utf-8 --*--
# @time:10/21/25 15:57
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_quantum_postprocess.py

"""
IDE-friendly top-level script for QSADPP post-processing.

- Fixed input directory:  quantum_data
- Fixed output directory: final_output
- Fixed MJ matrix path:  qsadpp/mj_matrix.txt (falls back to built-in if missing)

What it does:
1) Finds all CSV files under quantum_data/ (non-recursive by default; toggle RECURSIVE).
2) Runs the full QSADPP pipeline:
   - aggregate, decode, feature, cluster, select representatives
3) Writes analysis tables under final_output/analysis and representative XYZs under final_output/rep_structures
4) Emits simple PDF plots under final_output/analysis/reports (if matplotlib is available).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from qsadpp.pipeline import run_full_pipeline
from qsadpp.cluster import ClusterConfig
from qsadpp.features import TierAWeights


# ------------------ Mock builder (for out-of-the-box demo) ------------------ #
class MockProteinFoldingProblem:
    """
    A simple mock that turns a bitstring into a deterministic Cα polyline in 3D.

    This is only for end-to-end demonstration without external dependencies.
    Replace it with your real problem class that knows how to map bitstrings to coordinates.
    """

    def __init__(self, meta: Mapping[str, object]):
        self.L = int(meta.get("L", 6))
        self.sequence = str(meta.get("sequence", "ACDEFG"))
        self.seed = int(meta.get("seed", 0))

    def interpret(self, sample: Mapping[str, object]) -> Dict[str, object]:
        b = str(sample["bitstring"])
        rng = np.random.default_rng(abs(hash((b, self.seed))) % (2**32))
        # Generate a smooth random walk in 3D with unit steps
        dirs = rng.normal(size=(self.L, 3))
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        coords = np.cumsum(dirs, axis=0)
        coords -= coords.mean(axis=0, keepdims=True)
        return {"coords": coords}


def _find_csvs_in_dir(root: Path, recursive: bool = False) -> list[Path]:
    if recursive:
        files = sorted(root.rglob("*.csv"))
    else:
        files = sorted(root.glob("*.csv"))
    return files


def _maybe_convert_features_to_csv(out_dir: Path) -> None:
    """Create a human-readable CSV copy of features.parquet if present."""
    feats_parquet = out_dir / "analysis" / "features.parquet"
    if feats_parquet.exists():
        df = pd.read_parquet(feats_parquet)
        feats_csv = feats_parquet.with_suffix(".csv")
        df.to_csv(feats_csv, index=False)
        print(f"[+] Wrote {feats_csv.resolve()}")


def main():
    print("=== QSADPP Quantum Post-Processing ===")
    INPUT_DIR = Path("quantum_data")
    OUTPUT_DIR = Path("final_output")
    MJ_PATH = Path("qsadpp/mj_matrix.txt")  # optional; falls back to built-in if missing
    RECURSIVE = False

    INPUT_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    csvs = _find_csvs_in_dir(INPUT_DIR, recursive=RECURSIVE)
    if not csvs:
        print(f"[!] No CSV files found under {INPUT_DIR.resolve()}.")
        print("    Put your sampling CSVs there (columns: bitstring,count,prob,metadata...).")
        return

    cluster_cfg = ClusterConfig(
        method="kmedoids",
        n_clusters=8,
        per_cluster_max=3,
        beta_logq=0.2,
        seed=0,
    )

    weights = TierAWeights(
        w_clash=1.0,
        w_mj=1.0,
        w_rg=0.2,
    )

    mj_arg: Optional[str] = str(MJ_PATH) if MJ_PATH.exists() else None

    run_full_pipeline(
        csv_paths=[str(p) for p in csvs],
        output_dir=str(OUTPUT_DIR),
        build_problem=lambda meta: MockProteinFoldingProblem(meta),
        get_side_chain_hot_vector=lambda meta: [False] * int(meta.get("L", 0)),
        get_fifth_bit_flag=lambda meta: False,
        cluster_cfg=cluster_cfg,
        tierA_weights=weights,
        topK_per_group=5,
        per_cluster_max=2,
        mj_path=mj_arg,
    )

    # Also convert features.parquet -> features.csv for easy inspection
    _maybe_convert_features_to_csv(OUTPUT_DIR)

    print("\n[✓] Done.")
    print(f"    Analysis tables: {OUTPUT_DIR / 'analysis'}")
    print(f"    Reports (PDF) :  {OUTPUT_DIR / 'analysis' / 'reports'}")
    print(f"    TopK XYZ     :   {OUTPUT_DIR / 'rep_structures'}")

if __name__ == "__main__":
    main()
