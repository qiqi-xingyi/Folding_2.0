# --*-- conding:utf-8 --*--
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
   - reverse decode bitstrings -> Cα coordinates (mock builder by default)
   - compute energies/features
   - cluster and select Top-K
   - save analysis tables and PDF reports
   - export Top-K XYZ per group
3) Additionally writes a human-readable features.csv next to features.parquet.

You can click "Run" in your IDE to execute this script.
"""

from __future__ import annotations
from pathlib import Path
from typing import Mapping, List
import numpy as np
import pandas as pd

# QSADPP imports
from qsadpp.pipeline import run_full_pipeline
from qsadpp.cluster import ClusterConfig
from qsadpp.features import TierAWeights


# ----------------------------
# Fixed configuration
# ----------------------------
INPUT_DIR = Path("quantum_data")          # put your quantum CSVs here
OUTPUT_DIR = Path("final_output")         # results will be written here
MJ_PATH = Path("qsadpp/mj_matrix.txt")    # your MJ matrix file (20x20). If missing, fallback to built-in.
RECURSIVE = False                         # set True to search subfolders for *.csv
TOPK_PER_GROUP = 5                        # how many XYZ to export per group
PER_CLUSTER_MAX = 2                       # max per cluster when selecting Top-K


# ----------------------------
# Default hooks (safe, minimal)
# ----------------------------
class _MockProteinFoldingProblem:
    """Minimal stand-in so the pipeline can run without external dependencies."""
    def __init__(self, seq: str):
        self.seq = seq

    def interpret(self, binary_probs):
        class _DummyResult:
            def __init__(self, seq: str):
                self.seq = seq
                L = len(seq)
                t = np.linspace(0, 2 * np.pi, L)
                # simple helix-like coordinates for demo
                self.coords = np.stack([np.cos(t), np.sin(t), np.linspace(0, 1, L)], axis=1)

            def get_calpha_coords(self):
                return self.coords
        return _DummyResult(self.seq)


def build_problem(meta: Mapping[str, object]):
    """Use the mock problem by default; replace with your real builder if available."""
    seq = str(meta["sequence"])
    return _MockProteinFoldingProblem(seq)


def get_side_chain_hot_vector(meta: Mapping[str, object]) -> List[bool]:
    """Assume NO side chains (length-N all False)."""
    seq = str(meta["sequence"])
    return [False for _ in seq]


def get_fifth_bit_flag(meta: Mapping[str, object]) -> bool:
    """Conservative default for decoding; adjust if your encoding requires otherwise."""
    return True


# ----------------------------
# Helpers
# ----------------------------
def _collect_csvs(input_dir: Path, recursive: bool = False) -> List[str]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir.resolve()}")
    pattern = "**/*.csv" if recursive else "*.csv"
    files = [str(p) for p in input_dir.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {input_dir.resolve()}")
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
    print(f"Input dir : {INPUT_DIR.resolve()}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")
    print(f"MJ path   : {MJ_PATH.resolve()} (fallback to built-in if missing)")
    print(f"Recursive : {RECURSIVE}")

    # Collect input CSVs
    csv_paths = _collect_csvs(INPUT_DIR, recursive=RECURSIVE)
    print(f"[+] Found {len(csv_paths)} CSV file(s).")

    # Ensure output folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cluster and scoring config (tune as needed)
    cluster_cfg = ClusterConfig(
        method="kmedoids",  # or "kmeans"
        k=8,
        seed=42,
        beta_logq=0.2,
        temperature=1.0,
    )
    weights = TierAWeights(
        w_clash=1.0,
        w_mj=0.5,   # scale MJ contribution if needed
        w_rg=0.1,
    )

    # MJ path: if not present, pass None to use built-in placeholder
    mj_arg = str(MJ_PATH) if MJ_PATH.exists() else None
    if mj_arg is None:
        print("[!] MJ matrix file not found; using built-in placeholder matrix (all zeros).")
        print("    For real runs, please provide a valid 20x20 MJ matrix at qsadpp/mj_matrix.txt.")

    # Run the full pipeline
    run_full_pipeline(
        csv_paths=csv_paths,
        output_dir=OUTPUT_DIR,
        build_problem=build_problem,
        get_side_chain_hot_vector=get_side_chain_hot_vector,
        get_fifth_bit_flag=get_fifth_bit_flag,
        cluster_cfg=cluster_cfg,
        tierA_weights=weights,
        topK_per_group=TOPK_PER_GROUP,
        per_cluster_max=PER_CLUSTER_MAX,
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
