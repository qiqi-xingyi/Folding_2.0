# --*-- conding:utf-8 --*--
# @time:10/21/25 14:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test_pp.py

"""
Minimal functional test for qsadpp pipeline.

This script generates dummy sampling data, runs the full pipeline,
and prints where the analysis results and XYZ files are saved.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from qsadpp.pipeline import run_full_pipeline
from qsadpp.cluster import ClusterConfig


# -----------------------------------------------------------------------------
# 1. Create a mock ProteinFoldingProblem that returns fake coordinates
# -----------------------------------------------------------------------------

class MockProteinFoldingProblem:
    def __init__(self, seq):
        self.seq = seq

    def interpret(self, binary_probs):
        """Return a dummy ProteinFoldingResult-like object."""
        class DummyResult:
            def __init__(self, seq):
                self.seq = seq
                L = len(seq)
                # Make simple 3D spiral coordinates
                t = np.linspace(0, 2 * np.pi, L)
                self.coords = np.stack([np.cos(t), np.sin(t), np.linspace(0, 1, L)], axis=1)

            def get_calpha_coords(self):
                return self.coords
        return DummyResult(self.seq)


# -----------------------------------------------------------------------------
# 2. Provide helper functions required by pipeline
# -----------------------------------------------------------------------------

def build_problem(meta):
    return MockProteinFoldingProblem(meta["sequence"])

def get_side_chain_hot_vector(meta):
    # fake side-chain presence flags
    return [i % 2 == 0 for i in meta["sequence"]]

def get_fifth_bit_flag(meta):
    return True  # simplified assumption


# -----------------------------------------------------------------------------
# 3. Create a fake sampling CSV
# -----------------------------------------------------------------------------

def create_demo_csv(path: Path):
    data = {
        "L": [5]*5,
        "n_qubits": [6]*5,
        "shots": [1024]*5,
        "beta": [0.0]*5,
        "seed": [0]*5,
        "label": ["demo_sampling"]*5,
        "backend": ["simulator"]*5,
        "ibm_backend": [""]*5,
        "circuit_hash": ["demo"]*5,
        "protein": ["demo_protein"]*5,
        "sequence": ["YAGYS"]*5,
        "bitstring": ["000010", "100010", "001101", "011110", "001110"],
        "count": [10, 5, 8, 4, 3],
        "prob": [0.2, 0.1, 0.16, 0.08, 0.06],
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"✓ Demo CSV written to {path}")


# -----------------------------------------------------------------------------
# 4. Run the pipeline
# -----------------------------------------------------------------------------

def main():
    out_dir = Path("demo_output")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "samples_demo.csv"
    create_demo_csv(csv_path)

    cfg = ClusterConfig(k=3, seed=42)

    run_full_pipeline(
        csv_paths=[csv_path],
        output_dir=out_dir,
        build_problem=build_problem,
        get_side_chain_hot_vector=get_side_chain_hot_vector,
        get_fifth_bit_flag=get_fifth_bit_flag,
        cluster_cfg=cfg,
        mj_path="qsadpp/mj_matrix.txt",  # your matrix file
    )

    print("✓ Pipeline finished.")
    print(f"→ Check output under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
