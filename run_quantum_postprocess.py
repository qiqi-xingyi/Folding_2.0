# --*-- coding:utf-8 --*--
# @time: 10/22/25
# @Author: Yuqi Zhang
# @File: run_quantum_postprocess.py

# --*-- coding:utf-8 --*--
# @Author: Yuqi Zhang
# @File: run_postprocess.py
"""
Top-level driver for QSADPP minimal pipeline.

Pipeline:
  1. SamplingReader   -> read quantum sampling CSV
  2. StructureMapper  -> map bitstrings to coordinates (2D lattice)
  3. EnergyCalculator -> compute structural & Ising Hamiltonian energies
  4. ClusterAnalyzer  -> cluster and pick lowest-energy cluster
  5. StructureFitter  -> fit & save weighted structure from best cluster
"""

from __future__ import annotations
import argparse
from pathlib import Path
from qsadpp import (
    SamplingReader,
    StructureMapper,
    EnergyCalculator,
    ClusterAnalyzer,
    StructureFitter,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal QSADPP Post-Processing Pipeline")

    ap.add_argument("--input", required=True, help="Path to quantum sampling CSV file")
    ap.add_argument("--output", required=True, help="Directory for output files")

    ap.add_argument("--n-clusters", type=int, default=8, help="Number of clusters (default=8)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for clustering")

    ap.add_argument("--w-clash", type=float, default=1.0, help="Weight for clash energy term")
    ap.add_argument("--w-mj", type=float, default=1.0, help="Weight for MJ contact term")

    ap.add_argument("--clash-threshold", type=float, default=1.0, help="Distance threshold for clashes")
    ap.add_argument("--mj-cutoff", type=float, default=2.5, help="Cutoff for MJ contact detection")

    ap.add_argument("--no-xyz", action="store_true", help="Do not save fitted structure as XYZ file")
    ap.add_argument("--no-csv", action="store_true", help="Do not save best-cluster members CSV")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=== QSADPP Minimal Post-Processing ===")
    print(f"Input : {inp}")
    print(f"Output: {out}\n")

    # 1. Read sampling results
    reader = SamplingReader()
    df = reader.read(inp)
    print(f"[1] Loaded {len(df)} samples with {df['bitstring'].str.len().mode().iloc[0]} bits")

    # 2. Map bitstrings to coordinates (2D lattice, z=0)
    mapper = StructureMapper()
    df = mapper.map_dataframe(df)
    print(f"[2] Mapped bitstrings -> coords (shape: {df['coords'].iloc[0].shape})")

    # 3. Compute energies (structural + Ising Hamiltonian)
    energy_calc = EnergyCalculator(
        w_clash=args.w_clash,
        w_mj=args.w_mj,
        clash_threshold=args.clash_threshold,
        mj_cutoff=args.mj_cutoff,
    )
    df = energy_calc.compute(df, h=None, J=None)
    print("[3] Computed energies (E_total, H_ising)")

    # 4. Cluster & select lowest-energy cluster
    clusterer = ClusterAnalyzer(n_clusters=args.n_clusters, random_state=args.seed)
    clustered, best_cluster = clusterer.run(df)
    print(f"[4] Performed clustering (k={args.n_clusters}); best cluster = {best_cluster}")

    # 5. Fit structures in best cluster & save
    fitter = StructureFitter()
    fitted_coords, best_members = fitter.fit_and_save(
        clustered,
        best_cluster,
        out_dir=out,
        save_xyz=not args.no_xyz,
        save_csv=not args.no_csv,
    )
    print("[5] Fitted structure saved.")
    print(f"    XYZ : {out / 'fitted_ca.xyz'}")
    print(f"    CSV : {out / 'best_cluster_members.csv'}\n")

    print("=== Done. ===")


if __name__ == "__main__":
    main()

