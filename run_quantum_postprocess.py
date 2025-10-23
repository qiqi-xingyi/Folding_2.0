# --*-- coding:utf-8 --*--
# @time: 10/22/25
# @Author: Yuqi Zhang
# @File: run_quantum_postprocess.py


from __future__ import annotations
import time
from pathlib import Path

# Adjustable default parameters
N_CLUSTERS = 8
SEED = 0
W_CLASH = 1.0
W_MJ = 1.0
CLASH_THRESHOLD = 1.0
MJ_CUTOFF = 2.5
SAVE_XYZ = True
SAVE_CSV = True
DEFAULT_INPUT_NAME = "samples_demo.csv"

# Required imports
import numpy as np
import pandas as pd

from qsadpp import (
    SamplingReader,
    StructureMapper,
    EnergyCalculator,
    ClusterAnalyzer,
    StructureFitter,
)


def _maybe_select_csv(default_path: Path) -> Path:
    """Prefer default CSV; if not found, open a file dialog."""
    if default_path.exists():
        return default_path

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Select CSV",
            "Default sampling file not found.\nPlease choose a CSV file manually.",
        )
        fpath = filedialog.askopenfilename(
            title="Select quantum sampling CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.update()
        root.destroy()
        if fpath:
            return Path(fpath)
    except Exception:
        pass

    raise FileNotFoundError(
        f"CSV file not found: {default_path}\nPlease place the sampling CSV in the project root "
        "or modify DEFAULT_INPUT_NAME."
    )


def _prepare_out_dir(root: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"out_clickrun_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    project_root = Path(__file__).resolve().parent
    default_csv = project_root / DEFAULT_INPUT_NAME
    csv_path = _maybe_select_csv(default_csv)
    out_dir = _prepare_out_dir(project_root)

    print("=== QSADPP Minimal Post-Processing (Click-Run) ===")
    print(f"Input  : {csv_path}")
    print(f"Output : {out_dir}\n")

    # 1) Read quantum sampling results
    reader = SamplingReader()
    df = reader.read(csv_path)
    bit_len = int(df["bitstring"].str.len().mode().iloc[0])
    print(f"[1] Loaded {len(df)} samples, bit length ≈ {bit_len}")

    # 2) Map bitstrings to coordinates (2D lattice, z=0)
    mapper = StructureMapper()
    df = mapper.map_dataframe(df)
    print(f"[2] Mapped bitstrings → coords, shape example: {df['coords'].iloc[0].shape}")

    # 3) Compute structural and Hamiltonian energies
    energy_calc = EnergyCalculator(
        w_clash=W_CLASH,
        w_mj=W_MJ,
        clash_threshold=CLASH_THRESHOLD,
        mj_cutoff=MJ_CUTOFF,
    )
    df = energy_calc.compute(df, h=None, J=None)
    print("[3] Computed energies (E_total, H_ising)")

    # 4) Cluster and pick the lowest-energy cluster
    clusterer = ClusterAnalyzer(n_clusters=N_CLUSTERS, random_state=SEED)
    clustered, best_cluster = clusterer.run(df)
    print(f"[4] Clustering done (k={min(N_CLUSTERS, len(df))}); best cluster = {best_cluster}")

    # 5) Fit and save weighted structure from the best cluster
    fitter = StructureFitter()
    fitted_coords, best_members = fitter.fit_and_save(
        clustered,
        best_cluster,
        out_dir=out_dir,
        save_xyz=SAVE_XYZ,
        save_csv=SAVE_CSV,
    )
    print("[5] Fitted structure saved.")
    if SAVE_XYZ:
        print(f"    XYZ : {out_dir / 'fitted_ca.xyz'}")
    if SAVE_CSV:
        print(f"    CSV : {out_dir / 'best_cluster_members.csv'}")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()


