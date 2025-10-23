# --*-- conding:utf-8 --*--
# @time:10/22/25 21:52
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fitter.py


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass
class StructureFitter:
    """
    Aggregate structures inside the best cluster by probability weights to obtain a
    fitted (averaged) conformation. Save XYZ and CSV if requested.

    Assumptions:
      - 'coords' column exists and holds np.ndarray (L,3)
      - 'q_prob' column exists (will be renormalized within the cluster)
    """

    def fit_and_save(self,
                     clustered: pd.DataFrame,
                     best_cluster: int,
                     out_dir: str | Path,
                     sequence_col: str = "sequence",
                     save_xyz: bool = True,
                     save_csv: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

        g = clustered[clustered["cluster"] == best_cluster].copy()
        if g.empty:
            raise ValueError("Best cluster has no members.")

        w = g["q_prob"].to_numpy(dtype=float)
        s = w.sum()
        w = w / s if s > 0 else np.ones_like(w) / len(w)

        # weighted average coords
        coords_list = g["coords"].tolist()
        L = max(C.shape[0] for C in coords_list if C is not None)
        acc = np.zeros((L, 3), dtype=float)
        for wi, C in zip(w, coords_list):
            if C is None:
                continue
            if C.shape[0] < L:
                pad = np.zeros((L - C.shape[0], 3), dtype=float)
                Ci = np.vstack([C, pad])
            else:
                Ci = C
            acc += wi * Ci
        fitted = acc

        # Save XYZ (Cα-only)
        if save_xyz:
            seq = g[sequence_col].dropna().astype(str).head(1).tolist()
            sequence = seq[0] if seq else "A" * L
            xyz_lines = [f"{L}", "fitted Cα from qsadpp"]
            for i in range(L):
                aa = sequence[i] if i < len(sequence) else "A"
                x, y, z = fitted[i]
                xyz_lines.append(f"{aa} {x:.6f} {y:.6f} {z:.6f}")
            (out_dir / "fitted_ca.xyz").write_text("\n".join(xyz_lines), encoding="utf-8")

        # Save cluster table
        if save_csv:
            g.to_csv(out_dir / "best_cluster_members.csv", index=False)

        return fitted, g
