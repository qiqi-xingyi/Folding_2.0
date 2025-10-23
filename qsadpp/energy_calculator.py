# --*-- conding:utf-8 --*--
# @time:10/23/25 17:08
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:energy_calculator.py

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

_MJ_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "mj_matrix.txt")


@dataclass
class EnergyConfig:
    r_min: float = 0.5
    r_contact: float = 1.0
    d0: float = 0.57735
    lambda_overlap: float = 1000.0
    weights: Dict[str, float] = field(
        default_factory=lambda: {"steric": 1.0, "geom": 0.5, "bond": 0.2, "mj": 1.0}
    )
    normalize: bool = True
    output_path: str = "decoded_with_energy.jsonl"


class LatticeEnergyCalculator:
    """
    Compute approximate conformational energy for main-chain-only lattice proteins
    with an additional MJ contact potential.
    The MJ matrix is always loaded from qsadpp/mj_matrix.txt.
    """

    def __init__(self, cfg: EnergyConfig = EnergyConfig()):
        self.cfg = cfg
        self._mj_matrix, self._aa_index = self._load_mj_matrix()

    # ---------------- MJ Matrix ----------------
    def _load_mj_matrix(self) -> tuple[np.ndarray, Dict[str, int]]:
        """Load MJ matrix from the fixed package path."""
        path = _MJ_MATRIX_PATH
        if not os.path.exists(path):
            _LOG.warning("MJ matrix not found in package: %s; all E_MJ=0", path)
            return np.zeros((20, 20)), {}

        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        headers = lines[0].split()
        mat = np.zeros((len(headers), len(headers)))
        for i, line in enumerate(lines[1:]):
            vals = [float(x) for x in line.split()]
            for j in range(len(vals)):
                mat[i, j] = vals[j]
                mat[j, i] = vals[j]
        aa_index = {aa: i for i, aa in enumerate(headers)}
        _LOG.info("Loaded MJ matrix from package: %s (%d residues)", path, len(aa_index))
        return mat, aa_index

    # ---------------- Energy Components ----------------
    def compute_energy(self, main_positions: np.ndarray, sequence: str) -> Dict[str, float]:
        N = len(main_positions)
        if N < 3:
            return {"E_total": 0.0, "E_steric": 0.0, "E_geom": 0.0, "E_bond": 0.0, "E_mj": 0.0}

        # 1. bond energy
        v = np.diff(main_positions, axis=0)
        bond_lengths = np.linalg.norm(v, axis=1)
        E_bond = np.sum((bond_lengths - self.cfg.d0) ** 2)

        # 2. geometric energy
        v1, v2 = v[:-1], v[1:]
        cos_thetas = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        )
        cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
        E_geom = np.sum(1.0 - cos_thetas)

        # 3. steric repulsion
        diff = main_positions[:, None, :] - main_positions[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        mask = np.triu(np.ones_like(dist, dtype=bool), k=2)
        D = dist[mask]
        E_steric = np.sum((self.cfg.r_min - D[D < self.cfg.r_min]) ** 2)
        E_steric += np.sum(D[D == 0]) * self.cfg.lambda_overlap

        # 4. MJ contact energy
        E_mj = 0.0
        if len(self._aa_index) > 0:
            for i in range(N - 1):
                ai = self._aa_index.get(sequence[i])
                if ai is None:
                    continue
                for j in range(i + 2, N):  # skip bonded pairs
                    aj = self._aa_index.get(sequence[j])
                    if aj is None:
                        continue
                    if dist[i, j] <= self.cfg.r_contact:
                        E_mj += self._mj_matrix[ai, aj]

        w = self.cfg.weights
        E_total = (
            w["steric"] * E_steric
            + w["geom"] * E_geom
            + w["bond"] * E_bond
            + w["mj"] * E_mj
        )
        if self.cfg.normalize:
            E_total /= N

        return {
            "E_total": float(E_total),
            "E_steric": float(E_steric),
            "E_geom": float(E_geom),
            "E_bond": float(E_bond),
            "E_mj": float(E_mj),
        }

    # ---------------- Batch Evaluation ----------------
    def evaluate_jsonl(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        out_path = output_path or self.cfg.output_path
        written = 0
        with open(input_path, "r", encoding="utf-8") as fin, open(
            out_path, "w", encoding="utf-8"
        ) as fout:
            for idx, line in enumerate(fin):
                if limit and idx >= limit:
                    break
                rec = json.loads(line)
                if "main_positions" not in rec or "sequence" not in rec:
                    _LOG.warning("Skipping line %d: missing data", idx)
                    continue
                P = np.array(rec["main_positions"], dtype=float)
                seq = rec["sequence"]
                energies = self.compute_energy(P, seq)
                rec.update(energies)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        _LOG.info("Wrote %d records with energy to %s", written, out_path)
        return {"written": written, "path": out_path}

