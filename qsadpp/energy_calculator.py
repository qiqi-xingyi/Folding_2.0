# --*-- conding:utf-8 --*--
# @time:10/23/25 17:08
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:energy_calculator.py

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import numpy as np

_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass
class EnergyConfig:
    r_min: float = 0.5
    d0: float = 0.57735  # ideal bond length (≈ 1/sqrt(3))
    lambda_overlap: float = 1000.0
    weights: Dict[str, float] = field(default_factory=lambda: {"steric": 1.0, "geom": 0.5, "bond": 0.2})
    normalize: bool = True
    output_path: str = "decoded_with_energy.jsonl"


class LatticeEnergyCalculator:
    """
    Compute approximate conformational energy for main-chain-only lattice proteins.
    Input: records from CoordinateBatchDecoder (with main_positions).
    Output: JSONL file where each line contains the original record plus computed energy.
    """

    def __init__(self, cfg: EnergyConfig = EnergyConfig()):
        self.cfg = cfg

    # ---------- core computation ----------
    def compute_energy(self, main_positions: np.ndarray) -> Dict[str, float]:
        """
        Compute steric, geometric, and bond energy for one conformation.
        """
        N = len(main_positions)
        if N < 3:
            return {"E_total": 0.0, "E_steric": 0.0, "E_geom": 0.0, "E_bond": 0.0}

        # 1. bond energy
        v = np.diff(main_positions, axis=0)
        bond_lengths = np.linalg.norm(v, axis=1)
        E_bond = np.sum((bond_lengths - self.cfg.d0) ** 2)

        # 2. geometric energy
        v1 = v[:-1]
        v2 = v[1:]
        cos_thetas = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        )
        cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
        E_geom = np.sum(1.0 - cos_thetas)

        # 3. steric repulsion
        diff = main_positions[:, None, :] - main_positions[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        mask = np.triu(np.ones_like(dist, dtype=bool), k=2)  # exclude |i-j| <=1
        D = dist[mask]
        E_steric = np.sum((self.cfg.r_min - D[D < self.cfg.r_min]) ** 2)
        E_steric += np.sum(D[D == 0]) * self.cfg.lambda_overlap  # overlaps penalty

        # weighted sum
        E_total = (
            self.cfg.weights["steric"] * E_steric
            + self.cfg.weights["geom"] * E_geom
            + self.cfg.weights["bond"] * E_bond
        )

        if self.cfg.normalize:
            E_total /= N

        return {
            "E_total": float(E_total),
            "E_steric": float(E_steric),
            "E_geom": float(E_geom),
            "E_bond": float(E_bond),
        }

    # ---------- batch I/O ----------
    def evaluate_jsonl(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read decoded coordinate JSONL file line-by-line,
        compute energy for each conformation, and write a new JSONL with added fields.
        """
        out_path = output_path or self.cfg.output_path
        written = 0
        with open(input_path, "r", encoding="utf-8") as fin, open(
            out_path, "w", encoding="utf-8"
        ) as fout:
            for idx, line in enumerate(fin):
                if limit and idx >= limit:
                    break
                rec = json.loads(line)
                if "main_positions" not in rec:
                    _LOG.warning("Skipping line %d: missing main_positions", idx)
                    continue
                P = np.array(rec["main_positions"], dtype=float)
                energies = self.compute_energy(P)
                rec.update(energies)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        _LOG.info("Wrote %d records with energy to %s", written, out_path)
        return {"written": written, "path": out_path}
