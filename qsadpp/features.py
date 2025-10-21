# --*-- conding:utf-8 --*--
# @time:10/21/25 13:40
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:features.py

"""
Feature computation for QSAD post-processing.

Each bitstring corresponds to one sampled conformation.
We compute geometric and energetic descriptors used for clustering/ranking.

Main entry points
-----------------
- `compute_tierA_features(sequence, ca_coords, mj_table, weights=None)`
- `compute_features_for_group(sequence, mj_table, items)`
    where items = [(bitstring, prob, ca_coords), ...]

Outputs include:
  E_A: total pseudo-energy (weighted sum of subterms)
  E_clash: steric clash penalty
  E_mj: Miyazawa–Jernigan contact energy
  R_g: radius of gyration
  clash_cnt: number of close-contact pairs
  contact_cnt: number of hydrophobic contacts
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# Basic geometric utilities
# -----------------------------------------------------------------------------

def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances for an (L,3) array."""
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=-1))


def radius_of_gyration(coords: np.ndarray) -> float:
    """Compute radius of gyration."""
    center = coords.mean(axis=0)
    return float(np.sqrt(((coords - center) ** 2).sum(axis=1).mean()))


# -----------------------------------------------------------------------------
# Energy terms
# -----------------------------------------------------------------------------

def clash_energy(dmat: np.ndarray, threshold: float = 2.0) -> Tuple[float, int]:
    """
    Simple steric clash penalty:
    count pairs with distance < threshold and sum of squared overlaps.

    Returns
    -------
    energy, count
    """
    mask = (dmat < threshold) & (dmat > 0)
    overlap = np.maximum(threshold - dmat, 0.0)
    e = float(np.sum(overlap[mask] ** 2))
    cnt = int(mask.sum())
    return e, cnt


def contact_mask(dmat: np.ndarray, cutoff: float = 6.5) -> np.ndarray:
    """
    Boolean contact mask for distance-based contact definition.
    """
    return (dmat < cutoff) & (dmat > 0)


def mj_energy_from_seq(sequence: str, dmat: np.ndarray, mj_table: Mapping[str, Mapping[str, float]],
                       cutoff: float = 6.5) -> Tuple[float, int]:
    """
    Compute Miyazawa–Jernigan energy for contacts under a cutoff.

    Returns
    -------
    energy, n_contacts
    """
    seq = list(sequence)
    mask = contact_mask(dmat, cutoff)
    e_sum = 0.0
    n_contact = 0
    L = len(seq)
    for i in range(L):
        for j in range(i + 2, L):  # skip bonded neighbors
            if mask[i, j]:
                aa_i, aa_j = seq[i], seq[j]
                e_ij = mj_table.get(aa_i, {}).get(aa_j, 0.0)
                e_sum += e_ij
                n_contact += 1
    return float(e_sum), int(n_contact)


# -----------------------------------------------------------------------------
# Weighted sum / Tier-A combination
# -----------------------------------------------------------------------------

@dataclass
class TierAWeights:
    w_clash: float = 1.0
    w_mj: float = 1.0
    w_rg: float = 0.1

def compute_tierA_features(
    sequence: str,
    ca_coords: np.ndarray,
    mj_table: Mapping[str, Mapping[str, float]],
    weights: Optional[TierAWeights] = None,
) -> Dict[str, float]:
    """
    Compute Tier-A features for a single conformation.
    """
    coords = np.asarray(ca_coords, dtype=float)
    L = coords.shape[0]
    w = weights or TierAWeights()

    dmat = pairwise_distances(coords)
    E_clash, clash_cnt = clash_energy(dmat)
    E_mj, contact_cnt = mj_energy_from_seq(sequence, dmat, mj_table)
    R_g = radius_of_gyration(coords)

    E_A = w.w_clash * E_clash + w.w_mj * E_mj + w.w_rg * R_g

    return dict(
        E_A=float(E_A),
        E_clash=float(E_clash),
        E_mj=float(E_mj),
        R_g=float(R_g),
        clash_cnt=int(clash_cnt),
        contact_cnt=int(contact_cnt),
    )


# -----------------------------------------------------------------------------
# Group computation helpers
# -----------------------------------------------------------------------------

def compute_features_for_group(
    sequence: str,
    mj_table: Mapping[str, Mapping[str, float]],
    items: Iterable[Tuple[str, float, np.ndarray]],
    weights: Optional[TierAWeights] = None,
) -> List[Dict[str, object]]:
    """
    Compute Tier-A features for a set of conformations.

    Parameters
    ----------
    sequence : str
        Amino acid sequence
    mj_table : dict
        Nested dict of MJ potentials
    items : iterable of (bitstring, prob, ca_coords)
    weights : TierAWeights, optional

    Returns
    -------
    List of dict with features per conformation.
    """
    out = []
    for bitstring, prob, ca in items:
        feat = compute_tierA_features(sequence, ca, mj_table, weights=weights)
        feat["bitstring"] = bitstring
        feat["q_prob"] = float(prob)
        out.append(feat)
    return out


# -----------------------------------------------------------------------------
# MJ table conversion utility
# -----------------------------------------------------------------------------

def dense_to_mj_table(matrix: np.ndarray, aa_order: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """
    Convert a dense (20x20) matrix into a nested dict MJ table.
    """
    assert matrix.shape == (len(aa_order), len(aa_order))
    out: Dict[str, Dict[str, float]] = {}
    for i, ai in enumerate(aa_order):
        out[ai] = {aj: float(matrix[i, j]) for j, aj in enumerate(aa_order)}
    return out
