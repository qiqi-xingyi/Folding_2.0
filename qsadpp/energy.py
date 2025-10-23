# --*-- conding:utf-8 --*--
# @time:10/22/25 21:41
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:energy.py

# --*-- coding:utf-8 --*--
# qsadpp/energy.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def _pairwise_distances(C: np.ndarray) -> np.ndarray:
    diffs = C[:, None, :] - C[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=-1))


def _clash_energy(dmat: np.ndarray, threshold: float = 1.0) -> Tuple[float, int]:
    mask = (dmat < threshold) & (dmat > 0)
    overlap = np.maximum(threshold - dmat, 0.0)
    e = float(np.sum(overlap[mask] ** 2))
    cnt = int(mask.sum())
    return e, cnt


# a tiny built-in 20x20 potential (zeros), to keep code self-contained
_AA = list("ARNDCQEGHILKMFPSTWYV")
_MJ = np.zeros((20, 20), dtype=float)

def _mj_energy(sequence: Optional[str], dmat: np.ndarray, cutoff: float = 2.5) -> Tuple[float, int]:
    """Simple MJ-like contact energy. If sequence is absent or letters not in AA20, returns 0."""
    if not sequence:
        return 0.0, 0
    seq = list(sequence)
    n = len(seq)
    contact = 0
    e_sum = 0.0
    for i in range(n):
        ai = seq[i] if i < len(seq) else "A"
        if ai not in _AA:
            continue
        ii = _AA.index(ai)
        for j in range(i + 2, n):  # skip bonded neighbors
            if dmat[i, j] > 0 and dmat[i, j] < cutoff:
                aj = seq[j] if j < len(seq) else "A"
                if aj not in _AA:
                    continue
                jj = _AA.index(aj)
                e_sum += _MJ[ii, jj]
                contact += 1
    return float(e_sum), int(contact)


@dataclass
class EnergyCalculator:
    """
    Compute:
      - Structural pseudo-energy: E_total = w_clash*E_clash + w_mj*E_mj
      - Ising-like Hamiltonian:   H = - sum_i h_i s_i - sum_{i<j} J_ij s_i s_j
        where s_i in {+1,-1} derived from bit b_i: s = 2*b - 1
    """

    w_clash: float = 1.0
    w_mj: float = 1.0
    clash_threshold: float = 1.0
    mj_cutoff: float = 2.5

    def compute(self,
                df: pd.DataFrame,
                h: Optional[Sequence[float]] = None,
                J: Optional[np.ndarray] = None,
                sequence_col: str = "sequence") -> pd.DataFrame:
        out = df.copy()

        # Structural energy from coords
        E_total = []
        E_clash_arr = []
        E_mj_arr = []
        contact_cnt = []
        clash_cnt = []

        for _, row in out.iterrows():
            C = row.get("coords", None)
            seq = row.get(sequence_col, None)
            if C is None:
                # if mapping not done, structural terms = 0
                E_clash_arr.append(0.0); clash_cnt.append(0)
                E_mj_arr.append(0.0); contact_cnt.append(0)
                E_total.append(0.0)
                continue

            dmat = _pairwise_distances(C)
            e_clash, n_clash = _clash_energy(dmat, threshold=self.clash_threshold)
            e_mj, n_contact = _mj_energy(seq, dmat, cutoff=self.mj_cutoff)

            e_sum = self.w_clash * e_clash + self.w_mj * e_mj
            E_clash_arr.append(e_clash); clash_cnt.append(n_clash)
            E_mj_arr.append(e_mj);     contact_cnt.append(n_contact)
            E_total.append(e_sum)

        out["E_clash"] = np.array(E_clash_arr, dtype=float)
        out["clash_cnt"] = np.array(clash_cnt, dtype=int)
        out["E_mj"] = np.array(E_mj_arr, dtype=float)
        out["contact_cnt"] = np.array(contact_cnt, dtype=int)
        out["E_total"] = np.array(E_total, dtype=float)

        # Hamiltonian on bit-level
        # Build spin vector s from bitstring (use full bitstring)
        H_list = []
        for s in out["bitstring"].astype(str):
            b = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
            spins = 2 * b - 1  # {0,1} -> {-1,+1}
            # local fields
            H_local = 0.0
            if h is not None:
                n = min(len(h), len(spins))
                H_local = - float(np.dot(spins[:n], np.asarray(h[:n], dtype=float)))
            # couplings
            H_pair = 0.0
            if J is not None:
                Jarr = np.asarray(J, dtype=float)
                m = min(Jarr.shape[0], len(spins))
                S = spins[:m].astype(float)
                # use upper triangle i<j
                H_pair = 0.0
                for i in range(m):
                    for j in range(i+1, m):
                        H_pair += - Jarr[i, j] * S[i] * S[j]
            H_list.append(H_local + H_pair)

        out["H_ising"] = np.array(H_list, dtype=float)
        return out
