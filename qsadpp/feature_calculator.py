# --*-- conding:utf-8 --*--
# @time:10/30/25 18:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:feature_calculator.py

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # only used when output_format == "parquet"
    _PANDAS_OK = True
except Exception:
    _PANDAS_OK = False


# =========================
#         CONFIG
# =========================

@dataclass
class FeatureConfig:
    # IO
    output_path: str = "features.jsonl"
    output_format: str = "jsonl"  # "jsonl" | "parquet"
    carry_keys: Tuple[str, ...] = ("pdb_id", "case_id", "residue_range", "id", "bitstring")

    # Global/density
    contact_cutoff: float = 8.0
    clash_min_dist: float = 3.4
    nn_exclude: int = 1  # exclude |i-j| <= nn_exclude for NN distances

    # Secondary-structure proxies (Cα-only)
    alpha_i3_max: float = 5.4
    alpha_i4_max: float = 6.2
    beta_pair_max: float = 7.0  # |i-j|>=3 and within this => beta-like

    # Burial / packing (Cα proxies)
    burial_r: float = 6.0  # Gaussian kernel radius
    pack_rep_max: float = 4.5
    pack_att_min: float = 5.5
    pack_att_max: float = 7.5

    # Pseudo-atom reconstruction
    # All are rough, Cα-based estimates sufficient for lightweight features.
    use_pseudo_atoms: bool = True
    ca_n_dist: float = 1.46   # Å
    ca_cb_dist: float = 1.52  # Å
    ca_o_dist: float = 1.24   # Å (placed roughly along tangent)

    # H-bond (pseudo N-O) distance window
    hb_min: float = 2.6
    hb_max: float = 3.2

    # Sequence classes
    hydrophobic: str = "AVLIMFWYV"
    charged: str = "KRDEH"
    polar: str = "STNQYC"

    # Fail-soft behavior
    ignore_bad_rows: bool = True  # if True, write an _feature_error field instead of raising


# =========================
#      CORE CALCULATOR
# =========================

class StructuralFeatureCalculator:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

        # Precompute residue sets
        self._hydrophobic = set(cfg.hydrophobic.upper())
        self._charged = set(cfg.charged.upper())
        self._polar = set(cfg.polar.upper())

        # Validate output format
        if self.cfg.output_format not in ("jsonl", "parquet"):
            raise ValueError("output_format must be 'jsonl' or 'parquet'.")

    # ---------- public API ----------

    def evaluate_jsonl(self, decoded_jsonl_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Read a JSONL containing at least {sequence, main_positions}, compute features per record,
        and write to cfg.output_path (or output_path if provided).
        Returns summary dict.
        """
        out_path = output_path or self.cfg.output_path
        if self.cfg.output_format == "jsonl":
            written = self._run_streaming_jsonl(decoded_jsonl_path, out_path)
        else:
            if not _PANDAS_OK:
                raise RuntimeError("Pandas not available but output_format='parquet' requested.")
            written = self._run_to_parquet(decoded_jsonl_path, out_path)

        return {"written": written, "output_path": out_path, "format": self.cfg.output_format}

    # ---------- streaming writers ----------

    def _run_streaming_jsonl(self, in_path: str, out_path: str) -> int:
        cnt = 0
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                try:
                    features = self._compute_features(rec)
                except Exception as e:
                    if not self.cfg.ignore_bad_rows:
                        raise
                    features = {"_feature_error": str(e)}
                    for k in self.cfg.carry_keys:
                        if k in rec:
                            features[k] = rec[k]
                fout.write(json.dumps(features) + "\n")
                cnt += 1
        return cnt

    def _run_to_parquet(self, in_path: str, out_path: str) -> int:
        rows: List[Dict[str, Any]] = []
        with open(in_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                try:
                    rows.append(self._compute_features(rec))
                except Exception as e:
                    if not self.cfg.ignore_bad_rows:
                        raise
                    row = {"_feature_error": str(e)}
                    for k in self.cfg.carry_keys:
                        if k in rec:
                            row[k] = rec[k]
                    rows.append(row)
        if not rows:
            # Ensure an empty parquet still exists
            with open(out_path, "wb") as f:
                pass
            return 0
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_parquet(out_path, index=False)
        return len(rows)

    # ---------- per-record feature computation ----------

    def _compute_features(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        seq = str(rec["sequence"])
        X = self._as_np(rec.get("main_positions", []))  # (N,3)
        self._validate_lengths(seq, X)

        N = X.shape[0]
        D = self._pdist(X) if N else np.zeros((0, 0), dtype=float)

        out: Dict[str, Any] = {}
        # carry meta keys first (for readability/debug)
        for k in self.cfg.carry_keys:
            if k in rec:
                out[k] = rec[k]

        # ---------- A: global/density ----------
        out["length"] = float(N)
        out["Rg"] = self._radius_of_gyration(X) if N else 0.0
        out["end_to_end"] = float(np.linalg.norm(X[-1] - X[0])) if N > 1 else 0.0
        out["contact_density"] = self._contact_density(D) if N else 0.0

        # ---------- B: clashes / nearest-neighbor ----------
        out["clash_count"] = float(self._clash_count(D))
        p10, p50, p90 = self._nn_dists(D)
        out["nn_dist_p10"] = p10
        out["nn_dist_p50"] = p50
        out["nn_dist_p90"] = p90

        # ---------- C: secondary-structure-ish proxies ----------
        out["alpha_hits"] = float(self._alpha_hits(X))
        out["beta_hits"] = float(self._beta_hits(D))
        denom = max(1.0, float(N))
        out["rama_allowed_ratio"] = (out["alpha_hits"] + out["beta_hits"]) / denom

        # ---------- D: burial / packing ----------
        b_mean, b_max = self._burial_stats(D)
        out["burial_mean"] = b_mean
        out["burial_max"] = b_max
        rep, att = self._packing_counts(D)
        out["packing_rep_count"] = float(rep)
        out["packing_att_count"] = float(att)

        # ---------- E: sequence stats ----------
        out.update(self._seq_stats(seq))

        # ---------- F: pseudo atoms (optional) ----------
        if self.cfg.use_pseudo_atoms and N >= 2:
            Npos, Opos, CBpos = self._pseudo_atoms_from_ca(X)
            # N-O hbond-like counts
            if len(Npos) and len(Opos):
                D_no = self._pdist_cross(Npos, Opos)
                mask = (D_no >= self.cfg.hb_min) & (D_no <= self.cfg.hb_max)
                out["hb_count_pseudo"] = float(mask.sum())
                out["hb_density_pseudo"] = float(mask.sum()) / max(1.0, float(N))
            else:
                out["hb_count_pseudo"] = 0.0
                out["hb_density_pseudo"] = 0.0

            # Cβ–Cβ packing windows (using pseudo CB)
            if len(CBpos) >= 2:
                D_cb = self._pdist(CBpos)
                m = np.triu(np.ones_like(D_cb, dtype=bool), k=2)
                out["cbeta_rep_count"] = float((D_cb[m] < self.cfg.pack_rep_max).sum())
                m2 = (D_cb >= self.cfg.pack_att_min) & (D_cb <= self.cfg.pack_att_max)
                out["cbeta_att_count"] = float((m2 & m).sum())
            else:
                out["cbeta_rep_count"] = 0.0
                out["cbeta_att_count"] = 0.0
        else:
            out["hb_count_pseudo"] = 0.0
            out["hb_density_pseudo"] = 0.0
            out["cbeta_rep_count"] = 0.0
            out["cbeta_att_count"] = 0.0

        # ---------- G: include energy components present in rec (EXCLUDING E_total) ----------
        for k, v in rec.items():
            if isinstance(v, (int, float)) and k.startswith("E_") and k != "E_total":
                out[k] = float(v)

        return out

    # =========================
    #        UTILITIES
    # =========================

    @staticmethod
    def _as_np(positions: List[List[float]]) -> np.ndarray:
        X = np.asarray(positions, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"main_positions must be (N,3), got {X.shape}")
        return X

    @staticmethod
    def _validate_lengths(seq: str, X: np.ndarray) -> None:
        if len(seq) != len(X):
            raise ValueError(f"len(sequence)={len(seq)} != len(main_positions)={len(X)}")

    @staticmethod
    def _pdist(X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    @staticmethod
    def _pdist_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # shape: (len(A), len(B))
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    @staticmethod
    def _radius_of_gyration(X: np.ndarray) -> float:
        c = X.mean(axis=0, keepdims=True)
        return float(np.sqrt(((X - c) ** 2).sum(axis=1).mean()))

    def _contact_density(self, D: np.ndarray) -> float:
        N = D.shape[0]
        if N < 3:
            return 0.0
        mask = np.triu(np.ones_like(D, dtype=bool), k=2)
        return float((D[mask] < self.cfg.contact_cutoff).sum()) / float(N)

    def _clash_count(self, D: np.ndarray) -> int:
        if D.size == 0:
            return 0
        mask = np.triu(np.ones_like(D, dtype=bool), k=2)
        return int((D[mask] < self.cfg.clash_min_dist).sum())

    def _nn_dists(self, D: np.ndarray) -> Tuple[float, float, float]:
        N = D.shape[0]
        if N == 0:
            return (0.0, 0.0, 0.0)
        NN: List[float] = []
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            k = self.cfg.nn_exclude
            for t in range(1, k + 1):
                if i - t >= 0:
                    mask[i - t] = False
                if i + t < N:
                    mask[i + t] = False
            cand = D[i][mask]
            if cand.size:
                NN.append(float(np.min(cand)))
        if not NN:
            return (0.0, 0.0, 0.0)
        arr = np.array(NN, dtype=float)
        return (float(np.percentile(arr, 10)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 90)))

    def _alpha_hits(self, X: np.ndarray) -> int:
        N = len(X)
        cnt = 0
        for i in range(N):
            j = i + 3
            if j < N and np.linalg.norm(X[j] - X[i]) <= self.cfg.alpha_i3_max:
                cnt += 1
            j = i + 4
            if j < N and np.linalg.norm(X[j] - X[i]) <= self.cfg.alpha_i4_max:
                cnt += 1
        return cnt

    def _beta_hits(self, D: np.ndarray) -> int:
        N = D.shape[0]
        if N < 3:
            return 0
        mask = np.triu(np.ones_like(D, dtype=bool), k=3)  # |i-j| >= 3
        return int((D[mask] <= self.cfg.beta_pair_max).sum())

    def _burial_stats(self, D: np.ndarray) -> Tuple[float, float]:
        N = D.shape[0]
        if N == 0:
            return (0.0, 0.0)
        B = np.exp(- (D / self.cfg.burial_r) ** 2)
        np.fill_diagonal(B, 0.0)
        vec = B.sum(axis=1)
        return float(vec.mean()), float(vec.max(initial=0.0))

    def _packing_counts(self, D: np.ndarray) -> Tuple[int, int]:
        if D.size == 0:
            return (0, 0)
        mask = np.triu(np.ones_like(D, dtype=bool), k=2)
        rep = int(((D < self.cfg.pack_rep_max) & mask).sum())
        att = int((((D >= self.cfg.pack_att_min) & (D <= self.cfg.pack_att_max)) & mask).sum())
        return rep, att

    def _seq_stats(self, seq: str) -> Dict[str, float]:
        N = len(seq)
        if N == 0:
            return dict(frac_hydrophobic=0.0, frac_charged=0.0, frac_polar=0.0,
                        n_proline=0.0, n_glycine=0.0)
        s = seq.upper()
        n_h = sum(aa in self._hydrophobic for aa in s)
        n_c = sum(aa in self._charged for aa in s)
        n_p = sum(aa in self._polar for aa in s)
        return dict(
            frac_hydrophobic=n_h / N,
            frac_charged=n_c / N,
            frac_polar=n_p / N,
            n_proline=float(s.count("P")),
            n_glycine=float(s.count("G")),
        )

    # ---------- pseudo-atom reconstruction ----------
    def _pseudo_atoms_from_ca(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Very light Cα-only pseudo-atom construction:
        - Tangent t_i ~ normalize((CA_{i+1}-CA_i) + (CA_i-CA_{i-1}))  (Frenet-like)
        - Normal n_i ~ normalize(cross(t_{i-1}, t_i)) (fallback: perpendicular of local segment)
        - Place N_i at CA_i - t_i * d_CN_alpha  (pointing backward along chain)
        - Place O_i at CA_i + t_i * d_CO_alpha  (pointing forward)
        - Place Cβ_i at CA_i + n_i * d_Cbeta_alpha (rough side direction)
        Returns arrays (Npos, Opos, CBpos), each shape ~ (N,3), with NaN rows for undefined entries pruned.
        """
        N = len(X)
        if N == 1:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))

        # build tangents
        fwd = np.zeros_like(X)
        bwd = np.zeros_like(X)
        fwd[:-1] = X[1:] - X[:-1]
        bwd[1:] = X[1:] - X[:-1]
        bwd = -bwd
        t = fwd + bwd
        # endpoints: use single-sided difference
        t[0] = X[1] - X[0]
        t[-1] = X[-1] - X[-2]
        t = self._safe_normalize_rows(t)

        # a rough perpendicular "normal" using local segment cross a fixed axis then orth with t
        # if cross is near zero, rotate the vector
        n = np.cross(t, np.array([0.0, 0.0, 1.0]))
        zero_mask = (np.linalg.norm(n, axis=1) < 1e-8)
        n[zero_mask] = np.cross(t[zero_mask], np.array([0.0, 1.0, 0.0]))
        n = self._safe_normalize_rows(n)

        Npos = X - t * self.cfg.ca_n_dist
        Opos = X + t * self.cfg.ca_o_dist
        CBpos = X + n * self.cfg.ca_cb_dist

        return Npos, Opos, CBpos

    @staticmethod
    def _safe_normalize_rows(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        L = np.linalg.norm(V, axis=1, keepdims=True)
        L[L < eps] = 1.0
        return V / L
