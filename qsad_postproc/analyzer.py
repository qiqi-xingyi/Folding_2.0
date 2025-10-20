# --*-- conding:utf-8 --*--
# @time:10/20/25 01:54
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:analyzer.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from .config import AnalyzerConfig
from .metrics import shannon_entropy, effective_sample_size
from .utils import bitstrings_to_array, pairwise_hamming

def _weighted_mode(bitstrings: List[str], probs: np.ndarray) -> str:
    idx = int(np.argmax(probs))
    return bitstrings[idx]

class SamplingAnalyzer:
    """
    Post-processing for QSaD sampling outputs.

    Expected minimal columns:
      - bitstring, count, prob, seed, shots
    Group keys default to ('beta',), but any tuple of keys is allowed.
    """

    def __init__(self, df: pd.DataFrame, cfg: Optional[AnalyzerConfig] = None):
        self.df = df.copy()
        self.cfg = cfg or AnalyzerConfig()

        required = {self.cfg.bitstring_col, self.cfg.count_col, self.cfg.prob_col,
                    self.cfg.seed_col, self.cfg.shots_col}
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.cfg.n_qubits is None:
            if len(self.df) == 0:
                self.cfg.n_qubits = 0
            else:
                self.cfg.n_qubits = len(str(self.df[self.cfg.bitstring_col].iloc[0]))

        if self.cfg.prob_col not in self.df.columns or self.df[self.cfg.prob_col].isna().any():
            self.df[self.cfg.prob_col] = self.df[self.cfg.count_col] / self.df[self.cfg.shots_col]

    # ------------- per-experiment summaries -------------

    def per_experiment_summary(self) -> pd.DataFrame:
        gkeys = list(self.cfg.group_keys) + [self.cfg.seed_col]

        def _summ(grp: pd.DataFrame) -> pd.Series:
            probs = grp[self.cfg.prob_col].to_numpy(dtype=float)
            bitstrings = grp[self.cfg.bitstring_col].astype(str).tolist()

            total = probs.sum()
            if total > 0:
                probs = probs / total

            H = shannon_entropy(probs)
            ESS = effective_sample_size(probs)
            distinct = len(grp)
            max_states = 2 ** int(self.cfg.n_qubits or 0)
            distinct_ratio = float(distinct) / float(max_states) if max_states > 0 else np.nan
            top_idx = int(np.argmax(probs))
            return pd.Series({
                "entropy_bits": H,
                "ess": ESS,
                "distinct": distinct,
                "distinct_ratio": distinct_ratio,
                "top_bitstring": bitstrings[top_idx],
                "top_prob": probs[top_idx],
            })

        out = self.df.groupby(gkeys, dropna=False).apply(_summ).reset_index()
        return out

    def per_group_aggregate(self, agg: str = "mean") -> pd.DataFrame:
        exp = self.per_experiment_summary()
        gkeys = list(self.cfg.group_keys)
        if agg == "mean":
            out = exp.groupby(gkeys, dropna=False).mean(numeric_only=True).reset_index()
        elif agg == "median":
            out = exp.groupby(gkeys, dropna=False).median(numeric_only=True).reset_index()
        else:
            raise ValueError("agg must be 'mean' or 'median'")
        return out

    # ------------- bitwise marginals -------------

    def bit_marginals(self) -> pd.DataFrame:
        gkeys = list(self.cfg.group_keys) + [self.cfg.seed_col]
        rows: List[Dict] = []

        for _, grp in self.df.groupby(gkeys, dropna=False):
            bitstrings = grp[self.cfg.bitstring_col].astype(str).tolist()
            probs = grp[self.cfg.prob_col].to_numpy(dtype=float)
            total = probs.sum()
            if total <= 0:
                continue
            probs = probs / total

            A = bitstrings_to_array(bitstrings)  # (N, L)
            p1 = (probs[:, None] * A).sum(axis=0)
            base = {k: grp.iloc[0][k] for k in gkeys}
            for q, val in enumerate(p1):
                rows.append({**base, "qubit": int(q), "p1": float(val)})
        return pd.DataFrame(rows)

    # ------------- top-k and Hamming -------------

    def topk_states(self, k: int = 10) -> pd.DataFrame:
        gkeys = list(self.cfg.group_keys) + [self.cfg.seed_col]
        df = self.df.copy()
        df[self.cfg.prob_col] = df.groupby(gkeys)[self.cfg.prob_col].transform(lambda x: x / (x.sum() or 1.0))
        df["rank"] = df.groupby(gkeys)[self.cfg.prob_col].rank(ascending=False, method="first")
        out = df[df["rank"] <= k].sort_values(gkeys + ["rank"]).reset_index(drop=True)
        out["rank"] = out["rank"].astype(int)
        return out

    def mode_and_hamming(self) -> pd.DataFrame:
        gkeys = list(self.cfg.group_keys) + [self.cfg.seed_col]
        rows: List[Dict] = []

        for _, grp in self.df.groupby(gkeys, dropna=False):
            bitstrings = grp[self.cfg.bitstring_col].astype(str).tolist()
            probs = grp[self.cfg.prob_col].to_numpy(dtype=float)
            total = probs.sum()
            if total <= 0:
                continue
            probs = probs / total

            mode = _weighted_mode(bitstrings, probs)
            arr = bitstrings_to_array(bitstrings)
            mode_arr = bitstrings_to_array([mode])[0]
            hamm = float(np.sum(probs * np.sum(arr != mode_arr, axis=1)))
            base = {k: grp.iloc[0][k] for k in gkeys}
            rows.append({**base, "mode": mode, "mean_hamming_to_mode": hamm})
        return pd.DataFrame(rows)

    # ------------- constructors -------------

    @classmethod
    def from_csv(cls, path: str, cfg: Optional[AnalyzerConfig] = None) -> "SamplingAnalyzer":
        df = pd.read_csv(path)
        return cls(df, cfg=cfg)

    @classmethod
    def from_parquet(cls, path: str, cfg: Optional[AnalyzerConfig] = None) -> "SamplingAnalyzer":
        df = pd.read_parquet(path)
        return cls(df, cfg=cfg)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, cfg: Optional[AnalyzerConfig] = None) -> "SamplingAnalyzer":
        return cls(df, cfg=cfg)
