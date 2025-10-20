# --*-- conding:utf-8 --*--
# @time:10/20/25 01:55
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:clustering.py

from __future__ import annotations
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import AnalyzerConfig
from .utils import bitstrings_to_array, pairwise_hamming

try:
    from sklearn.cluster import AgglomerativeClustering
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

def cluster_by_group(
    df: pd.DataFrame,
    cfg: AnalyzerConfig,
    k: int = 3,
    linkage: str = "average",
    distance_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Cluster high-probability states within each group (e.g., beta),
    aggregating probabilities across seeds. Uses Hamming distance.
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for clustering.")

    gkeys = list(cfg.group_keys)
    rows: List[Dict] = []

    agg = (
        df.groupby(gkeys + [cfg.bitstring_col], dropna=False)[cfg.prob_col]
        .sum()
        .reset_index()
    )

    for _, grp in agg.groupby(gkeys, dropna=False):
        bitstrings = grp[cfg.bitstring_col].astype(str).tolist()
        probs = grp[cfg.prob_col].to_numpy(dtype=float)
        total = probs.sum()
        if total <= 0:
            continue
        probs = probs / total

        X = bitstrings_to_array(bitstrings)
        if X.shape[0] <= 1:
            base = {k: grp.iloc[0][k] for k in gkeys}
            rows.append({**base, "cluster_id": 0, "bitstring": bitstrings[0], "prob_sum": float(probs[0])})
            continue

        D = pairwise_hamming(X)
        model = AgglomerativeClustering(
            n_clusters=None if distance_threshold is not None else k,
            distance_threshold=distance_threshold,
            linkage=linkage
        )
        # metric/affinity compatibility
        try:
            model.set_params(affinity="precomputed")
        except Exception:
            try:
                model.set_params(metric="precomputed")
            except Exception:
                pass

        labels = model.fit_predict(D)
        base = {kk: grp.iloc[0][kk] for kk in gkeys}
        for label, b, p in zip(labels, bitstrings, probs):
            rows.append({**base, "cluster_id": int(label), "bitstring": b, "prob_sum": float(p)})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(gkeys + ["cluster_id", "prob_sum"], ascending=[True]*len(gkeys) + [True, False])
    return out
