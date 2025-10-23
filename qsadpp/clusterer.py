# --*-- conding:utf-8 --*--
# @time:10/22/25 21:48
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:clusterer.py


from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class ClusterAnalyzer:
    """
    Cluster bitstrings (or coords) and select the lowest-energy cluster.

    Feature choice:
      - Prefer bit features b0..b{Lbits-1} if present
      - Else, if 'coords' present, flatten (x,y,z)*L into a vector
      - Else, fallback to 'E_total' as 1D feature (not ideal, but minimal)

    After clustering, we compute cluster-wise mean(E_total) and pick the min.
    """

    n_clusters: int = 8
    random_state: int = 0
    max_iter: int = 300

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        bcols = [c for c in df.columns if isinstance(c, str) and c.startswith("b") and c[1:].isdigit()]
        if bcols:
            return df[bcols].to_numpy(dtype=float)

        if "coords" in df.columns:
            # flatten coords
            mats = df["coords"].tolist()
            maxL = max(C.shape[0] for C in mats if C is not None)
            flat = []
            for C in mats:
                if C is None:
                    flat.append(np.zeros(3 * maxL, dtype=float))
                    continue
                v = C.reshape(-1)  # (L*3,)
                if v.size < 3 * maxL:
                    pad = np.zeros(3 * maxL - v.size, dtype=float)
                    v = np.concatenate([v, pad])
                flat.append(v)
            return np.vstack(flat)

        # fallback: energy only
        if "E_total" in df.columns:
            return df[["E_total"]].to_numpy(dtype=float)

        raise ValueError("No usable features: need b* columns or coords or E_total.")

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        X = self._feature_matrix(df)
        k = min(self.n_clusters, len(df))
        if k <= 1:
            labels = np.zeros(len(df), dtype=int)
        else:
            km = KMeans(n_clusters=k, n_init="auto", random_state=self.random_state, max_iter=self.max_iter)
            labels = km.fit_predict(X)

        out = df.copy()
        out["cluster"] = labels

        # select cluster with lowest mean E_total (if not available, use mean H_ising)
        if "E_total" in out.columns:
            means = out.groupby("cluster")["E_total"].mean()
        else:
            means = out.groupby("cluster")["H_ising"].mean()
        best_cluster = int(means.idxmin())
        return out, best_cluster
