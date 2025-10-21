# --*-- conding:utf-8 --*--
# @time:10/21/25 14:10
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cluster.py

# qsadpp/cluster.py
"""
Clustering, representative selection, and ranking.

We cluster *per group* using feature space (energy + geometry stats),
with an optional sample weight (e.g., quantum probability reweighted
by Boltzmann). We provide:

- k-medoids implementation (pure NumPy; no external deps)
- fallback KMeans (if scikit-learn is available) for speed (optional)
- representative selection: per-cluster min-E and medoid
- ranking score: S = E_A - beta * log(q_prob + eps)

Inputs are pandas DataFrames produced by the pipeline (one row per bitstring).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# Distance and k-medoids (NumPy)
# -------------------------------

def _euclidean_cdist(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    if Y is None:
        Y = X
    # (n,d) · (m,d) -> (n,m)
    a2 = (X**2).sum(axis=1, keepdims=True)
    b2 = (Y**2).sum(axis=1, keepdims=True).T
    dist2 = a2 + b2 - 2.0 * X @ Y.T
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2, out=dist2)


def kmedoids(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    seed: int = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight k-medoids (PAM-lite) in pure NumPy.

    Returns
    -------
    medoid_idx : (k,) indices into X
    labels     : (n,) cluster assignment in [0,k-1]
    """
    n = X.shape[0]
    if k <= 0 or k > n:
        raise ValueError("Invalid k for k-medoids.")
    rng = np.random.default_rng(seed)
    medoid_idx = rng.choice(n, size=k, replace=False)
    D = _euclidean_cdist(X)  # (n,n)
    w = sample_weight if sample_weight is not None else np.ones(n, dtype=float)

    for _ in range(max_iter):
        # assign
        M = D[:, medoid_idx]  # (n,k)
        labels = M.argmin(axis=1)

        # update medoids for each cluster
        new_medoid_idx = medoid_idx.copy()
        changed = False
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            # choose member minimizing weighted sum of distances to other members
            subD = D[np.ix_(members, members)]
            subw = w[members]
            # cost_i = sum_l w_l * dist(i,l)
            cost = (subD * subw[None, :]).sum(axis=1)
            best_local = members[cost.argmin()]
            if best_local != medoid_idx[j]:
                new_medoid_idx[j] = best_local
                changed = True

        medoid_idx = new_medoid_idx
        if not changed:
            break

    # final assignment
    labels = D[:, medoid_idx].argmin(axis=1)
    return medoid_idx, labels


# -------------------------------
# Public API
# -------------------------------

@dataclass
class ClusterConfig:
    method: str = "kmedoids"          # "kmedoids" or "kmeans"
    k: int = 8
    seed: int = 0
    beta_logq: float = 0.2            # for ranking S = E_A - beta*log q
    temperature: float = 1.0          # for weight w = q * exp(-E_A/T)
    feature_cols: Tuple[str, ...] = ("E_A", "E_clash", "E_mj", "R_g", "clash_cnt", "contact_cnt")


def cluster_group(
    gdf: pd.DataFrame,
    cfg: ClusterConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cluster a single group's samples and produce:
      - clusters_df: gdf with added columns [cluster, is_medoid, is_minE, S, weight]
      - reps_df: representatives per cluster (medoid + minE; deduped)
      - stats_df: per-cluster summary statistics
    """
    df = gdf.copy()

    # weights and ranking score
    q = df["q_prob"].to_numpy(dtype=float)
    E = df["E_A"].to_numpy(dtype=float)
    w = q * np.exp(-E / max(1e-9, float(cfg.temperature)))
    w = w / (w.sum() + 1e-12)
    df["weight"] = w
    df["S"] = df["E_A"] - cfg.beta_logq * np.log(df["q_prob"].to_numpy(dtype=float) + 1e-12)

    # features
    X = df.loc[:, list(cfg.feature_cols)].to_numpy(dtype=float)
    # standardize (z-score)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    Z = (X - mu) / sd

    # choose method
    method = cfg.method.lower()
    labels: np.ndarray
    medoid_idx: np.ndarray
    if method == "kmedoids":
        medoid_idx, labels = kmedoids(Z, k=min(cfg.k, len(df)), seed=cfg.seed, sample_weight=w)
    elif method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=min(cfg.k, len(df)), n_init="auto", random_state=cfg.seed)
            labels = km.fit_predict(Z)
            # pseudo-medoid: closest to centroid
            centers = km.cluster_centers_
            D = _euclidean_cdist(Z, centers)
            medoid_idx = np.array([np.where(labels == j)[0][D[labels == j, j].argmin()]
                                   if (labels == j).any() else -1
                                   for j in range(centers.shape[0])])
        except Exception:
            # fallback to kmedoids
            medoid_idx, labels = kmedoids(Z, k=min(cfg.k, len(df)), seed=cfg.seed, sample_weight=w)
            method = "kmedoids"
    else:
        raise ValueError("Unknown clustering method.")

    df["cluster"] = labels

    # representatives: medoid & min-E per cluster
    medoid_mask = np.zeros(len(df), dtype=bool)
    minE_mask = np.zeros(len(df), dtype=bool)
    for j in np.unique(labels):
        members = np.where(labels == j)[0]
        if len(members) == 0:
            continue
        # medoid
        # find index == medoid_idx[j], but medoid_idx may be -1 for empty cluster w/ kmeans
        m_idx = medoid_idx[j] if j < len(medoid_idx) and medoid_idx[j] >= 0 else None
        if m_idx is not None and m_idx in members:
            medoid_mask[m_idx] = True
        # min-E
        m2 = members[E[members].argmin()]
        minE_mask[m2] = True

    df["is_medoid"] = medoid_mask
    df["is_minE"] = minE_mask

    # build reps_df (dedupe overlaps)
    reps_df = df[df["is_medoid"] | df["is_minE"]].copy()
    reps_df = reps_df.drop_duplicates(subset=["bitstring"])

    # stats per cluster
    stats = []
    for j in sorted(np.unique(labels)):
        g = df[df["cluster"] == j]
        if len(g) == 0:
            continue
        stats.append({
            "cluster": j,
            "size": len(g),
            "E_A_med": float(g["E_A"].median()),
            "E_A_min": float(g["E_A"].min()),
            "E_A_max": float(g["E_A"].max()),
            "q_prob_sum": float(g["q_prob"].sum()),
            "weight_sum": float(g["weight"].sum()),
            "S_med": float(g["S"].median()),
        })
    stats_df = pd.DataFrame(stats).sort_values("cluster").reset_index(drop=True)

    return df, reps_df, stats_df


def select_topK_per_group(
    gdf: pd.DataFrame,
    K: int = 5,
    per_cluster_max: int = 2,
) -> pd.DataFrame:
    """
    Final selection: sort by S ascending, take at most `per_cluster_max` per cluster, up to K total.
    """
    out = []
    counts = {}
    for _, row in gdf.sort_values("S", ascending=True).iterrows():
        c = int(row["cluster"])
        if counts.get(c, 0) >= per_cluster_max:
            continue
        out.append(row)
        counts[c] = counts.get(c, 0) + 1
        if len(out) >= K:
            break
    return pd.DataFrame(out)
