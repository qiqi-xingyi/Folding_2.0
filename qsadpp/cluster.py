# --*-- conding:utf-8 --*--
# @time:10/21/25 14:10
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cluster.py


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ClusterConfig:
    method: str = "kmedoids"  # or "kmeans"
    k: int = 8
    seed: int = 0
    beta_logq: float = 0.2
    temperature: float = 1.0

    def __post_init__(self):
        if self.n_clusters is not None and self.k == ClusterConfig.__dataclass_fields__["k"].default:
            self.k = int(self.n_clusters)


_NUM_CLIP = 1e6  # clip to avoid overflow in dot products


def _select_numeric_columns(df: pd.DataFrame, candidate_cols: Optional[Sequence[str]] = None) -> List[str]:
    if candidate_cols is None:
        candidate_cols = ["E_A", "E_clash", "E_mj", "R_g"]
    cols = [c for c in candidate_cols if c in df.columns]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def _prepare_numeric_matrix(df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    cols = _select_numeric_columns(df, feature_cols)
    if not cols:
        raise ValueError("No numeric feature columns found for clustering.")

    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    finite_mask = np.isfinite(X)
    if not finite_mask.all():
        col_median = np.nanmedian(np.where(finite_mask, X, np.nan), axis=0)
        col_median = np.where(np.isfinite(col_median), col_median, 0.0)
        inds = np.where(~finite_mask)
        X[inds] = col_median[inds[1]]

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    X = np.clip(X, -_NUM_CLIP, _NUM_CLIP)
    return X, cols


def _safe_sqeuclidean(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    if Y is None:
        Y = X
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)

    a2 = np.einsum("ij,ij->i", X, X)[:, None]
    b2 = np.einsum("ij,ij->i", Y, Y)[None, :]

    K = X @ Y.T
    if not np.isfinite(K).all():
        K = np.where(np.isfinite(K), K, 0.0)

    dist2 = a2 + b2 - 2.0 * K
    dist2 = np.where(np.isfinite(dist2), dist2, np.inf)
    np.maximum(dist2, 0.0, out=dist2)
    return dist2


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = []
    i0 = int(rng.integers(0, n))
    centers.append(X[i0])
    for _ in range(1, k):
        d2 = _safe_sqeuclidean(X, np.vstack(centers)).min(axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers.append(X[idx])
    return np.vstack(centers)


def _kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    k = min(k, n)

    C = _kmeans_pp_init(X, k, rng)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        d2 = _safe_sqeuclidean(X, C)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                C[j] = X[int(rng.integers(0, n))]
            else:
                C[j] = X[mask].mean(axis=0)
    return labels, C


def _pam(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    k = min(k, n)

    medoid_idx = rng.choice(n, size=k, replace=False)
    D = _safe_sqeuclidean(X)

    def total_cost(midx: np.ndarray) -> float:
        dmin = D[:, midx].min(axis=1)
        return float(dmin.sum())

    best_cost = total_cost(medoid_idx)

    for _ in range(max_iter):
        improved = False
        for i in range(k):
            for h in range(n):
                if h in medoid_idx:
                    continue
                new_idx = medoid_idx.copy()
                new_idx[i] = h
                c = total_cost(new_idx)
                if c + 1e-9 < best_cost:
                    best_cost = c
                    medoid_idx = new_idx
                    improved = True
        if not improved:
            break

    labels = D[:, medoid_idx].argmin(axis=1)
    medoids = X[medoid_idx]
    return labels, medoids


def cluster_group(
    df: pd.DataFrame,
    cfg: ClusterConfig,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.Series, np.ndarray, List[str]]:
    """
    Cluster a group's dataframe using numeric features.
    Returns (labels, centers, used_features) with labels aligned to df.index.
    """
    X, used = _prepare_numeric_matrix(df, feature_cols)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty matrix after cleaning.")
    k = max(1, min(cfg.k, n))

    if np.allclose(X.std(axis=0), 0.0):
        labels = np.zeros(n, dtype=int)
        centers = X[:1].copy()
        return pd.Series(labels, index=df.index, name="cluster"), centers, used

    if cfg.method.lower() == "kmeans":
        labels, centers = _kmeans(X, k=k, seed=cfg.seed)
    else:
        labels, centers = _pam(X, k=k, seed=cfg.seed)

    return pd.Series(labels, index=df.index, name="cluster"), centers, used


def select_topK_per_group(
    df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    per_cluster_max: int = 2,
    beta_logq: float = 0.2,
) -> pd.DataFrame:
    """
    Pick representative rows per cluster using score:
        S = E_A - beta_logq * log(q_prob_clipped)
    Lower score is better.
    """
    work = df.copy()
    if labels is not None:
        work["cluster"] = labels.values
    elif "cluster" not in work.columns:
        raise ValueError("Cluster labels are required: pass `labels` or provide a 'cluster' column in df.")


    q = pd.to_numeric(work.get("q_prob", 1.0), errors="coerce").fillna(1.0)
    q = q.clip(lower=1e-300)
    EA = pd.to_numeric(work.get("E_A", 0.0), errors="coerce").fillna(0.0)
    score = EA - beta_logq * np.log(q)
    work["__score__"] = score.replace([np.inf, -np.inf], np.nan).fillna(1e9)

    reps = []
    for c, g in work.groupby("cluster"):
        reps.append(g.sort_values("__score__").head(per_cluster_max))
    out = pd.concat(reps, ignore_index=True).drop(columns="__score__")
    return out

