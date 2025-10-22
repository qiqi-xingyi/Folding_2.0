# --*-- conding:utf-8 --*--
# @time:10/21/25 14:10
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cluster.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ClusterConfig:
    method: str = "kmedoids"  # or "kmeans"
    k: int = 8
    seed: int = 0
    beta_logq: float = 0.2     # used later for scoring S = E_A - beta*log(q)
    temperature: float = 1.0   # not used here directly

# -------------------------------
# Numeric matrix preparation
# -------------------------------

_NUM_CLIP = 1e6  # hard clip to avoid overflow in dot products


def _select_numeric_columns(df: pd.DataFrame, candidate_cols: Optional[Sequence[str]] = None) -> List[str]:
    if candidate_cols is None:
        # default features if caller did not specify
        candidate_cols = ["E_A", "E_clash", "E_mj", "R_g"]
    cols = []
    for c in candidate_cols:
        if c in df.columns:
            cols.append(c)
    # keep only numeric
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def _prepare_numeric_matrix(df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Build a clean numeric matrix:
      - select numeric feature columns,
      - coerce to float64,
      - replace non-finite with column medians,
      - standardize (z-score),
      - clip to [-_NUM_CLIP, _NUM_CLIP].
    """
    cols = _select_numeric_columns(df, feature_cols)
    if not cols:
        raise ValueError("No numeric feature columns found for clustering.")

    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    # replace non-finite with column medians
    finite_mask = np.isfinite(X)
    if not finite_mask.all():
        col_median = np.nanmedian(np.where(finite_mask, X, np.nan), axis=0)
        # if any column is all-NaN, set its median to 0
        col_median = np.where(np.isfinite(col_median), col_median, 0.0)
        inds = np.where(~finite_mask)
        X[inds] = col_median[inds[1]]

    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    # clip
    X = np.clip(X, -_NUM_CLIP, _NUM_CLIP)

    return X, cols


def _safe_sqeuclidean(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Numerically safer squared Euclidean distance:
      d^2 = ||x||^2 + ||y||^2 - 2 x·y
    - force float64
    - handle non-finite by mapping to large distances
    - ensure non-negative due to numerical noise
    """
    if Y is None:
        Y = X
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)

    # compute squared norms via einsum to avoid overflow patterns
    a2 = np.einsum("ij,ij->i", X, X)[:, None]
    b2 = np.einsum("ij,ij->i", Y, Y)[None, :]

    K = X @ Y.T  # this is where overflow used to happen
    # guard: if any non-finite sneaks in, set to 0 inner product (maximizes distance via a2+b2)
    K[~np.isfinite(K)] = 0.0

    dist2 = a2 + b2 - 2.0 * K
    # Replace non-finite distances with +inf
    dist2[~np.isfinite(dist2)] = np.inf
    # Numerical floor at zero
    np.maximum(dist2, 0.0, out=dist2)
    return dist2


# -------------------------------
# K-means (Lloyd) with k-means++ init
# -------------------------------

def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = []
    # pick first center
    i0 = int(rng.integers(0, n))
    centers.append(X[i0])
    # pick rest
    for _ in range(1, k):
        dist2 = _safe_sqeuclidean(X, np.vstack(centers)).min(axis=1)
        probs = dist2 / (dist2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers.append(X[idx])
    return np.vstack(centers)


def _kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    k = min(k, n)  # guard

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
                # re-seed empty cluster
                C[j] = X[int(rng.integers(0, n))]
            else:
                C[j] = X[mask].mean(axis=0)
    return labels, C


# -------------------------------
# K-medoids (PAM-lite)
# -------------------------------

def _pam(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    k = min(k, n)  # guard

    # initialize medoids by random unique indices
    medoid_idx = rng.choice(n, size=k, replace=False)
    D = _safe_sqeuclidean(X)  # full distance matrix

    def total_cost(med_idx: np.ndarray) -> float:
        dmin = D[:, med_idx].min(axis=1)
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

    # assign labels
    labels = D[:, medoid_idx].argmin(axis=1)
    medoids = X[medoid_idx]
    return labels, medoids


# -------------------------------
# Public API
# -------------------------------

def cluster_dataframe(
    df: pd.DataFrame,
    cfg: ClusterConfig,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.Series, np.ndarray, List[str]]:
    """
    Cluster rows of df using selected numeric features.

    Returns
    -------
    labels : pd.Series of shape (n,)
    centers : np.ndarray of shape (k, d)  (means or medoids)
    used_features : list of column names actually used
    """
    # build cleaned numeric matrix
    X, used = _prepare_numeric_matrix(df, feature_cols)

    # handle degenerate cases
    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty matrix after cleaning.")
    k = max(1, min(cfg.k, n))

    # if all features are constant after cleaning, skip clustering
    if np.allclose(X.std(axis=0), 0.0):
        labels = np.zeros(n, dtype=int)
        centers = X[:1].copy()
        return pd.Series(labels, index=df.index, name="cluster"), centers, used

    # run chosen method
    if cfg.method.lower() == "kmeans":
        labels, centers = _kmeans(X, k=k, seed=cfg.seed)
    else:
        labels, centers = _pam(X, k=k, seed=cfg.seed)

    return pd.Series(labels, index=df.index, name="cluster"), centers, used


def pick_representatives(
    df: pd.DataFrame,
    labels: pd.Series,
    per_cluster_max: int = 2,
    score_cols: Optional[Sequence[str]] = None,
    beta_logq: float = 0.2,
) -> pd.DataFrame:
    """
    Pick representative rows per cluster by a score.
    Default score: S = E_A - beta_logq * log(q_prob)
    """
    work = df.copy()
    work["cluster"] = labels.values

    # default score columns
    if score_cols is None:
        score_cols = ["E_A"]

    # build score
    work = work.copy()
    # safe log(q): replace non-positive with tiny epsilon
    q = pd.to_numeric(work.get("q_prob", 1.0), errors="coerce").fillna(1.0)
    q = q.clip(lower=1e-300)
    score = pd.to_numeric(work.get("E_A", 0.0), errors="coerce").fillna(0.0) - beta_logq * np.log(q)
    work["__score__"] = score.replace([np.inf, -np.inf], np.nan).fillna(1e9)

    reps = []
    for c, g in work.groupby("cluster"):
        g_sorted = g.sort_values("__score__").head(per_cluster_max)
        reps.append(g_sorted)
    out = pd.concat(reps, ignore_index=True).drop(columns=["__score__"])
    return out

# ---- Backward compatibility wrappers (keep old API names) ----

def cluster_group(
    df: pd.DataFrame,
    cfg: ClusterConfig,
    feature_cols: Optional[Sequence[str]] = None,
):
    """
    Backward-compatible wrapper for old API name.
    Delegates to cluster_dataframe.
    Returns (labels, centers, used_features).
    """
    return cluster_dataframe(df, cfg, feature_cols)


def select_topK_per_group(
    df: pd.DataFrame,
    labels: pd.Series,
    per_cluster_max: int = 2,
    beta_logq: float = 0.2,
):
    """
    Backward-compatible wrapper for old API name.
    Delegates to pick_representatives.
    """
    return pick_representatives(
        df,
        labels,
        per_cluster_max=per_cluster_max,
        beta_logq=beta_logq,
    )
