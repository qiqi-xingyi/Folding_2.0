# --*-- coding:utf-8 --*--
# @time:10/21/25 14:10
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cluster.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Iterable, Dict

import math
import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------

@dataclass
class ClusterConfig:
    """
    Configuration for clustering sampled bitstrings (or decoded vectors).

    Attributes
    ----------
    method : str
        "kmedoids" (default) or "kmeans".
    k : int
        Number of clusters. Backward-compatible alias: `n_clusters`.
    seed : int
        Random seed for deterministic behavior.
    beta_logq : float
        Weight for the -log(q_prob) term in scoring representative samples.
    temperature : float
        Reserved for soft-assignments / annealing; not used in this baseline.
    n_clusters : Optional[int]
        Backward-compat alias. If provided and `k` is left at its default,
        we adopt this value.
    per_cluster_max : int
        How many representatives to keep per cluster when selecting top-K.
    max_iter : int
        Max iterations for kmeans/kmedoids.
    tol : float
        Convergence tolerance.
    """
    method: str = "kmedoids"   # or "kmeans"
    k: int = 8
    seed: int = 0
    beta_logq: float = 0.2
    temperature: float = 1.0
    n_clusters: Optional[int] = None
    per_cluster_max: int = 3
    max_iter: int = 200
    tol: float = 1e-4

    def __post_init__(self):
        # alias adoption
        default_k = type(self).__dataclass_fields__["k"].default
        if self.n_clusters is not None and self.k == default_k:
            self.k = int(self.n_clusters)

        self.method = str(self.method).lower().strip()
        if self.method not in {"kmedoids", "kmeans"}:
            raise ValueError(f"Unsupported method={self.method}")

        if not (isinstance(self.k, int) and self.k >= 1):
            raise ValueError("k must be a positive integer")

        if not (isinstance(self.per_cluster_max, int) and self.per_cluster_max >= 1):
            raise ValueError("per_cluster_max must be a positive integer")

        if not (self.max_iter >= 1):
            raise ValueError("max_iter must be >= 1")

        if not (self.tol >= 0.0):
            raise ValueError("tol must be >= 0.0")


# ----------------------------
# Feature extraction
# ----------------------------

def _bitstrings_to_matrix(bits: Iterable[str]) -> np.ndarray:
    """Convert iterable of bitstrings (e.g., '010101') to a 2D {0,1} NumPy array."""
    arr = [np.frombuffer(bytes(b, "ascii"), dtype=np.uint8) - ord("0") for b in bits]
    # Validate same length
    Ls = {a.size for a in arr}
    if len(Ls) != 1:
        raise ValueError(f"Inconsistent bitstring lengths: {sorted(Ls)}")
    return np.vstack(arr)


def _infer_X(df: pd.DataFrame) -> np.ndarray:
    """
    Infer feature matrix X from DataFrame:
      1) Prefer 'bitstring' column if present
      2) Else use columns named 'b0'...'b{n-1}'
      3) Else try numeric columns
    """
    if "bitstring" in df.columns:
        return _bitstrings_to_matrix(df["bitstring"].astype(str).tolist())

    bcols = [c for c in df.columns if isinstance(c, str) and c.startswith("b")]
    if bcols:
        bcols_sorted = sorted(bcols, key=lambda s: int(s[1:]) if s[1:].isdigit() else 10**9)
        X = df[bcols_sorted].to_numpy(dtype=float)
        return X

    # fall back to all numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No usable features found (need 'bitstring' or numeric bit columns).")
    return df[num_cols].to_numpy(dtype=float)


# ----------------------------
# Distances
# ----------------------------

def _hamming(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between binary matrices A (n,d) and B (m,d)."""
    if A.dtype != np.uint8:
        A = A.astype(np.uint8, copy=False)
    if B.dtype != np.uint8:
        B = B.astype(np.uint8, copy=False)
    # Hamming = count of unequal bits
    n, m = A.shape[0], B.shape[0]
    out = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        out[i] = np.count_nonzero(A[i] != B, axis=1)
    return out


def _euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between A (n,d) and B (m,d)."""
    # (a-b)^2 = a^2 + b^2 - 2ab
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    ab = A @ B.T
    D2 = a2 + b2 - 2.0 * ab
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, dtype=np.float64)


# ----------------------------
# K-Means (NumPy only)
# ----------------------------

def _kmeans(X: np.ndarray, k: int, seed: int, max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k={k} larger than number of samples n={n}")

    # kmeans++ init
    centers = np.empty((k, X.shape[1]), dtype=float)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    dists = _euclidean(X, centers[0:1])[:, 0]
    for j in range(1, k):
        probs = dists ** 2
        s = probs.sum()
        if s <= 0:
            centers[j:] = X[rng.choice(n, size=k-j, replace=False)]
            break
        probs /= s
        idx = rng.choice(n, p=probs)
        centers[j] = X[idx]
        dists = np.minimum(dists, _euclidean(X, centers[j:j+1])[:, 0])

    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        # assign
        D = _euclidean(X, centers)
        new_labels = D.argmin(axis=1)
        # recompute centers
        new_centers = np.empty_like(centers)
        moved = 0.0
        for c in range(k):
            mask = (new_labels == c)
            if not np.any(mask):
                # re-seed empty cluster
                new_centers[c] = X[rng.integers(0, n)]
            else:
                new_centers[c] = X[mask].mean(axis=0)
            moved = max(moved, float(np.linalg.norm(new_centers[c] - centers[c])))
        centers = new_centers
        if np.array_equal(new_labels, labels) or moved <= tol:
            labels = new_labels
            break
        labels = new_labels
    return labels, centers


# ----------------------------
# K-Medoids (PAM, Hamming by default)
# ----------------------------

def _pam_kmedoids(X_bits: np.ndarray, k: int, seed: int, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    PAM on binary features using Hamming distance.
    Returns labels and medoid indices.
    """
    rng = np.random.default_rng(seed)
    n = X_bits.shape[0]
    if k > n:
        raise ValueError(f"k={k} larger than number of samples n={n}")

    # precompute distance matrix
    D = _hamming(X_bits, X_bits)

    # init: k unique medoids
    medoid_idx = rng.choice(n, size=k, replace=False)

    def total_cost(meds: np.ndarray) -> float:
        dmin = np.min(D[:, meds], axis=1)
        return float(dmin.sum())

    best_cost = total_cost(medoid_idx)

    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(k):
            for candidate in range(n):
                if candidate in medoid_idx:
                    continue
                new_meds = medoid_idx.copy()
                new_meds[i] = candidate
                c = total_cost(new_meds)
                if c + 1e-12 < best_cost:
                    best_cost = c
                    medoid_idx = new_meds
                    improved = True

    # final assignment
    labels = np.argmin(D[:, medoid_idx], axis=1)
    return labels, medoid_idx


# ----------------------------
# Public APIs
# ----------------------------

def cluster_group(df: pd.DataFrame, cfg: ClusterConfig) -> pd.DataFrame:
    """
    Cluster a dataframe of samples (expects 'bitstring' or bit columns).
    Adds a 'cluster' column and returns a new dataframe.
    """
    if df.empty:
        return df.assign(cluster=pd.Series(dtype=int))

    # extract features
    X = _infer_X(df)

    # choose metric/algorithm
    method = cfg.method
    rng_seed = int(cfg.seed)

    if method == "kmedoids":
        # ensure binary for Hamming
        if X.dtype != np.uint8:
            # If features aren't binary, try to round to {0,1}
            Xb = np.where(X > 0.5, 1, 0).astype(np.uint8)
        else:
            Xb = X
        labels, meds = _pam_kmedoids(Xb, cfg.k, rng_seed, cfg.max_iter)
        out = df.copy()
        out["cluster"] = pd.Series(labels, index=out.index, dtype=int)
        return out

    else:  # kmeans
        labels, _ = _kmeans(X.astype(float, copy=False), cfg.k, rng_seed, cfg.max_iter, cfg.tol)
        out = df.copy()
        out["cluster"] = pd.Series(labels, index=out.index, dtype=int)
        return out


def select_topK_per_group(
    df: pd.DataFrame,
    per_cluster_max: int,
    beta_logq: float,
    prob_col_candidates: Sequence[str] = ("q_prob", "prob", "p", "count"),
    energy_col_candidates: Sequence[str] = ("E_A", "energy", "score_A"),
) -> pd.DataFrame:
    """
    From a clustered dataframe, select top-K representatives per cluster using:
        score = E_A - beta_logq * log(q)
    where q is probability-like; if count is used, it's normalized within cluster.

    Returns a concatenated dataframe of representatives.
    """
    if df.empty:
        return df

    work = df.copy()

    # pick probability-like column
    prob_col = None
    for c in prob_col_candidates:
        if c in work.columns:
            prob_col = c
            break
    if prob_col is None:
        # fall back to uniform
        work["__q__"] = 1.0
    else:
        q_raw = pd.to_numeric(work[prob_col], errors="coerce")
        # if it's count, normalize within cluster
        if prob_col.lower() in {"count", "counts"}:
            q_raw = q_raw.fillna(0.0)
            work["__q__"] = q_raw / q_raw.groupby(work["cluster"]).transform("sum").clip(lower=1e-300)
        else:
            work["__q__"] = q_raw.fillna(0.0)

    # probability lower bound to avoid -inf
    q = work["__q__"].clip(lower=1e-300)

    # pick energy column (optional)
    E_col = None
    for c in energy_col_candidates:
        if c in work.columns:
            E_col = c
            break

    if E_col is None:
        EA = pd.Series(0.0, index=work.index)
    else:
        EA = pd.to_numeric(work[E_col], errors="coerce").fillna(0.0)

    # scoring
    score = EA - float(beta_logq) * np.log(q)
    work["__score__"] = score.replace([np.inf, -np.inf], np.nan).fillna(1e9)

    reps = []
    for c, g in work.groupby("cluster", dropna=False):
        reps.append(g.sort_values("__score__", kind="mergesort").head(int(per_cluster_max)))
    out = pd.concat(reps, ignore_index=True).drop(columns=["__score__", "__q__"], errors="ignore")
    return out
