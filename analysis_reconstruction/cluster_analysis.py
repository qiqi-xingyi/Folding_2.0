# --*-- conding:utf-8 --*--
# @time:10/23/26 1:30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cluster_analysis.py

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import hdbscan  # optional
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False


# -------------------------------
# Configuration
# -------------------------------

@dataclass
class ClusterConfig:
    # base clustering options
    method: str = "hdbscan"                # "hdbscan" | "kmedoids"
    min_cluster_size: Optional[int] = 2000  # for hdbscan
    k_candidates: Sequence[int] = field(default_factory=lambda: tuple(range(2, 9)))

    # columns and IO
    energy_key: str = "E_total"
    prefilter_rules: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 0
    output_dir: str = "./cluster_out"
    id_col: Optional[str] = None            # if None, DataFrame index will be used on save
    sequence_col: str = "sequence"
    main_vectors_col: str = "main_vectors"
    positions_col: str = "main_positions"   # backbone coordinates for geometry
    strict_same_length: bool = True

    # ---- EGDC: energy-geometry diffusion clustering ----
    distance_model: str = "geom_diffusion"  # "geom_diffusion" | "hamming" (fallback)
    # geometric kernel (RMSD -> Gaussian kernel), with kNN sparsification
    geom_kernel_eps: float = 0.5            # scale for RMSD Gaussian kernel
    knn: int = 40                           # keep top-k neighbors per row (symmetrized)
    # energy weighting to form a reversible kernel (detailed balance)
    temperature_kT: float = 0.5             # effective temperature

    # diffusion map embedding
    diffusion_time: int = 3                 # diffusion time t
    embedding_dim: int = 10                 # number of non-trivial eigenvectors

    # optional: blend energy into distances (usually unnecessary with the kernel)
    use_energy_in_distance: bool = False
    energy_alpha: float = 0.5
    energy_distance_method: str = "rank"    # "rank" | "mad"

    # weighted k-medoids (still available; we cluster in embedding space)
    use_weighted_pam: bool = True
    energy_weight_beta: float = 2.0         # larger -> stronger emphasis on low energy

    # stabilizers
    collapse_identical: bool = True
    silhouette_floor: float = 0.02
    max_cluster_frac: float = 0.60
    penalty_lambda: float = 0.5

    # optional: only low-energy portion enters clustering
    energy_quantile_for_clustering: Optional[float] = None

    # ---- NEW: robust batch rescaling of energy before kernel/weights ----
    energy_rescale_mode: str = "quantile"   # "none" | "quantile"
    energy_rescale_low_q: float = 0.01
    energy_rescale_high_q: float = 0.99
    energy_rescale_target: float = 10.0     # map [q_low, q_high] -> [0, target]
    energy_scaled_key: str = "E_total_scaled"

    def normalize(self):
        self.method = str(self.method).lower().strip()
        if self.min_cluster_size is None:
            self.min_cluster_size = 5
        if self.max_cluster_frac <= 0 or self.max_cluster_frac > 1:
            self.max_cluster_frac = 0.60
        if self.energy_distance_method not in ("rank", "mad"):
            self.energy_distance_method = "rank"
        if self.distance_model not in ("geom_diffusion", "hamming"):
            self.distance_model = "geom_diffusion"
        if self.energy_rescale_mode not in ("none", "quantile"):
            self.energy_rescale_mode = "quantile"


# -------------------------------
# Utilities
# -------------------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def read_table(path: str) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in [".csv"]:
        return pd.read_csv(path)
    if suffix in [".jsonl", ".json"]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if "\n" in content:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
            return pd.DataFrame(rows)
        obj = json.loads(content)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame([obj])
        raise ValueError("Unsupported JSON structure.")
    if suffix in [".parquet"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def parse_prefilter(df: pd.DataFrame, rules: Dict[str, Any]) -> pd.Series:
    """Apply simple prefilter rules. e.g., {"E_steric<=0": true, "E_geom<=7.5": true}"""
    if not rules:
        return pd.Series(True, index=df.index)
    mask = pd.Series(True, index=df.index)
    for expr, expect in rules.items():
        expr = expr.replace(" ", "")
        if "<=" in expr:
            col, val = expr.split("<=")
            val = float(val)
            if col not in df.columns:
                logging.warning("Prefilter column not found: %s", col)
                continue
            mask &= (df[col] <= val) == bool(expect)
        elif ">=" in expr:
            col, val = expr.split(">=")
            val = float(val)
            if col not in df.columns:
                logging.warning("Prefilter column not found: %s", col)
                continue
            mask &= (df[col] >= val) == bool(expect)
        elif "==" in expr:
            col, val = expr.split("==")
            val = float(val) if val.replace(".", "", 1).isdigit() else val
            if col not in df.columns:
                logging.warning("Prefilter column not found: %s", col)
                continue
            mask &= (df[col] == val) == bool(expect)
        else:
            logging.warning("Unsupported prefilter expression: %s", expr)
    return mask


def extract_main_vectors(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, List[int]]:
    """Return (matrix[N,L], valid_indices). Values must be iterable of ints or JSON-encoded lists."""
    valid_idx: List[int] = []
    vectors: List[List[int]] = []
    for i, v in enumerate(df[col].tolist()):
        if v is None:
            continue
        if isinstance(v, str):
            try:
                v_parsed = json.loads(v)
            except Exception:
                logging.warning("Row %d has string main_vectors but not JSON-decodable; skip.", i)
                continue
            v = v_parsed
        if not isinstance(v, (list, tuple)):
            continue
        try:
            row = [int(x) for x in v]
        except Exception:
            continue
        vectors.append(row)
        valid_idx.append(i)
    if not vectors:
        raise ValueError("No valid main_vectors found.")
    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        logging.warning("main_vectors have varying lengths: %s", sorted(lengths))
    L = max(lengths)
    if len(lengths) != 1:
        L = min(lengths)
        vectors = [v[:L] for v in vectors]
    mat = np.array(vectors, dtype=np.int16)
    return mat, valid_idx


def extract_positions(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, List[int]]:
    """Return (tensor[N,L,3], valid_indices). Values must be iterable (Lx3) or JSON."""
    valid_idx: List[int] = []
    pos_list: List[np.ndarray] = []
    for i, v in enumerate(df[col].tolist()):
        if v is None:
            continue
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                logging.warning("Row %d has string positions but not JSON-decodable; skip.", i)
                continue
        if not isinstance(v, (list, tuple)):
            continue
        try:
            arr = np.array(v, dtype=float)  # shape (L, 3) or (L+1, 3)
            if arr.ndim != 2 or arr.shape[1] != 3:
                continue
        except Exception:
            continue
        pos_list.append(arr)
        valid_idx.append(i)
    if not pos_list:
        raise ValueError("No valid positions found.")
    lengths = {p.shape[0] for p in pos_list}
    if len(lengths) != 1:
        logging.warning("positions have varying lengths: %s", sorted(lengths))
        L = min(lengths)
        pos_list = [p[:L] for p in pos_list]
    pos = np.stack(pos_list, axis=0)  # (N, L, 3)
    return pos, valid_idx


def hamming_distance_matrix(mat: np.ndarray) -> np.ndarray:
    """Pairwise normalized Hamming distance for integer-coded sequences."""
    n, L = mat.shape
    dm = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        diff = (mat[i] != mat).astype(np.float32)
        d = diff.mean(axis=1)
        dm[i, :] = d
        dm[:, i] = d
    return dm


# -------------------------------
# Geometry & EGDC helpers
# -------------------------------

def _kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """RMSD between two point sets after optimal superposition (Kabsch).
    A, B: (L,3)
    """
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Br = Bc @ R
    diff = Ac - Br
    return float(np.sqrt((diff * diff).sum() / A.shape[0]))


def rmsd_distance_matrix(pos: np.ndarray) -> np.ndarray:
    """Pairwise RMSD (Kabsch) for (N,L,3)."""
    N = pos.shape[0]
    dm = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        Ai = pos[i]
        for j in range(i + 1, N):
            d = _kabsch_rmsd(Ai, pos[j])
            dm[i, j] = dm[j, i] = d
    return dm


def energy_diff_matrix(E: np.ndarray, method: str = "rank") -> np.ndarray:
    """Energy difference 'distance' in [0,1]."""
    N = len(E)
    if N <= 1:
        return np.zeros((N, N), dtype=np.float32)
    if method == "rank":
        ranks = E.argsort().argsort().astype(np.float64)
        R = np.abs(ranks[:, None] - ranks[None, :]) / max(1, N - 1)
        return R.astype(np.float32)
    else:
        med = np.median(E)
        mad = np.median(np.abs(E - med)) + 1e-9
        D = np.abs(E[:, None] - E[None, :]) / (mad * 6.0)
        return np.clip(D, 0.0, 1.0).astype(np.float32)


def combine_distance(dm_base: np.ndarray, E: np.ndarray, alpha: float = 0.8, method: str = "rank") -> np.ndarray:
    """d = alpha * base + (1 - alpha) * EnergyDiff."""
    dm_E = energy_diff_matrix(E, method=method)
    return alpha * dm_base + (1.0 - alpha) * dm_E


def _symmetrize_max(mat: np.ndarray):
    return np.maximum(mat, mat.T)


def build_energy_geometry_kernel(dm_geom: np.ndarray,
                                 E: np.ndarray,
                                 eps: float,
                                 kT: float,
                                 knn: int) -> np.ndarray:
    """
    Reversible kernel K (satisfies detailed balance):
      base_ij = exp(-(d_ij^2) / eps^2) from geometry (RMSD);
      K_ij = base_ij * sqrt(pi_j / pi_i), where pi_i ∝ exp(-E_i/kT).
    The kernel is sparsified by symmetric kNN.
    """
    N = dm_geom.shape[0]
    base = np.exp(- (dm_geom ** 2) / max(1e-12, eps ** 2))
    np.fill_diagonal(base, 0.0)

    if knn is not None and knn > 0 and knn < N - 1:
        mask = np.zeros_like(base, dtype=bool)
        idx_sorted = np.argsort(-base, axis=1)
        for i in range(N):
            nn = idx_sorted[i, :knn]
            mask[i, nn] = True
        base = base * (mask | mask.T)

    pi = np.exp(- (E - float(np.min(E))) / max(1e-12, kT))
    pi = pi / (pi.sum() + 1e-12)
    sqrt_ratio = np.sqrt(np.clip(pi[None, :] / (pi[:, None] + 1e-300), 0.0, 1e300))
    K = base * sqrt_ratio
    K[K < 1e-15] = 0.0
    return K


def diffusion_map_embedding_from_kernel(K: np.ndarray, t: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric normalization: S = D^{-1/2} K D^{-1/2}.
    Eigendecompose S; take top m non-trivial eigenvectors (skip the first trivial mode).
    Embedding: Psi_k = (lambda_k)^t * u_k.
    Returns (embedding[N,m], lambdas[m]).
    """
    d = np.asarray(K.sum(axis=1)).reshape(-1)
    d[d <= 1e-30] = 1e-30
    inv_sqrt_d = 1.0 / np.sqrt(d)
    S = (inv_sqrt_d[:, None] * K) * inv_sqrt_d[None, :]
    vals, vecs = np.linalg.eigh(S)  # ascending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    m_eff = max(1, min(m, vecs.shape[1] - 1))
    lambdas = vals[1:1 + m_eff]
    U = vecs[:, 1:1 + m_eff]
    emb = U * (lambdas[None, :] ** max(1, t))
    return emb.astype(np.float32), lambdas.astype(np.float32)


def pairwise_euclidean_distance(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances for X[N,d]."""
    N = X.shape[0]
    G = X @ X.T
    nrm = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.clip(nrm + nrm.T - 2.0 * G, 0.0, None)
    D = np.sqrt(D2, dtype=np.float32)
    return D


# -------------------------------
# Classic helpers (kept)
# -------------------------------

def cluster_with_hdbscan(dm: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Run HDBSCAN clustering on a precomputed distance matrix."""
    if not _HDBSCAN_AVAILABLE:
        raise RuntimeError("hdbscan is not available.")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    labels = clusterer.fit_predict(dm)
    return labels


def silhouette_score_from_distance(dm: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette from a precomputed distance matrix. Returns -1 if invalid."""
    n = dm.shape[0]
    if n != len(labels):
        return -1.0
    uniq = [c for c in np.unique(labels) if c != -1]
    if len(uniq) < 2:
        return -1.0
    s_vals = []
    for i in range(n):
        li = labels[i]
        if li == -1:
            continue
        same = (labels == li)
        other = (labels != li) & (labels != -1)
        same_idx = np.where(same)[0]
        if len(same_idx) <= 1:
            continue
        a = dm[i, same_idx[same_idx != i]].mean() if len(same_idx) > 1 else 0.0
        b = np.inf
        for c in uniq:
            if c == li:
                continue
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            b = min(b, dm[i, idx].mean())
        if not np.isfinite(b):
            continue
        s = (b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0
        s_vals.append(s)
    if not s_vals:
        return -1.0
    return float(np.mean(s_vals))


def collapse_identical_vectors(mat: np.ndarray, idx: List[int]) -> Tuple[np.ndarray, List[List[int]], np.ndarray]:
    """Collapse identical rows. Return unique-matrix, groups (original df indices), and frequency weights."""
    from collections import defaultdict
    keys = [tuple(row.tolist()) for row in mat]
    bucket: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        bucket[k].append(i)
    mat_u, groups, weights = [], [], []
    for k, g in bucket.items():
        mat_u.append(np.array(k, dtype=mat.dtype))
        groups.append([idx[i] for i in g])
        weights.append(len(g))
    return np.stack(mat_u, axis=0), groups, np.array(weights, dtype=np.float64)


def energy_to_weights(E: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Map energy to clustering weights: lower energy -> larger weight."""
    N = len(E)
    ranks = E.argsort().argsort().astype(np.float64)
    r = ranks / max(1, N - 1)
    w = np.exp(-beta * r)
    w = np.clip(w, 1e-8, None)
    w = w * (N / w.sum())
    return w


def weighted_silhouette_from_distance(dm: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """Weighted silhouette on a precomputed distance matrix."""
    M = dm.shape[0]
    uniq = [c for c in np.unique(labels) if c != -1]
    if len(uniq) < 2:
        return -1.0
    s_sum, w_sum = 0.0, 0.0
    for i in range(M):
        li = labels[i]
        if li == -1:
            continue
        same_idx = np.where(labels == li)[0]
        if len(same_idx) <= 1:
            continue
        a = np.average(dm[i, same_idx], weights=weights[same_idx])
        b = np.inf
        for c in uniq:
            if c == li:
                continue
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            b = min(b, np.average(dm[i, idx], weights=weights[idx]))
        if not np.isfinite(b):
            continue
        s = (b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0
        s_sum += s * weights[i]
        w_sum += weights[i]
    if w_sum <= 0:
        return -1.0
    return float(s_sum / w_sum)


def pam_kmedoids(dm: np.ndarray, k: int, rng: np.random.RandomState, max_iter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """PAM (k-medoids) using a precomputed distance matrix. Returns (labels, medoid_indices)."""
    n = dm.shape[0]
    if k <= 1:
        return np.zeros(n, dtype=int), np.array([int(np.argmin(dm.sum(axis=1)))])
    medoids = []
    first = int(np.argmin(dm.sum(axis=1)))
    medoids.append(first)
    for _ in range(1, k):
        dist_to_nearest = np.min(dm[:, medoids], axis=1)
        probs = dist_to_nearest / (dist_to_nearest.sum() + 1e-12)
        candidate = int(rng.choice(np.arange(n), p=probs))
        medoids.append(candidate)
    medoids = np.array(sorted(set(medoids)))
    if len(medoids) < k:
        pool = [i for i in range(n) if i not in medoids]
        rng.shuffle(pool)
        medoids = np.array(list(medoids) + pool[: (k - len(medoids))])
    for _ in range(max_iter):
        dist_to_m = dm[:, medoids]
        labels = np.argmin(dist_to_m, axis=1)
        cost = float(np.sum(np.min(dist_to_m, axis=1)))
        improved = False
        for mi_idx in range(len(medoids)):
            non_medoids = [i for i in range(n) if i not in medoids]
            best_cost = cost
            best_swap = None
            for h in non_medoids:
                trial_medoids = medoids.copy()
                trial_medoids[mi_idx] = h
                trial_medoids.sort()
                d = np.min(dm[:, trial_medoids], axis=1)
                c = float(np.sum(d))
                if c + 1e-9 < best_cost:
                    best_cost = c
                    best_swap = h
            if best_swap is not None:
                medoids[mi_idx] = best_swap
                medoids.sort()
                cost = best_cost
                improved = True
        if not improved:
            break
    labels = np.argmin(dm[:, medoids], axis=1)
    return labels, medoids


def pam_kmedoids_weighted(dm: np.ndarray,
                          k: int,
                          rng: np.random.RandomState,
                          weights: Optional[np.ndarray] = None,
                          max_iter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted PAM (precomputed distance). weights >= 0."""
    n = dm.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones(n, dtype=np.float64)

    if k <= 1:
        labels = np.zeros(n, dtype=int)
        med = int(np.argmin(np.average(dm, axis=1, weights=weights)))
        return labels, np.array([med], dtype=int)

    medoids = [int(np.argmin(np.average(dm, axis=1, weights=weights)))]
    for _ in range(1, k):
        dist_to_nearest = np.min(dm[:, medoids], axis=1)
        probs = dist_to_nearest * (weights / (weights.sum() + 1e-12))
        s = probs.sum()
        if s <= 1e-12:
            candidate = int(rng.choice(np.arange(n)))
        else:
            probs = probs / s
            candidate = int(rng.choice(np.arange(n), p=probs))
        if candidate not in medoids:
            medoids.append(candidate)
    medoids = np.array(sorted(set(medoids)))
    if len(medoids) < k:
        pool = [i for i in range(n) if i not in medoids]
        rng.shuffle(pool)
        medoids = np.array(list(medoids) + pool[: (k - len(medoids))])

    for _ in range(max_iter):
        dist_to_m = dm[:, medoids]
        labels = np.argmin(dist_to_m, axis=1)
        cost = float(np.sum(weights * np.min(dist_to_m, axis=1)))
        improved = False
        for mi in range(len(medoids)):
            non_m = [i for i in range(n) if i not in medoids]
            best_cost = cost
            best_swap = None
            for h in non_m:
                trial = medoids.copy()
                trial[mi] = h
                trial.sort()
                d = np.min(dm[:, trial], axis=1)
                c = float(np.sum(weights * d))
                if c + 1e-9 < best_cost:
                    best_cost = c
                    best_swap = h
            if best_swap is not None:
                medoids[mi] = best_swap
                medoids.sort()
                cost = best_cost
                improved = True
        if not improved:
            break
    labels = np.argmin(dm[:, medoids], axis=1)
    return labels, medoids


def pick_k_balanced(dm: np.ndarray, rng: np.random.RandomState, k_candidates: Sequence[int],
                    weights: np.ndarray, max_cluster_frac: float, silhouette_floor: float,
                    penalty_lambda: float) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Return best_k and diagnostics per k: {"sil":..., "max_frac":..., "score":...}
    score = sil - penalty_lambda * max_frac; require sil >= silhouette_floor
    """
    M = dm.shape[0]
    report: Dict[int, Dict[str, float]] = {}
    best_k, best_score = None, -1e9

    for k in k_candidates:
        labels_u, _ = pam_kmedoids(dm, k, rng)
        frac = 0.0
        for c in np.unique(labels_u):
            if c == -1:
                continue
            w = weights[labels_u == c].sum()
            frac = max(frac, w / weights.sum())

        sil = weighted_silhouette_from_distance(dm, labels_u, weights)
        score = (sil if sil >= silhouette_floor else -1e9) - penalty_lambda * frac
        report[k] = {"sil": float(sil), "max_frac": float(frac), "score": float(score)}
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None or best_score <= -1e8:
        best_k = min(report.keys(), key=lambda kk: report[kk]["max_frac"])
    return best_k, report


def compute_cluster_energy_stats(df: pd.DataFrame, labels: np.ndarray, energy_key: str) -> pd.DataFrame:
    data = df.copy()
    data["_cluster"] = labels
    groups = []
    for cid, g in data.groupby("_cluster"):
        if cid == -1:
            continue
        energies = g[energy_key].astype(float).values
        median = float(np.median(energies)) if len(energies) else np.inf
        q05 = float(np.quantile(energies, 0.05)) if len(energies) > 1 else median
        size = int(len(energies))
        groups.append({
            "cluster": int(cid),
            "size": size,
            "energy_median": median,
            "energy_q05": q05,
        })
    stats = pd.DataFrame(groups).sort_values(["energy_median", "energy_q05", "size"], ascending=[True, True, False])
    return stats.reset_index(drop=True)


def choose_best_cluster(stats: pd.DataFrame) -> Optional[int]:
    if stats.empty:
        return None
    return int(stats.iloc[0]["cluster"])


# -------------------------------
# Main class
# -------------------------------

class ClusterAnalyzer:
    """Energy–Geometry Diffusion Clustering on backbone positions with energy-aware kernel."""
    def __init__(self, cfg: ClusterConfig):
        self.cfg = cfg
        self.cfg.normalize()
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_: Optional[pd.DataFrame] = None
        self.vec_mat: Optional[np.ndarray] = None
        self.valid_idx: Optional[List[int]] = None
        self.labels_: Optional[np.ndarray] = None
        self.dm_: Optional[np.ndarray] = None
        self.stats_: Optional[pd.DataFrame] = None
        self.best_cluster_id_: Optional[int] = None
        self.best_member_indices_: Optional[List[int]] = None
        self.embedding_: Optional[np.ndarray] = None
        self._energy_scale_info: Optional[Dict[str, Any]] = None
        self.rng = np.random.RandomState(self.cfg.random_seed)

    # ---- data I/O ----

    def load_dataframe(self, df: pd.DataFrame):
        """Set input DataFrame."""
        if self.cfg.id_col and self.cfg.id_col in df.columns:
            df = df.set_index(self.cfg.id_col, drop=False)
        self.df_raw = df

    def load_file(self, path: str):
        """Helper to read a table file and set as input."""
        df = read_table(path)
        self.load_dataframe(df)

    # ---- pipeline ----

    def _apply_prefilter(self):
        assert self.df_raw is not None
        mask = parse_prefilter(self.df_raw, self.cfg.prefilter_rules)
        kept = self.df_raw[mask].copy()
        self.df_ = kept

    def _prepare_vectors(self):
        assert self.df_ is not None
        if self.cfg.main_vectors_col not in self.df_.columns:
            raise ValueError(f"Column {self.cfg.main_vectors_col} not found.")
        mat, valid_idx = extract_main_vectors(self.df_, self.cfg.main_vectors_col)
        if self.cfg.strict_same_length:
            lengths = {len(x) for x in mat}
            if len(lengths) != 1:
                raise ValueError("main_vectors must have same length when strict_same_length=True.")
        self.vec_mat = mat
        self.valid_idx = valid_idx

    def _prepare_positions(self, idx: List[int]) -> np.ndarray:
        assert self.df_ is not None
        if self.cfg.positions_col not in self.df_.columns:
            raise ValueError(f"Column {self.cfg.positions_col} not found.")
        pos_all, valid_idx2 = extract_positions(self.df_, self.cfg.positions_col)
        mapper = {orig_i: k for k, orig_i in enumerate(valid_idx2)}
        rows = [mapper[i] for i in idx if i in mapper]
        if len(rows) != len(idx):
            raise RuntimeError("Positions index mismatch.")
        pos = pos_all[rows]
        return pos

    def _distance_matrix_hamming(self):
        assert self.vec_mat is not None
        self.dm_ = hamming_distance_matrix(self.vec_mat)

    # ---- NEW: robust batch energy rescaling ----
    def _rescale_energy_batch(self):
        assert self.df_ is not None
        ek = self.cfg.energy_key
        if ek not in self.df_.columns:
            raise ValueError(f"Energy key {ek} not found in data.")
        e = self.df_[ek].astype(float)

        if self.cfg.energy_rescale_mode.lower() != "quantile":
            self.df_[self.cfg.energy_scaled_key] = e.values
            self._energy_scale_info = {"mode": "none", "used_key": self.cfg.energy_scaled_key}
            return

        ql = float(np.quantile(e, self.cfg.energy_rescale_low_q))
        qh = float(np.quantile(e, self.cfg.energy_rescale_high_q))
        if not np.isfinite(ql) or not np.isfinite(qh) or qh <= ql + 1e-12:
            # degenerate; fall back to identity
            self.df_[self.cfg.energy_scaled_key] = e.values
            self._energy_scale_info = {
                "mode": "degenerate",
                "low_q": self.cfg.energy_rescale_low_q,
                "high_q": self.cfg.energy_rescale_high_q,
                "low": float(ql),
                "high": float(qh),
                "target": float(self.cfg.energy_rescale_target),
                "used_key": self.cfg.energy_scaled_key,
            }
            return
        t = float(self.cfg.energy_rescale_target)
        es = np.clip((e - ql) / (qh - ql), 0.0, 1.0) * t
        self.df_[self.cfg.energy_scaled_key] = es.values
        self._energy_scale_info = {
            "mode": "quantile",
            "low_q": self.cfg.energy_rescale_low_q,
            "high_q": self.cfg.energy_rescale_high_q,
            "low": float(ql),
            "high": float(qh),
            "target": t,
            "used_key": self.cfg.energy_scaled_key,
        }

    def fit(self):
        if self.df_raw is None:
            raise RuntimeError("No data loaded.")
        self._apply_prefilter()
        if self.df_ is None or len(self.df_) == 0:
            raise RuntimeError("No rows after prefilter.")
        self._prepare_vectors()

        # optional: energy-quantile preselection
        if self.cfg.energy_quantile_for_clustering is not None:
            ek = self.cfg.energy_key
            if ek not in self.df_.columns:
                raise ValueError(f"Energy key {ek} not found in data.")
            thr = float(self.df_[ek].quantile(self.cfg.energy_quantile_for_clustering))
            mask_low = self.df_[ek] <= thr
            idx_valid = np.array(self.valid_idx)
            keep_mask = mask_low.iloc[idx_valid].values
            self.vec_mat = self.vec_mat[keep_mask]
            self.valid_idx = list(idx_valid[keep_mask])

        # NEW: batch rescale energy to a fixed range for kernel/weights
        self._rescale_energy_batch()
        ekey_used = self.cfg.energy_scaled_key  # this column is used for kernel/weights

        # unique-level consolidation by main_vectors
        if self.cfg.collapse_identical:
            mat_u, groups, freq_w = collapse_identical_vectors(self.vec_mat, self.valid_idx)

            # median energy per unique group (scaled)
            if ekey_used not in self.df_.columns:
                raise ValueError(f"Energy key {ekey_used} not found in data.")
            colE_scaled = self.df_[ekey_used].astype(float)
            E_u = np.array([float(np.median(colE_scaled.iloc[grp].values)) for grp in groups], dtype=np.float64)

            # pick representative positions per unique vector: choose median-energy member (based on scaled energy)
            rep_indices = []
            for grp in groups:
                es = colE_scaled.iloc[grp].values.astype(float)
                mid_rank = np.argsort(es)[len(es)//2]
                rep_indices.append(grp[mid_rank])

            pos_all, valid_idx_pos = extract_positions(self.df_, self.cfg.positions_col)
            mapper = {orig_i: k for k, orig_i in enumerate(valid_idx_pos)}
            rep_rows = [mapper[i] for i in rep_indices if i in mapper]
            if len(rep_rows) != len(rep_indices):
                raise RuntimeError("Positions/indices mismatch when selecting representatives.")
            POS_u = pos_all[rep_rows]  # (M,L,3)

            if self.cfg.distance_model == "geom_diffusion":
                dm_geom = rmsd_distance_matrix(POS_u)
                K = build_energy_geometry_kernel(dm_geom, E_u,
                                                 eps=self.cfg.geom_kernel_eps,
                                                 kT=self.cfg.temperature_kT,
                                                 knn=self.cfg.knn)
                emb, lambdas = diffusion_map_embedding_from_kernel(
                    K, t=int(self.cfg.diffusion_time), m=int(self.cfg.embedding_dim)
                )
                self.embedding_ = emb
                dm_embed = pairwise_euclidean_distance(emb)
                dm_u = dm_embed
                if self.cfg.use_energy_in_distance:
                    dm_u = combine_distance(dm_embed, E_u, alpha=self.cfg.energy_alpha,
                                            method=self.cfg.energy_distance_method)
            else:
                dm_u_ham = hamming_distance_matrix(mat_u)
                dm_u = dm_u_ham
                if self.cfg.use_energy_in_distance:
                    dm_u = combine_distance(dm_u_ham, E_u, alpha=self.cfg.energy_alpha,
                                            method=self.cfg.energy_distance_method)

            # select k and cluster (unique level)
            if self.cfg.use_weighted_pam:
                w_energy = energy_to_weights(E_u, beta=self.cfg.energy_weight_beta)
                w_cluster = freq_w * w_energy
                best_k, _ = pick_k_balanced(
                    dm_u, self.rng, self.cfg.k_candidates,
                    weights=w_cluster,
                    max_cluster_frac=self.cfg.max_cluster_frac,
                    silhouette_floor=self.cfg.silhouette_floor,
                    penalty_lambda=self.cfg.penalty_lambda
                )
                labels_u, _ = pam_kmedoids_weighted(dm_u, best_k, self.rng, weights=w_cluster)
            else:
                best_k, _ = pick_k_balanced(
                    dm_u, self.rng, self.cfg.k_candidates,
                    weights=freq_w,
                    max_cluster_frac=self.cfg.max_cluster_frac,
                    silhouette_floor=self.cfg.silhouette_floor,
                    penalty_lambda=self.cfg.penalty_lambda
                )
                labels_u, _ = pam_kmedoids(dm_u, best_k, self.rng)

            # expand to sample level
            labels_full = np.empty(sum(len(g) for g in groups), dtype=int)
            ptr = 0
            for cid, grp in zip(labels_u, groups):
                labels_full[ptr:ptr + len(grp)] = cid
                ptr += len(grp)
            self.labels_ = labels_full

            # store a diagnostic distance matrix
            if self.cfg.distance_model == "geom_diffusion":
                self.dm_ = dm_geom.astype(np.float32)
            else:
                self.dm_ = dm_u.astype(np.float32)

        else:
            # sample-level path (not recommended for highly duplicated data)
            if self.cfg.distance_model == "geom_diffusion":
                pos_all, valid_idx_pos = extract_positions(self.df_, self.cfg.positions_col)
                mapper = {orig_i: k for k, orig_i in enumerate(valid_idx_pos)}
                rows = [mapper[i] for i in self.valid_idx if i in mapper]
                if len(rows) != len(self.valid_idx):
                    raise RuntimeError("Positions index mismatch on sample-level path.")
                POS = pos_all[rows]
                E = self.df_.iloc[self.valid_idx][ekey_used].astype(float).values

                dm_geom = rmsd_distance_matrix(POS)
                K = build_energy_geometry_kernel(dm_geom, E,
                                                 eps=self.cfg.geom_kernel_eps,
                                                 kT=self.cfg.temperature_kT,
                                                 knn=self.cfg.knn)
                emb, lambdas = diffusion_map_embedding_from_kernel(
                    K, t=int(self.cfg.diffusion_time), m=int(self.cfg.embedding_dim)
                )
                self.embedding_ = emb
                dm_embed = pairwise_euclidean_distance(emb)
                dm = dm_embed
                if self.cfg.use_energy_in_distance:
                    dm = combine_distance(dm_embed, E, alpha=self.cfg.energy_alpha,
                                          method=self.cfg.energy_distance_method)
            else:
                self._distance_matrix_hamming()
                dm = self.dm_
                E = self.df_.iloc[self.valid_idx][ekey_used].astype(float).values
                if self.cfg.use_energy_in_distance:
                    dm = combine_distance(dm, E, alpha=self.cfg.energy_alpha, method=self.cfg.energy_distance_method)

            n = dm.shape[0]
            if n < 2:
                self.labels_ = np.zeros(n, dtype=int)
            else:
                if self.cfg.method == "hdbscan" and _HDBSCAN_AVAILABLE:
                    min_cs = self.cfg.min_cluster_size or max(5, int(math.sqrt(n)))
                    self.labels_ = cluster_with_hdbscan(dm, min_cs)
                else:
                    if self.cfg.use_weighted_pam:
                        w = energy_to_weights(E, beta=self.cfg.energy_weight_beta)
                        best_k, _ = pick_k_balanced(dm, self.rng, self.cfg.k_candidates, weights=w,
                                                    max_cluster_frac=self.cfg.max_cluster_frac,
                                                    silhouette_floor=self.cfg.silhouette_floor,
                                                    penalty_lambda=self.cfg.penalty_lambda)
                        labels, _ = pam_kmedoids_weighted(dm, best_k, self.rng, weights=w)
                    else:
                        best_k = None
                        best_score = -1.0
                        for k in self.cfg.k_candidates:
                            lbl, _ = pam_kmedoids(dm, k, self.rng)
                            score = silhouette_score_from_distance(dm, lbl)
                            if score > best_score:
                                best_score = score
                                best_k = k
                        labels, _ = pam_kmedoids(dm, int(best_k), self.rng)
                    self.labels_ = labels
            self.dm_ = dm.astype(np.float32)

        # energy stats and best cluster selection
        assert self.labels_ is not None
        working_df = self.df_.iloc[self.valid_idx].copy()
        if ekey_used not in working_df.columns:
            raise ValueError(f"Energy key {ekey_used} not found in data.")
        # use scaled energy for stats and best cluster choice
        stats = compute_cluster_energy_stats(working_df, self.labels_, ekey_used)
        self.stats_ = stats
        best_cid = choose_best_cluster(stats)
        self.best_cluster_id_ = best_cid
        if best_cid is None:
            self.best_member_indices_ = []
        else:
            idx_valid = np.array(self.valid_idx)
            members = idx_valid[self.labels_ == best_cid]
            self.best_member_indices_ = list(map(int, members))

    # ---- outputs ----

    def get_best_cluster_indices(self) -> List[int]:
        return [] if self.best_member_indices_ is None else self.best_member_indices_

    def get_cluster_labels(self) -> Optional[np.ndarray]:
        return self.labels_

    def get_stats(self) -> Optional[pd.DataFrame]:
        return self.stats_

    def save_reports(self):
        """Persist clusters.csv, cluster_stats.json, best_cluster_ids.txt to cfg.output_dir."""
        if self.df_ is None or self.vec_mat is None or self.labels_ is None:
            raise RuntimeError("Run fit() before saving.")

        outdir = self.cfg.output_dir
        ensure_outdir(outdir)

        working_df = self.df_.iloc[self.valid_idx].copy()
        working_df["_cluster"] = self.labels_
        working_df["_is_best_cluster"] = False
        if self.best_cluster_id_ is not None:
            working_df.loc[working_df["_cluster"] == self.best_cluster_id_, "_is_best_cluster"] = True

        clusters_csv = os.path.join(outdir, "clusters.csv")
        working_df.to_csv(clusters_csv, index=bool(self.cfg.id_col is None))

        stats_json = os.path.join(outdir, "cluster_stats.json")
        stats_payload = {
            "method": self.cfg.method,
            "min_cluster_size": self.cfg.min_cluster_size,
            "k_candidates": list(self.cfg.k_candidates),
            "energy_key": self.cfg.energy_key,
            "energy_key_used": self.cfg.energy_scaled_key,
            "energy_rescale": self._energy_scale_info or {"mode": "unknown"},
            "distance_model": self.cfg.distance_model,
            "geom_kernel_eps": self.cfg.geom_kernel_eps,
            "knn": self.cfg.knn,
            "temperature_kT": self.cfg.temperature_kT,
            "diffusion_time": self.cfg.diffusion_time,
            "embedding_dim": self.cfg.embedding_dim,
            "n_rows_after_prefilter": int(len(self.df_)),
            "n_valid_for_vectors": int(len(self.valid_idx)),
            "best_cluster_id": None if self.best_cluster_id_ is None else int(self.best_cluster_id_),
            "stats": [] if self.stats_ is None else self.stats_.to_dict(orient="records"),
        }
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, indent=2)

        best_ids_path = os.path.join(outdir, "best_cluster_ids.txt")
        with open(best_ids_path, "w", encoding="utf-8") as f:
            if self.best_member_indices_:
                for ridx in self.best_member_indices_:
                    if self.cfg.id_col and self.cfg.id_col in self.df_raw.columns:
                        row_id = self.df_raw.iloc[ridx][self.cfg.id_col]
                        f.write(str(row_id) + "\n")
                    else:
                        f.write(str(ridx) + "\n")
