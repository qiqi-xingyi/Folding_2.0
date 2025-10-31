# --*-- coding:utf-8 --*--
# @time:10/30/25 19:40
# @Author : Yuqi Zhang (workflow) / Assistant (implementation)
# @Email : yzhan135@kent.edu
# @File:cluster_analysis.py
#
# Multi-view, energy-biased diffusion clustering (class-based)
# ------------------------------------------------------------
# - Load energies.jsonl + features.jsonl and inner-join on "bitstring".
# - Views: geometry (RMSD on main_positions), feature stats, bitstring/hamming.
# - Build Gaussian kernels per view, apply reversible energy bias (π from E_total).
# - Fuse kernels (weights), kNN sparsify, row-stochastic → diffusion embedding.
# - Repeat with jitter (n_runs) → consensus (co-association spectral).
# - Score clusters (compactness/energy smoothness/density/priors) and pick best.
# - Rank members in best cluster for downstream structure refine.
#
# Dependencies: numpy, pandas, scipy, scikit-learn; hdbscan (optional).
# Exposes a single class `ClusterAnalyzer` for external use,
# plus an optional CLI for quick testing.

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

try:
    import hdbscan  # optional
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False
    hdbscan = None  # type: ignore


# -------------------------------
# Config
# -------------------------------

@dataclass
class ClusterConfig:
    # I/O column keys
    id_col: Optional[str] = None              # if provided, will be preserved
    bitstring_col: str = "bitstring"
    positions_col: str = "main_positions"
    energy_key: str = "E_total"

    # Views & fusion weights
    use_geom: bool = True
    use_feat: bool = True
    use_ham: bool = True
    w_geom: float = 0.5
    w_feat: float = 0.3
    w_ham: float = 0.2

    # Geometry / kernels
    knn: int = 40                             # neighbors per node after fusion
    geom_max_n: int = 8000                    # set 0 to disable by size
    geom_sigma: float = -1.0                  # <=0 → auto median
    kT: float = -1.0                          # <=0 → IQR(E_total)

    # Diffusion
    diff_dim: int = 10
    diff_time: int = 2

    # Consensus (multi-run jitter)
    n_runs: int = 5
    weight_jitter: float = 0.08               # ± jitter on (w_geom,w_feat,w_ham)
    knn_jitter: int = 10

    # Clustering backend
    min_cluster_size: int = 25                # for HDBSCAN
    kmedoids_k_range: Tuple[int, int] = (3, 12)
    kmedoids_max_iter: int = 100

    # Ranking & selection
    target_rama: float = 1.5                  # heuristic sweet-spot
    top_k: int = 30

    # Reproducibility
    seed: int = 0


# -------------------------------
# JSONL loader & merge
# -------------------------------

def _read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def load_and_merge(energies_path: str,
                   features_path: str,
                   bitstring_col: str) -> pd.DataFrame:
    e = pd.DataFrame(_read_jsonl(energies_path))
    f = pd.DataFrame(_read_jsonl(features_path))
    if bitstring_col not in e.columns or bitstring_col not in f.columns:
        raise KeyError(f"Both files must contain '{bitstring_col}'")
    e = e.drop_duplicates(bitstring_col, keep="last")
    f = f.drop_duplicates(bitstring_col, keep="last")
    df = pd.merge(e, f, on=bitstring_col, how="inner", suffixes=("_E", "_F"))
    if len(df) == 0:
        raise ValueError("No overlapping rows after merge on bitstring.")
    return df.reset_index(drop=True)


# -------------------------------
# Geometry utilities (Kabsch RMSD)
# -------------------------------

def _as_xyz(arr_like) -> np.ndarray:
    A = np.asarray(arr_like, dtype=float)
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("Positions must have shape (L,3).")
    return A


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1, 1, d]) @ Wt
    R = Pc @ U
    diff = R - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def pairwise_rmsd(positions: List[np.ndarray]) -> np.ndarray:
    n = len(positions)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        Pi = positions[i]
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = kabsch_rmsd(Pi, positions[j])
    return D


# -------------------------------
# Bitstring / features distances
# -------------------------------

def normalized_hamming(bitstrings: List[str]) -> np.ndarray:
    # Convert to binary array (pad right with zeros to max length)
    n = len(bitstrings)
    L = max(len(s) for s in bitstrings)
    X = np.zeros((n, L), dtype=np.uint8)
    for i, s in enumerate(bitstrings):
        arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
        arr = np.clip(arr, 0, 1)
        X[i, :len(arr)] = arr
    D = squareform(pdist(X, metric="hamming")).astype(np.float32)
    return D


# -------------------------------
# Kernels, fusion, diffusion
# -------------------------------

def median_sigma(D: np.ndarray) -> float:
    X = D[np.triu_indices_from(D, k=1)]
    X = X[np.isfinite(X)]
    X = X[X > 0]
    if len(X) == 0:
        return 1.0
    return float(np.median(X))


def gaussian_kernel(D: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    if sigma is None or sigma <= 0:
        sigma = median_sigma(D)
    K = np.exp(-(D ** 2) / max(1e-12, sigma ** 2))
    np.fill_diagonal(K, 1.0)
    return K.astype(np.float32)


def energy_bias_reversible(K: np.ndarray, E: np.ndarray, kT: Optional[float]) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    med = np.median(E)
    if kT is None or kT <= 0:
        q75, q25 = np.percentile(E, [75, 25])
        iqr = max(q75 - q25, 1e-6)
        kT = iqr
    pi = np.exp(-(E - med) / kT)
    pi = np.clip(pi, 1e-12, None)
    s = np.sqrt(pi)
    Kb = K * (s[None, :] / s[:, None])  # detailed balance
    return Kb.astype(np.float32)


def knn_sparsify(K: np.ndarray, k: int) -> csr_matrix:
    n = K.shape[0]
    k = max(1, min(k, n - 1))
    indptr = [0]
    indices = []
    data = []
    for i in range(n):
        row = K[i]
        idx = np.argpartition(-row, k)[:k]
        idx = idx[np.argsort(-row[idx])]
        indices.extend(idx.tolist())
        data.extend(row[idx].tolist())
        indptr.append(len(indices))
    A = csr_matrix((data, indices, indptr), shape=(n, n), dtype=np.float32)
    A = A.maximum(A.T)  # symmetrize by max
    return A


def row_stochastic(A: csr_matrix) -> csr_matrix:
    if not issparse(A):
        A = csr_matrix(A)
    rowsum = np.array(A.sum(axis=1)).ravel()
    rowsum[rowsum == 0] = 1.0
    inv = 1.0 / rowsum
    Dinv = csr_matrix((inv, (np.arange(A.shape[0]), np.arange(A.shape[0]))), shape=A.shape)
    return Dinv @ A


def diffusion_map(T: csr_matrix, n_components: int, t: int, seed: int = 0) -> np.ndarray:
    A = (T + T.T) * 0.5  # make symmetric for stable eigs
    k = min(n_components + 1, A.shape[0] - 1)
    vals, vecs = eigsh(A, k=k, which="LM", tol=1e-4, maxiter=5000)
    idx = np.argsort(-vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals = vals[1:n_components + 1]
    vecs = vecs[:, 1:n_components + 1]
    lambdas_t = np.power(np.clip(vals, 0, None), t)
    Y = vecs * lambdas_t[None, :]
    return Y.astype(np.float32)


# -------------------------------
# Clustering backends
# -------------------------------

def run_hdbscan(Y: np.ndarray, min_cluster_size: int) -> np.ndarray:
    if not _HDBSCAN_AVAILABLE:
        raise RuntimeError("hdbscan not available")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(Y)
    return labels.astype(int)


def pam_kmedoids(X: np.ndarray, k: int, max_iter: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k <= 1:
        return np.zeros(n, dtype=int), np.array([int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))])
    D = squareform(pdist(X, metric="euclidean")).astype(np.float32)
    medoids = rng.choice(n, size=k, replace=False)
    labels = np.argmin(D[:, medoids], axis=1)
    for _ in range(max_iter):
        changed = False
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            current_med = medoids[j]
            current_cost = D[members][:, current_med].sum()
            best_med = current_med
            best_cost = current_cost
            for cand in members:
                cand_cost = D[members][:, cand].sum()
                if cand_cost < best_cost - 1e-6:
                    best_cost = cand_cost
                    best_med = cand
            if best_med != current_med:
                medoids[j] = best_med
                changed = True
        new_labels = np.argmin(D[:, medoids], axis=1)
        if not changed and np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels.astype(int), medoids


def auto_k_kmedoids(Y: np.ndarray, k_range: Tuple[int, int], max_iter: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    best_score = -1.0
    best = None
    best_meds = None
    for k in range(k_range[0], k_range[1] + 1):
        labels, medoids = pam_kmedoids(Y, k=k, max_iter=max_iter, seed=seed + k)
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(Y, labels, metric="euclidean")
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best = labels
            best_meds = medoids
    if best is None:
        best, best_meds = pam_kmedoids(Y, k=k_range[0], max_iter=max_iter, seed=seed)
    return best, best_meds


# -------------------------------
# Consensus clustering
# -------------------------------

def consensus_labels(all_labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(all_labels[0])
    C = np.zeros((n, n), dtype=np.float32)
    valid = 0
    for lab in all_labels:
        if lab.shape[0] != n:
            continue
        valid += 1
        same = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0) & (lab[None, :] >= 0)
        C += same.astype(np.float32)
    if valid == 0:
        raise RuntimeError("No valid labelings for consensus")
    C /= float(valid)
    A = (C + C.T) * 0.5
    A = A - np.diag(np.diag(A))
    vals = np.linalg.eigvalsh(A)[::-1]
    k_candidates = np.arange(2, min(12, len(vals) - 1) + 1)
    if len(k_candidates) == 0:
        k_final = 2
    else:
        gaps = np.diff(vals[: len(k_candidates) + 1])
        k_final = int(k_candidates[np.argmax(gaps)])
    sc = SpectralClustering(n_clusters=k_final, affinity="precomputed", assign_labels="kmeans", random_state=0)
    labels = sc.fit_predict(A)
    return labels.astype(int), C


# -------------------------------
# Scoring & ranking
# -------------------------------

def cluster_quality_scores(
    labels: np.ndarray,
    D_geom: Optional[np.ndarray],
    energies: np.ndarray,
    embed: np.ndarray,
    priors: Dict[str, np.ndarray],
    target_rama: float = 1.5,
) -> Tuple[int, pd.DataFrame]:
    uniq = np.unique(labels[labels >= 0])
    rows = []
    for k in uniq:
        idx = np.where(labels == k)[0]
        if len(idx) < 3:
            continue
        if D_geom is not None:
            sub = D_geom[np.ix_(idx, idx)]
            comp = float(np.median(sub[np.triu_indices_from(sub, k=1)]))
        else:
            comp = np.nan
        e = energies[idx]
        q75, q25 = np.percentile(e, [75, 25])
        e_iqr = float(max(q75 - q25, 1e-6))
        # density in embedding (kNN-5 radius)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(6, len(idx)), metric="euclidean").fit(embed[idx])
        dists, _ = nn.kneighbors(embed[idx])
        knn5 = float(np.mean(dists[:, -1]))
        clash = float(np.mean(priors.get("clash_count", np.full_like(energies, np.nan))[idx]))
        rama = float(np.mean(priors.get("rama_allowed_ratio", np.full_like(energies, np.nan))[idx]))
        hb = float(np.mean(priors.get("hb_count_pseudo", np.full_like(energies, np.nan))[idx]))
        size = len(idx)
        rows.append(dict(cluster=k, size=size, compactness=comp, energy_iqr=e_iqr, knn5=knn5,
                         clash_mean=clash, rama_mean=rama, hb_mean=hb))
    cdf = pd.DataFrame(rows)
    if cdf.empty:
        raise RuntimeError("No valid clusters for scoring.")

    def rscale(x: pd.Series) -> pd.Series:
        med = x.median()
        q75, q25 = np.percentile(x.dropna(), [75, 25])
        iqr = max(q75 - q25, 1e-6)
        return (x - med) / iqr

    score = (
        (-rscale(cdf["compactness"]).fillna(0)) * 0.4
        + (-rscale(cdf["energy_iqr"]).fillna(0)) * 0.2
        + (-rscale(cdf["knn5"]).fillna(0)) * 0.2
        + (-rscale(cdf["clash_mean"]).fillna(0)) * 0.08
        + (-rscale((cdf["rama_mean"] - target_rama).abs().fillna(0))) * 0.05
        + (rscale(cdf["hb_mean"]).fillna(0)) * 0.07
        - (rscale(cdf["size"]).fillna(0)) * 0.02
    )
    cdf["score"] = score
    best_row = cdf.iloc[int(np.argmax(score.values))]
    return int(best_row["cluster"]), cdf.sort_values("score", ascending=False).reset_index(drop=True)


def in_cluster_ranking(
    idx: np.ndarray,
    energies: np.ndarray,
    embed: np.ndarray,
    D_geom: Optional[np.ndarray],
    priors: Dict[str, np.ndarray],
    target_rama: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    subY = embed[idx]
    Dy = squareform(pdist(subY, metric="euclidean")).astype(np.float32)
    medoid_local = int(np.argmin(Dy.sum(axis=0)))

    e = energies[idx]
    q75, q25 = np.percentile(e, [75, 25])
    iqr = max(q75 - q25, 1e-6)
    e_score = -(e - np.median(e)) / iqr  # lower energy → higher score

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(6, len(idx)), metric="euclidean").fit(subY)
    dists, _ = nn.kneighbors(subY)
    dens = 1.0 / (dists[:, -1] + 1e-6)

    prox = -Dy[:, medoid_local]  # closer to medoid is better

    def z(arr: np.ndarray) -> np.ndarray:
        med = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = max(q75 - q25, 1e-6)
        return (arr - med) / iqr

    clash = priors.get("clash_count", np.full_like(energies, np.nan))[idx]
    rama = priors.get("rama_allowed_ratio", np.full_like(energies, np.nan))[idx]
    hb = priors.get("hb_count_pseudo", np.full_like(energies, np.nan))[idx]

    clash_score = -z(np.nan_to_num(clash, nan=np.nanmedian(clash)))
    rama_score = -np.abs(z(np.nan_to_num(rama, nan=np.nanmedian(rama)) - target_rama))
    hb_score = z(np.nan_to_num(hb, nan=np.nanmedian(hb)))

    geom_term = np.zeros_like(e_score)
    if D_geom is not None:
        subDg = D_geom[np.ix_(idx, idx)]
        gm = np.argmin(subDg.sum(axis=0))
        geom_term = -subDg[:, gm]

    score = (
        0.35 * e_score + 0.25 * z(dens) + 0.15 * z(prox) + 0.12 * clash_score
        + 0.05 * rama_score + 0.06 * hb_score + 0.02 * z(geom_term)
    )
    order = np.argsort(-score)
    return idx[order], score[order]


# -------------------------------
# Main class
# -------------------------------

class ClusterAnalyzer:
    """
    Class-based multi-view energy-biased diffusion clustering.
    Usage:
        ca = ClusterAnalyzer(cfg)
        ca.load_files(energies_jsonl, features_jsonl)
        ca.fit()
        labels = ca.labels_               # np.ndarray [n]
        best_idx = ca.get_best_cluster_indices()
        ranked = ca.best_cluster_rank_df  # pd.DataFrame (ranked members)
    """

    def __init__(self, cfg: ClusterConfig):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None
        self.embedding_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.coassoc_: Optional[np.ndarray] = None
        self.cluster_summary_: Optional[pd.DataFrame] = None
        self.best_cluster_id_: Optional[int] = None
        self.best_cluster_rank_df: Optional[pd.DataFrame] = None
        self._rng = np.random.default_rng(cfg.seed)

    # ---------- Load ----------

    def load_files(self, energies_path: str, features_path: str):
        self.df = load_and_merge(energies_path, features_path, self.cfg.bitstring_col)

    def load_dataframe(self, df: pd.DataFrame):
        self.df = df.copy()

    # ---------- Internal helpers ----------

    def _build_views(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[int]]:
        assert self.df is not None
        n0 = len(self.df)

        # Energy vector
        if self.cfg.energy_key not in self.df.columns:
            raise KeyError(f"Missing energy key: {self.cfg.energy_key}")
        E = self.df[self.cfg.energy_key].to_numpy(dtype=float)

        # Geometry positions (list)
        use_geom = self.cfg.use_geom and (self.cfg.geom_max_n == 0 or n0 <= self.cfg.geom_max_n)
        positions: List[np.ndarray] = []
        has_geom = False
        if use_geom and self.cfg.positions_col in self.df.columns:
            try:
                for v in self.df[self.cfg.positions_col].tolist():
                    if isinstance(v, str):
                        v = json.loads(v)
                    positions.append(_as_xyz(v))
                has_geom = True
            except Exception:
                warnings.warn("Failed to parse positions; disable geometry view.")
                has_geom = False
        else:
            has_geom = False

        # Bitstrings
        if self.cfg.bitstring_col not in self.df.columns:
            raise KeyError(f"Missing bitstring col: {self.cfg.bitstring_col}")
        bitstrings = self.df[self.cfg.bitstring_col].astype(str).tolist()

        # Feature matrix
        feat_cols = [
            "Rg", "end_to_end", "contact_density", "clash_count",
            "hb_count_pseudo", "rama_allowed_ratio", "packing_rep_count",
            "packing_att_count", "burial_mean", "burial_max",
            "frac_hydrophobic", "frac_charged", "frac_polar",
            "cbeta_rep_count", "cbeta_att_count",
        ]
        feat_cols = [c for c in feat_cols if c in self.df.columns]
        Xf = self.df[feat_cols].to_numpy(dtype=float) if feat_cols else None
        if Xf is not None and self.cfg.use_feat:
            Xf = RobustScaler().fit_transform(Xf)

        # Distances
        views_D: Dict[str, np.ndarray] = {}
        if has_geom:
            Dg = pairwise_rmsd(positions)
            views_D["geom"] = Dg
        if self.cfg.use_ham:
            Dh = normalized_hamming(bitstrings)
            views_D["ham"] = Dh
        if self.cfg.use_feat and Xf is not None:
            Df = squareform(pdist(Xf, metric="euclidean")).astype(np.float32)
            views_D["feat"] = Df

        return views_D, {"E": E, "Xf": Xf}, list(range(n0))

    def _fuse_to_transition(self, views_D: Dict[str, np.ndarray], E: np.ndarray) -> Tuple[csr_matrix, Optional[np.ndarray]]:
        # Kernels per view
        K_parts: List[Tuple[float, np.ndarray]] = []
        if "geom" in views_D and self.cfg.w_geom > 0:
            sig = self.cfg.geom_sigma
            Kg = gaussian_kernel(views_D["geom"], None if sig is None or sig <= 0 else sig)
            Kg = energy_bias_reversible(Kg, E, self.cfg.kT)
            K_parts.append((self.cfg.w_geom, Kg))
        if "feat" in views_D and self.cfg.w_feat > 0:
            Kf = gaussian_kernel(views_D["feat"], None)
            Kf = energy_bias_reversible(Kf, E, self.cfg.kT)
            K_parts.append((self.cfg.w_feat, Kf))
        if "ham" in views_D and self.cfg.w_ham > 0:
            Kh = gaussian_kernel(views_D["ham"], None)
            Kh = energy_bias_reversible(Kh, E, self.cfg.kT)
            K_parts.append((self.cfg.w_ham, Kh))

        if not K_parts:
            raise RuntimeError("No active views to fuse.")

        n = K_parts[0][1].shape[0]
        K = np.zeros((n, n), dtype=np.float32)
        for w, KK in K_parts:
            rs = KK.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            K += float(w) * (KK / rs)

        # kNN sparsify and row-stochastic
        A = knn_sparsify(K, k=self.cfg.knn)
        T = row_stochastic(A)

        # For optional cluster compactness scoring later (need a reference D_geom)
        D_geom = views_D.get("geom", None)
        return T, D_geom

    # ---------- Fit (full pipeline) ----------

    def fit(self):
        assert self.df is not None
        n = len(self.df)
        if n < 2:
            # trivial case
            self.embedding_ = np.zeros((n, max(1, self.cfg.diff_dim)), dtype=np.float32)
            self.labels_ = np.zeros(n, dtype=int)
            self.best_cluster_id_ = 0
            self.best_cluster_rank_df = self.df.copy()
            return

        views_D, cache, order = self._build_views()
        E = cache["E"]

        # Multiple runs (jitter) → collect labels
        label_runs: List[np.ndarray] = []
        embeds: List[np.ndarray] = []

        for r in range(self.cfg.n_runs):
            # jitter weights
            wg = np.clip(self.cfg.w_geom + self._rng.uniform(-self.cfg.weight_jitter, self.cfg.weight_jitter), 0, 1)
            wf = np.clip(self.cfg.w_feat + self._rng.uniform(-self.cfg.weight_jitter, self.cfg.weight_jitter), 0, 1)
            wh = np.clip(self.cfg.w_ham + self._rng.uniform(-self.cfg.weight_jitter, self.cfg.weight_jitter), 0, 1)

            # if some views inactive, renormalize remaining
            parts = []
            if "geom" in views_D and self.cfg.use_geom: parts.append(wg)
            if "feat" in views_D and self.cfg.use_feat: parts.append(wf)
            if "ham" in views_D and self.cfg.use_ham: parts.append(wh)
            if not parts:
                raise RuntimeError("All views disabled.")
            s = sum(parts)
            wg = (wg / s) if ("geom" in views_D and self.cfg.use_geom) else 0.0
            wf = (wf / s) if ("feat" in views_D and self.cfg.use_feat) else 0.0
            wh = (wh / s) if ("ham" in views_D and self.cfg.use_ham) else 0.0

            # temporarily override weights for this run
            w_geom_bak, w_feat_bak, w_ham_bak = self.cfg.w_geom, self.cfg.w_feat, self.cfg.w_ham
            self.cfg.w_geom, self.cfg.w_feat, self.cfg.w_ham = wg, wf, wh

            # jitter knn
            knn_base = self.cfg.knn
            self.cfg.knn = int(np.clip(knn_base + self._rng.integers(-self.cfg.knn_jitter, self.cfg.knn_jitter + 1), 5, max(5, n - 1)))

            # fuse to Markov operator and embed
            T, Dg_ref = self._fuse_to_transition(views_D, E)
            Y = diffusion_map(T, n_components=self.cfg.diff_dim, t=self.cfg.diff_time, seed=self.cfg.seed + r)
            embeds.append(Y)

            # clustering
            try:
                if _HDBSCAN_AVAILABLE:
                    labels = run_hdbscan(Y, min_cluster_size=self.cfg.min_cluster_size)
                else:
                    labels, _ = auto_k_kmedoids(Y, self.cfg.kmedoids_k_range, self.cfg.kmedoids_max_iter, seed=self.cfg.seed + r)
            except Exception as e:
                warnings.warn(f"Clustering failed in run {r}: {e}; assign one cluster.")
                labels = np.zeros(n, dtype=int)

            label_runs.append(labels)

            # restore base weights/knn for next run baseline
            self.cfg.w_geom, self.cfg.w_feat, self.cfg.w_ham = w_geom_bak, w_feat_bak, w_ham_bak
            self.cfg.knn = knn_base

        # consensus
        final_labels, C = consensus_labels(label_runs)
        self.labels_ = final_labels
        self.coassoc_ = C

        # choose a reference embedding (first run)
        self.embedding_ = embeds[0]

        # per-cluster summary & pick best
        priors = {
            "clash_count": self.df.get("clash_count", pd.Series([np.nan] * n)).to_numpy(dtype=float),
            "rama_allowed_ratio": self.df.get("rama_allowed_ratio", pd.Series([np.nan] * n)).to_numpy(dtype=float),
            "hb_count_pseudo": self.df.get("hb_count_pseudo", pd.Series([np.nan] * n)).to_numpy(dtype=float),
        }
        best_cluster, cdf = cluster_quality_scores(
            labels=self.labels_,
            D_geom=views_D.get("geom", None),
            energies=E,
            embed=self.embedding_,
            priors=priors,
            target_rama=self.cfg.target_rama,
        )
        self.cluster_summary_ = cdf
        self.best_cluster_id_ = best_cluster

        # ranking within best cluster
        best_idx = np.where(self.labels_ == best_cluster)[0]
        order_idx, scores = in_cluster_ranking(
            best_idx,
            E,
            self.embedding_,
            views_D.get("geom", None),
            priors,
            target_rama=self.cfg.target_rama,
        )
        rank_df = self.df.iloc[order_idx].copy()
        rank_df["rank_score"] = scores
        self.best_cluster_rank_df = rank_df.reset_index(drop=True)

    # ---------- Accessors ----------

    def get_best_cluster_indices(self) -> List[int]:
        if self.labels_ is None or self.best_cluster_id_ is None:
            return []
        return np.where(self.labels_ == self.best_cluster_id_)[0].astype(int).tolist()

    # ---------- Export ----------

    def save_reports(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        assert self.df is not None and self.labels_ is not None

        # labels
        lab = pd.DataFrame({
            self.cfg.bitstring_col: self.df[self.cfg.bitstring_col].values,
            "cluster": self.labels_,
            self.cfg.energy_key: self.df[self.cfg.energy_key].values,
        })
        lab.to_csv(os.path.join(outdir, "labels.csv"), index=False)

        # cluster summary
        if self.cluster_summary_ is not None:
            self.cluster_summary_.to_csv(os.path.join(outdir, "clusters_summary.csv"), index=False)

        # best cluster ranked
        if self.best_cluster_rank_df is not None:
            keep_cols = [
                self.cfg.bitstring_col, self.cfg.energy_key, "E_steric", "E_geom",
                "E_hb", "E_hydroph", "E_rama", "Rg", "end_to_end",
                "contact_density", "clash_count", "hb_count_pseudo", "rama_allowed_ratio",
                "rank_score",
            ]
            keep_cols = [c for c in keep_cols if c in self.best_cluster_rank_df.columns]
            self.best_cluster_rank_df[keep_cols].to_csv(os.path.join(outdir, "best_cluster_ranked.csv"), index=False)

        # co-association
        if self.coassoc_ is not None:
            np.save(os.path.join(outdir, "coassoc.npy"), self.coassoc_.astype(np.float32))

    # properties (optional sugar)
    @property
    def coassoc(self) -> Optional[np.ndarray]:
        return self.coassoc_

    @property
    def embedding(self) -> Optional[np.ndarray]:
        return self.embedding_


# -------------------------------
# Optional CLI for quick testing
# -------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Multi-view energy-biased diffusion clustering (class-based)")
    ap.add_argument("--energies", required=True, help="Path to energies.jsonl")
    ap.add_argument("--features", required=True, help="Path to features.jsonl")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--no-geom", action="store_true", help="Disable geometry view")
    ap.add_argument("--no-feat", action="store_true", help="Disable feature view")
    ap.add_argument("--no-ham", action="store_true", help="Disable bitstring/Hamming view")
    ap.add_argument("--knn", type=int, default=40)
    ap.add_argument("--diff-dim", type=int, default=10)
    ap.add_argument("--diff-time", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = ClusterConfig(
        use_geom=not args.no_geom,
        use_feat=not args.no_feat,
        use_ham=not args.no_ham,
        knn=args.knn,
        diff_dim=args.diff_dim,
        diff_time=args.diff_time,
        n_runs=args.runs,
        min_cluster_size=args.min_cluster_size,
        seed=args.seed,
    )

    ca = ClusterAnalyzer(cfg)
    ca.load_files(args.energies, args.features)
    ca.fit()
    ca.save_reports(args.outdir)
    print(f"[OK] Saved outputs to: {args.outdir}")
