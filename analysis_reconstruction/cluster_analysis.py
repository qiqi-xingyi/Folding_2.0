# --*-- coding:utf-8 --*--
# @time:10/31/25 21:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cluster_analysis.py
#
# Multi-view, sparse, large-N-friendly clustering for QSAD post-analysis.
# Key ideas:
#   - Build per-view kNN graphs directly (no dense N×N distance matrices).
#   - Views: bitstring (Hamming), geometric (RMSD on candidate neighbors), features (Euclidean).
#   - Convert distances -> sparse Gaussian kernels; fuse with weights; row-topk.
#   - Keep the largest connected component; do sparse spectral embedding (diffusion map).
#   - Cluster the embedding with energy-weighted KMeans (scikit-learn, no O(N^2)).
#   - Score clusters by energy statistics with safe fallbacks (never crash on empty/NaN).
#
# This file is self-contained and can replace the previous cluster_analysis.py.
# It exposes ClusterConfig and ClusterAnalyzer with the methods used by your analysis.py.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, coo_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


# ============================
# Config
# ============================

@dataclass
class ClusterConfig:
    # which views to use
    use_geom: bool = True
    use_feat: bool = True
    use_ham: bool = False

    # view weights in kernel fusion
    w_geom: float = 0.5
    w_feat: float = 0.3
    w_ham: float = 0.2

    # kNN and diffusion
    knn: int = 80
    diffusion_dim: int = 12
    diff_time: int = 2

    # clustering
    k_candidates: Sequence[int] = field(default_factory=lambda: (8, 10, 12))
    energy_weight_beta: float = 2.5
    max_cluster_frac: float = 0.60

    # runs / robustness (kept for compatibility; not used in this minimal version)
    n_runs: int = 1
    min_cluster_size: int = 15

    # columns / keys
    bitstring_col: str = "bitstring"
    positions_col: str = "main_positions"     # array of (L,3)
    energy_key: str = "E_total"

    # misc
    seed: int = 0
    output_dir: str = "./cluster_out"

    # feature selection
    # numeric columns from the merged DataFrame used for "feature view".
    # If empty -> auto-detect numeric columns except obvious identifiers/energies.
    feature_cols: Optional[List[str]] = None


# ============================
# I/O helpers
# ============================

def read_table(path: str) -> pd.DataFrame:
    suf = os.path.splitext(path)[1].lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".jsonl", ".json"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                    elif isinstance(obj, list):
                        rows.extend(obj)
                except Exception:
                    # try whole-file JSON once
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    if isinstance(data, dict):
                        return pd.DataFrame([data])
                    raise
        return pd.DataFrame(rows)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {suf}")


# ============================
# Bitstring / positions / features extractors
# ============================

def bitstrings_to_array(bits: Sequence[str]) -> np.ndarray:
    # Convert list of bitstrings to uint8 0/1 matrix (N, L); truncate to min length if needed.
    arrs: List[np.ndarray] = []
    Lmin = None
    for s in bits:
        if s is None or not isinstance(s, str) or len(s) == 0:
            continue
        a = (np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")).astype(np.uint8)
        if Lmin is None:
            Lmin = len(a)
        else:
            Lmin = min(Lmin, len(a))
        arrs.append(a)
    if not arrs:
        raise ValueError("No valid bitstrings.")
    if Lmin is None or Lmin <= 0:
        raise ValueError("Empty bitstrings.")
    X = np.stack([a[:Lmin] for a in arrs], axis=0)
    return X


def extract_positions(df: pd.DataFrame, positions_col: str) -> np.ndarray:
    # Return positions as (N, L, 3) float array; truncate to min L if needed.
    pos_list: List[np.ndarray] = []
    Lmin = None
    for v in df[positions_col].tolist():
        if v is None:
            continue
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                continue
        A = np.asarray(v, dtype=float)
        if A.ndim != 2 or A.shape[1] != 3:
            continue
        if Lmin is None:
            Lmin = A.shape[0]
        else:
            Lmin = min(Lmin, A.shape[0])
        pos_list.append(A)
    if not pos_list:
        raise ValueError("No valid positions.")
    P = np.stack([A[:Lmin] for A in pos_list], axis=0)
    return P


def extract_feature_matrix(df: pd.DataFrame, cfg: ClusterConfig) -> np.ndarray:
    if cfg.feature_cols and len(cfg.feature_cols) > 0:
        cols = [c for c in cfg.feature_cols if c in df.columns]
    else:
        # auto-pick numeric columns; drop obvious identifiers/energies/arrays
        drop_like = {cfg.bitstring_col, cfg.positions_col, cfg.energy_key,
                     "sequence", "main_vectors", "side_vectors", "side_positions"}
        num_cols = [c for c in df.columns
                    if (c not in drop_like)
                    and (pd.api.types.is_numeric_dtype(df[c]))]
        cols = num_cols
    if not cols:
        raise ValueError("No numeric feature columns for feature view.")
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    return X


# ============================
# Geometry (RMSD)
# ============================

def _kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Br = Bc @ R
    d = Ac - Br
    return float(np.sqrt((d * d).sum() / A.shape[0]))


def rmsd_knn_from_candidates(P: np.ndarray, candidates: csr_matrix) -> csr_matrix:
    # Compute RMSD only on candidate neighbors; return sparse distance graph (CSR).
    rows, cols, data = [], [], []
    N = P.shape[0]
    indptr, indices = candidates.indptr, candidates.indices
    for i in range(N):
        Pi = P[i]
        start, end = indptr[i], indptr[i + 1]
        nbrs = indices[start:end]
        for j in nbrs:
            if j == i:
                continue
            d = _kabsch_rmsd(Pi, P[j])
            rows.append(i); cols.append(j); data.append(d)
    G = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return G


# ============================
# Sparse kNN graphs
# ============================

def knn_graph_hamming(X: np.ndarray, n_neighbors: int) -> csr_matrix:
    nn = NearestNeighbors(metric="hamming", n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X)
    G = nn.kneighbors_graph(X, mode="distance")
    return G.tocsr()


def knn_graph_euclidean(X: np.ndarray, n_neighbors: int) -> csr_matrix:
    nn = NearestNeighbors(metric="euclidean", n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X)
    G = nn.kneighbors_graph(X, mode="distance")
    return G.tocsr()


# ============================
# Sparse kernels and fusion
# ============================

def estimate_eps_sparse(D: csr_matrix) -> float:
    if not issparse(D):
        raise ValueError("Expected sparse distance matrix.")
    x = D.data
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = np.median(x)
    if not np.isfinite(med) or med <= 0:
        med = np.percentile(x, 60) if x.size > 10 else float(np.mean(x))
        if med <= 0 or not np.isfinite(med):
            med = 1.0
    return float(med)


def gaussian_kernel_sparse(D: csr_matrix, eps: float) -> csr_matrix:
    D = D.tocsr().astype(np.float64)
    data = np.exp(-(D.data ** 2) / max(1e-12, eps ** 2))
    K = csr_matrix((data, D.indices, D.indptr), shape=D.shape)
    K.setdiag(0.0)
    return K


def row_topk_csr(M: csr_matrix, k: int) -> csr_matrix:
    M = M.tocsr()
    indptr = M.indptr
    new_indptr = [0]
    new_indices: List[int] = []
    new_data: List[float] = []
    for i in range(M.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        idx = M.indices[s:e]
        val = M.data[s:e]
        if len(val) > k:
            sel = np.argpartition(-val, k)[:k]
            idx = idx[sel]
            val = val[sel]
        new_indices.extend(idx.tolist())
        new_data.extend(val.tolist())
        new_indptr.append(len(new_indices))
    out = csr_matrix((np.array(new_data), np.array(new_indices), np.array(new_indptr)), shape=M.shape)
    out.setdiag(0.0)
    return out


def fuse_kernels_sparse(Ks: List[csr_matrix], ws: List[float], topk: int) -> csr_matrix:
    assert len(Ks) > 0 and len(Ks) == len(ws)
    S: Optional[csr_matrix] = None
    for K, w in zip(Ks, ws):
        Kw = K.copy()
        Kw.data *= float(w)
        S = Kw if S is None else (S + Kw)
    S = S.tocsr()
    S.setdiag(0.0)
    # final row-topk to keep O(N·k)
    S = row_topk_csr(S, topk)
    return S


# ============================
# Connectivity and diffusion
# ============================

def largest_cc(G: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
    n_components, labels = connected_components(G, directed=False, return_labels=True)
    if n_components <= 1:
        return G, np.arange(G.shape[0])
    counts = np.bincount(labels)
    keep = np.argmax(counts)
    keep_idx = np.where(labels == keep)[0]
    G2 = G[keep_idx][:, keep_idx]
    return G2, keep_idx


def diffusion_from_sparse(K: csr_matrix, t: int, m: int) -> np.ndarray:
    d = np.asarray(K.sum(axis=1)).reshape(-1)
    d[d <= 1e-30] = 1e-30
    inv_sqrt = 1.0 / np.sqrt(d)
    S = K.multiply(inv_sqrt[:, None]).multiply(inv_sqrt[None, :]).tocsr()
    # top-(m+1) eigenpairs of symmetric S
    k_use = max(2, min(m + 1, S.shape[0] - 1))
    vals, vecs = eigsh(S, k=k_use, which="LM")
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    # skip the trivial eigenvector
    lambdas = vals[1:m + 1]
    U = vecs[:, 1:m + 1]
    emb = U * (lambdas[None, :] ** max(1, t))
    return emb.astype(np.float32)


# ============================
# Energy helpers and scoring
# ============================

def energy_to_weights(E: np.ndarray, beta: float = 2.0) -> np.ndarray:
    N = len(E)
    ranks = E.argsort().argsort().astype(np.float64)
    r = ranks / max(1, N - 1)
    w = np.exp(-beta * r)
    w = np.clip(w, 1e-8, None)
    w *= (N / w.sum())
    return w


def compute_cluster_energy_stats(df: pd.DataFrame, labels: np.ndarray, energy_key: str) -> pd.DataFrame:
    tbl = pd.DataFrame({
        "_label": labels,
        "_energy": df[energy_key].astype(float).values
    })
    rows: List[Dict[str, Any]] = []
    for lab, g in tbl.groupby("_label"):
        size = int(len(g))
        if size <= 0:
            continue
        E = g["_energy"].values
        rows.append({
            "cluster": int(lab),
            "size": size,
            "energy_median": float(np.median(E)),
            "energy_q05": float(np.quantile(E, 0.05)) if size > 1 else float(np.median(E)),
        })
    if not rows:
        return pd.DataFrame(columns=["cluster", "size", "energy_median", "energy_q05"])
    out = pd.DataFrame(rows).sort_values(["energy_median", "energy_q05", "size"], ascending=[True, True, False])
    return out.reset_index(drop=True)


def rscale(x: pd.Series) -> pd.Series:
    s = pd.Series(x).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or s.nunique() <= 1:
        return pd.Series(0.0, index=x.index)
    q75, q25 = np.percentile(s.values, [75, 25])
    iqr = max(q75 - q25, 1e-9)
    z = (s - s.median()) / iqr
    out = pd.Series(0.0, index=x.index)
    out.loc[s.index] = z
    return out


# ============================
# ClusterAnalyzer
# ============================

class ClusterAnalyzer:
    def __init__(self, cfg: ClusterConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

        self.df: Optional[pd.DataFrame] = None            # merged energies + features
        self.embedding_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.best_cluster_id_: Optional[int] = None
        self.best_member_indices_: Optional[List[int]] = None
        self.stats_: Optional[pd.DataFrame] = None

    # ---------- I/O ----------

    def load_files(self, energies_path: str, features_path: Optional[str] = None):
        dfe = read_table(energies_path)
        if features_path is None:
            self.df = dfe.copy()
            return
        dff = read_table(features_path)
        key = self.cfg.bitstring_col
        if key not in dfe.columns or key not in dff.columns:
            raise ValueError(f"Merge key '{key}' not found in both files.")
        # inner join to keep rows that have both energy and features
        df = pd.merge(dfe, dff, on=key, how="inner", suffixes=("", "_feat"))
        self.df = df.reset_index(drop=True)

    def load_file(self, path: str):
        # kept for backward compatibility (energies only)
        self.df = read_table(path).reset_index(drop=True)

    # ---------- pipeline ----------

    def _build_views_sparse(self, df: pd.DataFrame) -> Tuple[List[csr_matrix], List[float], Dict[str, Any]]:
        K_list: List[csr_matrix] = []
        w_list: List[float] = []
        cache: Dict[str, Any] = {}

        N = len(df)
        if N < 2:
            return K_list, w_list, cache

        # feature view
        if self.cfg.use_feat:
            try:
                Xf = extract_feature_matrix(df, self.cfg)
                Df = knn_graph_euclidean(Xf, n_neighbors=self.cfg.knn)
                eps_f = estimate_eps_sparse(Df)
                Kf = gaussian_kernel_sparse(Df, eps=eps_f)
                K_list.append(Kf); w_list.append(self.cfg.w_feat)
                cache["feat_eps"] = eps_f
            except Exception as e:
                logging.warning("Feature view failed: %s", e)

        # bitstring view
        if self.cfg.use_ham and (self.cfg.bitstring_col in df.columns):
            try:
                Xh = bitstrings_to_array(df[self.cfg.bitstring_col].astype(str).tolist())
                Dh = knn_graph_hamming(Xh, n_neighbors=self.cfg.knn)
                eps_h = estimate_eps_sparse(Dh)
                Kh = gaussian_kernel_sparse(Dh, eps=eps_h)
                K_list.append(Kh); w_list.append(self.cfg.w_ham)
                cache["ham_eps"] = eps_h
            except Exception as e:
                logging.warning("Hamming view failed: %s", e)

        # geometric view (use candidates from feature or hamming)
        if self.cfg.use_geom and (self.cfg.positions_col in df.columns):
            if len(K_list) == 0:
                logging.warning("No candidate graph available; geometry view needs a candidate kNN. Skipped.")
            else:
                try:
                    # pick the densest candidate among built graphs
                    cand = max(K_list, key=lambda A: A.nnz)
                    P = extract_positions(df, self.cfg.positions_col)  # (N, L, 3)
                    Dg = rmsd_knn_from_candidates(P, cand)
                    eps_g = estimate_eps_sparse(Dg)
                    Kg = gaussian_kernel_sparse(Dg, eps=eps_g)
                    K_list.append(Kg); w_list.append(self.cfg.w_geom)
                    cache["geom_eps"] = eps_g
                except Exception as e:
                    logging.warning("Geometry view failed: %s", e)

        if not K_list:
            raise RuntimeError("No view was successfully constructed.")
        return K_list, w_list, cache

    def fit(self):
        if self.df is None or len(self.df) < 2:
            raise RuntimeError("No data loaded or not enough rows.")
        df = self.df

        # build per-view sparse kernels
        Ks, ws, cache = self._build_views_sparse(df)

        # fuse kernels and keep row top-k
        Kfused = fuse_kernels_sparse(Ks, ws, topk=self.cfg.knn)

        # largest CC to stabilize spectral embedding
        Kcc, keep_idx = largest_cc(Kfused)
        if Kcc.shape[0] < 2:
            raise RuntimeError("Largest connected component is too small.")

        # diffusion map embedding (sparse)
        emb = diffusion_from_sparse(Kcc, t=self.cfg.diff_time, m=self.cfg.diffusion_dim)
        self.embedding_ = emb

        # map back to original indices
        N = len(df)
        full_emb = np.zeros((N, emb.shape[1]), dtype=np.float32)
        full_emb[keep_idx] = emb

        # energy weights
        E = df[self.cfg.energy_key].astype(float).values
        w = energy_to_weights(E, beta=self.cfg.energy_weight_beta)

        # cluster with energy-weighted KMeans (no O(N^2) distance matrix)
        best_labels = None
        best_score = None
        best_stats = None

        for k in self.cfg.k_candidates:
            # mask to rows in CC (others will be assigned later)
            Xk = full_emb[keep_idx]
            wk = w[keep_idx]

            if Xk.shape[0] < k:
                continue

            km = KMeans(n_clusters=int(k), n_init=10, random_state=self.cfg.seed)
            km.fit(Xk, sample_weight=wk)
            labels_cc = km.labels_

            # assign out-of-CC rows (if any) to nearest centroid in embedding space
            labels_full = -1 * np.ones(N, dtype=int)
            labels_full[keep_idx] = labels_cc

            if len(keep_idx) < N:
                cent = km.cluster_centers_
                rest_idx = np.setdiff1d(np.arange(N), keep_idx, assume_unique=True)
                if len(rest_idx) > 0:
                    Xrest = full_emb[rest_idx]
                    # nearest centroid
                    d2 = np.sum((Xrest[:, None, :] - cent[None, :, :]) ** 2, axis=2)
                    assign = np.argmin(d2, axis=1)
                    labels_full[rest_idx] = assign

            # score by energy statistics and balance (no dense distances)
            cdf = compute_cluster_energy_stats(df, labels_full, self.cfg.energy_key)
            if cdf.empty:
                continue
            max_frac = float(
                cdf["size"].values.max() / max(1, N)
            )
            # lower median energy across clusters is better; penalize unbalanced partition
            score = -float(cdf["energy_median"].mean()) - 0.5 * max(0.0, max_frac - self.cfg.max_cluster_frac)

            if (best_score is None) or (score > best_score):
                best_score = score
                best_labels = labels_full
                best_stats = cdf

        if best_labels is None:
            # fallback: single KMeans with k=8
            k = int(self.cfg.k_candidates[0]) if len(self.cfg.k_candidates) > 0 else 8
            km = KMeans(n_clusters=k, n_init=10, random_state=self.cfg.seed)
            km.fit(full_emb[keep_idx], sample_weight=w[keep_idx])
            labels_cc = km.labels_
            labels_full = -1 * np.ones(N, dtype=int)
            labels_full[keep_idx] = labels_cc
            if len(keep_idx) < N:
                cent = km.cluster_centers_
                rest_idx = np.setdiff1d(np.arange(N), keep_idx, assume_unique=True)
                if len(rest_idx) > 0:
                    Xrest = full_emb[rest_idx]
                    d2 = np.sum((Xrest[:, None, :] - cent[None, :, :]) ** 2, axis=2)
                    assign = np.argmin(d2, axis=1)
                    labels_full[rest_idx] = assign
            best_labels = labels_full
            best_stats = compute_cluster_energy_stats(df, best_labels, self.cfg.energy_key)

        self.labels_ = best_labels
        self.stats_ = best_stats

        # pick best cluster: lowest median energy, then q05, then size desc
        if best_stats is None or best_stats.empty:
            self.best_cluster_id_ = None
            self.best_member_indices_ = []
        else:
            best_row = best_stats.iloc[0]
            cid = int(best_row["cluster"])
            self.best_cluster_id_ = cid
            members = np.where(self.labels_ == cid)[0]
            self.best_member_indices_ = members.tolist()

    # ---------- outputs ----------

    def get_best_cluster_indices(self) -> List[int]:
        return [] if self.best_member_indices_ is None else list(self.best_member_indices_)

    def get_cluster_labels(self) -> Optional[np.ndarray]:
        return self.labels_

    def get_stats(self) -> Optional[pd.DataFrame]:
        return self.stats_

    def save_reports(self, outdir: Optional[str] = None):
        if self.df is None or self.labels_ is None:
            raise RuntimeError("Run fit() before save_reports().")
        outdir = outdir or self.cfg.output_dir
        os.makedirs(outdir, exist_ok=True)

        df = self.df.copy()
        df["_cluster"] = self.labels_
        best_cid = self.best_cluster_id_
        df["_is_best_cluster"] = (df["_cluster"] == best_cid) if best_cid is not None else False

        # clusters.csv
        df.to_csv(os.path.join(outdir, "clusters.csv"), index=False)

        # cluster_stats.json
        payload = {
            "use_geom": self.cfg.use_geom,
            "use_feat": self.cfg.use_feat,
            "use_ham": self.cfg.use_ham,
            "weights": {"geom": self.cfg.w_geom, "feat": self.cfg.w_feat, "ham": self.cfg.w_ham},
            "knn": self.cfg.knn,
            "diffusion_dim": self.cfg.diffusion_dim,
            "diff_time": self.cfg.diff_time,
            "k_candidates": list(self.cfg.k_candidates),
            "energy_weight_beta": self.cfg.energy_weight_beta,
            "max_cluster_frac": self.cfg.max_cluster_frac,
            "bitstring_col": self.cfg.bitstring_col,
            "positions_col": self.cfg.positions_col,
            "energy_key": self.cfg.energy_key,
            "n_rows": int(len(self.df)),
            "best_cluster_id": None if self.best_cluster_id_ is None else int(self.best_cluster_id_),
            "stats": [] if self.stats_ is None else self.stats_.to_dict(orient="records"),
        }
        with open(os.path.join(outdir, "cluster_stats.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # best_cluster_ids.txt (row indices)
        with open(os.path.join(outdir, "best_cluster_ids.txt"), "w", encoding="utf-8") as f:
            if self.best_member_indices_:
                for ridx in self.best_member_indices_:
                    f.write(str(int(ridx)) + "\n")
