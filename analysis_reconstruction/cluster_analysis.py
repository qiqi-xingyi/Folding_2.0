# --*-- conding:utf-8 --*--
# @time:10/23/25 18:09
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
    method: str = "kmedoids"                # "hdbscan" | "kmedoids"
    min_cluster_size: Optional[int] = None  # for hdbscan
    k_candidates: Sequence[int] = field(default_factory=lambda: tuple(range(2, 9)))

    # columns and IO
    energy_key: str = "E_total"
    prefilter_rules: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 0
    output_dir: str = "./cluster_out"
    id_col: Optional[str] = None            # if None, DataFrame index will be used on save
    sequence_col: str = "sequence"
    main_vectors_col: str = "main_vectors"
    strict_same_length: bool = True

    # ---- energy-guided clustering (C) ----
    # C1: use energy in distance
    use_energy_in_distance: bool = True
    energy_alpha: float = 0.4               # d = alpha*Hamming + (1-alpha)*EnergyDiff
    energy_distance_method: str = "rank"    # "rank" | "mad"

    # C2: weighted PAM (energy->weights)
    use_weighted_pam: bool = True
    energy_weight_beta: float = 2.0         # larger -> stronger emphasis on low energy

    # stabilizers against giant clusters
    collapse_identical: bool = True
    silhouette_floor: float = 0.02
    max_cluster_frac: float = 0.60
    penalty_lambda: float = 0.5

    # optional: only low-energy portion enters clustering (keep None to disable)
    energy_quantile_for_clustering: Optional[float] = None

    def normalize(self):
        self.method = str(self.method).lower().strip()
        if self.min_cluster_size is None:
            self.min_cluster_size = 5
        if self.max_cluster_frac <= 0 or self.max_cluster_frac > 1:
            self.max_cluster_frac = 0.60
        if self.energy_distance_method not in ("rank", "mad"):
            self.energy_distance_method = "rank"


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
# Energy-guided helpers (C)
# -------------------------------

def energy_diff_matrix(E: np.ndarray, method: str = "rank") -> np.ndarray:
    """Return energy difference 'distance' in [0,1]."""
    N = len(E)
    if N <= 1:
        return np.zeros((N, N), dtype=np.float32)
    if method == "rank":
        ranks = E.argsort().argsort().astype(np.float64)  # 0..N-1
        R = np.abs(ranks[:, None] - ranks[None, :]) / max(1, N - 1)
        return R.astype(np.float32)
    else:
        med = np.median(E)
        mad = np.median(np.abs(E - med)) + 1e-9
        D = np.abs(E[:, None] - E[None, :]) / (mad * 6.0)  # ~IQR scale
        return np.clip(D, 0.0, 1.0).astype(np.float32)


def combine_distance(dm_hamming: np.ndarray, E: np.ndarray, alpha: float = 0.8, method: str = "rank") -> np.ndarray:
    """d = alpha * Hamming + (1 - alpha) * EnergyDiff."""
    dm_E = energy_diff_matrix(E, method=method)
    return alpha * dm_hamming + (1.0 - alpha) * dm_E

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
    ranks = E.argsort().argsort().astype(np.float64)  # 0=lowest energy
    r = ranks / max(1, N - 1)
    w = np.exp(-beta * r)
    w = np.clip(w, 1e-8, None)
    w = w * (N / w.sum())  # normalize to mean ~1
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
        # max cluster fraction by weights
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
    """Cluster on main_vectors with Hamming/energy-guided distance and pick the best-energy cluster."""
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

    def _distance_matrix(self):
        assert self.vec_mat is not None
        self.dm_ = hamming_distance_matrix(self.vec_mat)

    def fit(self):
        if self.df_raw is None:
            raise RuntimeError("No data loaded.")
        self._apply_prefilter()
        if self.df_ is None or len(self.df_) == 0:
            raise RuntimeError("No rows after prefilter.")
        self._prepare_vectors()

        # optional: energy-quantile preselection (low-energy portion only)
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

        # ---- unique-level clustering (recommended) ----
        if self.cfg.collapse_identical:
            mat_u, groups, freq_w = collapse_identical_vectors(self.vec_mat, self.valid_idx)

            # representative energy per unique vector: median over its group
            ek = self.cfg.energy_key
            if ek not in self.df_.columns:
                raise ValueError(f"Energy key {ek} not found in data.")
            colE = self.df_[ek].astype(float)
            E_u = np.array([float(np.median(colE.iloc[grp].values)) for grp in groups], dtype=np.float64)

            # base distances (Hamming on unique vectors)
            dm_u_ham = hamming_distance_matrix(mat_u)

            # composite distance (C1)
            if self.cfg.use_energy_in_distance:
                dm_u = combine_distance(dm_u_ham, E_u, alpha=self.cfg.energy_alpha,
                                        method=self.cfg.energy_distance_method)
            else:
                dm_u = dm_u_ham

            # choose k with balance (use frequency weights for diagnostics)
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

            # expand back to sample-level labels
            labels_full = np.empty(sum(len(g) for g in groups), dtype=int)
            ptr = 0
            for cid, grp in zip(labels_u, groups):
                labels_full[ptr:ptr + len(grp)] = cid
                ptr += len(grp)
            self.labels_ = labels_full

            # keep Hamming (unique) for optional later diagnostics
            self.dm_ = dm_u_ham

        else:
            # fallback: no collapsing (not recommended on highly duplicated data)
            self._distance_matrix()
            dm = self.dm_
            assert dm is not None
            n = dm.shape[0]

            # optionally blend energy into distance
            if self.cfg.use_energy_in_distance:
                ek = self.cfg.energy_key
                E = self.df_.iloc[self.valid_idx][ek].astype(float).values
                dm = combine_distance(dm, E, alpha=self.cfg.energy_alpha, method=self.cfg.energy_distance_method)

            if n < 2:
                self.labels_ = np.zeros(n, dtype=int)
            else:
                if self.cfg.method == "hdbscan" and _HDBSCAN_AVAILABLE:
                    min_cs = self.cfg.min_cluster_size or max(5, int(math.sqrt(n)))
                    self.labels_ = cluster_with_hdbscan(dm, min_cs)
                else:
                    if self.cfg.use_weighted_pam:
                        ek = self.cfg.energy_key
                        E = self.df_.iloc[self.valid_idx][ek].astype(float).values
                        w = energy_to_weights(E, beta=self.cfg.energy_weight_beta)
                        best_k, _ = pick_k_balanced(dm, self.rng, self.cfg.k_candidates, weights=w,
                                                    max_cluster_frac=self.cfg.max_cluster_frac,
                                                    silhouette_floor=self.cfg.silhouette_floor,
                                                    penalty_lambda=self.cfg.penalty_lambda)
                        labels, _ = pam_kmedoids_weighted(dm, best_k, self.rng, weights=w)
                    else:
                        # plain k-medoids with silhouette-driven k
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

        # ---- energy stats & pick best cluster by energy ----
        assert self.labels_ is not None
        working_df = self.df_.iloc[self.valid_idx].copy()
        if self.cfg.energy_key not in working_df.columns:
            raise ValueError(f"Energy key {self.cfg.energy_key} not found in data.")
        stats = compute_cluster_energy_stats(working_df, self.labels_, self.cfg.energy_key)
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
