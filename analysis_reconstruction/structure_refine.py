# --*-- conding:utf-8 --*--
# @time:10/23/25 18:09
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:structure_refine.py

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# Config
# -------------------------------

@dataclass
class RefineConfig:
    # subsampling inside the best cluster
    subsample_max: int = 64
    top_energy_pct: float = 0.3
    random_seed: int = 0

    # anchor/medoid selection
    anchor_policy: str = "lowest_energy"  # "lowest_energy" | "hamming_medoid"

    # refining mode
    refine_mode: str = "standard"         # "fast" | "standard" | "premium"

    # columns
    positions_col: str = "main_positions"
    vectors_col: str = "main_vectors"
    energy_key: str = "E_total"
    sequence_col: str = "sequence"
    id_col: Optional[str] = None

    # weighted averaging (sample weights)
    # weights act like exp(-sum alpha_k * E_k); tune alphas below (1.0 is fine)
    energy_weights: Dict[str, float] = field(default_factory=lambda: {
        "E_total": 1.0, "E_geom": 1.0, "E_steric": 1.0
    })
    weight_eps: float = 1e-9

    # continuous mean & projection (geometry)
    gpa_iters: int = 1                # we still allow 1 round of re-align to a proto-mean
    proj_enforce_bond: bool = True
    target_ca_distance: Optional[float] = 3.8  # Å
    proj_smooth_strength: float = 0.03         # weaker smoothing to avoid collapse
    proj_iters: int = 10
    min_separation: float = 3.0                # Cα-only non-neighbor collision threshold

    # local polishing (optional if energy_fn provided)
    do_local_polish: bool = False
    local_polish_steps: int = 20
    stay_lambda: float = 0.05
    step_size: float = 0.05              # Å for finite-diff gradients if energy_fn provided

    # output
    output_dir: str = "./refine_out"

    def normalize(self):
        self.refine_mode = str(self.refine_mode).lower().strip()
        self.anchor_policy = str(self.anchor_policy).lower().strip()
        if self.refine_mode not in ("fast", "standard", "premium"):
            self.refine_mode = "standard"


# -------------------------------
# Helpers
# -------------------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_positions(obj: Any) -> Optional[np.ndarray]:
    """Return (L,3) float array from a cell value that might be list or JSON string."""
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    if not isinstance(obj, (list, tuple)):
        return None
    arr = np.asarray(obj, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return R, t that aligns Q to P (both (L,3)), minimizing ||P - (R Q + t)||."""
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Qc.T @ Pc
    V, S, Wt = np.linalg.svd(C)
    R = V @ Wt
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ Wt
    t = P.mean(axis=0) - (R @ Q.mean(axis=0))
    return R, t


def align_to_reference(ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Align X to ref and return aligned coordinates (copy)."""
    R, t = kabsch(ref, X)
    return (R @ X.T).T + t


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """RMSD between two (L,3) arrays in the same frame."""
    dif = A - B
    return float(np.sqrt((dif * dif).sum() / A.shape[0]))


def pairwise_rmsd_sum(X_list: List[np.ndarray]) -> np.ndarray:
    """Approximate medoid: sum of RMSD to others in the same frame."""
    n = len(X_list)
    sums = np.zeros(n, dtype=float)
    for i in range(n):
        acc = 0.0
        Ai = X_list[i]
        for j in range(n):
            if i == j:
                continue
            acc += rmsd(Ai, X_list[j])
        sums[i] = acc
    return sums


def sample_weights(row: pd.Series, wcfg: Dict[str, float], eps: float = 1e-9) -> float:
    """Compute exp(-alpha_k * E_k ...). Larger energy -> smaller weight."""
    val = 0.0
    for k, alpha in wcfg.items():
        if k in row and pd.notna(row[k]):
            val += alpha * float(row[k])
    # soft clamp to prevent exp overflow
    val = min(max(val, -50.0), 50.0)
    w = math.exp(-val)
    return max(w, eps)


def write_pdb_ca(path: str, coords: np.ndarray, sequence: Optional[str] = None, chain_id: str = "A"):
    """Write Cα-only PDB (ATOM records). seqres optional; residue names fallback to 'GLY'."""
    with open(path, "w", encoding="utf-8") as f:
        resn = "GLY"
        for i, (x, y, z) in enumerate(coords, start=1):
            f.write(
                "ATOM  {serial:5d}  CA  {resn} {chain}{resi:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n".format(
                    serial=i, resn=resn, chain=chain_id, resi=i, x=x, y=y, z=z
                )
            )
        f.write("END\n")


def write_csv_ca(path: str, coords: np.ndarray):
    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    df.insert(0, "res_id", np.arange(1, len(coords) + 1))
    df.to_csv(path, index=False)


def laplacian_smooth(coords: np.ndarray, strength: float) -> np.ndarray:
    """Simple Laplacian smoothing for an open chain."""
    L = coords.shape[0]
    out = coords.copy()
    for i in range(L):
        if i == 0:
            lap = coords[1] - coords[0]
        elif i == L - 1:
            lap = coords[L - 2] - coords[L - 1]
        else:
            lap = (coords[i - 1] + coords[i + 1] - 2 * coords[i])
        out[i] += strength * lap
    return out


def enforce_neighbor_distance(coords: np.ndarray, target: float, iters: int = 1):
    """Iteratively adjust adjacent pairs to approach target distance (Jacobi style)."""
    X = coords.copy()
    L = X.shape[0]
    for _ in range(iters):
        for i in range(L - 1):
            a = X[i]
            b = X[i + 1]
            v = b - a
            d = np.linalg.norm(v)
            if d < 1e-12:
                # tiny random nudge to avoid zero-length edges
                v = np.array([1e-6, 0.0, 0.0], dtype=float)
                d = 1e-6
            delta = (target - d) * 0.5
            dir_ = v / d
            X[i] -= delta * dir_
            X[i + 1] += delta * dir_
    return X


def repel_min_separation(coords: np.ndarray, min_sep: float, factor: float = 0.15) -> np.ndarray:
    """Push points apart if closer than min_sep (skip immediate neighbors)."""
    X = coords.copy()
    L = X.shape[0]
    for i in range(L):
        for j in range(i + 2, L):
            v = X[j] - X[i]
            d = np.linalg.norm(v)
            if d < 1e-12:
                continue
            if d < min_sep:
                dir_ = v / d
                push = (min_sep - d) * factor
                X[i] -= push * dir_
                X[j] += push * dir_
    return X


def unit_directions_from_coords(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute unit edge directions from coordinate polyline."""
    V = X[1:] - X[:-1]
    N = np.linalg.norm(V, axis=1, keepdims=True)
    N = np.maximum(N, eps)
    D = V / N
    return D  # (L-1, 3)


def reconstruct_by_dirs(P0: np.ndarray, D: np.ndarray, step: float) -> np.ndarray:
    """Reconstruct coordinates by accumulating unit directions with fixed step."""
    Lm1 = D.shape[0]
    P = np.zeros((Lm1 + 1, 3), dtype=float)
    P[0] = P0
    for i in range(Lm1):
        P[i + 1] = P[i] + step * D[i]
    return P


def qc_geometry(coords: np.ndarray, min_sep_nonadj: float = 2.8) -> Dict[str, float]:
    """Simple geometry QC metrics."""
    dif = coords[1:] - coords[:-1]
    dists = np.linalg.norm(dif, axis=1)
    # min distance among non-neighbor pairs
    L = coords.shape[0]
    min_nonadj = np.inf
    for i in range(L):
        for j in range(i + 2, L):
            d = np.linalg.norm(coords[j] - coords[i])
            if d < min_nonadj:
                min_nonadj = d
    return {
        "bond_mean": float(dists.mean()) if len(dists) else float("nan"),
        "bond_std": float(dists.std()) if len(dists) else float("nan"),
        "bond_min": float(dists.min()) if len(dists) else float("nan"),
        "bond_max": float(dists.max()) if len(dists) else float("nan"),
        "min_nonadjacent": float(min_nonadj) if np.isfinite(min_nonadj) else float("nan"),
    }


# -------------------------------
# Main class
# -------------------------------

class StructureRefiner:
    """
    Refine a cluster of Cα-only conformations into a single continuous, projected and optionally polished structure.
    """

    def __init__(self, cfg: RefineConfig, energy_fn: Optional[Callable[[np.ndarray], Dict[str, float]]] = None):
        self.cfg = cfg
        self.cfg.normalize()
        self.energy_fn = energy_fn

        self.cluster_df: Optional[pd.DataFrame] = None
        self.sub_df: Optional[pd.DataFrame] = None

        self.anchor_idx: Optional[int] = None
        self.aligned_list: Optional[List[np.ndarray]] = None
        self.medoid_idx_in_sub: Optional[int] = None

        self.medoid_ca: Optional[np.ndarray] = None
        self.mean_ca: Optional[np.ndarray] = None
        self.projected_ca: Optional[np.ndarray] = None
        self.refined_ca: Optional[np.ndarray] = None

        self.report: Dict[str, Any] = {}

    # ---- data ----

    def load_cluster_dataframe(self, df: pd.DataFrame):
        """df must contain positions_col and energy_key."""
        self.cluster_df = df.copy()

    # ---- pipeline ----

    def run(self):
        if self.cluster_df is None or len(self.cluster_df) == 0:
            raise RuntimeError("Empty cluster dataframe.")

        self._subsample_cluster()
        if len(self.sub_df) == 0:
            raise RuntimeError("No rows selected for refinement.")

        self._select_anchor()
        self._align_to_anchor()
        self._select_medoid()

        # fast mode: return medoid only
        if self.cfg.refine_mode == "fast":
            self.medoid_ca = self.aligned_list[self.medoid_idx_in_sub].copy()
            self.refined_ca = self.medoid_ca.copy()
            self.report = self._build_report()
            return

        # standard/premium: build continuous mean (by averaged directions) and project
        self.medoid_ca = self.aligned_list[self.medoid_idx_in_sub].copy()
        self._continuous_mean()      # now does direction-averaging + 3.8Å reconstruction
        self._project_geometry()

        if self.cfg.refine_mode == "premium":
            # minor extra pass
            self._project_geometry()

        # optional local polishing with energy function
        if self.cfg.do_local_polish and self.energy_fn is not None:
            self._local_energy_polish()
            final = self.refined_ca
        else:
            final = self.projected_ca

        self.refined_ca = final.copy()
        self.report = self._build_report()

    # ---- steps ----

    def _subsample_cluster(self):
        cfg = self.cfg
        df = self.cluster_df.copy()

        # parse positions and drop invalid rows
        valid_mask = []
        parsed_positions = []
        for _, row in df.iterrows():
            pos = parse_positions(row.get(cfg.positions_col))
            valid_mask.append(pos is not None)
            parsed_positions.append(pos)
        df = df[valid_mask].copy()
        df["_parsed_positions"] = parsed_positions
        if len(df) == 0:
            raise RuntimeError("No valid positions in cluster.")

        # sort by energy
        if cfg.energy_key not in df.columns:
            raise ValueError(f"Energy key {cfg.energy_key} not in dataframe.")
        df = df.sort_values(cfg.energy_key, ascending=True).reset_index(drop=True)

        # take top percentile
        k_top = max(1, int(math.ceil(len(df) * cfg.top_energy_pct)))
        top_df = df.iloc[:k_top].copy()

        # random fill to subsample_max
        if len(top_df) < min(cfg.subsample_max, len(df)):
            rng = np.random.RandomState(cfg.random_seed)
            pool = df.iloc[k_top:]
            need = min(cfg.subsample_max, len(df)) - len(top_df)
            if need > 0 and len(pool) > 0:
                idx = rng.choice(pool.index.values, size=min(need, len(pool)), replace=False)
                extra = pool.loc[idx]
                sub_df = pd.concat([top_df, extra], axis=0).reset_index(drop=True)
            else:
                sub_df = top_df
        else:
            sub_df = top_df.iloc[: cfg.subsample_max].copy()

        self.sub_df = sub_df

    def _select_anchor(self):
        cfg = self.cfg
        df = self.sub_df
        if cfg.anchor_policy == "lowest_energy":
            self.anchor_idx = int(df[cfg.energy_key].astype(float).idxmin())
        else:
            # fallback to lowest energy
            self.anchor_idx = int(df[cfg.energy_key].astype(float).idxmin())

    def _align_to_anchor(self):
        df = self.sub_df
        assert self.anchor_idx is not None

        anchor_row = df.loc[self.anchor_idx]
        anchor = anchor_row["_parsed_positions"]
        L = anchor.shape[0]

        aligned = []
        for _, row in df.iterrows():
            X = row["_parsed_positions"]
            if X.shape[0] != L:
                # simple trimming to minimal length
                Lmin = min(L, X.shape[0])
                ref = anchor[:Lmin]
                tgt = X[:Lmin]
                aligned_X = align_to_reference(ref, tgt)
            else:
                aligned_X = align_to_reference(anchor, X)
            aligned.append(aligned_X)
        self.aligned_list = aligned

        # one optional GPA iteration to stabilize orientation around a proto-mean
        if self.cfg.gpa_iters > 0:
            mean0 = np.mean(np.stack(self.aligned_list, axis=0), axis=0)
            new_list = []
            for X in self.aligned_list:
                new_list.append(align_to_reference(mean0, X))
            self.aligned_list = new_list

    def _select_medoid(self):
        aligned = self.aligned_list
        sums = pairwise_rmsd_sum(aligned)
        idx = int(np.argmin(sums))
        self.medoid_idx_in_sub = idx

    def _continuous_mean(self, extra_iters: int = 0):
        """
        Build a continuous mean by averaging unit directions and reconstructing
        with a fixed Cα step (default 3.8 Å) from the anchor start point.
        """
        cfg = self.cfg
        assert self.aligned_list is not None
        L = self.aligned_list[0].shape[0]

        # sample weights by energy (softmax-like via exp(-sum alpha E))
        W = []
        for _, row in self.sub_df.iterrows():
            w = sample_weights(row, cfg.energy_weights, cfg.weight_eps)
            W.append(w)
        W = np.asarray(W, dtype=float)
        W /= W.sum() + 1e-12

        # compute unit edge directions for each aligned sample
        dirs_list = []
        for X in self.aligned_list:
            D = unit_directions_from_coords(X)  # (L-1,3)
            dirs_list.append(D)
        D_all = np.stack(dirs_list, axis=0)     # (M, L-1, 3)

        # energy-weighted mean direction and re-normalize per edge
        Dbar = (W[:, None, None] * D_all).sum(axis=0)  # (L-1,3)
        norms = np.linalg.norm(Dbar, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        Dbar = Dbar / norms

        step = float(cfg.target_ca_distance or 3.8)
        # use medoid's first CA as the start to minimize global drift
        if self.medoid_idx_in_sub is not None:
            P0 = self.aligned_list[self.medoid_idx_in_sub][0]
        else:
            P0 = np.zeros(3, dtype=float)

        mean = reconstruct_by_dirs(P0, Dbar, step)  # (L,3)
        self.mean_ca = mean

    def _project_geometry(self):
        """
        Projection: strong neighbor enforcement -> light smoothing -> repulsion,
        repeated for cfg.proj_iters; finally do an arc-length reconstruction to
        lock all Cα-Cα distances to the target step.
        """
        cfg = self.cfg
        assert self.mean_ca is not None
        X = self.mean_ca.copy()

        # set target Cα distance
        d0 = float(cfg.target_ca_distance or 3.8)

        for _ in range(cfg.proj_iters):
            if cfg.proj_enforce_bond:
                X = enforce_neighbor_distance(X, d0, iters=10)        # strong
            X = laplacian_smooth(X, cfg.proj_smooth_strength)         # light
            X = repel_min_separation(X, cfg.min_separation, factor=0.15)

        # exit insurance: arc-length reconstruction at exact d0
        D = unit_directions_from_coords(X)
        X_fixed = reconstruct_by_dirs(X[0], D, d0)
        self.projected_ca = X_fixed

        D = unit_directions_from_coords(X)
        X_fixed = reconstruct_by_dirs(X[0], D, d0)

        # --- NEW: post-pass to resolve residual close contacts ---
        for _ in range(5):  # a few safety passes
            # quick QC
            min_nonadj = np.inf
            L = X_fixed.shape[0]
            for i in range(L):
                for j in range(i + 2, L):
                    dij = np.linalg.norm(X_fixed[j] - X_fixed[i])
                    if dij < min_nonadj:
                        min_nonadj = dij
            if min_nonadj >= self.cfg.min_separation:
                break
            # repel & re-lock
            X_fixed = repel_min_separation(X_fixed, self.cfg.min_separation, factor=0.25)
            D = unit_directions_from_coords(X_fixed)
            X_fixed = reconstruct_by_dirs(X_fixed[0], D, d0)

        self.projected_ca = X_fixed

    def _local_energy_polish(self):
        cfg = self.cfg
        assert self.projected_ca is not None
        if self.energy_fn is None:
            self.refined_ca = self.projected_ca.copy()
            return

        Z = self.projected_ca.copy()
        Q = self.projected_ca.copy()

        def energy_total(coords: np.ndarray) -> float:
            try:
                d = self.energy_fn(coords)
                return float(d.get("E_total", np.nan))
            except Exception:
                return np.nan

        base_e = energy_total(Z)
        if not np.isfinite(base_e):
            self.refined_ca = self.projected_ca.copy()
            return

        for _ in range(max(1, cfg.local_polish_steps)):
            grad = np.zeros_like(Z)
            # finite difference gradient
            for i in range(Z.shape[0]):
                for k in range(3):
                    Z[i, k] += cfg.step_size
                    e_plus = energy_total(Z)
                    Z[i, k] -= 2 * cfg.step_size
                    e_minus = energy_total(Z)
                    Z[i, k] += cfg.step_size  # restore
                    if np.isfinite(e_plus) and np.isfinite(e_minus):
                        grad[i, k] = (e_plus - e_minus) / (2 * cfg.step_size)

            # add stay term gradient
            grad += 2.0 * cfg.stay_lambda * (Z - Q)

            # take a small step
            Z_new = Z - 0.1 * grad
            # projection-lite to keep geometry sane
            Z_new = enforce_neighbor_distance(Z_new, d0=cfg.target_ca_distance or 3.8, iters=2)
            Z_new = repel_min_separation(Z_new, self.cfg.min_separation, factor=0.10)

            e_new = energy_total(Z_new)
            if np.isfinite(e_new) and e_new <= base_e:
                Z = Z_new
                base_e = e_new
            else:
                break

        # final lock by arc-length reconstruction
        D = unit_directions_from_coords(Z)
        self.refined_ca = reconstruct_by_dirs(Z[0], D, float(cfg.target_ca_distance or 3.8))

    # ---- outputs ----

    def get_outputs(self) -> Dict[str, Any]:
        return dict(
            medoid_ca=self.medoid_ca,
            mean_ca=self.mean_ca,
            projected_ca=self.projected_ca,
            refined_ca=self.refined_ca,
            report=self.report,
        )

    def _build_report(self) -> Dict[str, Any]:
        rep: Dict[str, Any] = {}
        if self.medoid_ca is not None and self.refined_ca is not None:
            rep["rmsd_medoid_to_refined"] = rmsd(self.medoid_ca, self.refined_ca)
        if self.mean_ca is not None and self.refined_ca is not None:
            rep["rmsd_mean_to_refined"] = rmsd(self.mean_ca, self.refined_ca)
        rep["n_cluster"] = 0 if self.cluster_df is None else int(len(self.cluster_df))
        rep["n_used_for_refine"] = 0 if self.sub_df is None else int(len(self.sub_df))
        rep["mode"] = self.cfg.refine_mode

        # geometry QC
        if self.refined_ca is not None:
            qc = qc_geometry(self.refined_ca, min_sep_nonadj=self.cfg.min_separation)
            rep.update({
                "bond_mean": qc["bond_mean"],
                "bond_std": qc["bond_std"],
                "bond_min": qc["bond_min"],
                "bond_max": qc["bond_max"],
                "min_nonadjacent": qc["min_nonadjacent"],
            })
        return rep

    def save_outputs(self):
        if self.refined_ca is None:
            raise RuntimeError("Nothing to save. Run .run() first.")
        ensure_outdir(self.cfg.output_dir)

        # choose sequence if available (optional)
        seq = None
        if self.cluster_df is not None and self.cfg.sequence_col in self.cluster_df.columns:
            seq = str(self.cluster_df.iloc[0][self.cfg.sequence_col])

        # save PDB/CSV
        med = self.medoid_ca if self.medoid_ca is not None else self.refined_ca
        write_pdb_ca(os.path.join(self.cfg.output_dir, "medoid_ca.pdb"), med, seq)
        write_pdb_ca(os.path.join(self.cfg.output_dir, "refined_ca.pdb"), self.refined_ca, seq)
        write_csv_ca(os.path.join(self.cfg.output_dir, "refined_ca.csv"), self.refined_ca)

        # report
        rep = dict(self.report)
        with open(os.path.join(self.cfg.output_dir, "refine_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
