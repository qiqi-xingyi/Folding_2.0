# --*-- conding:utf-8 --*--
# @time:10/23/25 18:09
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:structure_refine.py

# -*- coding: utf-8 -*-

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    energy_weights: Dict[str, float] = field(default_factory=lambda: {"E_total": 1.0, "E_geom": 1.0, "E_steric": 1.0})
    weight_eps: float = 1e-9

    # continuous mean & projection
    gpa_iters: int = 3         # iterations of align-then-mean for continuous averaging (premium can increase)
    proj_enforce_bond: bool = True
    target_ca_distance: Optional[float] = 3.8  # Å; None -> infer from medoid mean
    proj_smooth_strength: float = 0.10   # Laplacian smoothing strength per iter
    proj_iters: int = 10
    min_separation: float = 2.5          # simple self-collision avoidance (Cα-only)

    # local polishing
    do_local_polish: bool = True
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
    """Compute exp(-alpha E_total) * exp(-beta E_geom) * exp(-gamma E_steric)."""
    val = 0.0
    for k, alpha in wcfg.items():
        if k in row and pd.notna(row[k]):
            val += alpha * float(row[k])
    w = math.exp(-max(val, -50.0))  # clamp
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
            if d < 1e-8:
                continue
            delta = (target - d) * 0.5
            dir_ = v / d
            X[i] -= delta * dir_
            X[i + 1] += delta * dir_
    return X


def repel_min_separation(coords: np.ndarray, min_sep: float, factor: float = 0.1) -> np.ndarray:
    """Push points apart if closer than min_sep (very simple repulsion)."""
    X = coords.copy()
    L = X.shape[0]
    for i in range(L):
        for j in range(i + 2, L):  # skip immediate neighbors
            v = X[j] - X[i]
            d = np.linalg.norm(v)
            if d < 1e-8:
                continue
            if d < min_sep:
                dir_ = v / d
                push = (min_sep - d) * factor
                X[i] -= push * dir_
                X[j] += push * dir_
    return X


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

        # standard/premium: build continuous mean and project
        self.medoid_ca = self.aligned_list[self.medoid_idx_in_sub].copy()
        self._continuous_mean()
        self._project_geometry()

        if self.cfg.refine_mode == "premium":
            # premium can slightly increase iterations for a tighter mean
            self._continuous_mean(extra_iters=max(0, self.cfg.gpa_iters - 3))
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

    def _select_medoid(self):
        aligned = self.aligned_list
        sums = pairwise_rmsd_sum(aligned)
        idx = int(np.argmin(sums))
        self.medoid_idx_in_sub = idx

    def _continuous_mean(self, extra_iters: int = 0):
        cfg = self.cfg
        assert self.aligned_list is not None
        L = self.aligned_list[0].shape[0]

        # initial mean as weighted average in current frame
        W = []
        for _, row in self.sub_df.iterrows():
            w = sample_weights(row, cfg.energy_weights, cfg.weight_eps)
            W.append(w)
        W = np.asarray(W, dtype=float)
        W /= W.sum() + 1e-12

        mean = np.zeros((L, 3), dtype=float)
        for w, X in zip(W, self.aligned_list):
            mean += w * X

        iters = max(0, cfg.gpa_iters) + max(0, extra_iters)
        # small number of re-alignments to the evolving mean
        for _ in range(iters):
            new_list = []
            for X in self.aligned_list:
                new_list.append(align_to_reference(mean, X))
            self.aligned_list = new_list
            mean = np.zeros((L, 3), dtype=float)
            for w, X in zip(W, self.aligned_list):
                mean += w * X

        self.mean_ca = mean

    def _project_geometry(self):
        cfg = self.cfg
        assert self.mean_ca is not None
        X = self.mean_ca.copy()

        # set target Cα distance
        if cfg.target_ca_distance is not None:
            d0 = float(cfg.target_ca_distance)
        else:
            # infer from medoid
            if self.medoid_ca is not None:
                dif = self.medoid_ca[1:] - self.medoid_ca[:-1]
                d0 = float(np.mean(np.linalg.norm(dif, axis=1)))
            else:
                dif = X[1:] - X[:-1]
                d0 = float(np.mean(np.linalg.norm(dif, axis=1)))

        # projection loop
        for _ in range(cfg.proj_iters):
            if cfg.proj_enforce_bond:
                X = enforce_neighbor_distance(X, d0, iters=1)
            X = laplacian_smooth(X, cfg.proj_smooth_strength)
            X = repel_min_separation(X, cfg.min_separation, factor=0.15)

        self.projected_ca = X

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
            # skip polishing if energy function fails
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
            Z_new = Z - 0.1 * grad  # simple step size
            # quick projection-lite to avoid drifting apart
            Z_new = enforce_neighbor_distance(Z_new, self.cfg.target_ca_distance or 3.8, iters=1)
            Z_new = repel_min_separation(Z_new, self.cfg.min_separation, factor=0.1)

            e_new = energy_total(Z_new)
            if np.isfinite(e_new) and e_new <= base_e:
                Z = Z_new
                base_e = e_new
            else:
                # early stop if no improvement
                break

        self.refined_ca = Z

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
