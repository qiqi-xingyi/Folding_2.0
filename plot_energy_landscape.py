# -*- coding: utf-8 -*-
# @file: plot_sampling_funnel_landmark_mds.py
#
# Purpose:
#   Distortion-free, memory-safe visualization of quantum sampling:
#     (1) True 2D distribution of sampled bitstrings (top view).
#     (2) Rank-driven 3D funnel using the SAME (X,Y) coordinates.
#
# Method:
#   • Landmark-MDS on Hamming distances:
#       - Randomly choose M "landmarks" (M << N).
#       - Compute M×M Hamming distances and run classical MDS (scikit-learn) to 2D.
#       - For the remaining points, use out-of-sample multilateration to place them
#         in the same 2D space using their distances to the landmarks only.
#   • The whole canvas is translated so rank-1 sits at the origin and then scaled
#     isotropically to fit in [-1,1]² (no aspect distortion).
#   • 2D figure: show sampling points colored by kNN density (contrast enhanced).
#   • 3D figure: Z = -(n - rank)/(n - 1) ∈ [-1,0]; rank-1 at the deepest center.
#
# Usage:
#   python plot_sampling_funnel_landmark_mds.py
#   python plot_sampling_funnel_landmark_mds.py --case 6czf \
#       --sampling_csv quantum_data/6czf/samples_6czf_all_ibm.csv \
#       --pred_csv predictions/6czf_pred.csv \
#       --landmarks 2000 --batch_size 8000 --seed 42 --topk 20
#
# Requirements:
#   pip install numpy pandas matplotlib scikit-learn
#
# Notes:
#   - No seaborn. Fonts set to Arial if available.
#   - For very large N, keep landmarks 1–5k; memory scales with M² (float32).


import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

# ======================== path =========================
INPUT_PATH = Path("e_results/1m7y/energies.jsonl")
OUTDIR = Path("result_summary/landscape")
# ========================================================

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def median_absolute_deviation(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)

def robust_scale_matrix(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, 1)
    vals = D[iu]
    med = np.median(vals)
    mad = median_absolute_deviation(vals)
    if mad == 0:
        mad = np.std(vals) + 1e-12
    S = (D - med) / mad
    S[S < 0] = 0.0
    np.fill_diagonal(S, 0.0)
    return S

def hamming_distance_bits(a: str, b: str) -> int:
    L = max(len(a), len(b))
    aa = a.ljust(L, '0')
    bb = b.ljust(L, '0')
    return sum(ch1 != ch2 for ch1, ch2 in zip(aa, bb))

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if P.shape != Q.shape:
        raise ValueError("P and Q must have same shape (N,3)")
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

def rmsd_after_kabsch(P: np.ndarray, Q: np.ndarray) -> float:
    R, t = kabsch(P, Q)
    Q_aligned = (R @ Q.T).T + t
    diff = P - Q_aligned
    return float(np.sqrt((diff ** 2).sum() / P.shape[0]))

def parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def extract_positions(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    pos = entry.get("main_positions", None)
    if (pos is None) or (len(pos) == 0):
        return None
    return np.array(pos, dtype=float)

def equalize_lengths(arrays: List[np.ndarray]) -> List[np.ndarray]:
    Lmin = min(a.shape[0] for a in arrays)
    return [a[:Lmin] for a in arrays]

# -----------------------------
# Distance Construction
# -----------------------------
def build_struct_distance(positions_list: List[np.ndarray],
                          bitstrings: List[str],
                          alpha: float = 0.8) -> np.ndarray:
    n = len(positions_list)
    # RMSD
    rmsd = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rmsd_ij = rmsd_after_kabsch(positions_list[i], positions_list[j])
            rmsd[i, j] = rmsd[j, i] = rmsd_ij
    rmsd_s = robust_scale_matrix(rmsd)
    # Hamming
    ham = np.zeros((n, n), dtype=float)
    for i in range(n):
        bi = bitstrings[i]
        for j in range(i + 1, n):
            ham[i, j] = ham[j, i] = hamming_distance_bits(bi, bitstrings[j])
    if np.max(ham) > 0:
        ham = ham / np.max(ham)
    ham_s = robust_scale_matrix(ham)
    D_struct = alpha * rmsd_s + (1 - alpha) * ham_s
    np.fill_diagonal(D_struct, 0.0)
    return D_struct

def build_energy_terms(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    Emin = float(np.min(E))
    dE = E - Emin
    tau = median_absolute_deviation(dE)
    if tau <= 1e-12:
        tau = float(np.percentile(dE, 75) - np.percentile(dE, 25) + 1e-12)
    Ehat = dE / (median_absolute_deviation(dE) + 1e-12)
    w = np.exp(-dE / (tau + 1e-12))
    return Ehat, w, Emin

def combine_distance(D_struct: np.ndarray,
                     Ehat: np.ndarray,
                     w: np.ndarray,
                     beta: float = 0.5,
                     lam: float = 0.4) -> np.ndarray:
    n = D_struct.shape[0]
    dE = np.abs(Ehat.reshape(-1, 1) - Ehat.reshape(1, -1))
    D = np.sqrt(np.maximum(D_struct, 0.0) ** 2 + beta * (dE ** 2))
    wavg = (w.reshape(-1, 1) + w.reshape(1, -1)) / 2.0
    D_tilde = (1.0 - lam * wavg) * D
    np.fill_diagonal(D_tilde, 0.0)
    return D_tilde

# -----------------------------
# Embeddings
# -----------------------------
def embed_angles_from_distance(D_tilde: np.ndarray, random_state: int = 42) -> np.ndarray:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state, n_init=4, max_iter=600)
    XY = mds.fit_transform(D_tilde)
    return XY

def build_funnel_3d(D_struct: np.ndarray,
                    D_tilde: np.ndarray,
                    Ehat: np.ndarray,
                    w: np.ndarray,
                    idx_min: int,
                    eta: float = 0.4,
                    kappa: float = 0.3,
                    p_power: float = 0.5,
                    a_slope: float = 1.0,
                    rough_rho: float = 0.1,
                    random_state: int = 42):
    n = D_struct.shape[0]
    r_raw = D_struct[idx_min, :].copy()
    r = r_raw * (1.0 - eta * w) + kappa * np.power(np.maximum(Ehat, 0.0), p_power)
    r[idx_min] = 0.0
    XY = embed_angles_from_distance(D_tilde, random_state=random_state)
    XYc = XY - XY[idx_min, :]
    theta = np.arctan2(XYc[:, 1], XYc[:, 0])
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = a_slope * Ehat.copy()

    # 粗糙度（本地多样性）
    k = min(8, n - 1) if n > 1 else 1
    if k >= 2:
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed")
        nbrs.fit(D_struct)
        idxs_mat = nbrs.kneighbors(D_struct, return_distance=False)
        eps = np.zeros(n, dtype=float)
        for i in range(n):
            idxs = idxs_mat[i][1:]
            local = D_struct[i, idxs]
            eps[i] = rough_rho * median_absolute_deviation(local)
        z += eps

    # 确保最低能量点 z 最低
    z[idx_min] = np.min(z) - 1e-6
    return x, y, z

def embed_2d_density(D_tilde: np.ndarray,
                     w: np.ndarray,
                     idx_min: int,
                     eta_prime: float = 0.4,
                     random_state: int = 42):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state, n_init=4, max_iter=600)
    XY = mds.fit_transform(D_tilde)
    XY = XY - XY[idx_min, :]
    XY = (1.0 - eta_prime * w).reshape(-1, 1) * XY
    return XY[:, 0], XY[:, 1]

# -----------------------------
# Plotting
# -----------------------------
def plot_funnel_3d(x, y, z, E, idx_min: int, Emin: float, outdir: Path):
    ensure_dir(outdir)
    fig = plt.figure(figsize=(8, 7), dpi=180)
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, s=18, c=E, cmap="viridis", alpha=0.9, depthshade=True)
    ax.scatter([x[idx_min]], [y[idx_min]], [z[idx_min]], s=90, c="red",
               marker="*", edgecolor="k", linewidths=0.6, zorder=5)

    try:
        tri = Triangulation(x, y)
        ax.plot_trisurf(tri, z, cmap="viridis", alpha=0.25, linewidth=0.2, antialiased=True)
    except Exception:
        pass

    ax.set_title(f"3D Funnel (min-E = {Emin:.3f})")
    ax.set_xlabel("X (structure-aware radius & angle)")
    ax.set_ylabel("Y (structure-aware radius & angle)")
    ax.set_zlabel("Relative Energy (z)")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label("E_total")

    ax.view_init(elev=28, azim=38)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"funnel_3d.{ext}", bbox_inches="tight")
    plt.close(fig)

def plot_density_2d(X, Y, E, idx_min: int, Emin: float, outdir: Path):
    ensure_dir(outdir)
    fig = plt.figure(figsize=(7.5, 7), dpi=180)
    ax = fig.add_subplot(111)

    hb = ax.hexbin(X, Y, gridsize=40, mincnt=1, cmap="Blues", linewidths=0.0)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Local density (counts per hex)")

    sc = ax.scatter(X, Y, c=E, s=14, cmap="viridis", alpha=0.85, edgecolors="none")
    ax.scatter([X[idx_min]], [Y[idx_min]], s=90, c="red",
               marker="*", edgecolor="k", linewidths=0.6, zorder=5)

    ax.set_title(f"2D Density (min-E = {Emin:.3f})")
    ax.set_xlabel("Dim-1 (energy-aware similarity)")
    ax.set_ylabel("Dim-2 (energy-aware similarity)")
    ax.axhline(0, color="gray", lw=0.6, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.6, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"density_2d.{ext}", bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main (IDE 直接运行)
# -----------------------------
def main():
    ensure_dir(OUTDIR)
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    rows = parse_jsonl(INPUT_PATH)
    if len(rows) == 0:
        raise RuntimeError("Input JSONL is empty.")

    bitstrings, positions, energies = [], [], []
    for r in rows:
        bs = r.get("bitstring")
        E = r.get("E_total")
        P = extract_positions(r)
        if (bs is None) or (P is None) or (E is None):
            continue
        bitstrings.append(str(bs))
        positions.append(P)
        energies.append(float(E))

    if len(positions) < 3:
        raise RuntimeError("Not enough valid entries (need >=3 with positions & energy).")

    positions = equalize_lengths(positions)
    E = np.array(energies, dtype=float)
    idx_min = int(np.argmin(E))

    D_struct = build_struct_distance(positions, bitstrings, alpha=0.8)
    Ehat, w, Emin = build_energy_terms(E)
    D_tilde = combine_distance(D_struct, Ehat, w, beta=0.5, lam=0.4)

    # 3D
    x, y, z = build_funnel_3d(
        D_struct=D_struct, D_tilde=D_tilde, Ehat=Ehat, w=w, idx_min=idx_min,
        eta=0.4, kappa=0.3, p_power=0.5, a_slope=1.0, rough_rho=0.1, random_state=42
    )
    plot_funnel_3d(x, y, z, E=E, idx_min=idx_min, Emin=Emin, outdir=OUTDIR)

    # 2D
    X2, Y2 = embed_2d_density(D_tilde=D_tilde, w=w, idx_min=idx_min,
                              eta_prime=0.4, random_state=42)
    plot_density_2d(X2, Y2, E=E, idx_min=idx_min, Emin=Emin, outdir=OUTDIR)

    np.savez(OUTDIR / "coords_outputs.npz",
             x3d=x, y3d=y, z3d=z, X2d=X2, Y2d=Y2, E=E,
             idx_min=idx_min, D_struct=D_struct, D_tilde=D_tilde)

    print(f"[saved] {OUTDIR/'funnel_3d.png'}")
    print(f"[saved] {OUTDIR/'funnel_3d.pdf'}")
    print(f"[saved] {OUTDIR/'density_2d.png'}")
    print(f"[saved] {OUTDIR/'density_2d.pdf'}")
    print(f"[saved] {OUTDIR/'coords_outputs.npz'}")

if __name__ == "__main__":
    main()

