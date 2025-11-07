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

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

# ---------- Global style ----------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 150,
})
CMAP_BLUES = "Blues"
CENTER_COLOR = "#1f77b4"
OUT_DIR_BASE = Path("result_summary/landscape")

# ---------- IO ----------
def read_sampling_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "bitstring" not in df.columns:
        raise ValueError("Sampling CSV must contain a 'bitstring' column.")
    df["bitstring"] = df["bitstring"].astype(str)
    return df[["bitstring"]].copy()

def read_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "bitstring" not in df.columns:
        raise ValueError("Prediction CSV must contain a 'bitstring' column.")
    df["bitstring"] = df["bitstring"].astype(str)
    if "rank" not in df.columns:
        df = df.sort_values("bitstring").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    else:
        df = df.sort_values("rank", ascending=True).reset_index(drop=True)
    return df[["bitstring", "rank"]].copy()

# ---------- Bit utilities ----------
def build_bit_matrix(bitstrings):
    """Return dense (N, L) float32 matrix with 0/1 values."""
    L = max(len(s) for s in bitstrings)
    padded = [s.zfill(L) for s in bitstrings]
    X = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8).reshape(-1, L) - ord("0")
    return X.astype(np.float32)  # (N, L)

def hamming_dm_via_mm(A, B):
    """
    Hamming distance matrix using matrix multiplications.
    A: (NA, L) in {0,1}, B: (NB, L) in {0,1}
    H = sum(A,1) + sum(B,1)^T - 2*A*B^T
    Returns float32 (NA, NB).
    """
    sum_a = A.sum(axis=1, dtype=np.float32)[:, None]       # (NA,1)
    sum_b = B.sum(axis=1, dtype=np.float32)[None, :]       # (1,NB)
    dot = A @ B.T                                          # (NA,NB)
    H = sum_a + sum_b - 2.0 * dot
    return H

# ---------- Landmark-MDS embedding ----------
def landmark_mds_embed(bitstrings, M=2000, batch_size=8000, seed=42, mds_max_iter=400, mds_n_init=2):
    """
    Returns:
      XY: (N,2) embedding for ALL points (landmarks first, then non-landmarks).
      order_idx: index array of length N describing the permutation applied
                 (landmarks first). Useful for reordering downstream.
      lm_idx: indices of landmarks within the original order.
    """
    rng = np.random.default_rng(seed)
    N = len(bitstrings)
    X = build_bit_matrix(bitstrings)  # (N,L)

    # Choose landmarks
    M = min(M, N)
    lm_idx = rng.choice(N, size=M, replace=False)
    non_lm_idx = np.setdiff1d(np.arange(N), lm_idx)

    X_lm = X[lm_idx]  # (M,L)

    # M×M Hamming distances among landmarks
    D_ll = hamming_dm_via_mm(X_lm, X_lm).astype(np.float32)

    # Classical MDS on landmarks (dissimilarity = Hamming)
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=seed, n_init=mds_n_init, max_iter=mds_max_iter)
    Y_lm = mds.fit_transform(D_ll).astype(np.float32)  # (M,2)

    # Precompute for out-of-sample multilateration
    y_norm2 = (Y_lm ** 2).sum(axis=1)                  # (M,)
    P_cache = []   # list of (2,M) arrays
    Ginv_cache = []  # list of (2,2) arrays
    for r in range(M):
        P_r = (2.0 * (Y_lm - Y_lm[r])).T              # (2,M)
        G_r = P_r @ P_r.T                              # (2,2)
        # regularize if near-singular
        G_r += 1e-8 * np.eye(2, dtype=np.float32)
        Ginv_cache.append(np.linalg.inv(G_r).astype(np.float32))
        P_cache.append(P_r.astype(np.float32))

    # Allocate result
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[lm_idx] = Y_lm

    # Stream the remaining points in batches
    if len(non_lm_idx) > 0:
        for s in range(0, len(non_lm_idx), batch_size):
            e = min(len(non_lm_idx), s + batch_size)
            idx_batch = non_lm_idx[s:e]
            Xb = X[idx_batch]                           # (B,L)

            # Distances to all landmarks: (B,M)
            d_bl = hamming_dm_via_mm(Xb, X_lm).astype(np.float32)
            d2 = d_bl ** 2

            # For each row, choose the closest landmark as reference r
            r_idx = np.argmin(d_bl, axis=1)             # (B,)

            # Prepare outputs
            Yb = np.zeros((len(idx_batch), 2), dtype=np.float32)

            for i in range(len(idx_batch)):
                r = int(r_idx[i])
                # b = (||y_j||^2 - ||y_r||^2) + (d_r^2 - d_j^2) for all j
                b = (y_norm2 - y_norm2[r]) + (d2[i, r] - d2[i, :])         # (M,)
                # v = P_r @ b ; x = Ginv_r @ v
                v = P_cache[r] @ b                                         # (2,)
                x = Ginv_cache[r] @ v                                      # (2,)
                Yb[i] = x

            Y[idx_batch] = Yb

    # Order: landmarks first, then non-landmarks
    order_idx = np.concatenate([lm_idx, non_lm_idx], axis=0)
    return Y[order_idx], order_idx, lm_idx

# ---------- Center & scale ----------
def center_and_scale(XY, center_idx=None, margin=0.02):
    XYc = XY.copy()
    if center_idx is not None:
        XYc -= XYc[center_idx]
    m = np.abs(XYc).max()
    s = 1.0 if m < 1e-9 else (1.0 - margin) / m
    XYc *= s
    return XYc

# ---------- Density & contrast ----------
def knn_density_2d(XY, k=20):
    if len(XY) == 0:
        return np.zeros(0, dtype=float)
    k = max(2, min(int(k), len(XY)))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(XY)
    dists, _ = nn.kneighbors(XY)
    mean_d = dists[:, 1:].mean(axis=1)
    dens = 1.0 / (mean_d + 1e-12)
    dmin, dmax = dens.min(), dens.max()
    if dmax > dmin:
        dens = (dens - dmin) / (dmax - dmin)
    else:
        dens = np.zeros_like(dens)
    return dens

def enhance_contrast(values, floor=0.28, vmax_pct=0.80, gamma=0.75):
    if len(values) == 0:
        return values
    v = values.astype(float)
    vmax = np.quantile(v, vmax_pct)
    if vmax <= 1e-12:
        vmax = 1.0
    v = np.clip(v / vmax, 0.0, 1.0)
    v = np.power(v, gamma)                  # lift lows
    v = floor + (1.0 - floor) * v           # non-zero floor
    return v

# ---------- Plots ----------
def plot_sampling_topview(XY_union, union_bits, sampling_bits,
                          case_id, out_dir, k_density=20,
                          floor=0.28, vmax_pct=0.80, gamma=0.75,
                          gridsize=60, point_size=8, alpha_pts=0.95):
    samp_set = set(sampling_bits)
    mask = np.array([b in samp_set for b in union_bits], dtype=bool)
    XYs = XY_union[mask]
    dens = knn_density_2d(XYs, k=k_density) if len(XYs) >= 3 else np.zeros(len(XYs))
    dens_vis = enhance_contrast(dens, floor=floor, vmax_pct=vmax_pct, gamma=gamma)

    fig, ax = plt.subplots(figsize=(8, 8))
    if len(XYs) > 0:
        ax.hexbin(XYs[:, 0], XYs[:, 1], gridsize=gridsize, mincnt=1,
                  cmap=CMAP_BLUES, alpha=0.22, linewidths=0.0)
        sc = ax.scatter(XYs[:, 0], XYs[:, 1], c=dens_vis, cmap=CMAP_BLUES,
                        s=point_size, alpha=alpha_pts, edgecolors="none")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Local density (enhanced)")
    ax.set_title(f"{case_id}: Sampling distribution (true top view)")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_sampling_topview_landmarkMDS.{ext}", dpi=300)
    plt.close(fig)

def plot_funnel_3d(XY_union, union_bits, df_pred, case_id, out_dir,
                   topk=20, cone_alpha=0.28, point_size=10):
    pred_bits = df_pred["bitstring"].tolist()
    n_pred = len(pred_bits)
    XYp = XY_union[:n_pred, :]  # predictions are placed first (see main)

    ranks = df_pred["rank"].to_numpy()
    n = len(ranks)
    if n == 1:
        Z = np.array([-1.0], dtype=float)
    else:
        Z = - (n - ranks) / (n - 1)  # [-1,0]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # reference cone
    Rg = np.linspace(0.0, 1.0, 90)
    Tg = np.linspace(0.0, 2 * np.pi, 180)
    Rm, Tm = np.meshgrid(Rg, Tg)
    Xs = Rm * np.cos(Tm)
    Ys = Rm * np.sin(Tm)
    Zs = -Rm
    ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2,
        cmap=CMAP_BLUES, alpha=cone_alpha, edgecolor="none")

    norm = Normalize(vmin=Z.min(), vmax=Z.max())
    colors = plt.cm.get_cmap(CMAP_BLUES)(norm(Z))
    ax.scatter(XYp[:, 0], XYp[:, 1], Z, s=point_size, c=colors,
               depthshade=False, edgecolors="k", linewidths=0.2)

    i_best = int(np.argmin(ranks))
    ax.scatter([0.0], [0.0], [Z[i_best]], s=160, c=CENTER_COLOR,
               edgecolors="k", linewidths=0.8, depthshade=False, label="Rank-1")

    kk = min(topk, n)
    order = np.argsort(ranks)[:kk]
    for pos, i in enumerate(order, start=1):
        ax.text(XYp[i, 0], XYp[i, 1], Z[i], f"{pos}", fontsize=10,
                ha="center", va="bottom", color="k")

    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(min(-1.2, Z.min() - 0.05), 0.15)
    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{case_id}: Re-ranked energy funnel (shared XY, landmark-MDS)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Normalized depth")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_funnel_3d_landmarkMDS.{ext}", dpi=300)
    plt.close(fig)

    out = df_pred.copy()
    out["x"] = XYp[:, 0]; out["y"] = XYp[:, 1]; out["z_norm_rank"] = Z
    out.to_csv(out_dir / f"{case_id}_funnel_mapping_landmarkMDS.csv", index=False)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Sampling top view + rank funnel with Landmark-MDS (Hamming).")
    parser.add_argument("--case", type=str, default="6czf", help="Case ID")
    parser.add_argument("--sampling_csv", type=Path, default=None, help="Path to sampling CSV")
    parser.add_argument("--pred_csv", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--landmarks", type=int, default=2000, help="Number of landmark points (M)")
    parser.add_argument("--batch_size", type=int, default=8000, help="Batch size for out-of-sample embedding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mds_max_iter", type=int, default=400, help="MDS max iterations for landmarks")
    parser.add_argument("--mds_n_init", type=int, default=2, help="MDS n_init for landmarks")
    # density/visual controls
    parser.add_argument("--k_density", type=int, default=20, help="k for kNN density")
    parser.add_argument("--density_floor", type=float, default=0.28, help="Color floor for low densities")
    parser.add_argument("--density_vmax_pct", type=float, default=0.80, help="Upper percentile clipping")
    parser.add_argument("--density_gamma", type=float, default=0.75, help="Gamma (<1 darkens lows)")
    parser.add_argument("--hex_gridsize", type=int, default=60, help="Hexbin gridsize for 2D backdrop")
    parser.add_argument("--topk", type=int, default=20, help="Top-K labels in 3D")
    args = parser.parse_args()

    case_id = args.case
    sampling_csv = args.sampling_csv or Path(f"quantum_data/{case_id}/samples_{case_id}_all_ibm.csv")
    pred_csv = args.pred_csv or Path(f"predictions/{case_id}_pred.csv")
    if not sampling_csv.exists():
        raise FileNotFoundError(f"Sampling CSV not found: {sampling_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    out_dir = OUT_DIR_BASE / case_id

    # Load data
    df_samp = read_sampling_csv(sampling_csv)
    df_pred = read_prediction_csv(pred_csv)  # rank ascending
    pred_bits = df_pred["bitstring"].tolist()
    samp_bits = df_samp["bitstring"].tolist()

    # Union with predictions FIRST (so XY[:n_pred] matches predictions)
    union_bits = pred_bits + [b for b in samp_bits if b not in set(pred_bits)]

    # Landmark-MDS embedding on union
    XY_raw, order_idx, lm_idx = landmark_mds_embed(
        union_bits,
        M=args.landmarks,
        batch_size=args.batch_size,
        seed=args.seed,
        mds_max_iter=args.mds_max_iter,
        mds_n_init=args.mds_n_init,
    )

    # Center at rank-1, then scale; the first element of union_bits is rank-1 bitstring
    best_idx = 0
    XY_cs = center_and_scale(XY_raw, center_idx=best_idx, margin=0.02)

    # Save union mapping
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bitstring": union_bits, "x": XY_cs[:, 0], "y": XY_cs[:, 1]}).to_csv(
        out_dir / f"{case_id}_union_xy_mapping_landmarkMDS.csv", index=False
    )

    # Plots
    plot_sampling_topview(
        XY_union=XY_cs,
        union_bits=union_bits,
        sampling_bits=samp_bits,
        case_id=case_id,
        out_dir=out_dir,
        k_density=args.k_density,
        floor=args.density_floor,
        vmax_pct=args.density_vmax_pct,
        gamma=args.density_gamma,
        gridsize=args.hex_gridsize,
        point_size=8,
        alpha_pts=0.95,
    )

    plot_funnel_3d(
        XY_union=XY_cs,
        union_bits=union_bits,
        df_pred=df_pred,
        case_id=case_id,
        out_dir=out_dir,
        topk=args.topk,
        cone_alpha=0.28,
        point_size=10,
    )

    print("[Done]", out_dir.resolve())

if __name__ == "__main__":
    main()
