# -*- coding: utf-8 -*-
# @file: plot_sampling_funnel_memorysafe.py
#
# Purpose
#   Memory-safe, distortion-free visualization of:
#     (1) The true 2D distribution of quantum sampling bitstrings.
#     (2) A re-ranked 3D funnel using the SAME (X,Y) coordinates.
#
# Key ideas
#   • ONE shared embedding for the union of {sampling ∪ predictions} using
#     TruncatedSVD/PCA on the 0/1 bit matrix (no NxN distance matrix → no OOM).
#   • Translate so rank-1 sits at (0,0); isotropically scale to [-1,1]^2.
#   • 2D: per-point density via kNN in the embedded 2D space (+contrast tuning).
#   • 3D: Z = -(n - rank) / (n - 1) ∈ [-1,0], rank-1 at the deepest center.
#   • For large N, skip triangulated surface; draw a light reference cone instead.
#
# Usage
#   python plot_sampling_funnel_memorysafe.py
#   python plot_sampling_funnel_memorysafe.py --case 6czf \
#       --sampling_csv quantum_data/6czf/samples_6czf_all_ibm.csv \
#       --pred_csv predictions/6czf_pred.csv \
#       --embed svd --seed 42 --k_density 20 --topk 20
#
# Requirements
#   pip install numpy pandas matplotlib scikit-learn
#
# Notes
#   - Fonts: default to Arial if available.
#   - No seaborn used; pure matplotlib.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import TruncatedSVD, PCA
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


# ---------- Bit matrix ----------
def build_bit_matrix(bitstrings):
    """
    Build an (N, L) float32 0/1 matrix from bitstrings without huge memory overhead.
    Suitable for very large N when L (qubits) is moderate (e.g., <= 128/256).
    """
    L = max(len(s) for s in bitstrings)
    padded = [s.zfill(L) for s in bitstrings]
    # Using frombuffer is very memory efficient for dense 0/1
    X = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8).reshape(-1, L) - ord("0")
    return X.astype(np.float32)  # (N, L)


# ---------- Embedding (memory-safe) ----------
def embed_union(bitstrings, method="svd", random_state=42):
    """
    Compute 2D embedding on (N, L) bit matrix using SVD/PCA (no NxN matrix → memory-safe).
    """
    X = build_bit_matrix(bitstrings)
    if method.lower() == "pca":
        model = PCA(n_components=2, random_state=random_state)
    else:
        # default: TruncatedSVD (works on dense; for huge L, still faster/stabler than PCA)
        model = TruncatedSVD(n_components=2, random_state=random_state)
    XY = model.fit_transform(X)
    return XY  # (N,2)


def center_and_scale(XY, center_idx=None, margin=0.02):
    XYc = XY.copy()
    if center_idx is not None:
        XYc -= XYc[center_idx]  # translate so best rank is at origin
    m = np.abs(XYc).max()
    s = 1.0 if m < 1e-12 else (1.0 - margin) / m
    XYc *= s
    return XYc


# ---------- Density on 2D (efficient in 2D) ----------
def knn_density_2d(XY, k=20):
    if len(XY) == 0:
        return np.zeros(0, dtype=float)
    k = max(2, min(int(k), len(XY)))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(XY)
    dists, _ = nn.kneighbors(XY)
    mean_d = dists[:, 1:].mean(axis=1)
    dens = 1.0 / (mean_d + 1e-12)
    # normalize to [0,1]
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
    v = np.power(v, gamma)        # gamma < 1 → lift low values
    v = floor + (1.0 - floor) * v # non-zero floor
    return v


# ---------- Plots ----------
def plot_sampling_topview(XY_union_ordered, union_order, sampling_set,
                          case_id, out_dir, k_density=20,
                          floor=0.28, vmax_pct=0.80, gamma=0.75,
                          gridsize=60, point_size=8, alpha_pts=0.95):
    samp_mask = np.array([b in sampling_set for b in union_order], dtype=bool)
    XYs = XY_union_ordered[samp_mask]
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
        fig.savefig(out_dir / f"{case_id}_sampling_topview.{ext}", dpi=300)
    plt.close(fig)


def plot_funnel_3d(XY_union_ordered, df_pred, case_id, out_dir,
                   topk=20, draw_surface=False, cone_alpha=0.28, point_size=10):
    # Predictions are first in union order (see main); use same XY for them
    pred_bits = df_pred["bitstring"].tolist()
    n_pred = len(pred_bits)
    XYp = XY_union_ordered[:n_pred, :]

    ranks = df_pred["rank"].to_numpy()
    n = len(ranks)
    if n == 1:
        Z = np.array([-1.0], dtype=float)
    else:
        Z = - (n - ranks) / (n - 1)  # [-1,0]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Reference cone surface (light)
    Rg = np.linspace(0.0, 1.0, 90)
    Tg = np.linspace(0.0, 2 * np.pi, 180)
    Rm, Tm = np.meshgrid(Rg, Tg)
    Xs = Rm * np.cos(Tm)
    Ys = Rm * np.sin(Tm)
    Zs = -Rm
    ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2,
                    cmap=CMAP_BLUES, alpha=cone_alpha, edgecolor="none")

    # Scatter predictions colored by depth
    norm = Normalize(vmin=Z.min(), vmax=Z.max())
    colors = plt.cm.get_cmap(CMAP_BLUES)(norm(Z))
    ax.scatter(XYp[:, 0], XYp[:, 1], Z, s=point_size, c=colors,
               depthshade=False, edgecolors="k", linewidths=0.2)

    # Highlight rank-1 at center
    i_best = int(np.argmin(ranks))
    ax.scatter([0.0], [0.0], [Z[i_best]], s=160, c=CENTER_COLOR,
               edgecolors="k", linewidths=0.8, depthshade=False, label="Rank-1")

    # Annotate top-k
    kk = min(topk, n)
    order = np.argsort(ranks)[:kk]
    for pos, i in enumerate(order, start=1):
        ax.text(XYp[i, 0], XYp[i, 1], Z[i], f"{pos}", fontsize=10,
                ha="center", va="bottom", color="k")

    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(min(-1.2, Z.min() - 0.05), 0.15)
    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{case_id}: Re-ranked energy funnel (shared XY)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Normalized depth")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_funnel_3d.{ext}", dpi=300)
    plt.close(fig)

    # Export mapping
    out = df_pred.copy()
    out["x"] = XYp[:, 0]; out["y"] = XYp[:, 1]; out["z_norm_rank"] = Z
    out.to_csv(out_dir / f"{case_id}_funnel_mapping.csv", index=False)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Memory-safe sampling top view and rank-based 3D funnel (shared XY).")
    parser.add_argument("--case", type=str, default="6czf", help="Case ID")
    parser.add_argument("--sampling_csv", type=Path, default=None, help="Path to sampling CSV")
    parser.add_argument("--pred_csv", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--embed", type=str, default="svd", choices=["svd", "pca"],
                        help="Embedding backend for 0/1 bit matrix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for embedding")
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
    samp_set = set(samp_bits)

    # Build union list with predictions FIRST (so XY[:n_pred] matches predictions)
    union_bits = pred_bits + [b for b in samp_bits if b not in set(pred_bits)]

    # Embed (memory-safe)
    XY = embed_union(union_bits, method=args.embed, random_state=args.seed)

    # Center at rank-1, then scale
    best_bit = df_pred.iloc[0]["bitstring"]
    best_idx = union_bits.index(best_bit)
    XY_cs = center_and_scale(XY, center_idx=best_idx, margin=0.02)

    # Save mapping (union)
    map_df = pd.DataFrame({"bitstring": union_bits, "x": XY_cs[:, 0], "y": XY_cs[:, 1]})
    out_dir.mkdir(parents=True, exist_ok=True)
    map_df.to_csv(out_dir / f"{case_id}_union_xy_mapping.csv", index=False)

    # Plots
    plot_sampling_topview(
        XY_union_ordered=XY_cs,
        union_order=union_bits,
        sampling_set=samp_set,
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
        XY_union_ordered=XY_cs,
        df_pred=df_pred,
        case_id=case_id,
        out_dir=out_dir,
        topk=args.topk,
        draw_surface=False,        # keep False for very large N (triangulation would be heavy)
        cone_alpha=0.28,
        point_size=10,
    )

    print("[Done]", out_dir.resolve())


if __name__ == "__main__":
    main()
