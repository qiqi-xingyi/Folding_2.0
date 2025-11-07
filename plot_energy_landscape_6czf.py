# -*- coding: utf-8 -*-
# @file: plot_sampling_funnel_truthful.py
#
# Goal:
#   Show the sampling distribution faithfully (no synthetic layouts) and the
#   re-ranked funnel without breaking the XY structure.
#
# Strategy (distortion-free):
#   1) Build ONE embedding for the union of {sampling bitstrings} ∪ {prediction bitstrings}
#      using metric-preserving MDS on Hamming distances (fallback: PCA on bits).
#   2) Translate the whole canvas so that the best-ranked prediction lies at the origin.
#      This recenters the view but preserves all pairwise XY distances.
#   3) Isotropic scale to fit a square [-1, 1] × [-1, 1] without changing aspect.
#   4) 2D figure = exact top view of this canvas:
#        - color each sampling point by kNN density
#        - add a light hexbin backdrop to reveal global density without clumping everything at center
#        - tuned contrast (percentile clip + gamma + color floor) so low densities are still visible
#   5) 3D figure uses the SAME (X, Y) for prediction points and sets Z by normalized rank:
#        z = - (n - rank) / (n - 1)  in [-1, 0]; rank-1 is lowest.
#      Draw a triangulated surface on prediction points plus the point cloud.
#
# Usage:
#   python plot_sampling_funnel_truthful.py
#   python plot_sampling_funnel_truthful.py --case 6czf \
#       --sampling_csv quantum_data/6czf/samples_6czf_all_ibm.csv \
#       --pred_csv predictions/6czf_pred.csv \
#       --k_density 20 --mds --seed 42 --topk 20
#
# Requirements:
#   pip install numpy pandas matplotlib scikit-learn

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

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

BLUES = "Blues"
CENTER_COLOR = "#1f77b4"
OUT_DIR_BASE = Path("result_summary/landscape")


# ---------------- I/O ----------------
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


# ---------------- Bit utilities ----------------
def pad_bitstrings(bitstrings):
    L = max(len(s) for s in bitstrings)
    return [s.zfill(L) for s in bitstrings], L


def bitstrings_to_matrix(bitstrings, max_len=None):
    if max_len is None:
        _, max_len = pad_bitstrings(bitstrings)
    padded = [s.zfill(max_len) for s in bitstrings]
    arr = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8)
    arr = arr.reshape(-1, max_len)
    arr = arr - ord("0")
    return arr.astype(np.float32)


def hamming_distance_matrix(arr):
    # arr: (N, L) with 0/1 values
    N = arr.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    # Chunk to avoid large memory spikes
    chunk = max(1, 2000 // max(1, arr.shape[1] // 64 + 1))
    for i in range(0, N, chunk):
        j = min(N, i + chunk)
        blk = arr[i:j]                           # (b, L)
        diff = np.abs(blk[:, None, :] - arr[None, :, :])  # (b, N, L)
        D[i:j] = diff.sum(axis=2)
    return D


def union_embed(bitstrings, use_mds=True, random_state=42):
    padded, L = pad_bitstrings(bitstrings)
    X = bitstrings_to_matrix(padded, L)
    if use_mds:
        D = hamming_distance_matrix(X)
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=random_state, n_init=4, max_iter=400)
        XY = mds.fit_transform(D)
    else:
        pca = PCA(n_components=2, random_state=random_state)
        XY = pca.fit_transform(X)
    return XY


def center_and_scale(XY, center_idx=None, margin=0.02):
    XYc = XY.copy()
    if center_idx is not None:
        XYc -= XYc[center_idx]  # translate so chosen point is at origin
    # isotropic scale to fit [-1,1] with margin
    mx = np.abs(XYc).max()
    if mx < 1e-9:
        s = 1.0
    else:
        s = (1.0 - margin) / mx
    XYc *= s
    return XYc


# ---------------- Density & contrast ----------------
def knn_density(xy, k=20):
    if len(xy) == 0:
        return np.zeros(0, dtype=float)
    k = max(2, min(int(k), len(xy)))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xy)
    dists, _ = nn.kneighbors(xy)
    d = dists[:, 1:].mean(axis=1)
    dens = 1.0 / (d + 1e-12)
    # normalize to [0,1]
    dmin, dmax = dens.min(), dens.max()
    if dmax > dmin:
        dens = (dens - dmin) / (dmax - dmin)
    else:
        dens = np.zeros_like(dens)
    return dens


def enhance_contrast(values, floor=0.25, vmax_pct=0.85, gamma=0.8):
    if len(values) == 0:
        return values
    v = values.astype(float)
    vmax = np.quantile(v, vmax_pct)
    if vmax <= 1e-12:
        vmax = 1.0
    v = np.clip(v / vmax, 0.0, 1.0)
    v = np.power(v, gamma)
    v = floor + (1.0 - floor) * v
    return v


# ---------------- Plots ----------------
def plot_sampling_2d(xy_union, order_union, sampling_set, case_id, out_dir,
                     k_density=20, floor=0.25, vmax_pct=0.85, gamma=0.8):
    samp_mask = np.array([b in sampling_set for b in order_union], dtype=bool)
    XYs = xy_union[samp_mask]
    dens = knn_density(XYs, k=k_density) if len(XYs) >= 3 else np.zeros(len(XYs))
    dens_vis = enhance_contrast(dens, floor=floor, vmax_pct=vmax_pct, gamma=gamma)

    fig, ax = plt.subplots(figsize=(8, 8))
    # light hexbin background to show region coverage without overpowering points
    if len(XYs) > 0:
        hb = ax.hexbin(XYs[:, 0], XYs[:, 1], gridsize=50, mincnt=1,
                       cmap=BLUES, alpha=0.25, linewidths=0.0)
    sc = ax.scatter(XYs[:, 0], XYs[:, 1], c=dens_vis, cmap=BLUES,
                    s=10, alpha=0.95, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Local density (enhanced)")

    ax.set_title(f"{case_id}: Sampling distribution (true top view)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_sampling_2d_truthful.{ext}", dpi=300)
    plt.close(fig)


def plot_funnel_3d(xy_union, order_union, df_pred, case_id, out_dir, topk=20):
    # prediction points occupy the first n_pred indices in the union order
    pred_bits = df_pred["bitstring"].tolist()
    n_pred = len(pred_bits)
    XYp = xy_union[:n_pred, :]

    rks = df_pred["rank"].to_numpy()
    n = len(rks)
    if n == 1:
        Z = np.array([-1.0], dtype=float)
    else:
        Z = - (n - rks) / (n - 1)  # [-1,0], rank-1 lowest

    # triangulated surface on prediction points
    triang = None
    if len(XYp) >= 3:
        triang = mtri.Triangulation(XYp[:, 0], XYp[:, 1])

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    if triang is not None:
        trisurf = ax.plot_trisurf(triang, Z, cmap=BLUES, linewidth=0.2,
                                  antialiased=True, alpha=0.75)
        cb = fig.colorbar(trisurf, ax=ax, shrink=0.6, aspect=12, pad=0.08)
        cb.set_label("Normalized depth (by rank)")
    # points
    norm = Normalize(vmin=Z.min(), vmax=Z.max())
    colors = plt.cm.get_cmap(BLUES)(norm(Z))
    ax.scatter(XYp[:, 0], XYp[:, 1], Z, s=12, c=colors,
               depthshade=False, edgecolors="k", linewidths=0.2)

    # highlight rank-1 at origin
    i_best = int(np.argmin(rks))
    ax.scatter([0.0], [0.0], [Z[i_best]], s=160, c=CENTER_COLOR,
               edgecolors="k", linewidths=0.8, depthshade=False, label="Rank-1")

    # annotate top-k
    kk = min(topk, n)
    order = np.argsort(rks)[:kk]
    for pos, i in enumerate(order, start=1):
        ax.text(XYp[i, 0], XYp[i, 1], Z[i], f"{pos}", fontsize=10,
                ha="center", va="bottom", color="k")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(min(-1.2, Z.min() - 0.05), 0.15)
    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{case_id}: Re-ranked energy funnel (true top view)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Normalized depth")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_funnel_3d_truthful.{ext}", dpi=300)
    plt.close(fig)

    # export mapping for traceability
    out = df_pred.copy()
    out["x"] = XYp[:, 0]
    out["y"] = XYp[:, 1]
    out["z_norm_rank"] = Z
    out.to_csv(out_dir / f"{case_id}_funnel_mapping_truthful.csv", index=False)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Distortion-free sampling map and re-ranked funnel (shared XY).")
    parser.add_argument("--case", type=str, default="6czf", help="Case ID")
    parser.add_argument("--sampling_csv", type=Path, default=None, help="Path to sampling CSV")
    parser.add_argument("--pred_csv", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--k_density", type=int, default=20, help="k for KNN density")
    parser.add_argument("--mds", action="store_true", help="Use MDS(Hamming) embedding (default).")
    parser.add_argument("--pca", action="store_true", help="Force PCA embedding instead of MDS.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for MDS/PCA")
    parser.add_argument("--topk", type=int, default=20, help="Top-K labels in 3D")
    # density contrast
    parser.add_argument("--density_floor", type=float, default=0.28, help="Color floor for low densities")
    parser.add_argument("--density_vmax_pct", type=float, default=0.80, help="Upper percentile clipping")
    parser.add_argument("--density_gamma", type=float, default=0.75, help="Gamma (<1 darkens lows)")
    args = parser.parse_args()

    case_id = args.case
    sampling_csv = args.sampling_csv or Path(f"quantum_data/{case_id}/samples_{case_id}_all_ibm.csv")
    pred_csv = args.pred_csv or Path(f"predictions/{case_id}_pred.csv")

    if not sampling_csv.exists():
        raise FileNotFoundError(f"Sampling CSV not found: {sampling_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    out_dir = OUT_DIR_BASE / case_id

    df_samp = read_sampling_csv(sampling_csv)
    df_pred = read_prediction_csv(pred_csv)  # rank asc

    # union set for one shared embedding
    union_bits = pd.Index(df_samp["bitstring"]).union(df_pred["bitstring"]).tolist()
    use_mds = not args.pca  # default True unless --pca
    XY = union_embed(union_bits, use_mds=use_mds, random_state=args.seed)

    # center at the best-ranked prediction without altering pairwise distances
    best_bit = df_pred.iloc[0]["bitstring"]
    best_idx = union_bits.index(best_bit)
    XY_cs = center_and_scale(XY, center_idx=best_idx, margin=0.02)

    # build union order and masks: predictions first to match mapping in 3D
    pred_bits = df_pred["bitstring"].tolist()
    samp_bits = df_samp["bitstring"].tolist()
    # order union as: predictions (in rank order) + remaining sampling (stable)
    samp_only = [b for b in samp_bits if b not in set(pred_bits)]
    union_order = pred_bits + samp_only
    # remap XY_cs to this order
    idx_map = {b: i for i, b in enumerate(union_bits)}
    XY_ordered = np.vstack([XY_cs[idx_map[b]] for b in union_order])

    # 2D sampling plot (top view)
    plot_sampling_2d(
        xy_union=XY_ordered,
        order_union=union_order,
        sampling_set=set(samp_bits),
        case_id=case_id,
        out_dir=out_dir,
        k_density=args.k_density,
        floor=args.density_floor,
        vmax_pct=args.density_vmax_pct,
        gamma=args.density_gamma,
    )

    # 3D funnel with the SAME XY for predictions
    plot_funnel_3d(
        xy_union=XY_ordered,
        order_union=union_order,
        df_pred=df_pred,
        case_id=case_id,
        out_dir=out_dir,
        topk=args.topk,
    )

    print("[Done]", out_dir.resolve())


if __name__ == "__main__":
    main()
