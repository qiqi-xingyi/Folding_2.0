# -*- coding: utf-8 -*-
# @time: 2025/11/06
# @file: plot_sampling_funnel_linked_v2.py
#
# Description:
#   Linked figures for a single case:
#     (1) Square 2D density scatter of sampling points (blue scale, darker overall).
#     (2) Rank-driven 3D funnel where rank-1 is at the center and lowest.
#         Only ~70% of points respect the rank order in depth; the rest are
#         intentionally disordered via controlled random perturbations.
#
# Usage:
#   python plot_sampling_funnel_linked_v2.py
#   python plot_sampling_funnel_linked_v2.py --case 6czf \
#       --sampling_csv quantum_data/6czf/samples_6czf_all_ibm.csv \
#       --pred_csv predictions/6czf_pred.csv \
#       --k_density 20 --jitter_frac 0.03 --disorder_frac 0.30 --seed 42
#
# Requirements:
#   pip install numpy pandas matplotlib scikit-learn

import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- Global style ----------------
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

BLUES_CMAP_NAME = "Blues"
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


# ---------------- Layout (shared by 2D & 3D) ----------------
def golden_spiral_xy(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    idx = np.arange(n, dtype=float)          # 0..n-1; rank-1 at index 0
    r = np.where(n > 1, idx / (n - 1), 0.0)  # 0..1
    phi = (3.0 - np.sqrt(5.0)) * np.pi
    theta = idx * phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    scale = 0.98  # cover square tightly but leave a small margin
    return np.column_stack([x * scale, y * scale])


def assign_union_layout(pred_bits, samp_bits):
    pred_bits = list(pred_bits)
    samp_bits = list(samp_bits)

    pred_set = set(pred_bits)
    samp_only = [b for b in samp_bits if b not in pred_set]

    def _key_sha1(s):
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    samp_only_sorted = sorted(samp_only, key=_key_sha1)
    order_union = pred_bits + samp_only_sorted
    n_union = len(order_union)

    xy_union = golden_spiral_xy(n_union)
    pred_mask_union = np.array([b in pred_set for b in order_union], dtype=bool)
    return xy_union, order_union, pred_mask_union


# ---------------- Density ----------------
def local_density_knn(xy, k=20):
    if len(xy) == 0:
        return np.zeros(0, dtype=float)
    k = max(2, min(int(k), len(xy)))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xy)
    distances, _ = nn.kneighbors(xy)
    d = distances[:, 1:].mean(axis=1)
    dens = 1.0 / (d + 1e-12)
    dmin, dmax = dens.min(), dens.max()
    if dmax > dmin:
        dens = (dens - dmin) / (dmax - dmin)
    else:
        dens = np.zeros_like(dens)
    return dens


# ---------------- Disorder control ----------------
def apply_depth_disorder(z_base: np.ndarray, ranks: np.ndarray, rng: np.random.Generator,
                         disorder_frac: float, jitter_frac: float) -> np.ndarray:
    """
    Start from monotonic depths z_base in [-1, 0] (rank-1 lowest).
    1) Add small jitter around the bottom (stronger for best ~2% ranks).
    2) Select a fraction of indices (excluding rank-1) and reshuffle their depths
       among themselves to break global order.
    3) Keep rank-1 strictly the lowest.
    """
    n = len(z_base)
    z = z_base.copy()

    # Step 1: bottom jitter
    if n > 1:
        k_low = max(5, int(0.02 * n))
        order_idx = np.argsort(ranks)  # ascending (best first)
        for pos in range(n):
            i = order_idx[pos]
            frac = 1.0 if pos < k_low else 0.3
            z[i] += rng.normal(loc=0.0, scale=jitter_frac * frac)

    # Step 2: disorder by partial reshuffle
    if n > 2 and disorder_frac > 0:
        m = max(1, int(disorder_frac * (n - 1)))  # exclude rank-1
        # exclude the best (rank==1)
        best_idx = int(np.argmin(ranks))
        candidates = np.setdiff1d(np.arange(n), np.array([best_idx]))
        picked = rng.choice(candidates, size=m, replace=False)
        # random permutation of their depths
        z[picked] = z[picked][rng.permutation(m)]

    # Step 3: enforce rank-1 as strict minimum
    best = int(np.argmin(ranks))
    z[best] = min(z.min(), -1.05)

    return z


# ---------------- Plots ----------------
def plot_sampling_density_2d(xy_union, order_union, sampling_set, case_id, out_dir,
                             k_density=20, density_gamma=0.7, alpha=0.95, size=14):
    samp_idx = np.array([b in sampling_set for b in order_union], dtype=bool)
    xy_samp = xy_union[samp_idx]

    dens = local_density_knn(xy_samp, k=k_density) if len(xy_samp) >= 3 else np.zeros(len(xy_samp))
    # Darker overall: gamma (<1) pushes values upward
    dens_gamma = np.power(dens, density_gamma)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        xy_samp[:, 0], xy_samp[:, 1],
        c=dens_gamma, cmap=BLUES_CMAP_NAME, s=size, alpha=alpha, edgecolors="none"
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Local density")

    ax.set_title(f"{case_id}: Sampling 2D density (linked canvas)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_sampling_density_2d_linked_v2.{ext}", dpi=300)
    plt.close(fig)


def plot_rank_funnel_3d(xy_union, df_pred_sorted, case_id, out_dir,
                        disorder_frac=0.30, jitter_frac=0.03, seed=42, topk_label=20):
    rng = np.random.default_rng(seed)

    n_pred = len(df_pred_sorted)
    xy_pred = xy_union[:n_pred]
    ranks = df_pred_sorted["rank"].to_numpy()

    if n_pred == 1:
        z_base = np.array([-1.0], dtype=float)
    else:
        z_base = - (n_pred - ranks) / (n_pred - 1)  # [-1,0]

    # Apply disorder: target ~70% order consistency
    z = apply_depth_disorder(z_base, ranks, rng,
                             disorder_frac=disorder_frac,
                             jitter_frac=jitter_frac)

    # Reference funnel surface (cone)
    Rg = np.linspace(0.0, 1.0, 90)
    Tg = np.linspace(0.0, 2 * np.pi, 180)
    Rm, Tm = np.meshgrid(Rg, Tg)
    Xs = Rm * np.cos(Tm)
    Ys = Rm * np.sin(Tm)
    Zs = -Rm

    norm = Normalize(vmin=z.min(), vmax=z.max())
    colors = plt.cm.get_cmap(BLUES_CMAP_NAME)(norm(z))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2,
                    cmap=BLUES_CMAP_NAME, alpha=0.28, edgecolor="none")

    ax.scatter(xy_pred[:, 0], xy_pred[:, 1], z,
               s=14, c=colors, depthshade=False, edgecolors="k", linewidths=0.2)

    # Rank-1 at center and lowest
    i_best = int(np.argmin(ranks))
    ax.scatter([0.0], [0.0], [z[i_best]],
               s=160, c=CENTER_COLOR, edgecolors="k", linewidths=0.8,
               depthshade=False, label="Rank-1")

    # Optional labels
    topk = min(topk_label, n_pred)
    order = np.argsort(ranks)[:topk]
    for pos, i in enumerate(order, start=1):
        xi, yi, zi = xy_pred[i, 0], xy_pred[i, 1], z[i]
        ax.text(xi, yi, zi, f"{pos}", fontsize=10, ha="center", va="bottom", color="k")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(min(-1.2, z.min() - 0.05), 0.15)
    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{case_id}: Rank-driven 3D funnel (70% order consistency)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Normalized depth")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_rank_funnel_3d_linked_v2.{ext}", dpi=300)
    plt.close(fig)

    # Export layout
    out = df_pred_sorted.copy()
    out["x"] = xy_pred[:, 0]
    out["y"] = xy_pred[:, 1]
    out["norm_depth"] = z
    out.to_csv(out_dir / f"{case_id}_rank_funnel_layout_linked_v2.csv", index=False)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Linked 2D density and 3D rank funnel with partial order (≈70%).")
    parser.add_argument("--case", type=str, default="6czf", help="Case ID")
    parser.add_argument("--sampling_csv", type=Path, default=None, help="Path to sampling CSV")
    parser.add_argument("--pred_csv", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--k_density", type=int, default=20, help="k for KNN local density")
    parser.add_argument("--density_gamma", type=float, default=0.7, help="Gamma (<1 darker) for 2D density colors")
    parser.add_argument("--alpha2d", type=float, default=0.95, help="Alpha for 2D points")
    parser.add_argument("--size2d", type=float, default=14, help="Marker size for 2D points")
    parser.add_argument("--disorder_frac", type=float, default=0.30, help="Fraction of prediction points to disorder")
    parser.add_argument("--jitter_frac", type=float, default=0.03, help="Z jitter scale near the bottom")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--topk", type=int, default=20, help="Top-K labels to annotate in 3D")
    args = parser.parse_args()

    case_id = args.case
    sampling_csv = args.sampling_csv or Path(f"quantum_data/{case_id}/samples_{case_id}_all_ibm.csv")
    pred_csv = args.pred_csv or Path(f"predictions/{case_id}_pred.csv")

    if not sampling_csv.exists():
        raise FileNotFoundError(f"Sampling CSV not found: {sampling_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    out_dir = OUT_DIR_BASE / case_id

    # Read data
    df_samp = read_sampling_csv(sampling_csv)
    df_pred = read_prediction_csv(pred_csv)  # sorted by rank asc
    pred_bits = df_pred["bitstring"].tolist()
    samp_bits = df_samp["bitstring"].tolist()
    samp_set = set(samp_bits)

    # Union layout (shared XY for both figures)
    xy_union, order_union, pred_mask_union = assign_union_layout(pred_bits, samp_bits)

    # 2D density (darker overall via gamma)
    plot_sampling_density_2d(
        xy_union=xy_union,
        order_union=order_union,
        sampling_set=samp_set,
        case_id=case_id,
        out_dir=out_dir,
        k_density=args.k_density,
        density_gamma=args.density_gamma,
        alpha=args.alpha2d,
        size=args.size2d,
    )

    # 3D funnel (≈70% order consistency)
    plot_rank_funnel_3d(
        xy_union=xy_union,
        df_pred_sorted=df_pred,
        case_id=case_id,
        out_dir=out_dir,
        disorder_frac=args.disorder_frac,
        jitter_frac=args.jitter_frac,
        seed=args.seed,
        topk_label=args.topk,
    )

    print("[Done]", out_dir.resolve())


if __name__ == "__main__":
    main()
