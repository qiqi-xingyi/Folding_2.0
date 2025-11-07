# -*- coding: utf-8 -*-
# @time: 2025/11/06
# @file: plot_sampling_funnel_linked.py
#
# Description:
#   Produce two linked figures for a single case:
#     (1) Square 2D density scatter of quantum sampling points (blue scale).
#         Layout covers the full square canvas and is IDENTICAL to the top-down
#         projection used in the 3D plot.
#     (2) Rank-driven 3D funnel where rank-1 is at the very bottom and at the center.
#         Depth is normalized purely by rank; near-bottom points receive slight random
#         perturbations in Z (rank-1 remains lowest and at the center).
#   The 2D plot is a strict top view of the 3D layout (same XY coordinates & limits).
#
# Usage:
#   python plot_sampling_funnel_linked.py
#   python plot_sampling_funnel_linked.py --case 6czf \
#       --sampling_csv quantum_data/6czf/samples_6czf_all_ibm.csv \
#       --pred_csv predictions/6czf_pred.csv \
#       --k_density 20 --jitter_frac 0.03 --seed 42
#
# Requirements:
#   pip install numpy pandas matplotlib

import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import Normalize
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

BLUES_CMAP = "Blues"
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
        # fallback: arbitrary ranking
        df = df.sort_values("bitstring").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    else:
        df = df.sort_values("rank", ascending=True).reset_index(drop=True)
    return df[["bitstring", "rank"]].copy()


# ---------------- Layout (shared by 2D & 3D) ----------------
def golden_spiral_xy(n: int) -> np.ndarray:
    """Vogel spiral in [-1, 1] square footprint, center at (0,0)."""
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    # Indices 0..n-1, rank-1 at index 0
    idx = np.arange(n, dtype=float)
    # Radius 0..1
    r = np.where(n > 1, idx / (n - 1), 0.0)
    phi = (3.0 - np.sqrt(5.0)) * np.pi  # ~137.5 deg
    theta = idx * phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Already within unit circle; to cover the square nicely, scale to ~0.98
    scale = 0.98
    return np.column_stack([x * scale, y * scale])


def assign_union_layout(pred_bits, samp_bits):
    """
    Build a single XY layout for the union of prediction and sampling bitstrings:
      - Predictions occupy the first N_pred spiral slots in rank order.
      - Sampling-only bitstrings are appended after predictions, in a stable order
        defined by SHA1 hash of bitstring (to make the mapping deterministic).
    Returns:
      XY_union: array of shape (N_union, 2)
      order_union: list of bitstrings (same order as XY_union rows)
      pred_mask_union: boolean mask of length N_union (True if in predictions)
    """
    pred_bits = list(pred_bits)
    samp_bits = list(samp_bits)

    pred_set = set(pred_bits)
    samp_only = [b for b in samp_bits if b not in pred_set]

    # Deterministic order for sampling-only via SHA1
    def _key_sha1(s):
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    samp_only_sorted = sorted(samp_only, key=_key_sha1)
    order_union = pred_bits + samp_only_sorted
    n_union = len(order_union)

    XY_union = golden_spiral_xy(n_union)
    pred_mask_union = np.array([b in pred_set for b in order_union], dtype=bool)
    return XY_union, order_union, pred_mask_union


# ---------------- Density ----------------
def local_density_knn(xy, k=20):
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


# ---------------- Plots ----------------
def plot_sampling_density_2d(xy_union, order_union, pred_mask_union,
                             sampling_set, case_id, out_dir, k_density=20):
    # Extract XY for sampling points (in union layout)
    samp_idx = np.array([b in sampling_set for b in order_union], dtype=bool)
    xy_samp = xy_union[samp_idx]
    if xy_samp.size == 0:
        # still produce an empty canvas for consistency
        xy_samp = np.zeros((0, 2))

    # Density on sampling points only (so the color truly reflects sampling density)
    dens = local_density_knn(xy_samp, k=k_density) if len(xy_samp) >= 3 else np.zeros(len(xy_samp))

    # Square figure, equal aspect, fixed limits identical to 3D top view
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        xy_samp[:, 0], xy_samp[:, 1],
        c=dens, cmap=BLUES_CMAP, s=10, alpha=0.9, edgecolors="none"
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Local density")

    ax.set_title(f"{case_id}: Sampling 2D density (top-view canvas)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_sampling_density_2d_linked.{ext}", dpi=300)
    plt.close(fig)


def plot_rank_funnel_3d(xy_union, order_union, pred_mask_union,
                        df_pred_sorted, case_id, out_dir,
                        jitter_frac=0.03, seed=42, topk_label=20):
    rng = np.random.default_rng(seed)

    # XY for prediction points are the FIRST N_pred slots in the union layout
    n_pred = len(df_pred_sorted)
    xy_pred = xy_union[:n_pred]

    # Depth purely by rank: rank-1 -> z = -1, rank-n -> z -> 0
    ranks = df_pred_sorted["rank"].to_numpy()
    n = len(ranks)
    if n == 1:
        z = np.array([-1.0], dtype=float)
    else:
        # z_i = - (n - rank_i) / (n - 1)  in [-1, 0]
        z = - (n - ranks) / (n - 1)

    # Small random perturbations near the bottom while keeping rank-1 the lowest
    # Apply stronger jitter to the best K_low ranks; taper for others.
    k_low = max(5, int(0.02 * n))
    jitter_scale = jitter_frac  # relative to the z-range (~1.0)
    z_j = z.copy()
    if n > 1:
        # ranks are 1..n; convert to 0..n-1 index
        order_idx = np.argsort(ranks)
        for pos in range(n):
            i = order_idx[pos]
            # stronger jitter for top ranks, weaker for others
            frac = 1.0 if pos < k_low else 0.3
            z_j[i] += rng.normal(loc=0.0, scale=jitter_scale * frac)
        # ensure monotonic min at rank-1
        i_best = np.argmin(ranks)
        z_j[i_best] = min(z_j.min(), -1.05)  # keep it strictly the lowest

    # Build a smooth reference funnel surface (cone-like)
    Rg = np.linspace(0.0, 1.0, 90)
    Tg = np.linspace(0.0, 2 * np.pi, 180)
    Rm, Tm = np.meshgrid(Rg, Tg)
    Xs = Rm * np.cos(Tm)
    Ys = Rm * np.sin(Tm)
    Zs = -Rm  # deepest at center

    # Colors for points by normalized depth
    norm = Normalize(vmin=z_j.min(), vmax=z_j.max())
    colors = plt.cm.get_cmap(BLUES_CMAP)(norm(z_j))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Surface
    ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2,
                    cmap=BLUES_CMAP, alpha=0.30, edgecolor="none")

    # Points
    ax.scatter(xy_pred[:, 0], xy_pred[:, 1], z_j,
               s=14, c=colors, depthshade=False, edgecolors="k", linewidths=0.2)

    # Highlight rank-1 at (0,0)
    ax.scatter([0.0], [0.0], [z_j[ranks.argmin()]],
               s=160, c=CENTER_COLOR, edgecolors="k", linewidths=0.8,
               depthshade=False, label="Rank-1")

    # Optional top-k labels
    topk = min(topk_label, n)
    order = np.argsort(ranks)[:topk]
    for pos, i in enumerate(order, start=1):
        xi, yi, zi = xy_pred[i, 0], xy_pred[i, 1], z_j[i]
        ax.text(xi, yi, zi, f"{pos}", fontsize=10, ha="center", va="bottom", color="k")

    # Fixed square footprint and view
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(min(-1.2, z_j.min() - 0.05), 0.15)
    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{case_id}: Rank-driven 3D funnel (linked top view)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Normalized depth")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_rank_funnel_3d_linked.{ext}", dpi=300)
    plt.close(fig)

    # Export layout for traceability
    out = df_pred_sorted.copy()
    out["x"] = xy_pred[:, 0]
    out["y"] = xy_pred[:, 1]
    out["norm_depth"] = z_j
    out.to_csv(out_dir / f"{case_id}_rank_funnel_layout_linked.csv", index=False)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Linked 2D density and 3D rank funnel (same XY canvas).")
    parser.add_argument("--case", type=str, default="6czf", help="Case ID")
    parser.add_argument("--sampling_csv", type=Path, default=None, help="Path to sampling CSV")
    parser.add_argument("--pred_csv", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--k_density", type=int, default=20, help="k for KNN local density")
    parser.add_argument("--jitter_frac", type=float, default=0.03, help="Z jitter scale near the bottom")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for jitter")
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
    df_pred = read_prediction_csv(pred_csv)         # sorted by rank asc
    pred_bits = df_pred["bitstring"].tolist()
    samp_bits = df_samp["bitstring"].tolist()
    samp_set = set(samp_bits)

    # Build union layout and masks
    xy_union, order_union, pred_mask_union = assign_union_layout(pred_bits, samp_bits)

    # 2D density (strict top-view of the 3D canvas)
    plot_sampling_density_2d(
        xy_union=xy_union,
        order_union=order_union,
        pred_mask_union=pred_mask_union,
        sampling_set=samp_set,
        case_id=case_id,
        out_dir=out_dir,
        k_density=args.k_density,
    )

    # 3D funnel (uses the first N_pred slots for predictions; rank-1 at center & lowest)
    plot_rank_funnel_3d(
        xy_union=xy_union,
        order_union=order_union,
        pred_mask_union=pred_mask_union,
        df_pred_sorted=df_pred,
        case_id=case_id,
        out_dir=out_dir,
        jitter_frac=args.jitter_frac,
        seed=args.seed,
        topk_label=args.topk,
    )

    print("[Done]", out_dir.resolve())


if __name__ == "__main__":
    main()
