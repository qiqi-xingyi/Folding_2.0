# -*- coding: utf-8 -*-
# @time: 2025/11/06
# @file: plot_sampling_funnel_linked_v3.py
#
# Description:
#   Linked figures with stricter controls:
#     (1) Square 2D density scatter (blue scale), darker overall:
#         - Contrast stretch with upper percentile clipping
#         - Non-zero color floor so low densities are still visible
#     (2) Rank-driven 3D funnel:
#         - Rank-1 fixed at the center and strictly the lowest
#         - Only the bottom 10% and top 10% of ranks are perturbed
#         - Middle 70% remain unchanged (order-preserving)
#   The 2D plot is the exact top view (same XY) of the 3D layout.
#
# Usage:
#   python plot_sampling_funnel_linked_v3.py
#   python plot_sampling_funnel_linked_v3.py --case 6czf \
#       --sampling_csv quantum_data/6czf/samples_6czf_all_ibm.csv \
#       --pred_csv predictions/6czf_pred.csv \
#       --k_density 20 --seed 42 \
#       --density_floor 0.25 --density_vmax_pct 0.85 --density_gamma 0.8 \
#       --jitter_frac 0.03 --extreme_frac 0.10 --topk 20
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
        df = df.sort_values("bitstring").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    else:
        df = df.sort_values("rank", ascending=True).reset_index(drop=True)
    return df[["bitstring", "rank"]].copy()


# ---------------- Layout (shared by 2D & 3D) ----------------
def golden_spiral_xy(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    idx = np.arange(n, dtype=float)                # 0..n-1; rank-1 at index 0
    r = np.where(n > 1, idx / (n - 1), 0.0)        # 0..1
    phi = (3.0 - np.sqrt(5.0)) * np.pi
    theta = idx * phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    scale = 0.98                                    # fill the square tightly
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
    xy_union = golden_spiral_xy(len(order_union))
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
    # Normalize to [0,1]
    dmin, dmax = dens.min(), dens.max()
    if dmax > dmin:
        dens = (dens - dmin) / (dmax - dmin)
    else:
        dens = np.zeros_like(dens)
    return dens


def stretch_density_for_visual(dens, floor=0.25, vmax_pct=0.85, gamma=0.8):
    """
    Make low densities darker and reduce the upper cap.
    1) Clip the top end by percentile (vmax_pct).
    2) Normalize to [0,1].
    3) Gamma (<1 darkens low values).
    4) Apply a non-zero color floor.
    """
    if len(dens) == 0:
        return dens
    dens = dens.astype(float)
    # clip upper
    p_hi = np.quantile(dens, vmax_pct)
    if p_hi <= 1e-12:
        p_hi = 1.0
    dens_cs = np.clip(dens / p_hi, 0.0, 1.0)
    # gamma
    dens_gamma = np.power(dens_cs, gamma)
    # floor
    dens_vis = floor + (1.0 - floor) * dens_gamma
    return dens_vis


# ---------------- Depth perturbation (extremes only) ----------------
def build_monotonic_depth_from_ranks(ranks: np.ndarray) -> np.ndarray:
    n = len(ranks)
    if n == 1:
        return np.array([-1.0], dtype=float)
    return - (n - ranks) / (n - 1)  # [-1, 0], rank-1 lowest


def perturb_extremes_only(z_base: np.ndarray, ranks: np.ndarray, rng: np.random.Generator,
                          extreme_frac: float = 0.10, jitter_frac: float = 0.03) -> np.ndarray:
    """
    Perturb only the bottom 10% and top 10% (by default).
    - Middle ~70% stay unchanged.
    - Rank-1 remains strictly the lowest and at center.
    - Inside the extreme sets: add jitter and a small in-set shuffle to break strict order.
    """
    n = len(z_base)
    z = z_base.copy()
    if n <= 2:
        # keep best strictly lowest
        if n == 2:
            best = int(np.argmin(ranks))
            z[best] = min(z.min(), -1.05)
        return z

    order = np.argsort(ranks)  # ascending
    best = int(order[0])

    k_ext = max(1, int(np.ceil(extreme_frac * n)))
    bottom_idx = order[:k_ext]
    top_idx = order[-k_ext:]

    # Jitter extremes
    z[bottom_idx] += rng.normal(0.0, jitter_frac * 1.2, size=len(bottom_idx))
    z[top_idx]    += rng.normal(0.0, jitter_frac * 0.8, size=len(top_idx))

    # In-set shuffle to break order (but keep identity of sets)
    if len(bottom_idx) > 1:
        z[bottom_idx] = z[bottom_idx][rng.permutation(len(bottom_idx))]
    if len(top_idx) > 1:
        z[top_idx] = z[top_idx][rng.permutation(len(top_idx))]

    # Keep rank-1 strictly the lowest
    z[best] = min(z.min(), -1.05)
    return z


# ---------------- Plots ----------------
def plot_sampling_density_2d(xy_union, order_union, sampling_set, case_id, out_dir,
                             k_density=20, density_floor=0.25, density_vmax_pct=0.85,
                             density_gamma=0.8, alpha=0.95, size=14):
    samp_idx = np.array([b in sampling_set for b in order_union], dtype=bool)
    xy_samp = xy_union[samp_idx]

    dens = local_density_knn(xy_samp, k=k_density) if len(xy_samp) >= 3 else np.zeros(len(xy_samp))
    dens_vis = stretch_density_for_visual(dens, floor=density_floor,
                                          vmax_pct=density_vmax_pct, gamma=density_gamma)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        xy_samp[:, 0], xy_samp[:, 1],
        c=dens_vis, cmap=BLUES_CMAP, s=size, alpha=alpha, edgecolors="none"
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Local density (contrast-enhanced)")

    ax.set_title(f"{case_id}: Sampling 2D density (linked canvas)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_sampling_density_2d_linked_v3.{ext}", dpi=300)
    plt.close(fig)


def plot_rank_funnel_3d(xy_union, df_pred_sorted, case_id, out_dir,
                        extreme_frac=0.10, jitter_frac=0.03, seed=42, topk_label=20):
    rng = np.random.default_rng(seed)

    n_pred = len(df_pred_sorted)
    xy_pred = xy_union[:n_pred]
    ranks = df_pred_sorted["rank"].to_numpy()

    z_base = build_monotonic_depth_from_ranks(ranks)
    z = perturb_extremes_only(z_base, ranks, rng,
                              extreme_frac=extreme_frac, jitter_frac=jitter_frac)

    # Reference funnel surface (cone)
    Rg = np.linspace(0.0, 1.0, 90)
    Tg = np.linspace(0.0, 2 * np.pi, 180)
    Rm, Tm = np.meshgrid(Rg, Tg)
    Xs = Rm * np.cos(Tm)
    Ys = Rm * np.sin(Tm)
    Zs = -Rm

    norm = Normalize(vmin=z.min(), vmax=z.max())
    colors = plt.cm.get_cmap(BLUES_CMAP)(norm(z))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2,
                    cmap=BLUES_CMAP, alpha=0.28, edgecolor="none")

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
    ax.set_title(f"{case_id}: 3D funnel (extremes perturbed; middle 70% fixed)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Normalized depth")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_rank_funnel_3d_linked_v3.{ext}", dpi=300)
    plt.close(fig)

    out = df_pred_sorted.copy()
    out["x"] = xy_pred[:, 0]
    out["y"] = xy_pred[:, 1]
    out["norm_depth_base"] = z_base
    out["norm_depth_final"] = z
    out.to_csv(out_dir / f"{case_id}_rank_funnel_layout_linked_v3.csv", index=False)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Linked 2D density and 3D funnel with extreme-only perturbations.")
    parser.add_argument("--case", type=str, default="6czf", help="Case ID")
    parser.add_argument("--sampling_csv", type=Path, default=None, help="Path to sampling CSV")
    parser.add_argument("--pred_csv", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--k_density", type=int, default=20, help="k for KNN local density")
    parser.add_argument("--density_floor", type=float, default=0.25, help="Color floor for low densities (0..1)")
    parser.add_argument("--density_vmax_pct", type=float, default=0.85, help="Upper percentile clipping (0..1)")
    parser.add_argument("--density_gamma", type=float, default=0.8, help="Gamma for density ( <1 darkens lows )")
    parser.add_argument("--extreme_frac", type=float, default=0.10, help="Fraction at each end to perturb (e.g., 0.10)")
    parser.add_argument("--jitter_frac", type=float, default=0.03, help="Z jitter stdev (relative to depth range)")
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

    df_samp = read_sampling_csv(sampling_csv)
    df_pred = read_prediction_csv(pred_csv)  # sorted by rank asc

    pred_bits = df_pred["bitstring"].tolist()
    samp_bits = df_samp["bitstring"].tolist()
    samp_set = set(samp_bits)

    xy_union, order_union, pred_mask_union = assign_union_layout(pred_bits, samp_bits)

    plot_sampling_density_2d(
        xy_union=xy_union,
        order_union=order_union,
        sampling_set=samp_set,
        case_id=case_id,
        out_dir=out_dir,
        k_density=args.k_density,
        density_floor=args.density_floor,
        density_vmax_pct=args.density_vmax_pct,
        density_gamma=args.density_gamma,
    )

    plot_rank_funnel_3d(
        xy_union=xy_union,
        df_pred_sorted=df_pred,
        case_id=case_id,
        out_dir=out_dir,
        extreme_frac=args.extreme_frac,
        jitter_frac=args.jitter_frac,
        seed=args.seed,
        topk_label=args.topk,
    )

    print("[Done]", out_dir.resolve())


if __name__ == "__main__":
    main()
