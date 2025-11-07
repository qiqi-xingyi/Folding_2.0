# -*- coding: utf-8 -*-
# @time: 2025/11/06
# @file: plot_sampling_and_funnel.py
#
# Description:
#   1) Plot a square 2D point cloud of quantum sampling results with blue-scale density
#      (higher local density -> darker blue).
#   2) Plot a rank-driven 3D funnel: rank-1 sits at the center with the deepest depth
#      (lowest energy), others normalized by rank order. Equal prediction signatures
#      (identical metrics) get identical normalized depth.
#
# Usage:
#   python plot_sampling_and_funnel.py
#   python plot_sampling_and_funnel.py --case 6czf \
#     --sampling_csv quantum_data/6czf/samples_6czf_all_ibm.csv \
#     --pred_csv predictions/6czf_pred.csv \
#     --k_density 20 --embed pca --topk 20
#
# Requirements:
#   pip install numpy pandas matplotlib scikit-learn

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
CENTER_COLOR = "#1f77b4"  # a blue tone to highlight the very center point
OUT_DIR_BASE = Path("result_summary/landscape")


# ---------------- Utilities ----------------
def pad_bitstrings(bitstrings):
    lengths = [len(s) for s in bitstrings]
    max_len = max(lengths)
    return [s.zfill(max_len) for s in bitstrings], max_len


def bitstrings_to_matrix(bitstrings, max_len=None):
    if max_len is None:
        _, max_len = pad_bitstrings(bitstrings)
    padded = [s.zfill(max_len) for s in bitstrings]
    arr = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8)
    arr = arr.reshape(-1, max_len)
    arr = arr - ord("0")
    return arr.astype(np.float32)


def pca_embed(bitstrings, random_state=42):
    padded, max_len = pad_bitstrings(bitstrings)
    X = bitstrings_to_matrix(padded, max_len=max_len)
    pca = PCA(n_components=2, random_state=random_state)
    XY = pca.fit_transform(X)
    # normalize to roughly unit scale for a stable square canvas
    XY = XY / (np.abs(XY).max() + 1e-9)
    return XY


def local_density_knn(xy, k=20):
    # Higher density -> smaller mean distance -> convert to density score
    k = max(2, min(k, len(xy)))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xy)
    distances, _ = nn.kneighbors(xy)  # (N, k)
    # Exclude the zero distance to itself by taking distances[:, 1:]
    d = distances[:, 1:].mean(axis=1)
    density = 1.0 / (d + 1e-12)
    # Normalize to [0,1]
    dmin, dmax = density.min(), density.max()
    if dmax > dmin:
        density = (density - dmin) / (dmax - dmin)
    else:
        density = np.zeros_like(density)
    return density


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
    # Ensure rank; if not present, build a ranking by p_rel2 desc, then logit2 desc, then score desc
    if "rank" not in df.columns:
        score_cols = [c for c in ["p_rel2", "logit2", "score"] if c in df.columns]
        ascending = [False if c in ["p_rel2", "logit2", "score"] else True for c in score_cols]
        if score_cols:
            df = df.sort_values(score_cols, ascending=ascending, na_position="last")
            df["rank"] = np.arange(1, len(df) + 1)
        else:
            # fallback: arbitrary ranking
            df["rank"] = np.arange(1, len(df) + 1)
    return df


def equal_signature_groups(df_pred: pd.DataFrame):
    # Build a signature from available prediction metrics
    sig_cols = [c for c in df_pred.columns if c.startswith("p_rel") or c.startswith("logit") or c == "score"]
    if not sig_cols:
        # fallback: use rank equality (unlikely ties)
        sig = df_pred["rank"].astype(str)
    else:
        sig = df_pred[sig_cols].astype(str).agg("|".join, axis=1)
    return sig


def normalize_depth_by_rank(df_pred: pd.DataFrame) -> pd.DataFrame:
    # Sort by rank ascending; map rank to radius/depth
    df = df_pred.copy()
    df = df.sort_values("rank", ascending=True).reset_index(drop=True)

    # If signatures are equal, assign identical normalized depth
    sig = equal_signature_groups(df)
    # Depth index: group by signature, assign the minimum order index for the group
    order_index = df.index.to_series()
    group_min_index = sig.groupby(sig).transform("min").index
    # Map each signature to the first occurrence index (stable order)
    first_idx_per_sig = sig.drop_duplicates().index
    sig_to_first = {sig[i]: int(i) for i in first_idx_per_sig}
    sig_first_idx = sig.map(sig_to_first)

    n = len(df)
    if n > 1:
        norm = sig_first_idx.astype(float) / (n - 1)
    else:
        norm = pd.Series([0.0], index=df.index)

    # Depth: rank-1 -> deepest (most negative), others closer to 0
    df["norm_depth"] = -norm.values
    return df


def golden_angle_layout(n):
    # Vogel spiral layout (radius grows with index, angle by golden angle)
    # Rank-1 sits at center with r=0
    if n <= 0:
        return np.zeros((0, 2))
    phi = (3 - np.sqrt(5)) * np.pi  # golden angle (~137.5 deg) in radians
    idx = np.arange(n)  # 0..n-1
    r = idx / max(1, n - 1)  # 0..1
    theta = idx * phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y]), r


# ---------------- Plots ----------------
def plot_sampling_density_2d(df_sample: pd.DataFrame, case_id: str, out_dir: Path, k_density=20, embed_seed=42):
    bits = df_sample["bitstring"].tolist()
    XY = pca_embed(bits, random_state=embed_seed)
    dens = local_density_knn(XY, k=k_density)

    # Square figure, equal aspect
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(XY[:, 0], XY[:, 1],
                    c=dens, cmap=BLUES_CMAP, s=16, alpha=0.95,
                    edgecolors="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Local density")

    ax.set_title(f"{case_id}: Sampling 2D density (PCA)")
    ax.set_xlabel("Embed-X")
    ax.set_ylabel("Embed-Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(XY[:, 0].min() - 0.05, XY[:, 0].max() + 0.05)
    ax.set_ylim(XY[:, 1].min() - 0.05, XY[:, 1].max() + 0.05)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_sampling_density_2d.{ext}", dpi=300)
    plt.close(fig)

    # Export embedding for reproducibility
    emb = pd.DataFrame({"bitstring": bits, "x": XY[:, 0], "y": XY[:, 1], "density": dens})
    emb.to_csv(out_dir / f"{case_id}_sampling_embedding_density.csv", index=False)


def plot_rank_funnel_3d(df_pred: pd.DataFrame, case_id: str, out_dir: Path, topk_label=20):
    # Normalize depths by rank and signature equality
    df_norm = normalize_depth_by_rank(df_pred)
    n = len(df_norm)

    # Layout in 2D using Vogel spiral (rank-1 at center)
    XY, radius = golden_angle_layout(n)
    Z = df_norm["norm_depth"].to_numpy()  # in [-1, 0], rank-1 -> -0

    # Build a smooth funnel surface (cone-like): Z_surf = -R
    Rg = np.linspace(0.0, 1.0, 64)
    Tg = np.linspace(0.0, 2 * np.pi, 128)
    Rm, Tm = np.meshgrid(Rg, Tg)
    Xs = Rm * np.cos(Tm)
    Ys = Rm * np.sin(Tm)
    Zs = -Rm  # deepest at center

    # Scatter colors by normalized depth (map to Blues)
    norm = Normalize(vmin=Z.min(), vmax=Z.max())
    colors = plt.cm.get_cmap(BLUES_CMAP)(norm(Z))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Surface
    ax.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2,
                    cmap=BLUES_CMAP, alpha=0.35, edgecolor="none")

    # Points
    ax.scatter(XY[:, 0], XY[:, 1], Z, s=18, c=colors, depthshade=False, edgecolors="k", linewidths=0.2)

    # Highlight top-1 at the very center
    ax.scatter([0.0], [0.0], [Z[0]], s=160, c=CENTER_COLOR, edgecolors="k", linewidths=0.8, depthshade=False, label="Rank-1")

    # Optionally annotate top-k points
    topk = min(topk_label, n)
    for i in range(topk):
        xi, yi, zi = XY[i, 0], XY[i, 1], Z[i]
        ax.text(xi, yi, zi, f"{i+1}", fontsize=10, ha="center", va="bottom", color="k")

    # Limits for a square footprint
    lim = 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(Z.min() - 0.05, 0.1)

    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{case_id}: Rank-driven 3D funnel (rank normalized)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Normalized depth")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_rank_funnel_3d.{ext}", dpi=300)
    plt.close(fig)

    # Export layout for traceability
    out = df_norm.copy()
    out["x"] = XY[:, 0]
    out["y"] = XY[:, 1]
    out["norm_depth"] = Z
    out.to_csv(out_dir / f"{case_id}_rank_funnel_layout.csv", index=False)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Plot sampling 2D density and rank-driven 3D funnel.")
    parser.add_argument("--case", type=str, default="6czf", help="Case ID")
    parser.add_argument("--sampling_csv", type=Path, default=None, help="Path to sampling CSV")
    parser.add_argument("--pred_csv", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--k_density", type=int, default=20, help="k for KNN local density")
    parser.add_argument("--embed", type=str, default="pca", choices=["pca"], help="Embedding method for sampling (fixed to PCA here)")
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

    # 2D density plot for sampling
    df_sample = read_sampling_csv(sampling_csv)
    plot_sampling_density_2d(df_sample, case_id, out_dir, k_density=args.k_density, embed_seed=42)

    # 3D rank-driven funnel
    df_pred = read_prediction_csv(pred_csv)
    plot_rank_funnel_3d(df_pred, case_id, out_dir, topk_label=args.topk)

    print("[Done]", out_dir.resolve())


if __name__ == "__main__":
    main()
