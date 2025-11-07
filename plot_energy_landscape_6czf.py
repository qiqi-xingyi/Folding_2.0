# --*-- conding:utf-8 --*--
# @time:11/6/25 20:42
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_energy_landscape_6czf.py

# Description:
#   Visualize the energy landscape for a case (default: 6czf).
#   - Read quantum sampling CSV and prediction CSV
#   - Unify bitstring length, build 2D embedding (PCA or MDS-Hamming)
#   - Define energy E by priority: -log(p_rel2+eps) -> -logit2 -> normalized rank
#   - Produce figures:
#       (A) Sampling scatter (2D)
#       (B) Energy scatter (2D)
#       (C) Energy contour (2D)
#       (D) 3D energy funnel surface
#       (E) Energy vs. rank
#       (F) Sampling coverage vs. TopK
#   - Export embedding coordinates to CSV
#
# Requirements:
#   pip install numpy pandas matplotlib scikit-learn

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

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

QSAD_COLOR = "#00A59B"
SAMPLE_COLOR = "#C7C7C7"
TOPK_EDGE = "black"
CMAP = "viridis"


# ---------------- I/O helpers ----------------
def read_sampling_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Choose weight: prefer 'prob', else count/shots, else fallback
    if "prob" in df.columns and df["prob"].notna().any():
        df["weight"] = pd.to_numeric(df["prob"], errors="coerce")
    elif {"count", "shots"}.issubset(set(df.columns)):
        c = pd.to_numeric(df["count"], errors="coerce")
        s = pd.to_numeric(df["shots"], errors="coerce").replace(0, np.nan)
        df["weight"] = c / s
    else:
        df["weight"] = 1.0 / max(len(df), 1)
    df["bitstring"] = df["bitstring"].astype(str)
    return df[["bitstring", "weight"]].copy()


def read_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["bitstring"] = df["bitstring"].astype(str)

    eps = 1e-12
    E = None
    if "p_rel2" in df.columns:
        with np.errstate(divide="ignore"):
            E = -np.log(pd.to_numeric(df["p_rel2"], errors="coerce") + eps)
    if (E is None) or (E.isna().all()):
        if "logit2" in df.columns:
            E = -pd.to_numeric(df["logit2"], errors="coerce")
    if (E is None) or (E.isna().all()):
        if "rank" in df.columns:
            rr = pd.to_numeric(df["rank"], errors="coerce")
            E = rr / rr.max()
        else:
            raise ValueError("No usable columns to define energy (need p_rel2 or logit2 or rank).")
    df["energy"] = E

    if "rank" not in df.columns or df["rank"].isna().all():
        df["rank"] = df["energy"].rank(method="first", ascending=True).astype(int)

    keep_cols = ["bitstring", "energy", "rank"]
    for c in ["sequence", "group_id", "p_rel2", "logit2", "score"]:
        if c in df.columns:
            keep_cols.append(c)
    return df[keep_cols].copy()


# ---------------- Bitstring utilities ----------------
def pad_bitstrings(bitstrings):
    lengths = [len(s) for s in bitstrings]
    max_len = max(lengths)
    return [s.zfill(max_len) for s in bitstrings], max_len


def bitstrings_to_matrix(bitstrings, max_len=None):
    if max_len is None:
        _, max_len = pad_bitstrings(bitstrings)
    padded = [s.zfill(max_len) for s in bitstrings]
    # Convert to 0/1 matrix (N, L)
    arr = np.frombuffer("".join(padded).encode("ascii"), dtype=np.uint8)
    arr = arr.reshape(-1, max_len)
    arr = arr - ord("0")
    return arr.astype(np.float32)


def hamming_distance_matrix(arr):
    N = arr.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    # Chunked computation to reduce memory pressure
    chunk = max(1, 2000 // max(1, arr.shape[1] // 64 + 1))
    for i in range(0, N, chunk):
        end = min(N, i + chunk)
        block = arr[i:end]  # (b,L)
        diff = np.abs(block[:, None, :] - arr[None, :, :])  # (b,N,L)
        D[i:end] = diff.sum(axis=2)
    return D


def embed_2d(bitstrings, method="pca", random_state=42):
    padded, max_len = pad_bitstrings(bitstrings)
    X = bitstrings_to_matrix(padded, max_len=max_len)
    N = X.shape[0]

    if method.lower() == "mds":
        if N > 5000:
            warnings.warn("N too large for MDS; falling back to PCA.", RuntimeWarning)
            method = "pca"
        else:
            try:
                D = hamming_distance_matrix(X)
                mds = MDS(
                    n_components=2,
                    dissimilarity="precomputed",
                    random_state=random_state,
                    n_init=4,
                    max_iter=300,
                )
                XY = mds.fit_transform(D)
                return XY
            except Exception as e:
                warnings.warn(f"MDS failed ({e}); falling back to PCA.", RuntimeWarning)
                method = "pca"

    pca = PCA(n_components=2, random_state=random_state)
    XY = pca.fit_transform(X)
    return XY


# ---------------- Plotting helpers ----------------
def ensure_outdir(case_id: str) -> Path:
    out_dir = Path("result_summary") / "landscape" / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_maxfinite(a: np.ndarray, default: float = 0.0) -> float:
    if a.size == 0:
        return default
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return default
    return float(finite.max())


# (A) sampling 2D
def plot_sampling_2d(xy_all, df_all, out_dir: Path, case_id: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    samp_idx = (df_all["source"] == "sample").to_numpy()
    if not np.any(samp_idx):
        ax.set_title(f"{case_id}: Quantum Sampling (no sampling points)")
        ax.set_xlabel("Embed-X")
        ax.set_ylabel("Embed-Y")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{case_id}_sampling_2d.{ext}", dpi=300)
        plt.close(fig)
        return

    s_w = df_all.loc[samp_idx, "weight"].to_numpy(dtype=float)
    maxw = safe_maxfinite(s_w, default=0.0)
    if maxw > 0:
        s = 20.0 + 180.0 * (np.nan_to_num(s_w, nan=0.0) / maxw)
    else:
        s = np.full(s_w.shape, 30.0)

    ax.scatter(
        xy_all[samp_idx, 0], xy_all[samp_idx, 1],
        s=s, c=SAMPLE_COLOR, alpha=0.6, edgecolors="none", label="Sampling"
    )
    ax.set_title(f"{case_id}: Quantum Sampling (2D)")
    ax.set_xlabel("Embed-X")
    ax.set_ylabel("Embed-Y")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_sampling_2d.{ext}", dpi=300)
    plt.close(fig)


# (B) energy scatter 2D
def plot_energy_scatter_2d(xy_all, df_all, out_dir: Path, case_id: str, top_k=10):
    fig, ax = plt.subplots(figsize=(9, 7))
    pred_idx = (df_all["source"] == "pred").to_numpy()
    if not np.any(pred_idx):
        ax.set_title(f"{case_id}: Prediction Energy (no prediction points)")
        ax.set_xlabel("Embed-X")
        ax.set_ylabel("Embed-Y")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{case_id}_energy_scatter_2d.{ext}", dpi=300)
        plt.close(fig)
        return

    E = df_all.loc[pred_idx, "energy"].to_numpy(dtype=float)
    if not np.isfinite(E).any():
        E = np.zeros_like(E)

    size = 30.0 + 120.0 * (1.0 / (np.nan_to_num(E, nan=0.0) + 1.0))
    sc = ax.scatter(
        xy_all[pred_idx, 0], xy_all[pred_idx, 1],
        c=E, s=size, cmap=CMAP, alpha=0.85, edgecolors="black", linewidths=0.2
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Energy E")

    df_pred = df_all.loc[pred_idx, ["bitstring", "energy"]].copy()
    df_pred = df_pred[np.isfinite(df_pred["energy"])]
    df_pred = df_pred.sort_values("energy", ascending=True).head(top_k)
    for _, row in df_pred.iterrows():
        mask = (df_all["bitstring"] == row["bitstring"]).to_numpy()
        xi = xy_all[mask, 0][0]
        yi = xy_all[mask, 1][0]
        ax.scatter([xi], [yi], s=160, facecolors="none", edgecolors=TOPK_EDGE, linewidths=1.2)
        ax.text(xi, yi, "Top", fontsize=12, color="black", ha="center", va="bottom")

    ax.set_title(f"{case_id}: Prediction Energy (2D)")
    ax.set_xlabel("Embed-X")
    ax.set_ylabel("Embed-Y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_energy_scatter_2d.{ext}", dpi=300)
    plt.close(fig)


# (C) energy contour 2D
def plot_energy_contour_2d(xy_all, df_all, out_dir: Path, case_id: str, top_k=10):
    pred_idx = (df_all["source"] == "pred").to_numpy()
    if not np.any(pred_idx):
        return
    X = xy_all[pred_idx, 0]
    Y = xy_all[pred_idx, 1]
    Z = df_all.loc[pred_idx, "energy"].to_numpy(dtype=float)
    if len(X) < 3 or not np.isfinite(Z).any():
        return

    triang = mtri.Triangulation(X, Y)
    fig, ax = plt.subplots(figsize=(9, 7))
    cntr = ax.tricontourf(triang, Z, levels=14, cmap=CMAP, alpha=0.9)
    plt.colorbar(cntr, ax=ax, label="Energy E")
    ax.tricontour(triang, Z, levels=14, colors="k", linewidths=0.3, alpha=0.5)

    df_pred = df_all.loc[pred_idx, ["bitstring", "energy"]].copy()
    df_pred = df_pred[np.isfinite(df_pred["energy"])]
    df_pred = df_pred.sort_values("energy", ascending=True).head(top_k)
    for _, row in df_pred.iterrows():
        mask = (df_all["bitstring"] == row["bitstring"]).to_numpy()
        xi = xy_all[mask, 0][0]
        yi = xy_all[mask, 1][0]
        ax.scatter([xi], [yi], s=120, c="white", edgecolors="black", linewidths=1.0, zorder=3)

    ax.set_title(f"{case_id}: Energy Contour (2D)")
    ax.set_xlabel("Embed-X")
    ax.set_ylabel("Embed-Y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_energy_contour_2d.{ext}", dpi=300)
    plt.close(fig)


# (D) energy funnel 3D
def plot_energy_funnel_3d(xy_all, df_all, out_dir: Path, case_id: str, top_k=10):
    pred_idx = (df_all["source"] == "pred").to_numpy()
    if not np.any(pred_idx):
        return
    X = xy_all[pred_idx, 0]
    Y = xy_all[pred_idx, 1]
    Z = df_all.loc[pred_idx, "energy"].to_numpy(dtype=float)
    if X.size < 3 or Y.size < 3 or not np.isfinite(Z).any():
        return

    triang = mtri.Triangulation(X, Y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    trisurf = ax.plot_trisurf(triang, Z, cmap=CMAP, linewidth=0.2, antialiased=True, alpha=0.85)
    cb = fig.colorbar(trisurf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    cb.set_label("Energy E")

    samp_idx = (df_all["source"] == "sample").to_numpy()
    if np.any(samp_idx):
        xs = xy_all[samp_idx, 0]
        ys = xy_all[samp_idx, 1]
        z_ref = safe_maxfinite(Z, default=0.0) + 0.2
        ax.scatter(xs, ys, z_ref, s=8, c=SAMPLE_COLOR, alpha=0.35, depthshade=False, edgecolors="none", label="Sampling")

    ax.scatter(X, Y, Z, s=15, c="k", alpha=0.3, depthshade=False)

    df_pred = df_all.loc[pred_idx, ["bitstring", "energy"]].copy()
    df_pred = df_pred[np.isfinite(df_pred["energy"])].sort_values("energy", ascending=True).head(top_k)
    for _, row in df_pred.iterrows():
        mask = (df_all["bitstring"] == row["bitstring"]).to_numpy()
        xi = xy_all[mask, 0][0]
        yi = xy_all[mask, 1][0]
        zi = df_all.loc[mask, "energy"].iloc[0]
        ax.scatter([xi], [yi], [zi], s=80, c="white", edgecolors="black", depthshade=False)

    if len(df_pred) > 0:
        row0 = df_pred.iloc[0]
        mask0 = (df_all["bitstring"] == row0["bitstring"]).to_numpy()
        xi = xy_all[mask0, 0][0]
        yi = xy_all[mask0, 1][0]
        zi = df_all.loc[mask0, "energy"].iloc[0]
        ax.scatter([xi], [yi], [zi], s=160, c=QSAD_COLOR, edgecolors="black", linewidths=0.8, depthshade=False, label="Min E")

    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{case_id}: 3D Energy Funnel")
    ax.set_xlabel("Embed-X")
    ax.set_ylabel("Embed-Y")
    ax.set_zlabel("Energy E")
    ax.legend(loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_energy_funnel_3d.{ext}", dpi=300)
    plt.close(fig)


# (E) energy vs. rank
def plot_energy_vs_rank(df_pred, out_dir: Path, case_id: str):
    dfp = df_pred.dropna(subset=["energy", "rank"]).copy()
    if len(dfp) == 0:
        return
    dfp = dfp.sort_values("rank")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.plot(dfp["rank"], dfp["energy"], marker="o", linestyle="-", linewidth=2, markersize=4)
    ax.set_xlabel("Rank (ascending)")
    ax.set_ylabel("Energy E")
    ax.set_title(f"{case_id}: Energy vs. Rank")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_energy_vs_rank.{ext}", dpi=300)
    plt.close(fig)


# (F) sampling coverage vs. TopK
def plot_sampling_coverage(df_sample, df_pred, out_dir: Path, case_id: str, topk_list=(10, 50, 100)):
    s_set = set(df_sample["bitstring"].tolist())
    dfp = df_pred.dropna(subset=["energy"]).copy().sort_values("energy", ascending=True)
    cov = []
    for K in topk_list:
        sub = dfp.head(K)
        inter = sum(1 for b in sub["bitstring"] if b in s_set)
        cov.append(inter / max(K, 1))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    x = np.arange(len(topk_list))
    ax.bar(x, cov, width=0.6, color="#8499BB", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in topk_list])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage (in sampling)")
    ax.set_title(f"{case_id}: Sampling Coverage of Low-E Predictions")
    for i, v in enumerate(cov):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{case_id}_sampling_coverage.{ext}", dpi=300)
    plt.close(fig)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Plot energy landscape (2D/3D) for a given case (default: 6czf).")
    parser.add_argument("--case", type=str, default="6czf", help="Case ID (e.g., 6czf)")
    parser.add_argument("--sampling_csv", type=Path, default=None, help="Path to sampling CSV")
    parser.add_argument("--pred_csv", type=Path, default=None, help="Path to prediction CSV")
    parser.add_argument("--embed", type=str, default="pca", choices=["pca", "mds"], help="Embedding method")
    parser.add_argument("--topk", type=int, default=10, help="Top-K minima to highlight")
    args = parser.parse_args()

    case_id = args.case
    sampling_csv = args.sampling_csv or Path(f"quantum_data/{case_id}/samples_{case_id}_all_ibm.csv")
    pred_csv = args.pred_csv or Path(f"predictions/{case_id}_pred.csv")

    if not sampling_csv.exists():
        raise FileNotFoundError(f"Sampling CSV not found: {sampling_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    out_dir = ensure_outdir(case_id)

    df_sample = read_sampling_csv(sampling_csv)
    df_pred = read_prediction_csv(pred_csv)

    all_bits = pd.Index(df_sample["bitstring"]).union(df_pred["bitstring"]).tolist()
    XY = embed_2d(all_bits, method=args.embed, random_state=42)

    df_all = pd.DataFrame({"bitstring": all_bits})
    df_all["source"] = "none"
    df_all["weight"] = np.nan
    df_all["energy"] = np.nan
    df_all["rank"] = np.nan

    s_map = df_sample.set_index("bitstring")["weight"].to_dict()
    smask = df_all["bitstring"].isin(s_map.keys())
    df_all.loc[smask, "source"] = "sample"
    df_all.loc[smask, "weight"] = df_all.loc[smask, "bitstring"].map(s_map)

    pE = df_pred.set_index("bitstring")["energy"].to_dict()
    pr = df_pred.set_index("bitstring")["rank"].to_dict()
    pmask = df_all["bitstring"].isin(pE.keys())
    df_all.loc[pmask, "source"] = "pred"
    df_all.loc[pmask, "energy"] = df_all.loc[pmask, "bitstring"].map(pE)
    df_all.loc[pmask, "rank"] = df_all.loc[pmask, "bitstring"].map(pr)

    emb_out = out_dir / f"{case_id}_embedding_xy.csv"
    emb_df = df_all.copy()
    emb_df["x"] = XY[:, 0]
    emb_df["y"] = XY[:, 1]
    emb_df.to_csv(emb_out, index=False)
    print(f"[Saved] {emb_out}")

    plot_sampling_2d(XY, df_all, out_dir, case_id)
    plot_energy_scatter_2d(XY, df_all, out_dir, case_id, top_k=args.topk)
    plot_energy_contour_2d(XY, df_all, out_dir, case_id, top_k=args.topk)
    plot_energy_funnel_3d(XY, df_all, out_dir, case_id, top_k=args.topk)
    plot_energy_vs_rank(df_pred, out_dir, case_id)
    plot_sampling_coverage(df_sample, df_pred, out_dir, case_id, topk_list=(10, 50, 100))

    print("[Done] All figures saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()

