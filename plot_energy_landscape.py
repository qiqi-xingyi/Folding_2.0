# -*- coding: utf-8 -*-
# @file: plot_landscape_light_en.py
# @desc: Lightweight O(n) landscape visualization with sharpened funnel bottom.
#        Reads e_results/1m7y/energies.jsonl
#        Outputs: result_summary/landscape/{funnel_3d,density_2d}.png/pdf

import json, hashlib, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = Path("e_results/1m7y/energies.jsonl")
OUTDIR = Path("result_summary/landscape")

# ---------- utils ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def median_absolute_deviation(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)

def parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def extract_positions(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    pos = entry.get("main_positions")
    if not pos:
        return None
    return np.array(pos, dtype=float)

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    Q_aln = (R @ Q.T).T + t
    diff = P - Q_aln
    return float(np.sqrt((diff ** 2).sum() / P.shape[0]))

def equalize_lengths(arrays: List[np.ndarray]) -> List[np.ndarray]:
    Lmin = min(a.shape[0] for a in arrays)
    return [a[:Lmin] for a in arrays]

def hamming_to_min(bs: str, bs_min: str) -> float:
    L = max(len(bs), len(bs_min))
    a = bs.ljust(L, '0')
    b = bs_min.ljust(L, '0')
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) / float(L)

def stable_angle_from_hash(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).digest()
    val = int.from_bytes(h[:8], 'big')
    return (val / 2**64) * 2.0 * math.pi

# ---------- main ----------
def main():
    ensure_dir(OUTDIR)
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    rows = parse_jsonl(INPUT_PATH)
    if not rows:
        raise RuntimeError("Input JSONL is empty.")

    bitstrings, positions, energies = [], [], []
    for r in rows:
        bs = r.get("bitstring")
        E = r.get("E_total")
        P = extract_positions(r)
        if (bs is None) or (E is None):
            continue
        bitstrings.append(str(bs))
        energies.append(float(E))
        positions.append(P)

    n = len(energies)
    if n < 3:
        raise RuntimeError(f"Too few valid entries: {n}")

    E = np.array(energies, dtype=float)
    idx_min = int(np.argmin(E))
    Emin = float(E[idx_min])

    # Energy normalization
    dE = E - Emin
    tau = median_absolute_deviation(dE)
    if tau <= 1e-12:
        tau = float(np.percentile(dE, 75) - np.percentile(dE, 25) + 1e-12)
    Ehat = dE / (median_absolute_deviation(dE) + 1e-12)
    rank_norm = (np.argsort(np.argsort(E)) / (n - 1)).astype(float)

    # Single-source structural distance to the lowest-energy sample
    bs_min = bitstrings[idx_min]
    use_rmsd = positions[idx_min] is not None and all(p is not None for p in positions)
    if use_rmsd:
        Ps = equalize_lengths([p for p in positions if p is not None])
        Pmin = Ps[idx_min]
        dist_to_min = np.array([rmsd_after_kabsch(Pmin, Ps[i]) for i in range(n)], dtype=float)
        d_med = np.median(dist_to_min)
        d_mad = median_absolute_deviation(dist_to_min)
        d_scale = (dist_to_min - d_med) / (d_mad + 1e-12)
        d_scale = np.clip(d_scale, 0.0, None)
        d_scale /= (np.percentile(d_scale, 95) + 1e-12)
        struct_to_min = d_scale
    else:
        struct_to_min = np.array([hamming_to_min(bitstrings[i], bs_min) for i in range(n)], dtype=float)

    struct_to_min[idx_min] = 0.0

    # Radius: energy-dominant with a small structural correction
    a, b, p = 0.75, 0.25, 0.6
    r = a * (rank_norm ** p) + b * struct_to_min
    r[idx_min] = 0.0

    # Angle: deterministic from hash
    theta = np.array([stable_angle_from_hash(bs) for bs in bitstrings], dtype=float)
    x2d = r * np.cos(theta)
    y2d = r * np.sin(theta)

    # ----------- sharpened funnel z -----------
    # Nonlinear energy-to-height mapping (sharper near minimum)
    Z_POWER = 1.25
    z_base = np.power(np.maximum(Ehat - Ehat.min(), 0.0) + 1e-12, Z_POWER)

    # Rugged walls: small deterministic noise
    rough = np.array([((int.from_bytes(hashlib.md5(bs.encode()).digest()[8:12], 'big') / 2**32) - 0.5)
                      for bs in bitstrings], dtype=float)
    z3d = z_base + 0.08 * rough

    # Sink the K lowest-energy structures
    SINK_K = max(3, int(0.002 * n))  # ~0.2% of the dataset
    SINK_DELTA = 0.35
    order = np.argsort(E)
    lowest_k = order[:SINK_K]
    if len(lowest_k) > 0:
        grades = np.linspace(1.0, 0.6, len(lowest_k))
        z3d[lowest_k] -= SINK_DELTA * grades

    # Lift other samples that tie (by rounded energy) with the K-th minimum
    ROUND_DEC = 4
    UPLIFT_SAME = 0.15
    if SINK_K > 0:
        Ek = np.round(E[order[SINK_K - 1]], decimals=ROUND_DEC)
        tied = np.where(np.round(E, ROUND_DEC) == Ek)[0]
        tied = np.setdiff1d(tied, lowest_k, assume_unique=False)
        if tied.size > 0:
            z3d[tied] += UPLIFT_SAME

    # Ensure the global minimum is strictly the lowest
    z3d[idx_min] = np.min(z3d) - 1e-6

    # ----------- 3D Funnel -----------
    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x2d, y2d, z3d, s=6, c=E, cmap="viridis", alpha=0.9, depthshade=True)
    ax.scatter([x2d[idx_min]], [y2d[idx_min]], [z3d[idx_min]], s=60, c="red",
               marker="*", edgecolor="k", linewidths=0.3, zorder=5)
    ax.set_title(f"3D Funnel (min-E = {Emin:.3f})")
    ax.set_xlabel("X (energy-dominant radial)")
    ax.set_ylabel("Y (hash-based angle)")
    ax.set_zlabel("Relative Energy (z)")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.06)
    cbar.set_label("E_total")
    ax.view_init(elev=28, azim=38)
    ensure_dir(OUTDIR)
    for ext in ["png", "pdf"]:
        fig.savefig(OUTDIR / f"funnel_3d.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ----------- 2D Density -----------
    fig = plt.figure(figsize=(7.5, 7), dpi=150)
    ax = fig.add_subplot(111)
    hb = ax.hexbin(x2d, y2d, gridsize=60, mincnt=1, cmap="Blues", linewidths=0.0)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Local density")
    sc2 = ax.scatter(x2d, y2d, c=E, s=6, cmap="viridis", alpha=0.85, edgecolors="none")
    ax.scatter([x2d[idx_min]], [y2d[idx_min]], s=60, c="red",
               marker="*", edgecolor="k", linewidths=0.3, zorder=5)
    ax.set_title(f"2D Density (min-E = {Emin:.3f})")
    ax.set_xlabel("Dim-1 (energy-dominant)")
    ax.set_ylabel("Dim-2 (hash-based angle)")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    for ext in ["png", "pdf"]:
        fig.savefig(OUTDIR / f"density_2d.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Save coordinates for downstream use
    np.savez(OUTDIR / "coords_outputs_light.npz",
             X2d=x2d, Y2d=y2d, Z3d=z3d, E=E, idx_min=idx_min)

    print(f"[n] {n}")
    print(f"[saved] {OUTDIR/'funnel_3d.png'}")
    print(f"[saved] {OUTDIR/'density_2d.png'}")
    print(f"[saved] {OUTDIR/'coords_outputs_light.npz'}")

if __name__ == "__main__":
    main()
