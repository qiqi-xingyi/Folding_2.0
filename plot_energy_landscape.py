# -*- coding: utf-8 -*-
# @file: plot_landscape_light_en.py
# @desc: Lightweight O(n) landscape visualization where z increases with distance from the global minimum.
#        Uses your requested parameters: Z_POWER=0.35, DIST_GAIN=0.55, DIST_POWER=0.95
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

    # Distance to the global minimum, scaled by 95th percentile (keeps small but non-zero)
    bs_min = bitstrings[idx_min]
    use_rmsd = positions[idx_min] is not None and all(p is not None for p in positions)
    if use_rmsd:
        Ps = equalize_lengths([p for p in positions if p is not None])
        Pmin = Ps[idx_min]
        dist_to_min_raw = np.array([rmsd_after_kabsch(Pmin, Ps[i]) for i in range(n)], dtype=float)
        p95 = np.percentile(dist_to_min_raw, 95)
        dist_w = dist_to_min_raw / (p95 + 1e-12)
    else:
        raw = np.array([hamming_to_min(bitstrings[i], bs_min) for i in range(n)], dtype=float)
        p95 = np.percentile(raw, 95)
        dist_w = raw / (p95 + 1e-12)

    dist_w = np.clip(dist_w, 0.0, 1.0)
    dist_w[idx_min] = 0.0

    # Radius: less compression near the bottom, more structural weight
    rank_norm = (np.argsort(np.argsort(E)) / (n - 1)).astype(float)
    a, b, p = 0.55, 0.45, 1.1
    r = a * (rank_norm ** p) + b * dist_w
    r[idx_min] = 0.0

    # Small radius floor for non-minimum points to avoid collapsing
    idx = np.arange(n)
    mask = idx != idx_min
    r_floor = 0.03 + 0.07 * rank_norm
    r[mask] = np.maximum(r[mask], r_floor[mask])

    # Optional: gently separate near-equal energies into thin rings
    E_bucket = np.round(E - Emin, 4)
    _, bucket_idx = np.unique(E_bucket, return_inverse=True)
    r += 0.012 * bucket_idx.astype(float)

    # Angle from hash to spread points deterministically
    theta = np.array([stable_angle_from_hash(bs) for bs in bitstrings], dtype=float)
    x2d = r * np.cos(theta)
    y2d = r * np.sin(theta)

    # ----------- z mapping: your requested parameters -----------
    Z_POWER = 0.5
    z_energy = np.power(np.maximum(Ehat - Ehat.min(), 0.0) + 1e-12, Z_POWER)

    DIST_GAIN = 0.45
    DIST_POWER = 0.98
    z_dist = DIST_GAIN * np.power(dist_w, DIST_POWER)

    rough = np.array([((int.from_bytes(hashlib.md5(bs.encode()).digest()[8:12], 'big') / 2**32) - 0.5)
                      for bs in bitstrings], dtype=float)
    z3d = z_energy + z_dist + 0.02 * rough

    # Sink a small set of lowest-energy structures to sharpen the tip
    SINK_K = max(3, int(0.002 * n))
    SINK_DELTA = 0.1
    order = np.argsort(E)
    lowest_k = order[:SINK_K]
    if len(lowest_k) > 0:
        grades = np.linspace(1.0, 0.7, len(lowest_k))
        z3d[lowest_k] -= SINK_DELTA * grades

    # Ensure the global minimum is strictly the lowest
    z3d[idx_min] = np.min(z3d) - 1e-6

    # ---------- 3D plot ----------
    fig = plt.figure(figsize=(8, 8), dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x2d, y2d, z3d, s=6, c=E, cmap="viridis", alpha=0.9, depthshade=True)
    ax.scatter([x2d[idx_min]], [y2d[idx_min]], [z3d[idx_min]], s=60, c="red",
               marker="*", edgecolor="k", linewidths=0.3, zorder=5)
    ax.set_title(f"3D Funnel (min-E = {Emin:.3f})")
    ax.set_xlabel("X (energy-dominant radial)")
    ax.set_ylabel("Y (hash-based angle)")
    ax.set_zlabel("Height: energy + distance-to-min")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.06)
    cbar.set_label("E_total")
    ax.view_init(elev=28, azim=38)
    ensure_dir(OUTDIR)
    for ext in ["png", "pdf"]:
        fig.savefig(outdir := OUTDIR / f"funnel_3d.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ---------- 2D plot ----------
    fig = plt.figure(figsize=(7.5, 7), dpi=600)
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

    # Save coordinates
    np.savez(OUTDIR / "coords_outputs_light.npz",
             X2d=x2d, Y2d=y2d, Z3d=z3d, E=E, idx_min=idx_min, dist_to_min=dist_w)

    print(f"[n] {n}")
    print(f"[saved] {OUTDIR/'funnel_3d.png'}")
    print(f"[saved] {OUTDIR/'density_2d.png'}")
    print(f"[saved] {OUTDIR/'coords_outputs_light.npz'}")

if __name__ == "__main__":
    main()
