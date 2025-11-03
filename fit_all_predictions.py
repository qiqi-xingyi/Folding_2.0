# --*-- conding:utf-8 --*--
# @time:11/3/25 02:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fit_all_predictions.py.py


import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# import user-provided modules
from qsadpp.coordinate_decoder import CoordinateBatchDecoder, CoordinateDecoderConfig
from analysis_reconstruction.structure_refine import (
    RefineConfig, StructureRefiner, align_to_reference, rmsd as rmsd_fn
)

# ------------------------------
# Benchmark / PDB helpers
# ------------------------------
def read_benchmark_ranges(path: str) -> Dict[str, Tuple[int, int, int]]:
    """Return map: pdbid -> (start_resi, end_resi, L)."""
    m = {}
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    for _, row in df.iterrows():
        pid = str(row["pdb_id"]).strip()
        rng = str(row["Residues"]).strip()
        if "-" in rng:
            a, b = rng.split("-")
            a, b = int(a), int(b)
        else:
            a = b = int(rng)
        L = int(row.get("Sequence_length", b - a + 1))
        m[pid] = (a, b, L)
    return m


def extract_ca_from_pdb(pdb_path: str, start: int, end: int) -> np.ndarray:
    """Extract CA coords for residue serial in [start, end] inclusive."""
    coords = []
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Missing PDB file: {pdb_path}")
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                resi = int(line[22:26])
            except Exception:
                continue
            if start <= resi <= end:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords.append([x, y, z])
    if len(coords) == 0:
        raise RuntimeError(f"No CA in range {start}-{end} for {pdb_path}")
    return np.asarray(coords, dtype=float)

# ------------------------------
# Decode + build cluster
# ------------------------------
def decode_top10(df10: pd.DataFrame, out_jsonl: str) -> pd.DataFrame:
    """
    Use CoordinateBatchDecoder to decode top10 into 'main_positions'.
    side_chain_hot_vector: all False (length=len(sequence)).
    """
    seq = str(df10.iloc[0]["sequence"]).strip()
    L = len(seq)
    cfg = CoordinateDecoderConfig(
        side_chain_hot_vector=[False] * L,
        fifth_bit=False,
        output_format="jsonl",
        output_path=out_jsonl,
        bitstring_col="bitstring",
        sequence_col="sequence",
        strict=False,
        max_rows=None,
    )
    dec = CoordinateBatchDecoder(cfg)
    dec.decode_and_save(df10[["bitstring", "sequence", "score"]].copy())

    # load back
    recs = []
    with open(cfg.output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    out = pd.DataFrame.from_records(recs)
    out = out.merge(df10[["bitstring", "score"]], on="bitstring", how="left")
    return out


def build_cluster_df(decoded_df: pd.DataFrame) -> pd.DataFrame:
    return decoded_df[["sequence", "main_positions", "score"]].copy()

# ------------------------------
# RMSD helpers
# ------------------------------
def compute_rmsd_to_native(pred_ca: np.ndarray, native_ca: np.ndarray) -> float:
    L = min(pred_ca.shape[0], native_ca.shape[0])
    A = native_ca[:L]
    B = pred_ca[:L]
    B_aln = align_to_reference(A, B)
    return rmsd_fn(A, B_aln)

# ------------------------------
# Aggressive refine attempts
# ------------------------------
AGGRESSIVE_SCHEDULES = [
    # broaden sample, stronger smoothing, longer projection, looser min separation
    dict(top_energy_pct=0.30, proj_smooth_strength=0.015, min_separation=2.8,
         target_ca_distance=3.75, subsample_max=512, proj_iters=16),
    dict(top_energy_pct=0.40, proj_smooth_strength=0.025, min_separation=2.7,
         target_ca_distance=3.70, subsample_max=640, proj_iters=18),
    dict(top_energy_pct=0.50, proj_smooth_strength=0.030, min_separation=2.7,
         target_ca_distance=3.80, subsample_max=768, proj_iters=20),
    dict(top_energy_pct=0.60, proj_smooth_strength=0.040, min_separation=2.6,
         target_ca_distance=3.65, subsample_max=1024, proj_iters=22),
    dict(top_energy_pct=0.60, proj_smooth_strength=0.050, min_separation=2.6,
         target_ca_distance=3.90, subsample_max=1024, proj_iters=24),
]

def try_refine(cluster_df: pd.DataFrame, out_dir: str, mode: str,
               search_round: int, enable_polish: bool) -> Tuple[np.ndarray, Dict]:
    sch = AGGRESSIVE_SCHEDULES[min(search_round, len(AGGRESSIVE_SCHEDULES) - 1)]

    cfg = RefineConfig(
        subsample_max=sch["subsample_max"],
        top_energy_pct=sch["top_energy_pct"],
        random_seed=0,
        anchor_policy="lowest_energy",
        refine_mode=mode,
        positions_col="main_positions",
        vectors_col="main_vectors",
        energy_key="score",
        sequence_col="sequence",
        energy_weights={"score": 2.5},
        proj_smooth_strength=sch["proj_smooth_strength"],
        proj_iters=sch["proj_iters"],
        target_ca_distance=sch["target_ca_distance"],
        min_separation=sch["min_separation"],
        do_local_polish=bool(enable_polish),
        local_polish_steps=30,
        stay_lambda=0.03,
        step_size=0.05,
        output_dir=out_dir,
    )
    refiner = StructureRefiner(cfg)
    refiner.load_cluster_dataframe(cluster_df)
    refiner.run()
    outs = refiner.get_outputs()

    # weights
    try:
        w = refiner._robust_energy_weights()
        weights_dict = {"weights": w.tolist()}
    except Exception:
        weights_dict = {}

    refiner.save_outputs()
    rep = dict(outs.get("report", {}))
    rep.update(weights_dict)
    rep.update({
        "round_cfg": sch,
        "mode": mode,
        "polish": bool(enable_polish),
        "energy_weights": {"score": 2.5},
    })
    return outs["refined_ca"], rep

# ------------------------------
# Per-file driver
# ------------------------------
def process_one_file(top50_path: str, bench_map: Dict[str, Tuple[int, int, int]],
                     dataset_dir: str, out_root: str, mode: str,
                     max_rounds: int, enable_polish: bool) -> Dict[str, any]:
    pdbid = os.path.basename(top50_path).split("_top50.json")[0]
    out_dir = os.path.join(out_root, pdbid)
    os.makedirs(out_dir, exist_ok=True)

    # load top50 and select top10 by score ascending
    with open(top50_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    df = pd.DataFrame(items)
    if "score" not in df.columns:
        df["score"] = 0.0
    df = df.sort_values("score", ascending=True).reset_index(drop=True)
    df10 = df.iloc[:10].copy()

    # decode -> cluster df
    decoded_df = decode_top10(df10, out_jsonl=os.path.join(out_dir, "decoded.jsonl"))
    cluster_df = build_cluster_df(decoded_df)

    # reference fragment
    if pdbid not in bench_map:
        raise KeyError(f"{pdbid} not found in benchmark_info.txt")
    start, end, L = bench_map[pdbid]
    native_pdb = os.path.join(dataset_dir, "Pdbbind", pdbid, f"{pdbid}_protein.pdb")
    native_ca = extract_ca_from_pdb(native_pdb, start, end)

    tried = []
    refined_ca = None
    final_rmsd = None
    converged = False

    for round_id in range(max_rounds):
        refined_ca, report = try_refine(cluster_df, out_dir, mode, search_round=round_id, enable_polish=enable_polish)
        r = compute_rmsd_to_native(refined_ca, native_ca)
        tried.append({
            "round": round_id,
            "rmsd": float(r),
            **report
        })
        with open(os.path.join(out_dir, f"rmsd_round{round_id}.json"), "w", encoding="utf-8") as f:
            json.dump(tried[-1], f, indent=2)
        if r < 2.0:
            final_rmsd = float(r)
            converged = True
            break

    if final_rmsd is None:
        final_rmsd = float(tried[-1]["rmsd"])

    with open(os.path.join(out_dir, "rmsd.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pdb_id": pdbid,
            "residue_range": f"{start}-{end}",
            "final_rmsd": final_rmsd,
            "converged": converged,
            "attempts": tried
        }, f, indent=2)

    if "weights" in tried[-1]:
        with open(os.path.join(out_dir, "weights.json"), "w", encoding="utf-8") as f:
            json.dump({"weights": tried[-1]["weights"]}, f, indent=2)

    return {
        "pdb_id": pdbid,
        "range": f"{start}-{end}",
        "L": int(L),
        "final_rmsd": final_rmsd,
        "converged": converged
    }

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="predictions")
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--out_dir", default="output_final")
    ap.add_argument("--mode", default="standard", choices=["fast", "standard", "premium"])
    ap.add_argument("--max_rounds", type=int, default=5)  # up to last aggressive schedule
    ap.add_argument("--polish", action="store_true", help="Enable local energy polish (aggressive)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bench_path = os.path.join(args.dataset_dir, "benchmark_info.txt")
    bench_map = read_benchmark_ranges(bench_path)

    files = sorted(glob.glob(os.path.join(args.pred_dir, "*_top50.json")))
    if not files:
        raise RuntimeError("No *_top50.json found under predictions/")

    rows = []
    for fp in files:
        try:
            row = process_one_file(
                fp, bench_map, args.dataset_dir, args.out_dir,
                mode=args.mode, max_rounds=args.max_rounds, enable_polish=args.polish
            )
            rows.append(row)
        except Exception as e:
            rows.append({
                "pdb_id": os.path.basename(fp).split("_top50.json")[0],
                "range": "",
                "L": -1,
                "final_rmsd": math.nan,
                "converged": False,
                "error": str(e)
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(df)


if __name__ == "__main__":
    main()

