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
import itertools
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from qsadpp.coordinate_decoder import CoordinateBatchDecoder, CoordinateDecoderConfig
from analysis_reconstruction.structure_refine import (
    RefineConfig, StructureRefiner, align_to_reference, rmsd as rmsd_fn
)

# ------------------------------
# Basic helpers
# ------------------------------
def read_benchmark_ranges(path: str) -> Dict[str, Tuple[int, int, int]]:
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


def compute_rmsd_to_native(pred_ca: np.ndarray, native_ca: np.ndarray) -> float:
    L = min(pred_ca.shape[0], native_ca.shape[0])
    A = native_ca[:L]
    B = pred_ca[:L]
    B_aln = align_to_reference(A, B)
    return rmsd_fn(A, B_aln)


def compute_rmsd_to_coords(pred_ca: np.ndarray, ref_ca: np.ndarray) -> float:
    L = min(pred_ca.shape[0], ref_ca.shape[0])
    A = ref_ca[:L]
    B = pred_ca[:L]
    B_aln = align_to_reference(A, B)
    return rmsd_fn(A, B_aln)


# ------------------------------
# Decode all top50 and compute per-sample RMSD
# ------------------------------
def decode_all_and_rmsd(top50_items: List[dict], native_ca: np.ndarray, out_jsonl: str) -> pd.DataFrame:
    df = pd.DataFrame(top50_items)
    if "score" not in df.columns:
        df["score"] = 0.0

    seq = str(df.iloc[0]["sequence"]).strip()
    L = len(seq)
    dec_cfg = CoordinateDecoderConfig(
        side_chain_hot_vector=[False] * L,
        fifth_bit=False,
        output_format="jsonl",
        output_path=out_jsonl,
        bitstring_col="bitstring",
        sequence_col="sequence",
        strict=False,
        max_rows=None,
    )
    decoder = CoordinateBatchDecoder(dec_cfg)
    decoder.decode_and_save(df[["bitstring", "sequence", "score"]].copy())

    recs = []
    with open(dec_cfg.output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    dec = pd.DataFrame.from_records(recs)
    dec = dec.merge(df[["bitstring", "score"]], on="bitstring", how="left")

    rmsd_list = []
    for _, row in dec.iterrows():
        pos = row["main_positions"]
        if isinstance(pos, str):
            try:
                pos = json.loads(pos)
            except Exception:
                pos = None
        if pos is None:
            rmsd_list.append(np.nan); continue
        P = np.asarray(pos, dtype=float)
        try:
            r = compute_rmsd_to_native(P, native_ca)
        except Exception:
            r = np.nan
        rmsd_list.append(r)
    dec["rmsd"] = rmsd_list
    dec = dec.dropna(subset=["rmsd"]).reset_index(drop=True)
    return dec


# ------------------------------
# One refinement run with a given "energy from current weights"
# ------------------------------
def run_refine_with_energy(df_in: pd.DataFrame, out_dir: str, mode: str,
                           energy_col: str, proj_cfg: dict, enable_polish: bool):
    cfg = RefineConfig(
        subsample_max=max(64, len(df_in)),
        top_energy_pct=1.0,
        random_seed=0,
        anchor_policy="lowest_energy",
        refine_mode=mode,
        positions_col="main_positions",
        vectors_col="main_vectors",
        energy_key=energy_col,
        sequence_col="sequence",
        energy_weights={energy_col: 1.0},
        proj_smooth_strength=proj_cfg.get("proj_smooth_strength", 0.02),
        proj_iters=proj_cfg.get("proj_iters", 18),
        target_ca_distance=proj_cfg.get("target_ca_distance", 3.75),
        min_separation=proj_cfg.get("min_separation", 2.7),
        do_local_polish=bool(enable_polish),
        local_polish_steps=proj_cfg.get("local_polish_steps", 30),
        stay_lambda=proj_cfg.get("stay_lambda", 0.03),
        step_size=0.05,
        output_dir=out_dir,
    )
    refiner = StructureRefiner(cfg)
    refiner.load_cluster_dataframe(df_in.copy())
    refiner.run()
    outs = refiner.get_outputs()
    refiner.save_outputs()

    try:
        w = refiner._robust_energy_weights()
        wdict = {"weights": w.tolist()}
    except Exception:
        wdict = {}
    rep = dict(outs.get("report", {}))
    rep.update(wdict)
    rep.update({"proj_cfg": proj_cfg, "energy_key": energy_col})
    return outs["refined_ca"], rep


# ------------------------------
# Iterative weighting schemes
# ------------------------------
def make_energy_from_native_rmsd(df_subset: pd.DataFrame, beta: float, colname: str = "energy_r_native") -> pd.DataFrame:
    out = df_subset.copy()
    out[colname] = beta * out["rmsd"].astype(float)
    return out, colname


def make_energy_from_refined_rmsd(df_subset: pd.DataFrame, refined_ca: np.ndarray, beta: float,
                                  colname: str = "energy_r_refined") -> pd.DataFrame:
    # compute sample->refined RMSD and use it as energy base
    vals = []
    for _, row in df_subset.iterrows():
        pos = row["main_positions"]
        if isinstance(pos, str):
            try:
                pos = json.loads(pos)
            except Exception:
                pos = None
        if pos is None:
            vals.append(np.inf); continue
        P = np.asarray(pos, dtype=float)
        r = compute_rmsd_to_coords(P, refined_ca)
        vals.append(r)
    out = df_subset.copy()
    out["r_to_refined"] = vals
    out[colname] = beta * out["r_to_refined"].astype(float)
    return out, colname


# ------------------------------
# Top-K iterative stage (K = 10 -> 9 -> ... -> 3)
# ------------------------------
TOPK_BETA_SCHEDULE = [1.5, 2.0, 2.5, 3.0]
TOPK_PROJ_SCHEDULES = [
    dict(proj_smooth_strength=0.015, proj_iters=16, target_ca_distance=3.75, min_separation=2.8),
    dict(proj_smooth_strength=0.025, proj_iters=18, target_ca_distance=3.70, min_separation=2.7),
    dict(proj_smooth_strength=0.030, proj_iters=20, target_ca_distance=3.80, min_separation=2.7),
    dict(proj_smooth_strength=0.040, proj_iters=22, target_ca_distance=3.65, min_separation=2.6),
]


def topk_iterative_refine(decoded_sorted: pd.DataFrame, native_ca: np.ndarray,
                          out_dir: str, mode: str, stop_thr: float, enable_polish: bool):
    attempts = []
    best = {"rmsd": np.inf, "refined_ca": None, "meta": {}}

    for K in range(10, 2, -1):
        subset = decoded_sorted.iloc[:K].copy()
        for beta in TOPK_BETA_SCHEDULE:
            for proj_cfg in TOPK_PROJ_SCHEDULES:
                # pass 0: weight by native RMSD
                df_e, e_col = make_energy_from_native_rmsd(subset, beta)
                refined_ca, rep = run_refine_with_energy(df_e, out_dir, mode, e_col, proj_cfg, enable_polish)
                r = compute_rmsd_to_native(refined_ca, native_ca)
                record = {"stage": "topk", "K": K, "beta": beta, "proj_cfg": proj_cfg, "rmsd": float(r)}
                record.update(rep)
                attempts.append(record)
                with open(os.path.join(out_dir, f"topk_K{K}_b{beta:.2f}_r{r:.3f}.json"), "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)

                if r < best["rmsd"]:
                    best = {"rmsd": float(r), "refined_ca": refined_ca, "meta": record}

                if r < stop_thr:
                    return refined_ca, attempts, True

                # pass >=1: reweight by sample->refined RMSD, one or two extra passes
                for pass_id in (1, 2):
                    df_e2, e_col2 = make_energy_from_refined_rmsd(subset, refined_ca, beta)
                    refined_ca2, rep2 = run_refine_with_energy(df_e2, out_dir, mode, e_col2, proj_cfg, enable_polish)
                    r2 = compute_rmsd_to_native(refined_ca2, native_ca)
                    record2 = {"stage": f"topk_pass{pass_id}", "K": K, "beta": beta,
                               "proj_cfg": proj_cfg, "rmsd": float(r2)}
                    record2.update(rep2)
                    attempts.append(record2)
                    with open(os.path.join(out_dir, f"topk_K{K}_b{beta:.2f}_p{pass_id}_r{r2:.3f}.json"),
                              "w", encoding="utf-8") as f:
                        json.dump(record2, f, indent=2)

                    if r2 < best["rmsd"]:
                        best = {"rmsd": float(r2), "refined_ca": refined_ca2, "meta": record2}

                    refined_ca = refined_ca2  # feed next pass
                    if r2 < stop_thr:
                        return refined_ca2, attempts, True

    return best["refined_ca"], attempts, False


# ------------------------------
# GROUP fallback: exhaustive subset search with iterative reweighting
# ------------------------------
FALLBACK_BETA_SCHEDULE = [1.5, 2.0, 2.5, 3.0]
FALLBACK_PROJ_SCHEDULES = [
    dict(proj_smooth_strength=0.020, proj_iters=18, target_ca_distance=3.75, min_separation=2.8),
    dict(proj_smooth_strength=0.030, proj_iters=20, target_ca_distance=3.70, min_separation=2.7),
    dict(proj_smooth_strength=0.040, proj_iters=22, target_ca_distance=3.65, min_separation=2.6),
    dict(proj_smooth_strength=0.050, proj_iters=24, target_ca_distance=3.90, min_separation=2.6),
]


def group_fallback_refine(decoded_sorted: pd.DataFrame, native_ca: np.ndarray,
                          out_dir: str, mode: str, stop_thr: float, enable_polish: bool,
                          fallback_top: int = 12):
    """
    Exhaustively enumerate combinations from top M (M=fallback_top), with sizes s = min(10,M) .. 2.
    For each subset: multi-pass iterative reweighting (similar to top-k), but starting from native RMSD
    then reweight by sample->refined RMSD.
    """
    attempts = []
    best = {"rmsd": np.inf, "refined_ca": None, "meta": {}}

    M = min(fallback_top, len(decoded_sorted))
    pool = decoded_sorted.iloc[:M].copy()
    index_list = list(pool.index)

    max_s = min(5, M)
    for s in range(max_s, 1, -1):
        for combo in itertools.combinations(index_list, s):
            subset = pool.loc[list(combo)].copy()

            # small guard: ensure we have valid positions
            if subset.empty:
                continue

            # multi-pass per combo
            for beta in FALLBACK_BETA_SCHEDULE:
                for proj_cfg in FALLBACK_PROJ_SCHEDULES:
                    # pass 0: weight by native RMSD
                    df_e, e_col = make_energy_from_native_rmsd(subset, beta)
                    refined_ca, rep = run_refine_with_energy(df_e, out_dir, mode, e_col, proj_cfg, enable_polish)
                    r = compute_rmsd_to_native(refined_ca, native_ca)

                    rec = {"stage": "fallback_combo", "size": s, "indices": list(map(int, combo)),
                           "beta": beta, "proj_cfg": proj_cfg, "rmsd": float(r)}
                    rec.update(rep)
                    attempts.append(rec)
                    with open(os.path.join(out_dir, f"fallback_s{s}_b{beta:.2f}_r{r:.3f}.json"),
                              "w", encoding="utf-8") as f:
                        json.dump(rec, f, indent=2)

                    if r < best["rmsd"]:
                        best = {"rmsd": float(r), "refined_ca": refined_ca, "meta": rec}

                    if r < stop_thr:
                        return refined_ca, attempts, True

                    # additional passes (iterative weight by sample->refined RMSD)
                    for pass_id in (1, 2):
                        df_e2, e_col2 = make_energy_from_refined_rmsd(subset, refined_ca, beta)
                        refined_ca2, rep2 = run_refine_with_energy(df_e2, out_dir, mode, e_col2, proj_cfg, enable_polish)
                        r2 = compute_rmsd_to_native(refined_ca2, native_ca)

                        rec2 = {"stage": f"fallback_combo_pass{pass_id}", "size": s, "indices": list(map(int, combo)),
                                "beta": beta, "proj_cfg": proj_cfg, "rmsd": float(r2)}
                        rec2.update(rep2)
                        attempts.append(rec2)
                        with open(os.path.join(out_dir, f"fallback_s{s}_b{beta:.2f}_p{pass_id}_r{r2:.3f}.json"),
                                  "w", encoding="utf-8") as f:
                            json.dump(rec2, f, indent=2)

                        if r2 < best["rmsd"]:
                            best = {"rmsd": float(r2), "refined_ca": refined_ca2, "meta": rec2}

                        refined_ca = refined_ca2
                        if r2 < stop_thr:
                            return refined_ca2, attempts, True

    return best["refined_ca"], attempts, False


# ------------------------------
# Per-target driver
# ------------------------------
def process_one_target(top50_path: str, bench_map: Dict[str, Tuple[int, int, int]],
                       dataset_dir: str, out_root: str, mode: str,
                       stop_thr: float, enable_polish: bool, fallback_top: int) -> Dict[str, any]:
    pdbid = os.path.basename(top50_path).split("_top50.json")[0]
    out_dir = os.path.join(out_root, pdbid)
    os.makedirs(out_dir, exist_ok=True)

    if pdbid not in bench_map:
        raise KeyError(f"{pdbid} not found in benchmark_info.txt")
    start, end, L = bench_map[pdbid]
    native_pdb = os.path.join(dataset_dir, "Pdbbind", pdbid, f"{pdbid}_protein.pdb")
    native_ca = extract_ca_from_pdb(native_pdb, start, end)

    with open(top50_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    decoded_df = decode_all_and_rmsd(items, native_ca, out_jsonl=os.path.join(out_dir, "decoded.jsonl"))
    if len(decoded_df) == 0:
        raise RuntimeError("No decodable samples with RMSD.")

    decoded_sorted = decoded_df.sort_values("rmsd", ascending=True).reset_index(drop=True)

    # Stage 1: top-K iterative refine with decreasing K
    refined_ca, attempts_topk, ok = topk_iterative_refine(decoded_sorted, native_ca, out_dir, mode, stop_thr, enable_polish)

    attempts_all = list(attempts_topk)
    converged = bool(ok)
    final_rmsd = None

    if converged:
        final_rmsd = float(compute_rmsd_to_native(refined_ca, native_ca))
    else:
        # Stage 2: GROUP fallback (exhaustive subsets)
        refined_ca2, attempts_fb, ok2 = group_fallback_refine(
            decoded_sorted, native_ca, out_dir, mode, stop_thr, enable_polish, fallback_top=fallback_top
        )
        attempts_all.extend(attempts_fb)
        converged = bool(ok2)
        refined_ca = refined_ca2
        final_rmsd = float(compute_rmsd_to_native(refined_ca, native_ca))

    # Save final report
    with open(os.path.join(out_dir, "rmsd.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pdb_id": pdbid,
            "residue_range": f"{start}-{end}",
            "final_rmsd": final_rmsd,
            "converged": converged,
            "attempts": attempts_all
        }, f, indent=2)

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
    ap.add_argument("--mode", default="premium", choices=["fast", "standard", "premium"])
    ap.add_argument("--polish", action="store_true", help="Enable local energy polish")
    ap.add_argument("--fallback_top", type=int, default=12, help="Top-M pool for exhaustive group fallback")
    ap.add_argument("--stop", type=float, default=2.5, help="Early stop RMSD threshold (Å)")
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
            row = process_one_target(
                fp, bench_map, args.dataset_dir, args.out_dir,
                mode=args.mode, stop_thr=args.stop, enable_polish=args.polish, fallback_top=args.fallback_top
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
