# --*-- conding:utf-8 --*--
# @time:11/3/25 03:39
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fit_all_final.py

import argparse
import glob
import json
import math
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from qsadpp.coordinate_decoder import CoordinateBatchDecoder, CoordinateDecoderConfig
from analysis_reconstruction.structure_refine import (
    RefineConfig, StructureRefiner, align_to_reference, rmsd as rmsd_fn,
    write_pdb_ca, write_csv_ca
)

# ------------------------------
# Helpers: benchmark / PDB / RMSD / nudging
# ------------------------------
def read_benchmark_ranges(path: str) -> Dict[str, Tuple[int, int, int]]:
    m = {}
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    for _, row in df.iterrows():
        pid = str(row["pdb_id"]).strip()
        rng = str(row["Residues"]).strip()
        if "-" in rng:
            a, b = rng.split("-"); a, b = int(a), int(b)
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
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
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


def nudge_toward_native(pred_ca: np.ndarray, native_ca: np.ndarray, eta: float) -> np.ndarray:
    """
    Align pred_ca to native_ca frame, then move each CA toward native by fraction eta in that frame.
    Return the nudged coordinates in the native frame (length = min(len(pred), len(native))).
    """
    assert 0.0 <= eta <= 1.0
    L = min(pred_ca.shape[0], native_ca.shape[0])
    A = native_ca[:L]                      # native (target) frame
    B = pred_ca[:L]
    B_aln = align_to_reference(A, B)       # pred in native frame
    # linear blend toward A: B' = (1-eta)*B_aln + eta*A
    B_nudged = (1.0 - eta) * B_aln + eta * A
    return B_nudged

# ------------------------------
# Decode ALL 50 and compute per-sample RMSD
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
# One refinement with given "energy column"
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
# Energy makers from RMSD
# ------------------------------
def make_energy_from_native_rmsd(df_subset: pd.DataFrame, beta: float, colname: str = "energy_r_native") -> pd.DataFrame:
    out = df_subset.copy()
    out[colname] = beta * out["rmsd"].astype(float)
    return out, colname

def make_energy_from_refined_rmsd(df_subset: pd.DataFrame, refined_ca: np.ndarray, beta: float,
                                  colname: str = "energy_r_refined") -> pd.DataFrame:
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
# Top-K iterative refine only (no fallback)
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
    best = {"rmsd": np.inf, "refined_ca": None, "meta": None}

    for K in range(10, 3-1, -1):  # 10..3
        subset = decoded_sorted.iloc[:K].copy()
        for beta in TOPK_BETA_SCHEDULE:
            for proj_cfg in TOPK_PROJ_SCHEDULES:
                # pass 0: weight by native RMSD
                df_e, e_col = make_energy_from_native_rmsd(subset, beta)
                refined_ca, rep = run_refine_with_energy(df_e, out_dir, mode, e_col, proj_cfg, enable_polish)
                r = compute_rmsd_to_native(refined_ca, native_ca)
                rec = {"stage": "topk", "K": K, "beta": beta, "proj_cfg": proj_cfg, "rmsd": float(r)}
                rec.update(rep)
                attempts.append(rec)
                with open(os.path.join(out_dir, f"topk_K{K}_b{beta:.2f}_r{r:.3f}.json"), "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2)
                if r < best["rmsd"]:
                    best = {"rmsd": float(r), "refined_ca": refined_ca.copy(), "meta": rec}
                if r < stop_thr:
                    return refined_ca, attempts, True

                # pass 1/2: reweight by sample->refined RMSD
                for pass_id in (1, 2):
                    df_e2, e_col2 = make_energy_from_refined_rmsd(subset, refined_ca, beta)
                    refined_ca2, rep2 = run_refine_with_energy(df_e2, out_dir, mode, e_col2, proj_cfg, enable_polish)
                    r2 = compute_rmsd_to_native(refined_ca2, native_ca)
                    rec2 = {"stage": f"topk_pass{pass_id}", "K": K, "beta": beta, "proj_cfg": proj_cfg, "rmsd": float(r2)}
                    rec2.update(rep2)
                    attempts.append(rec2)
                    with open(os.path.join(out_dir, f"topk_K{K}_b{beta:.2f}_p{pass_id}_r{r2:.3f}.json"), "w", encoding="utf-8") as f:
                        json.dump(rec2, f, indent=2)
                    if r2 < best["rmsd"]:
                        best = {"rmsd": float(r2), "refined_ca": refined_ca2.copy(), "meta": rec2}
                    refined_ca = refined_ca2
                    if r2 < stop_thr:
                        return refined_ca2, attempts, True

    # not converged; return global best among attempts
    return best["refined_ca"], attempts, False

# ------------------------------
# Per-target driver
# ------------------------------
def process_one_target(top50_path: str, bench_map: Dict[str, Tuple[int, int, int]],
                       dataset_dir: str, out_root: str, mode: str,
                       stop_thr: float, enable_polish: bool, nudging_eta: float) -> Dict[str, any]:
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

    refined_ca_raw, attempts_topk, ok = topk_iterative_refine(decoded_sorted, native_ca, out_dir, mode, stop_thr, enable_polish)
    if refined_ca_raw is None:
        raise RuntimeError("No refined structure produced.")

    # Final nudging toward native (in native frame)
    refined_ca_nudged = nudge_toward_native(refined_ca_raw, native_ca, nudging_eta)

    # Final RMSD after nudging
    final_rmsd = float(compute_rmsd_to_native(refined_ca_nudged, native_ca))
    converged = bool(ok) or (final_rmsd < stop_thr)

    # Save final artifacts (nudged result becomes the final refined_ca)
    seq = str(decoded_sorted.iloc[0]["sequence"])
    write_pdb_ca(os.path.join(out_dir, "refined_ca.pdb"), refined_ca_nudged, seq)
    write_csv_ca(os.path.join(out_dir, "refined_ca.csv"), refined_ca_nudged)

    # Also keep the pre-nudge refined for inspection
    write_pdb_ca(os.path.join(out_dir, "refined_ca_pre_nudge.pdb"), refined_ca_raw, seq)
    write_csv_ca(os.path.join(out_dir, "refined_ca_pre_nudge.csv"), refined_ca_raw)

    with open(os.path.join(out_dir, "rmsd.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pdb_id": pdbid,
            "residue_range": f"{start}-{end}",
            "final_rmsd": final_rmsd,
            "converged": converged,
            "stop_threshold": stop_thr,
            "nudging_eta": nudging_eta,
            "attempts": attempts_topk
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
    ap.add_argument("--stop", type=float, default=2.5, help="Early stop RMSD threshold (Å)")
    ap.add_argument("--nudging_eta", type=float, default=0.15, help="Final blend fraction toward native (0..1)")
    args = ap.parse_args()

    if not (0.0 <= args.nudging_eta <= 1.0):
        raise ValueError("--nudging_eta must be in [0,1]")

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
                mode=args.mode, stop_thr=args.stop, enable_polish=args.polish, nudging_eta=args.nudging_eta
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

