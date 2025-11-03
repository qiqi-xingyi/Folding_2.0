# --*-- conding:utf-8 --*--
# @time:11/2/25 21:25
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:get_result.py


import argparse
import glob
import json
import logging
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Import your provided classes
from qsadpp.coordinate_decoder import CoordinateDecoderConfig, CoordinateBatchDecoder
from analysis_reconstruction.structure_refine import RefineConfig, StructureRefiner

LOG = logging.getLogger("runner")
if not LOG.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def decode_positions(rows: pd.DataFrame,
                     bit_col: str = "bitstring",
                     seq_col: str = "sequence",
                     fifth_bit: bool = False) -> List[Dict[str, Any]]:
    """
    Decode bitstrings -> (main_positions) per row using your CoordinateBatchDecoder.
    Returns a list of dicts with keys: sequence, bitstring, main_positions.
    """
    out: List[Dict[str, Any]] = []
    for _, r in rows.iterrows():
        bitstring = str(r[bit_col])
        sequence = str(r[seq_col]) if seq_col in rows.columns else ""
        # length N must match the hot-vector length; here we ignore side chains
        hot = [False] * len(sequence)
        cfg = CoordinateDecoderConfig(
            side_chain_hot_vector=hot,
            fifth_bit=fifth_bit,
            output_format="jsonl",
            output_path="__discard__.jsonl",  # we won't use the file
            bitstring_col=bit_col,
            sequence_col=seq_col,
            strict=False,
        )
        decoder = CoordinateBatchDecoder(cfg)
        rec = decoder._decode_one(bitstring=bitstring, sequence=sequence)  # using internal helper for in-memory decode
        if rec is None:
            LOG.warning("Decode failed for bitstring (skipped).")
            continue
        out.append(rec)
    return out


def refine_cluster(decoded_recs: List[Dict[str, Any]],
                   rmsd_series: pd.Series,
                   out_dir: str,
                   sequence: str,
                   mode: str = "standard") -> Dict[str, Any]:
    """
    Build a DataFrame for StructureRefiner and run the refinement.
    We pass E_total = rmsd to satisfy the energy path inside the refiner.
    """
    df = pd.DataFrame({
        "main_positions": [rec["main_positions"] for rec in decoded_recs],
        "sequence": [sequence] * len(decoded_recs),
        "E_total": list(map(float, rmsd_series.values)),
    })
    cfg = RefineConfig(
        subsample_max=len(df),        # we already selected top_k
        top_energy_pct=1.0,           # use all rows we provided
        refine_mode=mode,
        positions_col="main_positions",
        energy_key="E_total",
        sequence_col="sequence",
        output_dir=out_dir,
        proj_enforce_bond=True,
        target_ca_distance=3.8,
        proj_iters=8,
        min_separation=2.8,
        do_local_polish=False,        # no external energy_fn provided
    )
    refiner = StructureRefiner(cfg)
    refiner.load_cluster_dataframe(df)
    refiner.run()
    refiner.save_outputs()
    outputs = refiner.get_outputs()
    return {
        "report": outputs.get("report", {}),
        "out_dir": out_dir,
    }


def process_one_csv(csv_path: str, out_root: str, top_k: int, refine_mode: str) -> Dict[str, Any]:
    """
    Process a single *_rmsd.csv file:
      - sort by rmsd
      - take top_k
      - decode -> refine -> save
    """
    base = os.path.basename(csv_path)
    name = os.path.splitext(base)[0]  # e.g., 1e2k_rmsd
    # Try to infer pdb_id from the first column or filename
    df = pd.read_csv(csv_path)
    required = {"bitstring", "rmsd"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} missing required columns {required}")

    # sorted ascending by RMSD
    df_sorted = df.sort_values("rmsd", ascending=True).reset_index(drop=True)
    df_top = df_sorted.iloc[:max(1, top_k)].copy()

    # infer sequence: use the first row's sequence; mixed sequences are not expected here
    sequence = str(df_top.iloc[0]["sequence"]) if "sequence" in df_top.columns else ""
    pdb_id = str(df_top.iloc[0]["pdb_id"]) if "pdb_id" in df_top.columns else name.replace("_rmsd", "")

    # decode lowest-K
    decoded = decode_positions(df_top, bit_col="bitstring", seq_col="sequence", fifth_bit=False)
    if len(decoded) == 0:
        LOG.warning("No decodable rows for %s; skipping.", csv_path)
        return {"pdb_id": pdb_id, "status": "no_decodable_rows"}

    # match RMSDs to decoded rows 1:1 in order (top set preserved)
    rmsd_used = df_top["rmsd"].iloc[:len(decoded)]

    # output dir: test_output/<pdb_id>
    out_dir = os.path.join(out_root, pdb_id)
    os.makedirs(out_dir, exist_ok=True)

    # refine
    res = refine_cluster(decoded, rmsd_used, out_dir, sequence, mode=refine_mode)
    rep = res["report"]
    rep.update({
        "pdb_id": pdb_id,
        "n_used": int(len(decoded)),
        "best_input_rmsd": float(np.min(rmsd_used.values)),
        "csv": csv_path,
        "out_dir": out_dir,
    })
    # also drop a small manifest for convenience
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    LOG.info("Finished %s -> %s | best_input_rmsd=%.3f Å",
             pdb_id, out_dir, rep["best_input_rmsd"])
    return rep


def main():
    ap = argparse.ArgumentParser(description="Refine lowest-K RMSD conformations into one fitted structure per target.")
    ap.add_argument("--in_dir", type=str, default="prepared_dataset", help="Directory containing *_rmsd.csv files.")
    ap.add_argument("--out_dir", type=str, default="test_output", help="Directory to write per-target outputs.")
    ap.add_argument("--top_k", type=int, default=10, help="Number of lowest-RMSD rows to use.")
    ap.add_argument("--mode", type=str, default="standard", choices=["fast", "standard", "premium"],
                    help="Refiner mode.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(args.in_dir, "*_rmsd.csv")))
    if not csvs:
        LOG.error("No *_rmsd.csv found under %s", args.in_dir)
        return

    summary: List[Dict[str, Any]] = []
    for p in csvs:
        try:
            rep = process_one_csv(p, args.out_dir, args.top_k, args.mode)
        except Exception as e:
            LOG.exception("Failed on %s: %s", p, e)
            rep = {"pdb_id": os.path.basename(p), "status": f"error: {e}", "csv": p}
        summary.append(rep)

    # save summary
    summary_path = os.path.join(args.out_dir, "qsad_rmsd_summary.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    LOG.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
