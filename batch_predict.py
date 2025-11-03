# --*-- coding:utf-8 --*--
# @time:11/2/25 23:59 (patched 11/3/25)
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:batch_predict.py
#
# Batch runner for Ghost_RMSD/predict.py over prepared_dataset/*_grn_input.jsonl

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # macOS OpenMP duplication workaround

import argparse
import csv
import subprocess
from pathlib import Path

def resolve_ckpt_path(ckpt_arg: str, predict_py: Path) -> str:
    p = Path(ckpt_arg)
    if p.exists():
        return str(p)
    alt = predict_py.parent / ckpt_arg
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(f"Cannot find checkpoint at '{ckpt_arg}' or '{alt}'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict_py", type=str, default="Ghost_RMSD/predict.py",
                    help="Path to predict.py")
    ap.add_argument("--inputs_glob", type=str, default="prepared_dataset/*_grn_input.jsonl",
                    help="Glob for input jsonl files")
    ap.add_argument("--ckpt", type=str, default="Ghost_RMSD/checkpoints_full/grn_best.pt",
                    help="Checkpoint path (relative to CWD or predict.py dir)")
    ap.add_argument("--out_dir", type=str, default="predictions",
                    help="Output directory for per-PDB CSVs")
    ap.add_argument("--device", type=str, default="auto", help="cpu|cuda|mps|auto")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--score_mode", type=str, default="expected_rel",
                    choices=["prob_rel3", "logit_rel3", "expected_rel"])
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    predict_py = Path(args.predict_py).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(Path().glob(args.inputs_glob))
    if not inputs:
        print(f"[WARN] No inputs matched: {args.inputs_glob}")
        return

    ckpt_path = resolve_ckpt_path(args.ckpt, predict_py)

    top1_rows = []
    for inp in inputs:
        pdb_id = inp.stem.replace("_grn_input", "")
        out_csv = out_dir / f"{pdb_id}_pred.csv"

        cmd = [
            "python", str(predict_py),
            "--ckpt", ckpt_path,
            "--input_jsonl", str(inp),
            "--out_csv", str(out_csv),
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--score_mode", args.score_mode,
            "--topk", str(args.topk),
            "--allow_omp_dup",
        ]
        print("[RUN]", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[ERROR] {pdb_id} failed:")
            print(res.stdout)
            print(res.stderr)
            continue
        else:
            print(f"[OK] {pdb_id} -> {out_csv}")

        # Collect top-1 (rank_in_group == 1) for (pdb_id, group_id=0)
        try:
            with out_csv.open("r", encoding="utf-8") as f:
                header = f.readline().rstrip("\n").split(",")
                col_idx = {h: i for i, h in enumerate(header)}
                need = ["pdb_id", "bitstring", "sequence", "score", "rank_in_group", "group_id"]
                if not all(c in col_idx for c in need):
                    continue
                top_line = None
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    try:
                        if parts[col_idx["pdb_id"]] == pdb_id and int(parts[col_idx["rank_in_group"]]) == 1:
                            top_line = parts
                            break
                    except Exception:
                        continue
                if top_line:
                    top1_rows.append({
                        "pdb_id": top_line[col_idx["pdb_id"]],
                        "bitstring": top_line[col_idx["bitstring"]],
                        "sequence": top_line[col_idx["sequence"]],
                        "score": top_line[col_idx["score"]],
                        "pred_csv": str(out_csv),
                    })
        except Exception:
            pass

    summary_path = out_dir / "summary_top1.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "bitstring", "sequence", "score", "pred_csv"])
        w.writeheader()
        for r in top1_rows:
            w.writerow(r)

    print(f"[DONE] Wrote {len(top1_rows)} top-1 rows to {summary_path.resolve()}")

if __name__ == "__main__":
    main()
