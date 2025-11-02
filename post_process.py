# --*-- conding:utf-8 --*--
# @time:10/21/25 14:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:post_process.py

# process_all.py
# Batch runner: iterate all subfolders under ./quantum_data and run the QSAD post-processing.
# Results go to ./pp_result/<pdbid>; an aggregate summary.csv/jsonl is produced at the end.

import os
import json
import csv
import traceback
from typing import List, Dict, Any

from qsadpp.orchestrator import OrchestratorConfig, PipelineOrchestrator
from qsadpp.io_reader import ReaderOptions
from qsadpp.feature_calculator import FeatureConfig

BASE_DIR = "./quantum_data"
OUT_ROOT = "./pp_result"

# ---- Orchestrator defaults you provided ----
def make_cfg(pdb_dir: str, out_dir: str) -> OrchestratorConfig:
    return OrchestratorConfig(
        pdb_dir=pdb_dir,
        reader_options=ReaderOptions(
            chunksize=100_000,
            strict=True,
            categorize_strings=True,
            include_all_csv=False,
        ),
        fifth_bit=False,
        out_dir=out_dir,
        compute_features=True,
        feature_from="decoded",
        combined_feature_name="features.jsonl",
        feature_config=FeatureConfig(
            output_format="jsonl",
        ),
    )

def discover_targets(base_dir: str) -> List[str]:
    items = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            items.append(path)
    return items

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("pdb_id,status,message\n")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def main():
    ensure_dir(OUT_ROOT)
    targets = discover_targets(BASE_DIR)
    if not targets:
        print(f"[WARN] No subfolders found under {BASE_DIR}")
        return

    aggregate: List[Dict[str, Any]] = []

    for pdb_path in targets:
        pdb_id = os.path.basename(pdb_path.rstrip("/"))
        out_dir = os.path.join(OUT_ROOT, pdb_id)
        ensure_dir(out_dir)

        print(f"==> Processing {pdb_id}")
        try:
            cfg = make_cfg(pdb_path, out_dir)
            runner = PipelineOrchestrator(cfg)
            summary = runner.run()  # expected to be dict-like
            print(f"[OK] {pdb_id}")

            row = {
                "pdb_id": pdb_id,
                "status": "ok",
                "out_dir": out_dir,
            }
            if isinstance(summary, dict):
                # Flatten a few common keys if present
                for k in ("num_decoded", "num_energy", "num_feature", "time_sec"):
                    if k in summary:
                        row[k] = summary[k]
            aggregate.append(row)

            # Also drop a per-target summary.json
            with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary if isinstance(summary, dict) else {"summary": str(summary)},
                          f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[FAIL] {pdb_id}: {e}")
            aggregate.append({
                "pdb_id": pdb_id,
                "status": "fail",
                "message": str(e),
                "out_dir": out_dir,
            })
            # Keep a traceback file for debugging
            with open(os.path.join(out_dir, "error.log"), "w", encoding="utf-8") as f:
                f.write("".join(traceback.format_exc()))

    # Write aggregate reports
    write_csv(os.path.join(OUT_ROOT, "summary.csv"), aggregate)
    write_jsonl(os.path.join(OUT_ROOT, "summary.jsonl"), aggregate)
    print(f"\nAll done. Summary written to {OUT_ROOT}/summary.csv and summary.jsonl")

if __name__ == "__main__":
    main()
