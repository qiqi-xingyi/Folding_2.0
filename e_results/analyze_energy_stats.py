# --*-- coding:utf-8 --*--
# @time: 10/25/25
# @Author: Yuqi Zhang
# @File: analyze_energy_stats.py
"""
Analyze energy statistics for a given QSAD energy file.
Input: <case_id>/energies.jsonl
Output: prints mean/std/min/max/range per energy component.
(No plotting version)
"""

import os
import json
import pandas as pd


# ========== User Config ==========
INPUT_PATH = "1m7y/energies.jsonl"
# =================================


def load_energy_jsonl(path: str) -> pd.DataFrame:
    """Load energies.jsonl as DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Energy file not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
    if not records:
        raise ValueError(f"No valid records found in {path}")
    return pd.DataFrame(records)


def analyze_energy(df: pd.DataFrame):
    """Print energy statistics."""
    cols = ["E_total", "E_geom", "E_steric", "E_bond", "E_mj"]
    print("=== Energy Statistics ===")
    for col in cols:
        if col not in df.columns:
            print(f"[!] Column missing: {col}")
            continue
        s = df[col].astype(float)
        print(
            f"{col:10s}  mean={s.mean():10.6f}  std={s.std():10.6f}  "
            f"min={s.min():10.6f}  max={s.max():10.6f}  range={(s.max()-s.min()):10.6f}"
        )
    print(f"\nSample count: {len(df)}")


def main():
    print("Loading:", INPUT_PATH)
    df = load_energy_jsonl(INPUT_PATH)
    analyze_energy(df)


if __name__ == "__main__":
    main()
