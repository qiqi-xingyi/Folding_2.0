# --*-- conding:utf-8 --*--
# @time:10/25/25 20:16
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:analyze_energy_stats.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ========== User Config ==========
INPUT_PATH = "e_results/1m7y/energies.jsonl"
SHOW_PLOT = True  # Set False if running on headless server
# =================================

def load_energy_jsonl(path: str) -> pd.DataFrame:
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
    df = pd.DataFrame(records)
    return df


def analyze_energy(df: pd.DataFrame):
    cols = ["E_total", "E_geom", "E_steric", "E_bond", "E_mj"]
    print("=== Energy Statistics ===")
    for col in cols:
        if col not in df.columns:
            print(f"[!] Column missing: {col}")
            continue
        s = df[col].astype(float)
        print(f"{col:10s}  mean={s.mean():10.6f}  std={s.std():10.6f}  "
              f"min={s.min():10.6f}  max={s.max():10.6f}  range={(s.max()-s.min()):10.6f}")

    print("\nSample count:", len(df))
    if SHOW_PLOT:
        plot_energy_hist(df, cols)


def plot_energy_hist(df: pd.DataFrame, cols):
    plt.figure(figsize=(10, 6))
    for col in cols:
        if col not in df.columns:
            continue
        plt.hist(df[col].astype(float), bins=50, alpha=0.6, label=col)
    plt.xlabel("Energy value")
    plt.ylabel("Frequency")
    plt.title("Energy Distributions (per component)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy_distributions.png", dpi=200)
    print("[+] Histogram saved as: energy_distributions.png")
    if SHOW_PLOT:
        plt.show()


def main():
    print("Loading:", INPUT_PATH)
    df = load_energy_jsonl(INPUT_PATH)
    analyze_energy(df)


if __name__ == "__main__":
    main()
