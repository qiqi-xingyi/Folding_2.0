# --*-- conding:utf-8 --*--
# @time:10/20/25 01:56
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test_postprocessing.py

#!/usr/bin/env python3
"""
Test script for the QSaD post-processing package.

This script:
  1. Loads the sampling output CSV file (samples_demo.csv);
  2. Runs entropy / ESS / coverage / bitwise / Hamming analyses;
  3. Saves summary CSV files into the ./analysis/ directory;
  4. Prints brief statistics to console for verification.
"""

import os
import pandas as pd
from pathlib import Path

from qsad_postproc import SamplingAnalyzer, AnalyzerConfig
from qsad_postproc.io import read_sampling_csv

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = "samples_demo.csv"
OUT_DIR = Path("analysis")
OUT_DIR.mkdir(exist_ok=True)
OUT_PREFIX = OUT_DIR / "qsad_demo"

# group_keys can include "beta", "label", etc. depending on your sampling CSV
CONFIG = AnalyzerConfig(
    n_qubits=None,           # infer from bitstrings
    group_keys=("beta",),    # aggregate by beta
)

if __name__ == '__main__':

    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------

    print(f"[*] Loading sampling data from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Cannot find file: {INPUT_FILE}")

    df = read_sampling_csv(INPUT_FILE, cfg=AnalyzerConfig(
        group_keys=("beta", "seed", "n_qubits", "label"),  # align with your schema
        normalize_bitstrings=True,
    ))
    print(f"  Loaded {len(df)} rows, columns = {list(df.columns)}")

    # ---------------------------------------------------------------------
    # Initialize analyzer
    # ---------------------------------------------------------------------

    analyzer = SamplingAnalyzer(df, cfg=CONFIG)
    print(f"[*] Initialized SamplingAnalyzer with {CONFIG.n_qubits or 'auto-detected'} qubits")

    # ---------------------------------------------------------------------
    # Run analyses
    # ---------------------------------------------------------------------

    print("[*] Computing summaries...")

    per_exp = analyzer.per_experiment_summary()
    per_group_mean = analyzer.per_group_aggregate("mean")
    per_group_median = analyzer.per_group_aggregate("median")
    bit_marginals = analyzer.bit_marginals()
    mode_hamming = analyzer.mode_and_hamming()
    topk = analyzer.topk_states(k=10)

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------

    print("[*] Writing summary CSV files...")
    out_map = write_many_csv({
        "per_experiment": per_exp,
        "per_group_mean": per_group_mean,
        "per_group_median": per_group_median,
        "bit_marginals": bit_marginals,
        "mode_hamming": mode_hamming,
        "top10": topk,
    }, str(OUT_PREFIX))

    for k, v in out_map.items():
        print(f"  [✓] {k:20s} -> {v}")

    # ---------------------------------------------------------------------
    # Print quick overview
    # ---------------------------------------------------------------------

    print("\n[*] Summary overview:")
    print(per_group_mean[["beta", "entropy_bits", "ess", "distinct_ratio"]].head())

    print("\n[*] Top-10 states (first few rows):")
    print(topk.head())

    print("\n[*] Bitwise marginals (first few rows):")
    print(bit_marginals.head())

    print("\n[✓] Post-processing complete. Results written to:")
    print(f"    {OUT_DIR.resolve()}")
