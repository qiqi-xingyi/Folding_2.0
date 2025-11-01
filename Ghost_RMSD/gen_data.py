# --*-- conding:utf-8 --*--
# @time:11/1/25 01:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:gen_data.py

# -- coding: utf-8 --
# @File: gen_random_samples.py
# @Author: Yuqi Zhang
# @Desc: Generate random bitstring samples for Ghost RMSD dataset

import os
import csv
import numpy as np
from pathlib import Path

# ==============================================================
#                   CONFIGURATION
# ==============================================================

INDEX_FILE = "benchmark_info.txt"
OUTPUT_DIR = "training_data"
PER_GROUP = 5000
SUBTRACT_BITS = 5
SHOTS = 2000                          # prob = count / shots
BETA = 0.0
SEED = 42
BACKEND = "ibm"
IBM_BACKEND = "ibm_strasbourg"
LABEL_PREFIX = "qsad_ibm"

# ==============================================================
#                       CORE FUNCTIONS
# ==============================================================

def read_benchmark_info(path):
    """读取 benchmark_info.txt 文件"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            rows.append(dict(zip(header, parts)))
    return rows


def gen_unique_bitstrings(bit_len, n, rng):
    """生成 n 条唯一的随机比特串"""
    out = set()
    while len(out) < n:
        need = n - len(out)
        arr = rng.integers(0, 2, size=(need, bit_len), dtype=np.uint8)
        for bits in arr:
            s = ''.join('1' if b else '0' for b in bits)
            out.add(s)
            if len(out) >= n:
                break
    return list(out)


# ==============================================================
#                          MAIN LOGIC
# ==============================================================

def main():
    root = Path(__file__).resolve().parent
    index_path = root / INDEX_FILE
    outdir = root / OUTPUT_DIR
    outdir.mkdir(exist_ok=True)

    rows = read_benchmark_info(index_path)
    print(f"[INFO] Loaded {len(rows)} benchmark entries from {index_path.name}")

    header = [
        "L","n_qubits","shots","beta","seed","label","backend",
        "ibm_backend","circuit_hash","protein","sequence",
        "group_id","bitstring","count","prob"
    ]

    base_rng = np.random.default_rng(SEED)

    for gi, r in enumerate(rows):
        pdb_id = r["pdb_id"].strip()
        seq = r["Residue_sequence"].strip()
        try:
            L = int(r["Sequence_length"])
        except:
            L = len(seq)

        n_qubits_raw = int(r["Number_of_qubits"])
        bit_len = n_qubits_raw - SUBTRACT_BITS
        if bit_len <= 0:
            print(f"[WARN] Skip {pdb_id}: invalid bit length {bit_len}")
            continue

        rng = np.random.default_rng(SEED + gi * 9973)
        bitstrings = gen_unique_bitstrings(bit_len, PER_GROUP, rng)

        prob = 1.0 / SHOTS
        count = 1
        label = f"{LABEL_PREFIX}_{pdb_id}_g{gi}"
        circuit_hash = ""

        out_path = outdir / f"samples_{pdb_id}_rand.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            for s in bitstrings:
                row = [
                    L, bit_len, SHOTS, BETA, SEED, label, BACKEND,
                    IBM_BACKEND, circuit_hash, pdb_id, seq,
                    gi, s, count, prob
                ]
                writer.writerow(row)

        print(f"[OK] {pdb_id}: {PER_GROUP} samples generated -> {out_path.name}")

    print(f"\n✅ All done! Generated {len(rows)} CSV files in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
