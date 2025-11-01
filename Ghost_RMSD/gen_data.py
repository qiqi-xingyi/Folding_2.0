# --*-- conding:utf-8 --*--
# @time:11/1/25 01:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:gen_data.py


"""
Generate random bitstring samples for ghost-RMSD training.
- Input : Ghost_RMSD/benchmark_info.txt  (TSV with header)
- Output: Ghost_RMSD/training_data/samples_{pdb_id}_rand.csv
Schema:
L,n_qubits,shots,beta,seed,label,backend,ibm_backend,circuit_hash,
protein,sequence,group_id,bitstring,count,prob
"""

import os
import csv
import argparse
from pathlib import Path
import numpy as np

def read_benchmark_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            row = dict(zip(header, parts))
            rows.append(row)
    return rows

def gen_unique_bitstrings(bit_len, n, rng):
    """Generate n unique random bitstrings of length bit_len."""
    out = set()
    # vectorized bulk generation + top-up loop to guarantee uniqueness
    while len(out) < n:
        # generate in chunks; size scales with remaining needed samples
        need = n - len(out)
        chunk = max(need * 2, 4096)
        arr = rng.integers(0, 2, size=(chunk, bit_len), dtype=np.uint8)
        # join to strings
        for bits in arr:
            s = ''.join('1' if b else '0' for b in bits)
            out.add(s)
            if len(out) >= n:
                break
    return list(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Ghost_RMSD", help="Project root containing benchmark_info.txt")
    ap.add_argument("--index", default="benchmark_info.txt", help="TSV index file")
    ap.add_argument("--outdir", default="training_data", help="Output directory")
    ap.add_argument("--per_group", type=int, default=5000, help="Samples per protein group")
    ap.add_argument("--subtract_bits", type=int, default=5, help="Bit length = Number_of_qubits - subtract_bits")
    ap.add_argument("--shots", type=int, default=2000, help="Shots used to compute prob = count/shots")
    ap.add_argument("--beta", type=float, default=0.0, help="Beta metadata field")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed")
    ap.add_argument("--backend", default="ibm", help="Backend metadata field")
    ap.add_argument("--ibm_backend", default="ibm_strasbourg", help="IBM backend name")
    ap.add_argument("--label_prefix", default="qsad_ibm", help="Label prefix")
    ap.add_argument("--group_start", type=int, default=0, help="Starting group_id")
    ap.add_argument("--one_file", action="store_true", help="Write a single merged CSV instead of per-PDB files")
    args = ap.parse_args()

    rows = read_benchmark_tsv(args.index)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    header = [
        "L","n_qubits","shots","beta","seed","label","backend","ibm_backend","circuit_hash",
        "protein","sequence","group_id","bitstring","count","prob"
    ]

    # single merged file?
    merged_writer = None
    merged_fp = None
    if args.one_file:
        merged_fp = open(os.path.join(args.outdir, "samples_random_merged.csv"), "w", newline="", encoding="utf-8")
        merged_writer = csv.writer(merged_fp)
        merged_writer.writerow(header)

    base_rng = np.random.default_rng(args.seed)

    for gi, r in enumerate(rows, start=args.group_start):
        pdb_id = r["pdb_id"].strip()
        seq = r["Residue_sequence"].strip()
        try:
            L = int(r["Sequence_length"])
        except:
            L = len(seq)

        nq_raw = int(r["Number_of_qubits"])
        bit_len = nq_raw - args.subtract_bits
        if bit_len <= 0:
            raise ValueError(f"{pdb_id}: invalid bit length {bit_len} from Number_of_qubits={nq_raw} - {args.subtract_bits}")

        # per-group RNG to keep reproducibility but diversity across groups
        rng = np.random.default_rng(args.seed + gi * 9973)

        bitstrings = gen_unique_bitstrings(bit_len, args.per_group, rng)

        prob = 1.0 / float(args.shots)
        count = 1
        label = f"{args.label_prefix}_{pdb_id}_g{gi}"
        circuit_hash = ""  # placeholder as in your example

        # choose writer
        if args.one_file:
            writer = merged_writer
        else:
            out_path = os.path.join(args.outdir, f"samples_{pdb_id}_rand.csv")
            fp = open(out_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(fp)
            writer.writerow(header)

        for s in bitstrings:
            row = [
                L, bit_len, args.shots, args.beta, args.seed, label,
                args.backend, args.ibm_backend, circuit_hash,
                pdb_id, seq, gi, s, count, prob
            ]
            writer.writerow(row)

        if not args.one_file:
            fp.close()
            print(f"[OK] {pdb_id}: {args.per_group} samples -> {out_path} (bit_len={bit_len})")

    if merged_fp is not None:
        merged_fp.close()
        print(f"[OK] Merged file written to {merged_fp.name}")

if __name__ == "__main__":
    main()
