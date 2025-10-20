# --*-- conding:utf-8 --*--
# @time:10/20/25 01:55
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cli.py

from __future__ import annotations
import argparse
from pathlib import Path

from .config import AnalyzerConfig
from .analyzer import SamplingAnalyzer
from .io import write_many_csv

def main():
    p = argparse.ArgumentParser(description="QSaD post-processing CLI")
    p.add_argument("--input", type=str, required=True, help="Input CSV file (sampling results)")
    p.add_argument("--out-prefix", type=str, required=True, help="Output path prefix for summaries")
    p.add_argument("--group-keys", type=str, default="beta", help="Comma-separated group keys, e.g., 'beta,label'")
    p.add_argument("--n-qubits", type=int, default=None, help="Override n_qubits; if omitted, infer from bitstrings")
    p.add_argument("--topk", type=int, default=10, help="Top-k states per experiment")
    args = p.parse_args()

    gkeys = tuple([s.strip() for s in args.group_keys.split(",") if s.strip()])
    cfg = AnalyzerConfig(n_qubits=args.n_qubits, group_keys=gkeys)

    an = SamplingAnalyzer.from_csv(args.input, cfg=cfg)

    exp = an.per_experiment_summary()
    grp_mean = an.per_group_aggregate("mean")
    grp_median = an.per_group_aggregate("median")
    marg = an.bit_marginals()
    mode = an.mode_and_hamming()
    topk = an.topk_states(k=args.topk)

    out_map = write_many_csv({
        "per_experiment": exp,
        "per_group_mean": grp_mean,
        "per_group_median": grp_median,
        "bit_marginals": marg,
        "mode_hamming": mode,
        f"top{args.topk}": topk
    }, args.out_prefix)

    for k, v in out_map.items():
        print(f"[written] {k}: {v}")

if __name__ == "__main__":
    main()
