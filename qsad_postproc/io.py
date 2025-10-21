# --*-- conding:utf-8 --*--
# @time:10/20/25 01:55
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:io.py

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from qsad_postproc.config import AnalyzerConfig
from qsad_postproc.utils import normalize_bitstrings

EXPECTED_COLUMNS = [
    "L", "n_qubits", "shots", "beta", "seed", "label",
    "backend", "ibm_backend", "circuit_hash",
    "protein", "sequence",
    "bitstring", "count", "prob"
]

def read_sampling_csv(path: str, cfg: Optional[AnalyzerConfig] = None) -> pd.DataFrame:
    """
    Read QSaD sampling CSV aligned to the schema you provided, enforce dtypes,
    fill missing optional fields, and normalize bitstring length per row n_qubits.
    """
    cfg = cfg or AnalyzerConfig()

    df = pd.read_csv(path)

    # Ensure all expected columns exist; fill optional ones if missing.
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col in ("ibm_backend", "circuit_hash", "protein", "sequence", "label"):
                df[col] = np.nan
            else:
                raise ValueError(f"Missing required column '{col}' in input CSV: {path}")

    # Cast dtypes (tolerant casting)
    int_like = ["L", "n_qubits", "shots", "seed", "count"]
    float_like = ["beta", "prob"]
    for c in int_like:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype("float").astype("Int64")
        # Then cast down to int (safe) if no NaN remains
        if df[c].isna().any():
            raise ValueError(f"Column '{c}' contains NaN after coercion.")
        df[c] = df[c].astype(int)
    for c in float_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise ValueError(f"Column '{c}' contains NaN after coercion.")

    # Fill optional strings
    for c in ["backend", "ibm_backend", "circuit_hash", "protein", "sequence", "label"]:
        df[c] = df[c].astype(str).fillna("")

    # Normalize bitstrings per-row against n_qubits if requested
    if cfg.normalize_bitstrings:
        # If cfg.n_qubits is set, use it globally; otherwise use per-row n_qubits
        if cfg.n_qubits is not None:
            target_len = int(cfg.n_qubits)
            df["bitstring"] = normalize_bitstrings(df["bitstring"].astype(str).tolist(), target_len=target_len)
        else:
            # per-row: bitstring must match row's n_qubits
            # do it in vectorized-ish way by grouping rows with same n_qubits
            pieces = []
            for nq, grp in df.groupby("n_qubits", sort=False):
                bs = grp["bitstring"].astype(str).tolist()
                fixed = normalize_bitstrings(bs, target_len=int(nq))
                g2 = grp.copy()
                g2["bitstring"] = fixed
                pieces.append(g2)
            df = pd.concat(pieces, ignore_index=True)

    # Ensure probability column consistent with count/shots if prob missing or zero-sum
    if "prob" not in df.columns or df["prob"].isna().any():
        df["prob"] = df["count"] / df["shots"]
    else:
        # Normalize prob within (beta, seed, n_qubits, label) to avoid tiny drift
        gkeys = list(cfg.group_keys)
        df["prob"] = df.groupby(gkeys)["prob"].transform(lambda x: x / (x.sum() or 1.0))

    return df

def write_many_csv(dfs: Dict[str, pd.DataFrame], path_prefix: str) -> Dict[str, str]:
    """
    Write multiple DataFrames to CSV files with name pattern {prefix}_{key}.csv.
    Returns mapping key -> path.
    """
    paths = {}
    for key, df in dfs.items():
        out = f"{path_prefix}_{key}.csv"
        df.to_csv(out, index=False)
        paths[key] = out
    return paths
