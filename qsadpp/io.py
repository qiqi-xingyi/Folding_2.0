# --*-- conding:utf-8 --*--
# @time:10/21/25 13:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:io.py

"""
I/O utilities for QSAD post-processing.

Responsibilities:
- Load one or multiple CSV files of raw samples.
- Normalize schema and dtypes.
- Group by metadata keys and aggregate counts to probabilities.
- Provide simple iterators over groups.

This module intentionally has no heavy dependencies beyond pandas/numpy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -------------------------------
# Default group keys (can be overridden by caller)
# -------------------------------
GROUP_KEYS: Tuple[str, ...] = (
    "protein",
    "sequence",
    "label",
    "backend",
    "ibm_backend",
    "beta",
    "seed",
    "circuit_hash",
)

# Columns expected in the raw CSVs
RAW_COLUMNS: Tuple[str, ...] = (
    "L",
    "n_qubits",
    "shots",
    "beta",
    "seed",
    "label",
    "backend",
    "ibm_backend",
    "circuit_hash",
    "protein",
    "sequence",
    "bitstring",
    "count",
    "prob",  # optional; will be recomputed if missing/inconsistent
)


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes and ensure required columns exist.

    - Missing optional "prob" will be filled later.
    - Use pandas StringDtype to preserve NA (avoid turning NA into the literal "nan").
    """
    missing = set(RAW_COLUMNS) - set(df.columns)
    # "prob" can be missing; allow it
    missing.discard("prob")
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # String-like columns
    str_cols = [
        "label",
        "backend",
        "ibm_backend",
        "circuit_hash",
        "protein",
        "sequence",
        "bitstring",
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    # Integer-like columns
    int_cols = ["L", "n_qubits", "shots", "seed"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Floats
    if "beta" in df.columns:
        df["beta"] = pd.to_numeric(df["beta"], errors="coerce")

    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    if "prob" in df.columns:
        df["prob"] = pd.to_numeric(df["prob"], errors="coerce")

    return df


def read_samples(paths: Sequence[Path | str]) -> pd.DataFrame:
    """
    Read one or more CSV files and vertically concatenate.

    Parameters
    ----------
    paths : list of str/Path
        CSV file paths. Globs are allowed; non-existing paths raise.

    Returns
    -------
    pd.DataFrame
        Raw concatenated samples with normalized dtypes.
    """
    files: List[Path] = []
    for p in paths:
        pth = Path(p)
        if any(ch in str(pth) for ch in "*?[]"):
            files.extend(sorted(Path().glob(str(pth))))
        else:
            if not pth.exists():
                raise FileNotFoundError(str(pth))
            files.append(pth)

    if not files:
        raise FileNotFoundError("No input files matched.")

    frames: List[pd.DataFrame] = []
    for f in files:
        # Force bitstring to be read as string to preserve leading zeros
        df = pd.read_csv(f, dtype={"bitstring": "string"})
        df = _coerce_columns(df)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def aggregate_counts_to_prob(
    df: pd.DataFrame,
    group_keys: Sequence[str] = GROUP_KEYS,
    validate_shots: bool = True,
) -> pd.DataFrame:
    """
    Aggregate duplicate (group, bitstring) rows and compute probabilities.

    For each group (by group_keys):
      - Drop rows with invalid/NA bitstrings.
      - Left-pad bitstrings with zeros to the group's n_qubits (if available) or to the max length in group.
      - Sum counts per bitstring.
      - Compute prob = count / sum(count) within group.

    Returns a tidy frame with columns:
      group_keys + ["bitstring", "count", "prob"] + ("shots" if present and unique per group)

    Other metadata columns like L, n_qubits are carried via groupby aggregation
    if they are constant within a group; otherwise the aggregated values will be NaN.
    """
    df = df.copy()

    def _normalize_bits_group(g: pd.DataFrame) -> pd.DataFrame:
        if "bitstring" not in g.columns:
            return g

        # Keep only valid binary strings; drop NA and invalid rows
        bs = g["bitstring"]
        mask = bs.notna() & bs.str.fullmatch(r"[01]+")
        g = g.loc[mask].copy()
        if g.empty:
            return g

        # Determine target width
        if "n_qubits" in g.columns and g["n_qubits"].notna().any():
            width_val = pd.to_numeric(g["n_qubits"], errors="coerce").max()
            if np.isfinite(width_val) and width_val > 0:
                width = int(width_val)
            else:
                width = int(g["bitstring"].str.len().max())
        else:
            width = int(g["bitstring"].str.len().max())

        # Zero-fill to width
        g["bitstring"] = g["bitstring"].astype("string").str.zfill(width)
        return g

    # Apply per-group normalization; pass include_groups to .apply (not to .groupby)
    gb_obj = df.groupby(list(group_keys), dropna=False, group_keys=False)
    try:
        # pandas >= 2.2
        df = gb_obj.apply(_normalize_bits_group, include_groups=True)
    except TypeError:
        # older pandas
        df = gb_obj.apply(_normalize_bits_group)

    # Sum counts per (group, bitstring)
    agg_cols = list(group_keys) + ["bitstring"]
    gb = df.groupby(agg_cols, dropna=False, as_index=False)["count"].sum()

    # Attach shots if available: if multiple shots per group, keep NaN
    if "shots" in df.columns:
        shots_df = df.groupby(list(group_keys), dropna=False)["shots"].nunique().reset_index(name="_nshots")
        shots_val = df.groupby(list(group_keys), dropna=False)["shots"].max().reset_index(name="shots")
        gb = gb.merge(shots_df, on=list(group_keys), how="left").merge(shots_val, on=list(group_keys), how="left")
        gb.loc[gb["_nshots"] != 1, "shots"] = np.nan
        gb = gb.drop(columns=["_nshots"])

    # Compute prob by normalizing counts within each group
    gb["prob"] = gb["count"] / gb.groupby(list(group_keys), dropna=False)["count"].transform("sum")

    # Optional validation: sum(count) close to shots
    if validate_shots and "shots" in gb.columns:
        check = gb.groupby(list(group_keys), dropna=False).agg(
            shots=("shots", "first"),
            sum_count=("count", "sum"),
        ).reset_index()
        check["rel_err"] = np.where(
            check["shots"].notna(),
            np.abs(check["sum_count"] - check["shots"]) / check["shots"].replace(0, np.nan),
            np.nan,
        )
        bad = check[check["rel_err"] > 0.05]
        if not bad.empty:
            # Attach for debugging if caller catches the exception
            bad_groups = bad.assign(_reason="sum(count) deviates from shots by >5%")
            bad_groups_attr = getattr(gb, "__dict__", None)
            if isinstance(bad_groups_attr, dict):
                gb.__dict__["bad_groups"] = bad_groups
            raise RuntimeError("Sum(count) deviates from shots by >5% for some groups.")

    return gb


def iter_groups(
    df: pd.DataFrame,
    group_keys: Sequence[str] = GROUP_KEYS,
) -> Iterator[Tuple[Mapping[str, object], pd.DataFrame]]:
    """
    Iterate (group_metadata, group_df) over groups.

    group_metadata is a dict mapping key->value for the group.
    group_df contains columns: group_keys + [bitstring, count, prob, (shots?)]
    """
    for gvals, gdf in df.groupby(list(group_keys), dropna=False):
        meta = {k: v for k, v in zip(group_keys, gvals)}
        yield meta, gdf.reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Save a DataFrame to Parquet (snappy)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def save_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Save a DataFrame to CSV (utf-8)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

