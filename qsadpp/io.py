# --*-- coding:utf-8 --*--
# @time: 10/22/25
# @Author: Yuqi Zhang (edited by ChatGPT)
# @File: io.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd


# ----------------------------
# Public exports
# ----------------------------

__all__ = [
    "find_csv_files",
    "load_sampling_csv",
    "load_all_samples",
    "ensure_standard_columns",
    "expand_bitstrings",
    "normalize_counts_to_prob_within",
    "coerce_numeric",
    "safe_read_table",
    "load_mj_matrix",
]


# ----------------------------
# Constants
# ----------------------------

# Known columns commonly seen in your pipeline (based on your logs)
STANDARD_GROUP_COLS: List[str] = [
    "L", "n_qubits", "shots", "beta", "seed", "label",
    "backend", "ibm_backend", "circuit_hash",
    "protein", "sequence",
]

# Default probability-like and count-like column names
PROB_COL_CANDIDATES = ("q_prob", "prob", "p")
COUNT_COL_CANDIDATES = ("count", "counts", "n")


# ----------------------------
# Filesystem helpers
# ----------------------------

def find_csv_files(
    input_dir: str | os.PathLike,
    recursive: bool = False,
    suffixes: Sequence[str] = (".csv",)
) -> List[Path]:
    """
    Find CSV files under a directory.

    Parameters
    ----------
    input_dir : str | Path
        Directory to search.
    recursive : bool
        Whether to recurse into subdirectories.
    suffixes : Sequence[str]
        File suffixes to match.

    Returns
    -------
    List[Path]
        Sorted list of CSV paths.
    """
    base = Path(input_dir).expanduser().resolve()
    if not base.exists():
        return []

    files: List[Path] = []
    if recursive:
        for s in suffixes:
            files.extend(base.rglob(f"*{s}"))
    else:
        for s in suffixes:
            files.extend(base.glob(f"*{s}"))

    # Deduplicate and sort for determinism
    uniq = sorted({p.resolve() for p in files})
    return uniq


# ----------------------------
# Data loading
# ----------------------------

def safe_read_table(path: str | os.PathLike, **kwargs) -> pd.DataFrame:
    """
    Read a CSV/TSV robustly with pandas.

    - If the file extension is .tsv/.txt, defaults to sep='\\t' or whitespace if not given.
    - If 'dtype' is passed, it is respected; otherwise, avoid aggressive dtype inference.

    Returns an empty dataframe if the path does not exist.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return pd.DataFrame()

    ext = p.suffix.lower()
    if "sep" not in kwargs:
        if ext in {".tsv"}:
            kwargs["sep"] = "\t"
        elif ext in {".txt"}:
            # Use whitespace by default for matrices like MJ tables
            kwargs["sep"] = r"\s+"

    try:
        df = pd.read_csv(p, **kwargs)
    except Exception:
        # Fallback try: minimal inference
        kwargs2 = dict(kwargs)
        kwargs2.pop("dtype", None)
        df = pd.read_csv(p, **kwargs2)
    return df


def load_sampling_csv(path: str | os.PathLike) -> pd.DataFrame:
    """
    Load a single sampling CSV.

    Expectations (best-effort; robust to missing columns):
    - Must have a 'bitstring' column or explicit bit columns (b0..b{n-1}).
    - May have 'count' or 'prob'/'q_prob'.
    - Other metadata columns (L, n_qubits, shots, beta, seed, label, backend,
      ibm_backend, circuit_hash, protein, sequence) are optional.

    Returns
    -------
    pd.DataFrame
        DataFrame with minimally cleaned columns.
    """
    df = safe_read_table(path)
    if df.empty:
        return df

    # Trim column names
    df.columns = [str(c).strip() for c in df.columns]

    # Common cleanup for bitstring column
    if "bitstring" in df.columns:
        df["bitstring"] = df["bitstring"].astype(str).str.strip()

    # Coerce numeric types for known numeric columns if present
    numeric_candidates = [
        "L", "n_qubits", "shots", "beta", "seed",
        "count", "counts", "n",
        "prob", "q_prob", "p",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    return df


def load_all_samples(
    input_dir: str | os.PathLike,
    recursive: bool = False,
    on_empty: str = "empty"
) -> pd.DataFrame:
    """
    Load and concatenate all sampling CSVs under a directory.

    Parameters
    ----------
    input_dir : str | Path
        The directory to search.
    recursive : bool
        Whether to search recursively.
    on_empty : {'empty', 'error'}
        Behavior when no CSVs found.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe with a 'source_file' column indicating provenance.
    """
    paths = find_csv_files(input_dir, recursive=recursive)
    if not paths:
        if on_empty == "error":
            raise FileNotFoundError(f"No CSV files found under: {input_dir}")
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = load_sampling_csv(p)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(p)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Standard cleaning pass
    out = ensure_standard_columns(out)
    return out


# ----------------------------
# Column normalization & helpers
# ----------------------------

def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure commonly used columns exist with sane defaults.
    This function does not invent domain-specific values; it only adds missing columns
    with neutral defaults to facilitate downstream grouping and analysis.

    For probability-like and count-like columns:
      - If both are missing, nothing is created here; use normalize_counts_to_prob_within later.
      - If only counts exist, you will typically want to call normalize_counts_to_prob_within
        to derive normalized probabilities within a group.

    Returns a shallow copy with any missing standard columns added as NA/empty.
    """
    if df.empty:
        return df

    out = df.copy()

    # Make sure standard metadata columns exist (filled with NA if missing)
    for c in STANDARD_GROUP_COLS:
        if c not in out.columns:
            out[c] = pd.Series([pd.NA] * len(out), index=out.index)

    # Unify typical backend field names to string
    for c in ("backend", "ibm_backend", "label", "protein", "sequence", "circuit_hash"):
        if c in out.columns:
            out[c] = out[c].astype(str)

    # Ensure shots is numeric if present
    if "shots" in out.columns:
        out["shots"] = coerce_numeric(out["shots"])

    # Bitstring column: trim spaces if it exists
    if "bitstring" in out.columns:
        out["bitstring"] = out["bitstring"].astype(str).str.strip()

    return out


def coerce_numeric(s: pd.Series, default: float | int | None = None) -> pd.Series:
    """
    Coerce a pandas Series into numeric dtype with safe fallback.

    Parameters
    ----------
    s : pd.Series
        Input series.
    default : scalar or None
        Value to fill NaNs after coercion. If None, leaves NaNs untouched.

    Returns
    -------
    pd.Series
        Numeric series (float64 by default).
    """
    x = pd.to_numeric(s, errors="coerce")
    if default is not None:
        x = x.fillna(default)
    return x


# ----------------------------
# Bitstring utilities
# ----------------------------

def _bitstrings_to_matrix(bits: Iterable[str]) -> np.ndarray:
    """
    Convert an iterable of bitstrings into a 2D numpy array of shape (n, L) with dtype=uint8.
    """
    arr: List[np.ndarray] = []
    L_ref: Optional[int] = None
    for b in bits:
        b = str(b)
        row = np.frombuffer(b.encode("ascii"), dtype=np.uint8) - ord("0")
        if L_ref is None:
            L_ref = row.size
        elif row.size != L_ref:
            raise ValueError(f"Inconsistent bitstring length: expected {L_ref}, got {row.size}")
        if np.any((row < 0) | (row > 1)):
            raise ValueError("Non-binary characters found in bitstring.")
        arr.append(row)
    if not arr:
        return np.empty((0, 0), dtype=np.uint8)
    return np.vstack(arr)


def expand_bitstrings(
    df: pd.DataFrame,
    bit_col: str = "bitstring",
    prefix: str = "b",
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Expand a 'bitstring' column into b0..b{L-1} numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a 'bitstring' column.
    bit_col : str
        Name of the bitstring column.
    prefix : str
        Prefix for expanded columns.
    drop_original : bool
        Whether to drop the original bitstring column.

    Returns
    -------
    pd.DataFrame
        Dataframe with new bit columns and 'n_bits' column.
    """
    if df.empty:
        return df

    if bit_col not in df.columns:
        # nothing to do; return as is
        return df

    bits = df[bit_col].astype(str).tolist()
    X = _bitstrings_to_matrix(bits)
    L = X.shape[1]

    out = df.copy()
    for j in range(L):
        out[f"{prefix}{j}"] = X[:, j].astype(np.uint8)
    out["n_bits"] = L
    if drop_original:
        out = out.drop(columns=[bit_col])
    return out


# ----------------------------
# Probability normalization
# ----------------------------

def _choose_existing(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _auto_group_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristic: choose a stable, reproducible group of columns to define an "experiment".
    Preference order: intersection with STANDARD_GROUP_COLS plus 'source_file' if present.
    """
    cols = df.columns.tolist()
    keys = [c for c in STANDARD_GROUP_COLS if c in cols]
    if "source_file" in cols:
        keys.append("source_file")
    # Ensure 'bitstring' is NOT in group keys (we group bitstrings within an experiment)
    return keys


def normalize_counts_to_prob_within(
    df: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    prob_col_candidates: Sequence[str] = PROB_COL_CANDIDATES,
    count_col_candidates: Sequence[str] = COUNT_COL_CANDIDATES,
    out_prob_col: str = "q_prob",
    min_prob: float = 1e-300,
) -> pd.DataFrame:
    """
    Normalize probabilities within experiment-defined groups.

    Behavior:
      1) If a probability-like column already exists, it is copied to `out_prob_col`.
         If multiple exist, the first found is used.
      2) Else, if a count-like column exists, it is normalized within group to sum to 1.
      3) If neither exists, assign a uniform probability within each group.

    This function avoids groupby.apply to prevent pandas FutureWarnings.
    It relies on vectorized groupby.transform operations only.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Should contain bitstrings/counts/probs.
    group_cols : Sequence[str] | None
        Columns that define an experiment. If None, will be chosen heuristically.
    prob_col_candidates : Sequence[str]
        Candidate names for existing probability column.
    count_col_candidates : Sequence[str]
        Candidate names for existing count column.
    out_prob_col : str
        The output probability column name.
    min_prob : float
        Lower bound to avoid zero-probability logs downstream.

    Returns
    -------
    pd.DataFrame
        Dataframe with a new/overwritten `out_prob_col`.
    """
    if df.empty:
        return df

    out = df.copy()
    cols = out.columns

    if group_cols is None:
        group_cols = _auto_group_columns(out)

    prob_col = _choose_existing(cols, prob_col_candidates)
    count_col = _choose_existing(cols, count_col_candidates)

    if prob_col is not None:
        # Trust provided probabilities; clip for numerical stability
        p = coerce_numeric(out[prob_col]).fillna(0.0).clip(lower=min_prob)
        out[out_prob_col] = p
        return out

    if count_col is not None:
        counts = coerce_numeric(out[count_col]).fillna(0.0).clip(lower=0.0)
        if group_cols:
            denom = counts.groupby(out[group_cols].apply(tuple, axis=1)).transform("sum").clip(lower=min_prob)
        else:
            denom = pd.Series([counts.sum()] * len(counts), index=counts.index).clip(lower=min_prob)
        out[out_prob_col] = (counts / denom).clip(lower=min_prob)
        return out

    # Neither prob nor counts: use uniform within group
    if group_cols:
        # Group sizes via transform
        sizes = out.groupby(out[group_cols].apply(tuple, axis=1))["bitstring"].transform("size")
        sizes = sizes.replace(0, np.nan).fillna(1.0)
        out[out_prob_col] = (1.0 / sizes).clip(lower=min_prob)
    else:
        n = max(len(out), 1)
        out[out_prob_col] = float(1.0 / n)
    return out


# ----------------------------
# MJ matrix loader
# ----------------------------

def load_mj_matrix(
    mj_path: Optional[str | os.PathLike],
    fallback_pkg_file: Optional[str | os.PathLike] = None
) -> pd.DataFrame:
    """
    Load Miyazawa–Jernigan (MJ) interaction matrix.

    Parameters
    ----------
    mj_path : str | Path | None
        Path to an external MJ matrix. If None or not found, try fallback.
    fallback_pkg_file : str | Path | None
        Fallback path relative to the package, e.g., Path(__file__).with_name("mj_matrix.txt").

    Returns
    -------
    pd.DataFrame
        Square symmetric matrix with amino-acid labels as index/columns if present.
        If labels are missing, numeric indices are used.

    Notes
    -----
    The function is tolerant to:
      - Extra header rows
      - Whitespace-separated values
      - Asymmetric inputs (it will symmetrize by (M + M.T) / 2)
    """
    df = pd.DataFrame()

    # Try user-provided path first
    if mj_path:
        df = safe_read_table(mj_path, header=0)
        if df.empty:
            # Some MJ files have no header; try header=None
            df = safe_read_table(mj_path, header=None)

    # Fallback to a bundled/default file if provided
    if df.empty and fallback_pkg_file:
        df = safe_read_table(fallback_pkg_file, header=0)
        if df.empty:
            df = safe_read_table(fallback_pkg_file, header=None)

    if df.empty:
        return df  # caller can decide how to handle missing MJ

    # If the first column looks like labels, set it as index
    # Otherwise keep numeric indices
    if df.shape[1] >= 2:
        # Heuristic: if the first column is non-numeric and unique, treat as index
        first_col = df.columns[0]
        col0 = df[first_col]
        if not pd.api.types.is_numeric_dtype(col0) and col0.is_unique:
            df = df.set_index(first_col)

    # Try to coerce to numeric and symmetrize
    df_num = df.apply(pd.to_numeric, errors="coerce")
    # If column names are not numeric, keep them; otherwise set generic col names
    if df_num.columns.duplicated().any():
        df_num.columns = [f"C{j}" for j in range(df_num.shape[1])]
    # Symmetrize
    try:
        df_num = (df_num + df_num.T) / 2.0
    except Exception:
        # If non-square due to malformed file, best-effort: clip to square
        m = min(df_num.shape[0], df_num.shape[1])
        df_num = df_num.iloc[:m, :m]
        df_num = (df_num + df_num.T) / 2.0

    return df_num
