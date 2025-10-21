# --*-- conding:utf-8 --*--
# @time:10/21/25 14:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:data.py

"""
Built-in data loaders for QSAD post-processing.

We ship a **default MJ (Miyazawa–Jernigan) potential** inside the package so users
can run the pipeline out-of-the-box. You may replace it with your lab-curated
matrix later. The loader also supports overriding via an explicit file path or
an environment variable `QSADPP_MJ_PATH`.

API
---
- `get_mj_table(path: str | Path | None = None) -> dict`
  Return the nested-dict MJ table using the canonical 20-AA order.

Implementation notes
--------------------
- Parsing is dependency-light (no pandas required).
- The embedded matrix below is a **placeholder** (zeros). Replace it with your
  true MJ matrix by editing `MJ_DEFAULT_TSV` or by providing a file path/env var.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import os

# Canonical 20-AA order. **Must** match the matrix header/order below and anywhere else.
AA20: List[str] = list("ACDEFGHIKLMNPQRSTVWY")

# -----------------------------------------------------------------------------
# Embedded default MJ matrix (TSV-like). Header row + 20 rows. Placeholder zeros.
# Replace with your lab's matrix for real experiments.
# -----------------------------------------------------------------------------
MJ_DEFAULT_TSV: str = (
    "	".join(["AA"] + AA20) + " "+ " ".join(
        ["	".join([aa] + ["0.0"] * 20) for aa in AA20]
    )
)


def _parse_mj_tsv(text: str) -> Dict[str, Dict[str, float]]:
    """
    Parse a Miyazawa–Jernigan matrix from a TSV or whitespace-delimited text.
    - Does NOT enforce a fixed AA20 order.
    - Uses the file header as the column order and the first field of each row as the row amino acid.
    - Returns a nested dict: table[row_aa][col_aa] = float value.
    - Case-insensitive (converted to uppercase).
    - Expects exactly 20 amino acids but allows any order.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("Empty MJ TSV content")

    # Header
    header = lines[0].split()
    if header[0].upper() in {"AA", "RES", "IDX"}:
        header = header[1:]
    cols = [h.strip().upper() for h in header]
    if len(cols) != 20:
        raise ValueError(f"Expected 20 AA columns, got {len(cols)}: {cols}")
    if any(len(c) != 1 for c in cols):
        raise ValueError(f"Column labels must be one-letter amino acid codes, got: {cols}")

    # Parse data rows
    table: Dict[str, Dict[str, float]] = {}
    seen_rows = set()
    for ln in lines[1:]:
        parts = ln.split()
        if not parts:
            continue
        if len(parts) != 1 + len(cols):
            raise ValueError(f"Row must have 1 + {len(cols)} fields (AA + values), got: {ln}")
        row_aa = parts[0].upper()
        if len(row_aa) != 1:
            raise ValueError(f"Row AA label must be one-letter code, got: {row_aa}")
        if row_aa in seen_rows:
            raise ValueError(f"Duplicate row for amino acid: {row_aa}")
        seen_rows.add(row_aa)

        vals = [float(x) for x in parts[1:]]
        row = {c: float(v) for c, v in zip(cols, vals)}
        table[row_aa] = row

    # Validate completeness
    if len(table) != 20:
        raise ValueError(f"MJ table must have 20 rows, got {len(table)} rows: {sorted(table.keys())}")

    missing_cols = [aa for aa in table.keys() if aa not in cols]
    if missing_cols:
        raise ValueError(f"Column labels do not cover all row AAs. Missing in columns: {missing_cols}")

    return table



def load_mj_table_file(path: str | Path) -> Dict[str, Dict[str, float]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    return _parse_mj_tsv(text)


# simple module-level cache
_MJ_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def get_mj_table(path: str | Path | None = None) -> Dict[str, Dict[str, float]]:
    """Return the MJ table as a nested dict.

    Resolution order:
    1) explicit `path` argument
    2) env var `QSADPP_MJ_PATH`
    3) built-in `MJ_DEFAULT_TSV`
    """
    global _MJ_CACHE
    if _MJ_CACHE is not None and path is None and os.getenv("QSADPP_MJ_PATH") is None:
        return _MJ_CACHE

    if path is not None:
        table = load_mj_table_file(path)
    else:
        envp = os.getenv("QSADPP_MJ_PATH")
        if envp:
            table = load_mj_table_file(envp)
        else:
            table = _parse_mj_tsv(MJ_DEFAULT_TSV)
    # cache only when using default/env (not caching explicit differing paths)
    if path is None:
        _MJ_CACHE = table
    return table