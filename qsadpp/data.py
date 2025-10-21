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
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("Empty MJ TSV content")
    header = lines[0].split()
    if header[0].upper() in {"AA", "RES", "IDX"}:
        header = header[1:]
    if len(header) != 20:
        raise ValueError(f"Expected 20 AA columns, got {len(header)}: {header}")

    # simple normalization to one-letter codes if needed
    cols = [h.strip().upper() for h in header]
    if cols != [a.upper() for a in AA20]:
        raise ValueError(
            "Header AA order does not match canonical AA20. "
            f"Got {cols}, expected {AA20}"
        )

    table: Dict[str, Dict[str, float]] = {aa: {} for aa in AA20}
    if len(lines) - 1 != 20:
        raise ValueError(f"Expected 20 data rows, got {len(lines)-1}")

    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 21:
            raise ValueError(f"Row must have 21 fields (AA + 20 numbers), got: {ln}")
        row_aa = parts[0].upper()
        if row_aa not in AA20:
            raise ValueError(f"Unknown AA code in row: {row_aa}")
        vals = [float(x) for x in parts[1:]]
        for col_aa, val in zip(AA20, vals):
            table[row_aa][col_aa] = float(val)

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