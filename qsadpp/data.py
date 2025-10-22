# --*-- coding:utf-8 --*--
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
- load_mj_table_file(path) -> dict[aa1][aa2] = float
- get_mj_table(path: Optional[str]) -> cached table (env var QSADPP_MJ_PATH honored)

Notes
-----
If you need a different default, you can plug in your lab's
true MJ matrix by editing `MJ_DEFAULT_TSV` or by providing a file path/env var.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path
import os

AA20: List[str] = list("ACDEFGHIKLMNPQRSTVWY")

# A minimal **symmetric zero** MJ matrix placeholder (21 lines: header + 20 AA lines),
# formatted as a proper TSV so the parser can always read it.
MJ_DEFAULT_TSV: str = """
AA	A	C	D	E	F	G	H	I	K	L	M	N	P	Q	R	S	T	V	W	Y
A	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
C	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
D	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
E	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
F	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
G	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
H	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
I	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
K	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
L	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
M	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
N	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
P	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Q	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
R	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
S	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
T	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
V	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
W	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
Y	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
"""

# data.py  —— replace _parse_mj_tsv with a more permissive parser
def _parse_mj_tsv(text: str) -> Dict[str, Dict[str, float]]:
    """
    Parse a Miyazawa–Jernigan matrix from whitespace/TSV text.

    Accepted headers:
      - "AA <20 AA letters>"
      - "<20 AA letters>"
    Accepted rows:
      - "<rowAA> <20 numbers>"
      - "<20 numbers>"  (row AA inferred from header order)
    The parser symmetrizes the final table.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    header = lines[0].split()
    if header[0] == "AA":
        aa_cols = header[1:]
    else:
        aa_cols = header
    assert len(aa_cols) == 20, "Header must contain 20 amino acids"

    rows = lines[1:]
    assert len(rows) == 20, "Expected 20 data rows"

    table: Dict[str, Dict[str, float]] = {aa: {} for aa in aa_cols}
    for i, row in enumerate(rows):
        parts = row.split()
        if len(parts) == 21:
            aa = parts[0]
            vals = parts[1:]
        elif len(parts) == 20:
            aa = aa_cols[i]
            vals = parts
        else:
            raise ValueError("Each row must have either 20 numbers or a label plus 20 numbers.")
        vals_f = [float(x) for x in vals]
        assert len(vals_f) == 20
        for j, aa2 in enumerate(aa_cols):
            table[aa].setdefault(aa2, vals_f[j])

    # Symmetrize
    for a in aa_cols:
        for b in aa_cols:
            v = 0.5 * (table[a].get(b, 0.0) + table[b].get(a, 0.0))
            table[a][b] = table[b][a] = v
    return table



_MJ_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def load_mj_table_file(path: str | Path) -> Dict[str, Dict[str, float]]:
    txt = Path(path).read_text(encoding="utf-8")
    return _parse_mj_tsv(txt)


def get_mj_table(path: Optional[str | Path] = None) -> Dict[str, Dict[str, float]]:
    """
    1) If `path` provided, load that file.
    2) Else if environment variable `QSADPP_MJ_PATH` is set, load it.
    3) Else use built-in `MJ_DEFAULT_TSV`
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
