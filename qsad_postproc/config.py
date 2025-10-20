# --*-- conding:utf-8 --*--
# @time:10/20/25 01:54
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:config.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class AnalyzerConfig:
    """
    Configuration for post-processing QSaD sampling outputs.
    """
    n_qubits: Optional[int] = None
    group_keys: Tuple[str, ...] = ("beta",)
    prob_col: str = "prob"
    count_col: str = "count"
    bitstring_col: str = "bitstring"
    shots_col: str = "shots"
    seed_col: str = "seed"
