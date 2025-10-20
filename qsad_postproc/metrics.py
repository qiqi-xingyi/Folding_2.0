# --*-- conding:utf-8 --*--
# @time:10/20/25 01:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:metrics.py

from __future__ import annotations
import numpy as np

def shannon_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """
    Shannon entropy in bits. Safe for zeros.
    """
    p = probs[probs > eps]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def effective_sample_size(probs: np.ndarray, eps: float = 1e-12) -> float:
    """
    ESS = 1 / sum p^2. Safe for zeros.
    """
    p = probs[probs > eps]
    if p.size == 0:
        return 0.0
    return float(1.0 / np.sum(p * p))
