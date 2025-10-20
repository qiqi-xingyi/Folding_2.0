# --*-- conding:utf-8 --*--
# @time:10/20/25 01:54
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:utils.py

from __future__ import annotations
import numpy as np
from typing import Iterable

def bitstrings_to_array(bitstrings: Iterable[str]) -> np.ndarray:
    """
    Convert iterable of bitstrings like ["0101","1110"] to array (N, L) of ints {0,1}.
    """
    bitstrings = list(bitstrings)
    if not bitstrings:
        return np.zeros((0, 0), dtype=int)
    L = len(bitstrings[0])
    arr = np.zeros((len(bitstrings), L), dtype=int)
    for i, s in enumerate(bitstrings):
        if len(s) != L:
            raise ValueError("Mixed-length bitstrings detected.")
        arr[i, :] = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
    return arr

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def pairwise_hamming(arr: np.ndarray) -> np.ndarray:
    """
    Pairwise Hamming distances (n x n) for an array of shape (n, L).
    """
    n = arr.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi = arr[i]
        for j in range(i + 1, n):
            dij = np.sum(xi != arr[j])
            d[i, j] = dij
            d[j, i] = dij
    return d
