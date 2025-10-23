# --*-- conding:utf-8 --*--
# @time:10/22/25 21:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:mapper.py.py


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class StructureMapper:
    """
    Minimal bitstring -> Cα coordinates mapper.

    Model: 2D square lattice self-avoiding *attempt* (no guarantees), z=0.
    - Interpret bitstring as a sequence of turns (2 bits -> 1 turn):
        "00": go +x, "01": +y, "10": -x, "11": -y
    - Start at (0,0,0); we need L residues -> L-1 moves (use first 2*(L-1) bits).
    - If bitstring shorter than needed -> wrap-around.
    - If collisions happen, we still place the bead (no rejection), just overlap—this is *deliberately* simple.

    Output:
      - Adds a 'coords' column to DataFrame, each entry is np.ndarray shape (L,3)
    """

    def _bits_to_moves(self, s: str) -> List[Tuple[int, int]]:
        mv = {"00": (1, 0), "01": (0, 1), "10": (-1, 0), "11": (0, -1)}
        k = len(s) // 2
        out: List[Tuple[int, int]] = []
        for i in range(k):
            step = mv[s[2*i:2*i+2]]
            out.append(step)
        return out

    def map_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        L = int(out["L"].mode().iloc[0])
        need_bits = 2 * max(L - 1, 0)

        coords_list: List[np.ndarray] = []
        for s in out["bitstring"].astype(str):
            if len(s) < need_bits and need_bits > 0:
                # wrap bits to reach needed length
                times = (need_bits + len(s) - 1) // len(s)
                s = (s * times)[:need_bits]
            else:
                s = s[:need_bits]

            moves = self._bits_to_moves(s)
            xy = [(0, 0)]
            x, y = 0, 0
            for dx, dy in moves:
                x += dx; y += dy
                xy.append((x, y))
            arr = np.zeros((L, 3), dtype=float)
            for i, (xx, yy) in enumerate(xy[:L]):
                arr[i, 0] = float(xx)
                arr[i, 1] = float(yy)
                arr[i, 2] = 0.0
            coords_list.append(arr)

        out["coords"] = coords_list
        return out
