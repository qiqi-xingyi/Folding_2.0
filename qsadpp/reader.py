# --*-- conding:utf-8 --*--
# @time:10/22/25 21:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:reader.py


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List
import numpy as np
import pandas as pd


@dataclass
class SamplingReader:
    """
    Read and normalize quantum sampling results from a single CSV file.

    Required columns (at least one of the following probability sources):
      - bitstring : str (e.g. "010110")
      - count or counts : int (shots count)
      - prob or q_prob or p : float (probability)

    Optional metadata columns kept as-is: ["L","sequence","shots","beta","seed","label","backend"]

    Normalization:
      - If 'prob/q_prob/p' exists -> copy into 'q_prob' (clip to [1e-300,1]).
      - Else if count-like exists -> normalize within the file so sum(q_prob)=1.
      - Else -> uniform probability over rows.
    """

    prob_cols: Sequence[str] = ("q_prob", "prob", "p")
    count_cols: Sequence[str] = ("count", "counts", "n")

    def read(self, csv_path: str | Path) -> pd.DataFrame:
        p = Path(csv_path)
        df = pd.read_csv(p)
        if "bitstring" not in df.columns:
            raise ValueError("CSV must contain a 'bitstring' column.")

        # Trim/str cast
        df["bitstring"] = df["bitstring"].astype(str).str.strip()

        # Infer q_prob
        prob_col = next((c for c in self.prob_cols if c in df.columns), None)
        if prob_col is not None:
            q = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0)
            df["q_prob"] = q.clip(lower=1e-300, upper=1.0)
        else:
            cnt_col = next((c for c in self.count_cols if c in df.columns), None)
            if cnt_col is not None:
                cnt = pd.to_numeric(df[cnt_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                s = float(cnt.sum()) if float(cnt.sum()) > 0 else 1.0
                df["q_prob"] = (cnt / s).clip(lower=1e-300)
            else:
                n = max(len(df), 1)
                df["q_prob"] = float(1.0 / n)

        # Useful numeric info
        if "L" not in df.columns:
            # Infer L from bitstring length if absent (pairs->turns, length L ≈ len(bits)//2 + 1)
            bl = df["bitstring"].str.len().mode().iloc[0]
            df["L"] = int(bl // 2 + 1)

        # Expand bits to b0..b{len-1}
        # Keep for clustering or quick use
        Lbits = int(df["bitstring"].str.len().mode().iloc[0])
        X = np.vstack([np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0") for s in df["bitstring"]])
        for j in range(Lbits):
            df[f"b{j}"] = X[:, j].astype(np.uint8)

        return df
