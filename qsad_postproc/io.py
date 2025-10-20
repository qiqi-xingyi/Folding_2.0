# --*-- conding:utf-8 --*--
# @time:10/20/25 01:55
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:io.py

from __future__ import annotations
from typing import Dict
import pandas as pd

def write_many_csv(dfs: Dict[str, pd.DataFrame], path_prefix: str) -> Dict[str, str]:
    """
    Write multiple DataFrames to CSV files with name pattern {prefix}_{key}.csv.
    Returns mapping key -> path.
    """
    paths = {}
    for key, df in dfs.items():
        out = f"{path_prefix}_{key}.csv"
        df.to_csv(out, index=False)
        paths[key] = out
    return paths
