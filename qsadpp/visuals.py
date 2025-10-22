# --*-- coding:utf-8 --*--
# @time:10/21/25 14:11
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:visuals.py

# qsadpp/visuals.py
"""
Minimal plotting helpers for reports (matplotlib only).

Generates:
- energy histogram (E_A)
- S vs -log q scatter
- per-cluster boxplot of E_A

Each plot saver returns the output path for convenience.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_energy_hist(df: pd.DataFrame, out_path: Union[str, Path], title: str = "E_A histogram") -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    EA = df["E_A"].to_numpy(dtype=float)
    plt.figure()
    plt.hist(EA, bins=50)
    plt.xlabel("E_A")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)


def plot_S_vs_logq(df: pd.DataFrame, out_path: str | Path, title: str = "S vs -log(q_prob)") -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = -np.log(df["q_prob"].to_numpy(dtype=float) + 1e-12)
    y = df["S"].to_numpy(dtype=float)
    plt.figure()
    plt.scatter(x, y, s=6)
    plt.xlabel("-log(q_prob)")
    plt.ylabel("S")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)


def plot_energy_box_by_cluster(df: pd.DataFrame, out_path: str | Path, title: str = "E_A by Cluster") -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    # ensure sorted cluster order
    tmp = df.copy()
    tmp = tmp.sort_values("cluster")
    clusters = sorted(tmp["cluster"].unique().tolist())
    data = [tmp[tmp["cluster"] == c]["E_A"].to_numpy(dtype=float) for c in clusters]
    plt.boxplot(data, labels=[str(c) for c in clusters], showfliers=False)
    plt.xlabel("Cluster")
    plt.ylabel("E_A")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)
