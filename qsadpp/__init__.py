# --*-- conding:utf-8 --*--
# @time:10/22/25 21:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py.py

# --*-- coding:utf-8 --*--
# qsadpp/__init__.py
"""
Minimal, end-to-end post-processing toolkit:
1) SamplingReader   - read/parse quantum sampling CSV
2) StructureMapper  - bitstring -> coarse Cα coordinates (2D lattice, z=0)
3) EnergyCalculator - structural pseudo-energy + Ising-like Hamiltonian
4) ClusterAnalyzer  - clustering and lowest-energy cluster selection
5) StructureFitter  - weighted averaging in the best cluster and save XYZ/CSV
"""

from .reader import SamplingReader
from .mapper import StructureMapper
from .energy import EnergyCalculator
from .clusterer import ClusterAnalyzer
from .fitter import StructureFitter

__all__ = [
    "SamplingReader",
    "StructureMapper",
    "EnergyCalculator",
    "ClusterAnalyzer",
    "StructureFitter",
]
