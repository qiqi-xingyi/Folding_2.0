# --*-- conding:utf-8 --*--
# @time:10/19/25 10:10
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any


@dataclass
class BackendConfig:
    """
    Configuration for the quantum backend.

    Attributes
    ----------
    kind : Literal["simulator", "ibm"]
        Type of backend to use.
    shots : int
        Number of measurement shots per (beta, seed) circuit.
    seed_sim : Optional[int]
        Random seed for local simulator reproducibility.
    ibm_backend : Optional[str]
        Name of the IBM backend if kind == "ibm".
    """
    kind: Literal["simulator", "ibm"] = "simulator"
    shots: int = 1024
    seed_sim: Optional[int] = None
    ibm_backend: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation."""
        return {
            "kind": self.kind,
            "shots": int(self.shots),
            "seed_sim": self.seed_sim,
            "ibm_backend": self.ibm_backend,
        }


@dataclass
class SamplingConfig:
    """
    Configuration for quantum sampling.

    Attributes
    ----------
    L : Optional[int]
        Sequence length (for bookkeeping only).
    betas : List[float]
        List of beta values controlling e^{-i β H} evolution.
    seeds : int
        Number of random ansatz seeds.
    reps : int
        Number of EfficientSU2 repetitions.
    entanglement : str
        Entanglement pattern for the ansatz.
    label : str
        Label for this sampling experiment.
    backend : BackendConfig
        Backend configuration.
    out_csv : str
        Path to output CSV file.
    write_parquet : bool
        Whether to also write Parquet output.
    extra_meta : Dict[str, Any]
        Arbitrary metadata written to each output row.
    """
    L: Optional[int]
    betas: List[float]
    seeds: int = 8
    reps: int = 1
    entanglement: str = "linear"
    label: str = "default"
    backend: BackendConfig = field(default_factory=BackendConfig)
    out_csv: str = "samples.csv"
    write_parquet: bool = False
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_qubits_hint(self) -> Optional[int]:
        """
        Optional hint for number of qubits when based on tetrahedral encoding.
        Usually n_qubits ≈ 2 * (L - 1). This is only informational.
        """
        if self.L is None:
            return None
        return max(1, 2 * (self.L - 1))

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation."""
        return {
            "L": self.L,
            "betas": list(self.betas),
            "seeds": int(self.seeds),
            "reps": int(self.reps),
            "entanglement": self.entanglement,
            "label": self.label,
            "out_csv": self.out_csv,
            "write_parquet": bool(self.write_parquet),
            "backend": self.backend.as_dict(),
            "extra_meta": dict(self.extra_meta),
            "n_qubits_hint": self.n_qubits_hint,
        }

    def validate(self) -> None:
        """Light validation of the configuration."""
        if not self.betas:
            raise ValueError("betas must be a non-empty list, e.g. [0.0, 0.5, 1.0].")
        if self.seeds <= 0:
            raise ValueError("seeds must be > 0.")
        if self.backend.shots <= 0:
            raise ValueError("backend.shots must be > 0.")
        if not isinstance(self.entanglement, str) or not self.entanglement:
            raise ValueError("entanglement must be a non-empty string.")
