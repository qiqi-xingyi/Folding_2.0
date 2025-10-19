# --*-- conding:utf-8 --*--
# @time:10/19/25 10:12
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:beck.py

from __future__ import annotations
from typing import Dict, Optional
import warnings
from qiskit import QuantumCircuit

# Try to import local samplers if available
try:
    from qiskit_aer.primitives import Sampler as AerSampler
except Exception:
    AerSampler = None

try:
    from qiskit.primitives import Sampler as LocalSampler
except Exception:
    LocalSampler = None


class SamplerBackend:
    """Abstract base for a sampler backend producing measurement counts."""

    def __init__(self, shots: int):
        self.shots = shots

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Execute a circuit and return measurement counts."""
        raise NotImplementedError


class LocalSimulatorBackend(SamplerBackend):
    """
    Local simulator backend using either Qiskit Aer (preferred)
    or the default Qiskit primitives Sampler.
    """

    def __init__(self, shots: int, seed: Optional[int] = None):
        super().__init__(shots)
        if AerSampler is not None:
            self._sampler = AerSampler(options={"shots": shots, "seed_simulator": seed})
        elif LocalSampler is not None:
            warnings.warn(
                "qiskit-aer not found; falling back to qiskit.primitives.Sampler "
                "(probabilities will be approximated as counts)."
            )
            self._sampler = LocalSampler(options={"shots": shots})
        else:
            raise RuntimeError(
                "No available local sampler. Please install qiskit-aer "
                "or use Qiskit >= 1.2 which includes primitives."
            )

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Run the given circuit and return integer counts."""
        job = self._sampler.run([circuit])
        result = job.result()
        probs = result[0].data.meas.get_probabilities()
        counts = {b: int(round(p * self.shots)) for b, p in probs.items() if p > 0.0}

        # Adjust rounding drift so total == shots
        drift = self.shots - sum(counts.values())
        if drift != 0 and counts:
            k = max(counts, key=counts.get)
            counts[k] += drift
        return counts


class IBMSamplerBackend(SamplerBackend):
    """IBM Runtime backend using SamplerV2."""

    def __init__(self, shots: int, backend_name: Optional[str]):
        super().__init__(shots)
        try:
            from qiskit_ibm_runtime import SamplerV2 as IBMSampler
        except Exception as e:
            raise RuntimeError(
                "qiskit-ibm-runtime must be installed to use IBM backends."
            ) from e

        self._sampler = IBMSampler(session=None, options={"default_shots": shots, "backend": backend_name})

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """Run the circuit on an IBM backend and return integer counts."""
        job = self._sampler.run([circuit])
        result = job.result()
        probs = result[0].data.meas.get_probabilities()
        counts = {b: int(round(p * self.shots)) for b, p in probs.items() if p > 0.0}

        drift = self.shots - sum(counts.values())
        if drift != 0 and counts:
            k = max(counts, key=counts.get)
            counts[k] += drift
        return counts


def make_backend(kind: str, shots: int, seed_sim: Optional[int], ibm_backend: Optional[str]) -> SamplerBackend:
    """
    Factory function returning the appropriate backend object.

    Parameters
    ----------
    kind : str
        Either "simulator" or "ibm".
    shots : int
        Number of measurement shots.
    seed_sim : Optional[int]
        RNG seed for simulator reproducibility.
    ibm_backend : Optional[str]
        Name of IBM backend (if kind == "ibm").
    """
    if kind == "simulator":
        return LocalSimulatorBackend(shots=shots, seed=seed_sim)
    elif kind == "ibm":
        return IBMSamplerBackend(shots=shots, backend_name=ibm_backend)
    else:
        raise ValueError(f"Unknown backend kind: {kind}")
