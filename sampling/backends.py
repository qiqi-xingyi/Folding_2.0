# --*-- conding:utf-8 --*--
# @time:10/19/25 10:12
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:beck.py

from __future__ import annotations
from typing import Dict, Optional
import warnings
import numpy as np

from qiskit import QuantumCircuit

# Prefer AerSampler if available
try:
    from qiskit_aer.primitives import AerSampler
except Exception:
    AerSampler = None

# Fallback local primitives Sampler if present
try:
    from qiskit.primitives import Sampler as LocalSampler
except Exception:
    LocalSampler = None

# Optional: statevector fallback
try:
    from qiskit.quantum_info import Statevector
except Exception:
    Statevector = None


class SamplerBackend:
    """Abstract base for a sampler backend producing measurement counts."""

    def __init__(self, shots: int, seed: Optional[int] = None):
        self.shots = int(shots)
        self.seed = seed

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        raise NotImplementedError


def _normalize_probs_to_counts(probs: Dict[str, float], shots: int) -> Dict[str, int]:
    counts = {b: int(round(max(0.0, float(p)) * shots)) for b, p in probs.items() if float(p) > 0.0}
    drift = shots - sum(counts.values())
    if drift != 0 and counts:
        k = max(counts, key=counts.get)
        counts[k] += drift
    return counts


def _extract_probabilities(result) -> Dict[str, float]:
    """
    Robust probability extraction supporting Qiskit 2.x and 1.x result layouts.
    Tries (in order):
      - result.quasi_dists[0]            (2.x AerSampler)
      - result[0].quasi_dist             (2.x per-result)
      - result[0].data.meas.get_probabilities()  (1.x path)
      - result[0].data.meas.get_counts() -> normalize to probs
    """
    # 2.x aggregated quasi_dists
    try:
        qd = result.quasi_dists[0]
        # qd can be dict or QuasiDistribution
        return dict(qd)
    except Exception:
        pass

    # 2.x per-result quasi_dist
    try:
        qd = result[0].quasi_dist
        return dict(qd)  # may already be a dict-like
    except Exception:
        pass

    # 1.x probabilities API
    try:
        probs = result[0].data.meas.get_probabilities()
        return dict(probs)
    except Exception:
        pass

    # 1.x counts API -> normalize
    try:
        counts = result[0].data.meas.get_counts()
        total = float(sum(counts.values())) or 1.0
        return {k: v / total for k, v in counts.items()}
    except Exception as e:
        raise RuntimeError(f"Unrecognized sampler result format: {e}") from e


class LocalSimulatorBackend(SamplerBackend):
    """
    Local simulator backend:
      - Prefer qiskit_aer.primitives.AerSampler (2.x)
      - Fallback to qiskit.primitives.Sampler
      - Fallback to statevector (if both missing)
    """

    def __init__(self, shots: int, seed: Optional[int] = None):
        super().__init__(shots, seed)
        self._sampler = None
        self._mode = None  # "aer" | "local" | "statevector"

        if AerSampler is not None:
            # Qiskit 2.x: do not pass options in ctor; provide in run()
            self._sampler = AerSampler()
            self._mode = "aer"
        elif LocalSampler is not None:
            warnings.warn(
                "qiskit-aer not found; falling back to qiskit.primitives.Sampler "
                "(probabilities will be converted to counts)."
            )
            self._sampler = LocalSampler()
            self._mode = "local"
        elif Statevector is not None:
            warnings.warn(
                "No local sampler available; falling back to Statevector-based sampling."
            )
            self._mode = "statevector"
        else:
            raise RuntimeError(
                "No local simulation path available. Install qiskit-aer or ensure primitives are present."
            )

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        if self._mode == "statevector":
            # Remove final measurements for pure statevector evolution
            unitary_circ = self._strip_final_measurements(circuit)
            sv = Statevector.from_instruction(unitary_circ)
            probs_dict = sv.probabilities_dict()
            return _normalize_probs_to_counts(probs_dict, self.shots)

        # AerSampler / LocalSampler path (2.x or 1.x)
        # Provide run_options at call time (2.x style). 1.x ignores unknown kwargs gracefully.
        job = self._sampler.run([circuit], run_options={"shots": self.shots, "seed_simulator": self.seed})
        res = job.result()
        probs = _extract_probabilities(res)
        return _normalize_probs_to_counts(probs, self.shots)

    @staticmethod
    def _strip_final_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
        try:
            return circuit.remove_final_measurements(inplace=False)
        except Exception:
            qc = QuantumCircuit(circuit.num_qubits)
            for instr, qargs, cargs in circuit.data:
                if instr.name.lower() == "measure":
                    continue
                qc.append(instr, qargs, cargs)
            return qc


class IBMSamplerBackend(SamplerBackend):
    """IBM Runtime SamplerV2 (2.x 可用；1.x 也兼容)."""

    def __init__(self, shots: int, backend_name: Optional[str], seed: Optional[int] = None):
        super().__init__(shots, seed)
        try:
            from qiskit_ibm_runtime import SamplerV2 as IBMSampler
        except Exception as e:
            raise RuntimeError(
                "qiskit-ibm-runtime must be installed to use IBM backends."
            ) from e

        # For SamplerV2, pass default shots in options; seed is not guaranteed to be honored on HW.
        self._sampler = IBMSampler(session=None, options={"default_shots": shots, "backend": backend_name})

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        job = self._sampler.run([circuit])
        res = job.result()
        probs = _extract_probabilities(res)
        return _normalize_probs_to_counts(probs, self.shots)


def make_backend(kind: str, shots: int, seed_sim: Optional[int], ibm_backend: Optional[str]) -> SamplerBackend:
    kind = (kind or "simulator").lower()
    if kind == "simulator":
        return LocalSimulatorBackend(shots=shots, seed=seed_sim)
    elif kind == "ibm":
        return IBMSamplerBackend(shots=shots, backend_name=ibm_backend, seed=seed_sim)
    else:
        raise ValueError(f"Unknown backend kind: {kind}")

