# --*-- conding:utf-8 --*--
# @time:10/19/25 10:12
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:beck.py

from __future__ import annotations
from typing import Dict, Optional
import warnings

from qiskit import QuantumCircuit

# Preferred local simulator: Aer SamplerV2 (Qiskit 2.x)
try:
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
except Exception:
    AerSamplerV2 = None  # type: ignore

# Optional statevector fallback
try:
    from qiskit.quantum_info import Statevector
except Exception:
    Statevector = None  # type: ignore


class SamplerBackend:
    """Abstract base for a sampler backend producing measurement counts."""
    def __init__(self, shots: int):
        self.shots = int(shots)

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        raise NotImplementedError


def _normalize_counts_to_total(counts: Dict[str, int], shots: int) -> Dict[str, int]:
    total = sum(counts.values())
    if total == shots or not counts:
        return counts
    drift = shots - total
    # Assign drift to the most frequent key
    k = max(counts, key=counts.get)
    counts[k] += drift
    return counts


def _extract_counts_v2(result) -> Dict[str, int]:
    """
    Qiskit 2.x result path:
      result[0].data.meas.get_counts()
    Fallbacks try probabilities if needed.
    """
    # Primary: counts
    try:
        return dict(result[0].data.meas.get_counts())
    except Exception:
        pass
    # Fallback: probabilities -> counts
    try:
        probs = dict(result[0].data.meas.get_probabilities())
        counts = {b: int(round(float(p) * 1_000_000)) for b, p in probs.items() if float(p) > 0.0}
        return counts
    except Exception as e:
        raise RuntimeError(f"Unrecognized SamplerV2 result format: {e}") from e


class LocalSimulatorBackend(SamplerBackend):
    """
    Local simulator using qiskit_aer.primitives.SamplerV2 (Qiskit 2.x).
    Falls back to Statevector if Aer is not available.
    """

    def __init__(self, shots: int, seed: Optional[int] = None):
        super().__init__(shots)
        self.seed = seed
        self._mode = None  # "aer" | "statevector"
        self._sampler = None

        if AerSamplerV2 is not None:
            self._sampler = AerSamplerV2()
            self._mode = "aer"
        elif Statevector is not None:
            warnings.warn(
                "No local SamplerV2 available; falling back to Statevector-based sampling."
            )
            self._mode = "statevector"
        else:
            raise RuntimeError(
                "No local simulation path available. Install qiskit-aer for SamplerV2."
            )

        print(f"[LocalSimulatorBackend] mode = {self._mode}")

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        if self._mode == "statevector":
            unitary = self._strip_final_measurements(circuit)
            sv = Statevector.from_instruction(unitary)
            probs = sv.probabilities_dict()
            counts = {b: int(round(float(p) * self.shots)) for b, p in probs.items() if float(p) > 0.0}
            return _normalize_counts_to_total(counts, self.shots)

        # Aer SamplerV2 path
        job = self._sampler.run([circuit], shots=self.shots)  # seed is not guaranteed/supported here
        res = job.result()
        counts = _extract_counts_v2(res)
        # Counts from Aer are already integer; align to requested shots just in case
        return _normalize_counts_to_total(counts, self.shots)

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
    """IBM Runtime SamplerV2 (Qiskit 2.x)."""

    def __init__(self, shots: int, backend_name: Optional[str]):
        super().__init__(shots)
        try:
            from qiskit_ibm_runtime import SamplerV2 as IBMSamplerV2
        except Exception as e:
            raise RuntimeError("qiskit-ibm-runtime must be installed to use IBM backends.") from e

        # default_shots in options sets the runtime default; run() can also pass shots explicitly
        self._sampler = IBMSamplerV2(session=None, options={"default_shots": shots, "backend": backend_name})

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        job = self._sampler.run([circuit], shots=self.shots)
        res = job.result()
        counts = _extract_counts_v2(res)
        return _normalize_counts_to_total(counts, self.shots)


def make_backend(kind: str, shots: int, seed_sim: Optional[int], ibm_backend: Optional[str]) -> SamplerBackend:
    kind = (kind or "simulator").lower()
    if kind == "simulator":
        return LocalSimulatorBackend(shots=shots, seed=seed_sim)
    elif kind == "ibm":
        return IBMSamplerBackend(shots=shots, backend_name=ibm_backend)
    else:
        raise ValueError(f"Unknown backend kind: {kind}")

