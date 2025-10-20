# --*-- conding:utf-8 --*--
# @time:10/19/25 10:12
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:beck.py

from __future__ import annotations
from typing import Dict, Optional
import warnings

from qiskit import QuantumCircuit, transpile

# Local simulator: AerSimulator (official recommended path)
try:
    from qiskit_aer import AerSimulator
except Exception:
    AerSimulator = None  # type: ignore

# Optional statevector fallback
try:
    from qiskit.quantum_info import Statevector
except Exception:
    Statevector = None  # type: ignore


class SamplerBackend:
    """Abstract base for a backend producing measurement counts."""
    def __init__(self, shots: int):
        self.shots = int(shots)

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        raise NotImplementedError


def _normalize_counts_to_total(counts: Dict[str, int], shots: int) -> Dict[str, int]:
    total = sum(counts.values())
    if total == shots or not counts:
        return counts
    drift = shots - total
    k = max(counts, key=counts.get)
    counts[k] += drift
    return counts


class LocalSimulatorBackend(SamplerBackend):
    """
    Local simulator using AerSimulator with the official transpile->run->get_counts flow.
    """
    def __init__(self, shots: int, seed: Optional[int] = None):
        super().__init__(shots)
        self.seed = seed
        if AerSimulator is None and Statevector is None:
            raise RuntimeError(
                "No local simulation path available. Install qiskit-aer (AerSimulator) or ensure Statevector is present."
            )
        self._mode = "aer" if AerSimulator is not None else "statevector"
        print(f"[LocalSimulatorBackend] mode = {self._mode}")

        if self._mode == "aer":
            # You can pass global options via set_options if needed (method, device, noise_model, etc.)
            self._sim = AerSimulator()
            if self.seed is not None:
                # Run-time seed is passed via run(..., seed_simulator=...), kept for completeness
                self._seed_kw = {"seed_simulator": int(self.seed)}
            else:
                self._seed_kw = {}
        else:
            self._sim = None  # statevector path

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        if self._mode == "statevector":
            unitary = self._strip_final_measurements(circuit)
            sv = Statevector.from_instruction(unitary)
            probs = sv.probabilities_dict()
            counts = {b: int(round(float(p) * self.shots)) for b, p in probs.items() if float(p) > 0.0}
            return _normalize_counts_to_total(counts, self.shots)

        # AerSimulator path: transpile to the simulator target, then run
        tqc = transpile(circuit, self._sim)
        job = self._sim.run(tqc, shots=self.shots, **self._seed_kw)
        res = job.result()
        # You can use res.get_counts(tqc) or res.get_counts(0). Both are standard.
        counts = res.get_counts(tqc)
        if isinstance(counts, list):
            # Defensive: some returns may be a list per-experiment; take the first
            counts = counts[0]
        # Ensure the total equals requested shots
        return _normalize_counts_to_total(dict(counts), self.shots)

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
    """IBM Runtime SamplerV2 for real backends."""
    def __init__(self, shots: int, backend_name: Optional[str]):
        super().__init__(shots)
        try:
            from qiskit_ibm_runtime import SamplerV2 as IBMSamplerV2
        except Exception as e:
            raise RuntimeError("qiskit-ibm-runtime must be installed to use IBM backends.") from e
        self._sampler = IBMSamplerV2(session=None, options={"default_shots": shots, "backend": backend_name})

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        job = self._sampler.run([circuit], shots=self.shots)
        res = job.result()
        # IBM SamplerV2 returns counts via result[0].data.meas.get_counts()
        counts = res[0].data.meas.get_counts()
        return _normalize_counts_to_total(dict(counts), self.shots)


def make_backend(kind: str, shots: int, seed_sim: Optional[int], ibm_backend: Optional[str]) -> SamplerBackend:
    kind = (kind or "simulator").lower()
    if kind == "simulator":
        return LocalSimulatorBackend(shots=shots, seed=seed_sim)
    elif kind == "ibm":
        return IBMSamplerBackend(shots=shots, backend_name=ibm_backend)
    else:
        raise ValueError(f"Unknown backend kind: {kind}")


