# --*-- coding:utf-8 --*--
# @time:10/19/25 10:12
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:beck.py

from __future__ import annotations
from typing import Dict, Optional, List
import warnings

from qiskit import QuantumCircuit, transpile

# Local simulator: AerSimulator
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


def _bit_reverse(s: str) -> str:
    """Reverse bitstring for msb/lsb alignment if needed."""
    return s[::-1]


def _quasi_to_counts(quasi: Dict[str, float], shots: int) -> Dict[str, int]:
    """Discretize a quasi-probability distribution to integer counts with total==shots."""
    raw = {k: int(round(max(0.0, float(p)) * shots)) for k, p in quasi.items()}
    return _normalize_counts_to_total(raw, shots)


def _normalize_counts_to_total(counts: Dict[str, int], shots: int) -> Dict[str, int]:
    """
    Make sure sum(counts.values()) equals shots without producing negative buckets.
    Uses proportional rescaling and drift correction over top-frequency keys.
    """
    if not counts:
        return {}
    total = sum(counts.values())
    if total == shots:
        return counts
    if total <= 0:
        scaled = {k: 0 for k in counts}
    else:
        scaled = {k: max(0, int(round(v * shots / total))) for k, v in counts.items()}
    diff = shots - sum(scaled.values())
    if diff != 0:
        keys = sorted(scaled.keys(), key=lambda k: counts[k], reverse=True)
        i = 0
        step = 1 if diff > 0 else -1
        while diff != 0 and keys:
            k = keys[i % len(keys)]
            if step < 0 and scaled[k] == 0:
                i += 1
                continue
            scaled[k] += step
            diff -= step
            i += 1
    return scaled


class LocalSimulatorBackend(SamplerBackend):
    """
    Local simulator using AerSimulator with transpile->run->get_counts flow.
    Falls back to Statevector to derive probabilities when Aer is unavailable.
    """
    def __init__(self, shots: int, seed: Optional[int] = None):
        super().__init__(shots)
        self.seed = seed
        if AerSimulator is None and Statevector is None:
            raise RuntimeError(
                "No local simulation path available. Install qiskit-aer or ensure qiskit.quantum_info.Statevector is present."
            )
        self._mode = "aer" if AerSimulator is not None else "statevector"
        print(f"[LocalSimulatorBackend] mode = {self._mode}")

        if self._mode == "aer":
            self._sim = AerSimulator()
            self._seed_kw = {"seed_simulator": int(self.seed)} if self.seed is not None else {}
        else:
            self._sim = None  # statevector path

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        if self._mode == "statevector":
            # Ensure measurement mapping consistency: emulate measure_all() ordering
            unitary = self._strip_final_measurements(circuit)
            sv = Statevector.from_instruction(unitary)
            probs = sv.probabilities_dict()  # keys in computational basis
            counts = {b: int(round(float(p) * self.shots)) for b, p in probs.items() if float(p) > 0.0}
            counts = _normalize_counts_to_total(counts, self.shots)
            # If your Aer/IBM path uses different endianness, toggle bit reverse here:
            # counts = { _bit_reverse(k): v for k, v in counts.items() }
            return counts

        tqc = transpile(
            circuit,
            self._sim,
            optimization_level=1,
            seed_transpiler=self.seed if self.seed is not None else None,
        )
        job = self._sim.run(tqc, shots=self.shots, **self._seed_kw)
        res = job.result()
        counts = res.get_counts(tqc)
        if isinstance(counts, list):
            counts = counts[0]
        return _normalize_counts_to_total(dict(counts), self.shots)

    @staticmethod
    def _strip_final_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Remove final measurements; this assumes there are no conditional gates
        depending on classical bits. For sampling circuits, this is typically safe.
        """
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
    """
    IBM Runtime SamplerV2 backend with robust result parsing.
    Uses Session(backend=...) and passes the session via mode=...
    """
    def __init__(self, shots: int, backend_name: Optional[str]):
        super().__init__(shots)
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
        except Exception as e:
            raise RuntimeError("qiskit-ibm-runtime must be installed to use IBM backends.") from e

        self._service = QiskitRuntimeService()
        # Resolve backend
        if backend_name:
            backend = self._service.backend(backend_name)
        else:
            # Fallback to a least-busy real backend if none provided
            backend = self._service.least_busy(operational=True, simulator=False)

        # Create a persistent session bound to the chosen backend
        self._session = Session(backend=backend)

        # Create the Sampler primitive; pass the session via 'mode'
        self._sampler = SamplerV2(
            mode=self._session,
            options={
                "default_shots": shots,  # used if 'shots' not provided at run()
            },
        )

    def __del__(self):
        # Best-effort cleanup of the session
        try:
            if hasattr(self, "_session") and self._session is not None:
                self._session.close()
        except Exception:
            pass

    def run_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        # You can pass 'shots' explicitly; otherwise default_shots is used.
        job = self._sampler.run([circuit], shots=self.shots)
        result = job.result()
        pub = result[0]
        data = pub.data

        # Path A: classical register access, e.g. data.cr.get_counts()
        try:
            # Prefer a register named 'cr' if present
            cr = getattr(data, "cr", None)
            if cr is not None and hasattr(cr, "get_counts"):
                counts = dict(cr.get_counts())
                return _normalize_counts_to_total(counts, self.shots)
            # Otherwise, search any attribute that exposes get_counts()
            for name in dir(data):
                obj = getattr(data, name)
                if hasattr(obj, "get_counts"):
                    counts = dict(obj.get_counts())
                    return _normalize_counts_to_total(counts, self.shots)
        except Exception:
            pass

        # Path B: explicit samples
        try:
            samples = getattr(data, "samples", None)
            if samples is not None:
                out: Dict[str, int] = {}
                for s in samples:
                    key = s if isinstance(s, str) else "".join(map(str, s))
                    out[key] = out.get(key, 0) + 1
                return _normalize_counts_to_total(out, self.shots)
        except Exception:
            pass

        # Path C: quasi distributions
        try:
            quasi = getattr(data, "quasi_dists", None)
            if quasi is None:
                quasi = getattr(data, "quasi_dist", None)
            if quasi is not None:
                q0 = dict(quasi[0]) if isinstance(quasi, list) else dict(quasi)
                return _normalize_counts_to_total(_quasi_to_counts(q0, self.shots), self.shots)
        except Exception:
            pass

        # Path D: rare 'meas.get_counts()' container
        try:
            meas = getattr(data, "meas", None)
            if meas is not None and hasattr(meas, "get_counts"):
                counts = dict(meas.get_counts())
                return _normalize_counts_to_total(counts, self.shots)
        except Exception:
            pass

        raise RuntimeError("Unsupported SamplerV2 result format; cannot extract counts.")



def make_backend(kind: str, shots: int, seed_sim: Optional[int], ibm_backend: Optional[str]) -> SamplerBackend:
    kind = (kind or "simulator").lower()
    if kind == "simulator":
        return LocalSimulatorBackend(shots=shots, seed=seed_sim)
    elif kind == "ibm":
        return IBMSamplerBackend(shots=shots, backend_name=ibm_backend)
    else:
        raise ValueError(f"Unknown backend kind: {kind}")
