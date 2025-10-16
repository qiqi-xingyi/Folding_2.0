from __future__ import annotations
from typing import Iterable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

class LocalStatevectorEstimator:
    """
    Analytic estimator that evaluates <psi|H|psi> using a statevector simulation.
    No external backends required. Suitable for prototyping and unit tests.
    """
    def __init__(self):
        pass

    def expectation(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> float:
        psi = Statevector.from_instruction(circuit)
        # SparsePauliOp @ statevector
        H = observable.to_operator()
        val = np.real((psi.data.conj().T @ (H.data @ psi.data)))
        return float(val)
