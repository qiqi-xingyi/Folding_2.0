# --*-- conding:utf-8 --*--
# @time:10/19/25 10:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:circuits.py

from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# EfficientSU2 import path compatibility (2.x vs 1.x)
try:
    from qiskit.circuit.library import EfficientSU2
except Exception:
    # Older path (unlikely needed, but kept as guard)
    from qiskit.circuit.library.n_local import EfficientSU2  # type: ignore

# Pauli evolution (paths are stable across 1.x/2.x)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter


def build_ansatz(n_qubits: int, reps: int = 1, entanglement: str = "linear") -> EfficientSU2:
    return EfficientSU2(n_qubits, entanglement=entanglement, reps=reps)


def random_params(ansatz: EfficientSU2, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 2 * np.pi * rng.random(ansatz.num_parameters)


def apply_beta_layer(circ: QuantumCircuit, H: SparsePauliOp, beta: float, trotter: int = 1) -> QuantumCircuit:
    if beta == 0:
        return circ
    evo = PauliEvolutionGate(H, time=beta, synthesis=SuzukiTrotter(order=2, reps=trotter))
    circ.append(evo, range(circ.num_qubits))
    return circ


def make_sampling_circuit(
    n_qubits: int,
    H: SparsePauliOp,
    beta: float,
    params: np.ndarray,
    reps: int = 1,
    entanglement: str = "linear",
    trotter: int = 1
) -> QuantumCircuit:
    ans = build_ansatz(n_qubits, reps=reps, entanglement=entanglement)
    circ = QuantumCircuit(n_qubits)
    circ.compose(ans.bind_parameters(params), inplace=True)
    apply_beta_layer(circ, H, beta, trotter=trotter)
    circ.measure_all()
    return circ

