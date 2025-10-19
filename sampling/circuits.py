# --*-- conding:utf-8 --*--
# @time:10/19/25 10:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:circuits.py

from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter


def build_ansatz(n_qubits: int, reps: int = 1, entanglement: str = "linear") -> EfficientSU2:
    """
    Build a parameterized ansatz circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    reps : int
        Number of repetition layers in EfficientSU2.
    entanglement : str
        Entanglement pattern ("linear", "full", etc.).

    Returns
    -------
    EfficientSU2
        The ansatz circuit object (unbound parameters).
    """
    return EfficientSU2(n_qubits, entanglement=entanglement, reps=reps)


def random_params(ansatz: EfficientSU2, seed: int) -> np.ndarray:
    """
    Generate a reproducible random parameter vector for the ansatz.

    Parameters
    ----------
    ansatz : EfficientSU2
        The ansatz circuit.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        A vector of parameter values in radians.
    """
    rng = np.random.default_rng(seed)
    return 2 * np.pi * rng.random(ansatz.num_parameters)


def apply_beta_layer(
    circ: QuantumCircuit,
    H: SparsePauliOp,
    beta: float,
    trotter: int = 1
) -> QuantumCircuit:
    """
    Optionally apply an evolution layer e^{-i β H} to bias sampling.

    Parameters
    ----------
    circ : QuantumCircuit
        Circuit to modify.
    H : SparsePauliOp
        Hamiltonian operator.
    beta : float
        Evolution time (β). If 0, the circuit is unchanged.
    trotter : int
        Number of Trotter steps for Suzuki-Trotter decomposition.

    Returns
    -------
    QuantumCircuit
        The modified circuit.
    """
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
    """
    Construct a complete sampling circuit:
    ansatz → optional β evolution → measurement.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    H : SparsePauliOp
        Problem Hamiltonian.
    beta : float
        Evolution parameter.
    params : np.ndarray
        Parameter vector for the ansatz.
    reps : int
        Repetitions in the ansatz.
    entanglement : str
        Entanglement topology.
    trotter : int
        Number of Suzuki-Trotter steps.

    Returns
    -------
    QuantumCircuit
        Fully parameterized and measured circuit.
    """
    ans = build_ansatz(n_qubits, reps=reps, entanglement=entanglement)
    circ = QuantumCircuit(n_qubits)
    circ.compose(ans.bind_parameters(params), inplace=True)
    circ = apply_beta_layer(circ, H, beta, trotter=trotter)
    circ.measure_all()
    return circ
