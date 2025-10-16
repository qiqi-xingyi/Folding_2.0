from __future__ import annotations
from typing import Iterable, Tuple, List
from qiskit.quantum_info import SparsePauliOp

def _pauli_label(n_qubits: int, ops: List[Tuple[str,int]]) -> str:
    """
    Build a label like "IIXZI" given a list of (op, index) with index in [0, n_qubits-1].
    Qiskit uses little-endian strings; we'll use the conventional ordering where qubit 0 is the rightmost.
    """
    label = ["I"] * n_qubits
    for p, idx in ops:
        assert p in ("I","X","Y","Z")
        assert 0 <= idx < n_qubits
        label[n_qubits - 1 - idx] = p
    return "".join(label)

def pauli_from_terms(n_qubits: int, terms: Iterable[Tuple[List[Tuple[str,int]], float]]) -> SparsePauliOp:
    """
    terms: iterable of ([(op,qubit), ...], coeff)
    Example: ([("Z",0),("Z",1)], 0.5) means 0.5 * Z0 Z1
    """
    paulis, coeffs = [], []
    for ops, c in terms:
        paulis.append(_pauli_label(n_qubits, ops))
        coeffs.append(c)
    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))
