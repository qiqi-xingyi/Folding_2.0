from __future__ import annotations
from typing import Iterable, Literal, Optional
from qiskit import QuantumCircuit

class AngleCircuit:
    """
    Builds a parameterized circuit where each angle θ is encoded by RY(θ) on one qubit.
    Optional shallow entanglement patterns to increase expressivity.
    """
    def __init__(self, n_angles: int, entangle: Literal["none","line","ring"]="none"):
        self.n_angles = int(n_angles)
        self.entangle = entangle

    def build(self, thetas: Iterable[float]) -> QuantumCircuit:
        thetas = list(thetas)
        assert len(thetas) == self.n_angles, "len(thetas) must equal n_angles"
        qc = QuantumCircuit(self.n_angles, name="angles")
        for i, th in enumerate(thetas):
            qc.ry(th, i)
        if self.entangle == "line":
            for i in range(self.n_angles-1):
                qc.cx(i, i+1)
        elif self.entangle == "ring":
            for i in range(self.n_angles-1):
                qc.cx(i, i+1)
            if self.n_angles > 2:
                qc.cx(self.n_angles-1, 0)
        return qc
