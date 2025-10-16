from __future__ import annotations
from typing import List, Optional, Tuple, Iterable
from dataclasses import dataclass, field
from qiskit.quantum_info import SparsePauliOp
from .observables import pauli_from_terms

@dataclass
class FourierAngleEnergyBuilder:
    """
    Construct an energy observable H(theta) as a linear combination of Pauli expectations
    derived from low-order Fourier expansions of single-angle and pairwise difference terms.

    For single angle i:
        V_i(θ_i) ≈ a1*cos θ_i + b1*sin θ_i + a2*cos 2θ_i + b2*sin 2θ_i + ...
    Using RY encoding and identities:
        ⟨Z_i⟩ = cos θ_i, ⟨X_i⟩ = sin θ_i
        cos 2θ = 2cos^2 θ - 1 = 2⟨Z⟩^2 - 1  -> we approximate using additional Pauli terms:
        For low order, we recommend up to k=2 and linearize via auxiliary trick:
           cos(2θ) = ⟨Z⟩^2*2 - 1  -> approximate by adding a ZZ self-penalty with a small coefficient if needed,
           or simply drop k>=2 for a minimal model.
    Pair coupling (difference form):
        w_ij * cos(θ_i - θ_j) = w_ij*(⟨Z_i Z_j⟩ + ⟨X_i X_j⟩)

    This builder focuses on k=1 exactly and allows k=2 as an optional approximation term with
    a pragmatic mapping:
        cos(2θ_i) ≈ α_ZZ * (Z_i Z_i) + α_I * I  (since Z_i Z_i = I, we fold into constants).
    In practice, keeping k=1 already yields smooth landscapes.
    """
    n_angles: int
    fourier_orders: List[int] = field(default_factory=lambda: [])
    constant: float = 0.0
    single_terms: List[Tuple[int, List[float], List[float]]] = field(default_factory=list)
    couplings: List[Tuple[int,int,float]] = field(default_factory=list)

    def add_constant(self, c0: float):
        self.constant += float(c0)

    def add_single(self, idx: int, a: List[float], b: List[float]):
        """
        Add single-angle Fourier series coefficients for angle `idx`.
        a[k-1] multiplies cos(kθ), b[k-1] multiplies sin(kθ). Typically keep up to order 1 or 2.
        """
        assert 0 <= idx < self.n_angles
        self.single_terms.append((idx, list(a), list(b)))

    def add_coupling(self, i: int, j: int, weight: float):
        """
        Add w * cos(θ_i - θ_j) which maps to w*(Z_i Z_j + X_i X_j).
        """
        assert 0 <= i < self.n_angles and 0 <= j < self.n_angles and i != j
        self.couplings.append((i,j,float(weight)))

    def to_sparse_pauli(self, n_qubits: Optional[int]=None) -> SparsePauliOp:
        """
        Map the accumulated terms to a SparsePauliOp over `n_qubits` (default: n_angles).
        Uses exact k=1 mapping; k>=2 folds into constants (approx) to keep the model minimal.
        """
        if n_qubits is None:
            n_qubits = self.n_angles
        terms = []
        # constant
        if abs(self.constant) > 0.0:
            terms.append(([], self.constant))

        # k=1 exact terms: a1*cosθ + b1*sinθ -> a1*Z_i + b1*X_i
        for idx, a, b in self.single_terms:
            if len(a) >= 1 and abs(a[0]) > 0:
                terms.append(( [("Z", idx)], float(a[0]) ))
            if len(b) >= 1 and abs(b[0]) > 0:
                terms.append(( [("X", idx)], float(b[0]) ))
            # optional: fold k=2 into constant (very rough); better is to increase order later with richer mapping
            if len(a) >= 2 and abs(a[1]) > 0:
                # cos(2θ) ≈ γ0*I + γ1*Z_i  (first-order Taylor around current solution would be better; keep simple)
                # Here we just add a small bias into I to avoid over-parameterizing:
                terms.append(( [], float(-abs(a[1]))*0.5 ))
            if len(b) >= 2 and abs(b[1]) > 0:
                # sin(2θ) average to ~0 over symmetric priors -> ignore for minimal model
                pass

        # couplings: w*(ZZ + XX)
        for i,j,w in self.couplings:
            if abs(w) > 0:
                terms.append(( [("Z",i),("Z",j)], float(w) ))
                terms.append(( [("X",i),("X",j)], float(w) ))

        return pauli_from_terms(n_qubits, terms)
