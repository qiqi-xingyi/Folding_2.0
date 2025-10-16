"""
Demo: 3-residue fragment -> 3 angles (for simplicity) with nearest-neighbor couplings.
Run:
    python -m qangle.examples.demo_tripeptide
or from repo root:
    python examples/demo_tripeptide.py
"""
import math
from qangle import FourierAngleEnergyBuilder, LocalStatevectorEstimator, AngleCircuit, spsa_minimize

def main():
    n = 3  # three angles θ0, θ1, θ2
    builder = FourierAngleEnergyBuilder(n_angles=n)

    # Single-angle priors (Ramachandran-like, order-1 terms only for demo)
    builder.add_single(0, a=[0.8], b=[0.1])
    builder.add_single(1, a=[0.5], b=[-0.2])
    builder.add_single(2, a=[0.6], b=[0.0])

    # Nearest-neighbor couplings
    builder.add_coupling(0,1, weight=0.4)
    builder.add_coupling(1,2, weight=0.4)

    H = builder.to_sparse_pauli(n_qubits=n)
    circ = AngleCircuit(n_angles=n, entangle="line")
    est = LocalStatevectorEstimator()

    def energy(thetas):
        qc = circ.build(thetas)
        return est.expectation(qc, H)

    theta0 = [0.0, 0.0, 0.0]
    print("E(theta0) =", energy(theta0))

    res = spsa_minimize(energy, theta0, maxiter=200, a=0.1, c=0.1)
    print("Result:", res)

if __name__ == "__main__":
    main()
