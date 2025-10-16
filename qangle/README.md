# qangle

Continuous-angle quantum modeling for protein fragments using a low-order Fourier energy mapped to Pauli expectation values.

- Encodes each dihedral angle `θ` with a single `RY(θ)`.
- Uses identities ⟨Z⟩=cosθ, ⟨X⟩=sinθ, and cos(θ_i-θ_j)=⟨Z_i Z_j⟩+⟨X_i X_j⟩.
- Builds `SparsePauliOp` observables from Fourier and pairwise-coupling coefficients.
- Includes a local **StatevectorEstimator** (no external backends) and a simple **SPSA** optimizer.
- Ready to swap-in IBM Runtime EstimatorV2 when needed.

## Install (editable)
```bash
pip install -e .
```

## Quickstart
```python
from qangle import FourierAngleEnergyBuilder, LocalStatevectorEstimator, AngleCircuit, spsa_minimize

# 2 angles (one residue: φ, ψ)
builder = FourierAngleEnergyBuilder(n_angles=2, fourier_orders=[2,2])
# add single-angle terms: a*cosθ + b*sinθ for angle 0 and 1
builder.add_single(0, a=[0.8, 0.2], b=[-0.1, 0.0])  # up to order=2
builder.add_single(1, a=[0.5, 0.1], b=[0.0, -0.05])

# add coupling cos(θ0-θ1) with weight 0.6  -> XX + ZZ with coeff 0.6
builder.add_coupling(0,1, weight=0.6)

H = builder.to_sparse_pauli(n_qubits=2)  # SparsePauliOp
circ = AngleCircuit(n_angles=2, entangle="none")
est = LocalStatevectorEstimator()

def energy(thetas):
    qc = circ.build(thetas)
    return est.expectation(qc, H)

E0 = energy([0.1, -1.2])
print("E([0.1,-1.2]) =", E0)

# optimize with SPSA
theta0 = [0.0, 0.0]
opt = spsa_minimize(energy, theta0, maxiter=200, a=0.1, c=0.1)
print(opt)
```

See `examples/demo_tripeptide.py` for a 3-residue toy with nearest-neighbor couplings.
