# --*-- conding:utf-8 --*--
# @time:10/19/25 10:30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_sampling.py

from qiskit.quantum_info import SparsePauliOp, Pauli
from sampling import SamplingRunner, SamplingConfig, BackendConfig

# ---------------------------------------------------------------------
# Example 1 — use a simple dummy Hamiltonian (for local testing)
# ---------------------------------------------------------------------
def demo_hamiltonian(n_qubits: int = 8) -> SparsePauliOp:
    """Example Hamiltonian: H = Z0 + Z1."""
    paulis = []
    coeffs = []
    for i in range(n_qubits):
        z = [False] * n_qubits
        x = [False] * n_qubits
        z[i] = True
        paulis.append(Pauli((z, x)))
        coeffs.append(1.0)
    return SparsePauliOp(paulis, coeffs)


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------
    cfg = SamplingConfig(
        L=5,
        betas=[0.0, 0.5, 1.0],
        seeds=4,
        reps=1,
        entanglement="linear",
        label="demo_sampling",
        backend=BackendConfig(kind="simulator", shots=1024, seed_sim=42),
        out_csv="samples_demo.csv",
        extra_meta={"protein": "6mu3", "sequence": "YAGYS"},
    )

    # --------------------------------------------------------------
    # Hamiltonian (replace this with your real one)
    # --------------------------------------------------------------
    H = demo_hamiltonian(6)

    # --------------------------------------------------------------
    # Run the sampling
    # --------------------------------------------------------------
    runner = SamplingRunner(cfg, H)
    df = runner.run()

    # --------------------------------------------------------------
    # Inspect results
    # --------------------------------------------------------------
    print("Sampling completed.")
    print(f"Wrote {len(df)} rows to {cfg.out_csv}")
    print(df.head())
