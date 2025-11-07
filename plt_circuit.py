# --*-- conding:utf-8 --*--
# @time:11/7/25 15:00
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_circuit.py

# --*-- coding:utf-8 --*--
# @time: 11/07/25
# @author: Yuqi Zhang
# @desc: Build H from protein sequence, construct circuit using existing sampling logic,
#        remove measurements (no sampling), transpile/decompose to backend basis,
#        and save full circuits (raw / no-measure / transpiled) as images & QASM.

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import qiskit
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

from sampling.circuits import make_sampling_circuit, build_ansatz, random_params


OUTPUT_DIR = Path("circuit_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROTEINS: List[Dict[str, str]] = [
    {"protein_name": "4mo4", "sequence": "NIGGF"},
    # {"protein_name": "toy14", "sequence": "YLVTHLMGADLNNI"},
]

REPS: int = 1
ENTANGLEMENT: str = "linear"
SEED: int = 0
BETA: float = 2.0

IBM_BACKEND_NAME: Optional[str] = None
DEFAULT_BASIS = ["rz", "sx", "x", "cx", "id"]

PENALTY_PARAMS: Tuple[int, int, int] = (10, 10, 10)
DRAW_FORMATS = ("png", "pdf")


def build_protein_hamiltonian(sequence: str, penalties: Tuple[int, int, int]) -> SparsePauliOp:
    side_chain_residue_sequences = ['' for _ in range(len(sequence))]
    peptide = Peptide(sequence, side_chain_residue_sequences)
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(*penalties)
    problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    H = problem.qubit_op()
    if isinstance(H, (list, tuple)) and len(H) > 0:
        H = H[0]
    if not isinstance(H, SparsePauliOp):
        H = SparsePauliOp(H)
    return H


def remove_final_measurements(circ: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Safely remove all final measurements."""
    if hasattr(circ, "remove_final_measurements"):
        return circ.remove_final_measurements(inplace=False)
    new_circ = qiskit.QuantumCircuit(circ.num_qubits, circ.num_clbits)
    for instr, qargs, cargs in circ.data:
        if instr.name.lower() == "measure":
            continue
        new_circ.append(instr, qargs, cargs)
    new_circ.metadata = dict(circ.metadata) if circ.metadata else None
    new_circ.name = (circ.name or "qc") + "_no_measure"
    return new_circ


def get_backend_and_basis(backend_name: Optional[str]):
    """Return (backend_or_None, basis_gates_list)."""
    if backend_name:
        try:
            svc = QiskitRuntimeService()
            be = svc.backend(backend_name)
            cfg = be.configuration()
            if hasattr(be, "target") and be.target is not None:
                basis = list(be.target.operation_names)
                prefer = ["id", "rz", "sx", "x", "cx"]
                basis = [g for g in prefer if g in basis] or DEFAULT_BASIS
            else:
                basis = getattr(cfg, "basis_gates", None) or DEFAULT_BASIS
            return be, basis
        except Exception:
            pass
    return None, DEFAULT_BASIS


def save_circuit_draw(circ: qiskit.QuantumCircuit, out_stem: Path):
    """Save the circuit as QASM and diagram (PNG/PDF)."""
    try:
        qasm_text = circ.qasm()
        (out_stem.with_suffix(".qasm")).write_text(qasm_text)
    except Exception:
        pass

    for ext in DRAW_FORMATS:
        try:
            circ.draw(output="mpl", fold=-1, idle_wires=False, scale=1.0).figure.savefig(
                out_stem.with_suffix(f".{ext}"),
                bbox_inches="tight", dpi=300
            )
        except Exception:
            (out_stem.with_suffix(f".{ext}.txt")).write_text(circ.draw(output="text"))


def main():
    be, basis = get_backend_and_basis(IBM_BACKEND_NAME)

    for item in PROTEINS:
        name = item["protein_name"]
        seq = item["sequence"]

        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        H = build_protein_hamiltonian(seq, PENALTY_PARAMS)
        n_qubits = getattr(H, "num_qubits", None)
        if n_qubits is None:
            raise RuntimeError("Cannot infer num_qubits from Hamiltonian.")

        ans = build_ansatz(n_qubits, reps=REPS, entanglement=ENTANGLEMENT)
        params = random_params(ans, seed=SEED)

        raw_circ = make_sampling_circuit(
            n_qubits=n_qubits,
            H=H,
            beta=BETA,
            params=params,
            reps=REPS,
            entanglement=ENTANGLEMENT,
        )
        no_meas_circ = remove_final_measurements(raw_circ)

        transpiled = transpile(
            no_meas_circ,
            backend=be,
            basis_gates=basis,
            optimization_level=3,
        )

        save_circuit_draw(raw_circ,      out_dir / f"{name}_raw_beta{BETA}")
        save_circuit_draw(no_meas_circ,  out_dir / f"{name}_no_measure_beta{BETA}")
        save_circuit_draw(transpiled,    out_dir / f"{name}_transpiled_beta{BETA}")

        print(f"[OK] {name}: saved to {out_dir}")


if __name__ == "__main__":
    main()
