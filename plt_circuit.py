# --*-- conding:utf-8 --*--
# @time:11/7/25 15:00
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_circuit.py

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import subprocess
import shutil
import tempfile

import qiskit
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

from sampling.circuits import make_sampling_circuit, build_ansatz, random_params


# ----------- configuration -----------
OUTPUT_DIR = Path("circuit_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROTEINS: List[Dict[str, str]] = [
    {"protein_name": "4mo4", "sequence": "NIGGF"},

]

REPS: int = 1
ENTANGLEMENT: str = "linear"
SEED: int = 0
BETA: float = 2.0

IBM_BACKEND_NAME: Optional[str] = None
DEFAULT_BASIS = ["rz", "sx", "x", "cx", "id"]

PENALTY_PARAMS: Tuple[int, int, int] = (10, 10, 10)
DRAW_FORMATS = ("png", "pdf")

# LaTeX export and compile
EXPORT_LATEX = True
LATEX_STANDALONE = True
# order of compilers to try; latexmk is most reliable
LATEX_COMPILERS = ["latexmk", "pdflatex", "xelatex"]
# -------------------------------------


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


def _textdrawing_to_str(td) -> str:
    try:
        return td.single_string()
    except Exception:
        return str(td)


def _compile_latex_to_pdf(tex_path: Path, out_pdf_path: Path) -> bool:
    """Compile LaTeX into a PDF next to out_pdf_path. Returns True on success."""
    # Work in a temp dir to avoid clutter; then copy the PDF out.
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        local_tex = tdir / tex_path.name
        local_tex.write_text(tex_path.read_text())

        # Choose a compiler
        compiler = None
        for c in LATEX_COMPILERS:
            if shutil.which(c):
                compiler = c
                break
        if compiler is None:
            return False

        try:
            if compiler == "latexmk":
                cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", local_tex.name]
                subprocess.run(cmd, cwd=tdir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                produced = local_tex.with_suffix(".pdf")
            elif compiler in ("pdflatex", "xelatex"):
                # Run twice to stabilize references
                cmd = [compiler, "-interaction=nonstopmode", "-halt-on-error", local_tex.name]
                subprocess.run(cmd, cwd=tdir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                subprocess.run(cmd, cwd=tdir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                produced = local_tex.with_suffix(".pdf")
            else:
                return False

            if produced.exists():
                out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                out_pdf_path.write_bytes(produced.read_bytes())
                return True
        except subprocess.CalledProcessError:
            return False
    return False


def save_circuit_draw(circ: qiskit.QuantumCircuit, out_stem: Path):
    """Save QASM, image (PNG/PDF), LaTeX, and compile LaTeX to PDF. ASCII fallback when needed."""
    # QASM
    try:
        qasm_text = circ.qasm()
        (out_stem.with_suffix(".qasm")).write_text(qasm_text)
    except Exception:
        pass

    # Images via matplotlib; fallback to ASCII text diagrams
    for ext in DRAW_FORMATS:
        try:
            fig = circ.draw(output="mpl", fold=-1, idle_wires=False, scale=1.0).figure
            fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
        except Exception:
            td = circ.draw(output="text", fold=-1)
            (out_stem.with_suffix(f".{ext}.txt")).write_text(_textdrawing_to_str(td))

    # LaTeX export and compile
    if EXPORT_LATEX:
        try:
            latex_src = circ.draw(
                output="latex_source",
                fold=-1,
                idle_wires=False,
                scale=1.0,
                standalone=LATEX_STANDALONE,
            )
            if not isinstance(latex_src, str):
                latex_src = str(latex_src)
            tex_path = out_stem.with_suffix(".tex")
            tex_path.write_text(latex_src)

            # Compile to PDF named *_latex.pdf to avoid clobbering matplotlib PDF
            latex_pdf_path = out_stem.with_name(out_stem.name + "_latex").with_suffix(".pdf")
            ok = _compile_latex_to_pdf(tex_path, latex_pdf_path)
            if not ok:
                # Provide a readable fallback note if compilation is unavailable
                note = (
                    "LaTeX compilation failed or LaTeX toolchain not found. "
                    "Install a TeX distribution (e.g., MacTeX/TeX Live) or ensure 'latexmk'/'pdflatex' is on PATH."
                )
                (out_stem.with_suffix(".tex.txt")).write_text(note + "\n\n" + latex_src)
        except Exception:
            # As a last resort, also dump ASCII diagram to .tex.txt
            td = circ.draw(output="text", fold=-1)
            ascii_text = _textdrawing_to_str(td)
            (out_stem.with_suffix(".tex.txt")).write_text(
                "LaTeX export failed; ASCII diagram instead:\n\n" + ascii_text
            )


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

        print(f"[OK] {name}: outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
