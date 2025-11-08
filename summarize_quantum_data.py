# -*- coding: utf-8 -*-
# summarize_quantum_data.py
#
# Aggregate per-PDB quantum sampling data:
# sequence, qubits, circuit depth, average time per shot, effective samples.

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp

from sampling.circuits import make_sampling_circuit, build_ansatz, random_params
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

# --- configuration ---
BETA_LIST = [1.0, 2.0, 3.0, 4.0]
SEEDS = 3
REPS = 1
ENTANGLEMENT = "linear"
SHOTS_PER_CIRCUIT = 2000
PENALTY_PARAMS = (10, 10, 10)

ROOT = Path("quantum_data")
SUMMARY_CSV = ROOT / "quantum_data_summary.csv"


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


def robust_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, header=None)


def parse_sequence_and_qubits_from_samples(df: pd.DataFrame) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if df.empty:
        return None, None, None

    candidates_seq = [c for c in df.columns if str(c).lower() in ("sequence", "seq", "main_chain_residue_seq")]
    candidates_nq = [c for c in df.columns if str(c).lower() in ("n_qubits", "nqubits", "num_qubits")]
    candidates_L = [c for c in df.columns if str(c).lower() in ("l", "length", "seq_len")]

    seq = df[candidates_seq[0]].iloc[0] if candidates_seq else None
    nq = int(df[candidates_nq[0]].iloc[0]) if candidates_nq else None
    L = int(df[candidates_L[0]].iloc[0]) if candidates_L else None

    # fallback for headerless CSV written by the sampling pipeline
    if seq is None or nq is None or L is None:
        try:
            L = int(df.iloc[0, 0]) if L is None else L
            nq = int(df.iloc[0, 1]) if nq is None else nq
            seq = str(df.iloc[0, 10]) if seq is None else seq
        except Exception:
            pass

    return seq, nq, L


def compute_depth_stats(sequence: str, betas=BETA_LIST, seeds=SEEDS, reps=REPS) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Build circuits locally and transpile without a backend.
    Returns (depth_mean, depth_max, num_qubits).
    """
    H = build_protein_hamiltonian(sequence, PENALTY_PARAMS)
    n_qubits = getattr(H, "num_qubits", None)
    if n_qubits is None:
        return None, None, None

    ans = build_ansatz(n_qubits, reps=reps, entanglement=ENTANGLEMENT)
    depths: List[int] = []
    for s in range(seeds):
        params = random_params(ans, s)
        for beta in betas:
            qc = make_sampling_circuit(
                n_qubits=n_qubits, H=H, beta=beta,
                params=params, reps=reps, entanglement=ENTANGLEMENT
            )
            try:
                qct = transpile(qc, optimization_level=1)
                depths.append(int(qct.depth()))
            except Exception:
                pass

    if not depths:
        return None, None, n_qubits

    depth_mean = round(sum(depths) / len(depths), 2)
    depth_max = max(depths)
    return depth_mean, depth_max, n_qubits


def compute_avg_time_per_shot(timing_df: pd.DataFrame) -> Tuple[Optional[float], int, float]:
    """
    Average over groups: seconds / (len(BETA_LIST)*SEEDS*REPS*SHOTS_PER_CIRCUIT)
    """
    if timing_df.empty or "seconds" not in timing_df.columns:
        return None, 0, 0.0

    circuits_per_group = len(BETA_LIST) * SEEDS * REPS
    denom = circuits_per_group * SHOTS_PER_CIRCUIT

    per_group = []
    for _, row in timing_df.iterrows():
        sec = float(row["seconds"])
        if denom > 0:
            per_group.append(sec / denom)

    avg_time_per_shot = float(sum(per_group) / len(per_group)) if per_group else None
    seconds_total = float(timing_df["seconds"].sum())
    return avg_time_per_shot, len(per_group), seconds_total


def main():
    rows_out: List[Dict] = []

    if not ROOT.exists():
        print(f"[ERROR] Folder not found: {ROOT}")
        return

    for sub in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        pdb_id = sub.name
        timing_csv = next(sub.glob(f"{pdb_id}_timing.csv"), None)
        samples_csv = next(sub.glob(f"samples_{pdb_id}_all_ibm.csv"), None)
        if not timing_csv or not samples_csv:
            continue

        tdf = robust_read_csv(timing_csv)
        sdf = robust_read_csv(samples_csv)

        sequence, nq_in_csv, L_in_csv = parse_sequence_and_qubits_from_samples(sdf)
        effective_samples = int(len(sdf))

        avg_tps, groups, seconds_total = compute_avg_time_per_shot(tdf)
        depth_mean, depth_max, nq_rebuild = compute_depth_stats(sequence)
        n_qubits = nq_rebuild if nq_rebuild is not None else nq_in_csv

        rows_out.append({
            "pdb_id": pdb_id,
            "sequence": sequence,
            "L": L_in_csv,
            "n_qubits": n_qubits,
            "depth_mean": depth_mean,
            "depth_max": depth_max,
            "avg_time_per_shot_sec": None if avg_tps is None else round(avg_tps, 6),
            "groups": groups,
            "seconds_total": round(seconds_total, 3),
            "effective_samples": effective_samples,
        })

    if not rows_out:
        print("[WARN] No valid PDB folders with both timing and samples CSV were found.")
        return

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(SUMMARY_CSV, index=False)
    print(f"[OK] Wrote summary -> {SUMMARY_CSV}")
    print(df_out.head(min(10, len(df_out))).to_string(index=False))


if __name__ == "__main__":
    main()
