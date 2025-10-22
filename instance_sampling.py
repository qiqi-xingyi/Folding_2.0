# --*-- coding:utf-8 --*--
# @time:10/21/25 11:09
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:instance_sampling.py

import os
from typing import Dict, Any, List
import pandas as pd

from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

from sampling import SamplingRunner, SamplingConfig, BackendConfig
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

IBM_CONFIG_FILE = "./ibm_config.txt"

def read_ibm_config(path: str) -> Dict[str, str]:
    """Read IBM Quantum credentials (TOKEN, INSTANCE, BACKEND) from a simple config file."""
    cfg: Dict[str, str] = {}
    try:
        with open(path, "r") as f:
            for line in f:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                cfg[key.strip().upper()] = value.strip()
    except Exception as e:
        print(f"Failed to read IBM config: {e}")
    return cfg

cfg_data = read_ibm_config(IBM_CONFIG_FILE)
IBM_TOKEN = cfg_data.get("TOKEN", "")
IBM_INSTANCE = cfg_data.get("INSTANCE", None)
IBM_BACKEND_NAME = cfg_data.get("BACKEND", None)


PENALTY_PARAMS = (10, 10, 10)
BETA_LIST: List[float] = [0.0, 0.5, 1.0]
SEEDS: int = 4
REPS: int = 1


GROUP_COUNT = 10
SHOTS_PER_GROUP = 2000

EXAMPLES: List[Dict[str, Any]] = [
    {"protein_name": "6mu3", "main_chain_residue_seq": "YAGYS"},
    {"protein_name": "1ppi", "main_chain_residue_seq": "PWWERYQP"},
    {"protein_name": "1m7y", "main_chain_residue_seq": "TAGATSANE"},
    {"protein_name": "4f5y", "main_chain_residue_seq": "GLAWSYYIGYL"},
    {"protein_name": "4zb8", "main_chain_residue_seq": "YFASGQPYRYER"}
]


def init_ibm_service() -> QiskitRuntimeService:
    if IBM_TOKEN:
        try:
            return QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=IBM_TOKEN,
                instance=IBM_INSTANCE
            )
        except Exception:
            return QiskitRuntimeService()
    return QiskitRuntimeService()



def build_protein_hamiltonian(sequence: str, penalties: tuple[int, int, int]) -> SparsePauliOp:
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


def per_example_sampling(protein_name: str, sequence: str) -> str:
    print(f"\n=== Running {protein_name} ({sequence}) ===")
    H = build_protein_hamiltonian(sequence, PENALTY_PARAMS)

    group_csvs: List[str] = []
    for group_id in range(GROUP_COUNT):
        cfg = SamplingConfig(
            L=len(sequence),
            betas=list(BETA_LIST),
            seeds=SEEDS,
            reps=REPS,
            entanglement="linear",
            label=f"qsad_ibm_{protein_name}_g{group_id}",
            backend=BackendConfig(
                kind="ibm",
                shots=SHOTS_PER_GROUP,
                seed_sim=None,
                ibm_backend=IBM_BACKEND_NAME,
            ),
            out_csv=f"samples_{protein_name}_group{group_id}_ibm.csv",
            extra_meta={
                "protein": protein_name,
                "sequence": sequence,
                "group_id": group_id,
                "shots": SHOTS_PER_GROUP,
            },
        )
        runner = SamplingRunner(cfg, H)
        df = runner.run()
        print(f"[Group {group_id}] wrote {len(df)} rows -> {cfg.out_csv}")
        group_csvs.append(cfg.out_csv)

    # Combine all group results for this protein
    combined = []
    for fpath in group_csvs:
        try:
            df = pd.read_csv(fpath)
            combined.append(df)
        except Exception:
            pass
    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        merged_csv = f"samples_{protein_name}_all_ibm.csv"
        all_df.to_csv(merged_csv, index=False)
        print(f"[Merged] {protein_name}: {len(all_df)} rows -> {merged_csv}")
        return merged_csv
    return ""

if __name__ == "__main__":

    service = init_ibm_service()

    all_combined = []
    for ex in EXAMPLES:
        merged_path = per_example_sampling(ex["protein_name"], ex["main_chain_residue_seq"])
        if merged_path:
            try:
                df = pd.read_csv(merged_path)
                all_combined.append(df)
            except Exception:
                pass

    if all_combined:
        all_df = pd.concat(all_combined, ignore_index=True)
        out_all = "samples_all_ibm.csv"
        all_df.to_csv(out_all, index=False)
        print(f"\n[Global merged] all proteins -> {out_all} ({len(all_df)} rows)")
    print("\nAll sampling runs completed.")
