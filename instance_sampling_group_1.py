# --*-- conding:utf-8 --*--
# @time:10/23/25 14:47
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:instance_sampling_group_1.py

import time
import random
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

from sampling import SamplingRunner, SamplingConfig, BackendConfig
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem


IBM_CONFIG_FILE = "./ibm_config.txt"
TASKS_FILE = "./tasks.csv"

PENALTY_PARAMS: Tuple[int, int, int] = (10, 10, 10)
BETA_LIST: List[float] = [1.0, 2.0, 3.0, 4.0]
SEEDS: int = 3
REPS: int = 1

GROUP_COUNT = 10
SHOTS_PER_GROUP = 2000

OUTPUT_ROOT = Path("sampling_results")  # Root directory for results


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read progress JSON {path}: {e}")
        return {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_progress(protein_name: str, sequence: str, out_dir: Path) -> Dict[str, Any]:
    now = utc_now_iso()
    return {
        "created_at": now,
        "updated_at": now,
        "pdbid": protein_name,
        "sequence": sequence,
        "status": "pending",
        "merged_csv": str(out_dir / f"samples_{protein_name}_all_ibm.csv"),
        "settings": {
            "PENALTY_PARAMS": list(PENALTY_PARAMS),
            "BETA_LIST": list(BETA_LIST),
            "SEEDS": SEEDS,
            "REPS": REPS,
            "GROUP_COUNT": GROUP_COUNT,
            "SHOTS_PER_GROUP": SHOTS_PER_GROUP,
            "IBM_BACKEND_NAME": IBM_BACKEND_NAME,
        },
        "groups": {},
    }


def load_timing_lookup(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read timing CSV {path}: {e}")
        return {}

    lookup: Dict[int, Dict[str, Any]] = {}
    if "group_id" not in df.columns:
        return lookup

    for _, row in df.iterrows():
        try:
            group_id = int(row["group_id"])
        except Exception:
            continue
        item: Dict[str, Any] = {}
        if "run_seed" in df.columns and pd.notna(row["run_seed"]):
            item["run_seed"] = int(row["run_seed"])
        if "rows" in df.columns and pd.notna(row["rows"]):
            item["rows"] = int(row["rows"])
        if "seconds" in df.columns and pd.notna(row["seconds"]):
            item["seconds"] = round(float(row["seconds"]), 6)
        lookup[group_id] = item
    return lookup


def ensure_progress(
    progress_path: Path,
    protein_name: str,
    sequence: str,
    out_dir: Path,
    timing_csv: Path,
) -> Dict[str, Any]:
    progress = load_json(progress_path) or default_progress(protein_name, sequence, out_dir)
    groups = progress.get("groups")
    if not isinstance(groups, dict):
        groups = {}
    progress["groups"] = groups
    progress["pdbid"] = protein_name
    progress["sequence"] = sequence
    progress["merged_csv"] = str(out_dir / f"samples_{protein_name}_all_ibm.csv")
    progress["settings"] = default_progress(protein_name, sequence, out_dir)["settings"]
    progress.setdefault("created_at", utc_now_iso())

    timing_lookup = load_timing_lookup(timing_csv)
    changed = False

    for group_id in range(GROUP_COUNT):
        key = str(group_id)
        csv_path = out_dir / f"samples_{protein_name}_group{group_id}_ibm.csv"
        entry = groups.get(key)
        if not isinstance(entry, dict):
            entry = {}
            groups[key] = entry
            changed = True

        if entry.get("csv") != str(csv_path):
            entry["csv"] = str(csv_path)
            changed = True

        timing_item = timing_lookup.get(group_id, {})
        if csv_path.exists():
            try:
                row_count = int(len(pd.read_csv(csv_path)))
            except Exception as e:
                print(f"[Resume] existing CSV unreadable, will rerun group {group_id}: {csv_path} ({e})")
                if entry.get("status") == "done":
                    entry["status"] = "pending"
                    changed = True
                continue

            updates: Dict[str, Any] = {
                "status": "done",
                "rows": row_count,
            }
            if "finished_at" not in entry:
                updates["finished_at"] = utc_now_iso()
            if "run_seed" in timing_item:
                updates["run_seed"] = timing_item["run_seed"]
            elif "run_seed" in entry:
                updates["run_seed"] = entry["run_seed"]
            if "seconds" in timing_item:
                updates["seconds"] = timing_item["seconds"]

            for k, v in updates.items():
                if entry.get(k) != v:
                    entry[k] = v
                    changed = True
        else:
            if entry.get("status") == "done":
                entry["status"] = "pending"
                changed = True

    done_groups = [
        gid for gid in range(GROUP_COUNT)
        if groups.get(str(gid), {}).get("status") == "done"
    ]
    progress["status"] = "done" if len(done_groups) == GROUP_COUNT else ("in_progress" if done_groups else "pending")
    progress["updated_at"] = utc_now_iso()

    if changed or not progress_path.exists():
        save_json(progress_path, progress)
    return progress


def write_timing_from_progress(progress: Dict[str, Any], protein_name: str, timing_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for group_id in range(GROUP_COUNT):
        entry = progress.get("groups", {}).get(str(group_id), {})
        if entry.get("status") != "done":
            continue
        rows.append({
            "protein_name": protein_name,
            "group_id": group_id,
            "run_seed": entry.get("run_seed"),
            "rows": entry.get("rows"),
            "seconds": entry.get("seconds"),
        })

    if not rows:
        return

    timing_df = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)
    numeric_seconds = pd.to_numeric(timing_df["seconds"], errors="coerce").fillna(0.0)
    total_elapsed = round(float(numeric_seconds.sum()), 6)
    timing_df["total_seconds_for_protein"] = total_elapsed
    timing_df.to_csv(timing_csv, index=False)


def done_group_csvs(progress: Dict[str, Any]) -> List[Path]:
    csvs: List[Path] = []
    for group_id in range(GROUP_COUNT):
        entry = progress.get("groups", {}).get(str(group_id), {})
        csv_path = Path(entry.get("csv", ""))
        if entry.get("status") == "done" and csv_path.exists():
            csvs.append(csv_path)
    return csvs


def read_ibm_config(path: str) -> Dict[str, str]:
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


def read_tasks(path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "protein_name" in cols:
        pn_col = df.columns[cols.index("protein_name")]
    elif "pdbid" in cols:
        pn_col = df.columns[cols.index("pdbid")]
    elif "pdb_id" in cols:
        pn_col = df.columns[cols.index("pdb_id")]
    elif "protein" in cols:
        pn_col = df.columns[cols.index("protein")]
    else:
        raise ValueError("Input must contain column: protein_name / pdbid / pdb_id / protein")

    if "main_chain_residue_seq" in cols:
        seq_col = df.columns[cols.index("main_chain_residue_seq")]
    elif "sequence" in cols:
        seq_col = df.columns[cols.index("sequence")]
    else:
        raise ValueError("Input must contain column: main_chain_residue_seq or sequence")

    tasks: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        protein_name = str(row[pn_col]).strip()
        sequence = str(row[seq_col]).strip()
        if protein_name and sequence:
            tasks.append({"protein_name": protein_name, "main_chain_residue_seq": sequence})
    return tasks


def per_example_sampling(protein_name: str, sequence: str) -> str:
    print(f"\n=== Running {protein_name} ({sequence}) ===")

    out_dir = OUTPUT_ROOT / protein_name
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    timing_csv = out_dir / f"{protein_name}_timing.csv"
    merged_csv = out_dir / f"samples_{protein_name}_all_ibm.csv"

    progress = ensure_progress(progress_path, protein_name, sequence, out_dir, timing_csv)
    existing_done = done_group_csvs(progress)
    if len(existing_done) == GROUP_COUNT and merged_csv.exists():
        print(f"[Resume] {protein_name}: all groups already completed, reusing existing outputs.")
        write_timing_from_progress(progress, protein_name, timing_csv)
        return str(merged_csv)

    H = None

    for group_id in range(GROUP_COUNT):
        key = str(group_id)
        csv_path = out_dir / f"samples_{protein_name}_group{group_id}_ibm.csv"
        entry = progress.get("groups", {}).get(key, {})
        if entry.get("status") == "done" and csv_path.exists():
            print(f"[Resume] {protein_name}: skip finished group {group_id} -> {csv_path}")
            continue

        if H is None:
            H = build_protein_hamiltonian(sequence, PENALTY_PARAMS)

        run_seed = int(entry["run_seed"]) if entry.get("run_seed") is not None else random.randint(1, 2**31 - 1)
        entry.update({
            "csv": str(csv_path),
            "status": "running",
            "run_seed": run_seed,
            "started_at": entry.get("started_at", utc_now_iso()),
        })
        progress["groups"][key] = entry
        progress["status"] = "in_progress"
        progress["updated_at"] = utc_now_iso()
        save_json(progress_path, progress)

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
                seed_sim=None,  # Running on real backend, not simulator
                ibm_backend=IBM_BACKEND_NAME,
            ),
            out_csv=str(csv_path),
            extra_meta={
                "protein": protein_name,
                "sequence": sequence,
                "group_id": group_id,
                "run_seed": run_seed,
            },
        )

        runner = SamplingRunner(cfg, H)

        try:
            t0 = time.perf_counter()
            df = runner.run()
            t1 = time.perf_counter()
            elapsed = t1 - t0
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            progress["groups"][key] = entry
            progress["status"] = "in_progress"
            progress["updated_at"] = utc_now_iso()
            save_json(progress_path, progress)
            raise

        print(f"[Group {group_id}] wrote {len(df)} rows -> {cfg.out_csv} (time: {elapsed:.2f}s)")
        entry.update({
            "status": "done",
            "rows": int(len(df)),
            "seconds": round(elapsed, 6),
            "finished_at": utc_now_iso(),
        })
        entry.pop("error", None)
        progress["groups"][key] = entry
        progress["updated_at"] = utc_now_iso()
        save_json(progress_path, progress)
        write_timing_from_progress(progress, protein_name, timing_csv)

    progress = ensure_progress(progress_path, protein_name, sequence, out_dir, timing_csv)
    write_timing_from_progress(progress, protein_name, timing_csv)
    group_csvs = [str(path) for path in done_group_csvs(progress)]

    if len(group_csvs) != GROUP_COUNT:
        progress["status"] = "in_progress" if group_csvs else "pending"
        progress["updated_at"] = utc_now_iso()
        save_json(progress_path, progress)
        print(f"[Resume] {protein_name}: {len(group_csvs)}/{GROUP_COUNT} groups completed; merged file not refreshed.")
        return ""

    timing_df = pd.read_csv(timing_csv)
    total_elapsed = pd.to_numeric(timing_df["seconds"], errors="coerce").fillna(0.0).sum()
    print(f"[Timing] -> {timing_csv} (total: {total_elapsed:.2f}s)")

    combined = []
    for fpath in group_csvs:
        try:
            df = pd.read_csv(fpath)
            combined.append(df)
        except Exception:
            pass
    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        all_df.to_csv(merged_csv, index=False)
        progress["merged_csv"] = str(merged_csv)
        progress["status"] = "done"
        progress["updated_at"] = utc_now_iso()
        save_json(progress_path, progress)
        print(f"[Merged] {protein_name}: {len(all_df)} rows -> {merged_csv}")
        return str(merged_csv)
    return ""


if __name__ == "__main__":

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    service = init_ibm_service()

    EXAMPLES: List[Dict[str, Any]] = read_tasks(TASKS_FILE)

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
        out_all = OUTPUT_ROOT / "samples_all_ibm.csv"
        all_df.to_csv(out_all, index=False)
        print(f"\n[Global merged] all proteins -> {out_all} ({len(all_df)} rows)")

    print("\nAll sampling runs completed.")
