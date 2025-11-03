# --*-- coding:utf-8 --*--
# @time:11/2/25 22:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:compute_refined_rmsd.py
#
# Compute RMSD for each test_output/<pdb_id>/refined_ca.pdb against reference CA
# defined by dataset/benchmark_info.txt (residue span) and PDBbind pocket/protein.
# Picks the best-matching reference chain and aligns with Kabsch. Writes a single CSV.

import argparse
import csv
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

import numpy as np

# ---------- IO utils ----------

def read_benchmark_info(path: Path) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = [h.strip() for h in line.split("\t")]
                continue
            parts = [p.strip() for p in line.split("\t")]
            rec = dict(zip(header, parts))
            pdb = rec["pdb_id"]
            idx[pdb] = rec
    return idx


def parse_residue_span(span: str) -> Tuple[int, int]:
    s = span.strip()
    if "-" not in s:
        raise ValueError(f"Invalid residue span: {s}")
    a, b = s.split("-", 1)
    return int(a), int(b)


# ---------- PDB parsing ----------

def load_ca_coords_refined(path: Path) -> np.ndarray:
    # refined_ca.pdb produced by our pipeline is CA-only with residue indices 1..L
    if not path.exists():
        raise FileNotFoundError(f"Missing refined PDB: {path}")
    ca = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            ca.append([x, y, z])
    if not ca:
        raise ValueError(f"No CA atoms found in refined PDB: {path}")
    return np.asarray(ca, dtype=np.float64)


def load_reference_ca_by_chain(pdb_path: Path, start_res: int, end_res: int) -> Dict[str, np.ndarray]:
    chains: Dict[str, np.ndarray] = {}
    if not pdb_path.exists():
        return chains
    tmp: Dict[str, List[List[float]]] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain_id = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq < start_res or resseq > end_res:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            tmp.setdefault(chain_id, []).append([x, y, z])
    for c, arr in tmp.items():
        if arr:
            chains[c] = np.asarray(arr, dtype=np.float64)
    return chains


# ---------- Geometry ----------

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
    # Align P to Q; both (N,3)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("Input shapes must be (N,3) and equal.")
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    C = P0.T @ Q0
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    P_rot = P0 @ R
    P_aligned = P_rot + Qc
    diff = P_aligned - Q
    rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    return P_aligned, float(rmsd)


# ---------- Discovery ----------

def discover_refined_targets(out_root: Path) -> List[str]:
    out = []
    if not out_root.exists():
        return out
    for d in out_root.iterdir():
        if not d.is_dir():
            continue
        pdb_id = d.name
        if (d / "refined_ca.pdb").exists():
            out.append(pdb_id)
    return sorted(out)


# ---------- Main computation ----------

def compute_one(pdb_id: str,
                final_root: Path,
                pdbbind_root: Path,
                bench: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    refined_path = final_root / pdb_id / "refined_ca.pdb"
    try:
        Q = load_ca_coords_refined(refined_path)
    except Exception as e:
        return {"pdb_id": pdb_id, "status": f"refined_load_failed: {e}"}

    if pdb_id not in bench:
        return {"pdb_id": pdb_id, "status": "no_benchmark_entry"}

    try:
        start_res, end_res = parse_residue_span(bench[pdb_id]["Residues"])
    except Exception as e:
        return {"pdb_id": pdb_id, "status": f"bad_span: {e}"}

    pocket_p = pdbbind_root / pdb_id / f"{pdb_id}_pocket.pdb"
    protein_p = pdbbind_root / pdb_id / f"{pdb_id}_protein.pdb"

    ref_src = "pocket"
    ref_chains = load_reference_ca_by_chain(pocket_p, start_res, end_res)
    ref_path_used = pocket_p
    if not ref_chains:
        ref_chains = load_reference_ca_by_chain(protein_p, start_res, end_res)
        ref_src = "protein"
        ref_path_used = protein_p
    if not ref_chains:
        return {"pdb_id": pdb_id, "status": "reference_not_found"}

    Lq = Q.shape[0]
    chain_choices = [(cid, arr.shape[0], arr) for cid, arr in ref_chains.items()]
    # Prefer closest length to refined; tie-breaker prefers longer chain
    chain_choices.sort(key=lambda x: (abs(x[1] - Lq), -x[1]))
    best = None

    for cid, Lref, Pfull in chain_choices:
        Lc = min(Lq, Lref)
        if Lc < 3:
            continue
        Q_use = Q[:Lc]
        P_use = Pfull[:Lc]
        try:
            _, r = kabsch_align(Q_use, P_use)
        except Exception:
            continue
        if (best is None) or (r < best["rmsd"]):
            best = {
                "pdb_id": pdb_id,
                "ref_source": ref_src,
                "ref_path": str(ref_path_used),
                "chain": cid,
                "L_refined": Lq,
                "L_ref": Lref,
                "L_common": Lc,
                "rmsd": float(r),
                "refined_path": str(refined_path),
                "status": "ok",
            }

    if best is None:
        return {"pdb_id": pdb_id, "status": "align_failed_or_too_short"}

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_root", type=str, default="test_output", help="Directory with per-PDB refined_ca.pdb")
    ap.add_argument("--dataset_root", type=str, default="dataset", help="Dataset root containing Pdbbind/")
    ap.add_argument("--bench_info", type=str, default="dataset/benchmark_info.txt", help="Benchmark index with residue spans")
    ap.add_argument("--only", type=str, default="", help="Comma-separated pdb_ids to restrict processing, e.g. '1e2k,4f5y'")
    ap.add_argument("--out_csv", type=str, default="test_rmsd_summary.csv", help="Output CSV path")
    args = ap.parse_args()

    final_root = Path(args.final_root)
    pdbbind_root = Path(args.dataset_root) / "Pdbbind"
    bench = read_benchmark_info(Path(args.bench_info))

    pdb_ids = discover_refined_targets(final_root)
    if args.only.strip():
        allow = {p.strip().lower() for p in args.only.split(",") if p.strip()}
        pdb_ids = [p for p in pdb_ids if p.lower() in allow]

    rows: List[Dict[str, Any]] = []
    for pdb_id in pdb_ids:
        rec = compute_one(pdb_id, final_root, pdbbind_root, bench)
        if rec is None:
            continue
        rows.append(rec)

    # Write CSV
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id", "status", "rmsd",
        "L_refined", "L_ref", "L_common",
        "ref_source", "chain",
        "refined_path", "ref_path"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "pdb_id": r.get("pdb_id", ""),
                "status": r.get("status", ""),
                "rmsd": f'{r.get("rmsd"):.4f}' if isinstance(r.get("rmsd"), (int, float)) else "",
                "L_refined": r.get("L_refined", ""),
                "L_ref": r.get("L_ref", ""),
                "L_common": r.get("L_common", ""),
                "ref_source": r.get("ref_source", ""),
                "chain": r.get("chain", ""),
                "refined_path": r.get("refined_path", ""),
                "ref_path": r.get("ref_path", ""),
            })

    print(f"[DONE] Wrote {len(rows)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
