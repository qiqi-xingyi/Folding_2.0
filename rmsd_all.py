# --*-- conding:utf-8 --*--
# @time:10/26/25 22:07
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd_all.py


"""
find_best_rmsd.py

Scan a JSON or CSV file of decoded structures, compute CA-aligned RMSD to the
ground-truth segment, and select the best candidate.

Examples:
  python find_best_rmsd.py \
    --pdb-id 1m7y \
    --index dataset/benchmark_info.txt \
    --data-root dataset/Pdbbind \
    --json decoded_structs.json \
    --out-dir e_results/1m7y/best_from_json

  python find_best_rmsd.py \
    --pdb-id 1m7y \
    --index dataset/benchmark_info.txt \
    --data-root dataset/Pdbbind \
    --csv cluster_results.csv \
    --pos-col main_positions \
    --out-dir e_results/1m7y/best_from_csv
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd


# ---------- math ----------
def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return R, t that aligns Q to P (both (N,3)), minimizing ||P - (R Q + t)||."""
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Qc.T @ Pc
    V, S, Wt = np.linalg.svd(C)
    R = V @ Wt
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ Wt
    t = P.mean(axis=0) - (R @ Q.mean(axis=0))
    return R, t


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    dif = A - B
    return float(np.sqrt((dif * dif).sum() / A.shape[0]))


# ---------- PDB IO ----------
def write_ca_pdb(path: str, ca_xyz: np.ndarray, chain_id: str = "A"):
    with open(path, "w", encoding="utf-8") as f:
        for i, (x, y, z) in enumerate(ca_xyz, start=1):
            f.write(
                "ATOM  {serial:5d}  CA  GLY {chain}{resi:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n".format(
                    serial=i, chain=chain_id, resi=i, x=float(x), y=float(y), z=float(z)
                )
            )
        f.write("END\n")


def read_ca_coords_from_pdb(path: str):
    out = []
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            name = line[12:16].strip()
            if name != "CA":
                continue
            chain = line[21].strip() or ""
            resseq = line[22:26].strip()
            icode = line[26].strip() or ""
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except Exception:
                continue
            out.append(dict(chain=chain, resseq=int(resseq), icode=icode,
                            xyz=np.array([x, y, z], dtype=float)))
    if not out:
        raise RuntimeError(f"No CA atoms found in {path}")
    return out


def build_chain_index(ca_list):
    idx: Dict[str, Dict[Tuple[int, str], np.ndarray]] = {}
    for rec in ca_list:
        ch = rec["chain"]
        key = (rec["resseq"], rec["icode"])
        idx.setdefault(ch, {})[key] = rec["xyz"]
    return idx


def extract_range_from_protein(protein_pdb: str, start: int, end: int) -> np.ndarray:
    ca = read_ca_coords_from_pdb(protein_pdb)
    chains = build_chain_index(ca)
    keys_range = [(i, "") for i in range(start, end + 1)]

    # strict pass: require exact (resseq, icode="")
    for ch, mapping in chains.items():
        coords = []
        ok = True
        for k in keys_range:
            if k in mapping:
                coords.append(mapping[k])
            else:
                ok = False
                break
        if ok:
            return np.vstack(coords)

    # fallback: allow insertion codes for each residue number (take the first)
    for ch, mapping in chains.items():
        coords = []
        ok = True
        for i in range(start, end + 1):
            cand = [mapping[k] for k in mapping.keys() if k[0] == i]
            if not cand:
                ok = False
                break
            coords.append(cand[0])
        if ok:
            return np.vstack(coords)

    raise RuntimeError(f"Could not extract residues {start}-{end} from {protein_pdb} on any chain.")


# ---------- index parsing ----------
def parse_index_tsv(tsv_path: str):
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        header = [h.strip() for h in header]
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts = parts + [""] * (len(header) - len(parts))
            row = {header[i]: parts[i] for i in range(len(header))}
            rows.append(row)
    return rows


def get_index_row(rows, pdb_id: str):
    for r in rows:
        if str(r.get("pdb_id", "")).lower() == pdb_id.lower():
            return r
    raise KeyError(f"pdb_id '{pdb_id}' not found in index file.")


# ---------- input parsers ----------
def _as_np_coords(obj: Any) -> np.ndarray:
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    return None


def load_candidates_from_json(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - JSON array file: [ {...}, {...}, ... ]
      - JSON Lines file: one JSON object per line
    Each object must contain "main_positions".
    """
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return []
        if txt[0] == "[":
            data = json.loads(txt)
            if not isinstance(data, list):
                raise ValueError("JSON root must be a list.")
            for i, obj in enumerate(data):
                if isinstance(obj, dict):
                    items.append(obj)
        else:
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
    out = []
    for i, obj in enumerate(items):
        coords = _as_np_coords(obj.get("main_positions"))
        if coords is None:
            continue
        rec = {
            "id": obj.get("bitstring") or obj.get("id") or f"json_{i}",
            "coords": coords,
            "raw": obj
        }
        out.append(rec)
    return out


def load_candidates_from_csv(path: str, pos_col: str = "main_positions") -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    out = []
    for i, row in df.iterrows():
        coords = _as_np_coords(row.get(pos_col))
        if coords is None:
            continue
        rid = None
        for key in ("bitstring", "id", "sequence"):
            if key in df.columns and pd.notna(row.get(key)):
                rid = str(row.get(key))
                break
        if rid is None:
            rid = f"row_{i}"
        out.append({
            "id": rid,
            "coords": coords,
            "raw": row.to_dict()
        })
    return out


# ---------- core ----------
def compute_rmsd_to_gt(candidate_xyz: np.ndarray,
                       gt_xyz: np.ndarray) -> Tuple[float, np.ndarray]:
    Lc = candidate_xyz.shape[0]
    Lg = gt_xyz.shape[0]
    L = min(Lc, Lg)
    cand = candidate_xyz[:L]
    gt = gt_xyz[:L]
    R, t = kabsch(gt, cand)
    cand_aligned = (R @ cand.T).T + t
    val = rmsd(gt, cand_aligned)
    return val, cand_aligned


def main():
    ap = argparse.ArgumentParser(description="Select the lowest-RMSD structure from JSON or CSV.")
    ap.add_argument("--pdb-id", required=True, help="PDB ID (e.g., 1m7y)")
    ap.add_argument("--index", required=True, help="Path to benchmark_info.txt")
    ap.add_argument("--data-root", required=True, help="Root folder for Pdbbind/<pdb_id>/<pdb_id>_protein.pdb")
    ap.add_argument("--json", default=None, help="Path to JSON array or JSON-Lines file of decoded structures")
    ap.add_argument("--csv", default=None, help="Path to CSV of cluster results")
    ap.add_argument("--pos-col", default="main_positions", help="Column name for positions in CSV")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    if (args.json is None) == (args.csv is None):
        raise SystemExit("Specify exactly one of --json or --csv.")

    os.makedirs(args.out_dir, exist_ok=True)

    # ground truth
    rows = parse_index_tsv(args.index)
    row = get_index_row(rows, args.pdb_id)
    residues = str(row.get("Residues", "")).strip()
    if "-" not in residues:
        raise RuntimeError(f"Bad 'Residues' format for {args.pdb_id}: '{residues}'")
    start, end = [int(x) for x in residues.split("-")]
    ref_dir = os.path.join(args.data_root, args.pdb_id.lower())
    protein_pdb = os.path.join(ref_dir, f"{args.pdb_id.lower()}_protein.pdb")
    if not os.path.isfile(protein_pdb):
        candidates = [fn for fn in os.listdir(ref_dir) if fn.lower().endswith("_protein.pdb")]
        if not candidates:
            raise FileNotFoundError(f"Reference PDB not found under {ref_dir}")
        protein_pdb = os.path.join(ref_dir, candidates[0])
    gt_ca = extract_range_from_protein(protein_pdb, start, end)

    # candidates
    if args.json:
        cands = load_candidates_from_json(args.json)
    else:
        cands = load_candidates_from_csv(args.csv, pos_col=args.pos_col)

    if not cands:
        raise RuntimeError("No valid candidates found.")

    # evaluate all
    records = []
    best = None
    best_aligned = None

    for rec in cands:
        cid = rec["id"]
        xyz = np.asarray(rec["coords"], dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            continue
        val, aligned = compute_rmsd_to_gt(xyz, gt_ca)
        records.append({"id": cid, "length": xyz.shape[0], "rmsd": val})
        if best is None or val < best["rmsd"]:
            best = {"id": cid, "length": xyz.shape[0], "rmsd": val, "raw": xyz}
            best_aligned = aligned

    # save table
    df = pd.DataFrame(records).sort_values("rmsd", ascending=True).reset_index(drop=True)
    df_path = os.path.join(args.out_dir, "rmsd_table.tsv")
    df.to_csv(df_path, sep="\t", index=False)

    # save best raw and aligned
    best_raw_pdb = os.path.join(args.out_dir, "best_raw_ca.pdb")
    write_ca_pdb(best_raw_pdb, best["raw"], chain_id="A")
    best_aln_pdb = os.path.join(args.out_dir, "best_aligned_to_gt.pdb")
    write_ca_pdb(best_aln_pdb, best_aligned, chain_id="A")

    # report
    print("=== Best RMSD candidate ===")
    print(f"pdb_id        : {args.pdb_id}")
    print(f"residue range : {start}-{end}")
    print(f"best id       : {best['id']}")
    print(f"length        : {best['length']}")
    print(f"RMSD (Å)      : {best['rmsd']:.4f}")
    print(f"Table         : {df_path}")
    print(f"Best raw PDB  : {best_raw_pdb}")
    print(f"Best aligned  : {best_aln_pdb}")


if __name__ == "__main__":
    main()
