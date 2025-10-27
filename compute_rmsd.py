# --*-- conding:utf-8 --*--
# @time:10/25/25 00:06
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:compute_rmsd.py

# -*- coding: utf-8 -*-
"""
IDE-click version: compute Ca-aligned RMSD between prediction and ground-truth segment.

Defaults:
  PRED_PDB : e_results/1m7y/refined_ca.pdb
  INDEX_TSV: dataset/benchmark_info.txt
  DATA_ROOT: dataset/Pdbbind/

Outputs:
  - Prints RMSD
  - Saves aligned predicted CA as e_results/<pdb_id>/pred_aligned_to_gt.pdb
"""

import os
import sys
import math
from typing import List, Dict, Tuple
import numpy as np


# ====== DEFAULT PATHS (edit here if needed) ======
PRED_PDB  = os.path.join("e_results", "1m7y", "allatom_refined.pdb")
INDEX_TSV = os.path.join("dataset", "benchmark_info.txt")
DATA_ROOT = os.path.join("dataset", "Pdbbind")


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
def read_ca_coords_from_pdb(path: str):
    """
    Read all CA atom coords from a PDB file.
    Returns: list of dict {chain, resseq(int), icode(str), xyz(np.array(3))}
    """
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


def write_ca_pdb(path: str, ca_xyz: np.ndarray, chain_id: str = "A"):
    """Write CA-only PDB with generic GLY residues from a (L,3) array."""
    with open(path, "w", encoding="utf-8") as f:
        for i, (x, y, z) in enumerate(ca_xyz, start=1):
            f.write(
                "ATOM  {serial:5d}  CA  GLY {chain}{resi:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n".format(
                    serial=i, chain=chain_id, resi=i, x=x, y=y, z=z
                )
            )
        f.write("END\n")


def build_chain_index(ca_list):
    """
    Make dict: chain -> { (resseq, icode) : xyz }
    """
    idx: Dict[str, Dict[Tuple[int, str], np.ndarray]] = {}
    for rec in ca_list:
        ch = rec["chain"]
        key = (rec["resseq"], rec["icode"])
        idx.setdefault(ch, {})[key] = rec["xyz"]
    return idx


def extract_range_candidates_per_chain(protein_pdb: str, start: int, end: int):
    """
    Return list of (chain_id, coords[L,3]) for all chains that can provide residues [start, end].
    First try exact insertion code "" for all residues; if not possible, allow any insertion code per resseq.
    """
    ca = read_ca_coords_from_pdb(protein_pdb)
    chains = build_chain_index(ca)
    keys_range = [(i, "") for i in range(start, end + 1)]

    candidates = []

    # strict pass (icode="")
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
            candidates.append((ch, np.vstack(coords)))

    # fallback pass (any icode per resseq), but only if strict didn't return anything for that chain set
    if not candidates:
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
                candidates.append((ch, np.vstack(coords)))

    if not candidates:
        raise RuntimeError(f"Could not extract residues {start}-{end} from {protein_pdb} on any chain.")
    return candidates


    # fallback: allow insertion codes for each residue number (take the first)
    for ch, mapping in chains.items():
        coords = []
        ok = True
        for i in range(start, end + 1):
            # all keys with same resseq (any icode)
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
    """
    Read benchmark_info.txt (tab-separated).
    Returns list of dict rows with keys: pdb_id, Residues, Sequence_length, ...
    """
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        # normalize header names
        header = [h.strip() for h in header]
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                # pad
                parts = parts + [""] * (len(header) - len(parts))
            row = {header[i]: parts[i] for i in range(len(header))}
            rows.append(row)
    return rows


def get_index_row(rows, pdb_id: str):
    for r in rows:
        if str(r.get("pdb_id", "")).lower() == pdb_id.lower():
            return r
    raise KeyError(f"pdb_id '{pdb_id}' not found in index file.")


# ---------- main ----------
def main():
    # Resolve absolute paths (relative to this script)
    here = os.path.dirname(os.path.abspath(__file__))

    pred_path  = os.path.join(here, PRED_PDB)
    index_path = os.path.join(here, INDEX_TSV)
    data_root  = os.path.join(here, DATA_ROOT)

    # Infer pdb_id from e_results/<pdb_id>/refined_ca.pdb
    pdb_id = os.path.basename(os.path.dirname(os.path.abspath(pred_path))).lower()

    # Read index row
    rows = parse_index_tsv(index_path)
    row = get_index_row(rows, pdb_id)
    residues = str(row.get("Residues", "")).strip()
    if "-" not in residues:
        raise RuntimeError(f"Bad 'Residues' format for {pdb_id}: '{residues}'")
    start, end = [int(x) for x in residues.split("-")]
    seq_len = int(row.get("Sequence_length", end - start + 1))

    # Reference protein pdb
    ref_dir = os.path.join(data_root, pdb_id)
    protein_pdb = os.path.join(ref_dir, f"{pdb_id}_protein.pdb")
    if not os.path.isfile(protein_pdb):
        # tolerate macOS '._' files and case variations
        candidates = [fn for fn in os.listdir(ref_dir) if fn.lower().endswith("_protein.pdb")]
        if not candidates:
            raise FileNotFoundError(f"Reference PDB not found under {ref_dir}")
        protein_pdb = os.path.join(ref_dir, candidates[0])

    # Load ground-truth segment (CA)
    # Load ground-truth candidates (CA) for all chains
    gt_candidates = extract_range_candidates_per_chain(protein_pdb, start, end)

    # Load predicted CA (file order) – this is Å, length L_pred
    pred_ca_list = read_ca_coords_from_pdb(pred_path)
    pred_ca = np.vstack([r["xyz"] for r in pred_ca_list])
    L_pred = pred_ca.shape[0]

    # We expect exact length match for a fragment; if not, truncate to min length
    best = dict(val=float("inf"), chain=None, reversed=False, pred_aligned=None, gt_used=None)
    for ch, gt_ca in gt_candidates:
        L_gt = gt_ca.shape[0]
        L = min(L_pred, L_gt)
        gt_use  = gt_ca[:L]
        pred_fw = pred_ca[:L]
        pred_rv = pred_ca[:L][::-1].copy()   # reversed order (C→N)

        # forward
        R, t = kabsch(gt_use, pred_fw)
        pred_aligned_fw = (R @ pred_fw.T).T + t
        val_fw = rmsd(gt_use, pred_aligned_fw)
        if val_fw < best["val"]:
            best.update(val=val_fw, chain=ch, reversed=False, pred_aligned=pred_aligned_fw, gt_used=gt_use)

        # reversed
        R, t = kabsch(gt_use, pred_rv)
        pred_aligned_rv = (R @ pred_rv.T).T + t
        val_rv = rmsd(gt_use, pred_aligned_rv)
        if val_rv < best["val"]:
            best.update(val=val_rv, chain=ch, reversed=True, pred_aligned=pred_aligned_rv, gt_used=gt_use)

    # Save best-aligned predicted CA for inspection
    out_dir = os.path.dirname(os.path.abspath(pred_path))
    out_pdb = os.path.join(out_dir, "pred_aligned_to_gt.pdb")
    write_ca_pdb(out_pdb, best["pred_aligned"], chain_id=str(best["chain"] or "A"))

    # Report
    print("=== RMSD (Cα-aligned; best over chains and orientation) ===")
    print(f"pdb_id           : {pdb_id}")
    print(f"residue range    : {start}-{end}")
    print(f"len(pred / gt)   : {best['pred_aligned'].shape[0]} / {best['gt_used'].shape[0]}")
    print(f"Best chain       : {best['chain']}")
    print(f"Orientation      : {'reversed' if best['reversed'] else 'forward'}")
    print(f"RMSD (Å)         : {best['val']:.4f}")
    print(f"Aligned PDB      : {out_pdb}")


    # Keep console open if double-clicked (optional)
    if sys.stdout.isatty() is False:
        input("\nPress <Enter> to exit...")


if __name__ == "__main__":
    main()
