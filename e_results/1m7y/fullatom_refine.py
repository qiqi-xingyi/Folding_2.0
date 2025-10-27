# --*-- coding:utf-8 --*--
# @time: 10/27/25
# @Author: SQD_Folding
# @File: fullatom_refine.py
"""
Build a full-atom peptide from sequence parsed out of a CA-only PDB,
rigidly align it to the CA trace, then energy-minimize in OpenMM with
ff14SB + GBn2 implicit solvent and soft CA positional restraints.

Inputs (relative to this script directory):
  - refined_ca.pdb  (CA-only PDB produced by your pipeline)

Outputs (same directory):
  - init_from_seq.pdb             (ideal all-atom from sequence)
  - init_fullatom_aligned.pdb     (rigidly aligned to refined CA)
  - allatom_refined.pdb           (minimized all-atom result)
"""

import os
import sys
import math
from typing import List, Tuple, Dict

import numpy as np

# BioPython & PeptideBuilder
from Bio.PDB import PDBParser, PDBIO
import PeptideBuilder

# OpenMM
import openmm as mm
from openmm import unit
from openmm import app


# -------------------------------
# Paths (relative to this script)
# -------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFINED_CA_PDB = os.path.join(SCRIPT_DIR, "refined_ca.pdb")
OUT_INIT_PDB = os.path.join(SCRIPT_DIR, "init_from_seq.pdb")
OUT_INIT_ALIGNED_PDB = os.path.join(SCRIPT_DIR, "init_fullatom_aligned.pdb")
OUT_MIN_PDB = os.path.join(SCRIPT_DIR, "allatom_refined.pdb")


# -------------------------------
# Helpers (math)
# -------------------------------

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return R, t that aligns Q to P (both Nx3), minimizing ||P - (R Q + t)||."""
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = P.mean(axis=0) - (R @ Q.mean(axis=0))
    return R, t


# -------------------------------
# Helpers (sequence & PDB IO)
# -------------------------------

_THREE_TO_ONE = {
    # standard amino acids
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common variants map to parent
    "HID": "H", "HIE": "H", "HIP": "H",
    "ASH": "D", "GLH": "E", "LYN": "K", "ARG+": "R",
    # termini labels occasionally appear; strip to base if needed
}

def three_to_one_safe(resname: str) -> str:
    r = resname.upper().strip()
    if r in _THREE_TO_ONE:
        return _THREE_TO_ONE[r]
    # fallback heuristics
    r2 = r.replace("NME", "GLY").replace("ACE", "GLY")
    if r2 in _THREE_TO_ONE:
        return _THREE_TO_ONE[r2]
    # last resort: treat unknown as gly
    return "G"


def read_ca_and_seq_from_pdb(path: str) -> Tuple[np.ndarray, str]:
    """
    Read CA coordinates and residue sequence (1-letter) from a PDB.
    For each residue, take the first CA encountered.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ca_xyz = []
    seq = []
    seen_res = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            name = line[12:16].strip()
            if name != "CA":
                continue
            resn = line[17:20].strip()
            chain = line[21].strip() or "A"
            resi = line[22:26].strip()
            icode = line[26].strip()

            key = (chain, resi, icode)
            if key in seen_res:
                continue
            seen_res.add(key)

            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except Exception:
                continue
            ca_xyz.append([x, y, z])
            seq.append(three_to_one_safe(resn))

    if not ca_xyz:
        raise RuntimeError(f"No CA atoms found in {path}")

    coords = np.asarray(ca_xyz, dtype=float)
    sequence = "".join(seq)
    return coords, sequence


def build_fullatom_from_sequence(seq: str, out_pdb_path: str) -> None:
    """
    Build an all-atom peptide with ideal geometry from a one-letter sequence and save to PDB.
    """
    if len(seq) == 0:
        raise ValueError("Empty sequence.")
    struct = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        PeptideBuilder.add_residue(struct, aa)
    io = PDBIO()
    io.set_structure(struct)
    io.save(out_pdb_path)


def read_ca_from_biopy(structure) -> np.ndarray:
    cas = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.has_id("CA"):
                    cas.append(res["CA"].get_coord())
    if not cas:
        raise RuntimeError("No CA atoms found in structure.")
    return np.asarray(cas, dtype=float)


def apply_rigid_transform_to_biopy(structure, R: np.ndarray, t: np.ndarray):
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    xyz = atom.get_coord()
                    atom.set_coord((R @ xyz) + t)


# -------------------------------
# OpenMM minimization
# -------------------------------

def minimize_with_soft_ca_restraints(
    pdb_path: str,
    ca_ref_angstrom: np.ndarray,
    k_pos_kj_per_mol_nm2: float = 2000.0,
    n_steps_min: int = 5000,
    run_short_anneal: bool = False,
    platform_name: str = None,
) -> Tuple[np.ndarray, app.PDBFile]:
    """
    Minimize an all-atom model with ff14SB + GBn2 implicit solvent and soft CA restraints.

    Returns (final_positions_angstrom (N,3), pdbfile_obj_for_topology).
    """
    # Force field without explicit implicit XML; enable GBn2 in createSystem
    ff = app.ForceField("amber14/protein.ff14SB.xml")

    pdb = app.PDBFile(pdb_path)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=7.0)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        removeCMMotion=True,
        implicitSolvent=app.GBn2,
        implicitSolventSaltConc=0.1 * unit.molar,
    )

    # Soft positional restraints on Cα
    ca_ref_nm = (ca_ref_angstrom * unit.angstrom).value_in_unit(unit.nanometer)

    pos_rest = mm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    pos_rest.addPerParticleParameter("x0")
    pos_rest.addPerParticleParameter("y0")
    pos_rest.addPerParticleParameter("z0")
    pos_rest.addGlobalParameter("k", float(k_pos_kj_per_mol_nm2))  # kJ/mol/nm^2

    ca_indices = [atom.index for atom in modeller.topology.atoms() if atom.name == "CA"]
    if len(ca_indices) != len(ca_ref_nm):
        # trim to shortest
        L = min(len(ca_indices), len(ca_ref_nm))
        ca_indices = ca_indices[:L]
        ca_ref_nm = ca_ref_nm[:L]

    for i, idx in enumerate(ca_indices):
        x, y, z = ca_ref_nm[i]
        pos_rest.addParticle(int(idx), [float(x), float(y), float(z)])

    system.addForce(pos_rest)

    # Integrator & Simulation
    integrator = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    platform = mm.Platform.getPlatformByName(platform_name) if platform_name else None
    sim = app.Simulation(modeller.topology, system, integrator, platform) if platform else app.Simulation(
        modeller.topology, system, integrator
    )
    sim.context.setPositions(modeller.positions)

    # Energy minimization
    sim.minimizeEnergy(maxIterations=int(n_steps_min))

    # Optional short anneal: 50K -> 300K -> 50K
    if run_short_anneal:
        def md_block(temp_K: float, nsteps: int):
            sim.context.setVelocitiesToTemperature(temp_K * unit.kelvin)
            sim.step(nsteps)

        md_block(50.0, 2000)
        md_block(300.0, 3000)
        md_block(50.0, 2000)

        # final touch-up
        sim.minimizeEnergy(maxIterations=2000)

    state = sim.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)  # in nm
    pos_A = pos.value_in_unit(unit.angstrom)
    pos_arr = np.array([[p[0], p[1], p[2]] for p in pos_A], dtype=float)
    return pos_arr, pdb  # return positions and a PDBFile for topology


def write_pdb_with_positions(topology: app.Topology, positions_A: np.ndarray, out_path: str):
    """Write a PDB given topology and positions in Å."""
    with open(out_path, "w", encoding="utf-8") as f:
        iatom = 1
        for chain in topology.chains():
            for res in chain.residues():
                for atom in res.atoms():
                    xyz = positions_A[atom.index]
                    f.write(
                        "ATOM  {serial:5d} {name:^4s}{altLoc}{resName:>3s} {chainID}{resSeq:>4s}{iCode:1s}"
                        "{x:12.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}          {elem:>2s}\n".format(
                            serial=iatom,
                            name=atom.name[:4],
                            altLoc=" ",
                            resName=res.name[:3],
                            chainID=(chain.id or "A")[:1],
                            resSeq=str(res.id) if res.id is not None else "1",
                            iCode=" ",
                            x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                            occ=1.00, temp=20.00,
                            elem=(atom.element.symbol if atom.element is not None else "C")[:2]
                        )
                    )
                    iatom += 1
        f.write("END\n")


# -------------------------------
# Main
# -------------------------------

def main():
    # 1) Read CA and sequence from CA-only PDB
    refined_ca, seq = read_ca_and_seq_from_pdb(REFINED_CA_PDB)
    print(f"[info] read {len(refined_ca)} CA atoms; sequence='{seq}'")

    if len(seq) != len(refined_ca):
        L = min(len(seq), len(refined_ca))
        print(f"[warn] seq length ({len(seq)}) != CA length ({len(refined_ca)}); truncating to {L}")
        seq = seq[:L]
        refined_ca = refined_ca[:L]

    # 2) Build ideal full-atom peptide from sequence
    build_fullatom_from_sequence(seq, OUT_INIT_PDB)

    # 3) Rigidly align ideal model to refined CA using Kabsch
    parser = PDBParser(QUIET=True)
    init_struct = parser.get_structure("init", OUT_INIT_PDB)
    ideal_ca = read_ca_from_biopy(init_struct)

    if len(ideal_ca) != len(refined_ca):
        L = min(len(ideal_ca), len(refined_ca))
        ideal_use = ideal_ca[:L]
        refined_use = refined_ca[:L]
    else:
        ideal_use = ideal_ca
        refined_use = refined_ca

    R, t = kabsch(refined_use, ideal_use)  # align ideal -> refined
    apply_rigid_transform_to_biopy(init_struct, R, t)

    io = PDBIO()
    io.set_structure(init_struct)
    io.save(OUT_INIT_ALIGNED_PDB)

    # 4) Minimize with soft CA restraints (GBn2 implicit solvent)
    pos_A, pdb_for_top = minimize_with_soft_ca_restraints(
        OUT_INIT_ALIGNED_PDB,
        ca_ref_angstrom=refined_ca,
        k_pos_kj_per_mol_nm2=2000.0,
        n_steps_min=6000,
        run_short_anneal=False,   # set True if you want extra relaxation
        platform_name=None        # e.g., "CUDA" if GPU build is available
    )

    # 5) Write minimized PDB
    write_pdb_with_positions(pdb_for_top.topology, pos_A, OUT_MIN_PDB)
    print(f"[ok] wrote minimized PDB: {OUT_MIN_PDB}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)
