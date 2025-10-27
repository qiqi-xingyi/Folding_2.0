# --*-- conding:utf-8 --*--
# @time:10/27/25 02:27
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fullatom_refine.py

# fullatom_refine.py
# Build all-atom peptide from sequence read from refined_ca.pdb, restrain CA, then minimize + restrained MD.

import os
import io
import tempfile
from typing import List, Tuple

import numpy as np

# OpenMM
from openmm import unit, Platform
import openmm as mm
from openmm import app

# PeptideBuilder / Biopython
from Bio.PDB import PDBIO
import PeptideBuilder


# ----------------------
# User config
# ----------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFINED_CA_PDB = os.path.join(SCRIPT_DIR, "refined_ca.pdb")
OUT_PDB = os.path.join(SCRIPT_DIR, "allatom_refined.pdb")

PLATFORM = "CPU"                       # "CUDA" | "OpenCL" | "CPU"
TEMPERATURE = 300.0 * unit.kelvin
FRICTION = 1.0 / unit.picosecond
STEP = 0.002 * unit.picoseconds        # 2 fs
REPORT_EVERY = 1000

# Restraint schedule (flat-bottom for CA)
SCHEDULE = [
    # (k, r0, minimize_steps, md_steps)
    (1000.0, 0.5, 2000, 0),
    (300.0,  0.8,  500, 25000),
    (100.0,  1.2,  500, 25000),
    (50.0,   1.5, 1000, 0),
]

# ----------------------
# Helpers
# ----------------------
def one_to_three(res1: str) -> str:
    table = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
        "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
        "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
        "B": "ASX", "Z": "GLX", "X": "GLY", "J": "LEU", "U": "SEC",
        "O": "PYL"
    }
    return table.get(res1.upper(), "GLY")

_AA3_TO_AA1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
    # common alt/ambiguous fallbacks:
    "HSD":"H","HSE":"H","HSP":"H","HIP":"H","HIE":"H","HID":"H",
    "ASX":"D","GLX":"E","SEC":"C","PYL":"K",
}

def read_ca_and_seq_from_pdb(path: str) -> Tuple[np.ndarray, str]:
    """
    Read CA coordinates and derive 1-letter sequence from 3-letter residue names.
    Assumes one CA per residue in order (as written by StructureRefiner.write_pdb_ca).
    """
    ca_xyz, aa1 = [], []
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            res3 = line[17:20].strip().upper()
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            ca_xyz.append([x, y, z])
            aa1.append(_AA3_TO_AA1.get(res3, "G"))  # unknown -> GLY
    if not ca_xyz:
        raise RuntimeError(f"No CA atoms found in {path}")
    coords = np.array(ca_xyz, dtype=float)
    seq = "".join(aa1)
    return coords, seq

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def build_peptide_pdb_from_sequence(seq: str) -> app.PDBFile:
    """Use PeptideBuilder to build an extended peptide for the sequence, return as OpenMM PDBFile."""
    struct = None
    for i, aa in enumerate(seq):
        res3 = one_to_three(aa.upper())
        if i == 0:
            struct = PeptideBuilder.initialize_res(res3)
        else:
            PeptideBuilder.add_residue(struct, res3)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name
    io_writer = PDBIO()
    io_writer.set_structure(struct)
    io_writer.save(tmp_path)
    pdb = app.PDBFile(tmp_path)
    os.remove(tmp_path)
    return pdb

def get_ca_indices(top: app.Topology, L: int) -> List[int]:
    idx, count = [], 0
    for atom in top.atoms():
        if atom.name == "CA":
            idx.append(atom.index)
            count += 1
            if count >= L:
                break
    if len(idx) < L:
        raise RuntimeError(f"Topology has only {len(idx)} CA atoms, expected {L}.")
    return idx

def align_model_ca_to_refined(model_pos: unit.Quantity, ca_idx: List[int], refined_ca: np.ndarray):
    """In-place alignment of the model CA set onto refined_ca using Kabsch (Quantity array shape (natoms,3))."""
    X = model_pos.value_in_unit(unit.angstrom)
    model_ca = X[ca_idx, :]
    L = min(model_ca.shape[0], refined_ca.shape[0])
    R, t = kabsch(refined_ca[:L], model_ca[:L])
    X_new = (R @ X.T).T
    X_new += t
    model_pos[:] = X_new * unit.angstrom

def make_flatbottom_ca_restraint(system: mm.System,
                                 topology: app.Topology,
                                 target_ca: np.ndarray,
                                 k_kcal_per_A2: float,
                                 r0_A: float) -> mm.CustomExternalForce:
    """
    Add flat-bottom restraint on CA: E = step(r-r0)*0.5*k*(r-r0)^2
    r0 in Å, k in kcal/mol/Å^2
    """
    k = k_kcal_per_A2 * (unit.kilocalories_per_mole / unit.angstrom**2)
    k_md = k.value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2)
    r0_nm = (r0_A * unit.angstrom).value_in_unit(unit.nanometer)

    expr = "step(sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2) - r0) * 0.5 * k * (sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2) - r0)^2"
    force = mm.CustomExternalForce(expr)
    force.addGlobalParameter("k", k_md)
    force.addGlobalParameter("r0", r0_nm)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    ca_atoms = [a for a in topology.atoms() if a.name == "CA"]
    L = min(len(ca_atoms), target_ca.shape[0])
    for i in range(L):
        x, y, z = target_ca[i]  # Å
        force.addParticle(ca_atoms[i].index, [x/10.0, y/10.0, z/10.0])  # nm
    system.addForce(force)
    return force

# ----------------------
# Main pipeline
# ----------------------
def main():
    # 1) Read refined CA and derive sequence from residue names
    refined_ca, seq = read_ca_and_seq_from_pdb(REFINED_CA_PDB)  # (L,3), str
    L = refined_ca.shape[0]
    print(f"[info] read {L} CA atoms; sequence='{seq}'")

    # 2) Build all-atom initial peptide (extended) from sequence
    pdb_init = build_peptide_pdb_from_sequence(seq)

    # 3) Force field and Modeller
    ff = app.ForceField("amber14/protein.ff14SB.xml")
    modeller = app.Modeller(pdb_init.topology, pdb_init.positions)
    modeller.addHydrogens(ff, pH=7.0)

    # 4) Create system
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,  # implicit solvent normally uses NoCutoff
        constraints=app.HBonds,
        removeCMMotion=True,
        implicitSolvent=app.GBn2,
        implicitSolventSaltConc=0.1 * unit.molar  # optional, tweak as you like
    )

    # 5) Align model CA to refined CA
    ca_idx = get_ca_indices(modeller.topology, L)
    positions = modeller.positions
    align_model_ca_to_refined(positions, ca_idx, refined_ca)

    # 6) Integrator & Simulation
    platform = Platform.getPlatformByName(PLATFORM)
    integrator = mm.LangevinIntegrator(TEMPERATURE, FRICTION, STEP)
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(positions)

    state = sim.context.getState(getEnergy=True)
    print(f"[init] Energy: {state.getPotentialEnergy()}")

    sim.reporters.append(app.StateDataReporter(
        file=open(os.devnull, "w"),
        reportInterval=REPORT_EVERY,
        step=True, potentialEnergy=True, temperature=True
    ))

    # 7) Restraint schedule
    for stage, (k_, r0_, min_steps, md_steps) in enumerate(SCHEDULE, start=1):
        print(f"[stage {stage}] k={k_} kcal/mol/A^2, r0={r0_} A, minimize={min_steps}, md_steps={md_steps}")
        # remove prior CustomExternalForce (if any)
        for fi in reversed(range(system.getNumForces())):
            if isinstance(system.getForce(fi), mm.CustomExternalForce):
                system.removeForce(fi)
        # add new CA restraint
        make_flatbottom_ca_restraint(system, modeller.topology, refined_ca, k_, r0_)
        # reinit context with preserved state
        sim.context.reinitialize(preserveState=True)

        if min_steps > 0:
            try:
                sim.minimizeEnergy(maxIterations=min_steps)
            except Exception as e:
                print(f"[warn] minimize failed at stage {stage}: {e}")

        if md_steps > 0:
            sim.step(md_steps)

        state = sim.context.getState(getEnergy=True)
        print(f"[stage {stage}] Energy: {state.getPotentialEnergy()}")

    # 8) Save final PDB
    final_state = sim.context.getState(getPositions=True)
    final_pos = final_state.getPositions()
    with open(OUT_PDB, "w") as f:
        app.PDBFile.writeFile(sim.topology, final_pos, f)
    print(f"[done] wrote: {OUT_PDB}")

if __name__ == "__main__":
    main()
