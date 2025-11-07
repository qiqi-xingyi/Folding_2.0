# --*-- conding:utf-8 --*--
# @time:11/6/25 19:48
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:merge_rmsd_results.py


#   Merge RMSD summaries from four methods (AF3, ColabFold, VQE, QSAD)
#   Align by QSAD's pdb_id and output a consolidated CSV file.

import os
import pandas as pd
from pathlib import Path

# ====== Configurable paths ======
BASE_DIR = Path(__file__).resolve().parent
AF3_FILE = BASE_DIR / "output_final/af3_rmsd_summary.txt"
COLAB_FILE = BASE_DIR / "output_final/colabfold_rmsd_summary.txt"
VQE_FILE = BASE_DIR / "output_final/vqe_rmsd_summary.txt"
QSAD_FILE = BASE_DIR / "output_final/qsad_rmsd_summary.csv"
OUT_DIR = BASE_DIR / "result_summary"
OUT_FILE = OUT_DIR / "result_rmsd_merged.csv"


def parse_two_column_file(path: Path):
    """Parse a text file with format: pdb_id <tab> rmsd"""
    data = {}
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                data[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return data


def main():
    # ====== Load QSAD base file ======
    if not QSAD_FILE.exists():
        raise FileNotFoundError(f"QSAD file not found: {QSAD_FILE}")
    df_qsad = pd.read_csv(QSAD_FILE)
    df_qsad.columns = [c.strip().lower() for c in df_qsad.columns]
    if "final_rmsd" not in df_qsad.columns:
        raise ValueError("QSAD file must contain column 'final_rmsd'")
    df = df_qsad.rename(columns={"final_rmsd": "qsad_rmsd"})[
        ["pdb_id", "sequence", "qsad_rmsd"]
    ]

    # ====== Parse other files ======
    af3 = parse_two_column_file(AF3_FILE)
    colab = parse_two_column_file(COLAB_FILE)
    vqe = parse_two_column_file(VQE_FILE)

    # ====== Merge ======
    df["af3_rmsd"] = df["pdb_id"].map(af3)
    df["colabfold_rmsd"] = df["pdb_id"].map(colab)
    df["vqe_rmsd"] = df["pdb_id"].map(vqe)

    # ====== Save ======
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"[Saved] {OUT_FILE.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()
