# --*-- conding:utf-8 --*--
# @time:10/21/25 14:53
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:save_parquet_to_csv.py

"""
Convert a QSADPP-generated Parquet file (e.g. features.parquet)
to a CSV file for easy inspection.

Usage:
    python save_parquet_to_csv.py /path/to/demo_output/analysis
"""

import sys
from pathlib import Path
import pandas as pd


def convert_parquet_to_csv(analysis_dir: Path):
    parquet_path = analysis_dir / "features.parquet"
    csv_path = analysis_dir / "features.csv"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    print(f"[*] Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"[*] Columns: {list(df.columns)}")
    print(f"[*] Writing to {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"[✓] Done. CSV saved at {csv_path.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python save_parquet_to_csv.py /path/to/demo_output/analysis")
        sys.exit(1)

    analysis_dir = Path(sys.argv[1])
    convert_parquet_to_csv(analysis_dir)
