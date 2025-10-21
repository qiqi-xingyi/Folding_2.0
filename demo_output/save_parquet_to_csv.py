# --*-- conding:utf-8 --*--
# @time:10/21/25 14:53
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:save_parquet_to_csv.py

"""
GUI-based Parquet → CSV converter for QSADPP output files.

You can run this directly in any IDE (PyCharm, VSCode, etc.)
or by double-clicking it in a file explorer if associated with Python.
"""

"""
Fixed-path Parquet → CSV converter for QSADPP output files.

Just click Run in your IDE.
It will read the specified Parquet file and save a CSV with the same name.
"""

import pandas as pd
from pathlib import Path


def main():
    # ---- CONFIG ----
    # Change this to your actual Parquet file path
    parquet_path = Path("./analysis/features.parquet")
    # ----------------

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    csv_path = parquet_path.with_suffix(".csv")

    print(f"[*] Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"[*] Columns: {list(df.columns)}")
    print(f"[*] Writing to {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"[✓] Done. CSV saved at {csv_path.resolve()}")


if __name__ == "__main__":
    main()
