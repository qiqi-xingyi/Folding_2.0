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

import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


def convert_parquet_to_csv(parquet_path: Path):
    try:
        print(f"[*] Loading {parquet_path}")
        df = pd.read_parquet(parquet_path)

        csv_path = parquet_path.with_suffix(".csv")
        print(f"[*] Saving to {csv_path}")
        df.to_csv(csv_path, index=False)

        messagebox.showinfo("Conversion Complete", f"CSV saved:\n{csv_path}")
        print(f"[✓] Done. CSV saved at {csv_path.resolve()}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to convert file:\n{e}")
        raise


def main():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Select File",
        "Please select a Parquet file (e.g., features.parquet) to convert to CSV."
    )

    file_path = filedialog.askopenfilename(
        title="Select Parquet File",
        filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")]
    )

    if not file_path:
        messagebox.showwarning("Cancelled", "No file selected. Exiting.")
        return

    convert_parquet_to_csv(Path(file_path))


if __name__ == "__main__":
    main()
