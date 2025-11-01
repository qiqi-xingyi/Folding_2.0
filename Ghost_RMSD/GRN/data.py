# --*-- conding:utf-8 --*--
# @time:11/1/25 03:42
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:data.py

# grn_simple/data.py
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


_ID_FIELDS = {"group_id", "protein", "sequence", "bitstring", "residues"}
_LABEL_FIELDS = {"rel", "rmsd"}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


class GRNDataset(Dataset):
    """Tensor-ready dataset for GRN."""

    def __init__(
        self,
        rows: List[Dict],
        feature_names: List[str],
        scaler: Dict[str, Dict[str, float]],
        device: Optional[torch.device] = None,
    ):
        self.rows = rows
        self.feature_names = feature_names
        self.scaler = scaler
        self.device = device

        X, y, gid, rmsd = self._vectorize(rows)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.group_id = torch.from_numpy(gid).long()
        self.rmsd = torch.from_numpy(rmsd).float()

    def _vectorize(self, rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(rows)
        d = len(self.feature_names)
        X = np.zeros((n, d), dtype=np.float32)
        y = np.zeros((n,), dtype=np.int64)
        gid = np.zeros((n,), dtype=np.int64)
        rmsd = np.zeros((n,), dtype=np.float32)

        for i, r in enumerate(rows):
            # features
            for j, col in enumerate(self.feature_names):
                val = r.get(col, 0.0)
                if not _is_number(val):
                    val = 0.0
                mu = self.scaler[col]["mean"]
                sd = self.scaler[col]["std"]
                if sd == 0.0:
                    sd = 1.0
                X[i, j] = (float(val) - mu) / sd
            # labels / meta
            y[i] = int(r.get("rel", 0))
            gid[i] = int(r.get("group_id", -1))
            rmsd[i] = float(r.get("rmsd", float("nan")))
        return X, y, gid, rmsd

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "group_id": self.group_id[idx],
            "rmsd": self.rmsd[idx],
        }


class GRNDataModule:
    """
    Minimal data module for GRN.
    - Auto-detect numeric feature columns from train.jsonl
    - Compute train-only mean/std and apply to all splits
    - Provide PyTorch DataLoaders
    """

    def __init__(
        self,
        data_dir: str = "training_dataset",
        train_file: str = "train.jsonl",
        valid_file: str = "valid.jsonl",
        test_file: str = "test.jsonl",
        batch_size: int = 1024,
        num_workers: int = 0,
        shuffle_train: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / train_file
        self.valid_path = self.data_dir / valid_file
        self.test_path = self.data_dir / test_file

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.drop_last = drop_last

        self.feature_names: List[str] = []
        self.scaler: Dict[str, Dict[str, float]] = {}

        self.ds_train: Optional[GRNDataset] = None
        self.ds_valid: Optional[GRNDataset] = None
        self.ds_test: Optional[GRNDataset] = None

    # ---------- public API ----------

    def setup(self) -> None:
        rows_train = _load_jsonl(self.train_path)
        rows_valid = _load_jsonl(self.valid_path)
        rows_test = _load_jsonl(self.test_path)

        self.feature_names = self._infer_feature_columns(rows_train)
        self.scaler = self._compute_scaler(rows_train, self.feature_names)

        self.ds_train = GRNDataset(rows_train, self.feature_names, self.scaler)
        self.ds_valid = GRNDataset(rows_valid, self.feature_names, self.scaler)
        self.ds_test = GRNDataset(rows_test, self.feature_names, self.scaler)

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.ds_train, shuffle=self.shuffle_train)

    def valid_dataloader(self) -> DataLoader:
        return self._make_loader(self.ds_valid, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.ds_test, shuffle=False)

    def feature_dim(self) -> int:
        return len(self.feature_names)

    # ---------- helpers ----------

    def _make_loader(self, ds: GRNDataset, shuffle: bool) -> DataLoader:
        assert ds is not None, "Call setup() first."
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def _infer_feature_columns(self, rows: List[Dict]) -> List[str]:
        # start from all numeric keys in train set
        cand = set()
        for r in rows:
            for k, v in r.items():
                if k in _ID_FIELDS or k in _LABEL_FIELDS:
                    continue
                if _is_number(v):
                    cand.add(k)
        # stable order
        cols = sorted(cand)
        if not cols:
            raise ValueError("No numeric feature columns found in train set.")
        return cols

    def _compute_scaler(self, rows: List[Dict], cols: List[str]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for c in cols:
            vals: List[float] = []
            for r in rows:
                v = r.get(c, None)
                if _is_number(v):
                    vals.append(float(v))
            if len(vals) == 0:
                stats[c] = {"mean": 0.0, "std": 1.0}
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            if std == 0.0:
                std = 1.0
            stats[c] = {"mean": mean, "std": std}
        return stats
