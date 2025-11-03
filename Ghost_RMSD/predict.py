# --*-- coding:utf-8 --*--
# @time:11/2/25 17:26 (patched 11/3/25)
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:predict.py
#
# GRN inference over GRN input JSONL.
# - Robust checkpoint path resolution (script-dir fallback).
# - Rebuild sklearn StandardScaler from dict (mean_/scale_) if needed.
# - Device auto-pick (cuda/mps/cpu).
# - Per-group ranking (group_id).
# - Optional top-k export.

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from GRN.model import GRNClassifier

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard residues


def pick_device(arg_device: Optional[str]) -> torch.device:
    if arg_device and arg_device != "auto":
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon / Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint_path(arg_path: str) -> Path:
    """Try user path first; if missing, try relative to this script's directory."""
    p = Path(arg_path)
    if p.exists():
        return p
    alt = Path(__file__).resolve().parent / arg_path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Cannot find checkpoint at '{arg_path}' or '{alt}'")


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def seq_features_from_sequence(seq: str, feature_names: List[str]) -> Dict[str, float]:
    seq = (seq or "").strip().upper()
    n = max(1, len(seq))
    counts = {aa: 0 for aa in AA_ORDER}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1

    feats: Dict[str, float] = {}
    if "seq_len" in feature_names:
        feats["seq_len"] = float(n)
    for aa in AA_ORDER:
        key = f"aa_count_{aa}"
        if key in feature_names:
            feats[key] = float(counts[aa])
    for aa in AA_ORDER:
        key = f"aa_frac_{aa}"
        if key in feature_names:
            feats[key] = float(counts[aa]) / float(n)
    return feats


def rebuild_scaler_if_needed(scaler_obj: Any) -> Any:
    """
    Accept either an sklearn-like object with .transform or a dict with mean_/scale_.
    Rebuild a StandardScaler when a dict is provided.
    """
    if hasattr(scaler_obj, "transform"):
        return scaler_obj
    if isinstance(scaler_obj, dict) and "mean_" in scaler_obj and "scale_" in scaler_obj:
        from sklearn.preprocessing import StandardScaler
        s = StandardScaler()
        s.mean_ = np.asarray(scaler_obj["mean_"], dtype=float)
        s.scale_ = np.asarray(scaler_obj["scale_"], dtype=float)
        s.var_ = s.scale_ ** 2
        s.n_features_in_ = int(s.mean_.shape[0])
        return s
    raise ValueError(
        "Invalid 'scaler' in checkpoint: expected an sklearn-like object with .transform "
        "or a dict containing mean_ and scale_."
    )


def build_design_matrix(
    df: pd.DataFrame,
    base_feature_names: List[str],
    seq_feature_names: List[str],
    scaler
) -> np.ndarray:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    missing = [c for c in base_feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base feature columns: {missing}")

    X_base = df[base_feature_names].astype(float).to_numpy(copy=True)
    X_base = scaler.transform(X_base)

    seq_rows: List[Dict[str, float]] = []
    if "sequence" not in df.columns:
        raise ValueError("Missing required column: sequence")
    for seq in df["sequence"].astype(str).tolist():
        seq_rows.append(seq_features_from_sequence(seq, seq_feature_names))
    X_seq = pd.DataFrame(seq_rows, columns=seq_feature_names).fillna(0.0).to_numpy(copy=True)

    X = np.concatenate([X_base, X_seq], axis=1)
    return X


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    input_dim: int,
    num_classes: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    model = GRNClassifier(
        in_dim=input_dim,
        hidden_dims=[512, 256, 128],
        dropout=dropout,
        use_rank_head=True,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer_batches(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    score_mode: str = "expected_rel",
) -> Tuple[np.ndarray, np.ndarray]:
    N = X.shape[0]
    logits_list = []
    scores_list = []
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).float().to(device, non_blocking=True)
        out = model(xb)
        logits = out["logits"]  # (B, C)
        prob = torch.softmax(logits, dim=-1)

        if score_mode == "prob_rel3":
            score = prob[:, 3]
        elif score_mode == "logit_rel3":
            score = logits[:, 3]
        else:  # expected_rel
            classes = torch.arange(logits.size(1), device=logits.device).float()
            score = (prob * classes[None, :]).sum(dim=-1)

        logits_list.append(logits.cpu().numpy())
        scores_list.append(score.cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    return logits, scores


def rank_within_groups(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    df_out = df.copy()
    df_out["score"] = scores
    if "group_id" not in df_out.columns:
        df_out["group_id"] = 0
    df_out["rank_in_group"] = (
        df_out.groupby("group_id")["score"]
             .rank(method="first", ascending=False)
             .astype(int)
    )
    return df_out.sort_values(["group_id", "rank_in_group", "score"], ascending=[True, True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints_full/grn_best.pt")
    ap.add_argument("--input_jsonl", type=str, required=True, help="New combined JSONL for inference")
    ap.add_argument("--out_csv", type=str, default="predictions.csv")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--score_mode", type=str, default="expected_rel",
                    choices=["prob_rel3", "logit_rel3", "expected_rel"])
    ap.add_argument("--topk", type=int, default=50, help="Optional per-group topk CSV export")
    ap.add_argument("--allow_omp_dup", action="store_true",
                    help="Set KMP_DUPLICATE_LIB_OK=TRUE for macOS OpenMP duplication issues")
    args = ap.parse_args()

    if args.allow_omp_dup and os.environ.get("KMP_DUPLICATE_LIB_OK") != "TRUE":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = pick_device(args.device)
    if device.type == "cuda":
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif device.type == "mps":
        print("[Device] Using Apple MPS backend")
    else:
        print("[Device] Using CPU")

    ckpt_path = resolve_checkpoint_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    base_feature_names: List[str] = ckpt["base_feature_names"]
    seq_feature_names: List[str] = ckpt["seq_feature_names"]
    scaler = rebuild_scaler_if_needed(ckpt["scaler"])

    df = load_jsonl(Path(args.input_jsonl))
    for c in ["group_id", "pdb_id", "sequence", "bitstring"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    X = build_design_matrix(df, base_feature_names, seq_feature_names, scaler)
    model = build_model_from_ckpt(
        ckpt, input_dim=X.shape[1], num_classes=4, dropout=ckpt.get("args", {}).get("dropout", 0.3)
    ).to(device)

    logits, scores = infer_batches(model, X, device, batch_size=args.batch_size, score_mode=args.score_mode)
    df_ranked = rank_within_groups(df, scores)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_ranked.to_csv(out_csv, index=False)
    print(f"[SAVED] Full predictions: {out_csv.resolve()}")

    if args.topk and args.topk > 0:
        tops = (
            df_ranked.sort_values(["group_id", "score"], ascending=[True, False])
                    .groupby("group_id")
                    .head(args.topk)
        )
        out_top = out_csv.with_name(out_csv.stem + f"_top{args.topk}.csv")
        tops.to_csv(out_top, index=False)
        print(f"[SAVED] Per-group top-{args.topk}: {out_top.resolve()}")


if __name__ == "__main__":
    main()
