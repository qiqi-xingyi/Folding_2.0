# --*-- conding:utf-8 --*--
# @time:11/2/25 17:26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:predict.py

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from GRN.model import GRN


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard residues


def pick_device(arg_device: str | None) -> torch.device:
    if arg_device and arg_device != "auto":
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def seq_features_from_sequence(seq: str, feature_names: List[str]) -> Dict[str, float]:
    # Build a consistent sequence feature vector following training-time names.
    # Common pattern: seq_len, aa_count_* or aa_frac_*.
    seq = (seq or "").strip().upper()
    n = max(1, len(seq))
    counts = {aa: 0 for aa in AA_ORDER}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1

    feats: Dict[str, float] = {}
    # populate supported fields if present in feature_names
    if "seq_len" in feature_names:
        feats["seq_len"] = float(n)
    # absolute counts
    for aa in AA_ORDER:
        key = f"aa_count_{aa}"
        if key in feature_names:
            feats[key] = float(counts[aa])
    # fractions
    for aa in AA_ORDER:
        key = f"aa_frac_{aa}"
        if key in feature_names:
            feats[key] = float(counts[aa]) / float(n)
    return feats


def build_design_matrix(
    df: pd.DataFrame,
    base_feature_names: List[str],
    seq_feature_names: List[str],
    scaler
) -> np.ndarray:
    # base features: numeric columns computed from energies/features jsonl
    for col in base_feature_names:
        if col not in df.columns:
            raise ValueError(f"Missing required base feature column: {col}")

    X_base = df[base_feature_names].astype(float).to_numpy(copy=True)
    # apply scaler saved in checkpoint (sklearn-like)
    X_base = scaler.transform(X_base)

    # sequence-level features computed from raw sequence
    seq_rows: List[Dict[str, float]] = []
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
    # Build GRN model with the same architecture signature used during training.
    model = GRN(
        in_dim=input_dim,
        hidden_dims=[512, 256, 128],
        num_classes=num_classes,
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
    # returns (logits, scores)
    # score_mode: "expected_rel" | "prob_rel3" | "logit_rel3"
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
    # rank descending by score within each group_id
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
    args = ap.parse_args()

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

    ckpt = torch.load(args.ckpt, map_location="cpu")
    base_feature_names: List[str] = ckpt["base_feature_names"]
    seq_feature_names: List[str] = ckpt["seq_feature_names"]
    scaler = ckpt["scaler"]  # sklearn-like scaler

    df = load_jsonl(Path(args.input_jsonl))
    necessary_cols = ["group_id", "pdb_id", "sequence", "bitstring"]
    for c in necessary_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    X = build_design_matrix(df, base_feature_names, seq_feature_names, scaler)
    model = build_model_from_ckpt(
        ckpt, input_dim=X.shape[1], num_classes=4, dropout=ckpt["args"].get("dropout", 0.3)
    ).to(device)

    logits, scores = infer_batches(model, X, device, batch_size=args.batch_size, score_mode=args.score_mode)
    df_ranked = rank_within_groups(df, scores)

    out_csv = Path(args.out_csv)
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
