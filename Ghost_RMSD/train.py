# --*-- conding:utf-8 --*--
# @time:11/1/25 03:53
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:train.py

import argparse
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from GRN.data import GRNDataModule
from GRN.model import build_grn_from_datamodule
from GRN.metrics import summarize_classification, summarize_ranking
from GRN.losses import build_pairs, ranknet_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits, all_scores, all_labels, all_groups, all_rmsd = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        out = model(x)
        all_logits.append(out["logits"].cpu().numpy())
        all_scores.append(out["score"].cpu().numpy())
        all_labels.append(batch["y"].cpu().numpy())
        all_groups.append(batch["group_id"].cpu().numpy())
        all_rmsd.append(batch["rmsd"].cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    rmsd = np.concatenate(all_rmsd, axis=0)

    cls = summarize_classification(logits, labels)
    rnk = summarize_ranking(logits, scores, labels, rmsd, groups, ks=(5, 10, 20))
    return {**cls, **rnk}


def train_one_epoch(model, loader, optimizer, device, ce_loss, lambda_ce: float, max_pairs_per_group: int):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        gid = batch["group_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = out["logits"]
        scores = out["score"]

        # classification loss
        L_ce = ce_loss(logits, y)

        # pairwise ranknet loss (build pairs on CPU, then move to device indexing)
        pairs = build_pairs(gid.cpu(), y.cpu(), max_pairs_per_group=max_pairs_per_group)
        L_rank = ranknet_loss(scores, pairs.to(device)) if pairs.numel() > 0 else torch.zeros([], device=device)

        loss = L_rank + lambda_ce * L_ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(1, total_n)


def compute_class_weight(train_ds) -> torch.Tensor:
    y = train_ds.y.numpy()
    counts = np.bincount(y, minlength=4).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv * (counts.sum() / len(counts))
    return torch.tensor(w, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--score_mode", type=str, default="expected_rel",
                        choices=["prob_rel3", "logit_rel3", "expected_rel"])
    parser.add_argument("--lambda_ce", type=float, default=0.5)
    parser.add_argument("--max_pairs_per_group", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    dm = GRNDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    dm.setup()

    # Model
    model = build_grn_from_datamodule(dm, dropout=args.dropout, score_mode=args.score_mode)
    model.to(device)

    # Optimizer / Loss (class weights to fight imbalance)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weight = compute_class_weight(dm.ds_train).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weight)

    best_val = -1.0
    best_epoch = -1
    patience_left = args.patience
    ckpt_path = Path(args.save_dir) / "grn_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, dm.train_dataloader(), optimizer, device,
            ce_loss, args.lambda_ce, args.max_pairs_per_group
        )
        val_metrics = evaluate(model, dm.valid_dataloader(), device)
        monitor = val_metrics.get("ndcg@10", 0.0)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} | val_ndcg@10={val_metrics['ndcg@10']:.4f} | "
              f"val_spearman={val_metrics['spearman']:.4f}")

        if monitor > best_val:
            best_val = monitor
            best_epoch = epoch
            patience_left = args.patience
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "base_feature_names": dm.base_feature_names,
                "seq_feature_names": dm.seq_feature_names,
                "scaler": dm.scaler,
                "args": vars(args),
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  -> Saved new best to {ckpt_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best epoch {best_epoch} (val_ndcg@10={best_val:.4f}).")
                break

    # Test with best ckpt
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_metrics = evaluate(model, dm.test_dataloader(), device)
        print(f"[TEST] acc={test_metrics['acc']:.4f} | "
              f"ndcg@5={test_metrics['ndcg@5']:.4f} | ndcg@10={test_metrics['ndcg@10']:.4f} | "
              f"ndcg@20={test_metrics['ndcg@20']:.4f} | spearman={test_metrics['spearman']:.4f}")
    else:
        print("No checkpoint found; skip test.")


if __name__ == "__main__":
    main()
