# --*-- conding:utf-8 --*--
# @time:11/1/25 03:53
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:train.py

# grn_simple/train.py
import argparse
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grn_simple.data import GRNDataModule
from grn_simple.model import build_grn_from_datamodule
from grn_simple.metrics import summarize_classification, summarize_ranking


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
    all_logits = []
    all_scores = []
    all_labels = []
    all_group = []
    all_rmsd = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].cpu().numpy()
        g = batch["group_id"].cpu().numpy()
        r = batch["rmsd"].cpu().numpy()

        out = model(x)
        logits = out["logits"].detach().cpu().numpy()
        score = out["score"].detach().cpu().numpy()

        all_logits.append(logits)
        all_scores.append(score)
        all_labels.append(y)
        all_group.append(g)
        all_rmsd.append(r)

    logits = np.concatenate(all_logits, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_group, axis=0)
    rmsd = np.concatenate(all_rmsd, axis=0)

    cls = summarize_classification(logits, labels)
    rnk = summarize_ranking(logits, scores, labels, rmsd, groups, ks=(5, 10, 20))
    return {**cls, **rnk}


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = out["logits"]
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)
    return total_loss / max(1, total_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--score_mode", type=str, default="prob_rel3", choices=["prob_rel3", "logit_rel3", "expected_rel"])
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

    # Optimizer / Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_epoch = -1
    patience_left = args.patience
    ckpt_path = Path(args.save_dir) / "grn_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, dm.train_dataloader(), optimizer, device, loss_fn)
        val_metrics = evaluate(model, dm.valid_dataloader(), device)
        monitor = val_metrics.get("ndcg@10", 0.0)  # primary early-stopping metric

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} | val_ndcg@10={val_metrics['ndcg@10']:.4f} | "
              f"val_spearman={val_metrics['spearman']:.4f}")

        if monitor > best_val:
            best_val = monitor
            best_epoch = epoch
            patience_left = args.patience

            # Save checkpoint with model + data normalization meta
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "feature_names": dm.feature_names,
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

    # Final test on best ckpt
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
