# predict_all.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from GRN.model import GRN  # uses the class you already defined and trained


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_LIST)
HYDRO = set(list("AVLIMFWY"))          # simple hydrophobic set
CHARGED = set(list("KRDEH"))           # basic+acidic(+His)
POLAR = set(list("STNQCYG"))           # coarse polar set


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def seq_features_from_names(seq: str, names: List[str]) -> Dict[str, float]:
    seq = (seq or "").strip().upper()
    L = max(len(seq), 1)
    counts = {aa: 0 for aa in AA_LIST}
    for ch in seq:
        if ch in AA_SET:
            counts[ch] += 1
    fracs = {aa: counts[aa] / L for aa in AA_LIST}
    feat: Dict[str, float] = {}
    for n in names:
        if n == "seq_len":
            feat[n] = float(L)
        elif n == "frac_hydrophobic":
            feat[n] = float(sum(1 for ch in seq if ch in HYDRO)) / L
        elif n == "frac_charged":
            feat[n] = float(sum(1 for ch in seq if ch in CHARGED)) / L
        elif n == "frac_polar":
            feat[n] = float(sum(1 for ch in seq if ch in POLAR)) / L
        elif n.startswith("aa_count_") and len(n) == 9+1:
            aa = n[-1]
            feat[n] = float(counts.get(aa, 0))
        elif n.startswith("aa_frac_") and len(n) == 8+1:
            aa = n[-1]
            feat[n] = float(fracs.get(aa, 0.0))
        else:
            # unknown key: default 0
            feat[n] = 0.0
    return feat


@torch.no_grad()
def predict_for_file(
    ckpt: Dict[str, Any],
    in_path: Path,
    out_dir: Path,
    device: torch.device,
    score_mode: str = "expected_rel",
    topk: int = 50,
) -> None:
    rows = load_jsonl(in_path)
    if not rows:
        return

    base_names: List[str] = ckpt["base_feature_names"]
    seq_names: List[str] = ckpt["seq_feature_names"]
    scaler = ckpt["scaler"]
    model_args = ckpt["args"]

    d_in = len(base_names) + len(seq_names)
    d_hidden = model_args.get("hidden", 256)
    dropout = model_args.get("dropout", 0.3)
    use_rank_head = model_args.get("use_rank_head", True)

    model = GRN(
        d_in=d_in,
        d_hidden=d_hidden,
        num_classes=4,
        dropout=dropout,
        use_rank_head=use_rank_head,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Build design matrix X in the same feature order
    X_list: List[List[float]] = []
    meta: List[Tuple[str, str, int]] = []  # (bitstring, sequence, group_id)
    for r in rows:
        seq = str(r.get("sequence", ""))
        base_vec = [float(r.get(k, 0.0)) for k in base_names]
        seq_feat = seq_features_from_names(seq, seq_names)
        seq_vec = [float(seq_feat[k]) for k in seq_names]
        X_list.append(base_vec + seq_vec)
        meta.append((str(r.get("bitstring", "")), seq, int(r.get("group_id", 0))))

    X = np.asarray(X_list, dtype=np.float32)
    # Apply saved scaler if present (sklearn StandardScaler or dict with mean/std)
    if hasattr(scaler, "transform"):
        X = scaler.transform(X)
    elif isinstance(scaler, dict) and "mean_" in scaler and "scale_" in scaler:
        mean = np.asarray(scaler["mean_"], dtype=np.float32)
        scale = np.asarray(scaler["scale_"], dtype=np.float32)
        X = (X - mean) / np.where(scale == 0, 1.0, scale)

    xt = torch.from_numpy(X).to(device, non_blocking=True)
    out = model(xt)
    logits = out["logits"].float()
    probs = F.softmax(logits, dim=-1)
    if score_mode == "prob_rel3":
        score = probs[:, 3]
    elif score_mode == "logit_rel3":
        score = logits[:, 3]
    else:
        # expected relevance under {0,1,2,3}
        weights = torch.arange(4, device=device, dtype=probs.dtype)
        score = (probs * weights).sum(dim=-1)

    score_np = score.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()

    df = pd.DataFrame({
        "bitstring": [m[0] for m in meta],
        "sequence": [m[1] for m in meta],
        "group_id": [m[2] for m in meta],
        "score": score_np,
        "p_rel0": probs_np[:, 0],
        "p_rel1": probs_np[:, 1],
        "p_rel2": probs_np[:, 2],
        "p_rel3": probs_np[:, 3],
        "logit0": logits_np[:, 0],
        "logit1": logits_np[:, 1],
        "logit2": logits_np[:, 2],
        "logit3": logits_np[:, 3],
    })
    df["rank"] = (-df["score"]).rank(method="first").astype(int)

    pdb_id = in_path.stem.replace("_grn_input", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{pdb_id}_pred.csv"
    df.sort_values("score", ascending=False).to_csv(csv_path, index=False)

    # Top-K json
    top = df.nlargest(topk, "score")[["bitstring", "sequence", "group_id", "score", "p_rel3"]]
    top_json = [
        dict(row._asdict()) if hasattr(row, "_asdict") else {k: row[k] for k in top.columns}
        for _, row in top.iterrows()
    ]
    with (out_dir / f"{pdb_id}_top{topk}.json").open("w", encoding="utf-8") as f:
        json.dump(top_json, f, indent=2)

    # Raw arrays for plotting
    np.savez_compressed(
        out_dir / f"{pdb_id}_pred.npz",
        X=X,
        score=score_np,
        probs=probs_np,
        logits=logits_np,
        bitstring=np.array([m[0] for m in meta], dtype=object),
        sequence=np.array([m[1] for m in meta], dtype=object),
        group_id=np.array([m[2] for m in meta], dtype=np.int32),
        base_feature_names=np.array(base_names, dtype=object),
        seq_feature_names=np.array(seq_names, dtype=object),
    )

    print(f"[OK] {pdb_id}: {len(df)} rows -> {csv_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="prepared_dataset")
    ap.add_argument("--ckpt", type=str, default="checkpoints_full/grn_best.pt")
    ap.add_argument("--out_dir", type=str, default="predictions")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_mode", type=str, default="expected_rel", choices=["expected_rel", "prob_rel3", "logit_rel3"])
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_grn_input.jsonl"))
    if not files:
        print(f"[WARN] No *_grn_input.jsonl found in {in_dir.resolve()}")
        return

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Found {len(files)} input files.")

    for f in files:
        predict_for_file(
            ckpt=ckpt,
            in_path=f,
            out_dir=out_dir,
            device=device,
            score_mode=args.score_mode,
            topk=args.topk,
        )

    print(f"[DONE] Results at {out_dir.resolve()}")


if __name__ == "__main__":
    main()
