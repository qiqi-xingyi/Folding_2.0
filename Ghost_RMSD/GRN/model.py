# --*-- conding:utf-8 --*--
# @time:11/1/25 03:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:model.py

# grn_simple/model.py
import math
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Linear -> ReLU -> Dropout (optional) with optional BatchNorm."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


class GRNClassifier(nn.Module):
    """
    Minimal classifier for Ghost RMSD labels (rel ∈ {0,1,2,3}).
    - Input: numeric feature vector (z-scored), shape [B, D]
    - Output:
        logits: [B, 4]
        probs : [B, 4] (softmax)
        score : [B]   (scalar ranking score; larger => better)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_bn: bool = True,
        score_mode: str = "prob_rel3",  # "prob_rel3" | "logit_rel3" | "expected_rel"
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: List[nn.Module] = []
        d_prev = in_dim
        for d in hidden_dims:
            layers.append(MLPBlock(d_prev, d, dropout=dropout, use_bn=use_bn))
            d_prev = d
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        self.classifier = nn.Linear(d_prev, 4)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        self.score_mode = score_mode

    @torch.no_grad()
    def _make_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to a scalar ranking score.
        Larger score should indicate higher quality (lower RMSD).
        """
        if self.score_mode == "prob_rel3":
            probs = F.softmax(logits, dim=-1)
            return probs[..., 3]  # probability of best class
        if self.score_mode == "logit_rel3":
            return logits[..., 3]
        if self.score_mode == "expected_rel":
            probs = F.softmax(logits, dim=-1)
            levels = torch.arange(0, 4, device=logits.device, dtype=probs.dtype)
            return (probs * levels).sum(dim=-1)
        # default fallback
        probs = F.softmax(logits, dim=-1)
        return probs[..., 3]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        logits = self.classifier(h)
        probs = F.softmax(logits, dim=-1)
        score = self._make_score(logits)
        return {"logits": logits, "probs": probs, "score": score}


def build_grn_from_datamodule(dm, dropout: float = 0.2, use_bn: bool = True, score_mode: str = "prob_rel3") -> GRNClassifier:
    """
    Convenience constructor when you already have a GRNDataModule.
    """
    in_dim = dm.feature_dim()
    return GRNClassifier(in_dim=in_dim, dropout=dropout, use_bn=use_bn, score_mode=score_mode)
