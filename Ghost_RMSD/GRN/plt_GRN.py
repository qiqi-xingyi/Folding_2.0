# --*-- conding:utf-8 --*--
# @time:11/5/25 12:17
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_GRN.py

import torch
from torchviz import make_dot
from model import GRNClassifier

# === Initialize your model ===
in_dim = 285  # feature_dim() from your datamodule
model = GRNClassifier(in_dim=in_dim, hidden_dims=[256,128], dropout=0.3, use_bn=True, use_rank_head=True)

# === Dummy input ===
x = torch.randn(1, in_dim)

if __name__ == '__main__':

    # === Forward once ===
    out = model(x)

    # === Visualize computation graph ===
    dot = make_dot(
        (out["logits"], out["score"]),  # two outputs
        params=dict(model.named_parameters()),
        show_attrs=False, show_saved=False
    )

    dot.format = "pdf"    # 'svg' or 'png' also fine
    dot.render("GRN_Architecture")   # produces GRN_Architecture.pdf

