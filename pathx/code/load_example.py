"""Minimal usage example: load the released Path-X checkpoint and classify.

Run from the pathx/ directory:  python code/load_example.py
"""

import json
import sys

import torch

sys.path.insert(0, "code")
from pcr_screening import build_pcr_classifier  # noqa: E402

cfg = json.load(open("model/config.json"))["pcr_config"]
model = build_pcr_classifier(seq_len=16384, vocab=256, **cfg)
model.load_state_dict(
    torch.load("model/pytorch_model.pt", map_location="cpu", weights_only=True),
    strict=True,
)
model.eval()

# Input: a flattened 128x128 Pathfinder-X image as 16,384 uint8 pixel tokens.
x = torch.randint(0, 256, (1, 16384))  # replace with real LRA Path-X data
with torch.no_grad():
    logits = model(x)  # [B, 2]: P(disconnected), P(connected)
print("logits:", logits.tolist())
