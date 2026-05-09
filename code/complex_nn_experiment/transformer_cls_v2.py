"""Extended classification-headed transformer for Phase 7 tasks.

Adds two output modes to FFTMnistClassifier:
  - 'pooled_classify': mean-pool over sequence, classify single output (existing)
  - 'multilabel': sigmoid output for multi-label classification (multi-pitch)
  - 'per_token': per-token logits (phase memory)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformer_cls import FFTMnistClassifier


class TaskTransformer(FFTMnistClassifier):
    """Extends FFTMnistClassifier with output mode flag."""

    def __init__(
        self,
        cell: str,
        num_classes: int = 10,
        output_mode: str = "pooled_classify",
        **kwargs,
    ):
        super().__init__(cell=cell, num_classes=num_classes, **kwargs)
        self.output_mode = output_mode
        # Replace classifier head if needed for per-token mode
        if output_mode == "per_token":
            in_dim = 2 * self.dim if self.is_complex else self.dim
            self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N] complex64."""
        B, N = x.shape

        if self.is_complex:
            x_in = x.unsqueeze(-1)
            h = self.embed(x_in)
        else:
            x_re_im = torch.stack([x.real, x.imag], dim=-1)
            h = self.embed(x_re_im)

        rope_inp = self._make_rope(N, x.device)

        for attn_norm, attn, ffn_norm, ffn in self.layers:
            normed = attn_norm(h)
            if self.cell.startswith("real_"):
                a = attn(normed, rope_cos_sin=rope_inp)
            else:
                a = attn(normed, rotary_emb=rope_inp)
            h = h + a
            h = h + ffn(ffn_norm(h))

        h = self.final_norm(h)

        if self.output_mode == "per_token":
            # Per-position logits, shape [B, N, num_classes]
            if self.is_complex:
                h_real = torch.cat([h.real, h.imag], dim=-1)  # [B, N, 2*dim]
                logits = self.classifier(h_real)
            else:
                logits = self.classifier(h)
            return logits
        else:
            # Pool then classify (pooled_classify or multilabel)
            h_pooled = h.mean(dim=1)
            if self.is_complex:
                h_real = torch.cat([h_pooled.real, h_pooled.imag], dim=-1)
                logits = self.classifier(h_real)
            else:
                logits = self.classifier(h_pooled)
            return logits  # [B, num_classes] (logit for classification, multilabel uses BCE)


def make_task_transformer(cell: str, output_mode: str = "pooled_classify", **kwargs):
    return TaskTransformer(cell=cell, output_mode=output_mode, **kwargs)
