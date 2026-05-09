"""Classification-headed transformer for FFT-MNIST and similar tasks.

Differences from `transformer.TransformerCell`:
- Input is pre-tokenized features `[B, N, D_in]` (real or complex), not integer ids
- Output is a fixed `num_classes` vector (10 for MNIST), pooled over sequence
- Token embedding: linear projection from D_in to dim (real or complex linear)

§4.6 fairness preserved:
- All cells share depth, heads, dim_head, FFN, optimizer
- softmax cells: row-softmax only (natural form)
- screening cells: TanhNorm only (natural form)
- RoPE used uniformly when sequence length > 1
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from baselines.lucidrains_cvt.complex_valued_transformer import (
    ComplexFeedForward,
    ComplexRMSNorm,
    RotaryEmbedding,
)
from screening import (
    ScreeningAttention,
    ScreeningAttentionReal,
    complex_glorot_init_,
)
from transformer import (
    RealRMSNorm,
    RealRotaryEmbedding,
    RealMultiHeadSoftmaxAttention,
    RealMultiHeadSigmoidAttention,
    RealMultiHeadTanh1Attention,
    RealMultiHeadReLUAttention,
    ComplexMultiHeadSoftmaxAttention,
    ComplexMultiHeadSigmoidAttention,
    ComplexMultiHeadSigmoidAttention_FreeReal,
    ComplexMultiHeadTanh1Attention,
    ComplexMultiHeadReLUAttention,
    RealFeedForward,
)


class FFTMnistClassifier(nn.Module):
    """4-cell transformer classifier for FFT-MNIST input.

    Input: [B, N] complex64 tensor (raw 2D FFT coefs flattened).
    Output: [B, num_classes] real logits.

    For real cells: input is interpreted as 2-channel real per token (Re, Im split).
    For complex cells: input is used as complex token directly.
    """

    def __init__(
        self,
        cell: str,
        num_classes: int = 10,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        ff_mult: int = 4,
        causal: bool = False,
        rotary: bool = True,
        use_softmask: bool = False,  # Phase 14 audit: default OFF (softmask=ON breaks learning)
        use_tanhnorm: bool = True,
        softmask_init_width: float = 32.0,
        s_r_init: float = -3.0,
        sigmoid_seq_len: int = 64,
        sigmoid_bias_init_override: float | None = None,
    ):
        super().__init__()
        assert cell in ("real_softmax", "real_sigmoid", "real_tanh1", "real_relu", "real_screen",
                        "complex_softmax", "complex_sigmoid", "complex_tanh1", "complex_relu", "complex_screen",
                        "complex_sigmoid_nocval", "complex_sigmoid_freereal", "complex_sigmoid_realqk",
                        "complex_softplus", "complex_cubic", "complex_clamped_relu")
        self.cell = cell
        self.is_complex = cell.startswith("complex_")
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads

        # Token embedding: 1 complex feature → dim features
        if self.is_complex:
            self.embed = nn.Linear(1, dim, bias=False, dtype=torch.complex64)
            complex_glorot_init_(self.embed)
            norm_cls = ComplexRMSNorm
        else:
            # Real cell: take (Re, Im) as 2 real features → dim
            self.embed = nn.Linear(2, dim, bias=False)
            nn.init.xavier_normal_(self.embed.weight)
            norm_cls = RealRMSNorm

        # Positional encoding
        self.rotary = rotary
        if rotary:
            if self.is_complex:
                self.rope = RotaryEmbedding(dim_head)
            else:
                self.rope = RealRotaryEmbedding(dim_head)
        else:
            self.rope = None

        # Transformer layers (re-use the building blocks from transformer.py)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            attn_norm = norm_cls(dim)
            ffn_norm = norm_cls(dim)
            if cell == "real_softmax":
                attn = RealMultiHeadSoftmaxAttention(dim, heads, dim_head, causal=causal)
                ffn = RealFeedForward(dim, mult=ff_mult)
            elif cell == "real_sigmoid":
                attn = RealMultiHeadSigmoidAttention(dim, heads, dim_head, causal=causal,
                                                       seq_len_for_bias=sigmoid_seq_len)
                ffn = RealFeedForward(dim, mult=ff_mult)
            elif cell == "real_tanh1":
                attn = RealMultiHeadTanh1Attention(dim, heads, dim_head, causal=causal,
                                                     seq_len_for_bias=sigmoid_seq_len)
                ffn = RealFeedForward(dim, mult=ff_mult)
            elif cell == "real_relu":
                attn = RealMultiHeadReLUAttention(dim, heads, dim_head, causal=causal)
                ffn = RealFeedForward(dim, mult=ff_mult)
            elif cell == "real_screen":
                attn = ScreeningAttentionReal(
                    dim, heads, dim_head, causal=causal,
                    use_softmask=use_softmask, use_tanhnorm=use_tanhnorm,
                    softmask_init_width=softmask_init_width, s_r_init=s_r_init,
                )
                ffn = RealFeedForward(dim, mult=ff_mult)
            elif cell == "complex_softmax":
                attn = ComplexMultiHeadSoftmaxAttention(dim, heads, dim_head, causal=causal)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_sigmoid":
                attn = ComplexMultiHeadSigmoidAttention(dim, heads, dim_head, causal=causal,
                                                          seq_len_for_bias=sigmoid_seq_len,
                                                          bias_init_override=sigmoid_bias_init_override)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_sigmoid_nocval":
                attn = ComplexMultiHeadSigmoidAttention(dim, heads, dim_head, causal=causal,
                                                          seq_len_for_bias=sigmoid_seq_len,
                                                          cval=False,
                                                          bias_init_override=sigmoid_bias_init_override)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_sigmoid_freereal":
                attn = ComplexMultiHeadSigmoidAttention_FreeReal(dim, heads, dim_head, causal=causal,
                                                                   seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_sigmoid_realqk":
                attn = ComplexMultiHeadSigmoidAttention(dim, heads, dim_head, causal=causal,
                                                          seq_len_for_bias=sigmoid_seq_len,
                                                          realqk=True,
                                                          bias_init_override=sigmoid_bias_init_override)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_tanh1":
                attn = ComplexMultiHeadTanh1Attention(dim, heads, dim_head, causal=causal,
                                                       seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_relu":
                attn = ComplexMultiHeadReLUAttention(dim, heads, dim_head, causal=causal,
                                                       seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_softplus":
                from transformer import ComplexMultiHeadSoftplusAttention  # noqa: E402
                attn = ComplexMultiHeadSoftplusAttention(dim, heads, dim_head, causal=causal,
                                                          seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_cubic":
                from transformer import ComplexMultiHeadCubicAttention  # noqa: E402
                attn = ComplexMultiHeadCubicAttention(dim, heads, dim_head, causal=causal,
                                                       seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_clamped_relu":
                from transformer import ComplexMultiHeadClampedReLUAttention  # noqa: E402
                attn = ComplexMultiHeadClampedReLUAttention(dim, heads, dim_head, causal=causal,
                                                              seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            else:  # complex_screen
                attn = ScreeningAttention(
                    dim, heads, dim_head, causal=causal,
                    use_softmask=use_softmask, use_tanhnorm=use_tanhnorm,
                    softmask_init_width=softmask_init_width, s_r_init=s_r_init,
                )
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            self.layers.append(nn.ModuleList([attn_norm, attn, ffn_norm, ffn]))

        self.final_norm = norm_cls(dim)

        # Classification head: pool over sequence (mean) + linear
        if self.is_complex:
            # complex pooled vector → concat[Re, Im] → real classifier
            self.classifier = nn.Linear(2 * dim, num_classes)
        else:
            self.classifier = nn.Linear(dim, num_classes)

    def _make_rope(self, N: int, device: torch.device):
        if self.rope is None:
            return None
        if self.is_complex:
            return self.rope(N).to(device)
        return self.rope(N, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N] complex64. Returns [B, num_classes] real."""
        B, N = x.shape

        if self.is_complex:
            # Project complex tokens (B, N, 1) → (B, N, dim) via complex linear
            x_in = x.unsqueeze(-1)  # [B, N, 1] complex
            h = self.embed(x_in)  # [B, N, dim] complex
        else:
            # Split complex into (Re, Im) features per token → real linear
            x_re_im = torch.stack([x.real, x.imag], dim=-1)  # [B, N, 2] real
            h = self.embed(x_re_im)  # [B, N, dim] real

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

        # Mean pool over sequence
        h_pooled = h.mean(dim=1)  # [B, dim]

        if self.is_complex:
            h_real = torch.cat([h_pooled.real, h_pooled.imag], dim=-1)  # [B, 2*dim]
            logits = self.classifier(h_real)
        else:
            logits = self.classifier(h_pooled)
        return logits


def make_fftmnist_classifier(cell: str, **kwargs) -> FFTMnistClassifier:
    return FFTMnistClassifier(cell=cell, **kwargs)
