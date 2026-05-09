"""4-cell transformer factory: real/complex × softmax/screening.

All 4 cells share the same skeleton (depth, heads, dim_head, ffn) and follow the
plan §4.6 fairness rule (each gate keeps its natural magnitude regulator).

Cells:
  real-softmax    — standard real transformer with softmax MHA
  real-screen     — standard real transformer with screening attention
  complex-softmax — lucidrains' Eilers ℂAtt complex transformer
  complex-screen  — complex transformer with paper-faithful screening attention

All cells use RoPE (positional encoding axis unified per plan §10).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from baselines.lucidrains_cvt.complex_valued_transformer import (
    ComplexMultiheadAttention as LucidrainsComplexMHA,
    ComplexRMSNorm,
    ComplexFeedForward,
    RotaryEmbedding,
)
from screening import (
    ScreeningAttention,
    ScreeningAttentionReal,
    complex_glorot_init_,
    complex_l2_normalize,
)


# --------------------------- Real building blocks ---------------------------


class RealRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class RealRotaryEmbedding(nn.Module):
    """RoPE for real q, k.

    Standard RoPE: split last dim into pairs (a, b), rotate each pair by angle θ_i.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))  # [N, D/2]
        return freqs.cos(), freqs.sin()  # each [N, D/2]


def apply_rope_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., N, D] real. cos, sin: [N, D/2]."""
    a, b = x[..., 0::2], x[..., 1::2]  # [..., N, D/2]
    cos_ = cos.view(*([1] * (x.ndim - 2)), x.shape[-2], -1)
    sin_ = sin.view(*([1] * (x.ndim - 2)), x.shape[-2], -1)
    rot_a = a * cos_ - b * sin_
    rot_b = a * sin_ + b * cos_
    out = torch.stack([rot_a, rot_b], dim=-1).flatten(start_dim=-2)
    return out


class RealMultiHeadSoftmaxAttention(nn.Module):
    """Standard real MHA with RoPE."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        dim_inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        nn.init.xavier_normal_(self.to_qkv.weight)
        nn.init.xavier_normal_(self.to_out.weight)

    def forward(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q = apply_rope_real(q, cos, sin)
            k = apply_rope_real(k, cos, sin)
        scale = self.dim_head ** -0.5
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * scale
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, float("-inf"))
        attn = sim.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def RealFeedForward(dim: int, mult: int = 4) -> nn.Module:
    inner = dim * mult
    return nn.Sequential(
        nn.Linear(dim, inner),
        nn.SiLU(),
        nn.Linear(inner, dim),
    )


# --------------------------- Complex softmax wrapper ---------------------------


class ComplexMultiHeadSoftmaxAttention(nn.Module):
    """Wraps lucidrains' ComplexMultiheadAttention with RoPE (Eilers ℂAtt, real-component score)."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.attn = LucidrainsComplexMHA(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            causal=causal,
            complete_complex=False,  # Eilers ℂAtt (real-component dot-product)
            flash=False,
        )

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        return self.attn(x, rotary_emb=rotary_emb)


# --------------------------- Sigmoid attention (Ramapuram 2024) ---------------------------


class RealMultiHeadSigmoidAttention(nn.Module):
    """Sigmoid attention per Ramapuram et al. 2024.

    Replaces row-softmax with element-wise sigmoid + learnable bias init at -log(N).
    No row normalization. Each (i,j) pair independently decides attention weight.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        dim_inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        nn.init.xavier_normal_(self.to_qkv.weight)
        nn.init.xavier_normal_(self.to_out.weight)
        # Per Ramapuram 2024: init bias to -log(N) for stability
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(float(seq_len_for_bias))))

    def forward(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q = apply_rope_real(q, cos, sin)
            k = apply_rope_real(k, cos, sin)
        scale = self.dim_head ** -0.5
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * scale
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, -1e9)
        attn = torch.sigmoid(sim + self.attn_bias)  # NO row-norm
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ComplexMultiHeadSigmoidAttention(nn.Module):
    """Complex sigmoid attention.

    Real-valued cosine score Re(⟨q̄, k̄⟩) ∈ [-1, 1] after L2-normalization.
    Element-wise sigmoid weight (real-valued in [0, 1]), aggregated against complex v.
    No row-norm. Bias init at -log(N) per Ramapuram 2024.

    Substrate ablation flags:
      cval=True  : complex value path (default, paper baseline). real α × complex V → complex out.
      cval=False : real value path. v.imag is forced to 0 → real α × real(V) → output Im=0.
                   Tests whether the complex value path is structurally necessary.
      realqk=False: complex Q, K (default).
      realqk=True : Im(Q) := Im(K) := 0 forced before cosine score. Tests whether complex
                    Q, K (and hence the phase-aware similarity score) is structurally
                    necessary. With realqk=True + cval=True, this is "real Q,K + complex V".
      bias_init_override : if not None, overrides −log(N) bias initialization.
                            Used for bias-init sweep ablation.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64, eps: float = 1e-8,
                 cval: bool = True, realqk: bool = False, bias_init_override: float | None = None,
                 chunk_size: int | None = None,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.eps = eps
        self.cval = cval
        self.realqk = realqk
        self.chunk_size = chunk_size
        # Path-X (16384 seq) memory: stored sigmoid attn-weights tensor is
        # ~25 GB at micro_batch=1, dominating peak GPU memory. Enabling
        # grad checkpointing on the chunked-attention call recomputes those
        # tensors during backward, freeing ~25 GB at the cost of ~30-50%
        # more compute. Turn on via env GRAD_CKPT=1 in the entrypoint.
        self.gradient_checkpointing = gradient_checkpointing
        dim_inner = heads * dim_head
        from screening import complex_glorot_init_, complex_l2_normalize  # local import to avoid cycle
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False, dtype=torch.complex64)
        self.to_out = nn.Linear(dim_inner, dim, bias=False, dtype=torch.complex64)
        complex_glorot_init_(self.to_qkv)
        complex_glorot_init_(self.to_out)
        self._l2 = complex_l2_normalize
        bias_val = bias_init_override if bias_init_override is not None else -math.log(float(seq_len_for_bias))
        self.attn_bias = nn.Parameter(torch.tensor(float(bias_val)))

    def _attend_chunked(self, q_bar: torch.Tensor, k_bar: torch.Tensor, v: torch.Tensor,
                        chunk: int) -> torch.Tensor:
        """Memory-efficient attention by chunking over Q.

        Sigmoid attention has no row-normalization (no Σ_j α_ij = 1 constraint), so
        each query row can be computed independently against the full K/V. We split
        Q into chunks of size `chunk` and accumulate the per-chunk output. Peak
        attention-matrix memory is O(B·H·chunk·N) instead of O(B·H·N²).

        Used for L=16K (Path-X) where the full N×N matrix is infeasible.
        """
        B, H, N, D = q_bar.shape
        scale = self.dim_head ** 0.5
        out = torch.zeros(B, H, N, D, dtype=v.dtype, device=v.device)
        for q_start in range(0, N, chunk):
            q_end = min(q_start + chunk, N)
            q_part = q_bar[:, :, q_start:q_end]                                          # [B, H, qc, D]
            sim_part = torch.einsum("bhid,bhjd->bhij", q_part.conj(), k_bar).real * scale  # [B, H, qc, N]
            if self.causal:
                qi = torch.arange(q_start, q_end, device=v.device).unsqueeze(-1)
                kj = torch.arange(N, device=v.device).unsqueeze(0)
                mask = qi < kj                                                            # [qc, N]
                sim_part = sim_part.masked_fill(mask, -1e9)
            attn_part = torch.sigmoid(sim_part + self.attn_bias)                          # real ∈ [0,1]
            out[:, :, q_start:q_end] = torch.einsum("bhij,bhjd->bhid", attn_part.to(v.dtype), v)
        return out

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rotary_emb is not None:
            q = q * rotary_emb
            k = k * rotary_emb
        if self.realqk:
            # realqk-on: force imaginary part of Q, K to zero before cosine score.
            # Tests whether the complex Q,K (and hence phase-aware similarity) is structurally
            # necessary. With realqk=True + cval=True: "real Q,K + complex V" variant.
            q = torch.complex(q.real, torch.zeros_like(q.imag))
            k = torch.complex(k.real, torch.zeros_like(k.imag))
        # L2 normalize → score in [-1, 1]
        q_bar = self._l2(q, dim=-1, eps=self.eps)
        k_bar = self._l2(k, dim=-1, eps=self.eps)
        if not self.cval:
            # cval-off: force imaginary part of value to zero (architectural ablation:
            # complex Q,K used for cosine score, but value path is real-valued).
            v = torch.complex(v.real, torch.zeros_like(v.imag))
        if self.chunk_size is not None and N > self.chunk_size:
            if self.gradient_checkpointing and self.training:
                out = torch.utils.checkpoint.checkpoint(
                    lambda qa, ka, va: self._attend_chunked(qa, ka, va, self.chunk_size),
                    q_bar, k_bar, v, use_reentrant=False,
                )
            else:
                out = self._attend_chunked(q_bar, k_bar, v, self.chunk_size)
        else:
            sim = torch.einsum("bhid,bhjd->bhij", q_bar.conj(), k_bar).real  # [B, h, N, N] real
            sim = sim * (self.dim_head ** 0.5)
            if self.causal:
                mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
                sim = sim.masked_fill(mask, -1e9)
            attn = torch.sigmoid(sim + self.attn_bias)  # real weights ∈ [0, 1]
            out = torch.einsum("bhij,bhjd->bhid", attn.to(v.dtype), v)  # complex v aggregated by real weights
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ComplexMultiHeadSigmoidAttention_FreeReal(nn.Module):
    """Complex sigmoid attention variant — free real linear instead of native complex linear.

    Substrate ablation: replaces nn.Linear(dtype=complex64) (which encodes Cauchy-Riemann
    constraint, i.e. (a+bi)·z form with 2 free reals per element) with two independent
    real nn.Linear layers, allowing a fully general 4-real-parameter map per element:
        out_re = W_rr·in_re + W_ir·in_im
        out_im = W_ri·in_re + W_ii·in_im
    where W_rr, W_ir, W_ri, W_ii are independent real weights.

    The native complex form ties W_rr = W_ii and W_ir = -W_ri (Cauchy-Riemann);
    free-real form has all four weights independent. If free-real ≈ native, the tie
    is interchangeable; if native > free-real, the holomorphic prior matters.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64, eps: float = 1e-8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.eps = eps
        dim_inner = heads * dim_head
        # Free real linear: 4 separate maps for (Re→Re, Im→Re, Re→Im, Im→Im) for each of QKV.
        # Implemented as two real linears producing real and imag outputs from concatenated [Re; Im] input.
        self.to_qkv_re = nn.Linear(2 * dim, 3 * dim_inner, bias=False)
        self.to_qkv_im = nn.Linear(2 * dim, 3 * dim_inner, bias=False)
        self.to_out_re = nn.Linear(2 * dim_inner, dim, bias=False)
        self.to_out_im = nn.Linear(2 * dim_inner, dim, bias=False)
        for L in (self.to_qkv_re, self.to_qkv_im, self.to_out_re, self.to_out_im):
            nn.init.xavier_normal_(L.weight, gain=1.0 / math.sqrt(2))  # split 1/sqrt(2) to match complex Glorot variance
        from screening import complex_l2_normalize
        self._l2 = complex_l2_normalize
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(float(seq_len_for_bias))))

    def _free_real_complex_linear(self, x_complex: torch.Tensor, L_re: nn.Linear, L_im: nn.Linear) -> torch.Tensor:
        x_concat = torch.cat([x_complex.real, x_complex.imag], dim=-1)
        out_re = L_re(x_concat)
        out_im = L_im(x_concat)
        return torch.complex(out_re, out_im)

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self._free_real_complex_linear(x, self.to_qkv_re, self.to_qkv_im).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rotary_emb is not None:
            q = q * rotary_emb
            k = k * rotary_emb
        q_bar = self._l2(q, dim=-1, eps=self.eps)
        k_bar = self._l2(k, dim=-1, eps=self.eps)
        sim = torch.einsum("bhid,bhjd->bhij", q_bar.conj(), k_bar).real
        sim = sim * (self.dim_head ** 0.5)
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, -1e9)
        attn = torch.sigmoid(sim + self.attn_bias)
        out = torch.einsum("bhij,bhjd->bhid", attn.to(v.dtype), v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self._free_real_complex_linear(out, self.to_out_re, self.to_out_im)


# --------------------------- (tanh+1) attention (sigmoid replacement, ad-hoc) ---------------------------
# Replaces sigmoid with (tanh(s + b) + 1) ∈ (0, 2).  Note tanh(s) + 1 = 2*σ(2s)
# — i.e. this is sigmoid with output range x2 and gradient steepness x2.
# Same structural choices as sigmoid attention (Ramapuram 2024 form): no row-norm,
# learnable bias init = -log(N).


class RealMultiHeadTanh1Attention(nn.Module):
    """Tanh+1 attention (real). attn = tanh(QK^T/sqrt(d) + b) + 1, no row-norm."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        dim_inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        nn.init.xavier_normal_(self.to_qkv.weight)
        nn.init.xavier_normal_(self.to_out.weight)
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(float(seq_len_for_bias))))

    def forward(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q = apply_rope_real(q, cos, sin)
            k = apply_rope_real(k, cos, sin)
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * (self.dim_head ** -0.5)
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, -1e9)
        attn = torch.tanh(sim + self.attn_bias) + 1.0  # ∈ (0, 2)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ComplexMultiHeadTanh1Attention(nn.Module):
    """Complex (tanh+1) attention (sigmoid-style complexification with σ replaced by tanh+1).

    Same structure as ComplexMultiHeadSigmoidAttention, ONLY the activation is changed:
      - L2-normalize q, k → score = Re⟨q̄, k̄⟩ × √d   (identical to complex_sigmoid)
      - attn = tanh(score + b) + 1                   (replaces σ; ∈ (0, 2), b init = -log(N))
      - real-weight × complex-value aggregation, no row-norm
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64, eps: float = 1e-8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.eps = eps
        dim_inner = heads * dim_head
        from screening import complex_glorot_init_, complex_l2_normalize
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False, dtype=torch.complex64)
        self.to_out = nn.Linear(dim_inner, dim, bias=False, dtype=torch.complex64)
        complex_glorot_init_(self.to_qkv)
        complex_glorot_init_(self.to_out)
        self._l2 = complex_l2_normalize
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(float(seq_len_for_bias))))

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rotary_emb is not None:
            q = q * rotary_emb
            k = k * rotary_emb
        q_bar = self._l2(q, dim=-1, eps=self.eps)
        k_bar = self._l2(k, dim=-1, eps=self.eps)
        sim = torch.einsum("bhid,bhjd->bhij", q_bar.conj(), k_bar).real * (self.dim_head ** 0.5)
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, -1e9)
        attn = torch.tanh(sim + self.attn_bias) + 1.0  # ∈ (0, 2)
        out = torch.einsum("bhij,bhjd->bhid", attn.to(v.dtype), v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# --------------------------- ReLU attention (Wortsman et al. 2023, paper-faithful) ---------------------------
# "Replacing softmax with ReLU in Vision Transformers" (Wortsman, Lee, Gilmer, Kornblith — 2023).
# Their proposal:  attention(Q, K, V) = [ReLU(QK^T / sqrt(d)) / L] V,   L = sequence length.
# Notes preserved exactly from the paper:
#   - plain ReLU (NOT squared ReLU)
#   - divided by L (sequence length normalization)
#   - NO row-softmax
#   - NO L2-normalization of q, k
#   - NO learnable bias / threshold
# Complex extension: minimal-complexification path matching Eilers ℂAtt (complete_complex=False) —
#   the score is the real component of the complex dot product, value path stays complex (cval).


class RealMultiHeadReLUAttention(nn.Module):
    """ReLU attention per Wortsman et al. 2023, "Replacing softmax with ReLU in ViT".

    attn_ij = ReLU(QK^T / sqrt(d)) / L      (no row-softmax; /L = seq-len normalization).
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        dim_inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        nn.init.xavier_normal_(self.to_qkv.weight)
        nn.init.xavier_normal_(self.to_out.weight)

    def forward(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q = apply_rope_real(q, cos, sin)
            k = apply_rope_real(k, cos, sin)
        scale = self.dim_head ** -0.5
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * scale     # QK^T / sqrt(d)
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, 0.0)  # post-ReLU these become 0; we want them killed
        attn = F.relu(sim) / float(N)                            # ReLU(.) / L
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ComplexMultiHeadReLUAttention(nn.Module):
    """Complex ReLU attention — sigmoid/screening-style complexification.

    Mirrors complex_sigmoid's structural choices but uses ReLU instead of σ:
      - complex L2-normalized q, k     → bounded cosine score (same as screen / complex_sigmoid)
      - score_ij = Re(<q_bar, k_bar>) * sqrt(d)                                ∈ [-sqrt(d), sqrt(d)]
      - attn_ij  = ReLU(score_ij + b)  with learnable b initialized to -log(N)  (same init as sigmoid)
      - NO row-softmax, NO /L
      - real-weight × complex-value aggregation (cval, same as Eilers ℂAtt / screen)

    Compared to complex_sigmoid: replaces soft saturating σ with hard cutoff ReLU, otherwise identical.
    Compared to complex_screen:  drops r² rescaling, squared term, TanhNorm, modReLU gate path.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64, eps: float = 1e-8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.eps = eps
        dim_inner = heads * dim_head
        from screening import complex_glorot_init_, complex_l2_normalize  # local import to avoid cycle
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False, dtype=torch.complex64)
        self.to_out = nn.Linear(dim_inner, dim, bias=False, dtype=torch.complex64)
        complex_glorot_init_(self.to_qkv)
        complex_glorot_init_(self.to_out)
        self._l2 = complex_l2_normalize
        # Learnable threshold bias, init = -log(N) (same as complex_sigmoid).
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(float(seq_len_for_bias))))

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rotary_emb is not None:
            q = q * rotary_emb
            k = k * rotary_emb
        # Complex L2-normalize → cosine score in [-1, 1], then scale by sqrt(d).
        q_bar = self._l2(q, dim=-1, eps=self.eps)
        k_bar = self._l2(k, dim=-1, eps=self.eps)
        sim = torch.einsum("bhid,bhjd->bhij", q_bar.conj(), k_bar).real
        sim = sim * (self.dim_head ** 0.5)
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, -1e9)
        attn = F.relu(sim + self.attn_bias)             # ReLU with learnable threshold; no /L, no row-norm
        out = torch.einsum("bhij,bhjd->bhid", attn.to(v.dtype), v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# --------------------------- C2-isolation cell (four-condition framework) ---------------------------
# See ssh/c2_isolation_experiment.md.
# complex_softplus = PCT (sigmoid) with the only difference that the gate output is unbounded above.
# C1✓ C2✗ C3✓ C4✓. Tests whether C2 alone is enough to break L2 cascade phase stability.


class ComplexMultiHeadSoftplusAttention(nn.Module):
    """Complex softplus attention — pure C2 violation, otherwise identical to PCT.

    α_ij = softplus(score_ij + b) = log(1 + exp(score_ij + b))   (real, ≥0, unbounded above)
    Gradient f'(s) = sigmoid(s) ∈ (0, 1)  — same shape as PCT's gate gradient.

    C1✓ C2✗ C3✓ C4✓. The only structural difference vs PCT (sigmoid) is C2 (gate output bound).
    Tests: cascade magnitude divergence at depth from C2 violation alone, with C3 strictly intact.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64, eps: float = 1e-8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.eps = eps
        dim_inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False, dtype=torch.complex64)
        self.to_out = nn.Linear(dim_inner, dim, bias=False, dtype=torch.complex64)
        complex_glorot_init_(self.to_qkv)
        complex_glorot_init_(self.to_out)
        self._l2 = complex_l2_normalize
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(float(seq_len_for_bias))))

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rotary_emb is not None:
            q = q * rotary_emb
            k = k * rotary_emb
        q_bar = self._l2(q, dim=-1, eps=self.eps)
        k_bar = self._l2(k, dim=-1, eps=self.eps)
        sim = torch.einsum("bhid,bhjd->bhij", q_bar.conj(), k_bar).real
        sim = sim * (self.dim_head ** 0.5)
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, -1e9)
        attn = F.softplus(sim + self.attn_bias)  # real, ≥0, unbounded above
        out = torch.einsum("bhij,bhjd->bhid", attn.to(v.dtype), v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# --------------------------- Two-axis isolation cells (C2 × C3) ---------------------------
# See ssh/c2_c3_isolation_experiment.md.
# complex_cubic        — strong C2 violation, C3 strict ✓ (smooth, derivative ≥ 1 everywhere)
# complex_clamped_relu — strict C2 ✓ (M=1, matches sigmoid), C3 fully violated (zero gradient
#                        on s+b<0 and s+b>1)
# Together with complex_sigmoid (PCT, C2 ✓ C3 ✓) and complex_relu (C2 partial, C3 ✗) these
# four cells form a 2×2 isolation of C2 vs C3.


class ComplexMultiHeadCubicAttention(nn.Module):
    """Complex cubic attention — pure C2 violation (large M), C3 strictly satisfied.

    α_ij = (s_ij + b) + (s_ij + b)³ / 6     (Taylor expansion of sinh up to cubic; real)
    Gradient f'(z) = 1 + z²/2 ≥ 1 everywhere — strictly C3 satisfied (with margin).

    In the L2-normalised operating range s ∈ [-√d, √d] (d=128 → ~11.3), the gate magnitude
    reaches |√d + d^{1.5}/6| ≈ 252, ~250× larger than sigmoid's M=1. C2 is violated
    *in operating range* (not merely on ℝ as for softplus), giving a clean genuine-C2-violation
    test.

    C1✓ C2✗ (M≈252) C3✓ C4✓.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64, eps: float = 1e-8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.eps = eps
        dim_inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False, dtype=torch.complex64)
        self.to_out = nn.Linear(dim_inner, dim, bias=False, dtype=torch.complex64)
        complex_glorot_init_(self.to_qkv)
        complex_glorot_init_(self.to_out)
        self._l2 = complex_l2_normalize
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(float(seq_len_for_bias))))

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rotary_emb is not None:
            q = q * rotary_emb
            k = k * rotary_emb
        q_bar = self._l2(q, dim=-1, eps=self.eps)
        k_bar = self._l2(k, dim=-1, eps=self.eps)
        sim = torch.einsum("bhid,bhjd->bhij", q_bar.conj(), k_bar).real
        sim = sim * (self.dim_head ** 0.5)
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, -1e9)
        z = sim + self.attn_bias
        attn = z + z.pow(3) / 6.0    # cubic gate; f'(z) = 1 + z²/2 ≥ 1 (C3 strict ✓), C2 violated
        out = torch.einsum("bhij,bhjd->bhid", attn.to(v.dtype), v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ComplexMultiHeadClampedReLUAttention(nn.Module):
    """Complex clamped-ReLU attention — strict C2 (M=1, matches sigmoid), C3 fully violated.

    α_ij = clamp(s_ij + b, 0, 1)            (real, bounded by [0, 1] — same M as sigmoid)
    Gradient f'(z) = 0 for z<0 or z>1, 1 for z∈(0, 1) — full C3 violation.

    Sits in the same M=1 bound as sigmoid but with full anti-phase deletion AND high-side
    saturation. The clean comparison vs complex_relu (which has M ≈ √d ≈ 11 in operating range)
    isolates the C3 violation effect from the partial C2 violation that complex_relu carries.

    C1✓ C2✓ (M=1 strict) C3✗ (full) C4✓.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, causal: bool = False,
                 seq_len_for_bias: int = 64, eps: float = 1e-8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.causal = causal
        self.eps = eps
        dim_inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * dim_inner, bias=False, dtype=torch.complex64)
        self.to_out = nn.Linear(dim_inner, dim, bias=False, dtype=torch.complex64)
        complex_glorot_init_(self.to_qkv)
        complex_glorot_init_(self.to_out)
        self._l2 = complex_l2_normalize
        self.attn_bias = nn.Parameter(torch.tensor(-math.log(float(seq_len_for_bias))))

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        if rotary_emb is not None:
            q = q * rotary_emb
            k = k * rotary_emb
        q_bar = self._l2(q, dim=-1, eps=self.eps)
        k_bar = self._l2(k, dim=-1, eps=self.eps)
        sim = torch.einsum("bhid,bhjd->bhij", q_bar.conj(), k_bar).real
        sim = sim * (self.dim_head ** 0.5)
        if self.causal:
            mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            sim = sim.masked_fill(mask, -1e9)
        attn = (sim + self.attn_bias).clamp(0.0, 1.0)   # M=1 strict ✓, C3 fully violated
        out = torch.einsum("bhij,bhjd->bhid", attn.to(v.dtype), v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# --------------------------- Cell-level Transformer ---------------------------


class TransformerCell(nn.Module):
    """One of {real, complex} × {softmax, screen} transformer cell.

    All cells follow: pre-norm + attention residual + pre-norm + FFN residual, repeated `depth` times.
    """

    def __init__(
        self,
        cell: str,
        num_tokens: int,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 16,
        ff_mult: int = 4,
        causal: bool = True,
        rotary: bool = True,
        use_softmask: bool = False,  # Phase 14 audit: default OFF (softmask=ON breaks learning)
        use_tanhnorm: bool = True,
        softmask_init_width: float = 32.0,
        s_r_init: float = -3.0,
        sigmoid_seq_len: int = 64,
        sigmoid_bias_init_override: float | None = None,
        attn_chunk_size: int | None = None,
        attn_grad_checkpoint: bool = False,
    ):
        super().__init__()
        assert cell in ("real_softmax", "real_sigmoid", "real_tanh1", "real_relu", "real_screen",
                        "complex_softmax", "complex_sigmoid", "complex_tanh1", "complex_relu", "complex_screen",
                        "complex_sigmoid_nocval", "complex_sigmoid_freereal", "complex_sigmoid_realqk",
                        "complex_softplus", "complex_cubic", "complex_clamped_relu")
        self.cell = cell
        self.is_complex = cell.startswith("complex_")
        self.depth = depth
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.causal = causal

        if self.is_complex:
            self.embed = nn.Parameter(torch.randn(num_tokens, dim, dtype=torch.complex64) * (1.0 / dim ** 0.5))
            norm_cls = ComplexRMSNorm
        else:
            self.embed = nn.Embedding(num_tokens, dim)
            nn.init.normal_(self.embed.weight, std=1.0 / dim ** 0.5)
            norm_cls = RealRMSNorm

        # Positional encoding (RoPE)
        self.rotary = rotary
        if rotary:
            if self.is_complex:
                self.rope = RotaryEmbedding(dim_head)
            else:
                self.rope = RealRotaryEmbedding(dim_head)
        else:
            self.rope = None

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
                    use_softmask=use_softmask, use_tanhnorm=use_tanhnorm, softmask_init_width=softmask_init_width, s_r_init=s_r_init,
                )
                ffn = RealFeedForward(dim, mult=ff_mult)
            elif cell == "complex_softmax":
                attn = ComplexMultiHeadSoftmaxAttention(dim, heads, dim_head, causal=causal)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_sigmoid":
                attn = ComplexMultiHeadSigmoidAttention(dim, heads, dim_head, causal=causal,
                                                          seq_len_for_bias=sigmoid_seq_len,
                                                          bias_init_override=sigmoid_bias_init_override,
                                                          chunk_size=attn_chunk_size,
                                                          gradient_checkpointing=attn_grad_checkpoint)
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
                attn = ComplexMultiHeadSoftplusAttention(dim, heads, dim_head, causal=causal,
                                                          seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_cubic":
                attn = ComplexMultiHeadCubicAttention(dim, heads, dim_head, causal=causal,
                                                       seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            elif cell == "complex_clamped_relu":
                attn = ComplexMultiHeadClampedReLUAttention(dim, heads, dim_head, causal=causal,
                                                              seq_len_for_bias=sigmoid_seq_len)
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            else:  # complex_screen
                attn = ScreeningAttention(
                    dim, heads, dim_head, causal=causal,
                    use_softmask=use_softmask, use_tanhnorm=use_tanhnorm, softmask_init_width=softmask_init_width, s_r_init=s_r_init,
                )
                ffn = ComplexFeedForward(dim, mult=ff_mult, relu_squared=True)
            self.layers.append(nn.ModuleList([attn_norm, attn, ffn_norm, ffn]))

        self.final_norm = norm_cls(dim)

        # Readout: complex → concat[Re; Im] → real_classifier; real → real_classifier
        if self.is_complex:
            self.to_logits = nn.Linear(2 * dim, num_tokens)
        else:
            self.to_logits = nn.Linear(dim, num_tokens)

    def _make_rope(self, N: int, device: torch.device):
        if self.rope is None:
            return None
        if self.is_complex:
            return self.rope(N).to(device)
        return self.rope(N, device)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [B, N] long. Returns logits [B, N, V]."""
        B, N = idx.shape
        if self.is_complex:
            x = self.embed[idx]  # [B, N, dim] complex
        else:
            x = self.embed(idx)  # [B, N, dim] real

        rope_inp = self._make_rope(N, x.device)

        for attn_norm, attn, ffn_norm, ffn in self.layers:
            normed = attn_norm(x)
            if self.cell.startswith("real_"):
                a = attn(normed, rope_cos_sin=rope_inp)
            else:
                a = attn(normed, rotary_emb=rope_inp)
            x = x + a
            x = x + ffn(ffn_norm(x))

        x = self.final_norm(x)

        if self.is_complex:
            x_real = torch.cat([x.real, x.imag], dim=-1)  # [B, N, 2*dim]
            logits = self.to_logits(x_real)
        else:
            logits = self.to_logits(x)
        return logits


def make_cell(cell: str, num_tokens: int, **kwargs) -> TransformerCell:
    return TransformerCell(cell=cell, num_tokens=num_tokens, **kwargs)
