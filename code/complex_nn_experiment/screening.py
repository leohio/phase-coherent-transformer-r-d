"""Paper-faithful screening primitive (Nakanishi 2026, "Screening Is Enough").

Implements the gate
    s_ij  = q̄_i · k̄_j         (q̄, k̄ L2-normalized)         (real, ∈ [-1, 1])
    α_ij  = r² · max(s_ij - t, 0)²,   r = exp(s_r) + 1,   t = 1 - 1/r
    α^d_ij = α_ij · m_ij(w)                                  (cosine softmask, optional)
    h_i   = Σ_j α^d_ij · v_j                                 (NO row-norm)
    u_i   = tanh(‖h‖) / ‖h‖ · h                              (post-aggregation TanhNorm)
    out_i = c_linear_out(u_i ⊙ g_act_i)                      (complex Hadamard, then complex linear)

Per `docs/complex_nn_screening_plan.md` §3.3, §3.4. Plan §4.6 fairness rule:
softmax keeps row-norm only, screening keeps TanhNorm only.

This module provides:
  - complex_l2_normalize, complex_tanh_norm  (helpers)
  - complex_glorot_init_                      (per-component std = 1/sqrt(fan_in+fan_out))
  - cosine_softmask                            (real, learnable width w)
  - ScreeningAttention                         (paper-faithful complex attention drop-in)
  - ScreeningAttentionReal                     (real-tensor twin for the real_screen cell)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def complex_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize a complex tensor: x / sqrt(Σ |x_d|²)."""
    norm = torch.sqrt((x.conj() * x).real.sum(dim=dim, keepdim=True) + eps)
    return x / norm


def complex_tanh_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """TanhNorm on complex magnitude: tanh(‖x‖)/‖x‖ · x.

    Bounds the magnitude into (0, 1) per vector while preserving phase.
    Natural complex extension of the paper's real TanhNorm.
    """
    norm = torch.sqrt((x.conj() * x).real.sum(dim=dim, keepdim=True) + eps)
    return torch.tanh(norm) / norm * x


def complex_glorot_init_(linear: nn.Linear) -> None:
    """Complex Glorot init: per-component std = 1/sqrt(fan_in + fan_out).

    Avoids PyTorch default Kaiming-on-real-storage which gives ~2x intended variance.
    """
    fan_in, fan_out = linear.in_features, linear.out_features
    sigma = math.sqrt(1.0 / (fan_in + fan_out))
    with torch.no_grad():
        linear.weight.real.normal_(0.0, sigma)
        linear.weight.imag.normal_(0.0, sigma)
        if linear.bias is not None:
            linear.bias.zero_()


class CosineSoftmask(nn.Module):
    """Real-valued positional softmask m_ij(w) = cos((i-j)/w)·1[|i-j| ≤ π·w].

    The paper's distance-aware acceptance window. Width w is learnable (positive via softplus).
    Half-window is π·w, so positions further than that get zero weight.
    """

    def __init__(self, init_width: float = 32.0):
        super().__init__()
        # softplus(s) = log(1+exp(s)), inverse for init: s = log(exp(w)-1)
        self.raw_w = nn.Parameter(torch.tensor(math.log(math.exp(init_width) - 1.0)))

    def width(self) -> torch.Tensor:
        return F.softplus(self.raw_w) + 1e-2

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(q_len, device=device).float().unsqueeze(1)  # [q_len, 1]
        j = torch.arange(k_len, device=device).float().unsqueeze(0)  # [1, k_len]
        d = (i - j).abs()
        w = self.width()
        half = math.pi * w
        cos_window = 0.5 * (torch.cos(d / w) + 1.0)  # rescale [-1,1] → [0,1]
        m = torch.where(d <= half, cos_window, torch.zeros_like(cos_window))
        return m  # [q_len, k_len]


class ModReLU(nn.Module):
    """modReLU(z) = ReLU(|z| + b) · z/|z|, b learnable. Restores killing capability of GLU."""

    def __init__(self, init_bias: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt((z.conj() * z).real + 1e-8)
        scaled_mag = F.relu(mag + self.bias)
        return scaled_mag * z / mag


class ScreeningAttention(nn.Module):
    """Paper-faithful complex screening attention block.

    Replaces the (softmax + scaled dot-product) attention with the screening gate.
    Layout matches lucidrains' ComplexMultiheadAttention so it can be a drop-in for
    the 4-cell substrate.

    Architecture (per plan §3.4):
      x → c_linear(q, k, v, g) → split heads → L2-norm q,k → score → trim-and-square
      → softmask (optional) → aggregate v (no row-norm) → TanhNorm
      → modReLU(g) → complex Hadamard u·g → c_linear_out
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        causal: bool = False,
        use_softmask: bool = False,  # Phase 14 audit: default OFF (softmask=ON breaks learning, ~10% chance)
        use_tanhnorm: bool = True,
        softmask_init_width: float = 32.0,
        s_r_init: float = -3.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.causal = causal
        self.use_softmask = use_softmask
        self.use_tanhnorm = use_tanhnorm

        dim_inner = heads * dim_head

        # 4-way input projection: q, k, v, g
        self.to_qkvg = nn.Linear(dim, 4 * dim_inner, bias=True, dtype=torch.complex64)
        complex_glorot_init_(self.to_qkvg)

        self.to_out = nn.Linear(dim_inner, dim, bias=True, dtype=torch.complex64)
        complex_glorot_init_(self.to_out)

        # Per-head learnable acceptance width: r = exp(s_r) + 1, t = 1 - 1/r
        # s_r_init = -3 → r ≈ 1.05, t ≈ 0.05 (default). For tau sweep at t = T set
        # s_r_init = log(T / (1-T)). t=0 → s_r = -∞ (use -10), t=0.1 → -2.20, etc.
        self.s_r = nn.Parameter(torch.full((heads,), float(s_r_init)))

        # gate side activation
        self.modrelu = ModReLU(init_bias=0.0)

        if use_softmask:
            self.softmask = CosineSoftmask(init_width=softmask_init_width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, rotary_emb: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, N, dim] complex.  rotary_emb: [N, d_head] complex (cos+i*sin) or None.
        Returns [B, N, dim] complex.
        """
        B, N, _ = x.shape
        qkvg = self.to_qkvg(x)  # [B, N, 4*dim_inner]
        q, k, v, g = qkvg.chunk(4, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        g = rearrange(g, "b n (h d) -> b h n d", h=self.heads)

        # Apply RoPE (complex form: q_i ← q_i · e^{iθ_i})
        if rotary_emb is not None:
            q = q * rotary_emb
            k = k * rotary_emb

        # L2 normalize q, k → score in [-1, 1]
        q_bar = complex_l2_normalize(q, dim=-1, eps=self.eps)
        k_bar = complex_l2_normalize(k, dim=-1, eps=self.eps)
        # score[b, h, i, j] = Re(<q_bar_i, k_bar_j>) = Re(Σ_d q_bar_d.conj() · k_bar_d)
        s = torch.einsum("bhid,bhjd->bhij", q_bar.conj(), k_bar).real

        # Per-head r, t = 1 - 1/r
        r = torch.exp(self.s_r) + 1.0  # [heads]
        t = 1.0 - 1.0 / r  # [heads]
        r2 = (r * r).view(1, self.heads, 1, 1)
        t_bcast = t.view(1, self.heads, 1, 1)

        alpha = r2 * F.relu(s - t_bcast).pow(2)  # [B, h, N, N]

        if self.use_softmask:
            m = self.softmask(N, N, x.device)  # [N, N], real
            alpha = alpha * m.unsqueeze(0).unsqueeze(0)

        if self.causal:
            causal_mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            alpha = alpha.masked_fill(causal_mask, 0.0)

        if mask is not None:
            alpha = alpha.masked_fill(~mask, 0.0)

        # Aggregate (NO row-norm)
        h = torch.einsum("bhij,bhjd->bhid", alpha.to(v.dtype), v)  # [B, h, N, d_head] complex

        # TanhNorm on magnitude (post-aggregation magnitude regulator). Optional for ablation.
        u = complex_tanh_norm(h, dim=-1, eps=self.eps) if self.use_tanhnorm else h

        # Gate side: modReLU
        g_act = self.modrelu(g)

        # Complex Hadamard
        out = u * g_act  # [B, h, N, d_head]

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ScreeningAttentionReal(nn.Module):
    """Real-tensor twin of ScreeningAttention for the real_screen cell.

    Same gate semantics (paper-faithful screening), but everything is real:
      score = q̄_i · k̄_j (real cosine similarity)
      v aggregation: real
      TanhNorm: tanh(‖h‖)/‖h‖ · h on real magnitude
      gate side: tanh(SiLU(g))  ← paper's natural real choice
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        causal: bool = False,
        use_softmask: bool = False,  # Phase 14 audit: default OFF (softmask=ON breaks learning, ~10% chance)
        use_tanhnorm: bool = True,
        softmask_init_width: float = 32.0,
        s_r_init: float = -3.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.causal = causal
        self.use_softmask = use_softmask
        self.use_tanhnorm = use_tanhnorm

        dim_inner = heads * dim_head
        self.to_qkvg = nn.Linear(dim, 4 * dim_inner, bias=True)
        nn.init.xavier_normal_(self.to_qkvg.weight)
        nn.init.zeros_(self.to_qkvg.bias)

        self.to_out = nn.Linear(dim_inner, dim, bias=True)
        nn.init.xavier_normal_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # See ScreeningAttention for the rationale of s_r_init.
        self.s_r = nn.Parameter(torch.full((heads,), float(s_r_init)))
        if use_softmask:
            self.softmask = CosineSoftmask(init_width=softmask_init_width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, rope_cos_sin=None) -> torch.Tensor:
        B, N, _ = x.shape
        qkvg = self.to_qkvg(x).chunk(4, dim=-1)
        q, k, v, g = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkvg)

        # Apply RoPE if provided (real form: rotate (a, b) pairs by angle θ)
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin  # each [N, D/2]
            def _apply(t):
                a, b = t[..., 0::2], t[..., 1::2]
                cos_ = cos.view(*([1] * (t.ndim - 2)), t.shape[-2], -1)
                sin_ = sin.view(*([1] * (t.ndim - 2)), t.shape[-2], -1)
                rot_a = a * cos_ - b * sin_
                rot_b = a * sin_ + b * cos_
                return torch.stack([rot_a, rot_b], dim=-1).flatten(start_dim=-2)
            q = _apply(q)
            k = _apply(k)

        q_bar = F.normalize(q, dim=-1, eps=self.eps)
        k_bar = F.normalize(k, dim=-1, eps=self.eps)
        s = torch.einsum("bhid,bhjd->bhij", q_bar, k_bar)

        r = torch.exp(self.s_r) + 1.0
        t = 1.0 - 1.0 / r
        alpha = (r * r).view(1, self.heads, 1, 1) * F.relu(s - t.view(1, self.heads, 1, 1)).pow(2)

        if self.use_softmask:
            m = self.softmask(N, N, x.device)
            alpha = alpha * m.unsqueeze(0).unsqueeze(0)
        if self.causal:
            causal_mask = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            alpha = alpha.masked_fill(causal_mask, 0.0)
        if mask is not None:
            alpha = alpha.masked_fill(~mask, 0.0)

        h = torch.einsum("bhij,bhjd->bhid", alpha, v)
        # Real TanhNorm (optional)
        if self.use_tanhnorm:
            h_norm = h.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            u = torch.tanh(h_norm) / h_norm * h
        else:
            u = h

        # Real gate side: tanh(SiLU(g)) — paper's natural choice
        g_act = torch.tanh(F.silu(g))

        out = u * g_act
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
