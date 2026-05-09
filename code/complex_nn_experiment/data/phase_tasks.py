"""Synthetic complex-favorable tasks for Phase 7.

Three tasks:
  - phase_sum: predict (Σ θ_t) mod K from N tokens with discrete phases
  - phase_memory: Copy Memory with phase keys (e^{iθ}) instead of integer ids
  - multi_pitch: synthetic STFT-of-multi-pitch-audio classification (MusicNet substitute)

All tasks are designed so that:
  - Real cell sees (Re, Im) as 2-channel real per token (information-equal)
  - Complex cell sees complex64 token directly
  - Difference is architectural inductive bias only
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ============================================================
# Task 1: Phase Sum Mod K
# ============================================================

def gen_phase_sum_batch(
    batch_size: int,
    N: int = 20,
    K: int = 8,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase Sum Mod K task.

    N tokens, each with phase θ_t ∈ {0, 2π/K, ..., (K-1)·2π/K}.
    Target: bucket index of (Σ θ_t) mod 2π in [0, K).

    The complex model can compute Σ via complex multiplication on the unit circle
    (each token is a unit vector); the real model has to handle (cos, sin) tuple
    addition manually.

    Returns:
      x: [B, N] complex64 (unit vectors)
      y: [B] long (target class in [0, K))
    """
    if generator is None:
        idx = torch.randint(0, K, (batch_size, N), device=device)
    else:
        idx = torch.randint(0, K, (batch_size, N), generator=generator, device=device)
    angles = idx.float() * (2.0 * math.pi / K)
    x = torch.complex(angles.cos(), angles.sin())
    y = idx.sum(dim=-1) % K
    return x, y


def phase_sum_seq_len(N: int = 20) -> int:
    return N


def phase_sum_num_classes(K: int = 8) -> int:
    return K


# ============================================================
# Task 2: Phase Memory (Copy Memory variant with phase keys)
# ============================================================

def gen_phase_memory_batch(
    batch_size: int,
    K: int = 5,
    delay: int = 30,
    M: int = 8,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase Copy Memory.

    Sequence of K source tokens with random phases θ_k ∈ {0, ..., (M-1)·2π/M},
    then `delay` zero tokens, then `K` zero tokens at predict positions.
    Target: at predict position i, output phase bucket of source[i] in [0, M).

    Real model sees each token as (cos, sin) 2-channel; complex model sees e^{iθ}.

    Returns:
      x: [B, T] complex64 (T = 2K + delay)
      y: [B, T] long (-100 outside predict positions, phase index at predict positions)
      mask: [B, T] bool (True at predict positions)
    """
    T = 2 * K + delay
    if generator is None:
        src_idx = torch.randint(0, M, (batch_size, K), device=device)
    else:
        src_idx = torch.randint(0, M, (batch_size, K), generator=generator, device=device)
    src_angles = src_idx.float() * (2.0 * math.pi / M)
    src_complex = torch.complex(src_angles.cos(), src_angles.sin())

    x = torch.zeros((batch_size, T), dtype=torch.complex64, device=device)
    x[:, :K] = src_complex
    # delay region (already zero)
    # predict region (already zero)

    y = torch.full((batch_size, T), -100, dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, T), dtype=torch.bool, device=device)
    y[:, K + delay:] = src_idx
    mask[:, K + delay:] = True

    return x, y, mask


def phase_memory_seq_len(K: int = 5, delay: int = 30) -> int:
    return 2 * K + delay


def phase_memory_num_classes(M: int = 8) -> int:
    return M


# ============================================================
# Task 3: Multi-Pitch STFT (synthetic MusicNet substitute)
# ============================================================

def gen_multi_pitch_batch(
    batch_size: int,
    n_pitches: int = 16,
    n_active: int = 3,
    n_samples: int = 128,
    sample_rate: int = 2000,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic multi-pitch detection from 1D FFT (MusicNet substitute, simplified).

    Audio: y(t) = Σ_k cos(2π f_k t + φ_k) for K random pitches.
    1D FFT (no STFT) → complex spectrum of n_samples//2 + 1 bins.
    Multi-label classification: for each of n_pitches in the bank, is it active?

    Real cell: (Re, Im) 2-channel input.
    Complex cell: complex spectrum directly.
    Magnitude alone is sufficient; phase is random per-sample. Complex model may
    benefit from architectural phase-invariance prior.

    Returns:
      x: [B, n_samples//2 + 1] complex64 (one-sided FFT)
      y: [B, n_pitches] float (0/1 multi-label)
    """
    pitches = torch.tensor(
        [100.0 * (8.0 ** (i / n_pitches)) for i in range(n_pitches)],  # log-spaced 100..800 Hz
        device=device,
    )

    active = torch.zeros((batch_size, n_pitches), device=device)
    for b in range(batch_size):
        chosen = torch.randperm(n_pitches, generator=generator, device=device)[:n_active]
        active[b, chosen] = 1.0

    if generator is None:
        phases = torch.rand((batch_size, n_pitches), device=device) * 2.0 * math.pi
    else:
        phases = torch.rand((batch_size, n_pitches), generator=generator, device=device) * 2.0 * math.pi

    t = torch.arange(n_samples, device=device, dtype=torch.float32) / sample_rate
    waves = torch.cos(2 * math.pi * pitches.view(1, -1, 1) * t.view(1, 1, -1)
                      + phases.view(batch_size, n_pitches, 1))
    audio = (waves * active.view(batch_size, n_pitches, 1)).sum(dim=1)
    audio = audio / max(1.0, n_active)

    # 1D rFFT: complex spectrum of n_samples//2 + 1 bins
    spec = torch.fft.rfft(audio)  # [B, n_samples//2 + 1]
    return spec, active


def multi_pitch_seq_len(n_samples: int = 128) -> int:
    return n_samples // 2 + 1


def multi_pitch_num_pitches(n_pitches: int = 16) -> int:
    return n_pitches
