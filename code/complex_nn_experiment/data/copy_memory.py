"""Copy Memory task (Arjovsky et al. 2016).

Sequence layout (length = K + delay + K + 1):
  [c_1, c_2, ..., c_K,  blank, blank, ..., blank,  delim,  blank, ..., blank]
  ^---- K random tokens   ^---- delay-1 blanks ----^   ^---- K blanks to predict ----^

Target: at the K positions after delim, output [c_1, ..., c_K]; elsewhere any (loss masked).

Vocabulary:
  tokens 0..K-1: the random source tokens
  token K:       blank
  token K+1:     delim

This is a retrieval / long-range memory task. Used by uRNN-family complex NNs at
delays 100-1000. We extend to delays 500-10000 for length extrapolation studies.
"""

from __future__ import annotations

import torch


def generate_copy_memory_batch(
    batch_size: int,
    K: int = 10,
    delay: int = 100,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a Copy Memory batch.

    Returns (input_seq, target_seq, target_mask), each [B, T] with T = 2*K + delay + 1.
    target_seq is the full sequence with -100 ignore_index outside the predict positions.
    target_mask is True at predict positions only.
    """
    blank = K
    delim = K + 1
    T = 2 * K + delay + 1  # K source + (delay - 1) blanks + 1 delim + K predict + 1 buffer? -- below

    # Layout: K source, (delay) blanks, 1 delim, K predict-blanks  →  total = 2K + delay + 1
    src = torch.randint(0, K, (batch_size, K), generator=generator, device=device)
    blanks_pre = torch.full((batch_size, delay), blank, dtype=torch.long, device=device)
    delim_col = torch.full((batch_size, 1), delim, dtype=torch.long, device=device)
    predict_blanks = torch.full((batch_size, K), blank, dtype=torch.long, device=device)
    inputs = torch.cat([src, blanks_pre, delim_col, predict_blanks], dim=1)  # [B, 2K+delay+1]

    targets = torch.full_like(inputs, -100)
    target_mask = torch.zeros_like(inputs, dtype=torch.bool)
    targets[:, K + delay + 1:] = src
    target_mask[:, K + delay + 1:] = True
    return inputs, targets, target_mask


def vocab_size(K: int = 10, **_) -> int:
    return K + 2  # K source tokens + blank + delim


def sequence_length(K: int = 10, delay: int = 100, **_) -> int:
    return 2 * K + delay + 1
