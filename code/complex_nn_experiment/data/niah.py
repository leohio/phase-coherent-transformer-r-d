"""Synthetic-vocab Needle-in-a-Haystack (NIAH) task for Phase 8 mid-scale.

Generalizes Copy Memory to test depth-invariance of retrieval:
  - Needle (1 token) inserted at variable depth in haystack of distractors
  - Query at end retrieves the needle position-aware via a marker token
  - vocab_size adjustable (controls discrimination difficulty)

Sequence layout (length = seq_len, includes terminal query block):
  [..haystack_tokens_with_needle_inserted_at_depth..,  delim_q,  retrieve_blank]
  ^---- random tokens with one being the unique "needle" ----^

This matches the standard NIAH evaluation paradigm (Greg Kamradt 2023, RULER 2024)
adapted for synthetic vocabulary so it works without LLM pre-training.

Used in Phase 8 to test whether complex_sigmoid maintains accuracy as seq_len
scales from 2K to 32K, at varying depths {0.1, 0.5, 0.9}.
"""
from __future__ import annotations

import torch


# Token semantics:
#   0 .. V-3: distractor / haystack tokens
#   V-2:      delim (query boundary)
#   V-1:      retrieve marker (predict here)


def vocab_size(vocab_size: int = 64, **_) -> int:
    """Returns the vocab size (passed through). Matches generate_*_batch signature."""
    return max(vocab_size, 4)


def sequence_length(seq_len: int = 8192, **_) -> int:
    """Returns the requested seq_len."""
    return seq_len


def generate_niah_batch(
    batch_size: int,
    seq_len: int = 8192,
    depth_ratio: float = 0.5,
    vocab_size: int = 64,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a NIAH batch.

    The "needle" is a unique token at depth_ratio × seq_len position in the haystack.
    The retrieve_marker at the end signals the model to predict the needle's value.

    Returns:
      inputs:  [B, T]  full sequence
      targets: [B, T]  −100 except at the predict position(s)
      mask:    [B, T]  True where targets are valid

    For NIAH-Single, we predict one token: the needle value, at the last position.
    """
    V = max(vocab_size, 4)
    delim_token = V - 2
    retrieve_token = V - 1
    haystack_vocab = V - 2  # tokens 0..V-3

    T = seq_len
    haystack_len = T - 2  # haystack + delim + retrieve

    # Uniform random haystack
    inputs = torch.randint(0, haystack_vocab, (batch_size, T), generator=generator, device=device)
    inputs[:, -2] = delim_token
    inputs[:, -1] = retrieve_token

    # Needle position
    needle_pos = max(0, min(haystack_len - 1, int(depth_ratio * haystack_len)))

    # Pick a random "needle value" per sample (uniform in vocab)
    needle_values = torch.randint(0, haystack_vocab, (batch_size,), generator=generator, device=device)
    inputs[:, needle_pos] = needle_values

    # Targets: predict the needle value at the last (retrieve_token) position
    targets = torch.full_like(inputs, -100)
    target_mask = torch.zeros_like(inputs, dtype=torch.bool)
    targets[:, -1] = needle_values
    target_mask[:, -1] = True

    return inputs, targets, target_mask
