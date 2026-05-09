"""Real RadioML loader (RML2016, 6dB SNR public mirror).

Reads pre-downloaded parquet files from `datasets/radioml/`. Each row has:
  - signal: [2, 128] float32 (I and Q channels, 128 samples)
  - label_id: int (0..10, 11 modulation classes)

Returns (x, y) compatible with TaskTransformer (output_mode='pooled_classify'):
  x: [B, T] complex64 (I + jQ)
  y: [B] long

Difficulty axes:
- `seq_len`: 128 default for L1 / 64 for L2 (truncate first half).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


DATASET_DIR = Path(os.environ.get(
    "RADIOML_REAL_DIR",
    "/Users/plasma/repos/complexscreening-00/.claude/worktrees/bridge-cse_011GzoPM6wXvkzzb2wjM5GL7/datasets/radioml",
))

NUM_CLASSES = 11

_train_cache: tuple[np.ndarray, np.ndarray] | None = None
_test_cache: tuple[np.ndarray, np.ndarray] | None = None


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (signals[N,2,128], labels[N]) for split in {train, test}."""
    global _train_cache, _test_cache
    if split == "train" and _train_cache is not None:
        return _train_cache
    if split == "test" and _test_cache is not None:
        return _test_cache
    fname = f"radioml_6db_{'train' if split == 'train' else 'test'}.parquet"
    path = DATASET_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_parquet(path)
    sigs = np.stack([np.stack(list(row), axis=0).astype(np.float32) for row in df["signal"]])  # [N, 2, 128]
    # The hitrs909/RML2016 mirror normalizes I/Q to roughly [0, 1] — bias the
    # complex inputs to one quadrant. Center to zero-mean and scale to unit
    # magnitude per sample so complex cells see a proper rotating-phase signal.
    sigs = sigs - sigs.mean(axis=-1, keepdims=True)        # zero-mean per channel per sample
    mag = np.sqrt((sigs ** 2).sum(axis=1, keepdims=True))  # [N, 1, T]
    rms = np.sqrt((mag ** 2).mean(axis=-1, keepdims=True)) # [N, 1, 1]
    sigs = sigs / np.clip(rms, 1e-8, None)                 # unit RMS power per sample
    labels = df["label_id"].to_numpy().astype(np.int64)  # [N]
    out = (sigs, labels)
    if split == "train":
        _train_cache = out
    else:
        _test_cache = out
    return out


def radioml_real_seq_len(seq_len: int = 128, **_) -> int:
    return seq_len


def radioml_real_num_classes() -> int:
    return NUM_CLASSES


def gen_radioml_real_batch(
    batch_size: int,
    seq_len: int = 128,
    split: str = "train",
    device: str | torch.device = "cpu",
    generator: Optional[torch.Generator] = None,
):
    """Sample a batch from the real RML2016 6dB subset.

    Returns:
      x: [B, seq_len] complex64
      y: [B] long
    """
    sigs, labels = _load_split(split)
    if seq_len > sigs.shape[-1]:
        raise ValueError(f"seq_len {seq_len} > native 128")
    N = sigs.shape[0]

    if generator is None:
        idx = torch.randint(0, N, (batch_size,))
    else:
        idx = torch.randint(0, N, (batch_size,), generator=generator, device="cpu")

    sel = sigs[idx.numpy()]  # [B, 2, 128]
    if seq_len < sigs.shape[-1]:
        sel = sel[..., :seq_len]
    # Build complex from I, Q
    x_complex = (sel[:, 0, :] + 1j * sel[:, 1, :]).astype(np.complex64)  # [B, seq_len]
    y = labels[idx.numpy()]  # [B]

    return torch.from_numpy(x_complex).to(device), torch.from_numpy(y).to(device)
