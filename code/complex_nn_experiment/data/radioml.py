"""RadioML 2016.10a modulation classification dataset loader.

Standard benchmark for ML on radio I/Q signals (O'Shea & West 2016, 2018).
Genuine physical complex domain — antenna I/Q samples with no permission
required (free download from DeepSig).

For Phase 8 mid-scale, this fills the "physical complex domain" axis without
the NYU fastMRI registration friction.

Dataset URL: https://www.deepsig.ai/datasets
Format: HDF5 with arrays:
  - X: [N, 2, 1024]  (I and Q channels, 1024 time samples)
  - Y: [N, 11]       (one-hot, 11 modulation classes)
  - Z: [N]           (SNR in dB)

For our Phase 8 use, we adapt to the standard generate_*_batch signature
returning (inputs, targets, mask) where:
  - inputs: [B, T, 2] float — sequence of (I, Q) pairs, T=1024
  - targets: [B, 1] int — modulation class label
  - mask: all True at last position only (classification target)

Note: this is NOT a token-prediction task, so the transformer is used as
a sequence encoder + classifier head on the last position. The existing
make_cell()'s output_head is a token vocabulary head; for RadioML we'd
need a 11-class head. See _adapt_for_classification() below.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import numpy as np

# Cached dataset path (set via env var or default)
DATASET_PATH = Path(os.environ.get(
    "RADIOML_PATH",
    "/var/lib/phase8/datasets/RML2016.10a_dict.pkl"
))

# 11 modulation classes in RadioML 2016.10a
MOD_CLASSES = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]

_cached_data = None  # lazy load


def _load_dataset():
    """Load RadioML 2016.10a (pickle format from DeepSig)."""
    global _cached_data
    if _cached_data is not None:
        return _cached_data
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"RadioML dataset not found at {DATASET_PATH}. "
            f"Download from https://www.deepsig.ai/datasets (RML2016.10a) "
            f"and set RADIOML_PATH env var or place at default path."
        )
    import pickle
    with open(DATASET_PATH, "rb") as f:
        # The pickle is a dict {(modulation, snr): array_of_examples}
        data = pickle.load(f, encoding="latin1")

    # Flatten into arrays
    X_list, Y_list = [], []
    for (mod, snr), examples in data.items():
        if mod not in MOD_CLASSES:
            continue
        cls_idx = MOD_CLASSES.index(mod)
        # Filter to high SNR for cleaner classification (standard practice)
        if snr < 0:
            continue
        for ex in examples:
            X_list.append(ex)            # [2, 128] I/Q
            Y_list.append(cls_idx)
    X = np.stack(X_list)                 # [N, 2, 128]
    Y = np.array(Y_list, dtype=np.int64)  # [N]
    _cached_data = (X, Y)
    return _cached_data


def vocab_size(version: str = "2016.10a", **_) -> int:
    """Returns number of modulation classes (11)."""
    return len(MOD_CLASSES)


def sequence_length(version: str = "2016.10a", **_) -> int:
    """RadioML 2016.10a uses 128-sample sequences."""
    return 128


def generate_radioml_batch(
    batch_size: int,
    version: str = "2016.10a",
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    split: str = "train",
    train_frac: float = 0.8,
):
    """Sample a batch from RadioML.

    Returns (inputs, targets, mask):
      inputs: [B, T] long  — placeholder, see note below
      targets: [B, T] long — class label at last position, -100 elsewhere
      mask: [B, T] bool   — True only at last position

    NOTE: RadioML inputs are continuous I/Q (real-valued 2D vectors), not
    token IDs. The current `make_cell()` factory takes a `num_tokens` input
    head, so this loader returns a quantized token representation:

      - I/Q value clipped to [-1, 1]
      - Quantized to V/2 buckets per channel
      - Concatenated as i_bucket * V_per_channel + q_bucket
      - Vocab size = V_per_channel² (use 64 → 64*64 = 4096 effective)

    This is a workaround for the existing token-based architecture. A proper
    implementation would use a learnable continuous-input projection. Tracked
    as a Phase 8 follow-up.
    """
    X, Y = _load_dataset()
    N = len(X)
    n_train = int(N * train_frac)
    if split == "train":
        idx_pool = slice(0, n_train)
    elif split == "test":
        idx_pool = slice(n_train, N)
    else:
        idx_pool = slice(0, N)

    # Pick batch_size random indices
    if generator is None:
        idx = torch.randint(0, n_train if split == "train" else N - n_train, (batch_size,))
    else:
        # Use the generator (different device aware)
        idx = torch.randint(0, n_train if split == "train" else N - n_train,
                            (batch_size,), generator=generator,
                            device="cpu")
    if split == "test":
        idx = idx + n_train

    X_batch = X[idx.numpy()]              # [B, 2, 128]
    Y_batch = Y[idx.numpy()]              # [B]

    # Quantize to tokens (workaround for token-based architecture)
    V_per_channel = 64
    X_tensor = torch.from_numpy(X_batch).float().to(device)  # [B, 2, 128]
    X_clipped = torch.clamp(X_tensor, -1.0, 1.0)
    X_quantized = ((X_clipped + 1.0) / 2.0 * (V_per_channel - 1)).long()  # [B, 2, 128] in [0, V_per_channel)
    i_tok = X_quantized[:, 0, :]  # [B, 128]
    q_tok = X_quantized[:, 1, :]  # [B, 128]
    inputs = i_tok * V_per_channel + q_tok  # [B, 128]

    # Targets: class at last position only
    T = inputs.shape[1]
    targets = torch.full((batch_size, T), -100, dtype=torch.long, device=device)
    target_mask = torch.zeros((batch_size, T), dtype=torch.bool, device=device)
    targets[:, -1] = torch.from_numpy(Y_batch).to(device)
    target_mask[:, -1] = True

    return inputs, targets, target_mask
