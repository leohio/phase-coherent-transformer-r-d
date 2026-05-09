"""LRA Text classification task — IMDB byte-level binary classification.

LRA Text (Tay et al. 2020):
  - IMDB movie reviews, byte-level tokenization
  - Truncate / pad to 4096 bytes
  - Binary sentiment classification

Tokens: vocab size 257 (256 byte values + 1 PAD).

## Data source

Two paths to obtain data:

1. **HuggingFace `imdb`** dataset (recommended, no LRA-specific TFRecord needed):
       pip install datasets
       python -c "from complex_nn_experiment.data.lra_text import _build_lra_text_pt; \
                  _build_lra_text_pt('/var/lib/phase8/datasets/lra_text.pt')"

2. LRA TFRecord (matches LRA paper exactly): use `_convert_lra_text_tfrecord` similar
   to `pathx.py`'s helper.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

DATASET_PATH = Path(os.environ.get(
    "LRA_TEXT_PATH",
    "/var/lib/phase8/datasets/lra_text.pt",
))

TEXT_SEQ_LEN = 4096
TEXT_VOCAB_SIZE = 257  # 256 byte values + PAD
PAD_TOKEN = 256

_cached_data = None


def _load_dataset():
    global _cached_data
    if _cached_data is not None:
        return _cached_data
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"LRA Text dataset not found at {DATASET_PATH}. "
            f"Build via `_build_lra_text_pt` (HuggingFace `imdb`) or "
            f"convert LRA TFRecord; see module docstring."
        )
    _cached_data = torch.load(DATASET_PATH, map_location="cpu", weights_only=True)
    for split in ("train", "val", "test"):
        if split not in _cached_data:
            raise ValueError(f"LRA Text dataset at {DATASET_PATH} is missing split '{split}'")
    return _cached_data


def vocab_size(**_) -> int:
    return TEXT_VOCAB_SIZE


def sequence_length(**_) -> int:
    return TEXT_SEQ_LEN


def generate_text_batch(
    batch_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    split: str = "train",
    **_,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = _load_dataset()
    if split not in data:
        raise ValueError(f"Unknown split '{split}'. Available: {list(data.keys())}")
    X, Y = data[split]
    N = X.shape[0]

    if generator is None:
        idx = torch.randint(0, N, (batch_size,))
    else:
        # torch.randint() requires generator and target device to match. Use Tensor.random_()
        # which accepts a same-device generator, then move to CPU for indexing.
        idx = torch.empty((batch_size,), dtype=torch.int64, device=generator.device).random_(0, N, generator=generator).cpu()

    X_batch = X[idx].long().to(device)
    Y_batch = Y[idx].long().to(device)

    T = X_batch.shape[1]
    targets = torch.full((batch_size, T), -100, dtype=torch.long, device=device)
    target_mask = torch.zeros((batch_size, T), dtype=torch.bool, device=device)
    targets[:, -1] = Y_batch
    target_mask[:, -1] = True

    return X_batch, targets, target_mask


def _build_lra_text_pt(out_pt_path: str, val_frac: float = 0.1, max_examples_per_split: int | None = None) -> None:
    """Build LRA-Text-style .pt from HuggingFace `imdb` dataset.

    LRA's official Text task is IMDB byte-level truncated to 4096. HuggingFace
    `imdb` provides train (25K) and test (25K) splits; we carve a val split
    from train.
    """
    from datasets import load_dataset
    ds = load_dataset("imdb")
    train_full = ds["train"]
    test = ds["test"]

    n_train = len(train_full)
    n_val = int(n_train * val_frac)
    train_idx = list(range(n_train - n_val))
    val_idx = list(range(n_train - n_val, n_train))
    if max_examples_per_split is not None:
        train_idx = train_idx[:max_examples_per_split]
        val_idx = val_idx[:max_examples_per_split]

    def _to_tensors(examples):
        X = torch.full((len(examples), TEXT_SEQ_LEN), PAD_TOKEN, dtype=torch.long)
        Y = torch.zeros(len(examples), dtype=torch.long)
        for i, ex in enumerate(examples):
            text_bytes = ex["text"].encode("utf-8", errors="replace")[:TEXT_SEQ_LEN]
            X[i, :len(text_bytes)] = torch.tensor(list(text_bytes), dtype=torch.long)
            Y[i] = int(ex["label"])
        return X, Y

    out = {
        "train": _to_tensors([train_full[i] for i in train_idx]),
        "val":   _to_tensors([train_full[i] for i in val_idx]),
        "test":  _to_tensors(list(test if max_examples_per_split is None else test.select(range(max_examples_per_split)))),
    }
    Path(out_pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt_path)
    for split, (X, Y) in out.items():
        print(f"  {split}: N={X.shape[0]}, X.shape={tuple(X.shape)}")
    print(f"Saved LRA Text (IMDB) to {out_pt_path}")
