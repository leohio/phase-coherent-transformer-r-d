"""LRA Pathfinder task — 32×32 image flattened to 1024-token binary classification.

Pathfinder is the visual long-range task in LRA (Tay et al. 2020):
  - 32×32 binary image flattened to 1024 tokens (pixel intensity)
  - Binary classification: are two highlighted endpoints connected by a path?

Smaller-scale precursor to Path-X (128×128 → 16384 tokens). Same pattern, just
1/16 the sequence length.

## Data source

LRA Pathfinder is distributed as TFRecord (Google Cloud / LRA release).
Convert once with `_convert_lra_pathfinder_tfrecord`, see docstring of
`complex_nn_experiment/data/pathx.py` for the same conversion idiom.

This file mirrors `pathx.py` exactly with `PATHFINDER_SEQ_LEN = 1024` and a
distinct env var.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

DATASET_PATH = Path(os.environ.get(
    "LRA_PATHFINDER_PATH",
    "/var/lib/phase8/datasets/lra_pathfinder.pt",
))

PATHFINDER_SEQ_LEN = 1024  # 32 × 32
PATHFINDER_VOCAB_SIZE = 256

_cached_data = None


def _load_dataset():
    global _cached_data
    if _cached_data is not None:
        return _cached_data
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"LRA Pathfinder dataset not found at {DATASET_PATH}. "
            f"Convert LRA TFRecord via `_convert_lra_pathfinder_tfrecord` (see docstring) "
            f"or set LRA_PATHFINDER_PATH to a pre-converted .pt with train/val/test splits."
        )
    _cached_data = torch.load(DATASET_PATH, map_location="cpu", weights_only=True)
    for split in ("train", "val", "test"):
        if split not in _cached_data:
            raise ValueError(f"LRA Pathfinder dataset at {DATASET_PATH} is missing split '{split}'")
    return _cached_data


def vocab_size(**_) -> int:
    return PATHFINDER_VOCAB_SIZE


def sequence_length(**_) -> int:
    return PATHFINDER_SEQ_LEN


def generate_pathfinder_batch(
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
        idx = torch.empty((batch_size,), dtype=torch.int64, device=generator.device).random_(0, N, generator=generator).cpu()

    X_batch = X[idx].long().to(device)
    Y_batch = Y[idx].long().to(device)

    T = X_batch.shape[1]
    targets = torch.full((batch_size, T), -100, dtype=torch.long, device=device)
    target_mask = torch.zeros((batch_size, T), dtype=torch.bool, device=device)
    targets[:, -1] = Y_batch
    target_mask[:, -1] = True

    return X_batch, targets, target_mask


def _convert_lra_pathfinder_tfrecord(tfrecord_dir: str, out_pt_path: str) -> None:
    """One-time conversion from LRA Pathfinder TFRecord to PyTorch .pt."""
    import tensorflow as tf

    feature_description = {
        "inputs": tf.io.FixedLenFeature([PATHFINDER_SEQ_LEN], tf.int64),
        "targets": tf.io.FixedLenFeature([1], tf.int64),
    }

    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        files = sorted(Path(tfrecord_dir).glob(f"{split}.tfrecord*"))
        if not files:
            raise FileNotFoundError(f"No {split}.tfrecord* under {tfrecord_dir}")
        ds = tf.data.TFRecordDataset([str(f) for f in files])
        ds = ds.map(_parse)
        X_list, Y_list = [], []
        for record in ds:
            x = record["inputs"].numpy().astype("uint8")
            y = int(record["targets"].numpy()[0])
            X_list.append(x)
            Y_list.append(y)
        X = torch.tensor(X_list, dtype=torch.uint8)
        Y = torch.tensor(Y_list, dtype=torch.int64)
        out[split] = (X, Y)
        print(f"  {split}: N={X.shape[0]}, X.shape={tuple(X.shape)}")

    Path(out_pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt_path)
    print(f"Saved LRA Pathfinder to {out_pt_path}")
