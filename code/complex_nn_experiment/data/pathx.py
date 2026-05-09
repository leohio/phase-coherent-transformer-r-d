"""Path-X (Long Range Arena) data loader.

Path-X is the longest-sequence task in LRA (Tay et al. 2020):
  - 128×128 binary image flattened to 16384-token sequence
  - Binary classification: are the two highlighted endpoints connected by a path?

Standard LRA evaluation:
  - Vanilla Transformer: ~50% (random) — task is famously not solved by attention
  - S4 (Gu 2021): 88.10%
  - S5 (Smith 2022): 98.58%
  - Mega (Ma 2022): 97.98%
  - Mamba (Gu & Dao 2023): 87.39%

Goal in this project: complex_sigmoid solo run, compare to literature baselines.
See `docs/total_bench_(till_phase_14).md` §11 for the policy.

## Data source

Path-X is distributed as part of the LRA benchmark, originally on Google Cloud
Storage. The format is TFRecord; this loader expects a pre-converted PyTorch
tensor file at `PATHX_PATH` (env var) or `/var/lib/phase8/datasets/pathx.pt`.

Conversion (one-time, on a machine with TF installed):
    pip install tensorflow tensorflow-datasets
    python -c "from complex_nn_experiment.data.pathx import _convert_lra_pathx_tfrecord; \
               _convert_lra_pathx_tfrecord('lra_release/pathx', '/var/lib/phase8/datasets/pathx.pt')"

Alternative: PyTorch-format mirrors of LRA on HuggingFace
(e.g., `tau/lra_pathfinder` family) can be adapted by overriding `_load_dataset`.

## Layout

  inputs:  [B, 16384] long  — pixel intensity tokens 0..255
  targets: [B, 16384] long  — −100 except at last position (class label 0/1)
  mask:    [B, 16384] bool — True only at last position

This matches the existing token-prediction signature so the existing
`make_cell` factory + classification training loop work unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# Cached dataset path
DATASET_PATH = Path(os.environ.get(
    "PATHX_PATH",
    "/var/lib/phase8/datasets/pathx.pt",
))

PATHX_SEQ_LEN = 16384  # 128 × 128
PATHX_VOCAB_SIZE = 256  # pixel intensity 0..255 (binary in original Path-X but kept as 256 for flexibility)

_cached_data = None  # lazy load: dict with train/val/test splits


def _load_dataset():
    """Load Path-X dataset from local PyTorch file.

    Expected format (from `_convert_lra_pathx_tfrecord`):
        {"train": (X_train, Y_train), "val": (X_val, Y_val), "test": (X_test, Y_test)}
        X: [N, 16384] uint8 (pixel intensities)
        Y: [N] int64    (binary class label 0 or 1)
    """
    global _cached_data
    if _cached_data is not None:
        return _cached_data
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Path-X dataset not found at {DATASET_PATH}. "
            f"See docstring of `complex_nn_experiment.data.pathx` for conversion instructions, "
            f"or set PATHX_PATH env var to a pre-converted .pt file with the expected layout."
        )
    _cached_data = torch.load(DATASET_PATH, map_location="cpu", weights_only=True)
    for split in ("train", "val", "test"):
        if split not in _cached_data:
            raise ValueError(f"Path-X dataset at {DATASET_PATH} is missing split '{split}'")
    return _cached_data


def vocab_size(**_) -> int:
    """Path-X vocab is pixel intensity 0..255 (treated as discrete tokens)."""
    return PATHX_VOCAB_SIZE


def sequence_length(**_) -> int:
    """Path-X sequence length is 128 * 128 = 16384."""
    return PATHX_SEQ_LEN


def generate_pathx_batch(
    batch_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    split: str = "train",
    **_,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch from Path-X.

    Returns (inputs, targets, mask):
      inputs: [B, 16384] long  — pixel tokens
      targets: [B, 16384] long — binary class label (0 or 1) at last position, -100 elsewhere
      mask: [B, 16384] bool   — True only at last position
    """
    data = _load_dataset()
    if split not in data:
        raise ValueError(f"Unknown split '{split}'. Available: {list(data.keys())}")
    X, Y = data[split]
    N = X.shape[0]

    if generator is None:
        idx = torch.randint(0, N, (batch_size,))
    else:
        # torch.randint() requires a CPU generator; use Tensor.random_() which
        # accepts a same-device generator, so this works on both CPU and CUDA.
        idx = (
            torch.empty((batch_size,), dtype=torch.int64, device=generator.device)
            .random_(0, N, generator=generator)
            .cpu()
        )

    X_batch = X[idx].long().to(device)  # [B, 16384]
    Y_batch = Y[idx].long().to(device)  # [B]

    T = X_batch.shape[1]
    targets = torch.full((batch_size, T), -100, dtype=torch.long, device=device)
    target_mask = torch.zeros((batch_size, T), dtype=torch.bool, device=device)
    targets[:, -1] = Y_batch
    target_mask[:, -1] = True

    return X_batch, targets, target_mask


def _convert_lra_pathx_tfrecord(tfrecord_dir: str, out_pt_path: str) -> None:
    """One-time conversion from LRA TFRecord to PyTorch .pt.

    Run once on a machine with TF installed. Output `.pt` is portable.
    Expected `tfrecord_dir` layout (from LRA release):
        tfrecord_dir/
          pathfinder_x/
            train.tfrecord*
            val.tfrecord*
            test.tfrecord*
    """
    import tensorflow as tf

    feature_description = {
        "inputs": tf.io.FixedLenFeature([PATHX_SEQ_LEN], tf.int64),
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
            x = record["inputs"].numpy().astype("uint8")  # [16384]
            y = int(record["targets"].numpy()[0])
            X_list.append(x)
            Y_list.append(y)
        X = torch.tensor(X_list, dtype=torch.uint8)
        Y = torch.tensor(Y_list, dtype=torch.int64)
        out[split] = (X, Y)
        print(f"  {split}: N={X.shape[0]}, X.shape={tuple(X.shape)}, Y.shape={tuple(Y.shape)}")

    Path(out_pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt_path)
    print(f"Saved Path-X to {out_pt_path}")
