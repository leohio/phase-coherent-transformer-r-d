"""LRA Image classification task — CIFAR-10 grayscale pixel sequence.

LRA Image (Tay et al. 2020):
  - CIFAR-10 (32×32×3) → grayscale 32×32 → flatten to 1024-token sequence
  - Each pixel is a token from vocab {0..255}
  - 10-class classification

Tokens: vocab size 256 (pixel intensities 0..255).

## Data source

torchvision CIFAR-10 (auto-download, ~170MB):
    pip install torchvision
    python -c "from data.lra_image import _build_lra_image_pt; \
               _build_lra_image_pt('/var/lib/phase8/datasets/lra_image.pt')"
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

DATASET_PATH = Path(os.environ.get(
    "LRA_IMAGE_PATH",
    "/var/lib/phase8/datasets/lra_image.pt",
))

IMAGE_SEQ_LEN = 1024  # 32 * 32 grayscale
IMAGE_VOCAB_SIZE = 256  # pixel intensities 0..255
NUM_CLASSES = 10

_cached_data = None


def _load_dataset():
    global _cached_data
    if _cached_data is not None:
        return _cached_data
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"LRA Image dataset not found at {DATASET_PATH}. "
            f"Build via `_build_lra_image_pt` (torchvision CIFAR-10); see module docstring."
        )
    _cached_data = torch.load(DATASET_PATH, map_location="cpu", weights_only=True)
    for split in ("train", "val", "test"):
        if split not in _cached_data:
            raise ValueError(f"LRA Image dataset at {DATASET_PATH} is missing split '{split}'")
    return _cached_data


def vocab_size(**_) -> int:
    return IMAGE_VOCAB_SIZE


def sequence_length(**_) -> int:
    return IMAGE_SEQ_LEN


def generate_image_batch(
    batch_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    split: str = "train",
    **_,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same protocol as lra_text/lra_pathfinder: ([B,N] long tokens, [B,N] targets w/ -100, [B,N] mask)."""
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


def _build_lra_image_pt(out_pt_path: str, val_frac: float = 0.1, root: str = "./data/cifar10") -> None:
    """Build LRA-Image-style .pt from torchvision CIFAR-10.

    LRA's official Image task is CIFAR-10 grayscale flattened to 1024-token seq.
    We carve a val split from train (10% by default).
    """
    from torchvision import datasets, transforms

    Path(root).mkdir(parents=True, exist_ok=True)

    # Convert RGB → grayscale → uint8 → flatten
    def _to_pixel_seq(x_rgb_tensor):
        """[3, 32, 32] float in [0,1] → [1024] uint8 grayscale."""
        # Standard luminance weights
        gray = 0.299 * x_rgb_tensor[0] + 0.587 * x_rgb_tensor[1] + 0.114 * x_rgb_tensor[2]  # [32, 32]
        gray_u8 = (gray * 255.0).clamp(0, 255).to(torch.uint8)
        return gray_u8.reshape(-1)  # [1024]

    transform = transforms.ToTensor()  # → [3, 32, 32] in [0,1]

    train_full = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    def _to_tensors(dataset, indices=None):
        n = len(dataset) if indices is None else len(indices)
        X = torch.empty((n, IMAGE_SEQ_LEN), dtype=torch.uint8)
        Y = torch.empty((n,), dtype=torch.long)
        iter_indices = range(len(dataset)) if indices is None else indices
        for i, src_idx in enumerate(iter_indices):
            x_rgb, y = dataset[src_idx]
            X[i] = _to_pixel_seq(x_rgb)
            Y[i] = int(y)
        return X, Y

    n_train = len(train_full)
    n_val = int(n_train * val_frac)
    train_idx = list(range(n_train - n_val))
    val_idx = list(range(n_train - n_val, n_train))

    out = {
        "train": _to_tensors(train_full, train_idx),
        "val":   _to_tensors(train_full, val_idx),
        "test":  _to_tensors(test),
    }
    Path(out_pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt_path)
    for split, (X, Y) in out.items():
        print(f"  {split}: N={X.shape[0]}, X.shape={tuple(X.shape)}")
    print(f"Saved LRA Image (CIFAR-10 grayscale) to {out_pt_path}")
