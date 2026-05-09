"""FFT-MNIST classification task.

A complex-NN-favorable benchmark in the literature (Trabelsi 2018, complexPyTorch
examples, torchcvnn examples). Each MNIST image is converted to its 2D Fourier
domain representation, which is intrinsically complex. Models must classify the
digit using complex frequency information.

Setup:
  - 28×28 MNIST grayscale image
  - Downsample to 16×16 via avgpool (reduce sequence length for CPU feasibility)
  - 2D FFT → 16×16 complex grid
  - Flatten to 256 complex tokens (each 1 complex feature)
  - 10-way classification

Real cells get (Re, Im) as 2-channel real per token (information-equal); complex
cells get complex64 tokens directly. The advantage real vs complex is purely
architectural inductive bias.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _ensure_mnist(root: str = "./data/mnist") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download MNIST once, return (train_x, train_y, test_x, test_y) as tensors."""
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor()])  # [0, 1] float
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    train_x = torch.stack([train[i][0] for i in range(len(train))]).squeeze(1)  # [N, 28, 28]
    train_y = torch.tensor([train[i][1] for i in range(len(train))])
    test_x = torch.stack([test[i][0] for i in range(len(test))]).squeeze(1)
    test_y = torch.tensor([test[i][1] for i in range(len(test))])
    return train_x, train_y, test_x, test_y


def _downsample_and_fft(x: torch.Tensor, target_size: int = 16) -> torch.Tensor:
    """Downsample to target_size×target_size via bilinear interpolation, normalize, 2D FFT.

    x: [B, 28, 28] real in [0, 1]
    returns: [B, target_size*target_size] complex
    """
    if x.shape[-1] != target_size:
        x = F.interpolate(x.unsqueeze(1), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(1)
    # Normalize to [-1, 1]
    x = x * 2.0 - 1.0
    # 2D FFT (returns complex)
    x_fft = torch.fft.fft2(x.to(torch.float32))  # [B, H, W] complex64
    # Flatten to sequence of complex tokens (row-major)
    x_flat = x_fft.reshape(x.shape[0], -1)  # [B, target_size²]
    return x_flat


_CACHE = {}


def get_fft_mnist(target_size: int = 16, device: str | torch.device = "cpu", root: str = "./data/mnist"):
    """Returns (train_complex, train_y, test_complex, test_y) on `device`.

    train_complex: [60000, target_size²] complex64
    train_y: [60000] long
    test_complex: [10000, target_size²] complex64
    test_y: [10000] long
    """
    key = (target_size, str(device), root)
    if key in _CACHE:
        return _CACHE[key]
    train_x, train_y, test_x, test_y = _ensure_mnist(root)
    train_c = _downsample_and_fft(train_x, target_size).to(device)
    test_c = _downsample_and_fft(test_x, target_size).to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)
    _CACHE[key] = (train_c, train_y, test_c, test_y)
    return _CACHE[key]


def fft_mnist_batch(
    train_complex: torch.Tensor,
    train_y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from the precomputed FFT-MNIST tensors.

    Returns (x, y) where x is [B, N] complex64 and y is [B] long.
    """
    n = train_complex.shape[0]
    if generator is not None:
        idx = torch.randint(0, n, (batch_size,), generator=generator, device=train_complex.device)
    else:
        idx = torch.randint(0, n, (batch_size,), device=train_complex.device)
    return train_complex[idx], train_y[idx]


def vocab_size(**_) -> int:
    return 10  # 10 digit classes


def sequence_length(t: int = 16, target_size: int | None = None, **_) -> int:
    # Accept both 't' (Phase 8 convention) and 'target_size' (legacy)
    sz = target_size if target_size is not None else t
    return sz * sz
