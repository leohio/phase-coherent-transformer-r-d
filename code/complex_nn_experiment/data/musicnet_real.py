"""Real MusicNet loader (small subset).

Loads pre-downloaded WAV+CSV pairs from `datasets/musicnet/<id>.{wav,csv}`,
extracts random windows, computes 1D rFFT, and emits multi-label note labels.

Returns (x, y) compatible with TaskTransformer (output_mode='multilabel'):
  x: [B, n_freq_bins] complex64 (rFFT of audio window)
  y: [B, n_notes] float (multi-label: 1 if MIDI note active in window)

Difficulty axes:
- `n_notes`: 21 (≈C4..G#5) for L1 / 88 (full piano) for L2
- `window_samples`: 512 default (at 11025 Hz; ≈46 ms)
- `audio_sr_target`: 11025 Hz default (4:1 decimation from native 44.1 kHz)
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import decimate


DATASET_DIR = Path(os.environ.get(
    "MUSICNET_REAL_DIR",
    "/Users/plasma/repos/complexscreening-00/.claude/worktrees/bridge-cse_011GzoPM6wXvkzzb2wjM5GL7/datasets/musicnet",
))

NATIVE_SR = 44100

# Cache: dict of {piece_id: (audio_target_sr_float32, list_of_notes)}
# Each note: (start_sample_target_sr, end_sample_target_sr, midi_note)
_cache: dict[str, tuple[np.ndarray, list[tuple[int, int, int]]]] = {}


def _load_piece(piece_id: str, target_sr: int = 11025) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Load audio at target_sr and parse CSV labels into target-sr sample indices."""
    cache_key = f"{piece_id}_{target_sr}"
    if cache_key in _cache:
        return _cache[cache_key]

    wav_path = DATASET_DIR / f"{piece_id}.wav"
    csv_path = DATASET_DIR / f"{piece_id}.csv"
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing {wav_path}")
    sr, audio = wavfile.read(str(wav_path))
    assert sr == NATIVE_SR, f"Unexpected SR {sr} for {piece_id}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if audio.dtype != np.float32 or audio.max() > 2.0:
        audio = audio / max(1.0, np.abs(audio).max())
    # Decimate by integer factor (NATIVE_SR / target_sr)
    if target_sr != NATIVE_SR:
        factor = NATIVE_SR // target_sr
        if factor < 1 or NATIVE_SR % target_sr != 0:
            raise ValueError(f"target_sr {target_sr} must divide {NATIVE_SR} evenly")
        if factor > 1:
            audio = decimate(audio, factor, ftype="fir", zero_phase=True).astype(np.float32)

    # Parse notes: rescale sample indices from native to target SR.
    notes: list[tuple[int, int, int]] = []
    factor = NATIVE_SR // target_sr
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = int(row["start_time"]) // factor
            end = int(row["end_time"]) // factor
            midi = int(row["note"])
            notes.append((start, end, midi))

    _cache[cache_key] = (audio, notes)
    return audio, notes


def _active_notes_in_window(notes: list[tuple[int, int, int]], win_start: int, win_end: int) -> set[int]:
    """Notes whose [start, end) overlaps the window [win_start, win_end)."""
    active = set()
    for s, e, midi in notes:
        if e > win_start and s < win_end:
            active.add(midi)
    return active


# Default piece IDs (the 10 test pieces we downloaded).
DEFAULT_PIECE_IDS = ["1759", "1819", "2106", "2191", "2298", "2303", "2382", "2416", "2556", "2628"]


def musicnet_real_seq_len(window_samples: int = 512, **_) -> int:
    """rFFT one-sided length = window_samples // 2 + 1."""
    return window_samples // 2 + 1


def musicnet_real_num_notes(min_midi: int = 60, max_midi: int = 80, **_) -> int:
    return max_midi - min_midi + 1


def gen_musicnet_real_batch(
    batch_size: int,
    window_samples: int = 512,
    target_sr: int = 11025,
    min_midi: int = 60,         # C4
    max_midi: int = 80,         # G#5
    piece_ids: Optional[list[str]] = None,
    require_active: bool = True,
    device: str | torch.device = "cpu",
    generator: Optional[torch.Generator] = None,
):
    """Sample a batch of (rFFT-of-audio-window, multi-label-pitch) pairs.

    Returns:
      x: [B, n_freq_bins] complex64
      y: [B, n_notes] float (multi-label, 1 if MIDI note active in window)

    `require_active`: if True, resamples until at least 1 note is active in
    the chosen MIDI range — avoids many silent / out-of-range examples.
    """
    if piece_ids is None:
        piece_ids = DEFAULT_PIECE_IDS
    n_notes = max_midi - min_midi + 1

    # Lazy-load all pieces on first call.
    pieces: list[tuple[np.ndarray, list[tuple[int, int, int]]]] = []
    for pid in piece_ids:
        pieces.append(_load_piece(pid, target_sr=target_sr))

    if generator is None:
        rng = np.random.default_rng()
    else:
        # Derive numpy RNG from torch generator's seed-like state for reproducibility.
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        rng = np.random.default_rng(seed)

    # Pre-allocate
    x = np.empty((batch_size, window_samples), dtype=np.float32)
    y = np.zeros((batch_size, n_notes), dtype=np.float32)

    for b in range(batch_size):
        for _attempt in range(20):
            piece_idx = int(rng.integers(0, len(pieces)))
            audio, notes = pieces[piece_idx]
            if audio.shape[0] <= window_samples:
                continue
            start = int(rng.integers(0, audio.shape[0] - window_samples))
            end = start + window_samples
            active = _active_notes_in_window(notes, start, end)
            in_range = [m for m in active if min_midi <= m <= max_midi]
            if (not require_active) or len(in_range) > 0:
                x[b] = audio[start:end]
                for m in in_range:
                    y[b, m - min_midi] = 1.0
                break

    # rFFT to complex spectrum.
    x_complex = np.fft.rfft(x, axis=-1).astype(np.complex64)  # [B, n_freq_bins]

    x_t = torch.from_numpy(x_complex).to(device)
    y_t = torch.from_numpy(y).to(device)
    return x_t, y_t
