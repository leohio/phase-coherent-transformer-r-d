"""LRA ListOps task — on-the-fly synthetic generation (no external dataset).

ListOps is the algorithmic sub-task of Long Range Arena (Tay et al. 2020).
The model sees a nested expression of operations over single-digit operands and
must predict the final result modulo 10.

Operations (matching LRA paper spec):
  [MAX a b c ...]    → max of operands
  [MIN a b c ...]    → min of operands
  [MED a b c ...]    → median (rounded)
  [SM  a b c ...]    → sum mod 10  (called SUM_MOD in some refs)

Operands are single digits (0-9). Operations can nest. The closing bracket ']'
marks end-of-operation. We pad sequences to a fixed `max_seq_len` (default 2048).

Tokens (vocab size 18):
   0..9           single digits (literal value = token index)
   10  '['        OPEN bracket — followed by operator token
   11  ']'        CLOSE bracket
   12  MAX
   13  MIN
   14  MED
   15  SM
   16  PAD          (only at end of sequence)
   17  CLS          (final retrieve position; used like a query)

Output: 10-class classification (the computed result digit) at the CLS position.

Implementation notes:
  - Depth is randomized per example to mimic LRA's distribution (median ~5 in LRA)
  - We reject samples whose tokenized length exceeds max_seq_len
  - The CLS token is appended at position max_seq_len-1; targets are -100 elsewhere
  - This is a synthetic on-the-fly generator; results are not directly comparable
    to LRA's published numbers (LRA uses a fixed pre-generated split). It is
    sufficient for correctness testing of the chunked attention path.

For matching LRA exactly, use the official LRA TFRecord (see `_load_lra_listops_pt`).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List

import torch


VOCAB_SIZE = 18
DIGIT_TOKEN = lambda d: d            # 0..9
OPEN_TOKEN = 10
CLOSE_TOKEN = 11
OP_MAX = 12
OP_MIN = 13
OP_MED = 14
OP_SM = 15
PAD_TOKEN = 16
CLS_TOKEN = 17

OP_TOKENS = (OP_MAX, OP_MIN, OP_MED, OP_SM)

DEFAULT_MAX_SEQ_LEN = 2048

# Optional .pt cache (LRA-faithful split if user runs the conversion helper)
DATASET_PATH = Path(os.environ.get(
    "LRA_LISTOPS_PATH",
    "/var/lib/phase8/datasets/lra_listops.pt",
))

_cached_data = None


def vocab_size(**_) -> int:
    return VOCAB_SIZE


def sequence_length(max_seq_len: int = DEFAULT_MAX_SEQ_LEN, **_) -> int:
    return max_seq_len


def _eval_op(op: int, args: List[int]) -> int:
    if op == OP_MAX:
        return max(args)
    elif op == OP_MIN:
        return min(args)
    elif op == OP_MED:
        sorted_args = sorted(args)
        return sorted_args[len(sorted_args) // 2]
    elif op == OP_SM:
        return sum(args) % 10
    raise ValueError(f"unknown op token {op}")


def _generate_tree(depth: int, max_args: int, rng: random.Random) -> tuple[List[int], int]:
    """Generate one ListOps tree. Returns (token_sequence, computed_value)."""
    if depth <= 0 or rng.random() < 0.3:
        d = rng.randint(0, 9)
        return [d], d

    op = rng.choice(OP_TOKENS)
    n_args = rng.randint(2, max_args)
    tokens: List[int] = [OPEN_TOKEN, op]
    arg_values: List[int] = []
    for _ in range(n_args):
        sub_tokens, sub_value = _generate_tree(depth - 1, max_args, rng)
        tokens.extend(sub_tokens)
        arg_values.append(sub_value)
    tokens.append(CLOSE_TOKEN)
    return tokens, _eval_op(op, arg_values)


def _generate_one(max_seq_len: int, max_depth: int, max_args: int, rng: random.Random,
                  max_attempts: int = 50) -> tuple[List[int], int]:
    """Generate one ListOps example fitting within max_seq_len-1 (room for CLS)."""
    for _ in range(max_attempts):
        depth = rng.randint(max(1, min(2, max_depth)), max(1, max_depth))
        tokens, value = _generate_tree(depth, max_args, rng)
        if len(tokens) <= max_seq_len - 1:
            return tokens, value
    # Fallback: trivial 1-digit example (always fits)
    d = rng.randint(0, 9)
    return [d], d


def generate_listops_batch(
    batch_size: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_depth: int = 6,
    max_args: int = 5,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    split: str = "train",
    **_,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of ListOps examples (on-the-fly generated).

    Returns (inputs, targets, mask):
      inputs: [B, max_seq_len] long  — tokens, PAD padded; CLS at last position
      targets: [B, max_seq_len] long — class label (0..9) at last position only, -100 elsewhere
      mask: [B, max_seq_len] bool   — True only at last position
    """
    if generator is not None:
        # Advance generator state so each call produces a different batch.
        # torch.randint() requires a CPU generator; use Tensor.random_() which
        # accepts a same-device generator, so this works on both CPU and CUDA.
        sub_seed = int(
            torch.empty((), dtype=torch.int64, device=generator.device)
            .random_(0, 2**31 - 1, generator=generator)
            .item()
        )
    else:
        sub_seed = random.randint(0, 2**31 - 1)
    rng = random.Random(sub_seed + (0 if split == "train" else 1_000_000))

    inputs = torch.full((batch_size, max_seq_len), PAD_TOKEN, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        tokens, value = _generate_one(max_seq_len, max_depth, max_args, rng)
        L = len(tokens)
        inputs[b, :L] = torch.tensor(tokens, dtype=torch.long)
        # CLS at last position (queries the result)
        inputs[b, -1] = CLS_TOKEN
        labels[b] = value

    inputs = inputs.to(device)
    targets = torch.full((batch_size, max_seq_len), -100, dtype=torch.long, device=device)
    target_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
    targets[:, -1] = labels.to(device)
    target_mask[:, -1] = True

    return inputs, targets, target_mask


def _load_lra_listops_pt() -> dict:
    """Load LRA-faithful ListOps split from a pre-converted .pt file (optional)."""
    global _cached_data
    if _cached_data is not None:
        return _cached_data
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"LRA ListOps dataset not found at {DATASET_PATH}. "
            f"Use generate_listops_batch() for synthetic on-the-fly examples, "
            f"or convert LRA TFRecord via `_convert_lra_listops_tfrecord` and set LRA_LISTOPS_PATH."
        )
    _cached_data = torch.load(DATASET_PATH, map_location="cpu", weights_only=True)
    return _cached_data


def _convert_lra_listops_tfrecord(tfrecord_dir: str, out_pt_path: str,
                                  max_seq_len: int = DEFAULT_MAX_SEQ_LEN) -> None:
    """One-time conversion from LRA ListOps TFRecord to PyTorch .pt.

    Run on a machine with TF installed.
    """
    import tensorflow as tf

    feature_description = {
        "inputs": tf.io.VarLenFeature(tf.int64),
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
            x = tf.sparse.to_dense(record["inputs"]).numpy().astype("int64")
            y = int(record["targets"].numpy()[0])
            x_padded = list(x[:max_seq_len])
            while len(x_padded) < max_seq_len:
                x_padded.append(PAD_TOKEN)
            x_padded[-1] = CLS_TOKEN
            X_list.append(x_padded)
            Y_list.append(y)
        X = torch.tensor(X_list, dtype=torch.int64)
        Y = torch.tensor(Y_list, dtype=torch.int64)
        out[split] = (X, Y)
        print(f"  {split}: N={X.shape[0]}, X.shape={tuple(X.shape)}")

    Path(out_pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt_path)
    print(f"Saved LRA ListOps to {out_pt_path}")
