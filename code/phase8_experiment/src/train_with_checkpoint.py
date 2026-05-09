"""Phase 8 training entry with periodic checkpoint + resume support.

Wraps the existing complex_nn_experiment/ training logic with:
  - Multi-task dispatch (Copy Memory, FFT-MNIST, phase_memory, NIAH)
  - Periodic checkpoint to disk (every N steps)
  - Resume from latest checkpoint (auto-detect)
  - Final model saved to SAKURA_ARTIFACT_DIR (DOK auto-uploads)
  - Heartbeat to mgr server (optional)
  - JSON metrics streaming for analysis

Usage (in DOK container):
    python3 train_with_checkpoint.py \\
        --task copymem --task-params '{"K":10,"delay":2000}' \\
        --cell complex_sigmoid --seed 0 \\
        --steps 30000 --checkpoint-every 1000 --eval-every 500 \\
        --output-dir $SAKURA_ARTIFACT_DIR/output \\
        --checkpoint-dir /workspace/ckpt \\
        --mgr-server-url $MGR_URL --mgr-job-id $JOB_ID

Architecture:
    /workspace/ckpt/  ← persistent across container restarts (mounted)
                        used for resume
    $SAKURA_ARTIFACT_DIR/output/  ← uploaded by DOK on task completion
                                    contains final model + metrics + final checkpoint
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Path setup: locate complex_nn_experiment/.
# Order of search:
#   1. $COMPLEX_NN_PATH                           (explicit override)
#   2. /workspace/complex_nn_experiment           (Docker layout)
#   3. <this file>/../../../complex_nn_experiment (repo layout — public release)
_HERE = Path(__file__).resolve()
_CANDIDATES = [
    os.environ.get("COMPLEX_NN_PATH"),
    "/workspace/complex_nn_experiment",
    str(_HERE.parents[2] / "complex_nn_experiment"),
]
for _cand in _CANDIDATES:
    if _cand and (Path(_cand) / "transformer.py").is_file():
        sys.path.insert(0, _cand)
        break
else:
    raise RuntimeError(
        "Could not locate complex_nn_experiment/. Tried: "
        + ", ".join(str(c) for c in _CANDIDATES)
        + ". Set $COMPLEX_NN_PATH to the directory containing transformer.py."
    )
from transformer import make_cell  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Task dispatch
# ─────────────────────────────────────────────────────────────────────────────

def get_task_module(task: str):
    """Import task-specific data generation."""
    if task == "copymem":
        from data.copy_memory import generate_copy_memory_batch, vocab_size, sequence_length
        return {
            "gen_batch": generate_copy_memory_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "copy_acc",
        }
    elif task == "fftmnist":
        # fft_mnist module exposes:
        #   get_fft_mnist(target_size, device) -> (train_complex, train_y, test_complex, test_y)
        #   fft_mnist_batch(train_complex, train_y, batch_size, generator) -> (x, y)
        # Uses FFTMnistClassifier-style model (pooled output) + cross-entropy loss.
        from data.fft_mnist import get_fft_mnist, fft_mnist_batch, vocab_size, sequence_length
        # Cache the precomputed FFT'd MNIST tensors per (target_size, device).
        _cache = {}
        def gen_batch(batch_size, device="cpu", generator=None, t=16, threshold=None, target_size=None, **_):
            sz = target_size if target_size is not None else (threshold if threshold is not None else t)
            key = (sz, str(device))
            if key not in _cache:
                tr_x, tr_y, te_x, te_y = get_fft_mnist(target_size=sz, device=device)
                _cache[key] = (tr_x, tr_y)
            train_complex, train_y = _cache[key]
            inputs, targets = fft_mnist_batch(train_complex, train_y, batch_size, generator=generator)
            T = inputs.shape[1] if inputs.ndim > 1 else (sz * sz)
            full_targets = torch.full((batch_size, T), -100, dtype=torch.long, device=device)
            mask = torch.zeros((batch_size, T), dtype=torch.bool, device=device)
            full_targets[:, -1] = targets
            mask[:, -1] = True
            return inputs, full_targets, mask
        return {
            "gen_batch": gen_batch,
            "vocab_size": vocab_size,    # =10 (digit classes), used as num_classes
            "seq_len": sequence_length,
            "metric_name": "test_acc",
            "model_type": "classifier",  # use FFTMnistClassifier instead of TransformerCell
            "loss_fn": "ce_classifier",  # cross-entropy on pooled logits w/ last-pos label extraction
        }
    elif task == "phase_memory":
        # Existing phase_tasks module
        from data.phase_tasks import (
            gen_phase_memory_batch, phase_memory_seq_len, phase_memory_num_classes,
        )
        def vocab_size(K=5, M=8, **_):
            return phase_memory_num_classes(M=M)
        def sequence_length(K=5, delay=30, **_):
            return phase_memory_seq_len(K=K, delay=delay)
        def gen_batch(batch_size, device="cpu", generator=None, K=5, delay=30, M=8, **_):
            return gen_phase_memory_batch(batch_size, K=K, delay=delay, M=M, device=device, generator=generator)
        return {
            "gen_batch": gen_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "phase_acc",
        }
    elif task == "multi_pitch":
        # Multi-pitch from phase_tasks: complex input (rFFT spectrum), multi-label output.
        # Uses FFTMnistClassifier-style model (pooled output) + BCE loss.
        from data.phase_tasks import (
            gen_multi_pitch_batch, multi_pitch_seq_len, multi_pitch_num_pitches,
        )
        def vocab_size(n_pitches=16, **_):
            # Used as num_classes for FFTMnistClassifier
            return multi_pitch_num_pitches(n_pitches=n_pitches)
        def sequence_length(n_samples=128, **_):
            return multi_pitch_seq_len(n_samples=n_samples)
        def gen_batch(batch_size, device="cpu", generator=None, n_pitches=16, n_active=3,
                      n_samples=128, sample_rate=2000, **_):
            x, y = gen_multi_pitch_batch(batch_size, n_pitches=n_pitches, n_active=n_active,
                                          n_samples=n_samples, sample_rate=sample_rate,
                                          device=device, generator=generator)
            # Pack into 3-tuple format expected by trainer; targets is multi-label float [B, n_pitches].
            # Mask is dummy (BCE loss path doesn't use it).
            mask = torch.ones((batch_size, n_pitches), dtype=torch.bool, device=device)
            return x, y.float(), mask
        return {
            "gen_batch": gen_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "multi_pitch_acc",
            "model_type": "classifier",   # use FFTMnistClassifier instead of TransformerCell
            "loss_fn": "bce",              # multi-label binary cross-entropy
        }
    elif task == "niah":
        # New: synthetic-vocab needle-in-a-haystack
        from data.niah import generate_niah_batch, vocab_size, sequence_length
        return {
            "gen_batch": generate_niah_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "needle_acc",
        }
    elif task == "radioml":
        # New: RadioML 2016.10a modulation classification
        from data.radioml import generate_radioml_batch, vocab_size, sequence_length
        return {
            "gen_batch": generate_radioml_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "mod_acc",
        }
    elif task == "pathx":
        # LRA Path-X: 16384-token binary classification (path connectivity)
        from data.pathx import generate_pathx_batch, vocab_size, sequence_length
        return {
            "gen_batch": generate_pathx_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "pathx_acc",
        }
    elif task == "lra_listops":
        # LRA ListOps: 2048 seq, 10-class algorithmic (synthetic on-the-fly)
        from data.lra_listops import generate_listops_batch, vocab_size, sequence_length
        return {
            "gen_batch": generate_listops_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "listops_acc",
        }
    elif task == "lra_pathfinder":
        # LRA Pathfinder: 1024 seq, binary visual (TFRecord-derived)
        from data.lra_pathfinder import generate_pathfinder_batch, vocab_size, sequence_length
        return {
            "gen_batch": generate_pathfinder_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "pathfinder_acc",
        }
    elif task == "lra_text":
        # LRA Text (IMDB byte-level): 4096 seq, binary
        from data.lra_text import generate_text_batch, vocab_size, sequence_length
        return {
            "gen_batch": generate_text_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "text_acc",
        }
    elif task == "lra_image":
        # LRA Image (CIFAR-10 grayscale, 1024 seq): 10-class
        from data.lra_image import generate_image_batch, vocab_size, sequence_length
        return {
            "gen_batch": generate_image_batch,
            "vocab_size": vocab_size,
            "seq_len": sequence_length,
            "metric_name": "image_acc",
        }
    else:
        raise ValueError(f"Unknown task: {task}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint protocol
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model, optimizer, scheduler, step: int,
                    metrics_history: list, job_spec: dict, rng_state: dict) -> None:
    """Atomic checkpoint save (write to .tmp then rename)."""
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "rng_state": rng_state,
        "metrics_history": metrics_history,
        "job_spec": job_spec,
        "timestamp": time.time(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on same filesystem


def load_checkpoint(path: Path, model, optimizer, scheduler, device: str) -> dict:
    """Load checkpoint into model/optimizer; return step + metrics history."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler and payload.get("scheduler_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    rng = payload.get("rng_state", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return {
        "step": payload["step"],
        "metrics_history": payload.get("metrics_history", []),
    }


def get_rng_state() -> dict:
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Latest non-tmp checkpoint, or None."""
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


# ─────────────────────────────────────────────────────────────────────────────
# Mgr server heartbeat (optional)
# ─────────────────────────────────────────────────────────────────────────────

def mgr_heartbeat(url: str | None, job_id: str, payload: dict) -> None:
    if not url:
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{url.rstrip('/')}/jobs/{job_id}/heartbeat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[heartbeat] WARN: {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    # Job identification
    p.add_argument("--task", required=True,
                   choices=["copymem", "fftmnist", "phase_memory", "niah", "radioml", "pathx",
                            "lra_listops", "lra_pathfinder", "lra_text", "lra_image", "multi_pitch"])
    p.add_argument("--task-params", default="{}",
                   help='JSON dict of task-specific params (e.g., {"K":10,"delay":2000})')
    p.add_argument("--cell", required=True,
                   choices=["real_softmax", "real_sigmoid", "real_tanh1", "real_screen",
                            "complex_softmax", "complex_sigmoid", "complex_tanh1", "complex_screen",
                            "complex_relu", "complex_softplus",
                            "complex_cubic", "complex_clamped_relu"])
    p.add_argument("--seed", type=int, required=True)

    # Architecture
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim-head", type=int, default=32)
    p.add_argument("--ff-mult", type=int, default=4)

    # Training
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Effective batch (gradients averaged across grad-accum micro-steps).")
    p.add_argument("--micro-batch", type=int, default=0,
                   help="Per-step micro batch size. 0 = use --batch-size (no accumulation). "
                        "If <batch-size, gradient accumulation runs batch_size/micro_batch micro-steps "
                        "per optimizer step. batch_size must be divisible by micro_batch.")
    p.add_argument("--eval-batch-size", type=int, default=0,
                   help="Eval batch size; 0 = use --micro-batch (or --batch-size if no accumulation).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--tau-init", type=float, default=None)
    # Phase 14 audit: softmask=OFF is the correct default for screening cells
    # (softmask=ON breaks learning, ~10% chance level). Provide both flags for backward compat:
    # --no-softmask (legacy, has no effect since OFF is now default) and --softmask (opt-in to ON for ablation).
    p.add_argument("--no-softmask", action="store_true",
                   help="(Legacy, kept for backward compat) Disable cosine softmask. softmask is OFF by default now.")
    p.add_argument("--softmask", action="store_true",
                   help="OPT-IN: enable cosine softmask in screening cells. Phase 14 audit found this breaks learning, "
                        "default is OFF. Use only for ablation/Phase-5 reproduction.")
    p.add_argument("--no-tanhnorm", action="store_true",
                   help="Disable post-aggregation TanhNorm in screening cells")
    p.add_argument("--attn-chunk-size", type=int, default=None,
                   help="If set and seq_len exceeds this, use chunked attention (only "
                        "complex_sigmoid; required for Path-X 16K to fit in memory). "
                        "Recommended: 128 or 256 for 16K seq.")
    p.add_argument("--attn-grad-checkpoint", action="store_true",
                   help="Recompute chunked attention during backward to free ~25 GB of "
                        "activation memory at micro_batch=1, 16K seq. Recommended for Path-X.")

    # Checkpoint + eval cadence
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)

    # I/O
    p.add_argument("--output-dir", required=True,
                   help="Final artifacts (DOK uploads automatically). Set to $SAKURA_ARTIFACT_DIR/output.")
    p.add_argument("--checkpoint-dir", default="/workspace/ckpt",
                   help="Persistent checkpoint dir (mounted volume, survives container restart).")
    p.add_argument("--device", default=None)

    # Mgr server (optional)
    p.add_argument("--mgr-server-url", default=os.environ.get("MGR_URL"))
    p.add_argument("--mgr-job-id", default=os.environ.get("JOB_ID"))

    args = p.parse_args()
    if args.micro_batch == 0:
        args.micro_batch = args.batch_size
    if args.batch_size % args.micro_batch != 0:
        raise SystemExit(f"--batch-size {args.batch_size} must be divisible by --micro-batch {args.micro_batch}")
    grad_accum_steps = args.batch_size // args.micro_batch
    if args.eval_batch_size == 0:
        args.eval_batch_size = args.micro_batch

    # ── Setup ──
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    task_params = json.loads(args.task_params)

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Task module
    task_mod = get_task_module(args.task)
    seq_len = task_mod["seq_len"](**task_params)
    V = task_mod["vocab_size"](**task_params)

    # Cell config (port from existing train.py)
    # Phase 14 audit: cosine softmask was empirically found to break screening learning
    # (Copy d=500, batch=256: softmask=ON → 10% chance, softmask=OFF → 90-100%).
    # Default: softmask OFF for screening cells. Use --softmask to opt-in (ablation only).
    # --no-softmask is a no-op (kept for backward compat with old yamls).
    use_softmask = args.cell.endswith("_screen") and args.softmask
    use_tanhnorm = not args.no_tanhnorm
    s_r_init = -3.0 if args.tau_init is None else _tau_to_sr_init(args.tau_init)

    cell_kwargs = dict(
        dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head,
        ff_mult=args.ff_mult, causal=(args.task in ("copymem", "niah", "pathx")), rotary=True,
        use_softmask=use_softmask, use_tanhnorm=use_tanhnorm,
        s_r_init=s_r_init, sigmoid_seq_len=seq_len,
        attn_chunk_size=args.attn_chunk_size,
        attn_grad_checkpoint=args.attn_grad_checkpoint,
    )
    # Branch: classifier-style tasks (fftmnist, multi_pitch) use FFTMnistClassifier
    # which projects continuous (complex) inputs to dim, mean-pools, and produces
    # [B, num_classes] logits. Token-style tasks use TransformerCell with embedding lookup.
    if task_mod.get("model_type") == "classifier":
        from transformer_cls import FFTMnistClassifier  # noqa: E402
        # FFTMnistClassifier doesn't accept attn_chunk_size or attn_grad_checkpoint
        cls_kwargs = {k: v for k, v in cell_kwargs.items()
                      if k not in ("attn_chunk_size", "attn_grad_checkpoint")}
        model = FFTMnistClassifier(args.cell, num_classes=V, **cls_kwargs).to(args.device)
    else:
        model = make_cell(args.cell, num_tokens=V, **cell_kwargs).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"job={args.mgr_job_id or 'local'}  task={args.task}  cell={args.cell}  seed={args.seed}  params={n_params}  seq_len={seq_len}")

    # Optimizer + scheduler (cosine with warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        # cosine decay from 1.0 to 0.1 over remaining steps
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        import math
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume from latest checkpoint, if any ──
    metrics_history: list[dict] = []
    start_step = 0
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None:
        print(f"[resume] loading {latest_ckpt}")
        loaded = load_checkpoint(latest_ckpt, model, optimizer, scheduler, args.device)
        start_step = loaded["step"]
        metrics_history = loaded["metrics_history"]
        print(f"[resume] resumed at step {start_step}, {len(metrics_history)} metrics in history")
    else:
        print("[fresh] no checkpoint, starting from step 0")

    # ── Generators (deterministic) ──
    train_gen = torch.Generator(device=args.device).manual_seed(args.seed * 1000 + 1)
    eval_gen = torch.Generator(device=args.device).manual_seed(args.seed * 1000 + 7)

    # ── Training loop ──
    job_spec = {**vars(args), "n_params": n_params, "seq_len": seq_len, "vocab_size": V}
    metrics_path = output_dir / "metrics.jsonl"
    final_model_path = output_dir / "final_model.pt"

    # ── Loss / eval helpers (branched on model_type / loss_fn) ──
    loss_fn_name = task_mod.get("loss_fn", "ce_sequence")  # default: token-task cross-entropy

    def compute_loss(logits, targets, mask):
        """Returns scalar loss tensor."""
        if loss_fn_name == "bce":
            # multi_pitch: logits [B, num_classes], targets [B, num_classes] float
            return F.binary_cross_entropy_with_logits(logits, targets)
        elif loss_fn_name == "ce_classifier":
            # fftmnist: logits [B, num_classes], targets [B, T] long with class label at last pos
            label = targets[:, -1]  # [B]
            return F.cross_entropy(logits, label)
        else:
            # token tasks: logits [B, T, V], targets [B, T] long with -100 mask
            return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)

    def compute_acc(logits, targets, mask):
        """Returns (correct_count, total_count) tensors so caller can aggregate."""
        if loss_fn_name == "bce":
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct = (pred == targets).float().sum()
            total = torch.tensor(float(targets.numel()), device=logits.device)
            return correct, total
        elif loss_fn_name == "ce_classifier":
            label = targets[:, -1]
            pred = logits.argmax(dim=-1)
            correct = (pred == label).float().sum()
            total = torch.tensor(float(label.numel()), device=logits.device)
            return correct, total
        else:
            pred = logits.argmax(dim=-1)
            correct = ((pred == targets) & mask).float().sum()
            total = mask.float().sum().clamp_min(1)
            return correct, total

    t0 = time.time()
    last_heartbeat = 0.0
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            inputs, targets, mask = task_mod["gen_batch"](
                args.micro_batch, device=args.device, generator=train_gen, **task_params,
            )
            logits = model(inputs)
            micro_loss = compute_loss(logits, targets, mask)
            (micro_loss / grad_accum_steps).backward()
            accum_loss += float(micro_loss.item())
        loss_val = accum_loss / grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()

        # Eval
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                e_inp, e_tgt, e_mask = task_mod["gen_batch"](
                    args.eval_batch_size, device=args.device, generator=eval_gen, **task_params,
                )
                e_logits = model(e_inp)
                e_correct, e_total = compute_acc(e_logits, e_tgt, e_mask)
                acc = e_correct / e_total
                eval_loss = compute_loss(e_logits, e_tgt, e_mask)
            entry = {
                "step": step,
                "train_loss": loss_val,
                "eval_loss": float(eval_loss.item()),
                task_mod["metric_name"]: float(acc.item()),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - t0,
            }
            metrics_history.append(entry)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[step {step:6d}] loss={loss_val:.4f} eval={eval_loss.item():.4f} "
                  f"{task_mod['metric_name']}={acc.item():.4f}", flush=True)

        # Heartbeat (every 30s)
        now = time.time()
        if now - last_heartbeat > 30:
            mgr_heartbeat(args.mgr_server_url, args.mgr_job_id,
                          {"step": step, "loss": loss_val,
                           "elapsed_s": now - t0})
            last_heartbeat = now

        # Checkpoint
        if step % args.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step,
                            metrics_history, job_spec, get_rng_state())
            # Keep only last 3 checkpoints to save disk
            all_ckpts = sorted(ckpt_dir.glob("step_*.pt"),
                               key=lambda p: int(p.stem.split("_")[1]))
            for old in all_ckpts[:-3]:
                old.unlink()
            print(f"[checkpoint] saved step_{step}.pt", flush=True)

    # ── Final model + summary ──
    final_payload = {
        "model_state_dict": model.state_dict(),
        "job_spec": job_spec,
        "final_step": args.steps,
        "metrics_history": metrics_history,
    }
    torch.save(final_payload, final_model_path)

    summary = {
        "job_id": args.mgr_job_id,
        "task": args.task,
        "cell": args.cell,
        "seed": args.seed,
        "task_params": task_params,
        "n_params": n_params,
        "final_metric": metrics_history[-1] if metrics_history else None,
        "total_elapsed_s": time.time() - t0,
        "completed": True,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Final heartbeat with completion
    mgr_heartbeat(args.mgr_server_url, args.mgr_job_id,
                  {"step": args.steps, "completed": True,
                   "final_metric": summary["final_metric"]})

    print(f"\n[done] final_model={final_model_path}  elapsed={time.time()-t0:.1f}s")


def _tau_to_sr_init(tau: float) -> float:
    import math
    if tau <= 0:
        return -10.0
    t = max(min(tau, 0.99), 1e-6)
    return math.log(t / (1.0 - t))


if __name__ == "__main__":
    main()
