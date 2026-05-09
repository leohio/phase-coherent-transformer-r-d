# LRA-ListOps L=1024 — `complex_sigmoid` depth scaling sweep (d ∈ {2, 4, 6, 10, 14, 20})

**Date**: 2026-05-08
**Source**: `docs/depthscale.md`
**Goal**: Show how `complex_sigmoid` performance on LRA-ListOps (seq=1024) scales with model depth, holding all other hyperparameters fixed. Probe the long-standing concern that complex transformers do not scale to deeper architectures.

## 1. Setup (held fixed across all depths)

| Item | Value |
|---|---|
| Cell | `complex_sigmoid` (= PCT) |
| Task | `lra_listops`, `max_seq_len=1024`, `max_depth=6` (LRA tree default), `max_args=5`, `vocab_size=18` |
| Model | dim=128, heads=4, dim_head=32, ff_mult=4, RoPE on |
| Attention | chunked (`attn_chunk_size=128`) |
| Batch | effective=32, micro_batch=32 |
| Optimizer | AdamW, weight_decay=0.01, clip_grad=1.0 |
| LR schedule | peak `lr=1e-3`, warmup=500, cosine decay (1.0 → 0.1) |
| Steps | 30 000 |
| Eval | every 500 steps, eval_batch=32 (16 needles → 0.0625 acc quanta) |
| Seed | 0 (single seed; **N=1**) |

Sweep variable: **`depth ∈ {2, 4, 6, 10, 14, 20}`** (6 points, all batch=32, methodology-aligned).

## 2. Headline result (6-point sweep)

| depth | params | final acc | **best acc** | best_acc @ step | final eval_loss | **best eval_loss** | best_evloss @ step | total time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 402 K | 0.594 | **0.8125** | 17 000 | 1.021 | **0.6516** | 17 000 | 40 min |
| 4 | 797 K | 0.563 | **0.7812** | 16 500 | 1.138 | **0.6562** | 28 000 | 62 min |
| 6 | 1.19 M | 0.563 | **0.8125** | 19 000 | 0.971 | **0.6106** | 28 000 | 86 min |
| 10 | 1.98 M | 0.656 | **0.8125** | 19 000 | 0.939 | **0.6370** | 28 000 | 132 min |
| 14 | 2.77 M | 0.656 | **0.8125** | 19 000 | 1.031 | **0.6154** | 28 000 | 179 min |
| 20 | 3.96 M | 0.594 | **0.7812** | 16 500 | 1.042 | **0.5397** | 28 000 | 250 min |

Total compute: 749 min ≈ 12.5 h, ¥12,591 (DOK h100, max_parallel=4).

## 3. What the data shows — and what it does NOT show

### 3.1 Headline: PCT trains cleanly across the full 10× depth range, no collapse

- All 6 jobs train to completion without divergence, NaN, or instability.
- **No depth-related accuracy collapse**: best_acc at depth=20 (3.96 M params, 4 h H100 wall) is 0.7812, indistinguishable from depth=2's 0.8125 within the eval-batch granularity.
- `complex_sigmoid + chunked attention` is therefore **not bottlenecked by depth** at this seq=1024 / dim=128 / batch=32 configuration.

This is the qualitative empirical claim the depth-scalability concern for complex transformers should be evaluated against.

### 3.2 best_acc is FLAT across depth — lucky-batch saturation, not a true scaling signal

| depth | params (M) | best_acc |
|---:|---:|---:|
| 2 | 0.40 | **0.8125** |
| 4 | 0.80 | 0.7812 |
| 6 | 1.19 | **0.8125** |
| 10 | 1.98 | **0.8125** |
| 14 | 2.77 | **0.8125** |
| 20 | 3.96 | 0.7812 |

Linear regression of best_acc vs log10(params): **slope = −0.0090, R² = 0.041 (N=6)** — flat noise.

4 of 6 points hit exactly **0.8125 = 13/16**, the **lucky-batch saturation pattern** for `eval_batch=32` (the chance of 13/16 needles being correct given a true accuracy somewhere ~0.5-0.7 is non-trivial). **The scaling signal is buried under measurement noise**.

→ Cannot read a "monotonic improvement with depth" claim off best_acc. It is an honest "flat or noisy" result.

### 3.3 best_eval_loss has a weak signal but is non-monotonic and N=1

| depth | params (M) | best_eval_loss |
|---:|---:|---:|
| 2 | 0.40 | 0.6516 |
| 4 | 0.80 | 0.6562 |
| 6 | 1.19 | 0.6106 |
| 10 | 1.98 | **0.6370** ← *worse* than d=6, non-monotonic |
| 14 | 2.77 | 0.6154 |
| 20 | 3.96 | **0.5397** |

Power-law fit `log10(best_eval_loss) = α · log10(params) + β`:

| Fit | slope α | intercept β | R² |
|---|---:|---:|---:|
| 4 points (d=2,4,14,20 only) | −0.0714 | 0.226 | **0.71** |
| **6 points (full, this work)** | **−0.0648** | **0.189** | **0.58** |

- Adding d=6, d=10 dropped R² from 0.71 → 0.58 — **the 4-point fit was over-confident**
- d=10 is *worse* than d=6 → non-monotonic, possibly seed noise (N=1)
- Compared with Chinchilla-style real-side LM scaling (`α ≈ −0.34`), our `α ≈ −0.065` is **~ 5× milder** and at much lower R²
- N=1 makes this fit **non-robust** — N=3 replication would be needed for paper-grade scaling claim

### 3.4 Archive context (batch=16 MPS, n=2048 stable eval) — independently NS

`docs/lra_listops_mps_complex_sigmoid_2026-05.md` (depth/width/batch/LR sweep at batch=16, MPS, with **n=2048 stable eval** instead of training-time eval_batch=32):

| arm | depth | batch | params | best (single batch, eval_batch=64) | **stable eval (n=2048)** | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Phase 2 baseline | 4 | 16 | 0.80 M | 0.7188 (lucky) | 0.5454 | [0.524, 0.567] |
| L=6 step LR | 6 | 16 | 1.19 M | 0.6875 | **0.5146** | [0.493, 0.536] |
| L=10 cosine 30K | 10 | 16 | 1.98 M | 0.6719 | **0.5630** | [0.541, 0.584] |

→ All 5 capacity/optim arms (width, depth=8, depth=6, batch, depth=10) land within the **0.50–0.56 noise band** — depth scaling NS at batch=16 with stable eval. **The batch=32 best=0.78–0.81 readings are likely lucky-batch artifact**.

### 3.5 What the data does NOT support

> **Claim like "complex_sigmoid follows a power-law accuracy scaling with depth on LRA-ListOps" cannot be made from this data alone**.
>
> - acc-axis: signal completely buried in lucky-batch saturation (4/6 points at 0.8125)
> - eval-loss-axis: 6-point R² = 0.58, non-monotonic d=6 → d=10, N=1 not robust
> - Archive (batch=16, n=2048 stable eval) confirms depth scaling is NS in the 0.50–0.56 band
> - Chinchilla-style claim cannot be substantiated at this measurement granularity

What is supported, qualitatively:
> **PCT does not exhibit depth-related accuracy collapse across the depths tested (2, 4, 6, 10, 14, 20 spanning a 10× param range)**.

This is a **negative claim about a failure mode** (no depth collapse), not a positive scaling claim (no fitted power law).

## 4. What is needed to make a real scaling claim

1. **n=2048 stable eval on the 6 saved checkpoints** (HIGHEST PRIORITY) — eval-only job, no new training. Removes lucky-batch noise; lets us compare directly with the archive's 0.50–0.56 plateau.
2. **N=3 seed-replication** of the 6-point sweep — disambiguates the d=10 non-monotonic dip vs seed noise.
3. **Larger eval_batch (≥ 128)** during training — `0.0625 quanta` becomes `0.0078 quanta`, lucky-batch ceiling pushed up.

(See `docs/depthscale.md` §6 for the full action-items list. )

## 5. Caveats

1. **N=1 (s=0 only)** — seed instability not assessed
2. **eval_batch=32 lucky-batch ceiling** — 4/6 points at exactly 13/16 = 0.8125 is the signature of saturation, not scaling
3. **task `max_depth=6` fixed** — task-side complexity not scaled with model
4. **LR=1e-3 fixed** — was tuned for shallow depth; may be sub-optimal at d=20
5. **Single-config sweep** — chunk_size, ff_mult, dim, batch all fixed; results may not transfer to other configs

## 6. DOK provenance

| depth | task_id | DOK status | artifact size | submitted |
|---:|---|---|---:|---|
| 2 | `30c6b2b9-fd11-44dc-91fd-2086ccaa46f3` | done (exit 0) | 3.0 MB | 2026-05-07 |
| 4 | `5a5fa635-7236-4d7b-80dc-2be8a10d104e` | done (exit 0) | 5.9 MB | 2026-05-07 |
| 6 | `06f77011-c51c-4919-ac78-ec122a0a5535` | done (exit 0) | 8.8 MB | 2026-05-08 |
| 10 | `5b56d08a-f2bd-4ed0-9bed-2de0e4406068` | done (exit 0) | 14.7 MB | 2026-05-08 |
| 14 | `593ce8c4-14ca-4b10-b571-cdcff3d54859` | done (exit 0) | 20.5 MB | 2026-05-07 |
| 20 | `84dbd9fb-4d01-4213-8d82-9cd0957522bb` | done (exit 0) | 29.3 MB | 2026-05-07 |

Local extracted: `/tmp/art_d{2,4,6,10,14,20}/output/` (`summary.json`, `metrics.jsonl`, `final_model.pt`).

## 7. Cross-references

- archive batch=16 plateau analysis (origin of n=2048 stable eval methodology): [`03_apple_silicon_mps_complex_sigmoid_plateau.md`](03_apple_silicon_mps_complex_sigmoid_plateau.md)
- PCT-fairness L=1024 dep=2 6-cell N=3 (different config, complementary data): [`02_pct_fairness_dep2_L128_L1024_2026-05-08.md`](02_pct_fairness_dep2_L128_L1024_2026-05-08.md)
- summary.md depth-scaling section
