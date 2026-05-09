# LRA-ListOps on Apple-Silicon MPS — `complex_sigmoid` plateau analysis

**Period**: 2026-05-05 – 2026-05-06
**Hardware**: MacBook (Apple Silicon, 36 GB RAM), MPS backend
**Cell focus**: `complex_sigmoid` (chunked attention path)
**Task**: synthetic LRA-ListOps generator (`data/lra_listops.py`), `seq_len=1024`, `max_depth=5`, `max_args=5`
**Reference baselines** (literature): Vanilla Transformer (Tay et al. 2020 LRA paper) ≈ 0.36; S4 ≈ 0.59; S5 ≈ 0.62.
**Source**: `lra_listops_mps_complex_sigmoid_2026-05.md`

## 1. What was found

**Original goal**: benchmark `complex_sigmoid` at standard LRA-ListOps scale (seq ≥ 500) on the laptop GPU, then push toward `eval_acc = 1`.

**Result**:
- Baseline at `dim=128, depth=4` plateaus at **0.545** [0.524, 0.567] (n=2048 stable eval)
- Adjusting width / depth / batch / LR schedule (within the explored range) cannot push the plateau beyond the per-arm noise band (± 0.044 = 2 × CI half-width)
- The ~0.55 plateau is consistent with the literature ceiling for attention-only architectures on tree-structured ListOps (S4/S5 reach 0.59-0.62)
- `eval_acc = 1` is **not reachable** in the explored hyperparameter family

## 2. Setup

- Synthetic generator (`max_depth=5, max_args=5`) — close to but not the official LRA paper TFRecord median-depth distribution
- **Architecture**: `complex_sigmoid` cell — Eilers-style complex Q/K/V, real-valued cosine score `Re(⟨q̄, k̄⟩)`, element-wise sigmoid weighting (no row-norm), bias init `−log N`, RoPE-modulated attention, ComplexFeedForward. Chunked attention path enabled (`attn_chunk_size`)
- MPS shims: `F.linear(complex)` → `x @ w.t() (+ b)`, `F.normalize(complex)` → manual L2, `torch.linalg.vector_norm(complex)` → `clip_grad_norm_mps` helper
- AdamW β=(0.9, 0.98), eps=1e-9, wd=1e-4, grad-clip=1.0, cosine LR + linear warmup (`step_half` for L=6 run), `eval_batch=64` per training-time eval, final n=2048 stable eval per arm (Wilson 95% CI)

## 3. Per-arm results

| arm | dim | depth | batch | LR schedule | chunk | steps | wallclock | params | best (single batch) | **stable eval** | 95 % CI | Δ vs Phase 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Phase 1** | 128 | 4 | 16 | cosine 5e-4 | 256 | 4 000 | 1.1 h | 0.80 M | 0.3125 | 0.3125† | – | – |
| **Phase 2** (baseline) | 128 | 4 | 16 | cosine 5e-4, warmup 1 500 | 256 | 30 000 | 8.6 h | 0.80 M | 0.7188 (lucky) | **0.5454** | [0.524, 0.567] | 0 |
| **A1** width | 192 | 4 | 16 | cosine 4e-4 | 256 | 8 000 | 2.8 h | 1.79 M | 0.7031 | 0.5347 | [0.513, 0.556] | −0.011 (NS) |
| **A2** depth | 128 | 8 | 16 | cosine 3.5e-4 | 128‡ | 8 000 | 4.8 h | 1.59 M | 0.4688 | 0.3320 | [0.312, 0.353] | **−0.213** (crash) |
| **A3** batch‖ | 128 | 4 | 32 | cosine 5e-4 | 128 | 8 000 | 4.7 h | 0.80 M | 0.6719 | 0.4863 | [0.465, 0.508] | −0.059 (marginal) |
| **L=6 step LR** | 128 | 6 | 16 | step 1e-3 → 2e-4 (50% drop) | 128 | 8 000 | 3.4 h | 1.19 M | 0.6875 | 0.5146 | [0.493, 0.536] | −0.031 (NS) |
| **real_softmax** | 192 | 4 | 16 | cosine 4e-4 | – | 8 000 | 0.5 h | 1.78 M | 0.2188 | **0.1270** | [0.113, 0.142] | −0.418 (caveat §6) |

† Phase 1 has no n=2048 stable eval (early run)
‡ A2 was first attempted with chunk=256 and hung the MPS scheduler at ~35 GB peak (swap)
‖ A3 was originally scheduled at batch=64 but OOM-ed on MPS (~38 GB)

Pre-registered significance gate: `|Δ| > 0.044` (= 2 × Wilson half-width at n=2048, p≈0.55). NS = "not significant".

## 4. Convergence-step observation

Phase 2 ran 30 000 steps but the **8-eval rolling mean** crossed 0.524 (final stable-eval CI lower bound) at **step 8 250** — i.e. plateaued at ~ 1/3 of budget; the remaining 22 000 steps (5.7 h) added only +0.05 in rolling mean, comparable to rolling std. **8000 steps are sufficient.**

## 5. Hypothesis update (Bayesian)

| hypothesis | prior | posterior | reasoning |
|---|---:|---:|---|
| H_C width | 0.50 | 0.05 | A1 (1.5× width) NS vs baseline |
| H_C depth | (0.50) | 0.05 | A2 (2× depth) crashed; L=6 (1.5× depth) NS — neither helps |
| H_O batch | 0.20 | 0.05 | A3 marginally worse; gradient-noise reduction did not lift plateau |
| H_O LR-stability | (0.20) | 0.10 | step LR avoided A2-style crash → optim affects *stability*, not *ceiling* |
| H_I (inductive bias) | 0.20 | 0.55 | survives by elimination; consistent with literature S4/S5 > Transformer |
| H_D (synth ceiling) | 0.10 | 0.20 | survives by elimination; only resolvable with real LRA TFRecord |

## 6. Caveat on the real_softmax comparison

real_softmax @ d=192, L=4, 8 000 steps → stable eval = 0.127 (barely above random 0.10).

This is **not a clean complex_sigmoid comparison**:
1. LR=4e-4 with warmup=500 is too aggressive for vanilla softmax. The LRA paper used 8 000-step warmup (16×) and 50 000 steps total.
2. complex_sigmoid's `−log N` bias init starts attention very sparse — well-suited to sparse-evidence tasks like ListOps. softmax has a diffuse initial distribution.
3. Softmax's row-sum constraint makes it hard to learn competition with many distractors on ListOps (single arg per op).

→ **Honest reading**: in the short-budget regime, complex_sigmoid trains *more easily*; whether softmax can reach the same level under proper LR/warmup tuning is a separate question.

## 7. Operational lessons

1. **MPS RAM ceiling on this machine ≈ 22 GB usable per process**. The peak attention matrix `batch × heads × chunk × N` should stay below ~ 4 GB
2. `chunk_size = 128` is the safe default on MPS for `seq=1024` complex-sigmoid attention with `B ∈ {16, 32}`
3. **eval_batch 64 is too small** — use rolling 8-eval mean for in-run signals, final n=2048 stable eval for the headline
4. **Cosine LR floor + deep complex stack = late-stage crash risk**. Switching to `step_half` (drop to 20% LR at 50%) eliminated this in the L=6 run. Recommended for any `depth ≥ 6` run
5. **Budget guidance**: 8 000 steps is sufficient at `d=128, L=4`; 30 000 was overkill by 3×
6. **Throughput**: real_softmax is **~5× faster** than complex cells on MPS (no MPS shim, native `F.scaled_dot_product_attention`)

## 8. Recommended next steps

1. **Real LRA TFRecord conversion** — resolves H_D (synthetic-vs-real distribution gap)
2. **Best-on-eval ckpt selection** — makes A2-class results reportable without crash artifact
3. **real_softmax retry with LRA-paper-style LR/warmup** (LR=5e-4, warmup=2 000–8 000, 16 000 steps) — required before any "complex_sigmoid > softmax" headline
4. **Multi-seed at chosen baseline** — currently every arm is `seed=0`. 3 seeds at Phase 2 config gives the variance band
5. **Different inductive bias arm** to test H_I directly (state-space block, recurrent attention, or tree-attention)

## 9. Run artifact map

All under `complex_nn_experiment/runs/`:

| arm | directory |
|---|---|
| Phase 1 | `bench_lra_listops_mps_standard_phase1_4ksteps/complex_sigmoid_s0/` |
| Phase 2 | `bench_lra_listops_mps_standard_phase2_30ksteps/complex_sigmoid_s0/` |
| A1 | `bench_lra_listops_mps_standard_A1_d192L4_lr4e-4/complex_sigmoid_s0/` |
| A2 (final) | `bench_lra_listops_mps_standard_A2_d128L8_lr3.5e-4_chunk128/complex_sigmoid_s0/` |
| A3 (final) | `bench_lra_listops_mps_standard_A3_d128L4_batch32_chunk128/complex_sigmoid_s0/` |
| L=6 step LR | `bench_lra_listops_mps_standard_complex_sigmoid_d128L6_lr1e-3_step_half_2e-4/complex_sigmoid_s0/` |
| real_softmax | `bench_lra_listops_mps_standard_real_softmax_d192L4_lr4e-4/real_softmax_s0/` |

Each dir contains `config.json`, `metrics.jsonl`, `summary.json`, `ckpt.pt`, `stable_eval.json`.
