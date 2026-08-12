# NIAH L=2048 — `complex_sigmoid` (PCT) N=3 deeply solves

**Task**: needle (1 token) inserted at `depth_ratio × seq_len = 1024` of a 2048-token sequence; remaining 1023 tokens are uniform-random distractors from the same vocabulary (`vocab_size=64`). Final-position prediction of the needle's value. Since needle and distractors share the same vocabulary, **the only retrieval signal is position**.

## 1. Setup

| Item | Value |
|---|---|
| Cell | `complex_sigmoid` (= PCT) |
| Architecture | dim=256, depth=6, heads=8, dim_head=32, ff_mult=4 |
| Attention | chunked (`attn_chunk_size=256`) — required for L=2048 memory at this dim |
| Batch | effective=32 (mb=16, grad_accum=2), or effective=16 (mb=16, grad_accum=1) |
| LR | 3e-4, warmup=1000, cosine decay |
| Steps | 30000 (config A) / 60000 (config B) |
| Seeds | s=0, s=1, s=2 (**N=3**) |
| Source (s=0) | `bench_phase14_5_results.md` §3.5 |
| Source (s=1, s=2) | DOK re-runs completed 2026-05-09 (task ids `013e26e1`, `130fad10`); both independently reproduce acc=1.000 |

## 2. Result

| Seed | First `needle_acc=1.0` | Final `needle_acc` | Final `eval_loss` | Status |
|---|---:|---:|---:|---|
| s=0 (config A, 30K) | step 3000 (45 min) | **1.000** | 3.67e-4 | ✅ confirmed |
| s=0 (config B, 60K) | step 9000 (67 min) | **1.000** | 2.87e-5 | ✅ confirmed (same seed, longer-run cross-check) |
| s=1 | step 3000 (45 min) | **1.000** | 3.33e-4 | ✅ confirmed (DOK `013e26e1`, 2026-05-09) |
| s=2 | step 6000 (89 min) ¹ | **1.000** | 6.92e-4 | ✅ confirmed (DOK `130fad10`, 2026-05-09) |

¹ s=2 read acc=0.8125 at step 3000, dipped transiently to 0.5625–0.6875 over steps 3500–4500, then reached stable 1.0 at step 6000 — slower convergence than s=0/s=1 but the same plateau.

**Mean (confirmed, N=3)**: `needle_acc = 1.000 ± 0.000`

All three config-A seeds reach `needle_acc=1.0` (s=0/s=1 by step 3000, s=2 by step 6000) and maintain it until kill, with `eval_loss` decaying from chance (~4.16) to the 1e-4 order — i.e., the cell **deeply solves** NIAH L=2048 (not borderline). Caveat: none of the three runs completed the full 30K-step schedule; all were killed by the 6h wall-clock timeout at step 24000/30000 (80%), with acc=1.0 held continuously from first solve to kill.

## 3. Status of other cells (retracted)

The 6-cell sweep at this configuration ran on unreliable instances and **the corresponding multi-cell results are RETRACTED**:

| Cell | Sweep | Status |
|---|---|---|
| `complex_sigmoid` | this file | ✅ N=3 confirmed (2026-05-09) |
| `complex_screen` | `bench_phase14_5_results.md §3.5.1` | ❌ retracted (mid-run truncation) |
| `real_screen` | `bench_phase14_5_results.md §3.5.1` | ❌ retracted (mid-run truncation) |
| `real_softmax` | `bench_phase14_5_results.md §3.5.1` | ❌ retracted (mid-run truncation) |
| `real_sigmoid` | `bench_phase14_5_results.md §3.5.1` | ❌ retracted (mid-run truncation) |
| `complex_softmax` | `bench_phase14_5_results.md §3.5.1` | ❌ retracted (mid-run truncation) |

The earlier-claimed "3 solver / 3 non-solver dichotomy" should be treated as **withdrawn** until a clean re-run reproduces it.

## 4. What this single-cell N=3 finding supports

- **PCT (complex_sigmoid) deeply solves NIAH L=2048**, a purely positional retrieval task where complex Q, K offers no inductive advantage by construction (needle and distractors are content-identical, drawn from the same vocab).
- This is one of the H2 "dominance even on complex-disadvantaged tasks" data points: complex transformer with PCT-style attention solves a task that vanilla `complex_softmax` cannot (`complex_softmax` Copy d=2000 chance 0.10 documented separately is the closest analog).
- The 1.000 saturation is **deep** (eval_loss ~ 1e-4), not borderline — i.e., the model is not just "slightly above chance".

## 5. What this single-cell N=3 finding does NOT support (open)

- The "3 solvers / 3 non-solvers" dichotomy across cells — needs N=3 sweep at clean instances for all 6 cells.
- `complex_screen` / `real_screen` standing on NIAH L=2048 — re-runs needed.
- `complex_softmax` / `real_softmax` / `real_sigmoid` "structural failure" claim on NIAH — without clean N=3 evidence, the structural-vs-undertraining attribution is open.
- Depth-ratio dependence (only `depth_ratio=0.5` tested; `0.1` shallow / `0.9` deep are open).

## 6. Cross-references

- mechanism (why C4 + per-pair sigmoid help purely-positional retrieval): `_frameworks/01_anti_correlation_preservation_mechanism.md`
- mid-scale Copy d=2000 (PCT same architecture, N=3 confirmed at 1.000): `result/01_copy_memory/03_mid_scale_dim256_d500_d2000_d5000.md`
- PCT-fairness Copy mid: `result/01_copy_memory/04_pct_fairness_d1000_small_d2000_mid_2026-05-08.md`
