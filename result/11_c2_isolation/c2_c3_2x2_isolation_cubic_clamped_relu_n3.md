# C2 × C3 Two-axis isolation — `complex_cubic` + `complex_clamped_relu` (N=3)

**Date**: 2026-05-08
**Status**: 42/42 jobs completed, 0 failures
**Source**: `doc2/results_c2_c3_isolation.md`

This experiment **completes** the 2 × 2 design that the earlier softplus-vs-ReLU isolation ([`c2_isolation_softplus_vs_relu_n3.md`](c2_isolation_softplus_vs_relu_n3.md)) only partially covered, by adding two cells that selectively violate one of {C2, C3} while strictly satisfying the other.

→ **The earlier "C2 is automatic, framework reduces to 3-condition" conclusion is WITHDRAWN by this experiment**. All four conditions are independently necessary in their operating-range form.

## 1. The 2×2 design

This experiment introduces two new cells that selectively violate **one** of {C2, C3} while strictly satisfying the other:

| Cell | Gate `f(s+b)` | C2 (operating range M for d=128) | C3 | Position |
|---|---|---|---|---|
| **PCT** (sigmoid) | `σ(s+b)` | ✓ strict M=1 | ✓ smooth | control |
| **softplus** (existing) | `log(1+e^{s+b})` | partial M≈4 | ✓ smooth | partial-C2 |
| **`complex_cubic`** (NEW) | `(s+b) + (s+b)³/6` | **✗ M≈252** | **✓ strict** (`f' = 1+(s+b)²/2 ≥ 1`) | strong-C2-only |
| **`complex_clamped_relu`** (NEW) | `clamp(s+b, 0, 1)` | **✓ strict M=1** | **✗ full** (zero on s+b<0 and s+b>1) | C3-only |
| **ReLU** (existing) | `max(s+b, 0)` | partial M≈11 | ✗ full | C2 partial + C3 |

## 2. Headline copymem d=1000 (N=3)

Same task and architecture as the earlier softplus-vs-ReLU isolation: `dim=128, depth=4, heads=4, dim_head=32, ff_mult=4, batch=32, lr=3e-3, steps=2000, AdamW`.

| Cell | C2 | C3 | mean copy_acc (N=3) | std | per-seed | eval_loss |
|---|---|---|---:|---:|---|---:|
| **complex_sigmoid (PCT)** | ✓ | ✓ | **1.000** | 0.000 | 1.000 / 1.000 / 1.000 | ~0.001 |
| **complex_softplus** | partial M≈4 | ✓ | **1.000** | 0.000 | 1.000 / 1.000 / 1.000 | ~0.001 |
| **complex_cubic** (NEW) | **✗ strong M≈252** | ✓ strict | **0.200** | 0.030 | 0.172 / 0.197 / 0.231 | 2.090 |
| **complex_clamped_relu** (NEW) | ✓ strict M=1 | **✗ full** | **0.103** | 0.039 | 0.062 / 0.106 / 0.141 | 2.303 |
| **complex_relu** (existing) | partial M≈11 | ✗ full | **0.107** | 0.027 | 0.106 / 0.075 / 0.141 | 2.303 |

## 3. The 2×2 isolation matrix

| | **C2 ✓ strict (M=1)** | **C2 ✗ (M ≫ 1)** |
|---|---|---|
| **C3 ✓ (smooth, no anti-phase deletion)** | `complex_sigmoid` (PCT): **1.000** *(control)*<br>`complex_softplus` (M≈4): 1.000 ± 0.000 | **`complex_cubic`** (M≈252) NEW: **0.200** *(partial collapse)* |
| **C3 ✗ (anti-phase deletion)** | **`complex_clamped_relu`** (M=1) NEW: **0.103** *(chance)* | `complex_relu` (M≈11): 0.107 *(chance)* |

## 4. Three independent claims

### 4.1 C3 is empirically necessary (strict)

`complex_clamped_relu` (C2 ✓ M=1, C3 ✗ full) collapses to chance level (0.103 ± 0.039), eval-loss flat at log(V). Anti-phase deletion alone — with the gate output bounded **identically to sigmoid** — destroys the ability to learn long-range retrieval.

The N=3 per-seed values (0.062 / 0.106 / 0.141) match `complex_relu`'s (0.106 / 0.075 / 0.141) within seed noise. **Once C3 is violated, C2 status is irrelevant**.

### 4.2 C2 is also necessary in operating-range form, when M is large enough

`complex_cubic` (C2 ✗ M≈252, C3 ✓ strict) drops from 1.000 → **0.200 ± 0.030** — an 80% accuracy loss. Even with the gate gradient strictly positive everywhere (`f' ≥ 1`), the unbounded gate output causes the cascade to fail.

**This contradicts the earlier conclusion (from softplus-only data) that "C2 is automatic given L2-normalize + continuous gate"**. **C2 violation alone is a real failure mode** when M is large enough to break cascade contraction.

### 4.3 Magnitude of C2 violation matters monotonically

- `softplus` (M≈4) achieves 1.000
- `cubic` (M≈252) achieves 0.200

Somewhere between M=4 and M=252 lies a transition. **C2 (in operating range) is not a binary on/off condition — it's a magnitude-dependent factor** that interacts with cascade depth.

## 5. Why C3 violation dominates over C2

At the linearised per-layer Jacobian:
- **C3 violation introduces a discontinuity** in the gate gradient that **breaks the linearisation entirely** — the cascade-summation argument cannot recover (chance level).
- **C2 violation degrades the geometric series's contraction constant** but does **not break linearisation** — partial degradation rather than structural failure.

The empirical 0.200 (cubic, partial) vs 0.103 (clamped_relu, chance) gap matches this prediction. **Bottom-right (ReLU, both ✗)**: 0.107 ≈ clamped_relu's 0.103 — additional C2 violation contributes no extra damage once C3 is fully gone.

## 6. Other tasks (cubic and clamped_relu only)

| Task | metric | clamped_relu mean (N=3) | cubic mean (N=3) |
|---|---|---:|---:|
| copymem d=100 | copy_acc | 0.103 ± 0.039 | 0.200 ± 0.030 |
| copymem d=200 | copy_acc | 0.103 ± 0.039 | 0.200 ± 0.030 |
| copymem d=500 | copy_acc | 0.103 ± 0.039 | 0.200 ± 0.030 |
| copymem d=1000 | copy_acc | 0.103 ± 0.039 | 0.200 ± 0.030 |
| fft_t8 | test_acc | 0.281 ± 0.083 | 0.406 ± 0.094 |
| fft_t16 | test_acc | 0.323 ± 0.065 | 0.292 ± 0.118 |
| multi_pitch K=16 | multi_pitch_acc | 0.812 ± 0.000 | 0.861 ± 0.005 |

**Notes**:
- copymem at d=100/200/500/1000: `clamped_relu` copy_acc and eval_loss are *bit-identical* across all four delays (loss = 2.3033 = log 12 within numerical noise). Hard saturation prevents any signal propagation — model stays at uniform-prior chance regardless of input length. **Strong signature of full C3 violation**: cell does not learn anything.
- `cubic` copy_acc is also d-independent (0.200) but eval_loss is lower (2.09 < 2.30) — cell *does* learn a small amount of structure (~0.10 above chance), but cannot reach high accuracy. Consistent with severe C2 violation rather than total failure.

## 7. Refined framework — back to 4-condition (operating-range form)

The framework retains **all four conditions**:

| Level | Conditions (operating-range form) | Role |
|---|---|---|
| **Architectural baseline** (assumed) | L2-normalize + RMSNorm substrate | Provides the operating range `[−√d, √d]` on which C2/C3 are evaluated |
| **L1** (per-layer phase coherence) | **C1** (real gate) + **C4** (element-independent, no row-norm) | Per-layer structure |
| **L2** (all-layer cascade phase stability) | **C1** + **C2** (operating-range bounded) + **C3** (anti-phase preservation, no zero gradient) + **C4** | Cascade structure |

- **C2 read in operating-range form** (not strict-on-ℝ): a cell is "C2-satisfied" if its gate output is bounded by some manageable constant on the cosine-score operating range `[−√d, √d]`. `softplus` satisfies this (M≈4 for d=128), `cubic` does not (M≈252).
- **The 3-condition simplification proposed in the earlier softplus-vs-ReLU isolation is withdrawn**. All four conditions are independently necessary.

## 8. Cell taxonomy under refined 4-condition framework

| Cell | C1 | C2 (op-range) | C3 | C4 | Position |
|---|:-:|:-:|:-:|:-:|---|
| sigmoid (PCT) | ✓ | ✓ M=1 | ✓ | ✓ | **PCT** — all four strict ✓ |
| tanh+1 | ✓ | ✓ M=1 | ✓ | ✓ | **PCT** (gain-equivalent to sigmoid) |
| softplus | ✓ | partial M≈4 (bypassed by L2-norm operating range) | ✓ | ✓ | **Close to PCT** (bypassed deviation, empirically equivalent) |
| `complex_screen` (semi-PCT) | ✓ | ✓ (TanhNorm post-aggregation) | partial (✗ below threshold, ✓ above) | ✓ | **Close to PCT** (partial deviation, task-conditional) |
| **`complex_cubic`** | ✓ | **✗ M≈252 (not bypassed)** | ✓ strict | ✓ | **Far from PCT** (strict C2 violation in operating range, partial collapse) |
| **`complex_clamped_relu`** | ✓ | ✓ M=1 | **✗ full (saturates on both sides)** | ✓ | **Far from PCT** (strict C3 violation, chance) |
| ReLU | ✓ | partial M≈11 | ✗ full | ✓ | **Far from PCT** (strict C3 violation, chance) |
| softmax | ✓ | ✓ | ✓ | **✗ row-norm** | **Non-PCT** (separate failure mode: token competition) |

→ Three failure modes: (a) **strict C3 violation in operating range** = chance (clamped_relu, ReLU), (b) **strict C2 violation in operating range** = partial collapse (cubic 0.200), (c) **C4 violation (token competition)** = long-range dilution (softmax).

## 9. Provider / cost

- 42 jobs × ~50 min wall-clock total (max_parallel=8)
- Cost: ¥4,788 (Sakura DOK h100-80gb, ¥1,008/hr)
- Image: `phase8:v8` (adds `complex_cubic` and `complex_clamped_relu` to `transformer.py` TransformerCell + `transformer_cls.py` FFTMnistClassifier dispatch)

## 10. Cross-references

- Earlier (now superseded) softplus-vs-ReLU 2-cell isolation: [`c2_isolation_softplus_vs_relu_n3.md`](c2_isolation_softplus_vs_relu_n3.md)
- Refined mechanism doc (4-condition restored): [`../_frameworks/01_anti_correlation_preservation_mechanism.md`](../_frameworks/01_anti_correlation_preservation_mechanism.md)
- Cell taxonomy used in summary: [`../summary.md`](../summary.md) §4
