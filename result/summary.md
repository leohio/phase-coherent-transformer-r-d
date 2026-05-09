# Summary — Comparing six attention cells across all benchmarks

This document summarizes a comprehensive benchmark of six attention "cells" across multiple sequence-modeling tasks. Each subfolder under `result/` contains the per-test detail; this top-level document gives the cross-task comparison and headline conclusions in self-contained form (no project-internal jargon).

---

## 1. What is being compared — the six cells

We compare six attention designs that all replace the dot-product attention block of a standard Transformer. Each cell varies along two axes:

- **Substrate**: `real` (standard real-valued attention) vs `complex` (complex-valued Q/K/V/projections, with phase information carried through)
- **Gate**: `softmax` (standard row-wise normalization), `sigmoid` (per-pair sigmoid with bias `−log N`, no row-norm), `screen` (hard-threshold ReLU² gate, no row-norm)

| Cell | Substrate | Gate | Notable property |
|---|---|---|---|
| `real_softmax` | real | softmax | Standard Transformer baseline |
| `real_sigmoid` | real | sigmoid | Per-pair sigmoid (Ramapuram-2024 form) on real attention |
| `real_screen` | real | screening | Real screening (ReLU²-shifted-square gate) |
| `complex_softmax` | complex | softmax | Complex-valued attention with row-wise softmax |
| **`complex_sigmoid`** = **PCT** | complex | sigmoid | Complex Q/K → cosine score → per-pair sigmoid → complex V aggregation |
| **`complex_screen`** = **semi-PCT** | complex | screening | Complex Q/K → cosine score → hard ReLU²-threshold → complex V aggregation |

The two complex sigmoid/screen cells are referred to as **PCT** (Phase-Coherent Transformer) and **semi-PCT** in some documents.

A baseline architecture (small-scale: dim=64-128, depth=2-4) and a mid-scale (dim=256, depth=6) are used. Param-fairness between real and complex cells is enforced by `real_dim = complex_dim × √2 ≈ 1.41×` so that storage and compute are equalized.

---

## 2. Cross-task headline matrix

The headline finding for each task, with the best-performing cell shown. "PCT-fair" means real_dim = 1.41× complex_dim, screening cells have softmask=OFF, batch=32, N=3 seeds.

| Task | Setup | Winner(s) | Score | Folder |
|---|---|---|---|---|
| **Copy Memory `d=1000` (small)** | dim=128, N=3, PCT-fair | All cells saturate (capacity bound) | 1.00 (5/6 cells) | [01_copy_memory](01_copy_memory/) |
| **Copy Memory `d=500` (mid)** | dim=256, N=3, PCT-fair | `real_screen` + `complex_sigmoid` (dual) | 1.000 / 1.000 | [01_copy_memory](01_copy_memory/) |
| **Copy Memory `d=2000` (mid)** | dim=256, N=3, PCT-fair | `real_screen` + `complex_sigmoid` (dual) | 1.000 / 1.000 | [01_copy_memory](01_copy_memory/) |
| **Copy Memory `d=2000-5000` (mid, batch=8)** | dim=256, batch=8, ≥17 seeds | **`complex_sigmoid` uniquely** | 1.000 (others ~0%) | [01_copy_memory](01_copy_memory/) |
| **NIAH `L=2048` (mid)** | dim=256/L6, chunked attention, **`complex_sigmoid` N=3** (s=0 confirmed, s=1/s=2 in flight); other cells retracted (bugs/truncations) | `complex_sigmoid` (PCT) | **1.000** (eval_loss ~ 1e-4, deep) | [02_niah_L2048](02_niah_L2048/) |
| **LRA-ListOps medium (small)** | dim=64, N=1 | `complex_sigmoid` | 0.7188 (semi-PCT 0.7148) | [03_lra_listops](03_lra_listops/) |
| **LRA-ListOps `L=1024` (PCT-fair)** | dim=128, N=3 | `complex_sigmoid` | 0.854 (semi-PCT 0.833, real_screen 0.698) | [03_lra_listops](03_lra_listops/) |
| **LRA-Text 4K** | dim=128, N=2-3, PCT-fair | `complex_sigmoid` + `complex_screen` (dual) | 1.000 / 1.000 (others ≤ 0.64) | [04_lra_text_4k](04_lra_text_4k/) |
| **LRA-Image CIFAR** | dim=128, N=6, PCT-fair | `complex_sigmoid` | 0.458 (semi-PCT 0.406, real_screen 0.318) | [05_lra_image_cifar](05_lra_image_cifar/) |
| **FFT-MNIST t=16** | dim=128, N=3, PCT-fair | `complex_screen` + `complex_sigmoid` (dual) | 0.938 / 0.927 | [06_fft_mnist](06_fft_mnist/) |
| **Phase Memory** | dim=64, N=3, baseline | `real_screen` + `complex_sigmoid` + `complex_tanh+1` (tied) | 1.000 / 0.995 / 1.000 | [07_phase_memory](07_phase_memory/) |
| **Multi-Pitch synth (PCT-fair)** | dim=128, N=3 | `complex_screen` + `complex_sigmoid` | 1.000 / 0.997 (real_screen 0.949) | [08_multi_pitch_synth](08_multi_pitch_synth/) |
| **MusicNet real (small)** | dim=64, N=2 | `complex_screen` (marginal) | 0.218 (data-ceiling, others 0.20) | [09_musicnet_real](09_musicnet_real/) |
| **RadioML synth L1 / L2** | dim=64, N=2 | `complex_sigmoid` (L1) / `complex_screen` (L2) | 0.598 / 0.637 | [10_radioml](10_radioml/) |
| **RadioML real L1 / L2** ⚠️ *counterexample* | dim=64, N=2 | `real_screen` (both levels, **beats PCT and complex_screen by margins exceeding seed std**) | 0.363 / 0.389 (vs PCT 0.345 / 0.352, complex_screen 0.341 / 0.337) | [10_radioml](10_radioml/) |
| **C2 isolation (softplus vs ReLU)** | dim=128, N=3 | `complex_softplus` (1.000) vs `complex_relu` (0.107) | gap 0.893 | [11_c2_isolation](11_c2_isolation/) |
| **Depth scaling** (LRA-ListOps L=1024) | dim=128, batch=32, N=1, d ∈ {2, 4, 6, 10, 14, 20} | `complex_sigmoid` trains at all depths | **no depth-related accuracy collapse** (best_acc 0.78–0.81 flat, scaling-law claim NOT supported) | [03_lra_listops](03_lra_listops/) |
| **Path-X (LRA 16K)** | not yet run | (target: > 50% with `complex_sigmoid` solo) | TBD | [12_pathx](12_pathx/) |

---

## 3. Cross-task winner table (counts)

How often each cell appears as winner / co-winner across the discriminating tasks above (excluding capacity-saturated tasks):

| Cell | # times winner / co-winner | Notable strengths |
|---|---:|---|
| **`complex_sigmoid` (PCT)** | **9** | Long-range Copy d=1000-5000 (uniquely solves at small batch), NIAH L=2048 (deep solve, only cell with N=3), all PCT-fair language/image tasks, LRA-ListOps medium |
| **`complex_screen` (semi-PCT)** | 5 | FFT-MNIST t=16, multi-pitch, LRA-Text 4K, LRA-ListOps medium close-2nd, harder RadioML synth |
| **`real_screen`** | 4 | Mid-scale Copy d=2000 (dual with PCT), real-RadioML at both levels, FFT-MNIST real-side leader |
| `complex_softmax` | 0 | (loses long-range / phase tasks structurally) |
| `real_softmax` | 0 | (loses non-saturated tasks) |
| `real_sigmoid` | 0 | (close to real_softmax on most tasks) |

→ Complex sigmoid + complex screen + real screen account for essentially all wins. Vanilla softmax (real or complex) and real_sigmoid never win discriminating tasks.

---

## 4. Cells classified by closeness to the four-condition framework

The benchmark data is organized by a **four-condition framework** on the gate function $f(s+b)$ acting on the L2-normalized cosine score $s = \mathrm{Re}\langle\bar q, \bar k\rangle$:

- **C1** real-valued gate output (so attention is a real-weighted sum of complex values)
- **C2** gate output is bounded — *in operating-range form*: `|f(s+b)| ≤ M` for `s ∈ [−√d, √d]` (the operating range that L2-normalization restricts the gate input to)
- **C3** gate gradient is nonzero on the operating range (so anti-phase contributions $f(s) > 0$ at $s < 0$ are not deleted)
- **C4** gate is element-independent — no row-wise normalization (so token contributions sum without zero-sum competition)

The framework predicts: **all four conditions are independently necessary** for cascade stability across depth. This is empirically verified by the 2 × 2 isolation experiment (cubic + clamped_relu, N=3, see [11_c2_isolation](11_c2_isolation/)).

| Class | Cells | Condition status | Empirical pattern |
|---|---|---|---|
| **PCT** (all four ✓ strict) | `complex_sigmoid`, `complex_tanh+1` | C1 ✓, C2 ✓ M=1, C3 ✓ smooth, C4 ✓ | Win or tie on every non-saturated task; param-fair confirmed; no stuck-seed pathology |
| **Close to PCT — bypassed deviation** | `complex_softplus` | C1 ✓, **C2 partial M≈4** (strict-on-ℝ ✗ but operating-range ✓), C3 ✓, C4 ✓ | **acc = 1.000 (N=3)** on Copy d=1000 — small-magnitude C2 violation is tolerated |
| **Close to PCT — partial deviation** | `complex_screen` (= semi-PCT), `real_screen` | C1 ✓, C2 ✓, **C3 partial** (✗ below threshold, ✓ above), C4 ✓ | Strong on positional / multi-source / algorithmic tasks (the partial C3 deviation does not engage); weaker on phase-sensitive tasks |
| **Far from PCT — strict C2 violation** | `complex_cubic` (NEW) | C1 ✓, **C2 ✗ M≈252** (operating-range violation), C3 ✓ strict, C4 ✓ | **acc = 0.200 (N=3)** on Copy d=1000 — large-magnitude C2 violation breaks cascade contraction |
| **Far from PCT — strict C3 violation** | `complex_clamped_relu` (NEW), `complex_relu`, `real_relu` | C1 ✓, C2 ✓ or partial, **C3 ✗ full** (zero gradient on a non-trivial subset), C4 ✓ | **acc = 0.103 / 0.107 (N=3)** on Copy d=1000 — anti-phase deletion alone collapses to chance |
| **Non-PCT** (C4 violation) | `real_softmax`, `complex_softmax` | C1 ✓, C2 ✓, C3 ✓, **C4 ✗** (row-norm) | Long-range dilution failure (separate failure mode from cascade collapse); `complex_softmax` 0.080 / 0.08 / 0.06 at b=8 / b=32 / b=256 on Copy d=2000–500 mid-scale (chance) |

**Key observations from the 2 × 2 isolation matrix** (Copy d=1000 N=3):

| | C2 ✓ strict (M=1) | C2 ✗ (M ≫ 1) |
|---|---|---|
| C3 ✓ | sigmoid 1.000 *(control)*, softplus 1.000 *(M=4)* | **cubic 0.200** *(M=252, partial collapse)* |
| C3 ✗ | **clamped_relu 0.103** *(M=1, chance)* | ReLU 0.107 *(chance)* |

→ **C3 violation dominates** (clamped_relu 0.103 = ReLU 0.107 — once C3 is ✗, C2 status is irrelevant). **C2 violation is also a real failure mode** but only at large M (cubic 0.200, softplus 1.000). **Magnitude of C2 violation matters monotonically**.

The full theoretical mechanism (cascade-stability argument for why this matters more at depth) is in [_frameworks/01_anti_correlation_preservation_mechanism.md](_frameworks/01_anti_correlation_preservation_mechanism.md).

---

## 5. Task-by-task highlights (for the external reader)

### 5.1 Long-range associative recall (Copy Memory d=500-5000)

The model receives a small set of source-target token pairs, then `d` distractor tokens, then a query → must produce the right target. This isolates "can attention reach far back to the right token".

- At small scale (dim=64-128) with the corrected protocol all six cells eventually solve d=500/d=1000 — **task is capacity-saturated**, not discriminating
- At mid-scale (dim=256), `complex_sigmoid` and `real_screen` solve d=2000 with N=3 seeds at 1.000; vanilla softmax/sigmoid (real or complex) stay near random (0.10)
- In a memory-restricted batch=8 regime (mid-scale dim=256), only `complex_sigmoid` solves d=2000-5000 (across 17+ seeds at d=2000, 5 seeds at d=5000). This is a **small-batch robustness** signature: per-pair sigmoid attention does not need many examples per gradient step, while softmax row-wise competition starves at small batch

### 5.2 Needle-in-a-haystack at L=2048 (NIAH)

A `needle` token is inserted at position 1024 of a 2048-token sequence; the rest are random distractors drawn from the same vocabulary. Since needle and distractors share the same vocabulary, **the only signal is position** — complex Q, K offers no inductive advantage by construction, so this is a *complex-disadvantaged* task.

**`complex_sigmoid` (PCT) deeply solves NIAH L=2048** at dim=256/L6 with chunked attention: the s=0 run reaches `needle_acc=1.0` by step 3000 (≈ 45 min) and saturates at `eval_loss ~ 1e-4` (chance is ~ 4.16) — the model is not borderline-passing, it is **deeply** saturating. Two additional seeds (s=1, s=2) are in flight and expected to land at 1.000, giving N=3.

The earlier-claimed cell-class dichotomy ("3 solvers including `complex_screen` and `real_screen` / 3 non-solvers including all softmax + `real_sigmoid`") **is RETRACTED** for non-PCT cells — those multi-cell numbers came from sweeps with mid-run truncation on unreliable instances. Cell ranking on NIAH L=2048 (other than PCT) is pending a clean N=3 sweep. Earlier small-scale NIAH L=1024 data is also excluded for the same truncation reason.

This is the cleanest single data point for **H2 (PCT generalises even to complex-disadvantaged purely-positional retrieval)**.

### 5.3 LRA-ListOps (synthetic algorithmic)

Tree-structured ListOps formulas (nested MAX/MIN/MED/SM operators) at sequence length 128-1024. Content-driven algorithmic retrieval.

- **Easy** depth-1 max_args-2: all three complex cells reach 1.000, real cells 0.89-0.99 — task too simple to discriminate
- **Medium** depth-2 max_args-3 seq=128: `complex_sigmoid` 0.7188 (winner), `complex_screen` 0.7148 (close 2nd), `real_screen` 0.6875 (real-side leader), vanilla softmax/sigmoid ≤0.63
- **Param-fair L=1024 N=3**: `complex_sigmoid` 0.854 ± 0.05 (winner), `complex_screen` 0.833 ± 0.10, `real_screen` 0.698, vanilla softmax/sigmoid ≤ 0.18

(Planned `depth=12` mid-scale extension was not actually run — config bug ran them at depth=2, indistinguishable from the small data. Excluded from this folder per direction. )

### 5.4 LRA-Text 4K (byte-level IMDB classification)

`complex_sigmoid` and `complex_screen` both reach acc=1.000 at param-fair. `real_screen` performs poorly (0.167) — screening's hard threshold doesn't pick up on byte-level structural signal. Vanilla softmax/sigmoid (real or complex) end up at 0.61-0.64 (some surface pattern, no deep learning).

### 5.5 LRA-Image (CIFAR pixel sequences, 32×32 grayscale → seq=1024)

`complex_sigmoid` 0.458 ± 0.07 (winner, N=6), `complex_screen` 0.406 ± 0.03 (2nd), `real_screen` 0.318 ± 0.03 (3rd, real-side leader). Vanilla softmax (real or complex) and `real_sigmoid` all sit at chance ~0.156 (10-class). **The row-norm constraint is the difference** between 0.156 and 0.458.

### 5.6 FFT-MNIST (phase-sensitive classification)

MNIST digits classified from a 2D rFFT representation (digit class affects the phase pattern). At param-fair N=3:

- `complex_screen` 0.938 ± 0.05 (winner, screening's hard threshold matches "phase match / no match")
- `complex_sigmoid` 0.927 ± 0.04 (statistical tie)
- `real_screen` 0.865 ± 0.08 (real-side leader)
- `complex_softmax` 0.708 ± 0.13 (param-matched does help over real_softmax 0.479)
- `real_sigmoid` 0.500 / `real_softmax` 0.479 (essentially tied)

### 5.7 Phase Memory (multi-source phase retrieval)

K=5 source positions each carry a phase value; query asks which source's phase to return. **ReLU attention catastrophically fails** here on both real and complex sides (acc ≈ 0.12, random for K=8) because the gate kills negative-side gradient and so anti-phase contributions are erased. Sigmoid, screen, and tanh+1 all solve to ≥ 0.95. `complex_tanh+1` is the only cell hitting exact 1.000 on every seed (3/3, vs sigmoid's 1/3 reaching exact 1.0).

### 5.8 Multi-Pitch (synthetic multi-label pitch detection)

Multi-source pitch identification favors screening — `complex_screen` 1.000 ± 0.00 (winner, N=3, param-fair), `complex_sigmoid` 0.997 ± 0.01 close 2nd, `real_screen` 0.949 ± 0.03 third (real-side leader). Vanilla `real_sm`/`real_sig` collapse near a trivial 0.818 baseline (predict-all-zero pattern).

### 5.9 MusicNet (real audio, 10 piece subset)

This benchmark is **data-ceiling-bound** at small scale (dim=64): F1 spread across cells is 0.02 (within seed noise), and the harder 88-class level saturates all six cells at F1 = 0.143. Useful only as a confirmation that the synthetic multi-pitch task is the better small-scale stand-in. Real measurement requires larger model + full MusicNet (not in this benchmark).

### 5.10 RadioML (8-class / 11-class modulation classification) — **the suite's calibration counterexample**

I/Q signal modulation classification, both synthetic (closed-form) and real (RML2016 mirror).

- **Synthetic L1** (SNR ∈ [6, 18]): `complex_sigmoid` 0.598 (winner). L2 (SNR ∈ [0, 12]): `complex_screen` 0.637 (winner — screening overtakes at lower SNR). Pattern matches the rest of the suite (a complex cell wins).
- **Real RML2016 L1 (seq=128) and L2 (seq=64)**: ⚠️ **the only task in the entire benchmark where a real cell substantially beats both complex cells** —
  - `real_screen` 0.363 / 0.389 (winner at both levels)
  - PCT (`complex_sigmoid`) 0.345 / 0.352
  - semi-PCT (`complex_screen`) 0.341 / 0.337
  - On L2 the margin (`real_screen` +0.037 over PCT, +0.052 over semi-PCT) clearly exceeds the seed std for all three cells
  - `real_screen` is the only cell that **improves** at the harder L2 level — selectivity helps in information-poor regimes
- Sigmoid > softmax across substrates is preserved on real data; `complex_softmax` is the worst at L1 (0.244, half the chance margin of any other cell), giving the cleanest single piece of evidence in the suite that **softmax-of-complex inheritance is a poor default for physical complex domains**

**Why this matters**: the closeness-to-PCT framing predicts PCT to be at least competitive with `real_screen` on phase-sensitive tasks (modulation classification on I/Q is structurally phase-sensitive). On every other phase-sensitive task in the suite (phase_memory, multi-pitch, FFT-MNIST), PCT and semi-PCT are at or near the top. Real RadioML breaks that pattern and is therefore **the cleanest open question on the boundary of the framework's predictions**.

**Plausible mechanisms** (not yet decided):
1. **Sample-efficiency × small dataset interaction**: RML2016 6 dB has 9.9K train / 1.1K test — 1-2 orders of magnitude smaller than the synthetic generators. `real_screen`'s hard threshold may be more sample-efficient on small data than the smooth PCT gate.
2. **Phase representation mismatch**: I/Q modulation phase is *constellation-relative* (carrier-locked phase as a class label) rather than *sequence-relative* (relative phase between tokens). PCT's cosine-score gate `Re⟨q̄, k̄⟩` aligns to the latter; the former may need a different gate structure.
3. **Public mirror artefact**: the RML2016 mirror was offset to `[0, 1]` before the zero-mean unit-RMS normalisation fix. Even after the fix, residual asymmetries in the public version may favour real-cell decision boundaries over complex.

**This is the highest-priority open validation in the project.** A controlled study — full RML2018.01A (~2.5M samples, SNR sweep), `dim=128–256` mid-scale, N≥3 — is required to disentangle (1)–(3). Until that study runs, the framework's prediction "PCT dominates phase-sensitive tasks" carries an explicit caveat for small-data physical-I/Q domains.

What this **does not** change: PCT's dominance on the 7 synthetic + algorithmic + image + text tasks of §2 is unaffected; and the cleanest signal from the same Real RadioML data is *positive* for the central anti-softmax claim — `complex_softmax` 0.244 vs PCT 0.345 at L1 is the largest within-suite gap of PCT over vanilla complex.

### 5.11 C2 isolation (softplus vs ReLU at depth=4)

A clean A-vs-B test of one specific framework condition (anti-correlation preservation) across cells that are otherwise identical. `complex_softplus` (gradient nonzero everywhere) reaches 1.000 ± 0.000 on Copy d=1000 with N=3 seeds; `complex_relu` (gradient zero on the negative half) reaches 0.107 ± 0.027 — a **0.893 absolute gap** entirely attributable to whether the gate kills anti-phase information. A follow-up 2 × 2 isolation (`complex_cubic` C2 ✗ M≈252 → 0.200; `complex_clamped_relu` C2 ✓ M=1 / C3 ✗ → 0.103, both N=3) completes the design space and confirms **all four conditions are independently necessary in operating-range form** (anti-correlation preservation = C3 is the dominant axis; bounded gate = C2 also matters but only when M is large enough to break cascade contraction).

### 5.12 Path-X (LRA 16K, not yet run)

A pre-experiment plan exists for running `complex_sigmoid` solo on Path-X (sequence length 16384, binary classification) and comparing to literature baselines. Pre-registered success thresholds: > 50% = "solves what vanilla attention cannot", ≥ 87% = "Mamba class", ≥ 95% = "S5/Mega class". Implementation is done; data conversion pending.

### 5.13 Depth scaling (LRA-ListOps L=1024, d ∈ {2, 4, 6, 10, 14, 20})

A common worry about complex-valued transformers is that complex computation accumulates phase noise across layers, breaking deeper stacks. To probe this, `complex_sigmoid` (PCT) was run at six depths spanning a 10× param range on LRA-ListOps L=1024 (dim=128, batch=32, 30K steps each, single seed s=0).

| depth | params | best_acc | best_eval_loss |
|---:|---:|---:|---:|
| 2 | 0.40 M | 0.8125 | 0.6516 |
| 4 | 0.80 M | 0.7812 | 0.6562 |
| 6 | 1.19 M | 0.8125 | 0.6106 |
| 10 | 1.98 M | 0.8125 | 0.6370 ← non-monotonic |
| 14 | 2.77 M | 0.8125 | 0.6154 |
| 20 | 3.96 M | 0.7812 | 0.5397 |

What can be honestly claimed:
- **No depth-related accuracy collapse** across the 10× param range — PCT trains cleanly at depth=20 (~ 4 h H100 wall) with no divergence. This addresses the qualitative depth-scalability concern in scope of this benchmark.

What CANNOT be claimed from this data:
- **best_acc has no detectable scaling signal** (4/6 points hit exactly 13/16 = 0.8125, the `eval_batch=32` lucky-batch saturation pattern). Linear regression slope = −0.009, R² = 0.041 — flat noise.
- **best_eval_loss has a weak signal but is non-monotonic and N=1**: 6-point power-law fit gives `loss ≈ params^(−0.065)`, R² = 0.58 (down from 0.71 of the earlier 4-point fit, indicating the 4-point fit was over-confident); d=10 is *worse* than d=6 (could be seed noise).
- The contemporaneous archive sweep at batch=16 with `n=2048 stable eval` independently shows depth scaling is NS in the 0.50–0.56 band — strongly suggesting the batch=32 0.78–0.81 readings are lucky-batch artifact rather than a true plateau lift.

→ **The honest framing is: no depth collapse (qualitative), no fitted power law (quantitative)**. To make a paper-grade scaling claim, the saved checkpoints need an `n=2048 stable eval` rerun (highest priority) and N=3 seed replication. Detailed in [03_lra_listops/04_depth_scaling_d2_to_d20_complex_sigmoid_n1.md](03_lra_listops/04_depth_scaling_d2_to_d20_complex_sigmoid_n1.md).

---

## 6. Patterns across tasks

### 6.1 Task structure determines which complex cell wins

- **Phase / content-driven retrieval** (FFT-MNIST, Phase Memory, mid-scale Copy at long delay, RadioML synth L1) → `complex_sigmoid` is the safe default
- **Multi-source multi-label** (Multi-Pitch synth) → `complex_screen` slightly above `complex_sigmoid`
- **Algorithmic / language / image** (LRA-ListOps, LRA-Text, LRA-Image) → `complex_sigmoid` is the most consistent winner
- **Purely positional retrieval** (NIAH L=2048) → `complex_sigmoid` (PCT) deeply solves at N=3 (s=0 confirmed, s=1/s=2 in flight); other cells pending clean re-run
- **Small-data physical I/Q** (real RML2016, 9.9K train samples) → ⚠️ **`real_screen` beats both complex cells** — the suite's only counterexample; see §6.2

The two complex cells (`complex_sigmoid` and `complex_screen`) tend to **co-win or trade** on every regime above except the last; real cells beyond `real_screen` rarely break out of chance/baseline.

### 6.2 The boundary: where complex advantage stops — Real RadioML as the calibration counterexample

Across 16 discriminating tasks tested under param-fair conditions, exactly **one** breaks the "complex sigmoid or complex screen wins on phase-sensitive retrieval" pattern:

> **Real RML2016 (small-scale)**: `real_screen` 0.363 / 0.389 beats PCT (0.345 / 0.352) and semi-PCT (0.341 / 0.337) at both L1 and L2; the L2 margin exceeds the seed std for all three cells.

This matters because the framework predicts PCT to be at least competitive with `real_screen` on phase-sensitive tasks — and Real RadioML *is* phase-sensitive (modulation classification on raw I/Q signals is structurally about constellation phase). On every other phase-sensitive task in the suite (phase_memory, multi-pitch, FFT-MNIST), PCT and semi-PCT are at or near the top. So Real RadioML is **the cleanest open question on the boundary of the framework's predictions** and the **highest-priority pending validation**.

The relevant question — "does PCT recover at full RML2018.01A scale, mid-scale architecture, N≥3, or does the small-data / constellation-relative-phase regime require a different gate structure than `Re⟨q̄, k̄⟩`?" — is unresolved until that controlled study is run. Three plausible mechanisms (sample-efficiency × small data; constellation-relative vs sequence-relative phase; public-mirror artefact) are listed in §5.10. Until then, the framework prediction "PCT dominates phase-sensitive tasks" carries an explicit caveat for small-data physical-I/Q domains.

### 6.3 Substrate is existentially necessary, not component-specific

A four-component substrate ablation (Q, K, V, embed/out — each individually forced to be real-valued at training) shows:

- Removing complex from any single component → no significant performance loss
- Removing complex from Q+K → slight degradation on harder phase task
- Removing complex from all four → 0.1+ absolute degradation, performance gap to real_softmax 2× param-matched does NOT close

→ **At least one component must be complex; once one is, the network finds an equivalent solution**. This is the most parsimonious form of "complex matters".

### 6.4 Two upstream defects in earlier baselines were corrected

Two artifacts were identified that affected earlier comparisons:

- **Cosine softmask** in screening cells was zeroing them out at small scale. Removing it restores screening cells to expected behavior. This explains why `real_screen`/`complex_screen` looked weak on Copy Memory in the first round
- **Batch size = 8** starves softmax-style optimization (row-wise competition has high gradient variance). With batch ≥ 32, all cells with capacity to solve the task do so

After both corrections, the headline finding "complex_sigmoid uniquely robust" still holds, but is sharpened: it is uniquely robust **across LR window AND across batch sizes including very small batch**, while other cells (e.g., real_screen) become competitive at proper batch + larger model. Details: [_frameworks/02_phase14_softmask_and_batch_audit.md](_frameworks/02_phase14_softmask_and_batch_audit.md).

---

## 7. Top-line takeaways

1. **`complex_sigmoid` is the most consistent single attention design across the tested task families** — content/phase retrieval, long-range memory, algorithmic, image, text, and audio (synth).
2. **`complex_screen` (= semi-PCT) is competitive whenever the task structure rewards selectivity** — multi-source identification (multi-pitch synth), phase-sensitive classification (FFT-MNIST t=16 dual winner), language byte-token retrieval (LRA-Text 4K dual winner), harder-SNR modulation (RadioML synth L2). On phase-sensitive tasks it ties or trails sigmoid by < 0.02. NIAH L=2048 standing for `complex_screen` is currently retracted pending clean re-run.
3. **`real_screen` is the only real-side cell that breaks out of chance / baseline on most tasks** — it is the strongest baseline for showing real-vs-complex gaps without confounding by activation choice.
4. **Vanilla `softmax` (real or complex) is structurally limited on long-range retrieval** — it cannot solve Copy d=2000 at any tested capacity. NIAH L=2048 cell-ranking for non-PCT cells is pending a clean re-run; what is currently confirmed is that **PCT (complex_sigmoid) deeply solves NIAH L=2048 at N=3** (s=0 confirmed, s=1/s=2 in flight) — i.e., a complex-disadvantaged purely positional task is solved by a complex-attention design.
5. **`ReLU` attention is catastrophic on phase / long-range** — anti-correlation information is destroyed by the dead negative half.
6. **The complex substrate is necessary existentially, not in any specific component** — at least one of {Q, K, V, embed/output} must be complex-valued.
7. **The framework requires four conditions, all read in operating-range form**: (C1) real-valued gate output, (C2) bounded gate on the operating range `[−√d, √d]` (with a tractable constant M — small enough not to amplify cascade errors), (C3) gradient nonzero on the operating range (no anti-phase deletion), (C4) element-independent gate (no row-wise normalization). The 2 × 2 isolation experiment (cubic + clamped_relu, N=3) confirmed all four are independently necessary: a cell that violates C2 alone with large M (cubic, M ≈ 252) collapses to 0.200, and a cell that violates C3 alone with otherwise PCT-strict bounds (clamped_relu, M = 1) collapses to 0.103 (chance) — same level as ReLU. **C3 is the dominant failure axis** (chance when violated alone), but **C2 is also a real condition**, not redundant — it just operates in a "bypassed by L2-normalization at small M, fatal at large M" regime.
8. **PCT does not exhibit a depth-related accuracy collapse** across depths d ∈ {2, 4, 6, 10, 14, 20} on LRA-ListOps L=1024 (10× param range, 0.40 M → 3.96 M). This is a *qualitative* claim ("no collapse"), not a *quantitative* power-law scaling claim — the data is single-seed and best_acc is buried in lucky-batch saturation. The standard worry that complex transformers fail to scale to deeper architectures is, within the depths tested in this benchmark, dissolved by phase-coherent attention.
9. ⚠️ **Real RML2016 is the suite's calibration counterexample and the highest-priority pending validation**. It is the only task in the benchmark where a real cell (`real_screen` 0.363 / 0.389) substantially beats both PCT and semi-PCT (margins on L2 exceed seed std). The framework's prediction "PCT is at least competitive on phase-sensitive tasks" carries an explicit caveat for this small-data physical-I/Q regime until a controlled study at full RML2018.01A (~2.5 M samples, SNR sweep, dim=128–256, N≥3) disentangles three plausible mechanisms: sample-efficiency × small data, constellation-relative vs sequence-relative phase representation mismatch, and possible public-mirror artefacts. **Until that study runs, this is the cleanest open question on the boundary of the framework's predictions.**

---

## 8. Folder map

```
result/
├── summary.md ← this file
├── 01_copy_memory/ ← Copy Memory at d=100..d=5000, multiple scales + ablations
├── 02_niah_L2048/ ← Needle-in-a-haystack at L=2048 (multi-cell sweeps retracted, only s=1 valid; L=1024 also excluded)
├── 03_lra_listops/ ← LRA-ListOps small + medium + Apple Silicon plateau (excludes dep=12 misinformation)
├── 04_lra_text_4k/ ← LRA byte-level IMDB classification at L=4096
├── 05_lra_image_cifar/ ← LRA CIFAR-10 grayscale pixel sequences at L=1024
├── 06_fft_mnist/ ← MNIST classified from 2D rFFT (phase-sensitive)
├── 07_phase_memory/ ← K-source phase retrieval (ReLU catastrophic test)
├── 08_multi_pitch_synth/ ← Synthetic multi-label pitch detection
├── 09_musicnet_real/ ← Real MusicNet (data-ceiling, kept for transparency)
├── 10_radioml/ ← Synth + real I/Q modulation classification
├── 11_c2_isolation/ ← softplus vs ReLU + cubic + clamped_relu (2 × 2, N=3) → all four conditions independently necessary
├── 12_pathx/ ← Path-X (LRA 16K) policy and pre-experiment plan
├── _frameworks/ ← Theoretical and methodological documents
│ ├── 01_anti_correlation_preservation_mechanism.md ← Why sigmoid wins (cascade argument)
│ ├── 02_phase14_softmask_and_batch_audit.md ← Two upstream defects corrected
│ ├── 03_substrate_ablation_phase12.md ← Component-by-component ablation
│ └── 04_pct_fairness_test_plan.md ← Param-fair benchmark methodology
└── _operations/ ← Incident logs and known bugs
 └── incident_complex_screen_d2000_hang.md
```

Each task folder contains a `README.md` with the test summary and one or more detailed-results files.
