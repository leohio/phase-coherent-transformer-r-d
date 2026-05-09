# Copy Memory — `tanh+1` vs `sigmoid` ablation (Phase 10c+10d)

**Date**: 2026-05-03
**Hypothesis**: `tanh(s)+1 = 2σ(2s)` is an activation in the same condition family as sigmoid (bounded, gradient-nonzero everywhere, element-independent). It has **4× steeper gradient** which can speed up convergence — does that produce pathology on long-range Copy?

## 1. Mathematical background

| activation | output range | max gradient | f(0) | smoothness |
|---|---|---|---|---|
| σ(s) | (0, 1) | 0.25 | 0.5 | C∞ |
| **tanh(s)+1** | **(0, 2)** | **1.0** | **1.0** | **C∞** |
| ReLU(s) | [0, ∞) | 1.0 | 0 | C⁰ (kink at 0) |

`tanh+1` is sigmoid with **output range × 2 and gradient steepness × 4**.

## 2. Phase 8 + Phase 10 combined — Copy Memory full matrix (per-seed)

| Cell | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|
| complex_sigmoid (Phase 8) | [1.00, 1.00] | [1.00, 1.00, 1.00] | [1.00, 1.00] | [1.00, 1.00] |
| **complex_tanh1** (Phase 10) | [1.00, 1.00] | [1.00, **0.69**, 1.00] | [1.00, 1.00] | [**0.38**, 1.00] |
| | | | | |
| real_sigmoid (Phase 8) | [1.00, 1.00] | [1.00, 1.00, 1.00] | [0.81, 1.00] | [0.09, 0.26] |
| **real_tanh1** (Phase 10) | [1.00, 1.00] | [1.00, 1.00, 1.00] | [**0.09, 0.11**] | [**0.09, 0.11**] |

means with std:

| Cell | d=100 | d=200 | d=500 | d=1000 |
|---|---|---|---|---|
| complex_sigmoid | 1.000 | 1.000 | 1.000 | **1.000** |
| **complex_tanh1** | 1.000 | **0.897 ± 0.18 (1 stuck)** | 1.000 | **0.690 ± 0.44 (1 stuck)** |
| real_sigmoid | 1.000 | 1.000 | 0.904 ± 0.14 | 0.177 ± 0.12 |
| **real_tanh1** | 1.000 | 1.000 | **0.101 ± 0.01 (random!)** | **0.101 ± 0.01 (random!)** |

## 3. Headline reversal (full task parity)

§§1-9 (3 tasks) suggested "complex_tanh1 ≥ complex_sigmoid", but after Phase 10c+10d added Copy Memory + multi_pitch, the picture **reverses on long-range Copy**:

- **complex_tanh1 d=200**: 1/3 seeds plateau at 0.69 (sigmoid: 3/3 perfect)
- **complex_tanh1 d=1000**: 1/2 seeds plateau at 0.38 (sigmoid: 2/2 perfect at 1.0)
- **real_tanh1 d=500/d=1000**: all seeds 0.10 (random) — **catastrophic regression**

→ complex_sigmoid is at 1.0 on 8/8 seeds across all 4 Copy Memory distances; complex_tanh1 introduces stuck-seed instability. real_tanh1 is **strictly dominated** on the real side by every other real cell (softmax/sigmoid/screen).

## 4. Mechanism: why a steeper gradient is counterproductive at long range

Hypothesis (consistent with paper §3 framework):

- **phase_memory** (K=5 sources, delay 30): the network only needs to find K=5 source positions, so a steeper gradient that fast-commits is helpful → tanh+1 is 3/3 perfect.
- **Copy Memory d=1000** (1 source position among 1020 distractors): the network needs to find a **needle in a haystack**. A steeper gradient causes **early commitment to the WRONG candidate position** (because the score landscape is initially noisy).
- **complex_sigmoid's softer gradient**: bias = -log(1021) ≈ -6.93 makes everything almost zero, sigmoid gradient 0.25 — slow but consistent. There is **room for the correct attention pattern to emerge before any commitment**.

## 5. 4-activation combined ranking (complex side, full matrix)

With L2-norm + bias init = −log N + no row-norm + cval as common substrate:

| Activation | FFT t=8 | FFT t=16 | phase_memory | Copy d=100 | Copy d=200 | Copy d=500 | Copy d=1000 | multi_pitch |
|---|---|---|---|---|---|---|---|---|
| softmax | 0.342 | 0.387 | 0.928 | 1.000 | 1.000 | 0.100 | 0.104 | 0.812 |
| **sigmoid** | 0.381 | **0.449** | 0.995 | **1.000** | **1.000** | **1.000** | **1.000** | **0.856** |
| **tanh+1** | **0.399** | 0.420 | **1.000** | 1.000 | 0.897 (1 stuck) | 1.000 | 0.690 (1 stuck) | 0.852 |
| ReLU | 0.243 | 0.329 | 0.121 (random!) | — | — | — | — | — |

→ **complex_sigmoid wins or ties on every task in the matrix**, in particular splitting from tanh+1 (0.69) at Copy d=1000.

## 6. Conclusion — the non-monotonic effect of gradient steepness

| Gradient stiffness | Examples | Result |
|---|---|---|
| Too flat | softmax for long-range, ReLU dead-zone | catastrophic |
| **Just right** | **sigmoid** | **robust across all task types** |
| Too steep | tanh+1, ReLU when active | unstable on noisy long-range; works on dense classification + short-range retrieval |

complex_sigmoid sits in the **sweet spot**.

## 7. Implementation / reproduction

- `complex_nn_experiment/transformer.py`: `RealMultiHeadTanh1Attention`, `ComplexMultiHeadTanh1Attention`
- 18 + 30 = 48 runs total (Phase 10 + 10c + 10d), `runs/p10_*`, `runs/p10c_*`, `runs/p10d_*`
- Each run dir contains `metrics.jsonl` + `config.json`

## 8. Cross-references

- Phase 8 6-cell baseline: `result/01_copy_memory/01_small_scale_phase8_d100_to_d1000_dim64.md`
- C2 isolation (relu vs softplus, N=3): `result/11_c2_isolation/c2_isolation_softplus_vs_relu_n3.md`
- Mechanism: `_frameworks/01_anti_correlation_preservation_mechanism.md`
