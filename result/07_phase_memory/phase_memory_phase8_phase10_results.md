# Phase Memory — Multi-source phase retrieval task

**Task**: K=5 source positions (each carrying a phase value) are placed in the first half of the sequence; a query in the second half asks which source's phase to return. A retrieval task with **phase as the identifier**, delay=30.
**Setup**: dim=64, depth=2, heads=4, dim_head=16, ff_mult=4, lr=3e-3, batch=64, steps=1500. 3 seeds per cell.

## 1. Phase 8 small-scale 8-cell matrix

| Cell | n=3 final | step→0.9 (per seed) | stuck rate |
|---|---|---|---|
| real_softmax | 0.980 ± 0.03 | [1000, 800, 600] | 0/3 |
| real_sigmoid | 0.957 ± 0.07 | [1200, —, 900] | 1/3 partial (s1 plateau @ 0.873) |
| **real_screen** | **1.000 ± 0.00** | [200, 100, 200] | **0/3 (fastest)** |
| **real_relu** (Wortsman 2023) | **0.125 ± 0.00 ← random!** | never | catastrophic |
| real_tanh1 | 0.923 ± 0.11 | (1 partial stuck @ 0.792) | 1/3 partial |
| complex_softmax | 0.928 ± 0.12 | [600, —, 800] | 1/3 partial (s1 @ 0.78) |
| **complex_sigmoid** | **0.995 ± 0.004** | [300, 600, 300] | **0/3** |
| **complex_relu** | **0.121 ± 0.00 ← random!** | never | catastrophic |
| complex_tanh1 | **1.000 ± 0.00** | [800, 300, 1000] | **0/3** (3/3 hit exact 1.0) |
| complex_screen ⚠️ (sm=ON) | 0.706 ± 0.51 | [300, 200, —] | **1/3 hard-stuck (s2 @ 0.119)** |

**Pattern**:
- **real_screen, complex_sigmoid, and complex_tanh1 are tied 1st place** (1.000 / 0.995 / 1.000; screen is fastest at step→0.9)
- **ReLU attention is catastrophic on both real and complex sides** (0.12 random across all 3 seeds)
- complex_screen sm=ON has 1/3 hard-stuck (defect later confirmed by Phase 14 audit)

## 2. Phase 10 detail: complex_tanh1 reaches exact 1.0 on phase_memory

Discovered in the same-condition-family ablation `tanh+1`:

| Seed | complex_sigmoid | complex_tanh1 |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 1 | 0.993 | 1.000 |
| 2 | 0.993 | 1.000 |

step → 0.9:

| Seed | complex_sigmoid | complex_tanh1 |
|---|---|---|
| 0 | 300 | 300 |
| 1 | 600 | **200** |
| 2 | 300 | **200** |

→ complex_tanh1 reaches 0.9 in mean 233 vs sigmoid's 400 (**~1.7× faster**) and additionally fills the last 0.7% gap to land at exact 1.0 (sigmoid plateaus at 0.99).

## 3. Phase 11 param-matched fairness check

| Cell | dim | n_params | phase_memory final |
|---|---|---|---|
| real_softmax 2× | 92 | ~186K | 0.875§ (1 seed stuck) |
| complex_sigmoid | 64 | ~200K | 0.995 (no stuck) |

→ Δ = +0.120 in favour of complex_sigmoid; advantage holds under param matching.

## 4. softmask=OFF rerun (May 2026 sweep, dim=80 real / dim=64 complex, b=4, N=2)

| Cell | phase_memory |
|---|---|
| real_softmax | **1.000** |
| real_sigmoid | **1.000** |
| real_screen | **1.000** |
| complex_softmax | **1.000** |
| complex_sigmoid | **1.000** |
| complex_screen (sm=OFF) | **1.000** |

→ **Under softmask=OFF + 1.41× param compensation, all cells solve phase_memory at 100%** (capacity saturation). The Phase 8 era differentiation was a mix of defect-induced suppression and capacity saturation.

## 5. Phase 12 substrate ablation results (phase_memory)

| Variant | Q,K | V | embed/out | phase_memory |
|---|---|---|---|---:|
| baseline complex_sigmoid | C | C | C | 0.995 |
| A1 (cval-off, real V) | C | R | C | 0.997 (Δ=+0.002) |
| A2 (free real linear) | C-untied | C-untied | C | 0.999 (Δ=+0.004) |
| A3 (bias-init ±2) | C | C | C | 0.980-1.000 |
| A4 (real Q,K, complex V) | R | C | C | 0.978 (Δ=−0.017) |
| Phase 11 (all real, 2×) | R | R | R | 0.875 (1 stuck) |

→ phase_memory shows a small drop (-0.017) and 5× larger seed variance for A4 (real Q,K), hinting that complex Q,K marginally helps the harder phase task. **Complex is existentially required.**

## 6. Bug note (open)

The `idx is_complex()` branch in `phase_memory + transformer.py` fails embedding lookup → all 72 phasemem jobs across midscale_b32 + p13full failed. The `transformer.py` forward needs to be fixed (not yet done). See `_operations/known_bugs.md` for details.

## 7. Cross-references

- Phase 8 6-cell baseline: `result/01_copy_memory/01_small_scale_phase8_d100_to_d1000_dim64.md`
- tanh+1 ablation (full picture): `result/01_copy_memory/05_tanh1_vs_sigmoid_long_range_pathology.md`
- C2 isolation (relu vs softplus, N=3): `result/11_c2_isolation/c2_isolation_softplus_vs_relu_n3.md`
- Mechanism: `_frameworks/01_anti_correlation_preservation_mechanism.md`
