# Copy Memory — Small-scale (dim=64, depth=2, batch=64) — Phase 8 baseline

**Setup**: dim=64, depth=2, heads=4, dim_head=16, ff_mult=4, lr=3e-3, batch=64, steps=1500. 2-3 seeds per cell.
**Source**: Phase 8 / 9 / 10 / 11 small-scale full-cell sweep.
**Note**: screening cells originally ran with `softmask=ON` per the Phase 5 default (later identified as a defect by the Phase 14 audit). **The Copy d=500/d=1000 numbers for `complex_screen` may be suppressed by this defect.** `real_screen` is non-catastrophic at dim=64. Phase 14.5 has produced a softmask=OFF rerun, so for the current corrected values see `result/01_copy_memory/02_small_scale_phase14_audit_corrected_dim32.md`.

## 1. Final accuracy matrix (mean over seeds)

| Cell | Copy d=100 | Copy d=200 | Copy d=500 | Copy d=1000 |
|---|---:|---:|---:|---:|
| real_softmax | 1.000 | 1.000 | 0.956 | 0.050 |
| real_softmax 2× *(param-matched, dim=92)* | 1.000 | 1.000 | **1.000** | 0.101 |
| real_sigmoid | 1.000 | 1.000 | 0.904 | 0.177 |
| real_screen ⚠️ (softmask=ON) | 1.000 | 1.000 | **1.000** | 0.109 (1 seed) |
| real_tanh1 | 1.000 | 1.000 | **0.10 (rand!)** | **0.10 (rand!)** |
| complex_softmax | 1.000 | 1.000 | 0.100 | 0.104 |
| **complex_sigmoid** | 1.000 | 1.000 | **1.000** | **1.000** |
| complex_tanh1 | 1.000 | 0.897 (1 stuck) | 1.000 | 0.690 (1 stuck) |
| complex_screen ⚠️ (softmask=ON) | 0.743 (1 stuck) | 0.873 (1 stuck) | 0.252 ⚠️ | failed ⚠️ |

**Saturation note**: Copy d=100 / d=200 saturate at 100% across all cells (no discriminative power); d=500 and d=1000 are the discriminating regime.

## 2. Headline finding (Phase 8)

**complex_sigmoid is the only cell that solves Copy d=1000 at 100%.** It hits step→1.0 at **step 300** for both seeds; the d=1000 trajectory:

```
step 1 100 200 300 400 500 600 700 800
acc 0.11 0.10 0.78 1.00 1.00 1.00 1.00 1.00 1.00 (mean of 2 seeds)
```

complex_softmax never escapes the random level (0.10); complex_screen fails to train under softmask=ON.

## 3. Phase 11 param-matched fairness check

real_softmax was re-run at `dim=92 (~186K params, 2× complex)` and compared against complex_sigmoid at the same param count:

| Task | real_softmax 2× (186K) | complex_sigmoid (200K) | Δ |
|---|---:|---:|---:|
| Copy d=500 | **1.000** | 1.000 | tied |
| Copy d=1000 | 0.101 (rand) | **1.000** | **+0.899** |

→ complex_sigmoid's Copy d=1000 advantage **persists under param matching**. The "structural advantage from complex" claim holds.

## 4. Implications of the Phase 10 tanh+1 ablation

Trying `tanh(s)+1 = 2σ(2s)` — an activation in the same condition family — on the complex side:

- complex_tanh1 d=200: 1/3 seeds stuck at 0.69 (sigmoid: 3/3 perfect)
- complex_tanh1 d=1000: 1/2 seeds stuck at 0.38 (sigmoid: 2/2 perfect)
- real_tanh1 d=500/d=1000: all seeds at 0.10 (random) — **catastrophic regression**

→ **A gradient that is too steep produces early-commitment failure on long-range Copy.** complex_sigmoid is more stable. Details in `tanh1_vs_sigmoid_long_range_pathology.md` in this folder.

## 5. Phase 12 substrate ablation (deviations from the complex_sigmoid baseline)

Δ on Copy d=500 when each substrate component is turned off one axis at a time. All n=3 seeds:

| Variant | Q,K | V | embed/out | Copy d=500 | phase_memory |
|---|---|---|---|---:|---:|
| baseline complex_sigmoid | C | C | C | 1.000 | 0.995 |
| A1 (cval-off, real V) | C | R | C | 1.000 (Δ=0) | 0.997 (Δ=+0.002) |
| A2 (free real linear) | C-untied | C-untied | C | 1.000 (Δ=0) | 0.999 (Δ=+0.004) |
| A3 (bias-init ±2) | C | C | C | 1.000 (every δ) | 0.980–1.000 |
| A4 (real Q,K, complex V) | R | C | C | 0.996 (Δ=−0.004) | 0.978 (Δ=−0.017) |
| Phase 11 (all real, 2×, dim=92) | R | R | R | **1.000** | 0.875 (1 stuck) |

→ The 4 substrate constraints (cval / native / bias / Q,K complexity) each stay within seed variance individually. **It only collapses when all components are forced real** (phase_memory regression). Complex is **existentially required (at least one component must be complex)**.

## 6. Old §4.6-violating (Phase 5–7) artifacts — not used

The "complex × screening synergy" numbers withdrawn after the Phase 5 §8.4 sign-flip came from §4.6 violations such as one-sided TanhNorm / softmask=ON, and are excluded from this table. The screening primitive itself is re-evaluated under the softmask=OFF rerun.

## 7. Cross-references

- Predecessor: Phase 14 audit (softmask defect confirmed) — `_frameworks/02_phase14_softmask_and_batch_audit.md`
- Successor: PCT-fairness rerun (1.41× param ratio) — `01_copy_memory/04_pct_fairness_d1000_small_d2000_mid_2026-05-08.md`
- Mechanism doc: `_frameworks/01_anti_correlation_preservation_mechanism.md`
