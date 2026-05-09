# FFT-MNIST — t=8 / t=16 phase-sensitive classification

**Task**: 10-class classification of MNIST representations obtained by 2D rFFT laid out along time (t=8 → seq_len 64; t=16 → seq_len 256). **Digit class affects the 2D-FFT phase pattern**, so this is a representative phase-sensitive task.

## 1. Phase 8 / 9 / 10 / 11 small-scale (dim=64, batch=64, sm=ON for screen — defect-affected)

**Setup**: dim=64, depth=2, heads=4, dim_head=16, ff_mult=4, lr=3e-3, batch=64, steps=1500. 2-3 seeds per cell.

| Cell | t=8 | t=16 |
|---|---:|---:|
| real_softmax | 0.292 | 0.316 |
| real_softmax 2× *(param-matched, dim=92)* | 0.287 | 0.328 |
| real_sigmoid | 0.316 | 0.305 |
| real_screen ⚠️ (sm=ON) | 0.400 | 0.428 |
| real_relu | 0.324 | 0.384 |
| real_tanh1 | 0.306 | 0.367 |
| complex_softmax | 0.342 | 0.387 |
| **complex_sigmoid** | 0.381 | **0.449** |
| complex_relu | 0.243 | 0.329 |
| complex_tanh1 | **0.399** | 0.420 |
| complex_screen ⚠️ (sm=ON) | **0.436** | **0.452** |

**Pattern (under defect)**:
- t=8: complex_screen (0.436) > complex_tanh1 (0.399) > complex_sigmoid (0.381) > real_screen (0.400)
- t=16: complex_screen (0.452) ≈ complex_sigmoid (0.449) > real_screen (0.428) > complex_tanh1 (0.420)

## 2. Phase 11 param-matched (real_sm dim=92 ≈ 186K params)

| Task | real_softmax 2× (186K) | complex_sigmoid (200K) | Δ |
|---|---:|---:|---:|
| FFT t=8 | 0.287 | 0.381 | **+0.094** |
| FFT t=16 | 0.328 | 0.449 | **+0.121** |

→ complex_sigmoid's advantage **persists under param matching** (+0.12 on t=16).

## 3. softmask=OFF rerun (May 2026 sweep, dim=80 real / dim=64 complex, b=4)

Rerun applying softmask=OFF to the screen cells:

```
 fft_t8 fft_t16
real_softmax 0.281 0.319
real_sigmoid 0.281 0.323
real_screen 0.319 0.449
complex_softmax 0.324 0.381
complex_sigmoid 0.439 0.417
complex_screen 0.506 0.497
```

→ **complex_screen rises further at softmask=OFF** (t=8: 0.436 → 0.506, t=16: 0.452 → 0.497); real_screen also holds the Phase 8 value at t=16 (0.449).

## 4. PCT-fairness rerun (2026-05-08, dim=128 complex / dim=184 real, depth=4, batch=32, N=3)

**Setup**: param-fair (1.41× compensation), screening softmask=OFF, 3 seeds (s0/s1/s2).
**Source**: `doc2/results_pct_fairness.md` §1, §2 FFT-MNIST t=16 entry.

`task=fftmnist params={"t": 16}`

| Cell | s0 | s1 | s2 | mean ± std |
|---|---:|---:|---:|---|
| real_sm | 0.406 | 0.500 | 0.531 | 0.479 ± 0.065 (N=3) |
| real_sig | 0.469 | 0.469 | 0.562 | 0.500 ± 0.054 (N=3) |
| real_scr | 0.938 | 0.781 | 0.875 | **0.865 ± 0.079 (N=3)** |
| csm | 0.562 | 0.812 | 0.750 | 0.708 ± 0.130 (N=3) |
| **csg (PCT)** | 0.906 | 0.906 | 0.969 | **0.927 ± 0.036 (N=3)** |
| **cscr (semi-PCT)** | 0.969 | 0.875 | 0.969 | **0.938 ± 0.054 (N=3)** |

→ **At PCT-fair, complex_sigmoid and complex_screen are tied at 0.93/0.94 (N=3); real_screen at 0.86 (N=3); real_sm/sig at 0.48-0.50.** **Going to dim=128 + N=3 + softmask=OFF + param compensation lifts overall scores significantly** (Phase 8 dim=64 era 0.45 → PCT-fair 0.94).

## 5. Key findings (FFT-MNIST family)

1. **PCT (csg) and semi-PCT (cscr) are equivalent on phase-sensitive classification** (gap 0.011, within seed std)
2. **real_screen is uniquely strong on the real side** (0.86 N=3) — Phase 8/10 "real_screen is strong on phase/algorithmic" pattern persists at PCT-fair
3. **complex_softmax is at 0.71 (N=3) under param-matched fair, slightly behind real_screen 0.86** — complex-ifying alone does not fully overcome the row-norm constraint
4. **vanilla real_sm/real_sig at ~0.50 (N=3)** — weakest even at short distance (seq=256)
5. The Phase 8 numbers under softmask=ON era may have been partly suppressed for screen cells; with softmask=OFF and PCT-fair, screening cells fully express their capability

## 6. Cross-references

- Phase 14 audit (softmask defect): `_frameworks/02_phase14_softmask_and_batch_audit.md`
- PCT-fairness test plan: `_frameworks/04_pct_fairness_test_plan.md`
- Mechanism (Anti-correlation Preservation framework): `_frameworks/01_anti_correlation_preservation_mechanism.md`
- Phase 11 substrate ablation: `_frameworks/03_substrate_ablation_phase12.md`
