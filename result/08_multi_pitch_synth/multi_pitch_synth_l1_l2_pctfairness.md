# Multi-Pitch — Synthetic MusicNet substitute (L1 / L2 + PCT-fairness)

**Task**: pick `n_active` pure tones from a log-spaced [100..800 Hz] pitch bank, sum them with random phase → 1D rFFT → multi-label prediction of active pitches (multi-label BCE). Adopted as the "MusicNet substitute" in the Phase 8 plan because real MusicNet at small scale saturates.

**Source**: `bench_musicnet_radioml.md` L1+L2 + `musicnet_all_results.md` + `doc2/results_pct_fairness.md` Multi-pitch entry

## 1. Phase 11 small-scale L1 / L2 (synth, 2 seeds, sm=ON)

**Setup**: dim=64, depth=3, heads=4, dim_head=16, ff_mult=4, AdamW, lr=1e-3, batch=64, steps=1500, 2 seeds.

### L1 (n_pitches=8, n_active=3, n_samples=128, seq_len=65)

| cell | params | F1 | ham_acc | eval_loss | wall (s) |
|---|---:|---:|---:|---:|---:|
| real_softmax | 149,512 | 0.588 ± 0.004 | 0.686 ± 0.003 | 0.726 | 79.3 |
| real_sigmoid | 149,515 | 0.594 ± 0.003 | 0.679 ± 0.006 | 0.728 | 84.1 |
| real_screen | 162,775 | 0.661 ± 0.043 | 0.727 ± 0.015 | 0.618 | 134.8 |
| complex_softmax | 149,963 | 0.634 ± 0.007 | 0.733 ± 0.002 | 0.633 | 345.5 |
| complex_sigmoid | 149,966 | 0.675 ± 0.015 | 0.718 ± 0.004 | 0.595 | 635.0 |
| **complex_screen** | 163,229 | **★0.710 ± 0.000** | 0.724 ± 0.001 | **0.534** | 766.8 |

Random-baseline floor F1 ≈ 0.43; trivial all-zero gives ham_acc=0.625 (5/8).

### L2 (n_pitches=16, n_active=3 — harder)

| cell | params | F1 | ham_acc | eval_loss | wall (s) |
|---|---:|---:|---:|---:|---:|
| real_softmax | 150,032 | 0.381 ± 0.003 | 0.607 ± 0.007 | 1.034 ± 0.002 | 122.3 |
| real_sigmoid | 150,035 | 0.377 ± 0.001 | 0.605 ± 0.001 | 1.034 ± 0.000 | 126.8 |
| real_screen | 163,295 | 0.450 ± 0.025 | 0.633 ± 0.015 | 0.920 ± 0.052 | 206.8 |
| complex_softmax | 150,995 | 0.425 ± 0.005 | 0.657 ± 0.002 | 0.938 ± 0.013 | 489.6 |
| complex_sigmoid | 150,998 | 0.493 ± 0.001 | 0.679 ± 0.008 | 0.826 ± 0.005 | 622.4 |
| **complex_screen** | 164,261 | **★0.511 ± 0.002** | **0.684 ± 0.006** | **0.772 ± 0.011** | 732.7 |

→ **complex_screen is the winner at both L1 and L2 difficulty** (F1=0.71 → 0.51). Order preserved: sigmoid > softmax in both substrates; complex > real for screen-rows.

## 2. May 2026 softmask=OFF rerun (dim=80 real / dim=64 complex, b=4)

```
 multi_pitch
real_softmax 0.813* (* trivial collapse)
real_sigmoid 0.813*
real_screen 0.845
complex_softmax 0.821
complex_sigmoid 0.865
complex_screen 0.877 (winner)
```

→ At softmask=OFF, complex_screen 0.877; complex_sigmoid 0.865; real_screen 0.845 — Phase 11 sm=ON era pattern (complex_screen winner) continues.

## 3. PCT-fairness rerun (2026-05-08, dim=128 complex / dim=184 real, depth=4, batch=32, N=3)

`task=multi_pitch params={"n_active": 3, "n_pitches": 16, "n_samples": 128}`

| Cell | s0 | s1 | s2 | mean ± std |
|---|---:|---:|---:|---|
| real_sm | 0.812 | 0.830 | 0.812 | 0.818 ± 0.010 (N=3) |
| real_sig | 0.811 | 0.830 | 0.812 | 0.818 ± 0.011 (N=3) |
| real_scr | 0.926 | 0.980 | 0.941 | 0.949 ± 0.028 (N=3) |
| csm | 0.922 | 0.977 | 0.982 | 0.960 ± 0.033 (N=3) |
| **csg (PCT)** | 1.000 | 0.990 | 1.000 | **0.997 ± 0.006 (N=3)** |
| **cscr (semi-PCT)** | 1.000 | 1.000 | 1.000 | **1.000 ± 0.000 (N=3)** |

→ **At PCT-fair, cscr 1.000 (N=3) and csg 0.997 (N=3) are tied at 1st**; real_screen 0.949 is 3rd; real_sm/sig at 0.818 (near collapse); complex_softmax at 0.960 (unexpectedly competitive under param matching).

## 4. Cell ranking (multi_pitch overall)

| Rank | Cell | PCT-fair (N=3) | small-scale L1 (sm=ON, N=2) |
|---|---|---|---|
| 🥇 | cscr (semi-PCT, complex_screen) | 1.000 | 0.710 |
| 🥈 | csg (PCT, complex_sigmoid) | 0.997 | 0.675 |
| 🥉 | real_screen | 0.949 | 0.661 |
| 4 | csm (complex_softmax) | 0.960 | 0.634 |
| 5 | real_sig | 0.818 | 0.594 |
| 6 | real_sm | 0.818 | 0.588 |

**Pattern**: sigmoid > softmax in both substrates; complex > real for screen-rows; semi-PCT > PCT (close) → screening's hard threshold is effective for multi-source identification.

## 5. Cross-references

- Real MusicNet (data ceiling): `result/09_musicnet_real/musicnet_real_d3_d6_data_ceiling.md`
- PCT-fairness test plan: `_frameworks/04_pct_fairness_test_plan.md`
- Mechanism: `_frameworks/01_anti_correlation_preservation_mechanism.md`
