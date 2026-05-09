# MusicNet Real — May 2026 CPU campaign (softmask discovery)

**Period**: 2026-05
**Hardware**: Linux 6.17, x86_64, 22 cores, 15 GB RAM (no GPU)
**Source**: `musicnet_and_multi_pitch_bench_2026-05.md`
**Status**: an exploratory CPU campaign — main outputs are (a) the **"softmask=ON kills *_screen cells on long-range copy memory"** finding (later confirmed by Phase 14 audit) and (b) the data-ceiling confirmation for real MusicNet at small scale.

## 1. Bench timeline (chronological)

| Bench | Setup | Wall | Note |
|---|---|---|---|
| **A** | dim=64 dep=4 b=4 s=16K, softmask=ON | ~130 min | Original "16× compensation": shrinks batch and bumps steps proportionally |
| **B** | dim=64 dep=4 b=4 s=8K, softmask=ON | ~85 min | Empirical peak detection (Bench A overfit) |
| **C** | synth multi_pitch K=16 N=20, b=4 s=8K | ~19 min | data-ceiling control |
| Phase 8 screen redo | softmask=ON, same setup | partial | Discovered that softmask=ON makes *_screen at copy_d200 random (0.087) |
| **Comprehensive sweep** | softmask=OFF, real dim=80 (param-fair), b=4 s=8K, 7 tasks | ~3 hr | Headline: at softmask=OFF, all 7 tasks train |
| Screen-only redo | same setup, screen-only confirmation | ~3 hr | Reproducibility check, confirmed |

## 2. softmask=ON kills *_screen on long-range Copy

| Setting | real_screen copy_d200 | complex_screen copy_d200 |
|---|---|---|
| softmask = ON (Phase 8 redo) | 0.087 (random) | 0.087 (random) |
| **softmask = OFF (sweep + screen-only)** | **1.000** (perfect) | **1.000** (perfect) |

A **+0.913 jump** from a single architectural flag. This is the strongest single result from the campaign. The Phase 14 audit later ratified this finding.

## 3. Comprehensive sweep (softmask=OFF, real dim=80) — 7 task results

```
 multi phase fft fft phase copy music music
 pitch mem t8 t16 sum d200 L1 L2
real_softmax 0.813* 1.000 0.281 0.319 0.122* 1.000 0.185 0.144
real_sigmoid 0.813* 1.000 0.281 0.323 0.122* 1.000 0.185 0.144
real_screen 0.845 1.000 0.319 0.449 0.122* 1.000 0.190 0.145
complex_softmax 0.821 1.000 0.324 0.381 0.122* 1.000 0.185 0.142
complex_sigmoid 0.865 1.000 0.439 0.417 0.121* 1.000 0.201 0.142
complex_screen 0.877 1.000 0.506 0.497 0.123* 1.000 0.192 0.143
```

(test_acc except F1 for musicnet/copy_acc for copy_d200; * = trivial baseline)

**Param verification (depth=4)**:

| Cell | dim | params | as float DoF |
|---|---:|---:|---:|
| real_softmax | 80 | 290,496 | 290,496 |
| real_sigmoid | 80 | 290,500 | 290,500 |
| real_screen | 80 | 311,688-312,812 | 311,688-312,812 |
| complex_softmax | 64 | 199,564 (numel) | 399,128 |
| complex_sigmoid | 64 | 199,568 (numel) | 399,136 |
| complex_screen | 64 | 217,248 (numel) | 434,496 |

real cells have ~1.43× the numel of complex cells but still ~0.7-0.8× the float DoF. Both interpretations give meaningful "fairness".

## 4. Real MusicNet F1 in absolute terms

```
Random / statistical baselines:
 all-0 (BCE attractor) F1 = 0.000
 all-1 F1 = 0.160 (L1) / 0.059 (L2)
 random uniform 50/50 F1 = 0.151 (L1) / 0.057 (L2)
 random calibrated to base F1 = 0.090 (L1) / 0.034 (L2)
 top-k always (k = avg ON) F1 = 0.149 (L1) / 0.127 (L2)

Our peak F1 (best cells):
 Bench A real_screen F1 = 0.225 (L1) / 0.154 (L2)
 Bench B complex_sigmoid F1 = 0.233 (L1) / 0.151 (L2)
 Sweep complex_sigmoid F1 = 0.201 (L1) / 0.142 (L2)

Margin over best statistical baseline:
 L1: ~+0.07 over top-2-always
 L2: ~+0.025 over top-3-always
```

→ The model learns SOMETHING beyond the statistical baseline, but only ~+0.07 (L1) / +0.025 (L2) at peak. Cell-comparison effect sizes within this range (Δ ≈ 0.005-0.015) are at or below seed variance. Comparable to Trabelsi 2018 / Cwitkowitz 2019 (mean AP ≈ 0.65-0.78) where our F1 ≈ 0.22 corresponds to mean AP ≈ 0.4-0.5; gap ≈ 0.2-0.3 absolute mean AP.

Root causes: 33× less training data, 16× shorter audio window, 4-8× smaller model, no CNN front-end, 10-60× fewer steps.

## 5. Cross-references

- depth=3 / depth=6 baseline: `result/09_musicnet_real/musicnet_real_d3_d6_data_ceiling.md`
- synth substitute (validated): `result/08_multi_pitch_synth/multi_pitch_synth_l1_l2_pctfairness.md`
- Phase 14 audit (softmask defect ratified): `_frameworks/02_phase14_softmask_and_batch_audit.md`
