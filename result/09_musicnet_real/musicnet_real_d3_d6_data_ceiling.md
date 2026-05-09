# MusicNet (Real) — depth=3 / depth=6, L1/L2 (data ceiling)

**Status**: After running the real-data follow-up, the user retracted the request to use real MusicNet ("we wanted MusicNet, not multi_pitch"): real MusicNet at the same small scale produces no inter-cell discrimination at L2 (all six cells F1=0.143) and only 0.02 F1 spread at L1. The synth `multi_pitch` task ([result/08_multi_pitch_synth/](../08_multi_pitch_synth/)) is the more discriminative MusicNet stand-in for small-scale work, consistent with the Phase 8 plan's positioning.

**Source**: `bench_real_musicnet_radioml.md` + `musicnet_all_results.md` + `musicnet_and_multi_pitch_bench_2026-05.md`

## 1. Dataset

- **Real MusicNet**: 10 test pieces from `DreamyWanderer/MusicNet` on HuggingFace (~250 MB WAV + CSV labels). Audio decimated to 11025 Hz, random 256-sample windows → 1D rFFT (seq_len=129 complex bins). Multi-label output: 1 if MIDI note is active in the window. Loader: `complex_nn_experiment/data/musicnet_real.py`.
- 10 piece IDs: 1759, 1819, 2106, 2191, 2298, 2303, 2382, 2416, 2556, 2628 (gitignored under `datasets/musicnet/`).

Difficulty axes:

| level | MIDI range | # notes |
|---|---|---|
| L1 | 60..80 | 21 notes (mid-keyboard range) |
| L2 | 21..108 | 88 notes (full piano) |

## 2. depth=3 results (1000 steps, batch=64, N=2 seeds)

### Real MusicNet L1 (21 notes) — F1

| cell | params | F1 | ham_acc | eval_loss | wall (s) |
|---|---:|---:|---:|---:|---:|
| real_softmax | 150,357 | 0.201 ± 0.012 | 0.738 ± 0.003 | 1.116 ± 0.004 | 153.6 |
| real_sigmoid | 150,360 | 0.201 ± 0.012 | 0.738 ± 0.003 | 1.116 ± 0.004 | 164.5 |
| real_screen | 163,620 | 0.214 ± 0.007 | 0.724 ± 0.002 | 1.099 ± 0.003 | 231.3 |
| complex_softmax | 151,640 | 0.201 ± 0.014 | 0.737 ± 0.002 | 1.116 ± 0.004 | 497.8 |
| complex_sigmoid | 151,643 | 0.203 ± 0.013 | 0.742 ± 0.006 | 1.114 ± 0.004 | 712.3 |
| **complex_screen** | 164,906 | **★0.218 ± 0.010** | **0.743 ± 0.016** | **1.099 ± 0.002** | 872.4 |

→ complex_screen is the nominal winner, but the spread of 0.02 F1 is within seed variance. Marginally informative.

### Real MusicNet L2 (88 notes) — F1

```
All 6 cells fully saturate at F1 = 0.143 ± 0.001 (identification floor at this scale)
```

→ The combination of 88 classes / pos_weight cap=20 (vs natural ratio ~35) / model size saturates. **Treat as a ceiling artifact, not a comparative result**. A larger model or a focal-loss objective is needed.

## 3. depth=6 results (truncated, 500 steps, batch=64)

Both L1 and L2 show **exactly the same saturation pattern** — in fact complex_screen regresses on L1 from 0.218 (depth=3) to 0.187 (possibly insufficient steps). **Increasing depth does not help.**

The bottleneck is not depth but:
1. **Representation capacity at dim=64**: too small for 88-class multi-label
2. **pos_weight cap=20 vs natural ratio ~35** mismatch (L2)
3. **MusicNet's label structure**: per-piece pitch range is biased; a short window (256 samples ≈ 23 ms) gives weak fundamental-frequency analysis

## 4. May 2026 8K-step bench (softmask=OFF, dim=80 real / dim=64 complex)

```
 musicnet L1 (F1) musicnet L2 (F1)
real_sm 0.185 0.144
real_sig 0.185 0.144
real_scr 0.190 0.145
csm 0.185 0.142
csg 0.201 0.142
cscr 0.192 0.143
```

→ `complex_sigmoid` is at 0.201 on L1; the other cells are 0.185-0.192 (spread ~0.02). L2 fully saturates across all cells at 0.142-0.145. Reconfirms that cell discrimination is hard at small scale + 10-piece data.

## 5. Cross-comparison: synth vs real, L1 vs L2

| benchmark | L1 winner | L2 winner |
|---|---|---|
| synth multi_pitch (≈MusicNet substitute) | complex_screen (F1=0.710) | complex_screen (F1=0.511) |
| **real MusicNet** | complex_screen (F1=0.218) | (saturated, F1≈0.143) |

→ **At small scale, the synth multi_pitch is the discriminative MusicNet stand-in**. Real MusicNet at this scale is data-ceiling-bound (10 pieces × 7 min ≈ 10K unique 256-sample windows; batch=4 × 16K steps = 64K examples → 6× revisits, past the memorization threshold).

## 6. Followups (deferred, would need GPU or fundamentally different recipe)

1. softmask=OFF copy_d=500/1000/2000/5000 — confirm scaling
2. **Full MusicNet (Zenodo, 11 GB, 330 pieces)** — 33× more data
3. Window=4096+ STFT-frame-sequence input — attention over time
4. mean AP metric (instead of F1@0.5) — comparable to literature (Trabelsi 2018 / Cwitkowitz 2019, mean AP ≈ 0.65-0.78)
5. CNN front-end + transformer hybrid — closer to Trabelsi/Cwitkowitz recipes

## 7. Cross-references

- synth multi_pitch (validated MusicNet substitute): `result/08_multi_pitch_synth/multi_pitch_synth_l1_l2_pctfairness.md`
- 2026-05 campaign details: `result/09_musicnet_real/musicnet_real_2026-05_campaign.md`
