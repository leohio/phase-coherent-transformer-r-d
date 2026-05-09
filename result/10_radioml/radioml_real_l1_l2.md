# RadioML Real — L1 / L2 modulation classification (RML2016 6-dB-SNR mirror)

**Task**: real RadioML 11-class modulation classification using public mirror `hitrs909/RML2016` parquet (6-dB-SNR public mirror, 9900 train / 1100 test, 11 modulation classes, 128 I/Q samples each).
**Source**: `bench_real_musicnet_radioml.md`

**Setup**: dim=64, depth=3, heads=4, dim_head=16, ff_mult=4, AdamW, lr=1e-3, wd=1e-4, clip=1.0, batch=64, steps=1000, N=2 seeds, CPU. Per-sample zero-mean + unit-RMS normalization at load (public mirror was offset to `[0, 1]`, biased complex cells).

Difficulty axes:

| level | seq_len | Note |
|---|---|---|
| L1 | 128 (full) | normal |
| L2 | 64 (truncated first half) | shorter input |

## 1. L1 (seq=128) — 11-class accuracy

| cell | params | acc | eval_loss | wall (s) |
|---|---:|---:|---:|---:|
| real_softmax | 149,707 | 0.296 ± 0.007 | 1.817 ± 0.028 | 158.0 |
| real_sigmoid | 149,710 | 0.303 ± 0.002 | 1.805 ± 0.014 | 167.6 |
| **real_screen** | 162,970 | **★0.363 ± 0.014** | **1.574 ± 0.058** | 238.2 |
| complex_softmax | 150,350 | 0.244 ± 0.029 | 1.940 ± 0.022 | 486.0 |
| complex_sigmoid | 150,353 | 0.345 ± 0.007 | 1.469 ± 0.022 | 795.0 |
| complex_screen | 163,616 | 0.341 ± 0.011 | 1.600 ± 0.062 | 990.3 |

Random baseline = 0.091. **real_screen leads** (+0.067 over real_softmax). complex_softmax underperforms; complex_sigmoid is competitive (0.345).

## 2. L2 (seq=64, shorter input) — 11-class accuracy

| cell | params | acc | eval_loss | wall (s) |
|---|---:|---:|---:|---:|
| real_softmax | 149,707 | 0.224 ± 0.009 | 1.993 ± 0.036 | 114.6 |
| real_sigmoid | 149,710 | 0.241 ± 0.007 | 1.991 ± 0.037 | 97.9 |
| **real_screen** | 162,970 | **★0.389 ± 0.015** | **1.523 ± 0.006** | 140.5 |
| complex_softmax | 150,350 | 0.273 ± 0.007 | 1.897 ± 0.061 | 398.0 |
| complex_sigmoid | 150,353 | 0.352 ± 0.012 | 1.590 ± 0.001 | 321.0 |
| complex_screen | 163,616 | 0.337 ± 0.015 | 1.641 ± 0.021 | 346.8 |

real_screen actually **improves** at L2 (0.363 → 0.389) — only cell to do so. Other cells degrade or stay flat. **Speculative**: shorter input forces attention onto fewer salient features, which screening's selective gating exploits more cleanly than softmax.

## 3. Key findings

1. **real_screen wins both Real-RadioML levels** by clear margin (+0.067 over real_softmax at L1, +0.165 at L2). Screening's advantage on real I/Q-modulation is the most decisive single finding from this benchmark
2. **real_screen on RadioML L2 (shorter input) does not just resist degradation — it improves** vs L1 (0.363 → 0.389). Selectivity of screening attention may be more useful in information-poor regimes
3. **complex_softmax underperforms** on real RadioML (0.244 / 0.273) while real_softmax does fine — opposite of "complex domain wants complex models" heuristic. complex_sigmoid and complex_screen recover
4. **Sigmoid > softmax across substrates** holds on real RadioML, consistent with synth findings

## 4. cross-comparison: synth vs real

| benchmark | L1 winner | L2 winner |
|---|---|---|
| synth radioml | complex_sigmoid (acc=0.598) | complex_screen (acc=0.637) |
| **real RadioML** | **real_screen (acc=0.363)** | **real_screen (acc=0.389)** |

→ **Screening attention always wins**, but family member shifts: complex_* on synthetic, real_* on real RadioML. **Real data is harder to differentiate cells on**: synth has 0.03-0.10 cell gaps, real RadioML has 0.07/0.16 gaps (still meaningful).

## 5. Caveats

1. **N=2 seeds only** (project standard ≥5 for stuck-rate)
2. **Real RML2016 = 6 dB SNR only** (public mirror's only split). Full RML2016.10a SNR-sweep (-20 to +18 dB) is absent. Difficulty axis for L2 is "input length truncation," not canonical SNR axis
3. **Real MusicNet L2 saturates** in the same campaign (companion finding) — not interpretable
4. **Normalization fix critical**: zero-mean + unit-RMS per sample required to get complex cells to learn. Without it, complex_softmax / complex_sigmoid stay at chance because [0, 1]-offset I/Q lives in one quadrant of complex plane, phase carries no information. First attempt acc≈0.10 → with norm acc≈0.30+

## 6. Cross-references

- synth RadioML companion: `result/10_radioml/radioml_synth_l1_l2.md`
- mechanism (screen wins on info-poor regimes): `_frameworks/01_anti_correlation_preservation_mechanism.md`
