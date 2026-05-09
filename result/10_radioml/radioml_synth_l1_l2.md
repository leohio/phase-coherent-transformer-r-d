# RadioML Synthetic — L1 / L2 modulation classification

**Task**: closed-form I/Q signal generation for 8 modulation classes (BPSK, QPSK, 8PSK, QAM16, PAM4, GFSK, CPFSK, AM-DSB) with carrier offset + AWGN, 8-class CE.
**Source**: `bench_musicnet_radioml.md` L1 + L2

**Setup (both levels)**:
- dim=64, depth=3, heads=4, dim_head=16, ff_mult=4
- AdamW, lr=1e-3, wd=1e-4, clip=1.0, batch=64, steps=1500
- N=2 seeds (sub-stuck-detection — Project standard is N≥5)
- CPU (14 cores)
- Loader: `complex_nn_experiment/data/radioml_synth.py`

Difficulty axis: SNR range
- L1: snr_db ∈ [6, 18] (easy)
- L2: snr_db ∈ [0, 12] (harder, lower SNR)

## 1. L1 results — 8-class accuracy (snr ∈ [6, 18])

| cell | params | acc | eval_loss | wall (s) |
|---|---:|---:|---:|---:|
| real_softmax | 149,512 | 0.357 ± 0.012 | 1.411 ± 0.048 | 257.5 |
| real_sigmoid | 149,515 | 0.361 ± 0.004 | 1.366 ± 0.013 | 282.5 |
| real_screen | 162,775 | 0.555 ± 0.174 | 1.135 ± 0.214 | 381.5 |
| complex_softmax | 149,963 | 0.434 ± 0.051 | 1.396 ± 0.122 | 802.6 |
| **complex_sigmoid** | 149,966 | **★0.598 ± 0.051** | **0.824 ± 0.046** | 1235.1 |
| complex_screen | 163,229 | 0.427 ± 0.122 | 1.256 ± 0.182 | 1453.2 |

Random-baseline floor acc = 1/8 = 0.125.

**Observations**:
- **complex_sigmoid wins L1** (0.598 ± 0.051), complex_screen underperforms with high seed variance (0.549 / 0.305)
- real_screen shows project-memory-documented "stuck-seed" split (0.381 / 0.729) — single seed pulls the mean up
- **Complex >> real on radioml** (best complex 0.598 vs best non-screen real 0.361): I/Q substrate is the natural domain, complex cells exploit it
- Sigmoid > softmax in both substrates

## 2. L2 results — 8-class accuracy (snr ∈ [0, 12], harder)

| cell | params | acc | eval_loss | wall (s) |
|---|---:|---:|---:|---:|
| real_softmax | 149,512 | 0.265 ± 0.013 | 1.760 ± 0.100 | 250.5 |
| real_sigmoid | 149,515 | 0.276 ± 0.003 | 1.741 ± 0.084 | 261.8 |
| real_screen | 162,775 | 0.615 ± 0.047 | 1.182 ± 0.142 | 361.6 |
| complex_softmax | 149,963 | 0.413 ± 0.017 | 1.483 ± 0.019 | 754.6 |
| complex_sigmoid | 149,966 | 0.591 ± 0.052 | 1.015 ± 0.096 | 1196.6 |
| **complex_screen** | 163,229 | **★0.637 ± 0.016** | **0.985 ± 0.035** | 1409.0 |

**Observations**:
- **complex_screen wins L2** (0.637) — was joint-winner / runner-up at L1
- **L2 separation between cell families widens**:
 - real_softmax/sigmoid collapse to ≈0.27 (3 SNR-impaired classes barely above chance)
 - *_screen and complex_sigmoid stay above 0.59
- **Stuck-seed pattern attenuated at L2**: real_screen radioml L1 had a stuck/unstuck split (0.381 / 0.729). At L2 both seeds learn (0.568 / 0.662) — lower SNR forces both seeds out of easy-confidence regime

## 3. L1 → L2 movement

| metric | L1 winner | L2 winner |
|---|---|---|
| radioml | complex_sigmoid (0.598) | **complex_screen (0.637)** |

→ At harder difficulty (lower SNR), complex_screen overtakes; selectivity becomes effective in signal-poor regimes.

## 4. Caveats

- N=2 seeds is sub-stuck-detection
- Synthetic RadioML uses fixed-form modulation generators rather than DeepSig's channel-impaired RML2016.10a; lower-difficulty than the published dataset

## 5. Cross-references

- Real RadioML companion: `result/10_radioml/radioml_real_l1_l2.md`
- mechanism (screen vs sigmoid trade-off): `_frameworks/01_anti_correlation_preservation_mechanism.md`
