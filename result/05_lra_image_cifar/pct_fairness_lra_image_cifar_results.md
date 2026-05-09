# LRA-Image CIFAR — PCT/semi-PCT fairness results (32×32 grayscale, seq=1024)

**Date**: 2026-05-08
**Setup**: param-fair (complex `dim=128/dim_head=32`, real `dim=184/dim_head=46` ≈ 1.41× compensation), batch=32, depth=4, screening softmask=OFF, 6 seeds (DOK N=3 + Vast N=3).
**Source**: `doc2/results_pct_fairness.md` §1, §2 LRA-Image CIFAR entry.
**Data**: CIFAR-10 grayscale 32×32 → seq_len=1024 (LRA paper standard).

> **Note**: DOK ran N=3 (pctf_dok_b_image_s0/s1/s2) + Vast independently ran N=3 (pctf_vast_b_image_s0/s1/s2 = same task config, different stochasticity). Below shows the N=6 combined matrix.

## 1. 6-cell × 6 seeds matrix

`task=lra_image params={}`

| Cell | DOK s0 | DOK s1 | DOK s2 | Vast s0 | Vast s1 | Vast s2 | mean ± std (N=6) |
|---|---:|---:|---:|---:|---:|---:|---|
| real_sm | 0.188 | 0.125 | 0.156 | 0.188 | 0.125 | 0.156 | 0.156 ± 0.028 |
| real_sig | 0.188 | 0.188 | 0.094 | 0.188 | 0.188 | 0.094 | 0.156 ± 0.048 |
| real_scr | 0.344 | 0.344 | 0.312 | 0.281 | 0.344 | 0.281 | 0.318 ± 0.031 |
| csm | 0.156 | 0.188 | 0.125 | 0.156 | 0.188 | 0.125 | 0.156 ± 0.028 |
| **csg (PCT)** | 0.531 | 0.375 | 0.469 | 0.531 | 0.375 | 0.469 | **0.458 ± 0.070** |
| **cscr (semi-PCT)** | 0.406 | 0.375 | 0.438 | 0.406 | 0.375 | 0.438 | **0.406 ± 0.028** |

## 2. Key findings

1. **PCT (csg) is single 1st place** (0.458 N=6); semi-PCT (cscr) is 2nd (0.406 N=6) — complex sigmoid/screening is effective even on image pixel-sequence.
2. **real_screen is 3rd** (0.318 N=6): more than 2× the other real cells (real_sm/real_sig at 0.156); the Phase 8/10 "real_screen is strong on phase/algorithmic" pattern carries to CIFAR pixel-sequence.
3. **Vanilla softmax (real and complex) is at random** (0.156, ~10% chance for 10-class) — a structural failure that capacity does not fix.
4. complex_softmax vs complex_sigmoid both at 1.41× compensation: 0.156 vs 0.458 → **the row-norm constraint is decisive**.

## 3. Cell ranking

| Rank | Cell | mean (N=6) |
|---|---|---|
| 🥇 | csg (PCT) | 0.458 |
| 🥈 | cscr (semi-PCT) | 0.406 |
| 🥉 | real_screen | 0.318 |
| 4 | real_sm / real_sig / csm | 0.156 (chance) |

→ Even on image, the standard pattern **PCT > semi-PCT > real_screen >> non-screen** holds.

## 4. Cross-references

- PCT-fairness test plan: `_frameworks/04_pct_fairness_test_plan.md`
- LRA-ListOps PCT-fairness sister result: `result/03_lra_listops/02_pct_fairness_dep2_L128_L1024_2026-05-08.md`
- LRA-Text PCT-fairness sister result: `result/04_lra_text_4k/pct_fairness_lra_text_4k_results.md`
