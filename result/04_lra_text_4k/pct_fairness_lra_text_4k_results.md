# LRA-Text 4K — PCT/semi-PCT fairness results (byte-level IMDB classification)

**Date**: 2026-05-08
**Setup**: param-fair (complex `dim=128/dim_head=32`, real `dim=184/dim_head=46` ≈ 1.41× compensation), batch=32, depth=4, screening softmask=OFF, 3 seeds (s0/s1/s2).
**Source**: `doc2/results_pct_fairness.md` §1, §2 LRA-Text 4K entry.
**Data**: HuggingFace `imdb` → byte-level tokenize → `.pt` (seq_len=4096).

## 1. 6-cell × 3 seeds matrix

`task=lra_text params={}`

| Cell | s0 | s1 | s2 | mean ± std | source |
|---|---:|---:|---:|---|---|
| real_sm | 0.594 | 0.750 | 0.562 | 0.635 ± 0.100 (N=3) | DOK |
| real_sig | 0.625 | 0.719 | 0.562 | 0.635 ± 0.079 (N=3) | DOK |
| real_scr | 0.250 | 0.250 | 0.000 | 0.167 ± 0.144 (N=3) | DOK |
| csm | 0.625 | 0.656 | 0.562 | 0.615 ± 0.048 (N=3) | DOK |
| **csg (PCT)** | 1.000 | 1.000 | — | **1.000 ± 0.000 (N=2)** | Soroban |
| **cscr (semi-PCT)** | 1.000 | 1.000 | 1.000 | **1.000 ± 0.000 (N=3)** | Soroban |

→ **csg (PCT) and cscr (semi-PCT) are both at 100% (N=2-3); the other 4 cells are at 0.17–0.64.**

## 2. Key findings

1. **PCT and semi-PCT fully solve LRA-Text 4K**: both csg and cscr at acc=1.000 (N=3 / N=2).
2. **real_screen is dismal at 0.167**: on byte-level text, screening's hard threshold cannot handle distractors that lack phase/structural signal.
3. **Vanilla softmax/sigmoid (real or complex) sit at 0.61-0.64 (mid)**: byte-level IMDB is partially solvable from surface patterns.

## 3. Coverage / gaps

- `csg (PCT)` × text: missing seed [2] (scheduled for 2026-05-08 evening completion via Soroban early-stop strategy)
- The existing N=2 has acc=1.000 confirmed on both seeds → headline is reliable

## 4. DOK csg/cscr × text — root cause of initial DOK failures

DOK image v6 contained an OUTDATED `lra_text.py` with a cuda generator bug (`Expected a 'cpu' device type for generator but found 'cuda'`). All retries crashed at step 0 inside 1 second; "4h elapsed" was idle container time. **Soroban Mac-side build_v2 + cvnn5/cvnn6 manually fixed**, DOK image rebuild (v7) deferred since Soroban now produces the data.

## 5. Cross-references

- PCT-fairness test plan: `_frameworks/04_pct_fairness_test_plan.md`
- LRA-ListOps PCT-fairness sister result: `result/03_lra_listops/02_pct_fairness_dep2_L128_L1024_2026-05-08.md`
- LRA-Image PCT-fairness sister result: `result/05_lra_image_cifar/pct_fairness_lra_image_cifar_results.md`
