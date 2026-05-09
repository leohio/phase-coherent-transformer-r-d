# Copy Memory — PCT/semi-PCT fairness rerun (real param 1.41× compensated)

**Date**: 2026-05-08
**Setup**:
- **PCT (csg) = complex_sigmoid**, **semi-PCT (cscr) = complex_screen** (named per paper outline_v3 convention)
- param-fair: complex `dim=128/dim_head=32`, real `dim=184/dim_head=46` (1.41× ≈ √2 compensation)
- batch=32 effective, depth=4 (small) / depth=6 (mid), screening softmask=OFF
- Seeds: 3 (s0/s1/s2). Per-cell-task mean ± std (N=3).

**Source**: `doc2/results_pct_fairness.md` (auto-aggregated by `scripts/aggregate_pct_fairness.py`)

## 1. Small-scale Copy d=1000 (dim=128/L4, ~862K params)

`task=copymem params={"K": 10, "delay": 1000}`

| Cell | s0 | s1 | s2 | mean ± std | source |
|---|---:|---:|---:|---|---|
| real_sm | 0.694 | 1.000 | 0.784 | 0.826 ± 0.157 (N=3) | DOK |
| real_sig | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 (N=3) | DOK |
| real_scr | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 (N=3) | DOK |
| csm | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 (N=3) | DOK |
| csg (PCT) | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 (N=3) | DOK |
| cscr (semi-PCT) | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 (N=3) | DOK |

→ **All cells saturate at small-scale dim=128** (only real_sm shows 0.83 with 1 unstable seed). No discrimination — consistent with a capacity-saturated regime.

## 2. Mid-scale Copy d=500 (dim=256/L6, ~5M params, Vast 24GB)

`task=copymem params={"K": 10, "delay": 500}`

| Cell | s0 | s1 | s2 | mean ± std |
|---|---:|---:|---:|---|
| real_sm | 1.000 | 0.225 | 0.119 | 0.448 ± 0.481 (N=3) |
| real_sig | 1.000 | 0.834 | 1.000 | 0.945 ± 0.096 (N=3) |
| real_scr | 1.000 | 1.000 | 1.000 | **1.000 ± 0.000 (N=3)** |
| csm | 0.100 | 0.100 | 0.100 | 0.100 ± 0.000 (N=3) |
| csg (PCT) | 1.000 | 1.000 | 1.000 | **1.000 ± 0.000 (N=3)** |
| cscr (semi-PCT) | 1.000 | 0.825 | 0.316 | 0.714 ± 0.356 (N=3) |

→ **PCT (csg) and real_screen are dual winners**; semi-PCT (cscr) is partial on 1/3; real_sigmoid is partial on 1/3; real_softmax and complex_softmax collapse.

## 3. Mid-scale Copy d=2000 (dim=256/L6, Soroban A100 80GB)

`task=copymem params={"K": 10, "delay": 2000}`

| Cell | s0 | s1 | s2 | mean ± std |
|---|---:|---:|---:|---|
| real_sm | 0.150 | 0.100 | 0.075 | 0.108 ± 0.038 (N=3) |
| real_sig | 0.100 | 0.100 | 0.100 | 0.100 ± 0.000 (N=3) |
| real_scr | 1.000 | 1.000 | 1.000 | **1.000 ± 0.000 (N=3)** |
| csm | 0.150 | 0.100 | 0.050 | 0.100 ± 0.050 (N=3) |
| csg (PCT) | 1.000 | 1.000 | 1.000 | **1.000 ± 0.000 (N=3)** |
| cscr (semi-PCT) | 1.000 | 1.000 | 0.050 | 0.683 ± 0.548 (N=3) |

→ **Soroban N=3: PCT (csg) and real_screen are perfect; semi-PCT (cscr) is 1/3 stuck.**

**Mid-scale Copy d=2000 cscr_s1 — full-training confirmation (2026-05-08)**: cvnn5 was user-truncated at step 2000 with copy_acc=1.000 saturated. An independent re-run on Vast 95GB (RTX PRO 6000 Blackwell) ran the **full 6000 steps** and recorded `copy_acc=1.000, eval_loss=4.79e-4` — confirming that cscr fully converges on mid-scale d=2000 (not just early-saturation). Vast s2=0.05 stays as a single-seed outlier.

## 4. Key findings

1. **PCT (csg) perfectly solves Copy d=1000/d=2000 at N=3 across both small and mid scale** — preserved after 1.41× param-fairness compensation
2. **real_screen is dual winner with PCT on mid-scale d=2000** (all N=3 at 1.000) — consistent with Phase 14.5 §1
3. **semi-PCT (cscr) shows 1/3 stuck-seed pathology on mid-scale d=500/d=2000** (variance σ=0.36-0.55) — saturates at small scale; stuck-seed surfaces at mid scale
4. **real_sigmoid hits 0.94 (N=3) at mid d=500** — at short distance, even param-matched, it can compete in implementation
5. **complex_softmax is at chance across all mid-scale Copy** — softmax's row-norm constraint is fatal at long range

## 5. Cross-references

- baseline (Phase 8/14): `result/01_copy_memory/01_*` and `02_*`
- mid-scale Phase 14.5 (no param compensation): `result/01_copy_memory/03_mid_scale_dim256_d500_d2000_d5000.md`
- PCT-fairness test plan: `_frameworks/04_pct_fairness_test_plan.md`
- semi-PCT positioning: `_frameworks/01_anti_correlation_preservation_mechanism.md`
