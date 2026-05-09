# LRA-ListOps — Test summaries

LRA tree-structured operation evaluation task. Uses a synthetic generator (variable depth, max_args, max_seq_len) to evaluate nested MAX/MIN/MED/SM operator expressions at the CLS-position. A content-driven algorithmic retrieval task.

## Scope of this folder

**Excluded**: LRA-ListOps L=1024 dep=12 (planned mid-scale extension, never actually executed — `sb_listops_L1024_d12_*` Soroban jobs ran with `max_depth=2` due to yaml config bug; merged into the dep=2 row. **True dep=12 data does not exist** per `doc2/results_pct_fairness.md` Status note).

## Files

1. [`01_small_scale_easy_medium_6cell_2026-05-05.md`](01_small_scale_easy_medium_6cell_2026-05-05.md) — CPU small-scale 6-cell × 2 difficulty (easy / medium), N=1. At easy, the 3 complex cells (csm/csg/cscr) all reach 1.000; at medium, **complex_sigmoid 0.7188 is single 1st**.
2. [`02_pct_fairness_dep2_L128_L1024_2026-05-08.md`](02_pct_fairness_dep2_L128_L1024_2026-05-08.md) — PCT-fairness rerun (real param 1.41× compensated, N=3). **PCT (csg) 0.854 single 1st**; semi-PCT (cscr) 0.833 2nd; real_screen 0.698 3rd.
3. [`03_apple_silicon_mps_complex_sigmoid_plateau.md`](03_apple_silicon_mps_complex_sigmoid_plateau.md) — Deep dive into `complex_sigmoid` on Apple Silicon MPS: Bayesian hypothesis update of the ~0.55 plateau across 7 arms. `eval_acc=1` is unreachable in the explored range; consistent with the literature ceiling for attention-only architectures.
4. [`04_depth_scaling_d2_to_d20_complex_sigmoid_n1.md`](04_depth_scaling_d2_to_d20_complex_sigmoid_n1.md) — **Depth scaling sweep** (d ∈ {2, 4, 6, 10, 14, 20}, N=1, dim=128, batch=32, 30K steps, DOK h100). **No depth-related accuracy collapse** across a 10× param range (0.40 M → 3.96 M). best_acc plateaus at 13/16 = 0.8125 lucky-batch saturation on 4/6 points; the data alone cannot establish a scaling-law claim. n=2048 stable eval + N=3 are pending.

## Headline findings

> **PCT (complex_sigmoid) is 1st place on multiple LRA-ListOps medium-difficulty setups** (param-fair N=3 at 0.854), nearly tied with semi-PCT (complex_screen). On the real side, real_screen leads. Vanilla softmax (real or complex) is at chance / near-chance without LR/warmup tuning.
>
> **Depth scaling**: PCT shows no accuracy collapse across depths d ∈ {2, 4, 6, 10, 14, 20} (10× param range). **A positive power-law scaling claim cannot be made from this data alone**, but the qualitative claim that **"the conventional concern that complex transformers fail to scale to deeper architectures is not supported within this sweep"** does hold.
