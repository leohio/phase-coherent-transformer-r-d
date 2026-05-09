# Copy Memory — Test summaries

Synthetic associative-recall benchmark. K source-target pairs at the start of the sequence are followed by a query token; the model must produce the target. Distractor blank tokens between source and query control the **delay (`d`)**.

## Files (chronological + by scale)

1. [`01_small_scale_phase8_d100_to_d1000_dim64.md`](01_small_scale_phase8_d100_to_d1000_dim64.md) — Phase 8 / 9 / 10 / 11 small-scale (dim=64 batch=64) baseline. Discovery: **complex_sigmoid uniquely solves d=1000 at 100%**, others at random. Screening cells run with softmask=ON (later defect, not corrected here).
2. [`02_small_scale_phase14_audit_corrected_dim32_batch256.md`](02_small_scale_phase14_audit_corrected_dim32_batch256.md) — Phase 14 audit: batch corrected to 256, softmask defect identified. **complex_sigmoid uniquely LR-robust** (100% at d=1000 across all LRs).
3. [`03_mid_scale_dim256_d500_d2000_d5000.md`](03_mid_scale_dim256_d500_d2000_d5000.md) — Mid-scale (dim=256 depth=6) results. complex_sigmoid is the only cell solving long-range Copy across batch ∈ {8, 32, 256}. real_screen also solves d=2000 at 1.000 (N=3) under batch=32.
4. [`04_pct_fairness_d1000_small_d2000_mid_2026-05-08.md`](04_pct_fairness_d1000_small_d2000_mid_2026-05-08.md) — PCT/semi-PCT fairness rerun (real param 1.41× compensated). **PCT (csg) and real_screen are dual winners on mid-scale d=2000**; semi-PCT (cscr) shows 1/3 stuck-seed pathology.
5. [`05_tanh1_vs_sigmoid_long_range_pathology.md`](05_tanh1_vs_sigmoid_long_range_pathology.md) — `tanh+1` vs `sigmoid` ablation. Within the same condition family, **a too-steep gradient produces stuck-seed at long range**. Evidence that complex_sigmoid sits in the sweet spot.

## Headline finding (across all files)

> **complex_sigmoid (PCT) is the only cell that solves long-range Copy Memory (d=500-5000) at 100% across all batch / all LR.** At mid-scale (dim=256), real_screen is also a dual winner on d=2000; the rest (real_softmax / real_sigmoid / complex_softmax) are capacity-bound + structural failure.
