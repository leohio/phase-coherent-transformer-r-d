# MusicNet Real — Test summary

Real MusicNet (10 test pieces, ~250 MB) — 6-cell × multi-label pitch detection. The true MusicNet midscale of the Phase 8 plan (dim=256, depth=6) is out of scope here; what is recorded here is small-scale CPU-environment exploration.

## Files

1. [`musicnet_real_d3_d6_data_ceiling.md`](musicnet_real_d3_d6_data_ceiling.md) — depth=3 / depth=6 results. **complex_screen is nominally 1st on L1 (F1=0.218 ± 0.010)** but the spread is 0.02 — within seed variance. **L2 saturates across all cells at F1=0.143** (the 88-class regime is hard to discriminate at small scale). Larger scale (full MusicNet from Zenodo / mid-scale GPU) is required.
2. [`musicnet_real_2026-05_campaign.md`](musicnet_real_2026-05_campaign.md) — May 2026 CPU campaign (the "softmask=ON kills *_screen" finding, the real param-fair sweep across 7 tasks, and absolute F1 baseline comparisons).

## Headline

> **Real MusicNet at small scale (dim=64, 10-piece test split) is data-ceiling-bound** — F1 spread of 0.02-0.07 makes cell discrimination hard; L2 88-class saturates across all cells at F1=0.143. **The synth `multi_pitch` is the discriminative substitute at small scale.** Meaningful cell comparison on real MusicNet requires the full Zenodo MusicNet (11 GB, 330 pieces) + dim≥128 + a CNN front-end (Phase 8 mid-scale scope).
