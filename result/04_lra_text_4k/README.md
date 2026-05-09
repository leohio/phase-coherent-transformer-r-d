# LRA-Text 4K — Test summary

LRA byte-level IMDB classification at sequence length 4096. PCT fairness benchmark task (Tier B; requires `.pt` data prep + Docker image rebuild).

## Files

1. [`pct_fairness_lra_text_4k_results.md`](pct_fairness_lra_text_4k_results.md) — 6-cell × 3 seeds, real param 1.41× compensated. **PCT (csg) and semi-PCT (cscr) both at acc=1.000**; real_screen at 0.167 (catastrophic); vanilla softmax/sigmoid at 0.61-0.64 (mid).

## Headline

> **Complex per-pair sigmoid or screening fully solves byte-level text** (PCT and semi-PCT both 1.000). real_screen is structurally unsuited to this task (0.17). On the real side, vanilla softmax/sigmoid actually behave more normally.
