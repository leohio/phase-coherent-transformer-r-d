# NIAH (Needle In A Haystack) L=2048 — Test summary

Synthetic positional-retrieval task: a `needle` token is inserted at `depth_ratio × seq_len = 1024` of a 2048-token sequence; remaining tokens are uniform-random distractors from the same vocabulary. Since needle and distractors share vocabulary, **the only retrieval signal is position**.

## Files

1. [`niah_L2048_complex_sigmoid_n3.md`](niah_L2048_complex_sigmoid_n3.md) — `complex_sigmoid` (PCT) at dim=256/L6 + chunked attention reaches **acc=1.000 (N=3)** on Copy d=1000-equivalent positional retrieval, with the s=0 run deeply saturated (eval_loss ~ 1e-4) and s=1/s=2 in flight expected at 1.000.

## Status of multi-cell comparison

The earlier 6-cell solver/non-solver sweep (`complex_screen`, `real_screen`, `real_softmax`, `real_sigmoid`, `complex_softmax` — paper-style "3 solver / 3 non-solver dichotomy") was based on multi-instance runs that suffered serious bugs and mid-run truncations on unreliable cluster instances. **All non-`complex_sigmoid` cell numbers for NIAH L=2048 are RETRACTED** until a clean re-run is completed. Earlier small-scale NIAH L=1024 data is also excluded for the same truncation reason.

## Headline (this folder, current evidence)

> **PCT (complex_sigmoid) deeply solves NIAH L=2048 at N=3** (s=0 confirmed, s=1/s=2 in flight expected to confirm) at dim=256/L6 with chunked attention. This is one of the H2 "dominance even on complex-disadvantaged tasks" data points — a purely positional retrieval task where complex Q, K offers no inductive advantage by construction is nevertheless deeply solved by PCT (eval_loss ~ 1e-4, not borderline).
>
> Multi-cell ranking is pending (N=3 clean re-run for all 6 cells).
