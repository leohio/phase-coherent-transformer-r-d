# RadioML — Test summary

I/Q-domain modulation classification benchmark. Two axes: synthetic (closed-form 8 modulations) and real (RML2016 6dB-SNR public mirror, 11 modulations).

## Files

1. [`radioml_synth_l1_l2.md`](radioml_synth_l1_l2.md) — Synthetic 8-class, L1 (snr ∈ [6,18]) / L2 (snr ∈ [0,12]). **complex_sigmoid is single 1st on L1 (0.598); complex_screen is single 1st on L2 (0.637)** — at the harder difficulty, screen overtakes.
2. [`radioml_real_l1_l2.md`](radioml_real_l1_l2.md) — Real RML2016 11-class, L1 (seq=128) / L2 (seq=64). **real_screen is single winner at both levels** (0.363 / 0.389); it is the only cell that improves at L2.

## Headline

> **Screening attention always wins on I/Q-modulation tasks** (synth: complex_*; real: real_*). **Real data is harder to differentiate cells on** — synth has 0.03-0.10 cell gaps; real has 0.07-0.16 cell gaps (still meaningful). **Selectivity is effective in signal-poor regimes** (real_screen is the only cell that improves at L2).
