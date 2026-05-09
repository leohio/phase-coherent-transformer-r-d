# Multi-Pitch (Synthetic) — Test summary

Synthetic multi-label pitch detection task (log-spaced 100..800 Hz pitch bank; sum pure tones with random phase → 1D rFFT → multi-label BCE). Adopted in the Phase 8 plan as the "MusicNet substitute" because real MusicNet at small scale saturates and `multi_pitch` produces actual cell discrimination at small scale.

## Files

1. [`multi_pitch_synth_l1_l2_pctfairness.md`](multi_pitch_synth_l1_l2_pctfairness.md) — Phase 11 small-scale L1/L2 (synth, N=2, sm=ON) + softmask=OFF rerun + PCT-fairness rerun (dim=128, N=3). **At PCT-fair, semi-PCT (cscr) and PCT (csg) are tied 1st** (1.000 / 0.997 N=3); **real_screen 0.949 is 3rd** (preserved under param-fair).

## Headline

> **On multi-source pitch retrieval, screening's hard threshold is effective for multi-pitch identification** — semi-PCT (complex_screen) is the single 1st place at PCT-fair (1.000 N=3); PCT (complex_sigmoid) is a close 2nd at 0.997 N=3. real_sm/sig sit at 0.82 (near collapse); real_screen at 0.95 N=3 is the only real-side cell above 0.94.
