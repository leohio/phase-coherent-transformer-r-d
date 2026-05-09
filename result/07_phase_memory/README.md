# Phase Memory — Test summary

K=5 source positions (each carrying a phase value) followed by a query specifying which source's phase to return. A retrieval task with **phase as identifier** (delay=30, K=5, total seq_len=40).

## Files

1. [`phase_memory_phase8_phase10_results.md`](phase_memory_phase8_phase10_results.md) — Phase 8 8-cell matrix + Phase 10 tanh+1 detail + Phase 11 param-matched + softmask=OFF rerun + Phase 12 substrate ablation. **ReLU is catastrophic on both real and complex (0.12)**; complex_sigmoid / complex_tanh1 / real_screen are tied 1st; under softmask=OFF + dim=80, all cells reach 100% (capacity saturation).

## Headline

> **For retrieval where phase is the identifier, gradient-nonzero on the full domain (condition B) is decisive** — ReLU attention is at random (0.12) on both real and complex; smooth bounded gates (sigmoid / tanh+1 / screening) all sit at 0.95-1.00. **complex_tanh1 hits exact 1.0 (3/3 perfect; sigmoid 1/3); screening cells are fastest on step→0.9.**
