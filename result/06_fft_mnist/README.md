# FFT-MNIST — Test summary

MNIST digit classification on a 2D rFFT representation (`t` time-frames × frequency bins → seq_len = `t × 8`). Digit class affects the 2D-FFT phase pattern → phase-sensitive classification task.

## Files

1. [`fft_mnist_t8_t16_phase8_to_pctfairness.md`](fft_mnist_t8_t16_phase8_to_pctfairness.md) — Phase 8/9/10/11 small-scale baseline + Phase 11 param-matched + softmask=OFF rerun + PCT-fairness rerun (dim=128, N=3). **PCT (csg) 0.927 and semi-PCT (cscr) 0.938 are tied 1st**; real_screen 0.865 leads the real side alone.

## Headline

> **On phase-sensitive classification (FFT-MNIST t=16), PCT and semi-PCT are equally strong** (PCT-fair N=3: 0.93/0.94). **real_screen is the only strong cell on the real side at 0.86** (1.7-1.8× the vanilla real cells at 0.48-0.50). **Vanilla complex_softmax sits in the middle at 0.71** — the row-norm constraint refrains it.
