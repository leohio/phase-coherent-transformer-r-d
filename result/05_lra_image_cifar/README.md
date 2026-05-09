# LRA-Image CIFAR — Test summary

LRA CIFAR-10 grayscale (32×32 → seq_len=1024) image classification under the PCT-fairness benchmark (Tier B; requires `lra_image.py` loader + `.pt` data prep + Docker image rebuild).

## Files

1. [`pct_fairness_lra_image_cifar_results.md`](pct_fairness_lra_image_cifar_results.md) — 6-cell × 6 seeds (DOK N=3 + Vast N=3), real param 1.41× compensated. **PCT (csg) 0.458 single 1st**; semi-PCT (cscr) 0.406 2nd; real_screen 0.318 3rd; vanilla softmax (real and complex) at chance (0.156).

## Headline

> **Complex sigmoid (PCT) is single 1st even on image pixel-sequence** (0.458 N=6). Screening exceeds chance by 2× on both real and complex sides. Vanilla softmax (real and complex) cannot solve the task even with more capacity — stuck at chance. **Whether row-norm is present is decisive** (0.156 → 0.458 is a 3× gap).
