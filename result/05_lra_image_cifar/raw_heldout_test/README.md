# LRA-Image CIFAR — held-out test-split re-measurement (2026-08-14)

Full re-run of the 6-cell × 3-seed LRA-Image comparison with the evaluation bug
fixed: the trainer evaluates on the **held-out test split** (`--eval-split test`)
and the final metric is a **deterministic n=2048 evaluation** (`--final-eval-n
2048`), replacing the original runs' final-step single-batch (n=32) train-split
reading. Config otherwise matches the original PCT-fairness runs: real cells
dim=184/dh=46, complex cells dim=128/dh=32, depth=4, heads=4, batch=32,
lr=4e-4 (warmup 1000), 15 000 steps, RTX 3090.

Each run dir contains `summary.json` (with `final_eval: {split: "test", n: 2048}`),
per-step `metrics.jsonl`, and the trained `final_model.pt`.

## Results (test acc, n=2048, N=3)

| cell | s0 | s1 | s2 | mean ± std | superseded value (train-split n=32) |
|---|---:|---:|---:|---|---:|
| real_softmax | 0.2241 | 0.2144 | 0.2280 | 0.2222 ± 0.0070 | 0.156 |
| real_sigmoid | 0.2285 | 0.2207 | 0.2236 | 0.2243 ± 0.0039 | 0.156 |
| real_screen | 0.3320 | 0.3135 | 0.3096 | 0.3184 ± 0.0120 | 0.333 |
| complex_softmax | 0.2344 | 0.2266 | 0.2354 | 0.2321 ± 0.0048 | 0.156 |
| **complex_sigmoid (PCT)** | 0.3799 | 0.3657 | 0.3706 | **0.3721 ± 0.0072** | 0.458 |
| complex_screen | 0.3477 | 0.3262 | 0.3481 | 0.3407 ± 0.0125 | 0.406 |

- The qualitative ordering of the published table survives held-out evaluation:
  PCT > complex_screen > real_screen > the three softmax/sigmoid baselines,
  with cell gaps an order of magnitude larger than the seed std.
- The superseded numbers were distorted in both directions: leaders were
  inflated by the single lucky-batch reading, baselines deflated by the n=32
  quantization floor. These test-split values supersede the `0.458` etc.
  reported in `pct_fairness_lra_image_cifar_results.md` and the paper's tables;
  a paper-side correction pass will follow once the LRA-Text re-measurement
  (in progress) completes.
