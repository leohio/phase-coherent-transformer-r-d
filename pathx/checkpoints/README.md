# Path-X hybrid (complex screening + PCR) — all three seed checkpoints

Training endpoints (step 250,000) of the §6 hybrid model, one per seed. These are the
runs behind the paper's Path-X result **92.71 ± 0.89** (held-out split, n=20,000,
deterministic sweep):

| file | seed | held-out test acc |
|---|---:|---:|
| `hybrid_pathx_seed0_step250000.pt` | 0 | **0.9350** (best) |
| `hybrid_pathx_seed1_step250000.pt` | 1 | 0.9175 |
| `hybrid_pathx_seed2_step250000.pt` | 2 | 0.9288 |

The seed-0 weights are also published in HF format under `../model/`. Model code:
`../code/pcr_screening.py`. Evaluate with the checkpoint loader documented in
`../README.md` (deterministic full-split eval, no bootstrap resampling).
