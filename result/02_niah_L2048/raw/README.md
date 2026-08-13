# NIAH L=2048 `complex_sigmoid` (PCT) — raw per-run artifacts

Recovered 2026-08-13 from Sakura DOK task artifacts (`artifact.tar.gz` of each task).
Each run dir contains the per-step `metrics.jsonl` (eval every 500 steps) and the
container `console.log`. The DOK artifact bundles do not include a `config.json`;
the full hyperparameters are in the paper's Appendix A2 and the table below, and are
also encoded in the DOK task names.

| run | DOK task id | task name | config |
|---|---|---|---|
| `s0_configA_b32_30k` | `94cb7e68-b396-4889-b2f1-1459dccc95bd` | `niah-L2048-cs-d256L6-b32-s30k-mb16` | dim=256, depth=6, heads=8, dim_head=32, chunk=256, eff. batch=32 (mb=16, ga=2), lr=3e-4, warmup=1000, 30K steps, seed 0 |
| `s0_configB_b16_60k` | `50d938fd-81f4-47fc-8d8d-29e3414567b0` | `niah-L2048-cs-d256L6-b16-s60k` | same arch, eff. batch=16, 60K steps, seed 0 |
| `s1_configA_b32_30k` | `013e26e1-bc80-464d-8c84-89eb124f390b` | `niah-L2048-cs-d256L6-b32-s30k-mb16-s1` | config A, seed 1 |
| `s2_configA_b32_30k` | `130fad10-0d2e-4238-b816-0c0ed249be75` | `niah-L2048-cs-d256L6-b32-s30k-mb16-s2` | config A, seed 2 |

Key readings (verifiable from the included `metrics.jsonl`):

| run | first `needle_acc=1.0` | last logged step | final `needle_acc` | final `eval_loss` |
|---|---:|---:|---:|---:|
| s0 config A | step 3000 | 24000 / 30000 | 1.000 | 3.67e-4 |
| s0 config B | step 9000 | 48000 / 60000 | 1.000 | 2.86e-5 |
| s1 config A | step 3000 | 24000 / 30000 | 1.000 | 3.33e-4 |
| s2 config A | step 6000 ¹ | 24000 / 30000 | 1.000 | 6.92e-4 |

¹ s2 read 0.8125 at step 3000, dipped transiently to 0.5625–0.6875 (steps 3500–4500),
then held 1.0 from step 6000.

Caveat (also stated in `../niah_L2048_complex_sigmoid_n3.md`): all runs were killed by a
6h wall-clock timeout at 80% of the step budget (hence DOK status `canceled`), with
`needle_acc=1.0` held continuously from first solve to kill. No trained checkpoints were
retained in the DOK artifact for these runs.
