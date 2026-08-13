# Depth-scaling sweep (`complex_sigmoid` × LRA-ListOps L=1024) — raw per-run artifacts, seed 0

Recovered 2026-08-13 from Sakura DOK task artifacts. Each run dir contains
`metrics.jsonl` (eval every 500 steps), `summary.json`, `console.log`, and the
trained `final_model.pt` (30K-step endpoint).

Shared config: task `lra_listops` with `max_seq_len=1024, max_depth=6, max_args=5`;
dim=128, heads=4, dim_head=32, ff_mult=4, chunked attention (chunk=128); batch=32
(micro=32), AdamW wd=0.01, clip 1.0, lr=1e-3 cosine (warmup 500), 30 000 steps,
seed 0. H100-80GB, 2026-05-07/08. See `04_depth_scaling_d2_to_d20_complex_sigmoid_n1.md`
for the summary write-up (single seed, N=1).

| depth | DOK task id |
|---:|---|
| 2 | `30c6b2b9-fd11-44dc-91fd-2086ccaa46f3` |
| 4 | `5a5fa635-7236-4d7b-80dc-2be8a10d104e` |
| 6 | `06f77011-c51c-4919-ac78-ec122a0a5535` |
| 10 | `5b56d08a-f2bd-4ed0-9bed-2de0e4406068` |
| 14 | `593ce8c4-14ca-4b10-b571-cdcff3d54859` |
| 20 | `84dbd9fb-4d01-4213-8d82-9cd0957522bb` |

An N=3 replication (seeds 0/1/2 re-run end-to-end with an additional n=2048
deterministic final eval) is in progress as of 2026-08-13 and will be added under
`raw_depth_scaling_n3/` when complete.
