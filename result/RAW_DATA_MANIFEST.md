# MANIFEST — public raw-data staging tree

Generated 2026-08-13 by staging script from internal stores:
`phase8_results/dok_pulled/`, `phase8_results/sync_pulled/` (soroban + vast),
`phase8_results/sakura_artifacts_pulled/`, `complex_nn_experiment/runs/` (local bench),
plus git-history extraction (commit `52be6aa`) for the RadioML/MusicNet real-data benches.

Layout: `<family>/<job_id>__<provenance>/{summary.json, metrics.jsonl, config.json}`.
Provenance suffixes: `__dok_<batch>` (Sakura DOK pull batch), `__soroban<host>`,
`__vast<instance>`, `__sakura` (sakura_artifacts_pulled), `__bench` (local runs/),
`__git52be6aa` (extracted from git history). Suffix `_ckpt` marks copies that were
pulled from an instance's `_ckpt` mirror. Every copy of a job is kept — divergent
duplicates stay transparent. Checkpoints (`*.pt`) are intentionally excluded.
The `09_musicnet_radioml_real` family additionally carries per-run `*.jsonl` and `run.log`,
since those benches log one jsonl per (task, cell, seed) instead of metrics.jsonl.

| family | run copies | distinct job_ids | duplicate copies | size |
|---|---:|---:|---:|---:|
| 01_copy_memory | 1170 | 759 | 411 | 9.9 MB |
| 02_niah_L2048 | 172 | 126 | 46 | 1.7 MB |
| 03_lra_listops | 59 | 58 | 1 | 524 KB |
| 04_lra_text_4k | 25 | 19 | 6 | 192 KB |
| 05_lra_image_cifar | 36 | 36 | 0 | 288 KB |
| 06_fft_mnist | 150 | 150 | 0 | 1.2 MB |
| 08_multi_pitch_synth | 73 | 73 | 0 | 584 KB |
| 09_musicnet_radioml_real | 5 | 5 | 0 | 640 KB |
| 11_c2_isolation | 27 | 6 | 21 | 216 KB |
| 99_other | 155 | 155 | 0 | 1.3 MB |
| **total** | **1872** | **1387** | **485** | **17.0 MB** |

## Notes / gaps

- `07_phase_memory` (public result dir): raw phase_memory / phase_sum runs are staged
  under `99_other/` per the staging spec (jobs `*phase_memory*`, `*phase_sum*`, `*pm_N5*`,
  `*ps_N20*`, `p12_substrate_*`, `p13_modrelu_fix_pm_*`).
- `10_radioml` and `09_musicnet_real` (public result dirs) are merged here into
  `09_musicnet_radioml_real/`: `bench_mn_rml*` = synth/substitute L1+L2 (commit a85dccf
  content, identical blobs re-extracted from 52be6aa), `bench_real*` = real RadioML +
  real MusicNet L1+L2.
- `11_c2_isolation`: `c2iso_*` orchestrator jobs (softplus vs relu on copymem d1000) are
  given their own family to mirror the public `result/11_c2_isolation` dir, rather than
  being folded into 01/99.
- No internal store contains `config.json` for the DOK/soroban/vast/sakura jobs — only
  the local bench runs carry config.json. Remote-job configs live in the launch yamls,
  not in the pulled artifact dirs.
- `99_other` also contains non-paper tracks pulled in by the blanket bench sweep
  (binctabl / fi2010 / stocknet investment benches, smoke runs, tinysmoke).
- All eight paper families listed in the staging spec have at least one raw run;
  no family is empty.

## Duplicate copies with final metrics differing by > 0.05 (top 10)

| Δ | family | job_id | metric | values (copy: value) |
|---:|---|---|---|---|
| 0.9500 | 01_copy_memory | micro_b256_d500_complex_softmax_lr3e-03_s1 | copy_acc | soroban220: 0.0500; vast36076958: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_real_softmax_lr3e-03_s2 | copy_acc | soroban203: 0.0750; vast36077724: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_real_softmax_lr3e-03_s0 | copy_acc | soroban242: 0.0750; vast36076951: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_real_softmax_lr1e-03_s0 | copy_acc | soroban242: 0.0750; vast36077719: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_real_sigmoid_lr3e-03_s2 | copy_acc | soroban203: 0.0750; vast36078897: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_real_sigmoid_lr3e-03_s0 | copy_acc | soroban216: 0.0750; vast36078892: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_real_sigmoid_lr1e-02_s2 | copy_acc | soroban203: 0.0750; vast36076955: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_real_sigmoid_lr1e-02_s0 | copy_acc | soroban216: 0.0750; vast36078897: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_complex_softmax_lr3e-03_s2 | copy_acc | soroban238: 0.0750; vast36096615: 1.0000 |
| 0.9250 | 01_copy_memory | micro_d1000_complex_softmax_lr3e-03_s0 | copy_acc | soroban220: 0.0750; vast36077726: 1.0000 |

Total flagged job_ids (Δ>0.05 on some acc/f1 metric): 65

