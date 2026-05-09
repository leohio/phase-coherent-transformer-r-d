# Phase-Coherent Transformer — benchmark code

This directory bundles all code used to run the benchmarks reported in the
Phase-Coherent Transformer (PCT) paper / appendix. It is laid out so that
external readers can reproduce single-job runs, smoke tests, and (with their
own GPU compute) the full sweeps.

## Layout

```
code/
├── README.md                          ← this file
├── requirements.txt                   ← minimal deps for local runs
├── complex_nn_experiment/             ← model + data + baselines
│   ├── transformer.py                 ← 6-cell factory `make_cell(...)`
│   ├── transformer_cls.py             ← classifier head (FFT-MNIST etc.)
│   ├── transformer_cls_v2.py
│   ├── screening.py                   ← screening attention (semi-PCT)
│   ├── data/                          ← per-task data generators / loaders
│   │   ├── copy_memory.py
│   │   ├── fft_mnist.py
│   │   ├── lra_image.py / lra_listops.py / lra_pathfinder.py / lra_text.py
│   │   ├── musicnet_real.py / radioml.py / radioml_real.py
│   │   ├── niah.py
│   │   ├── pathx.py
│   │   └── phase_tasks.py
│   └── baselines/lucidrains_cvt/      ← complex-valued transformer baseline
└── phase8_experiment/                 ← training driver + container recipe
    ├── src/train_with_checkpoint.py   ← multi-task entry with checkpoint/resume
    ├── configs/                       ← example job specs (smoke / Path-X)
    ├── docker/                        ← Dockerfile + container deps
    └── scripts/entrypoint.sh, entrypoint_8gpu.sh
```

## Cells

`make_cell(name, ...)` in [`complex_nn_experiment/transformer.py`](complex_nn_experiment/transformer.py)
exposes the 6 cells used in the paper's main sweep, plus two extras used in
the §6.7 closeness anchor:

| name              | softmax / non-PCT | sigmoid (PCT) | screening (semi-PCT) |
|-------------------|-------------------|---------------|----------------------|
| real-valued       | `real_softmax`    | `real_sigmoid`| `real_screen`        |
| complex-valued    | `complex_softmax` | `complex_sigmoid` | `complex_screen` |
| extras (closeness)|                   | `complex_softplus`, `complex_relu` |   |

### Important defaults (Phase 14 audit)

- **Screening cells default to `use_softmask=False`.** Cosine softmask was
  found to keep screening at chance level on Copy-Memory; passing `--softmask`
  re-enables it for the Phase-5 ablation only. Do **not** add
  `no_softmask: true` to new job specs — it is redundant.
- **Param-fairness:** real cells use `dim=184, heads=4, dim_head=46` to match
  the parameter count of `dim=128, heads=4, dim_head=32` complex cells
  (≈ ×1.41). `dim_head` must be even due to the RoPE constraint.

## Quickstart — single local run

```bash
# 1. Install deps (CPU-only is fine for the smoke task; CUDA recommended otherwise).
pip install -r requirements.txt

# 2. Run a 100-step Copy-Memory smoke on the complex_sigmoid cell.
python phase8_experiment/src/train_with_checkpoint.py \
    --task copymem --task-params '{"K":10,"delay":200}' \
    --cell complex_sigmoid --seed 0 \
    --steps 100 --batch-size 8 --eval-every 50 \
    --dim 128 --depth 4 --heads 4 --dim-head 32 \
    --output-dir ./out_smoke --checkpoint-dir ./ckpt_smoke
```

`--output-dir` receives `metrics.jsonl`, `summary.json`, and
`final_model.pt` on completion. `--checkpoint-dir` defaults to
`/workspace/ckpt` (the Docker mount); pass an explicit path on a
laptop / non-container host.

### Path discovery

The trainer looks up `complex_nn_experiment/` in this order:

1. `$COMPLEX_NN_PATH` (explicit override),
2. `/workspace/complex_nn_experiment` (Docker layout),
3. the sibling directory of `phase8_experiment/` (this repo layout — what you
   get by default after cloning).

So no extra setup is needed when running from this folder.

## Reproducing a paper benchmark

The full job specs live under [`phase8_experiment/configs/`](phase8_experiment/configs/).
Each YAML lists `(task, task_params, cell, seed, dim, depth, ...)` tuples;
`scripts/entrypoint.sh` shows the exact CLI translation expected by
`train_with_checkpoint.py`. A minimal local invocation matching one job is:

```bash
python phase8_experiment/src/train_with_checkpoint.py \
    --task <task> --task-params '<json>' \
    --cell <cell> --seed <seed> \
    --steps <steps> --batch-size <batch_size> --lr <lr> --warmup <warmup> \
    --dim <dim> --depth <depth> --heads <heads> --dim-head <dim_head> --ff-mult <ff_mult> \
    --checkpoint-every <ckpt> --eval-every <eval> \
    --output-dir <outdir> --checkpoint-dir <ckptdir>
```

### Datasets

- **Synthetic tasks** (`copymem`, `fftmnist`, `niah`, `phase_memory`,
  `multi_pitch`) are generated on the fly. No download required.
- **LRA / Path-X** tasks (`lra_listops`, `lra_text`, `lra_image`,
  `lra_pathfinder`, `pathx`) expect preprocessed `.pt` tensors. Set the
  corresponding env var (`LRA_LISTOPS_PATH`, `LRA_TEXT_PATH`,
  `LRA_PATHFINDER_PATH`, `PATHX_PATH`) to the local path. The loaders in
  `complex_nn_experiment/data/` document the expected tensor shapes.
- **RadioML / MusicNet** tasks expect the standard public releases; loaders
  live in `radioml.py` / `musicnet_real.py`.

## Container build (optional)

The `phase8_experiment/docker/Dockerfile` produces the exact image used for
the cluster runs (Sakura DOK / multi-provider orchestrator). To build:

```bash
# Run from the parent of phase8_experiment/ and complex_nn_experiment/
docker build -f phase8_experiment/docker/Dockerfile -t pct-bench:latest .
```

Inside the image, `entrypoint.sh` reads `TASK`, `CELL`, `SEED`, `STEPS`, ...
from the environment and invokes the trainer.

## License

See `LICENSE` once added by the authors. All third-party baseline code in
`complex_nn_experiment/baselines/` retains its original license.
