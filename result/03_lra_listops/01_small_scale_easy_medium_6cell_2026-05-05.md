# LRA-ListOps — Small-scale 6-cell × 2 difficulty bench (CPU, N=1)

**Date**: 2026-05-05 (CPU 60 min completed)
**Setup**: small-scale architecture (dim=64, depth=2, heads=4, dim_head=16, ~100K params), batch=32, N=1 seed, CPU. 6 cells × 2 difficulties = 12 runs. Screening cells use softmask=OFF (Phase 14 §4.6 revision); only complex_sigmoid is given a chunk_size to trigger the chunked attention path.
**Source**: `total_bench_(till_phase_14).md` §5.6
**Data**: synthetic on-the-fly LRA-ListOps generator (`complex_nn_experiment/data/lra_listops.py`). Slightly simplified relative to the LRA paper's standard distribution (the generator's seed bug is fixed; each batch is sampled independently).

## 1. Difficulty A — easy (depth=1, max_args=2, seq=64, chunk=16)

| Cell | best | final | time | Note |
|---|---:|---:|---:|---|
| real_softmax | 0.9883 | 0.9883 | 42s | strong |
| real_sigmoid | 0.8906 | 0.8867 | 41s | mid |
| real_screen | 0.9336 | 0.9023 | 57s | mid |
| **complex_softmax** | **1.0000** | **1.0000** | 163s | strongest |
| **complex_sigmoid** | **1.0000** | 0.9844 | 251s | strongest (tied) |
| **complex_screen** | **1.0000** | **1.0000** | 240s | strongest (tied) |

**Observation**: at the easy setting **all 3 complex cells reach 1.000** (real cells top out at 0.99 — slightly worse). The task is simple enough that no differentiation appears, so the complex side wins across the board.

## 2. Difficulty B — medium (depth=2, max_args=3, seq=128, chunk=32)

| Cell | best | final | time | Note |
|---|---:|---:|---:|---|
| real_softmax | 0.6250 | 0.6250 | 129s | mid |
| real_sigmoid | 0.5938 | 0.5938 | 109s | weak |
| **real_screen** | **0.6875** | **0.6875** | 154s | real-side strongest |
| complex_softmax | 0.6367 | 0.6367 | 416s | mid |
| **complex_sigmoid** | **0.7188** | **0.7188** | 1039s | **single 1st place** |
| complex_screen | 0.7148 | 0.7148 | 928s | 2nd (close) |

**Observations**:
- **complex_sigmoid is single 1st place** (0.7188); complex_screen 0.7148 is a close 2nd
- On the real side, **real_screen 0.6875 is the strongest** (0.05-0.09 above the other real cells, continuing the Phase 8/10 "real_screen is strong on phase/algorithmic" pattern)
- complex_sigmoid > complex_screen (margin +0.004) — sigmoid edges screen even at small-scale L=128
- real ↔ complex gap: complex side +0.03 to +0.08 (standard pattern for content-driven retrieval tasks)

## 3. Three points the result suggests

1. **Consistent with the Anti-correlation Preservation framework**: even on an algorithmic task without phase information at the medium setting, complex_sigmoid takes the trade win. Tasks that need content / relational retrieval are robust under complex + sigmoid.
2. **real_screen's standalone strength**: on the real-side classification, real_screen pulls ahead of the other real cells (also consistent with prior findings such as real_screen 1.000 on phase_memory).
3. **Contrast with the NIAH-flip**: NIAH (purely positional retrieval) is where screening dominates; ListOps (content-driven algorithmic) is where sigmoid wins. **Task structure (positional vs content) flips the cell ranking** — confirmed again.

## 4. Caveats

- N=1 only — N=3 replication is the next step
- Synthetic generator instead of the official LRA paper TFRecord (simplified to medium=2/3 instead of max_depth=5, max_args=5)
- CPU only (equivalent to MPS shim)

## 5. Cross-references

- PCT-fairness rerun (same dim, real param compensated): `result/03_lra_listops/02_pct_fairness_dep2_L128_L1024_2026-05-08.md`
- Apple Silicon MPS plateau analysis: `result/03_lra_listops/03_apple_silicon_mps_complex_sigmoid_plateau.md`
- Mechanism: `_frameworks/01_anti_correlation_preservation_mechanism.md`
