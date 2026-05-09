# Copy Memory — Small-scale (dim=32, depth=2, batch=256) — Phase 14 audit corrected

**Setup**: dim=32, depth=2, heads=2, dim_head=16 (~26K params), batch=256, lr ∈ {1e-3, 3e-3, 1e-2}. 3 seeds per cell.
**Source**: `phase_14_audit.md` lines 49-57.
**§4.6 fairness rule (revised)**: softmask is changed to OFF default (Phase 14 §4.6 revision).

## 1. Final-acc per LR

| Cell | d=200 (1e-3 / 3e-3 / 1e-2) | d=500 (1e-3 / 3e-3 / 1e-2) | d=1000 (1e-3 / 3e-3 / 1e-2) |
|---|---|---|---|
| real_softmax | 100% / 100% / 100% | 58% / 100% / 100% | 37% / 69% / 31% |
| real_sigmoid | 64% / 100% / 100% | 25% / 100% / 100% | 7% / 100% / 22% |
| complex_softmax | 67% / 100% / 94% | 18% / 70% / 42% | 9% / 39% / 35% |
| **complex_sigmoid** | **100% / 100% / 100%** | **100% / 100% / 100%** | **100% / 100% / 100%** |
| real_screen (softmask=OFF) | **96–100% (3 LRs)** | 100% (so far) | (in-flight) |
| complex_screen (softmask=OFF) | **95–100% (3 LRs)** | (in-flight) | (in-flight) |

## 2. Best-LR vs LR-window-robust tally

| Criterion | Eligible cells |
|---|---|
| 100% at best LR (d=200) | real_softmax / real_sigmoid / complex_softmax (3e-3 only) / **complex_sigmoid** (all LRs) / real_screen (softmask off, 3 LRs) |
| 100% at best LR (d=500) | real_softmax (3e-3, 1e-2) / real_sigmoid (3e-3, 1e-2) / **complex_sigmoid** (all LRs) |
| 100% at best LR (d=1000) | real_sigmoid (3e-3 only) / **complex_sigmoid** (all LRs) |
| **100% at all 3 LRs (d=1000) — truly LR-robust** | **complex_sigmoid** (uniquely) |

→ **complex_sigmoid has the widest LR window** — uniquely so on the "narrow LR window" axis from Arora et al. 2025.

## 3. 4-condition truth table (real_screen, copymem d=500, batch=256)

| softmask | TanhNorm | final_acc |
|:-:|:-:|---:|
| ON | ON | 10% (chance) ← Phase 5 default ("fairness rule") |
| ON | OFF | 11% |
| **OFF** | **ON** | **90%** |
| **OFF** | **OFF** | **100%** ← Phase 4 default |

**Conclusion**: the cosine softmask was pushing screening cells into a chance-level basin. **softmask is the main defect for screening cells**; TanhNorm alone is benign.

→ Phase 14 changed both the trainer and implementation defaults to softmask=OFF.

## 4. batch=64 (Phase 8 era) vs batch=256 (Phase 14) comparison

batch=64 is at dim=64 and batch=256 is at dim=32, so this is not a strict fair comparison, only a pattern check:

| Cell × Task | batch=64 dim=64 (best LR=3e-3) | batch=256 dim=32 (best LR) | Observation |
|---|---:|---:|---|
| real_softmax × Copy d=500 | 0.956 | 100% | as the Phase 14 audit claimed |
| real_softmax × Copy d=1000 | 0.050 (rand) | 69% (best LR) / 37%, 31% | narrow LR window |
| real_sigmoid × Copy d=500 | 0.904 | 100% | stable at batch=256 |
| real_sigmoid × Copy d=1000 | 0.177 | 100% (best LR) / 7-22% | solvable only at the best LR |
| complex_softmax × Copy d=500 | 0.100 (rand) | 70% (best LR) / 18-42% | even at dim=32 does not reach 100% |
| complex_softmax × Copy d=1000 | 0.104 (rand) | 9-39% | collapses even at batch=256 |
| **complex_sigmoid × Copy d=500** | **1.000** | **100% (all LRs)** | robust at both settings |
| **complex_sigmoid × Copy d=1000** | **1.000** | **100% (all LRs)** | **uniquely LR-robust** |

**Pattern**: at batch=256 the other cells start showing capability (softmax cells 100% at short distances, sigmoid cells 100% at mid distances), but **only complex_sigmoid is LR-robust at long distance d=1000**.

## 5. Implications for the §3.4 four-condition framework

The 4 conditions of complex_sigmoid:
1. **complex** Q/K (phase-aware similarity)
2. **L2 normalization** (cosine score in [-1, 1])
3. **per-pair sigmoid** (no row competition)
4. **complex value path** (phase information through aggregation)

Phase 14 confirmed (3) per-pair sigmoid with no row competition is **the single biggest contributor to LR robustness**.

## 6. Cross-references

- Predecessor (defect period): `result/01_copy_memory/01_small_scale_phase8_d100_to_d1000_dim64.md`
- Mid-scale batch-correction result: `result/01_copy_memory/03_mid_scale_dim256_d2000_d5000.md`
- Full audit: `_frameworks/02_phase14_softmask_and_batch_audit.md`
