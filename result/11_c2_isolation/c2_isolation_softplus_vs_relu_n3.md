# C2 Isolation Experiment — softplus vs ReLU (N=3)

**Date**: 2026-05-07
**Provider**: Vast.ai Tier-S RTX 3090 (orchestrator path + direct dispatch)
**Status**: 6/6 jobs completed successfully
**Source**: `doc2/results_c2_isolation.md`

> ⚠️ **§3-§4 of this document are SUPERSEDED**. The 2-cell isolation here (softplus vs ReLU) initially suggested that C2 might be droppable from the framework (3-condition simplification). **That conclusion was withdrawn 2026-05-08** by the completed 2 × 2 isolation with `complex_cubic` (M≈252 → 0.200) and `complex_clamped_relu` (M=1, C3 ✗ → 0.103) at N=3. See [`c2_c3_2x2_isolation_cubic_clamped_relu_n3.md`](c2_c3_2x2_isolation_cubic_clamped_relu_n3.md) for the **definitive** result and the restored 4-condition framework.
>
> The N=3 numbers reported in §1 (softplus 1.000, ReLU 0.107) are still correct — only the *interpretation* in §3-§4 has been overturned.

## 1. Summary table (N=3 each)

| Cell | C1 | C2 | C3 | C4 | mean copy_acc | std | individual seeds |
|---|:-:|:-:|:-:|:-:|---:|---:|---|
| **complex_softplus** | ✓ | ✗ | ✓ | ✓ | **1.0000** | 0.0000 | 1.000 / 1.000 / 1.000 |
| **complex_relu** | ✓ | ✗ | ✗ | ✓ | **0.1073** | 0.0274 | 0.106 / 0.075 / 0.141 |

**Task**: copymem K=10 d=1000 (sequence length 1021)
**Architecture**: dim=128, depth=4, heads=4, dim_head=32, ff_mult=4
**Training**: batch=32 effective, lr=3e-3, warmup=200, steps=2000, AdamW
**Seeds**: 0, 1, 2

## 2. Headline finding

**The only structural difference between complex_softplus and complex_relu is C3 (anti-correlation preservation):**
- Both share C1 ✓ (real gate), C2 ✗ (gate unbounded above on ℝ), C4 ✓ (no row-norm)
- softplus: `f(s) = log(1 + e^{s+b})`, `f'(s) = sigmoid(s) ∈ (0, 1)` — **C3 ✓** (gradient nonzero everywhere, anti-phase preserved)
- relu: `f(s) = max(s + b, 0)`, `f'(s) = 1{s + b > 0}` — **C3 ✗** (zero gradient on negative side, anti-phase deleted)

**Result**: 0.893 absolute accuracy gap (softplus 1.0 vs relu 0.107), N=3 clean separation.

→ **C3 is empirically essential** for long-range retrieval at depth=4. ReLU's catastrophic failure is **C3-driven**, not C2-driven.

## 3. Resolution of C2's role *(SUPERSEDED — see [c2_c3_2x2_isolation_cubic_clamped_relu_n3.md](c2_c3_2x2_isolation_cubic_clamped_relu_n3.md))*

This 2-cell isolation (softplus M≈4 vs ReLU) initially suggested C2 might be droppable, since softplus violates strict-on-ℝ C2 but reaches 1.000 due to operating-range boundedness.

**However**, the completed 2 × 2 isolation 2026-05-08 with `complex_cubic` (M≈252) added a cell with **strong** operating-range C2 violation, which collapses to **0.200** — showing that **C2 *is* necessary at sufficient M**. The earlier reasoning that "M ≈ 4 for softplus is harmless ⇒ C2 is automatic" overstated the case based on a single low-M data point.

The current C3-isolation reading remains correct: ReLU and softplus differ structurally only in C3, and the 0.893 gap (1.000 vs 0.107) is **unambiguously attributable to C3**. But C2 is **not** redundant — see the refined framework in §4.

## 4. Implication for the framework — 4-condition restored, in operating-range form *(REVISED)*

The framework retains **all four conditions**, with C2 / C3 read in **operating-range form** (cosine score restricted to `[−√d, √d]` by L2-normalisation):

| Level | Conditions (operating-range form) | Role |
|---|---|---|
| **Architectural baseline** (assumed) | L2-normalize + RMSNorm substrate | Defines the operating range `[−√d, √d]` on which C2/C3 are evaluated |
| **L1** (per-layer phase coherence) | **C1** (real gate) + **C4** (element-independent) | Per-layer structure |
| **L2** (all-layer cascade phase stability) | **C1** + **C2** (operating-range bounded, M tractable) + **C3** (anti-phase preservation, no zero gradient) + **C4** | Cascade structure |

**Both C2 and C3 are independently necessary in their operating-range form**, with C3 the dominant factor (chance-level when violated alone). Magnitude of C2 violation matters **monotonically**: softplus M≈4 → 1.000, cubic M≈252 → 0.200 → there is a transition somewhere between.

### Cell taxonomy under restored 4-condition framework

| Cell | C1 | C2 (op-range) | C3 | C4 | L1 | L2 | Empirical |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| PCT (sigmoid) | ✓ | ✓ M=1 | ✓ | ✓ | ✓ | ✓ | universally strong |
| softplus (this exp) | ✓ | partial M≈4 (bypassed) | ✓ | ✓ | ✓ | ✓ | **acc=1.000 (N=3)** |
| tanh+1 | ✓ | ✓ M=1 | ✓ | ✓ | ✓ | ✓ | strong (Phase 10) |
| semi-PCT (screen) | ✓ | ✓ (TanhNorm) | partial | ✓ | ✓ | partial | task-conditional |
| **cubic** (2026-05-08) | ✓ | **✗ M≈252** | ✓ strict | ✓ | ✓ | **✗** | **acc=0.200 (N=3)** partial collapse |
| **clamped_relu** (2026-05-08) | ✓ | ✓ M=1 | **✗ full** | ✓ | ✓ | **✗** | **acc=0.103 (N=3)** chance |
| ReLU (this exp) | ✓ | partial M≈11 | ✗ full | ✓ | ✓ | ✗ | **acc=0.107 (N=3)** chance |
| softmax | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | catastrophic (L1 broken) |

## 5. What this experiment did NOT settle

- **C2 strict necessity in non-PCT architectures**: if substrate or L2-normalize is removed, C2 might re-emerge as essential. Not tested
- **L2 at depth > 4**: cascade phase stability (L2) is properly tested by depth sweep. softplus at depth=4 achieves 1.000, but whether it stays at 1.000 at depth=12 with C2 violated is open
- **C3 partial vs full distinction**: semi-PCT (C3 partial) vs ReLU (C3 fully ✗) shows the *degree* of C3 violation matters. Empirical separation but no formal characterization of "C3 partial"

## 6. Mapping to paper outline (current = v4)

The 2-cell isolation here was originally mapped to a "drop C2 column → 3-condition table" change in paper outline_v3. This proposal is **withdrawn** by the 2026-05-08 2 × 2 completion ([`c2_c3_2x2_isolation_cubic_clamped_relu_n3.md`](c2_c3_2x2_isolation_cubic_clamped_relu_n3.md)).

In paper outline_v4 the framework is back to the 4-condition form (`C1 + C2 + C3 + C4 ⇒ L2`) with C2 / C3 explicitly read in **operating-range form**, and the 2 × 2 isolation (cubic + clamped_relu, N=3) is paper §5.4 's central evidence:

- **C3 strict in operating range**: clamped_relu 0.103 (chance) — anti-phase deletion alone is fatal
- **C2 strict in operating range with large M**: cubic 0.200 (partial collapse) — large M is also fatal
- **C2 partial / bypassed**: softplus 1.000 (this experiment) — small-M C2 violation is tolerated
- **Both ✗**: ReLU 0.107 (this experiment) — chance, indistinguishable from clamped_relu

The 3-tier cell taxonomy still works, but the "PCT-class / semi-PCT-class / failed-condition class" labels are **structurally** explained by the closeness-to-PCT axis (whether deviations are partial / bypassed / strict-in-operating-range), not by dropping conditions.

## 7. Provider / cost

- 6 jobs × ~373–694s per job on RTX 3090 (avg ~443s)
- Total compute: ~2660 GPU-seconds = 0.74 GPU-hr
- Estimated cost: ~$0.18 (Vast.ai Tier-S at $0.16–0.25/hr)
- Wall time: ~30 min

## 8. Files / artifacts

- YAML: `/var/lib/phase8/queue/jobs_c2_isolation.yaml` (now in queue_done)
- Logs: per-instance `/root/phase8/runs_jobs_c2_isolation/c2iso_orch_*/log.txt` and `summary.json`
- Aggregation script: `/tmp/collect_c2iso.sh` on mgr
- Trainer: `/var/lib/phase8/build_v2/phase8_experiment/src/train_with_checkpoint.py` (with `complex_softplus` and `complex_relu` choices)
- Cell impl: `/var/lib/phase8/build_v2/complex_nn_experiment/transformer.py` (with `ComplexMultiHeadSoftplusAttention`)
- Local: `complex_nn_experiment/transformer.py`, `phase8_experiment/src/train_with_checkpoint.py`
