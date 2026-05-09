# C2 / C3 Isolation — Test summary

These experiments test the four-condition framework's C2 (boundedness) and C3 (anti-correlation preservation) by comparing complex-attention cells that selectively violate one condition while keeping the other strictly satisfied.

## Files

1. [`c2_isolation_softplus_vs_relu_n3.md`](c2_isolation_softplus_vs_relu_n3.md) (2026-05-07, 2-cell N=3) — softplus 1.000 vs ReLU 0.107 (Δ=0.893) at depth=4 on Copy d=1000. Establishes **C3 (anti-correlation preservation) is empirically essential**. **§3-§4 superseded** (the 3-condition simplification was retracted by the next experiment).
2. [`c2_c3_2x2_isolation_cubic_clamped_relu_n3.md`](c2_c3_2x2_isolation_cubic_clamped_relu_n3.md) (2026-05-08, 2 × 2 N=3) — Adds `complex_cubic` (C3 ✓ strict, C2 ✗ M≈252) → 0.200 and `complex_clamped_relu` (C2 ✓ M=1, C3 ✗ full) → 0.103. Completes the 2 × 2 design **and restores the framework to 4-condition** in operating-range form.

## Headline (2026-05-08, after 2 × 2 completion)

> **All four conditions (C1 + C2 + C3 + C4) are independently necessary in their operating-range form**:
> - **C3 violation alone** (clamped_relu, M=1) → chance (0.103) — anti-phase deletion is the dominant failure mode
> - **C2 violation alone** (cubic, M≈252) → partial collapse (0.200) — strong magnitude breaks cascade contraction even with C3 satisfied
> - **C2 violation, small magnitude** (softplus, M≈4) → fully tolerated (1.000) — operating-range bypass works at low M
> - **Both ✗** (ReLU) → chance (0.107) — once C3 is fully violated, additional C2 status is irrelevant
>
> The framework is therefore correctly stated as **C1 + C2 + C3 + C4 ⇒ L2** with both C2 and C3 read in operating-range form.

The 2-cell experiment's earlier "drop C2 → 3-condition" simplification was based on softplus alone (M≈4 → 1.000) and is **withdrawn**.
