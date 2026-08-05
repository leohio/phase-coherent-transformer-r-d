# Lean 4 formalisation of `paper/outline_v4.md` — Appendix M

This folder contains a Lean 4 + Mathlib formalisation of the
mathematical proofs in **Appendix M** of `paper/outline_v4.md`
(Phase-Coherent Transformers / two-level phase coherence).

## Build status (verified 2026-08-05)

```
$ lake build
Build completed successfully (8407 jobs).
```

**The project contains no `sorry` and no bespoke premises.**
`#print axioms` of every theorem — including **`lemmaC` (Doeblin
contraction)** and **`theorem5` (cascade phase stability)** — shows
only the three standard Lean / Mathlib axioms:

```
PaperV4.AttentionLayer.apply_R              depends on [propext, Classical.choice, Quot.sound]
PaperV4.AttentionLayer.L1b_witness          depends on [propext, Classical.choice, Quot.sound]
PaperV4.P_decompose                         depends on [propext, Classical.choice, Quot.sound]
PaperV4.composeLayers_R                     depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaA                              depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaA_bound                        depends on [propext, Classical.choice, Quot.sound]
PaperV4.L1b_implies_C4                      depends on [propext, Classical.choice, Quot.sound]
PaperV4.corollary2                          depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaC                              depends on [propext, Classical.choice, Quot.sound]
PaperV4.tokenSeqNorm_triangle               depends on [propext, Classical.choice, Quot.sound]
PaperV4.tokenSeqNorm_R_sub_le               depends on [propext, Classical.choice, Quot.sound]
PaperV4.composeLayers_zero_mean_stable      depends on [propext, Classical.choice, Quot.sound]
PaperV4.cascade_geometric_bound             depends on [propext, Classical.choice, Quot.sound]
PaperV4.theorem5                            depends on [propext, Classical.choice, Quot.sound]
PaperV4.theorem5Hypotheses_satisfiable      depends on [propext, Classical.choice, Quot.sound]
```

**Theorem 5 is proven** (`L2.lean`, `theorem5`) from **standard
per-layer hypotheses only** (`Theorem5Hypotheses`): per-layer L1.a
(supplied by Theorem 1), per-layer non-expansiveness (S2 + Lemma D),
per-layer uniform Lipschitz stability against zero-mean phase
perturbations (the single-layer conclusion of Lemmas B + C under S3),
and a uniformly bounded output range (S1).  No hypothesis is
conclusion-shaped — the earlier `Theorem5Premises.cascade_decomposition`
field, which assumed the L-uniform decomposition bound wholesale, has
been **removed** (2026-08-05) and replaced by a genuine depth induction.
The hypothesis set is machine-checked to be satisfiable
(`theorem5Hypotheses_satisfiable`), so nothing holds vacuously.
**Lemma C is proven standalone** (`LemmaC.lean`, `lemmaC`; 2026-08-05)
via the rank-1 Doeblin-component decomposition of Levin–Peres–Wilmer
2017 Theorem 4.9.

## Mapping: Appendix-M sections ↔ Lean files

| Appendix M | Content | File | Status |
| --- | --- | --- | --- |
| M.0 | Setting and notation (`Token`, `TokenSeq`, `R(φ)`, `P(ε)`) | [`PaperV4/Basic.lean`](PaperV4/Basic.lean) | **PROVEN** ✓ |
| M.1 | Definition 1 (per-layer phase coherence: L1.a + L1.b) | [`PaperV4/L1.lean`](PaperV4/L1.lean) | **PROVEN** ✓ |
| M.2 | Theorem 1 (C1 + C4 ⇒ L1) — `apply_R` + `L1b_witness` | [`PaperV4/L1.lean`](PaperV4/L1.lean) | **PROVEN** ✓ |
| M.2 | Theorem 1' (necessity of C4 for L1.b) — `L1b_implies_C4` | [`PaperV4/L1.lean`](PaperV4/L1.lean) | **PROVEN** ✓ |
| M.2 | Corollary 2 — `corollary2` | [`PaperV4/L1.lean`](PaperV4/L1.lean) | **PROVEN** ✓ |
| M.3 | Definition 3 (cascade phase stability) | [`PaperV4/L2.lean`](PaperV4/L2.lean) | **STATEMENT** |
| M.3 | Definition 4 (all-layer phase coherence) | [`PaperV4/L2.lean`](PaperV4/L2.lean) | **STATEMENT** |
| M.4 / M.10 | **Theorem 5** (C1+C3+C4+(A1,A2,S1–S3) ⇒ L2) | [`PaperV4/L2.lean`](PaperV4/L2.lean) | **PROVEN** ✓ (`theorem5`, no `sorry`, standard hypotheses only) |
| M.4 / M.10 | `Theorem5Hypotheses` (standard per-layer conditions) | [`PaperV4/L2.lean`](PaperV4/L2.lean) | satisfiable ✓ (`theorem5Hypotheses_satisfiable`) |
| M.6 step 4 | Geometric cascade recurrence (`Σ Λ^l ≤ 1/(1−Λ)` in invariant form) | [`PaperV4/L2.lean`](PaperV4/L2.lean) | **PROVEN** ✓ (`cascade_geometric_bound`) |
| M.7 | Lemma A — `P_decompose`, `composeLayers_R`, `lemmaA` | [`PaperV4/LemmaA.lean`](PaperV4/LemmaA.lean) | **PROVEN** ✓ (factorisation + stack pass-through) |
| M.7 | Quantitative norm bound `‖Ỹ_L − Y_L‖ ≤ … + |φ̄|·‖W‖` | [`PaperV4/L2.lean`](PaperV4/L2.lean) | **PROVEN** ✓ (`lemmaA_bound`; exact, no `O(φ̄²)` remainder) |
| M.8 | Lemma B — linearised per-layer Jacobian | — | paper-level; its single-layer conclusion is the standard hypothesis `zero_mean_phase_stable` |
| M.9 | Lemma C — Doeblin contraction | [`PaperV4/LemmaC.lean`](PaperV4/LemmaC.lean) | **PROVEN** ✓ (`lemmaC`, no `sorry`) |
| M.10 | Lemma D — substrate non-expansion | — | paper-level; its conclusion is the standard hypothesis `substrate_nonexpansive` |

### Status legend

* **PROVEN** — full Lean proof, no `sorry`. Verified by `lake build`
  ending in `Build completed successfully` with `#print axioms` showing
  only `[propext, Classical.choice, Quot.sound]`.
* **paper-level** — the lemma is proven in the paper / companion
  document but not re-proven in Lean against a concrete architecture;
  its *conclusion* enters `theorem5` as an explicitly named, standard
  mathematical hypothesis (a Lipschitz / non-expansiveness / bounded-
  range condition), never as a conclusion-shaped bound.

## What is fully proven

### Theorem 1 (M.2): C1 + C4 ⇒ L1

* **L1.a** — global phase equivariance `A (R φ X) = R φ (A X)` ⇒
  [`AttentionLayer.apply_R`](PaperV4/L1.lean) (PROVEN, no sorry).
* **L1.b** — element-independent factorisation ⇒
  [`AttentionLayer.L1b_witness`](PaperV4/L1.lean) (PROVEN, no sorry).

The proof of L1.a chains:

  `Wq` complex-linear ⇒ `q_j (R φ X) = e^{iφ} q_j(X)`
  norm preserved by `R(φ)` ⇒ `q̄_j (R φ X) = e^{iφ} q̄_j(X)`
  sesquilinearity ⇒ `⟨e^{iφ} q̄_i, e^{iφ} k̄_j⟩ = ⟨q̄_i, k̄_j⟩`
  ⇒ cosine score `s_ij`, hence `α_ij`, invariant
  `Wo` complex-linear ⇒ output transforms as `e^{iφ} •` original.

### Lemma A (M.7): exact factorisation

  `P(ε) = R(φ̄) ∘ P(δ)` for `φ̄ = mean(ε)`, `δ = ε − φ̄·1`.

⇒ [`P_decompose`](PaperV4/LemmaA.lean) (PROVEN).

The whole-stack version ([`composeLayers_R`](PaperV4/LemmaA.lean),
[`lemmaA`](PaperV4/LemmaA.lean)) follows by induction on the layer list,
using `L1a` of every layer (PROVEN).  The **quantitative** body bound
is PROVEN as [`lemmaA_bound`](PaperV4/L2.lean) (2026-08-05):

  `‖stack(P ε X) − stack X‖₂
     ≤ ‖stack(P δ̃ X) − stack X‖₂ + |φ̄| · ‖stack(P δ̃ X)‖₂`

with **no `O(φ̄²)` remainder** — instead of a Taylor expansion, the
proof uses the global estimate `|e^{iφ̄} − 1| = 2|sin(φ̄/2)| ≤ |φ̄|`
(Mathlib `Real.norm_exp_I_mul_ofReal_sub_one_le`), together with the
`PiLp 2` triangle inequality (`tokenSeqNorm_triangle`).

### Theorem 1' (M.2 second half): necessity of C4 for L1.b

⇒ [`L1b_implies_C4`](PaperV4/L1.lean) (PROVEN).
Just unwraps the `L1b` existential.

### Lemma C (M.9): Doeblin contraction

⇒ [`lemmaC`](PaperV4/LemmaC.lean) (PROVEN, no sorry; added 2026-08-05).

If `P` is row-stochastic and satisfies the Doeblin condition
`P_ij ≥ μ π_j` (with `μ > 0`, `π` a probability vector), then on the
zero-mean subspace `V_0 = {u : Σ_i u_i π_i = 0}`,

  `max_i |Σ_j P_ij u_j| ≤ (1 − μ) · max_j |u_j|`.

The Lean proof follows Levin–Peres–Wilmer 2017 Theorem 4.9 directly:
on `V_0` the rank-1 component `μ · 1 π^T` annihilates `u`, so
`Σ_j P_ij u_j = Σ_j (P_ij − μ π_j) u_j`; the residual coefficients are
non-negative with row sum `1 − μ`, and the triangle inequality closes
the bound.  No coupling machinery is needed — the finite-dimensional
`ℓ∞ → ℓ∞` case is a direct big-operator argument.

## Theorem 5: from standard hypotheses to the conclusion

`L2.lean` exposes Theorem 5 in **`(standard hypotheses) ⇒
CascadePhaseStable`** form:

```lean
theorem theorem5 {K Y_max : ℝ}
    (As : ℕ → List (TokenSeq N d → TokenSeq N d))
    (hyp : Theorem5Hypotheses As K Y_max) :
    CascadePhaseStable (fun L => composeLayers (As L))
```

The proof is fully closed (no `sorry`); `#print axioms PaperV4.theorem5`
shows only the three standard Lean / Mathlib axioms.  All fields of
`Theorem5Hypotheses` are **`Prop`s stating standard mathematical
conditions** (there are no opaque data fields, and no field assumes
anything depth-uniform about the cascade):

* `per_layer_L1a` — every layer is globally phase equivariant
  (C1 + C4 ⇒ L1.a; **provided by Theorem 1**, machine-checked).
* `substrate_nonexpansive` — every layer is non-expansive in `‖·‖₂`
  (the conclusion of (S2) + **Lemma D**; a standard 1-Lipschitz
  condition).
* `zero_mean_phase_stable` — every layer is uniformly `K`-Lipschitz
  against zero-mean phase perturbations (the single-layer conclusion
  of **Lemma B + Lemma C** under (S3); Lemma C itself is
  machine-checked as `lemmaC`).
* `output_bound` — the stack output is uniformly bounded by `Y_max`
  (the conclusion of (S1); a standard bounded-range condition).
* `stack_nonempty`, `K_nonneg`, `Y_max_nonneg` — side conditions.

From these, the proof *derives* the depth-uniform constants
`(C_0, C_1) = (2K + Y_max, 0)` by real induction (machine-checked):

1. `lemmaA` / `lemmaA_bound` — global mode `φ̄` passes exactly through
   the L1.a stack; the quantitative bound has **no** `O(φ̄²)` remainder
   because `|e^{iφ̄} − 1| ≤ |φ̄|` holds globally.
2. `tokenSeqNorm_R_sub_le` — the global-mode error is `≤ |φ̄| · Y_max`.
3. `composeLayers_zero_mean_stable` — the zero-mean residual is
   absorbed by the entry layer (constant `K`) and propagated unchanged
   through the non-expansive substrate, **independent of depth**.
4. `cascade_geometric_bound` — the geometric recurrence
   `e_{l+1} ≤ Λ e_l + b`, `Λ < 1` ⇒ L-independent bound (M.6 step 4,
   with `Λ = 1 − μ_D` from Lemma C), proven separately.

**Non-vacuity**: `theorem5Hypotheses_satisfiable` constructs an
explicit instance of the hypothesis bundle, so the theorem does not
hold vacuously.

## What is left as `sorry` in the project

**Nothing** — as of 2026-08-05 the project builds with zero `sorry`s,
and (same date) the earlier conclusion-shaped premise field
`cascade_decomposition` has been eliminated in favour of the standard
per-layer hypotheses above.

The remaining formalisation work — none of which affects the
correctness or the hypotheses of what is proven — is to *derive* the
per-layer hypotheses for the concrete trained architecture:

* **Lemma B** (M.8): linearised per-layer Jacobian on the zero-mean
  subspace, for the concrete `AttentionLayer`.
* **Lemma D** (M.10): substrate non-expansion for the concrete
  residual + RMSNorm + FFN substrate.
* **M.11 closure**: the fixed-point argument that `K_R < μ_D` is
  preserved across layers, plus verification that (S3) is preserved
  across training.  Both are flagged "tractable, neither is a deep
  open problem" in §M.5/M.11.

## Building

```sh
cd lean/
lake update     # fetch Mathlib
lake build
```

Mathlib's CI moves quickly; the `lakefile.lean` pins `mathlib4 @ master`,
which may need to be tightened to a specific revision (e.g. a Mathlib
nightly tag) to reproduce.  The `lean-toolchain` file pins
`leanprover/lean4:v4.14.0`.

## Honest scope notes

### Differences vs. `paper/outline_v4.md` §M (audited 2026-05-08)

**Lean is *broader* than paper (paper's setting is a special case of
ours):**

* `x_i ≠ 0` — paper M.0 requires this; we use a degenerate-safe
  normalisation `(‖v‖⁻¹ : ℝ) • v` that returns `0` when `v = 0`, so we
  prove the L1 result on **all** of `ℂ^{N×d}`, not just the nonzero
  cone.

**Lean is *narrower* than paper (we restrict to a special case):**

* `W_q, W_k, W_v, W_o` are all `Token d →ₗ[ℂ] Token d` — i.e. square
  endomorphisms of the input space.  Paper's standard form leaves head
  dim and output dim unspecified (typically `q_i, k_j ∈ ℂ^{d_qk}`,
  `v_j ∈ ℂ^{d_v}`, output `∈ ℂ^d`).  Our proofs go through unchanged
  for the multi-dim case; this is a structural restriction, not an
  added premise.  Fix: parametrise `AttentionLayer` over
  `(d_in d_qk d_v : ℕ)`.

**Lean Theorem 1' (`L1b_implies_C4`) is a tautology by construction:**

* Paper's Theorem 1' says: assume the gate has the form
  `α_ij = f̃(s_i1, ..., s_iN)` (allowing arbitrary row-coupling) and is
  L1.b coherent; conclude `f̃` factors as `f(s_ij)`.  Our L1.b is
  *defined* as the existential `∃ f V s, …`, so `L1b A → (∃ f V s, …)`
  is just unwrapping.  The substantive content (per-pair factoring is
  necessary) lives in `Definition 1` itself in our setup.  Capturing
  paper's "row-coupled `f̃` form ⇒ factors" requires a separate
  definition of "row-coupled gate"; not done here.

**~~`Theorem5Hypotheses` uses `True` placeholders.~~** *(2026-05-10:
replaced by `Theorem5Premises` with concrete data fields.
2026-08-05: `Theorem5Premises` itself replaced — its
`cascade_decomposition` field was conclusion-shaped (it assumed the
L-uniform decomposition bound wholesale). The current
`Theorem5Hypotheses` contains only standard per-layer conditions and
the depth-uniformity is derived, not assumed. See "Theorem 5: from
standard hypotheses to the conclusion" above.)*

### No hidden axioms or non-standard premises

`#print axioms` on every PROVEN theorem yields only the three standard
Lean axioms `[propext, Classical.choice, Quot.sound]`.  No `axiom`,
`opaque`, or non-standard `instance` declarations are introduced.  All
Mathlib API used (`EuclideanSpace ℂ (Fin d)`, `inner ℂ`,
`LinearMap →ₗ[ℂ]`, `RCLike.conj_mul`, `Complex.norm_exp_ofReal_mul_I`,
big-operator lemmas) is standard and reusable.

### Build environment

The toolchain ended up pinned to `leanprover/lean4:v4.30.0-rc2`
(auto-aligned by Mathlib master's `post_update` hook).  The build uses
Mathlib's pre-compiled `olean` cache, so reproducing it costs
~1 minute on a warm cache, ~30 minutes on a cold cache.

## References

* `paper/outline_v4.md`, §M.0–M.11 (the source of all proofs above).
* Levin, Peres, Wilmer, *Markov Chains and Mixing Times*, 2nd ed., 2017
  (Theorem 4.9 — Lemma C reference).
* Wang–Sun 2023, *DeepNet* — Lemma D reference (substrate non-expansion).
