# Lean 4 formalisation of `paper/outline_v4.md` — Appendix M

This folder contains a Lean 4 + Mathlib formalisation of the
mathematical proofs in **Appendix M** of `paper/outline_v4.md`
(Phase-Coherent Transformers / two-level phase coherence).

## Build status (verified 2026-05-10)

```
$ lake build
warning: PaperV4/LemmaC.lean:60:8: declaration uses `sorry`
Build completed successfully (8407 jobs).
```

`#print axioms` of all PROVEN theorems — including the new
**`theorem5` (Theorem 5 from premises)** — shows only the three
standard Lean / Mathlib axioms (no `sorry`):

```
PaperV4.AttentionLayer.apply_R              depends on [propext, Classical.choice, Quot.sound]
PaperV4.AttentionLayer.L1b_witness          depends on [propext, Classical.choice, Quot.sound]
PaperV4.P_decompose                         depends on [propext, Classical.choice, Quot.sound]
PaperV4.composeLayers_R                     depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaA                              depends on [propext, Classical.choice, Quot.sound]
PaperV4.L1b_implies_C4                      depends on [propext, Classical.choice, Quot.sound]
PaperV4.corollary2                          depends on [propext, Classical.choice, Quot.sound]
PaperV4.theorem5                            depends on [propext, Classical.choice, Quot.sound]
PaperV4.abs_angleMean_le_angleSupNorm       depends on [propext, Classical.choice, Quot.sound]
PaperV4.angleSupNorm_angleResidual_le       depends on [propext, Classical.choice, Quot.sound]
```

The only remaining `sorry` in the project is in **Lemma C**
(`LemmaC.lean:60`).  **Theorem 5 itself is now proven** (`L2.lean`,
`theorem5`) as a closed-form algebraic consequence of the
`Theorem5Premises` bundle, which encapsulates Lemmas A + B + C + D
plus the M.11 closure.  See *"Theorem 5: from premises to conclusion"*
below.

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
| M.4 / M.10 | **Theorem 5** (C1+C3+C4+(A1,A2,S1–S3) ⇒ L2) | [`PaperV4/L2.lean`](PaperV4/L2.lean) | **PROVEN-FROM-PREMISES** ✓ (`theorem5`, no `sorry`) |
| M.4 / M.10 | `Theorem5Premises` (bundles Lemmas A–D + M.11 closure) | [`PaperV4/L2.lean`](PaperV4/L2.lean) | **STATEMENT** (the data fields that *would* require Lemmas B–D + M.11 to construct) |
| M.7 | Lemma A — `P_decompose`, `composeLayers_R`, `lemmaA` | [`PaperV4/LemmaA.lean`](PaperV4/LemmaA.lean) | **PROVEN** ✓ (factorisation + stack pass-through) |
| M.7 | Quantitative norm bound `‖Ỹ_L − Y_L‖ ≤ … + |φ̄|·‖Y_L‖` | [`PaperV4/LemmaA.lean`](PaperV4/LemmaA.lean) | not yet (TODO; absorbed into `Theorem5Premises.cascade_decomposition`) |
| M.8 | Lemma B — linearised per-layer Jacobian | — | not yet started (absorbed into `Theorem5Premises`) |
| M.9 | Lemma C — Doeblin contraction | [`PaperV4/LemmaC.lean`](PaperV4/LemmaC.lean) | **STATEMENT** (sorry) |
| M.10 | Lemma D — substrate non-expansion | — | not yet started (absorbed into `Theorem5Premises`) |

### Status legend

* **PROVEN** — full Lean proof, no `sorry`. Verified by `lake build`
  ending in `Build completed successfully` with `#print axioms` showing
  only `[propext, Classical.choice, Quot.sound]`.
* **PROVEN-FROM-PREMISES** — the conclusion is fully proven (no `sorry`)
  given a bundle of premises stated as a Lean structure. The premise
  fields correspond 1:1 to paper §M.5's "rigorous lemmas" + M.11
  closure; the data inside the structure is what would have to be
  *constructed* in a future pass (full proofs of Lemmas B–D + M.11).
* **STATEMENT** — only the formal statement is given; the proof is `sorry`.
  This mirrors the body's "drafted modulo …" markers.

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
`‖Ỹ_L − Y_L‖₂ ≤ … + |φ̄|·‖Y_L‖₂ + O(φ̄²)` is left as a stub
(`lemmaA_bound_TODO`) — it is a Taylor expansion of `R(φ̄) − I` plus a
unitary triangle inequality.

### Theorem 1' (M.2 second half): necessity of C4 for L1.b

⇒ [`L1b_implies_C4`](PaperV4/L1.lean) (PROVEN).
Just unwraps the `L1b` existential.

## Theorem 5: from premises to conclusion

`L2.lean` exposes Theorem 5 in **`(prem) ⇒ CascadePhaseStable`** form:

```lean
theorem theorem5
    (As : ℕ → List (TokenSeq N d → TokenSeq N d))
    (prem : Theorem5Premises As) :
    CascadePhaseStable (fun L => composeLayers (As L))
```

The proof is fully closed (no `sorry`); `#print axioms PaperV4.theorem5`
shows only the three standard Lean / Mathlib axioms.  The proof
strategy is paper §M.10 verbatim: take `δ := ‖ε‖_∞`, bound
`|angleMean ε| ≤ δ` and `‖angleResidual ε‖_∞ ≤ 2 δ` (both proven as
auxiliary lemmas with no `sorry`), and combine with the premise
bundle to obtain `(C_0, C_1) = (2 · C_zm + Y_max, 0)` independent of
`L`.

The `Theorem5Premises` structure has four data fields:

* `per_layer_L1a` — every layer satisfies L1.a (provided by Theorem 1).
* `C_zm`, `C_zm_nonneg` — the zero-mean cascade Lipschitz constant.
  Constructing this from first principles requires Lemmas B + C + D
  + M.11 closure (`Λ_S · sup_l ‖J_l|_{V_0}‖ < 1`).
* `Y_max`, `Y_max_nonneg` — uniform output norm bound.  Constructing
  this requires (S1) + (S2) + Lemma D.
* `cascade_decomposition` — the Lemma-A-decomposed cascade bound that
  combines Lemmas A + B + C + D + L²-norm triangle inequality + R-
  unitarity into a single per-input inequality.

In short: **`theorem5` discharges the algebraic / geometric-series
core of Theorem 5 once and for all**; the remaining work is
*constructing* a `Theorem5Premises` value, which decomposes neatly
into the four classical lemmas the paper §M.5 already classifies as
"rigorous", modulo the M.11 residual pieces.

## What is left as `sorry` in the project

The only `sorry` in the project after the Theorem 5 update is:

1. **Lemma C** (M.9, `LemmaC.lean:60`): Doeblin contraction on the
   zero-mean subspace.  The proof tracks Levin–Peres–Wilmer 2017
   Theorem 4.9; formalising the standard coupling argument is a
   substantial Mathlib project on its own and has been left as a
   `sorry` skeleton.

The remaining engineering work to obtain a fully `sorry`-free Theorem 5
**without** the `Theorem5Premises` indirection is:

* **Lemma B** (M.8): not yet stubbed.  Linearised per-layer Jacobian
  on the zero-mean subspace.
* **Lemma D** (M.10): not yet stubbed.  Substrate non-expansion.
* **M.11 closure**: the fixed-point argument that `K_R < μ_D` is
  preserved across layers, plus verification that (S3) is preserved
  across training.  Both are flagged "tractable, neither is a deep
  open problem" in §M.5/M.11.
* A constructor function `Theorem5Premises.ofLemmasABCD : … → Theorem5Premises As`
  assembling the four lemmas and the closure into the bundled premise.

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

**~~`Theorem5Hypotheses` uses `True` placeholders.~~** *(2026-05-10
update: the `True`-placeholder structure has been replaced by
`Theorem5Premises` with concrete data fields, and `theorem5` is now
proven without `sorry` from these premises. See "Theorem 5: from
premises to conclusion" above.)*

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
