# Lean 4 formalisation of `paper/outline_v4.md` — Appendix M

This folder contains a Lean 4 + Mathlib formalisation of the
mathematical proofs in **Appendix M** of `paper/outline_v4.md`
(Phase-Coherent Transformers / two-level phase coherence).

## Build status (verified 2026-05-08)

```
$ lake build
warning: PaperV4/LemmaC.lean:60:8: declaration uses `sorry`
warning: PaperV4/L2.lean:143:8: declaration uses `sorry`
Build completed successfully (8407 jobs).
```

`#print axioms` of the M.2 / M.7 theorems shows only the three standard
Lean / Mathlib axioms (no `sorry`):

```
PaperV4.AttentionLayer.apply_R       depends on [propext, Classical.choice, Quot.sound]
PaperV4.AttentionLayer.L1b_witness   depends on [propext, Classical.choice, Quot.sound]
PaperV4.P_decompose                  depends on [propext, Classical.choice, Quot.sound]
PaperV4.composeLayers_R              depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaA                       depends on [propext, Classical.choice, Quot.sound]
PaperV4.L1b_implies_C4               depends on [propext, Classical.choice, Quot.sound]
PaperV4.corollary2                   depends on [propext, Classical.choice, Quot.sound]
```

The only two remaining `sorry`s are in **Lemma C** (`LemmaC.lean:60`) and
**Theorem 5** (`L2.lean:143`), which the paper itself marks as "drafted
modulo two residual technical pieces" in §M.11.

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
| M.4 | Conjecture/Theorem 5 (C1+C3+C4+(A1,A2,S1–S3) ⇒ L2) | [`PaperV4/L2.lean`](PaperV4/L2.lean) | **STATEMENT** (sorry) |
| M.7 | Lemma A — `P_decompose`, `composeLayers_R`, `lemmaA` | [`PaperV4/LemmaA.lean`](PaperV4/LemmaA.lean) | **PROVEN** ✓ (factorisation + stack pass-through) |
| M.7 | Quantitative norm bound `‖Ỹ_L − Y_L‖ ≤ … + |φ̄|·‖Y_L‖` | [`PaperV4/LemmaA.lean`](PaperV4/LemmaA.lean) | not yet (TODO) |
| M.8 | Lemma B — linearised per-layer Jacobian | — | not yet started |
| M.9 | Lemma C — Doeblin contraction | [`PaperV4/LemmaC.lean`](PaperV4/LemmaC.lean) | **STATEMENT** (sorry) |
| M.10 | Lemma D — substrate non-expansion | — | not yet started |

### Status legend

* **PROVEN** — full Lean proof, no `sorry`. Verified by `lake build`
  ending in `Build completed successfully` with `#print axioms` showing
  only `[propext, Classical.choice, Quot.sound]`.
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

## What is left as `sorry`

These match the body's own status markers:

1. **Theorem 5** (M.10): cascade phase stability under
   `(C1) + (C3) + (C4) + (A1) + (A2) + (S1) + (S2) + (S3)`.
   The body marks this as *drafted modulo two residual technical pieces*
   (M.11): (i) verification that (S3) is preserved across training,
   (ii) the fixed-point argument that the bounded-input regime where
   `K_R < μ_D` holds is preserved across layers.

2. **Lemma C** (M.9): Doeblin contraction on the zero-mean subspace.
   The proof tracks Levin–Peres–Wilmer 2017 Theorem 4.9; formalising
   the standard coupling argument is a substantial Mathlib project on
   its own and has been left as a `sorry` skeleton.

3. **Lemma B** (M.8) and **Lemma D** (M.10): not yet stubbed.

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

**`Theorem5Hypotheses` uses `True` placeholders** for the side
conditions (C1, C3, A1, A2, S1, S2, S3).  Formalising the full
functional-analytic conditions is on the same footing as formalising
Theorem 5 / Lemmas B–D themselves and was deliberately scoped out for
this first pass.  Since `theorem5_statement` itself is `sorry`'d, no
*claim* is asserted with these placeholders; they are structural
stubs only.

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
