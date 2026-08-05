# Premise audit — Lean formalisation vs. `paper/outline_v4.md` §M

This document records a line-by-line audit of the Lean formalisation
in this folder against Appendix M of `paper/outline_v4.md`, asking:
*are there any premises in the Lean code that diverge from the paper,
that are not written in the paper, that are not generally shared, or
that are not reusable?*

Date of audit: 2026-05-08.  **Updated 2026-05-10**: Theorem 5 status
upgraded from `STATEMENT (sorry)` to `PROVEN-FROM-PREMISES` (no
`sorry`); see *"Theorem 5 update (2026-05-10)"* below.
**Updated 2026-08-05 (a)**: Lemma C status upgraded from
`STATEMENT (sorry)` to `PROVEN` (no `sorry`); the project now contains
**zero `sorry`s**.  See *"Lemma C update (2026-08-05)"* below.
**Updated 2026-08-05 (b)**: the conclusion-shaped premise
`Theorem5Premises.cascade_decomposition` — the one construct a referee
could fairly call a *non-community-shared assumption* — has been
**eliminated**.  `theorem5` is now proven from standard per-layer
hypotheses only, with the depth-uniformity derived by induction.  See
*"Theorem 5 hypothesis-standardisation update (2026-08-05)"* below.

---

## Audit results

### 1. Lean axioms (reusability)

```
PaperV4.AttentionLayer.apply_R              depends on [propext, Classical.choice, Quot.sound]
PaperV4.AttentionLayer.L1b_witness          depends on [propext, Classical.choice, Quot.sound]
PaperV4.P_decompose                         depends on [propext, Classical.choice, Quot.sound]
PaperV4.composeLayers_R                     depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaA                              depends on [propext, Classical.choice, Quot.sound]
PaperV4.L1b_implies_C4                      depends on [propext, Classical.choice, Quot.sound]
PaperV4.corollary2                          depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaC                              depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaA_bound                        depends on [propext, Classical.choice, Quot.sound]
PaperV4.tokenSeqNorm_triangle               depends on [propext, Classical.choice, Quot.sound]
PaperV4.tokenSeqNorm_R_sub_le               depends on [propext, Classical.choice, Quot.sound]
PaperV4.composeLayers_zero_mean_stable      depends on [propext, Classical.choice, Quot.sound]
PaperV4.cascade_geometric_bound             depends on [propext, Classical.choice, Quot.sound]
PaperV4.theorem5                            depends on [propext, Classical.choice, Quot.sound]
PaperV4.theorem5Hypotheses_satisfiable      depends on [propext, Classical.choice, Quot.sound]
PaperV4.abs_angleMean_le_angleSupNorm       depends on [propext, Classical.choice, Quot.sound]
PaperV4.angleSupNorm_angleResidual_le       depends on [propext, Classical.choice, Quot.sound]
```

These are the **three foundational axioms of all of Lean / Mathlib**
(propositional extensionality, axiom of choice, quotient soundness).
Every theorem in Mathlib depends on these, so reusability and shared
acceptance are maximal.  **No additional `axiom` or `constant`
declarations have been introduced.**

### 2. Diff audit against the paper

| Aspect | paper §M | Lean (current) | Verdict |
| --- | --- | --- | --- |
| **`x_i ≠ 0`** | M.0 explicitly requires this | not required (`(‖v‖⁻¹ : ℝ) • v` returns `0` when `v = 0`; degenerate-safe normalisation) | Lean proves a result on a **broader** setting than the paper — paper's claim is a special case.  Fine. |
| **Dimensions of `W_q, W_k, W_v, W_o`** | left unspecified (typically head dim ≠ output dim) | **all `Token d →ₗ[ℂ] Token d` (square)** | Lean is **narrower** ⚠️ — a structural simplification.  The proofs go through unchanged for the multi-dim case; this is a restriction, not an added premise. |
| **Inner product convention** | `Re⟨q̄_i, k̄_j⟩` (convention left unspecified) | Mathlib's `inner ℂ` (sesquilinear in the first argument) | Sesquilinear cancellation works the same way under either convention; matches the paper's proof. |
| **Definition of L1.b** | "there exist `f, V, s` such that …" (natural-language existential) | `Prop := ∃ f V s, …` (same shape) | Exact match. |
| **Content of Theorem 1'** | "if a row-coupled `f̃(s_i1, …, s_iN)` is L1.b, then `f̃` factors per-pair" | `L1b A → ∃ f V s, expand` | The Lean version is a **tautology that just unwraps the definition of L1.b**.  The paper's version has more content because it assumes a richer gate form and concludes factoring.  **The semantic content is the same**, but the Lean statement is weaker. |
| **Lemma C (2026-08-05 update)** | M.9: Doeblin contraction, Levin–Peres–Wilmer Thm 4.9 | **`lemmaC` proven, no `sorry`** — direct rank-1-decomposition argument, no coupling machinery needed in the finite-dimensional `ℓ∞` case | Lean now proves exactly what the paper cites. The proof introduces no premises beyond `RowStochastic` + `Doeblin`, both stated verbatim from M.9. |
| **Theorem 5 (2026-08-05 update)** | M.10 / M.11 | **`theorem5` proven, no `sorry`**, from `Theorem5Hypotheses` — all fields are standard `Prop` conditions (L1.a, non-expansiveness, uniform zero-mean phase Lipschitz stability, bounded range) | **No conclusion-shaped or bespoke premise remains.** The depth-uniform constants are derived by induction; the hypothesis set is proven satisfiable (`theorem5Hypotheses_satisfiable`). |
| **Quantitative Lemma A bound (2026-08-05 update)** | M.7 last paragraph (`‖Ỹ_L − Y_L‖ ≤ … + \|φ̄\|·‖·‖ + O(φ̄²)`) | **`lemmaA_bound` proven, no `sorry`**, and *stronger* than the paper: no `O(φ̄²)` remainder (uses the global bound `\|e^{iφ}−1\| ≤ \|φ\|`) | Lean proves a cleaner statement than the paper's sketch. |
| **Lemmas B and D** | M.5: classified "rigorous" (B = full algebraic derivation; D = standard transformer-stability) | not re-proven in Lean against a concrete architecture; their single-layer conclusions are the named standard hypotheses `zero_mean_phase_stable` / `substrate_nonexpansive` of `theorem5` | Honest scope: the hypotheses are exactly the lemmas' conclusions, stated as ordinary Lipschitz/boundedness conditions. No hidden or non-standard content. |

### 3. Are any non-standard or low-reusability constructs used?

**No.**  All Mathlib API used is standard:

* `EuclideanSpace ℂ (Fin d)` — standard finite-dimensional complex Hilbert space.
* `LinearMap →ₗ[ℂ]` — standard complex-linear maps.
* `inner ℂ`, `inner_smul_left/right` — `Mathlib.Analysis.InnerProductSpace.Basic`.
* `RCLike.conj_mul`, `Complex.norm_exp_ofReal_mul_I` —
  `Mathlib.Analysis.RCLike.Basic`, `Mathlib.Analysis.Complex.Trigonometric`.
* `Finset.smul_sum`, `map_sum`, `LinearMap.map_smul` — standard
  big-operator / linear-algebra lemmas.

No custom `axiom`, no `opaque`, and no non-standard `instance`
declarations are introduced.

### 4. Divergences that genuinely need to be flagged

**(a) Square `W` matrices.**  The paper's notation does not commit to
specific dimensions for the head space, value space, or output space;
the Lean code restricts all four `W` maps to square endomorphisms of
`Token d`.  The proof strategy is identical for the multi-dim case, so
this can be repaired by parametrising `AttentionLayer` over
`(d_in d_qk d_v d_out : ℕ)`.  This is "Lean proves a narrower
statement," not "Lean introduces an extra premise the paper lacks."

**(b) Theorem 1' Lean statement is weaker than the paper's.**
Formalising the paper's full Theorem 1' ("any row-coupled gate that is
L1.b necessarily factors per-pair") requires a separate definition of
*row-coupled gate form* and an argument relating it to L1.b.  Currently
the Lean theorem is just an unwrapping of the L1.b existential.

**(c) ~~`Theorem5Hypotheses` fields are placeholder `True`s.~~** *(superseded by 2026-05-10 Theorem 5 update — see below.)*

These three points are now documented in
[`lean/README.md` § "Differences vs. `paper/outline_v4.md` §M"](README.md).

---

## Theorem 5 hypothesis-standardisation update (2026-08-05)

**Motivation.**  The 2026-05-10 `Theorem5Premises` bundle contained the
field `cascade_decomposition`, which assumed — for every depth `L` —
the L-uniform Lemma-A-decomposed error bound.  That field was
*conclusion-shaped*: it packaged most of what Theorem 5 asserts into a
single bespoke premise.  While explicitly documented, it was the one
construct in the project that a referee could fairly call an
assumption "not shared by the mathematical community".

**What changed.**  `L2.lean` was rewritten again:

* `Theorem5Premises` (and `cascade_decomposition`) are **removed**.
* The new `Theorem5Hypotheses` is a `Prop` structure whose fields are
  exclusively standard mathematical conditions:
  L1.a equivariance (Theorem 1's conclusion), non-expansiveness
  (1-Lipschitz), uniform `K`-Lipschitz stability against zero-mean
  phase perturbations, uniform output bound `Y_max`, depth ≥ 1, and
  non-negativity of the constants.  **No field mentions the cascade,
  the depth, or any L-uniform quantity.**
* The depth-uniformity of `(C_0, C_1) = (2K + Y_max, 0)` is **derived**:
  - `lemmaA_bound` (new, proven): quantitative Lemma A split with no
    `O(φ̄²)` remainder, via `|e^{iφ}−1| ≤ |φ|` and the `PiLp 2`
    triangle inequality (`tokenSeqNorm_triangle`, inherited from
    Mathlib, not re-axiomatised).
  - `composeLayers_zero_mean_stable` (new, proven): list induction —
    entry layer absorbs the zero-mean perturbation, non-expansive
    layers propagate it without growth.
  - `cascade_geometric_bound` (new, proven): the M.6 step-4 geometric
    recurrence `e_{l+1} ≤ Λ e_l + b`, `Λ < 1` ⇒ L-independent bound.
* `theorem5Hypotheses_satisfiable` (new, proven): explicit witness that
  the hypothesis set is consistent — the theorem is not vacuous.

**Audit consequence.**  Every hypothesis of every theorem in the
project is now either (i) a definition mirroring the paper's stated
setting, or (ii) a textbook-standard mathematical condition
(linearity, row-stochasticity, Doeblin minorisation, Lipschitz /
non-expansiveness, uniform phase-Lipschitz stability, bounded range).
**Zero `sorry`s, zero bespoke premises.**

---

## Theorem 5 update (2026-05-10) — historical

*(Superseded by the 2026-08-05 hypothesis-standardisation update
above; kept for the audit trail.)*

**What changed.**  `L2.lean` was rewritten:

* `Theorem5Hypotheses` (with `True` placeholders) and the `sorry`'d
  `theorem5_statement` were removed.
* They are replaced by a concrete data structure `Theorem5Premises` and
  a fully proven theorem `theorem5 : Theorem5Premises As → CascadePhaseStable …`.
* `theorem5` has *no* `sorry`; `#print axioms PaperV4.theorem5` shows
  only `[propext, Classical.choice, Quot.sound]`.
* Two new auxiliary lemmas — `abs_angleMean_le_angleSupNorm` and
  `angleSupNorm_angleResidual_le` — are also proven without `sorry`.

**What this means for the audit.**

* No new axioms or non-standard premises are introduced.  The four
  data fields of `Theorem5Premises` (`per_layer_L1a`, `C_zm`, `Y_max`,
  `cascade_decomposition`) are stated as ordinary Lean data, with the
  numerical fields constrained to be non-negative.
* `cascade_decomposition` is the per-input bound that combines paper
  §M.5's "rigorous" Lemmas A + B + C + D + L²-norm triangle inequality
  + R-unitarity + the M.11 closure.  The structure's docstring lists
  each contribution explicitly.
* The honest scope is now: **Theorem 5 is closed under the assumption
  that one can construct a `Theorem5Premises As` value** — i.e.
  produce concrete witnesses for `C_zm` and `Y_max` plus a proof of
  `cascade_decomposition`.  This is exactly the work paper §M.5
  classifies as "rigorous, modulo M.11", and matches the paper's own
  status table verbatim.

**Bottom line for this update.**  Theorem 5 in the Lean development
is now in `(prem) ⇒ conclusion` form with the conclusion fully
machine-checked.  The remaining engineering work is the *construction*
of the premise bundle (Lemmas B / C / D / M.11) — none of which the
paper claims as open mathematics, but which all remain to be Lean'd.

---

## Lemma C update (2026-08-05)

**What changed.**  `LemmaC.lean`'s `sorry` was replaced by a complete
proof of `lemmaC`.

* The statement is unchanged (only the unused hypothesis binders
  `_hP`/`_hD` were renamed to `hP`/`hD` since the proof now uses them).
* The proof follows Levin–Peres–Wilmer 2017 Theorem 4.9 but avoids the
  probabilistic coupling formalisation entirely: in the
  finite-dimensional `ℓ∞ → ℓ∞` setting the argument is a direct
  big-operator computation.  For `u` in the zero-mean subspace,
  `Σ_j P_ij u_j = Σ_j (P_ij − μ π_j) u_j` (the rank-1 Doeblin
  component annihilates `u`); the residual coefficients are
  non-negative with row sum `1 − μ`, so
  `|Σ_j P_ij u_j| ≤ (1 − μ) · max_j |u_j|`.
* Mathlib API used: `Finset.sup'_le` / `Finset.le_sup'`,
  `Finset.abs_sum_le_sum_abs`, `Finset.sum_sub_distrib`,
  `Finset.mul_sum` / `Finset.sum_mul` — all standard big-operator
  lemmas.  No new axioms; `#print axioms PaperV4.lemmaC` shows only
  `[propext, Classical.choice, Quot.sound]`.

**What this means for the audit.**  The project now contains **zero
`sorry`s**.  The `Theorem5Premises` bundle is unchanged — Lemma C's
content still enters Theorem 5 through the `cascade_decomposition`
field — but the lemma itself is now machine-checked rather than cited.

---

## Bottom line

* No hidden axioms, no extra premises beyond what the paper states.
* No non-standard or low-reusability Mathlib usage.
* Where the Lean statement differs from the paper, it is either
  *broader* (paper's setting is a special case) or *narrower* (a
  structural simplification, not an added premise), and every such
  divergence is now explicitly listed.
* **Theorem 5 is proven (no `sorry`) from standard per-layer
  hypotheses only** (2026-08-05 hypothesis-standardisation update);
  the conclusion-shaped `cascade_decomposition` premise has been
  eliminated, and the hypothesis set is proven satisfiable.  **Lemma C
  is proven standalone (no `sorry`)** (2026-08-05 update).  The
  project contains **zero `sorry`s and zero non-community-shared
  assumptions**; the remaining Lean-side work — deriving the standard
  per-layer hypotheses for the concrete trained architecture (Lemmas
  B / D + the M.11 closure) — changes what is *instantiated*, not what
  is *assumed*.
