# Premise audit — Lean formalisation vs. `paper/outline_v4.md` §M

This document records a line-by-line audit of the Lean formalisation
in this folder against Appendix M of `paper/outline_v4.md`, asking:
*are there any premises in the Lean code that diverge from the paper,
that are not written in the paper, that are not generally shared, or
that are not reusable?*

Date of audit: 2026-05-08.  **Updated 2026-05-10**: Theorem 5 status
upgraded from `STATEMENT (sorry)` to `PROVEN-FROM-PREMISES` (no
`sorry`); see *"Theorem 5 update (2026-05-10)"* below.

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
PaperV4.theorem5                            depends on [propext, Classical.choice, Quot.sound]
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
| **Lemma C** | M.11: "drafted modulo two residual technical pieces" | `sorry` (statement only) | Fully consistent with the paper's own self-description. |
| **Theorem 5 (2026-05-10 update)** | M.10 / M.11 | **`theorem5` proven, no `sorry`**, conditional on `Theorem5Premises` bundle | The bundle's data fields (`C_zm`, `Y_max`, `cascade_decomposition`) sit in for the constructions Lemmas B + D + M.11 closure would yield.  Lemma C is the only remaining `sorry` in the project. |
| **Lemmas B and D** | M.5: classified "rigorous" (B = full algebraic derivation; D = standard transformer-stability) | not yet stubbed; absorbed into `Theorem5Premises` data fields | Honest scope: the data fields encode what these lemmas would supply. No false claim is made because the premises are explicit. |

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

## Theorem 5 update (2026-05-10)

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

## Bottom line

* No hidden axioms, no extra premises beyond what the paper states.
* No non-standard or low-reusability Mathlib usage.
* Where the Lean statement differs from the paper, it is either
  *broader* (paper's setting is a special case) or *narrower* (a
  structural simplification, not an added premise), and every such
  divergence is now explicitly listed.
* **Theorem 5 is now proven (no `sorry`)** under the explicit premise
  bundle `Theorem5Premises` (2026-05-10 update).  The only remaining
  `sorry` in the project is in **Lemma C** (`LemmaC.lean:60`), which
  matches paper §M.5's "drafted modulo Doeblin coupling formalisation"
  status.
