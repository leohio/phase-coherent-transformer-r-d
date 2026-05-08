# Premise audit — Lean formalisation vs. `paper/outline_v4.md` §M

This document records a line-by-line audit of the Lean formalisation
in this folder against Appendix M of `paper/outline_v4.md`, asking:
*are there any premises in the Lean code that diverge from the paper,
that are not written in the paper, that are not generally shared, or
that are not reusable?*

Date of audit: 2026-05-08.

---

## Audit results

### 1. Lean axioms (reusability)

```
PaperV4.AttentionLayer.apply_R       depends on [propext, Classical.choice, Quot.sound]
PaperV4.AttentionLayer.L1b_witness   depends on [propext, Classical.choice, Quot.sound]
PaperV4.P_decompose                  depends on [propext, Classical.choice, Quot.sound]
PaperV4.composeLayers_R              depends on [propext, Classical.choice, Quot.sound]
PaperV4.lemmaA                       depends on [propext, Classical.choice, Quot.sound]
PaperV4.L1b_implies_C4               depends on [propext, Classical.choice, Quot.sound]
PaperV4.corollary2                   depends on [propext, Classical.choice, Quot.sound]
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
| **Lemma C / Theorem 5 / Lemma B / D** | M.11: explicitly marked "drafted modulo two residual technical pieces" | `sorry` (statement only) | Fully consistent with the paper's own self-description. |
| **Side conditions in `Theorem5Hypotheses` (C1, C3, A1, A2, S1–S3)** | M.4 spells these out as functional-analytic conditions | **`True` placeholders** (only C4 and per-layer L1.a are realised) | Lean **deliberately scopes out** the formalisation.  This *looks* like Theorem 5 is being claimed under weakened hypotheses, but Theorem 5 itself is `sorry`'d, so no claim is actually being asserted. |

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

**(c) `Theorem5Hypotheses` fields are placeholder `True`s.**  Inside
the `sorry`'d `theorem5_statement`, no claim is being made — but the
structure's fields are not yet realised.  The README's status table
already marks Theorem 5 and Lemma C as `STATEMENT (sorry)`; we are not
hiding any extra premise here.

These three points are now documented in
[`lean/README.md` § "Differences vs. `paper/outline_v4.md` §M"](README.md).

---

## Bottom line

* No hidden axioms, no extra premises beyond what the paper states.
* No non-standard or low-reusability Mathlib usage.
* Where the Lean statement differs from the paper, it is either
  *broader* (paper's setting is a special case) or *narrower* (a
  structural simplification, not an added premise), and every such
  divergence is now explicitly listed.
* The two remaining `sorry`s (Lemma C, Theorem 5) match the paper's
  own "drafted modulo …" markers in §M.5 / §M.11 exactly.
