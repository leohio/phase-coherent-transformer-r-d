/-
PaperV4.LemmaC — Appendix M.9 (Lemma C: Doeblin contraction).

  [STATEMENT]   Lemma C is given as a Lean statement only.  The proof
                tracks Levin–Peres–Wilmer 2017 Theorem 4.9; formalising
                the standard Doeblin coupling argument is a substantial
                Mathlib project on its own and is left as a `sorry`.

Mirrors:

> Lemma C (Doeblin contraction).  Suppose the gate satisfies (C1)–(C4)
> and (S3) — there exist μ > 0 and a probability vector π such that
> α_ij ≥ μ · π_j for all i, j and all inputs in the bounded set.
> Define the row-stochastic matrix P by P_{ij} := α_ij / Σ_k α_ik.
> Then P satisfies the Doeblin condition P_{ij} ≥ μ_D · π_j with
> μ_D := μ / M, and the operator norm of P restricted to the zero-mean
> subspace V_0 := {u ∈ ℝ^N : Σ_i u_i · w_i = 0} is bounded by
> ‖P|_{V_0}‖_{∞→∞} ≤ 1 − μ_D.
-/

import PaperV4.Basic

noncomputable section

namespace PaperV4

variable {N : ℕ}

/-- A matrix `P : Fin N → Fin N → ℝ` is **row-stochastic** if every row
is nonneg and sums to 1. -/
def RowStochastic (P : Fin N → Fin N → ℝ) : Prop :=
  (∀ i j, 0 ≤ P i j) ∧ (∀ i, (∑ j : Fin N, P i j) = 1)

/-- The **Doeblin condition** for `P`: there exist `μ > 0` and a probability
vector `π` such that every entry `P i j ≥ μ · π_j`. -/
def Doeblin (P : Fin N → Fin N → ℝ) : Prop :=
  ∃ (μ : ℝ) (π : Fin N → ℝ),
    0 < μ ∧ (∀ j, 0 ≤ π j) ∧ (∑ j : Fin N, π j) = 1 ∧
    (∀ i j, μ * π j ≤ P i j)

/-- The **zero-mean subspace** of `ℝ^N` with respect to weights `π`. -/
def zeroMeanSubspace (π : Fin N → ℝ) : Set (Fin N → ℝ) :=
  {u | (∑ i : Fin N, u i * π i) = 0}

/-- **Lemma C — Doeblin contraction (statement).**
If `P` is row-stochastic and Doeblin with constant `μ` and stationary
distribution `π`, then `P` acts as a contraction on the zero-mean
subspace `V_0` with operator-norm bound `1 - μ` (in the `ℓ∞ → ℓ∞`
norm; an equivalent `ℓ² → ℓ²` bound follows by norm equivalence on
`ℝ^N`).

*Proof.* (Levin–Peres–Wilmer 2017, Theorem 4.9.)  Decompose
`P = μ · 1 π^T + (1 − μ) Q` for some row-stochastic `Q`; the rank-1
component annihilates `V_0` (because `⟨π, u⟩ = 0` on `V_0`), and `Q` is
non-expansive in the `ℓ∞` norm.  ∎

The Lean form below picks the `ℓ∞ → ℓ∞` norm; we use `Finset.univ.sup'`
because `Fin N` is non-empty under the explicit `[NeZero N]` hypothesis.
-/
theorem lemmaC [NeZero N]
    (P : Fin N → Fin N → ℝ)
    (_hP : RowStochastic P) (_hD : Doeblin P) :
    ∃ (μ_D : ℝ) (π : Fin N → ℝ),
        0 < μ_D ∧ ∀ u ∈ zeroMeanSubspace π,
      (Finset.univ.sup' Finset.univ_nonempty
        (fun i : Fin N => |∑ j, P i j * u j|))
      ≤ (1 - μ_D) *
        (Finset.univ.sup' Finset.univ_nonempty (fun j : Fin N => |u j|)) := by
  -- Standard coupling argument; proof omitted in this skeleton.
  sorry

end PaperV4
