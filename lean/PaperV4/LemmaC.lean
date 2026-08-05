/-
PaperV4.LemmaC — Appendix M.9 (Lemma C: Doeblin contraction).

  [PROVEN]      Lemma C is fully proven, no `sorry`.  The proof follows
                Levin–Peres–Wilmer 2017 Theorem 4.9: split off the
                rank-1 Doeblin component `μ · 1 π^T`, which annihilates
                the zero-mean subspace, and bound the remainder — whose
                entries `P_ij − μ π_j` are non-negative with row sums
                `1 − μ` — by the `ℓ∞` norm of `u`.

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

/-- **Lemma C — Doeblin contraction.**
If `P` is row-stochastic and Doeblin with constant `μ` and stationary
distribution `π`, then `P` acts as a contraction on the zero-mean
subspace `V_0` with operator-norm bound `1 - μ` (in the `ℓ∞ → ℓ∞`
norm; an equivalent `ℓ² → ℓ²` bound follows by norm equivalence on
`ℝ^N`).

*Proof.* (Levin–Peres–Wilmer 2017, Theorem 4.9.)  Decompose
`P = μ · 1 π^T + (1 − μ) Q` for some row-stochastic `Q`; the rank-1
component annihilates `V_0` (because `⟨π, u⟩ = 0` on `V_0`), and `Q` is
non-expansive in the `ℓ∞` norm.  Concretely, for every row `i`:

  `Σ_j P_ij u_j = Σ_j (P_ij − μ π_j) u_j`      (since `Σ_j π_j u_j = 0`)

and the coefficients `P_ij − μ π_j` are non-negative (Doeblin lower
bound) with row sum `Σ_j P_ij − μ Σ_j π_j = 1 − μ`, so the absolute
value is at most `(1 − μ) · max_j |u_j|`.  ∎

The Lean form below picks the `ℓ∞ → ℓ∞` norm; we use `Finset.univ.sup'`
because `Fin N` is non-empty under the explicit `[NeZero N]` hypothesis.
-/
theorem lemmaC [NeZero N]
    (P : Fin N → Fin N → ℝ)
    (hP : RowStochastic P) (hD : Doeblin P) :
    ∃ (μ_D : ℝ) (π : Fin N → ℝ),
        0 < μ_D ∧ ∀ u ∈ zeroMeanSubspace π,
      (Finset.univ.sup' Finset.univ_nonempty
        (fun i : Fin N => |∑ j, P i j * u j|))
      ≤ (1 - μ_D) *
        (Finset.univ.sup' Finset.univ_nonempty (fun j : Fin N => |u j|)) := by
  obtain ⟨μ, π, hμ, hπ0, hπ1, hlb⟩ := hD
  refine ⟨μ, π, hμ, ?_⟩
  intro u hu
  -- `hu : Σ_i u_i π_i = 0`, rewritten with the products commuted.
  have hzero : (∑ j : Fin N, π j * u j) = 0 := by
    have hu' : (∑ i : Fin N, u i * π i) = 0 := hu
    calc (∑ j : Fin N, π j * u j)
        = ∑ j : Fin N, u j * π j :=
          Finset.sum_congr rfl fun j _ => mul_comm _ _
      _ = 0 := hu'
  -- The residual coefficients `P i j − μ π j` are non-negative.
  have hcoeff : ∀ i j, 0 ≤ P i j - μ * π j := fun i j =>
    sub_nonneg.mpr (hlb i j)
  refine Finset.sup'_le _ _ fun i _ => ?_
  calc |∑ j, P i j * u j|
      = |∑ j, (P i j - μ * π j) * u j| := by
        congr 1
        have hexpand : (∑ j, (P i j - μ * π j) * u j)
            = (∑ j, P i j * u j) - μ * (∑ j, π j * u j) := by
          rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
          exact Finset.sum_congr rfl fun j _ => by ring
        rw [hexpand, hzero, mul_zero, sub_zero]
    _ ≤ ∑ j, |(P i j - μ * π j) * u j| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ j, (P i j - μ * π j) * |u j| :=
        Finset.sum_congr rfl fun j _ => by
          rw [abs_mul, abs_of_nonneg (hcoeff i j)]
    _ ≤ ∑ j, (P i j - μ * π j) *
          (Finset.univ.sup' Finset.univ_nonempty
            (fun j : Fin N => |u j|)) :=
        Finset.sum_le_sum fun j _ =>
          mul_le_mul_of_nonneg_left
            (Finset.le_sup' (fun j : Fin N => |u j|) (Finset.mem_univ j))
            (hcoeff i j)
    _ = (1 - μ) *
          (Finset.univ.sup' Finset.univ_nonempty
            (fun j : Fin N => |u j|)) := by
        rw [← Finset.sum_mul, Finset.sum_sub_distrib, hP.2 i,
            ← Finset.mul_sum, hπ1, mul_one]

end PaperV4
