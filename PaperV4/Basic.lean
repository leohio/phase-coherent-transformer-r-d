/-
PaperV4.Basic — Appendix M.0 (Setting and notation).

Mirrors:

> - Token sequence  X = (x_1, …, x_N) ∈ ℂ^{N×d}.
> - Global phase shift R(φ) : X ↦ (e^{iφ} x_1, …, e^{iφ} x_N) for φ ∈ ℝ.
> - Per-token phase shift P(ε) : X ↦ (e^{iε_i} x_i)_i for ε ∈ ℝ^N.

We use Mathlib's `EuclideanSpace ℂ (Fin d)` for ℂ^d (a finite-dimensional
complex inner product space) and `Fin N → EuclideanSpace ℂ (Fin d)` for
the token sequence. All definitions in this file are `[PROVEN]`.
-/

import Mathlib

noncomputable section

open Complex
open scoped ComplexConjugate

namespace PaperV4

/-- A single token: a vector in ℂ^d. -/
abbrev Token (d : ℕ) : Type := EuclideanSpace ℂ (Fin d)

/-- A token sequence of length `N` with feature dimension `d`. -/
abbrev TokenSeq (N d : ℕ) : Type := Fin N → Token d

variable {N d : ℕ}

/-- The unit-modulus complex scalar `e^{iφ}`. -/
@[simp] def cisR (φ : ℝ) : ℂ := Complex.exp ((φ : ℂ) * Complex.I)

@[simp] lemma cisR_zero : cisR 0 = 1 := by
  simp [cisR]

lemma cisR_add (φ ψ : ℝ) : cisR (φ + ψ) = cisR φ * cisR ψ := by
  unfold cisR
  rw [← Complex.exp_add]
  congr 1
  push_cast
  ring

/-- `|e^{iφ}| = 1`. -/
@[simp] lemma norm_cisR (φ : ℝ) : ‖cisR φ‖ = 1 := by
  unfold cisR
  exact Complex.norm_exp_ofReal_mul_I φ

/-- `(e^{iφ})^* · e^{iφ} = 1` (used heavily in Theorem 1).
For the complex field, `star = conj`, and the standard identity
`conj z * z = ‖z‖^2` from `RCLike.conj_mul` reduces the LHS to
`‖cisR φ‖^2 = 1^2 = 1`. -/
@[simp] lemma star_cisR_mul_cisR (φ : ℝ) : star (cisR φ) * cisR φ = 1 := by
  rw [show star (cisR φ) = conj (cisR φ) from rfl,
      RCLike.conj_mul, norm_cisR]
  norm_num

/-- Global phase shift `R(φ) X = e^{iφ} • X`, applied tokenwise. -/
def R (φ : ℝ) (X : TokenSeq N d) : TokenSeq N d :=
  fun i => cisR φ • X i

/-- Per-token phase shift `P(ε) X = (e^{iε_i} x_i)_i`. -/
def P (ε : Fin N → ℝ) (X : TokenSeq N d) : TokenSeq N d :=
  fun i => cisR (ε i) • X i

@[simp] lemma R_zero (X : TokenSeq N d) : R 0 X = X := by
  funext i; simp [R]

@[simp] lemma P_zero (X : TokenSeq N d) : P (fun _ => (0 : ℝ)) X = X := by
  funext i; simp [P]

/-- `R` is a one-parameter group: `R(φ + ψ) = R(φ) ∘ R(ψ)`. -/
lemma R_add (φ ψ : ℝ) (X : TokenSeq N d) :
    R (φ + ψ) X = R φ (R ψ X) := by
  funext i
  simp only [R, cisR_add, mul_smul]

/-- `R(φ)` written as `P` of the constant-`φ` sequence. -/
lemma R_eq_P_const (φ : ℝ) (X : TokenSeq N d) :
    R φ X = P (fun _ => φ) X := by
  funext i; simp [R, P]

/-- Two `P`'s compose by adding the angle vectors pointwise. -/
lemma P_add (ε ε' : Fin N → ℝ) (X : TokenSeq N d) :
    P (fun i => ε i + ε' i) X = P ε (P ε' X) := by
  funext i
  simp only [P, cisR_add, mul_smul]

/-- `R(φ)` is unitary (norm-preserving) on each token. -/
@[simp] lemma R_norm (φ : ℝ) (X : TokenSeq N d) (i : Fin N) :
    ‖R φ X i‖ = ‖X i‖ := by
  simp [R, norm_smul]

/-- `P(ε)` is unitary on each token. -/
@[simp] lemma P_norm (ε : Fin N → ℝ) (X : TokenSeq N d) (i : Fin N) :
    ‖P ε X i‖ = ‖X i‖ := by
  simp [P, norm_smul]

end PaperV4
