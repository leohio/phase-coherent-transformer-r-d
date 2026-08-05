/-
PaperV4.LemmaA — Appendix M.7 (Lemma A: global-mode decomposition).

  [PROVEN]      The exact factorisation `P(ε) = R(φ̄) ∘ P(δ)` for
                `φ̄ = (1/N) Σ ε_i`, `δ = ε - φ̄ · 1`.
  [PROVEN]      Whole-stack pass-through under `L1a` of every layer.
  [PROVEN]      The full quantitative bound (body §M.7 last paragraph)
                is `lemmaA_bound` in `L2.lean` (where the `ℓ²` norm is
                defined): the global estimate `|e^{iφ̄} − 1| ≤ |φ̄|`
                makes it exact, with no `O(φ̄²)` remainder.

Mirrors:

> Lemma A (global-mode pass-through).  Let ε ∈ ℝ^N and write
> ε = φ̄·1 + δ with φ̄ = (1/N) Σ ε_i and Σ δ_i = 0.  Suppose every A_l
> is per-layer phase-coherent.  Then
>   Ỹ_L = R(φ̄) · (A_L ∘ … ∘ A_1)(P(δ) X).
-/

import PaperV4.Basic
import PaperV4.L1

noncomputable section

namespace PaperV4

variable {N d : ℕ}

/-- The mean of a real-valued angle vector. -/
def angleMean (ε : Fin N → ℝ) : ℝ :=
  (∑ i : Fin N, ε i) / (N : ℝ)

/-- The zero-mean residual `δ = ε − φ̄·1`. -/
def angleResidual (ε : Fin N → ℝ) : Fin N → ℝ :=
  fun i => ε i - angleMean ε

/-- `ε i = φ̄ + δ i` pointwise. -/
lemma angle_decomp (ε : Fin N → ℝ) (i : Fin N) :
    ε i = angleMean ε + angleResidual ε i := by
  unfold angleResidual; ring

/-- **Exact factorisation** (the algebraic core of Lemma A):
`P(ε) = R(φ̄) ∘ P(δ)` for `φ̄ = angleMean ε`, `δ = angleResidual ε`. -/
theorem P_decompose (ε : Fin N → ℝ) (X : TokenSeq N d) :
    P ε X = R (angleMean ε) (P (angleResidual ε) X) := by
  funext i
  show cisR (ε i) • X i = cisR (angleMean ε) • (cisR (angleResidual ε i) • X i)
  rw [smul_smul, ← cisR_add, ← angle_decomp]

/-! ### Global-mode pass-through across an L1 stack -/

/-- A layer stack as a finite list of maps; we only require each to be
per-layer L1.a coherent. -/
def composeLayers (As : List (TokenSeq N d → TokenSeq N d)) :
    TokenSeq N d → TokenSeq N d :=
  As.foldr (fun A acc => A ∘ acc) id

@[simp] lemma composeLayers_nil :
    composeLayers ([] : List (TokenSeq N d → TokenSeq N d)) = id := rfl

@[simp] lemma composeLayers_cons (A : TokenSeq N d → TokenSeq N d) (As) :
    composeLayers (A :: As) = A ∘ composeLayers As := rfl

/-- **Global-mode commutation**: if every layer in the stack satisfies
`L1a`, then the *whole stack* commutes with `R(φ)`. -/
lemma composeLayers_R
    (As : List (TokenSeq N d → TokenSeq N d))
    (h : ∀ A ∈ As, L1a A) (φ : ℝ) (X : TokenSeq N d) :
    composeLayers As (R φ X) = R φ (composeLayers As X) := by
  induction As with
  | nil => simp
  | cons A As ih =>
    have hA : L1a A := h A (List.mem_cons_self)
    have hAs : ∀ A' ∈ As, L1a A' :=
      fun A' hA' => h A' (List.mem_cons_of_mem A hA')
    -- (A ∘ composeLayers As) (R φ X) = R φ ((A ∘ composeLayers As) X)
    simp only [composeLayers_cons, Function.comp_apply]
    rw [ih hAs, hA φ]

/-- **Lemma A — global-mode pass-through (Lean form).**
For an L1 stack `A_L ∘ … ∘ A_1` and any phase vector `ε`, the perturbed
output factors as `R(φ̄) · (stack)(P(δ) X)`. -/
theorem lemmaA
    (As : List (TokenSeq N d → TokenSeq N d))
    (hL1a : ∀ A ∈ As, L1a A) (ε : Fin N → ℝ) (X : TokenSeq N d) :
    composeLayers As (P ε X) =
      R (angleMean ε) (composeLayers As (P (angleResidual ε) X)) := by
  rw [P_decompose ε X]
  exact composeLayers_R As hL1a (angleMean ε) (P (angleResidual ε) X)

/-! ### Quantitative bound (body §M.7)

The body's quantitative bound

  `‖Ỹ_L − Y_L‖₂ ≤ ‖(stack)(P(δ)X) − Y_L‖₂ + |φ̄|·‖(stack)(P(δ)X)‖₂`

is proven as `lemmaA_bound` in `L2.lean` (which defines the `ℓ²` norm
`tokenSeqNorm`).  No Taylor remainder is needed: the global estimate
`|e^{iφ̄} − 1| ≤ |φ̄|` (`norm_cisR_sub_one_le`) is exact for all `φ̄`. -/

end PaperV4
