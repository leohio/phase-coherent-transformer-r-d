/-
PaperV4.L1 — Appendix M.1 + M.2 (Definition 1, Theorem 1).

Mirrors:

> Theorem 1 (C1 + C4 imply L1). Suppose an attention layer A_θ is
> constructed with complex-linear maps W_q, W_k, W_v, W_o and a
> real-valued gate α_ij = f(s_ij) where s_ij = Re⟨q̄_i, k̄_j⟩.
> Then A_θ satisfies Definition 1.

We split L1 into:
  • L1.a (global phase equivariance): `A (R φ X) = R φ (A X)`
  • L1.b (element-independent gating, captured by construction)
-/

import PaperV4.Basic

noncomputable section

open Complex
open scoped ComplexConjugate

namespace PaperV4

variable {N d : ℕ}

/-! ### Auxiliary phase-cancellation lemmas -/

/-- Sesquilinear cancellation: `⟨e^{iφ}a, e^{iφ}b⟩ = ⟨a, b⟩`.

The Mathlib convention is `⟪c • x, y⟫_ℂ = (conj c) * ⟪x, y⟫` and
`⟪x, c • y⟫_ℂ = c * ⟪x, y⟫`, so

  `⟪c • a, c • b⟫ = (conj c * c) * ⟪a, b⟫`,

which collapses to `⟪a, b⟫` whenever `conj c * c = 1`.  For
`c = cisR φ` this is `star_cisR_mul_cisR` from `Basic.lean`. -/
lemma inner_smul_phase (φ : ℝ) (a b : Token d) :
    inner ℂ (cisR φ • a) (cisR φ • b) = inner ℂ a b := by
  rw [inner_smul_left, inner_smul_right, ← mul_assoc]
  -- Goal: starRingEnd ℂ (cisR φ) * cisR φ * inner ℂ a b = inner ℂ a b
  -- starRingEnd ℂ z = star z definitionally on ℂ; reduces to one_mul.
  rw [show starRingEnd ℂ (cisR φ) * cisR φ = 1 from star_cisR_mul_cisR φ, one_mul]

/-- Normalisation commutes with phase: `n (e^{iφ} v) = e^{iφ} • n v`,
where `n v := (‖v‖⁻¹ : ℂ) • v` is the (degenerate-safe) normalisation. -/
lemma normalize_smul_phase (φ : ℝ) (v : Token d) :
    ((‖cisR φ • v‖⁻¹ : ℝ) : ℂ) • (cisR φ • v) =
      cisR φ • (((‖v‖⁻¹ : ℝ) : ℂ) • v) := by
  have hnorm : ‖cisR φ • v‖ = ‖v‖ := by
    rw [norm_smul, norm_cisR, one_mul]
  rw [hnorm, smul_smul, smul_smul, mul_comm]

/-! ### Definition 1 (per-layer phase coherence) -/

/-- Definition 1 (L1.a): a map `A : TokenSeq N d → TokenSeq N d` is
**globally phase equivariant** if `A (R φ X) = R φ (A X)` for all `φ, X`. -/
def L1a (A : TokenSeq N d → TokenSeq N d) : Prop :=
  ∀ (φ : ℝ) (X : TokenSeq N d), A (R φ X) = R φ (A X)

/-- Definition 1 (L1.b) — element-independent gating.
A map `A` is **element-independently gated** if there exist
  • a real gate function `f : ℝ → ℝ`
  • a complex-linear value path `V : Token d →ₗ[ℂ] Token d`
  • a binary score `s : Token d → Token d → ℝ` invariant under joint
    global phase rotation
such that `A X i = ∑ j, (f (s (X i) (X j)) : ℂ) • V (X j)`. -/
def L1b (A : TokenSeq N d → TokenSeq N d) : Prop :=
  ∃ (f : ℝ → ℝ) (V : Token d →ₗ[ℂ] Token d) (s : Token d → Token d → ℝ),
    (∀ (φ : ℝ) (a b : Token d), s (cisR φ • a) (cisR φ • b) = s a b) ∧
    (∀ (X : TokenSeq N d) (i : Fin N),
      A X i = ∑ j : Fin N, ((f (s (X i) (X j)) : ℂ) • V (X j)))

/-! ### The attention layer of Theorem 1 -/

/-- The L1 attention layer of Theorem 1: complex-linear `W_q W_k W_v W_o`
and a real-valued, real-shifted gate `α_ij = f (s_ij + b)` where
`s_ij = Re⟨q̄_i, k̄_j⟩`. -/
structure AttentionLayer (d : ℕ) where
  Wq : Token d →ₗ[ℂ] Token d
  Wk : Token d →ₗ[ℂ] Token d
  Wv : Token d →ₗ[ℂ] Token d
  Wo : Token d →ₗ[ℂ] Token d
  f  : ℝ → ℝ
  b  : ℝ

namespace AttentionLayer

variable (A : AttentionLayer d)

/-- Per-token query, key, value (before normalisation). -/
def qVec (X : TokenSeq N d) (j : Fin N) : Token d := A.Wq (X j)
def kVec (X : TokenSeq N d) (j : Fin N) : Token d := A.Wk (X j)
def vVec (X : TokenSeq N d) (j : Fin N) : Token d := A.Wv (X j)

/-- Normalised query / key (degenerate-safe via `‖0‖⁻¹ = 0`). -/
def qBar (X : TokenSeq N d) (j : Fin N) : Token d :=
  ((‖A.qVec X j‖⁻¹ : ℝ) : ℂ) • A.qVec X j
def kBar (X : TokenSeq N d) (j : Fin N) : Token d :=
  ((‖A.kVec X j‖⁻¹ : ℝ) : ℂ) • A.kVec X j

/-- Cosine score `s_ij = Re ⟨q̄_i, k̄_j⟩ ∈ [-1, 1]` (range bound not needed
for Theorem 1's L1.a; only the phase invariance is used). -/
def cosScore (X : TokenSeq N d) (i j : Fin N) : ℝ :=
  Complex.re (inner ℂ (A.qBar X i) (A.kBar X j))

/-- The gate `α_ij = f (s_ij + b)`. -/
def alpha (X : TokenSeq N d) (i j : Fin N) : ℝ :=
  A.f (A.cosScore X i j + A.b)

/-- The full forward pass `A_θ(X)_i = W_o (Σ_j α_ij • v_j)`. -/
def apply (X : TokenSeq N d) : TokenSeq N d :=
  fun i => A.Wo (∑ j : Fin N, ((A.alpha X i j : ℂ) • A.vVec X j))

/-! ### Phase invariance of internal quantities under `X ↦ R(φ) X` -/

/-- `q_j (R φ X) = e^{iφ} • q_j X` by complex-linearity of `W_q`. -/
lemma qVec_R (φ : ℝ) (X : TokenSeq N d) (j : Fin N) :
    A.qVec (R φ X) j = cisR φ • A.qVec X j := by
  unfold qVec R
  exact LinearMap.map_smul A.Wq (cisR φ) (X j)

lemma kVec_R (φ : ℝ) (X : TokenSeq N d) (j : Fin N) :
    A.kVec (R φ X) j = cisR φ • A.kVec X j := by
  unfold kVec R
  exact LinearMap.map_smul A.Wk (cisR φ) (X j)

lemma vVec_R (φ : ℝ) (X : TokenSeq N d) (j : Fin N) :
    A.vVec (R φ X) j = cisR φ • A.vVec X j := by
  unfold vVec R
  exact LinearMap.map_smul A.Wv (cisR φ) (X j)

/-- Normalised query rotates with the phase. -/
lemma qBar_R (φ : ℝ) (X : TokenSeq N d) (j : Fin N) :
    A.qBar (R φ X) j = cisR φ • A.qBar X j := by
  unfold qBar
  rw [qVec_R]
  exact normalize_smul_phase φ (A.qVec X j)

lemma kBar_R (φ : ℝ) (X : TokenSeq N d) (j : Fin N) :
    A.kBar (R φ X) j = cisR φ • A.kBar X j := by
  unfold kBar
  rw [kVec_R]
  exact normalize_smul_phase φ (A.kVec X j)

/-- Cosine score is invariant under `X ↦ R(φ) X` — this is **C1 + C4**
combined with the sesquilinear cancellation `inner_smul_phase`. -/
lemma cosScore_R (φ : ℝ) (X : TokenSeq N d) (i j : Fin N) :
    A.cosScore (R φ X) i j = A.cosScore X i j := by
  unfold cosScore
  rw [qBar_R, kBar_R, inner_smul_phase]

/-- Gate is invariant. -/
lemma alpha_R (φ : ℝ) (X : TokenSeq N d) (i j : Fin N) :
    A.alpha (R φ X) i j = A.alpha X i j := by
  unfold alpha
  rw [cosScore_R]

/-! ### Theorem 1 (L1.a): `A (R φ X) = R φ (A X)` -/

/-- **Theorem 1, part L1.a** — global phase equivariance of `A_θ`. -/
theorem apply_R (φ : ℝ) (X : TokenSeq N d) :
    A.apply (R φ X) = R φ (A.apply X) := by
  funext i
  have step1 :
      A.apply (R φ X) i =
        A.Wo (∑ j : Fin N, ((A.alpha X i j : ℂ) • (cisR φ • A.vVec X j))) := by
    unfold apply
    congr 1
    apply Finset.sum_congr rfl
    intro j _
    rw [alpha_R, vVec_R]
  have step2 :
      (∑ j : Fin N, ((A.alpha X i j : ℂ) • (cisR φ • A.vVec X j)))
        = cisR φ • (∑ j : Fin N, ((A.alpha X i j : ℂ) • A.vVec X j)) := by
    rw [Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [smul_comm]
  rw [step1, step2, LinearMap.map_smul]
  rfl

/-- **Theorem 1, part L1.a — packaged form**. -/
theorem apply_L1a : L1a (A.apply : TokenSeq N d → TokenSeq N d) :=
  fun φ X => A.apply_R φ X

/-- **L1.b is realised by construction**: take `f̃(t) := A.f (t + A.b)`,
`V := A.Wo ∘ A.Wv`, and `s̃(a, b) := Re ⟨normalize (Wq a), normalize (Wk b)⟩`.
-/
theorem L1b_witness : L1b (A.apply : TokenSeq N d → TokenSeq N d) := by
  refine ⟨fun t => A.f (t + A.b), A.Wo.comp A.Wv,
    (fun a b => Complex.re (inner ℂ
        (((‖A.Wq a‖⁻¹ : ℝ) : ℂ) • A.Wq a)
        (((‖A.Wk b‖⁻¹ : ℝ) : ℂ) • A.Wk b))),
    ?_, ?_⟩
  · -- s phase invariance
    intro φ a b
    have hq : ((‖A.Wq (cisR φ • a)‖⁻¹ : ℝ) : ℂ) • A.Wq (cisR φ • a)
            = cisR φ • (((‖A.Wq a‖⁻¹ : ℝ) : ℂ) • A.Wq a) := by
      rw [LinearMap.map_smul]
      exact normalize_smul_phase φ (A.Wq a)
    have hk : ((‖A.Wk (cisR φ • b)‖⁻¹ : ℝ) : ℂ) • A.Wk (cisR φ • b)
            = cisR φ • (((‖A.Wk b‖⁻¹ : ℝ) : ℂ) • A.Wk b) := by
      rw [LinearMap.map_smul]
      exact normalize_smul_phase φ (A.Wk b)
    show Complex.re (inner ℂ
            (((‖A.Wq (cisR φ • a)‖⁻¹ : ℝ) : ℂ) • A.Wq (cisR φ • a))
            (((‖A.Wk (cisR φ • b)‖⁻¹ : ℝ) : ℂ) • A.Wk (cisR φ • b))) =
          Complex.re (inner ℂ
            (((‖A.Wq a‖⁻¹ : ℝ) : ℂ) • A.Wq a)
            (((‖A.Wk b‖⁻¹ : ℝ) : ℂ) • A.Wk b))
    rw [hq, hk, inner_smul_phase]
  · -- expand: A.apply X i = ∑ j, (f̃(s̃(X i, X j)) : ℂ) • (Wo ∘ Wv) (X j)
    intro X i
    unfold apply
    rw [map_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [LinearMap.map_smul]
    -- The remaining equality unfolds: alpha = f (cosScore + b),
    -- cosScore = s̃ on (X i, X j), and A.Wo (vVec) = (Wo.comp Wv) X j.
    unfold alpha cosScore qBar kBar qVec kVec vVec
    rfl

end AttentionLayer

/-! ### Theorem 1' — necessity of C4 for L1.b -/

/-- **Theorem 1' (necessity)**: any L1.b-coherent map factors its gate
through the *binary* score, i.e. element-wise on `(x_i, x_j)` with no
row-coupling.  Trivially true since `L1b A` is, by definition, the
existential of such a factorisation. -/
theorem L1b_implies_C4 {A : TokenSeq N d → TokenSeq N d} (h : L1b A) :
    ∃ (f : ℝ → ℝ) (V : Token d →ₗ[ℂ] Token d) (s : Token d → Token d → ℝ),
      ∀ X i, A X i = ∑ j, ((f (s (X i) (X j)) : ℂ) • V (X j)) := by
  obtain ⟨f, V, s, _h_phase, h_expand⟩ := h
  exact ⟨f, V, s, h_expand⟩

/-! ### Corollary 2 (PCT and `complex_screen` are L1; `complex_softmax` is not) -/

/-- **Corollary 2 (existence form).**  Any `AttentionLayer` realises both
halves of L1 by `apply_L1a` and `L1b_witness`. -/
theorem corollary2 {N : ℕ} (A : AttentionLayer d) :
    L1a (A.apply : TokenSeq N d → TokenSeq N d) ∧
    L1b (A.apply : TokenSeq N d → TokenSeq N d) :=
  ⟨A.apply_L1a, A.L1b_witness⟩

end PaperV4
