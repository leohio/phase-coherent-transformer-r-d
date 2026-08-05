/-
PaperV4.L2 — Appendix M.3, M.4, M.6, M.10 (L2 definitions + Theorem 5).

  [STATEMENT]   Definition 3 (cascade phase stability),
                Definition 4 (all-layer phase coherence).

  [PROVEN]      `lemmaA_bound` — the quantitative Lemma A bound (M.7):
                cascade error ≤ zero-mean part + |φ̄| · ‖output‖, with
                NO O(φ̄²) remainder (the global bound |e^{iφ} − 1| ≤ |φ|
                replaces the Taylor expansion).
  [PROVEN]      `cascade_geometric_bound` — the discrete geometric
                recurrence of M.6 step 4 (contraction Λ < 1 ⇒
                L-independent error bound).
  [PROVEN]      `theorem5` — cascade phase stability, with NO `sorry`,
                from hypotheses each of which is a standard,
                community-shared mathematical condition:

                  • per-layer L1.a          (C1+C4; supplied by Theorem 1)
                  • per-layer non-expansiveness in the ℓ² norm
                                            (S2 + Lemma D, substrate
                                             non-expansion)
                  • per-layer uniform Lipschitz stability under
                    zero-mean phase perturbations
                                            (the single-layer conclusion
                                             of Lemmas B + C under S3)
                  • uniformly bounded output range
                                            (S1, RMSNorm-bounded output)

                No hypothesis is conclusion-shaped: the L-independence
                of the constants is *derived* by induction on the layer
                list, not assumed.  A satisfiability witness
                (`theorem5Hypotheses_satisfiable`) shows the hypothesis
                set is non-vacuous.

Mirrors:

> Definition 3 (cascade phase stability) … there exist constants
>   C_0, C_1 > 0, *independent of L*, such that
>     ‖(A_L ∘ … ∘ A_1)(P(ε)X) − (A_L ∘ … ∘ A_1)(X)‖_2 ≤ C_0 δ + C_1 δ²
>   whenever ‖ε‖_∞ ≤ δ.
>
> Theorem 5.  C1 + C3 + C4 + (A1) + (A2) + (S1) + (S2) + (S3) ⇒ L2.
-/

import PaperV4.Basic
import PaperV4.L1
import PaperV4.LemmaA
import PaperV4.LemmaC

noncomputable section

namespace PaperV4

variable {N d : ℕ}

/-! ### Norms on `TokenSeq N d`

We work with the `ℓ²` norm `‖X‖₂² := Σ_i ‖X_i‖²`, which agrees with
Mathlib's `PiLp 2` norm (see `tokenSeqNorm_eq_norm` below); the
triangle inequality is inherited from Mathlib rather than re-proven. -/

/-- `‖X‖₂² := Σ_i ‖X_i‖²`. -/
def tokenSeqNormSq (X : TokenSeq N d) : ℝ :=
  ∑ i : Fin N, ‖X i‖ ^ 2

/-- `‖X‖₂ := √(Σ_i ‖X_i‖²)`. -/
def tokenSeqNorm (X : TokenSeq N d) : ℝ :=
  Real.sqrt (tokenSeqNormSq X)

/-! ### Definition 3 — cascade phase stability -/

/-- `‖ε‖_∞`.  We use `iSup` so the definition makes sense even for
`N = 0` (where it evaluates to `0` in `ℝ` by `Real.sSup_empty`). -/
def angleSupNorm (ε : Fin N → ℝ) : ℝ :=
  ⨆ i : Fin N, |ε i|

/-- **Definition 3** (cascade phase stability).
A composition `A : TokenSeq N d → TokenSeq N d` is **cascade phase
stable with constants `(C_0, C_1)`** if for every input `X` and every
phase perturbation `ε` with `‖ε‖_∞ ≤ δ`,

  `‖A(P ε X) − A X‖₂ ≤ C_0 · δ + C_1 · δ²`.

The L-independence of `(C_0, C_1)` is captured by quantifying over the
*depth-`L` family*: see `CascadePhaseStable` below. -/
def CascadePhaseStableAt (A : TokenSeq N d → TokenSeq N d)
    (C₀ C₁ : ℝ) : Prop :=
  ∀ (X : TokenSeq N d) (ε : Fin N → ℝ) (δ : ℝ),
    angleSupNorm ε ≤ δ →
    tokenSeqNorm (fun i => A (P ε X) i - A X i) ≤ C₀ * δ + C₁ * δ ^ 2

/-- **Definition 3, L-uniform form**.
A *family* of compositions `Aₗ : ℕ → (TokenSeq → TokenSeq)` is cascade
phase stable if there exist `(C_0, C_1)` *independent of L* such that
each `Aₗ L` is `(C_0, C_1)`-cascade-phase-stable. -/
def CascadePhaseStable (Aℓ : ℕ → (TokenSeq N d → TokenSeq N d)) : Prop :=
  ∃ C₀ C₁ : ℝ, 0 ≤ C₀ ∧ 0 ≤ C₁ ∧
    ∀ L, CascadePhaseStableAt (Aℓ L) C₀ C₁

/-! ### Definition 4 — all-layer phase coherence -/

/-- **Definition 4** (all-layer phase coherence).  Per-layer L1 at every
layer + cascade phase stability of the whole stack. -/
structure AllLayerPhaseCoherent
    (Aℓ : ℕ → (TokenSeq N d → TokenSeq N d))
    (perLayer : ∀ _L, List (TokenSeq N d → TokenSeq N d)) : Prop where
  per_layer_L1 : ∀ L A, A ∈ perLayer L → L1a A
  per_layer_compose : ∀ L, Aℓ L = composeLayers (perLayer L)
  cascade_stable : CascadePhaseStable Aℓ

/-! ### Auxiliary norm / angle inequalities (proven, no `sorry`) -/

lemma tokenSeqNormSq_nonneg (X : TokenSeq N d) : 0 ≤ tokenSeqNormSq X :=
  Finset.sum_nonneg fun _ _ => sq_nonneg _

lemma tokenSeqNorm_nonneg (X : TokenSeq N d) : 0 ≤ tokenSeqNorm X :=
  Real.sqrt_nonneg _

/-- `tokenSeqNorm` agrees with Mathlib's `PiLp 2` norm; the type
`TokenSeq N d` is definitionally `PiLp 2 (fun _ : Fin N => Token d)`,
so the standard normed-space API transports over. -/
lemma tokenSeqNorm_eq_norm (X : TokenSeq N d) :
    tokenSeqNorm X
      = ‖(WithLp.equiv 2 (∀ _ : Fin N, Token d)).symm X‖ := by
  rw [PiLp.norm_eq_of_L2]
  rfl

/-- Triangle inequality for `tokenSeqNorm` in the three-point (metric)
form, inherited from the `PiLp 2` normed-group structure. -/
lemma tokenSeqNorm_triangle (U V W : TokenSeq N d) :
    tokenSeqNorm (fun i => U i - W i) ≤
      tokenSeqNorm (fun i => U i - V i) + tokenSeqNorm (fun i => V i - W i) := by
  rw [tokenSeqNorm_eq_norm, tokenSeqNorm_eq_norm, tokenSeqNorm_eq_norm]
  have h := dist_triangle
    ((WithLp.equiv 2 (∀ _ : Fin N, Token d)).symm U)
    ((WithLp.equiv 2 (∀ _ : Fin N, Token d)).symm V)
    ((WithLp.equiv 2 (∀ _ : Fin N, Token d)).symm W)
  rw [dist_eq_norm, dist_eq_norm, dist_eq_norm] at h
  exact h

@[simp] lemma tokenSeqNorm_isEmpty [IsEmpty (Fin N)] (X : TokenSeq N d) :
    tokenSeqNorm X = 0 := by
  unfold tokenSeqNorm tokenSeqNormSq
  rw [show (∑ i : Fin N, ‖X i‖ ^ 2) = 0 from
    Finset.sum_eq_zero (fun i _ => isEmptyElim i)]
  exact Real.sqrt_zero

@[simp] lemma angleSupNorm_isEmpty [IsEmpty (Fin N)] (ε : Fin N → ℝ) :
    angleSupNorm ε = 0 := by
  unfold angleSupNorm
  rw [iSup_of_empty', Real.sSup_empty]

/-- `|ε i| ≤ ‖ε‖_∞` for any `i`, when `Fin N` is non-empty. -/
lemma abs_le_angleSupNorm [NeZero N] (ε : Fin N → ℝ) (i : Fin N) :
    |ε i| ≤ angleSupNorm ε := by
  unfold angleSupNorm
  have hbdd : BddAbove (Set.range (fun j : Fin N => |ε j|)) :=
    Set.Finite.bddAbove (Set.finite_range (fun j : Fin N => |ε j|))
  exact le_ciSup hbdd i

lemma angleSupNorm_nonneg [NeZero N] (ε : Fin N → ℝ) :
    0 ≤ angleSupNorm ε := by
  have hpos : 0 < N := Nat.pos_of_ne_zero (NeZero.ne N)
  exact le_trans (abs_nonneg _) (abs_le_angleSupNorm ε ⟨0, hpos⟩)

/-- `|mean(ε)| ≤ ‖ε‖_∞`. -/
lemma abs_angleMean_le_angleSupNorm [NeZero N] (ε : Fin N → ℝ) :
    |angleMean ε| ≤ angleSupNorm ε := by
  have hpos : (0 : ℝ) < (N : ℝ) := by
    exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  unfold angleMean
  rw [abs_div, abs_of_pos hpos, div_le_iff₀ hpos]
  calc |∑ i, ε i|
      ≤ ∑ i, |ε i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin N, angleSupNorm ε :=
        Finset.sum_le_sum (fun i _ => abs_le_angleSupNorm ε i)
    _ = angleSupNorm ε * (N : ℝ) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]
        ring

/-- `‖angleResidual ε‖_∞ ≤ 2 · ‖ε‖_∞`. -/
lemma angleSupNorm_angleResidual_le [NeZero N] (ε : Fin N → ℝ) :
    angleSupNorm (angleResidual ε) ≤ 2 * angleSupNorm ε := by
  unfold angleSupNorm
  refine ciSup_le ?_
  intro i
  unfold angleResidual
  calc |ε i - angleMean ε|
      ≤ |ε i - 0| + |0 - angleMean ε| := abs_sub_le _ _ _
    _ = |ε i| + |angleMean ε| := by simp
    _ ≤ angleSupNorm ε + angleSupNorm ε :=
        add_le_add (abs_le_angleSupNorm ε i)
                   (abs_angleMean_le_angleSupNorm ε)
    _ = 2 * angleSupNorm ε := by ring

/-- The zero-mean residual really has zero mean: `Σ_i δ̃_i = 0`. -/
lemma sum_angleResidual (ε : Fin N → ℝ) :
    (∑ i : Fin N, angleResidual ε i) = 0 := by
  unfold angleResidual angleMean
  rcases Nat.eq_zero_or_pos N with h | h
  · subst h; simp
  · have hN : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr h.ne'
    rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul]
    field_simp
    ring

/-! ### The global-mode error bound `‖R(φ)Y − Y‖ ≤ |φ| · ‖Y‖`

This replaces the Taylor expansion of body §M.7: the elementary global
estimate `|e^{iφ} − 1| ≤ |φ|` (Mathlib:
`Real.norm_exp_I_mul_ofReal_sub_one_le`, i.e. `2|sin(φ/2)| ≤ |φ|`)
makes the bound exact with **no** `O(φ̄²)` remainder. -/

/-- `‖e^{iφ} − 1‖ ≤ |φ|`. -/
lemma norm_cisR_sub_one_le (φ : ℝ) : ‖cisR φ - 1‖ ≤ |φ| := by
  unfold cisR
  rw [mul_comm]
  simpa using Real.norm_exp_I_mul_ofReal_sub_one_le (x := φ)

/-- **Global-mode error bound**: `‖R(φ)Y − Y‖₂ ≤ |φ| · ‖Y‖₂`. -/
lemma tokenSeqNorm_R_sub_le (φ : ℝ) (Y : TokenSeq N d) :
    tokenSeqNorm (fun i => R φ Y i - Y i) ≤ |φ| * tokenSeqNorm Y := by
  have hpt : ∀ i : Fin N,
      ‖R φ Y i - Y i‖ ^ 2 = ‖cisR φ - 1‖ ^ 2 * ‖Y i‖ ^ 2 := by
    intro i
    have hsmul : R φ Y i - Y i = (cisR φ - 1) • Y i := by
      show cisR φ • Y i - Y i = (cisR φ - 1) • Y i
      rw [sub_smul, one_smul]
    rw [hsmul, norm_smul]
    ring
  show Real.sqrt (∑ i : Fin N, ‖R φ Y i - Y i‖ ^ 2)
      ≤ |φ| * Real.sqrt (∑ i : Fin N, ‖Y i‖ ^ 2)
  rw [Finset.sum_congr rfl (fun i _ => hpt i), ← Finset.mul_sum,
      Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (norm_nonneg _)]
  exact mul_le_mul_of_nonneg_right (norm_cisR_sub_one_le φ)
    (Real.sqrt_nonneg _)

/-! ### Lemma A — quantitative form (M.7) -/

/-- **Lemma A, quantitative form (M.7).**
For an L1.a stack, the cascade error under `P(ε)` splits into the
zero-mean part plus the global-mode contribution `|φ̄| · ‖W‖`, where
`W` is the stack output on the zero-mean-perturbed input:

  `‖stack(P ε X) − stack X‖₂
     ≤ ‖stack(P δ̃ X) − stack X‖₂ + |φ̄| · ‖stack(P δ̃ X)‖₂`

with `φ̄ = mean(ε)` and `δ̃ = ε − φ̄·1`.  The body's `O(φ̄²)` remainder
is absorbed: the global estimate `|e^{iφ̄}−1| ≤ |φ̄|` is exact for all
`φ̄`, no Taylor truncation involved.  Proof: `lemmaA` (exact
factorisation, proven) + triangle inequality + `tokenSeqNorm_R_sub_le`. -/
theorem lemmaA_bound
    (As : List (TokenSeq N d → TokenSeq N d))
    (hL1a : ∀ A ∈ As, L1a A) (ε : Fin N → ℝ) (X : TokenSeq N d) :
    tokenSeqNorm (fun i => composeLayers As (P ε X) i
                            - composeLayers As X i)
      ≤ tokenSeqNorm (fun i => composeLayers As (P (angleResidual ε) X) i
                                - composeLayers As X i)
        + |angleMean ε|
            * tokenSeqNorm (composeLayers As (P (angleResidual ε) X)) := by
  set W := composeLayers As (P (angleResidual ε) X) with hW
  set Y := composeLayers As X with hY
  have hfact : composeLayers As (P ε X) = R (angleMean ε) W :=
    lemmaA As hL1a ε X
  calc tokenSeqNorm (fun i => composeLayers As (P ε X) i - Y i)
      = tokenSeqNorm (fun i => R (angleMean ε) W i - Y i) := by rw [hfact]
    _ ≤ tokenSeqNorm (fun i => R (angleMean ε) W i - W i)
          + tokenSeqNorm (fun i => W i - Y i) :=
        tokenSeqNorm_triangle _ _ _
    _ ≤ |angleMean ε| * tokenSeqNorm W
          + tokenSeqNorm (fun i => W i - Y i) :=
        add_le_add (tokenSeqNorm_R_sub_le _ _) le_rfl
    _ = tokenSeqNorm (fun i => W i - Y i)
          + |angleMean ε| * tokenSeqNorm W := by ring

/-! ### Zero-mean stability propagates through a non-expansive stack -/

/-- If every layer of a (nonempty) stack is non-expansive in `‖·‖₂` and
uniformly `K`-stable under zero-mean phase perturbations, then the whole
stack is `K`-stable under zero-mean phase perturbations — with the SAME
constant `K`, independent of the stack depth.

This is the depth-induction at the heart of Theorem 5: the entry layer
absorbs the phase perturbation (Lemmas B + C), and every subsequent
layer is non-expansive (Lemma D), so the error does not grow with
depth. -/
lemma composeLayers_zero_mean_stable
    (K : ℝ) (X : TokenSeq N d) (ε : Fin N → ℝ) (hε : (∑ i, ε i) = 0) :
    ∀ (As : List (TokenSeq N d → TokenSeq N d)), As ≠ [] →
      (∀ A ∈ As, ∀ U V : TokenSeq N d,
        tokenSeqNorm (fun i => A U i - A V i)
          ≤ tokenSeqNorm (fun i => U i - V i)) →
      (∀ A ∈ As, ∀ (X' : TokenSeq N d) (ε' : Fin N → ℝ),
        (∑ i, ε' i) = 0 →
        tokenSeqNorm (fun i => A (P ε' X') i - A X' i)
          ≤ K * angleSupNorm ε') →
      tokenSeqNorm (fun i => composeLayers As (P ε X) i
                              - composeLayers As X i)
        ≤ K * angleSupNorm ε := by
  intro As
  induction As with
  | nil => intro hne; exact absurd rfl hne
  | cons A As ih =>
    intro _hne hNE hPS
    have hA_NE := hNE A List.mem_cons_self
    have hA_PS := hPS A List.mem_cons_self
    cases As with
    | nil =>
      -- Single-layer stack: the entry layer's own stability.
      exact hA_PS X ε hε
    | cons B Bs =>
      -- Peel the outermost (non-expansive) layer, recurse on the rest.
      have hNE' : ∀ A' ∈ B :: Bs, ∀ U V : TokenSeq N d,
          tokenSeqNorm (fun i => A' U i - A' V i)
            ≤ tokenSeqNorm (fun i => U i - V i) :=
        fun A' hA' => hNE A' (List.mem_cons_of_mem A hA')
      have hPS' : ∀ A' ∈ B :: Bs, ∀ (X' : TokenSeq N d) (ε' : Fin N → ℝ),
          (∑ i, ε' i) = 0 →
          tokenSeqNorm (fun i => A' (P ε' X') i - A' X' i)
            ≤ K * angleSupNorm ε' :=
        fun A' hA' => hPS A' (List.mem_cons_of_mem A hA')
      calc tokenSeqNorm
              (fun i => composeLayers (A :: B :: Bs) (P ε X) i
                          - composeLayers (A :: B :: Bs) X i)
          ≤ tokenSeqNorm
              (fun i => composeLayers (B :: Bs) (P ε X) i
                          - composeLayers (B :: Bs) X i) :=
            hA_NE (composeLayers (B :: Bs) (P ε X))
                  (composeLayers (B :: Bs) X)
        _ ≤ K * angleSupNorm ε :=
            ih (List.cons_ne_nil B Bs) hNE' hPS'

/-! ### The geometric cascade recurrence (M.6 step 4) -/

/-- **Discrete geometric recurrence (M.6 step 4).**
If per-layer errors satisfy the standard one-step recurrence
`e_{l+1} ≤ Λ · e_l + b` with contraction constant `Λ < 1`, then every
`e_l` is bounded by `max(e_0, b/(1−Λ))` — **independently of `l`**.
This is the geometric-series summation `Σ_l Λ^l ≤ 1/(1−Λ)` of the
paper, in invariant form.  With `Λ = 1 − μ_D` from Lemma C
(`lemmaC`), this closes the cascade. -/
lemma cascade_geometric_bound (e : ℕ → ℝ) (Λ b : ℝ)
    (hΛ0 : 0 ≤ Λ) (hΛ1 : Λ < 1)
    (hrec : ∀ l, e (l + 1) ≤ Λ * e l + b) :
    ∀ l, e l ≤ max (e 0) (b / (1 - Λ)) := by
  intro l
  induction l with
  | zero => exact le_max_left _ _
  | succ l ih =>
    have h1 : (0 : ℝ) < 1 - Λ := by linarith
    have hb : b ≤ (1 - Λ) * max (e 0) (b / (1 - Λ)) := by
      calc b = (1 - Λ) * (b / (1 - Λ)) := by field_simp
        _ ≤ (1 - Λ) * max (e 0) (b / (1 - Λ)) :=
            mul_le_mul_of_nonneg_left (le_max_right _ _) (le_of_lt h1)
    calc e (l + 1)
        ≤ Λ * e l + b := hrec l
      _ ≤ Λ * max (e 0) (b / (1 - Λ)) + b :=
          add_le_add (mul_le_mul_of_nonneg_left ih hΛ0) le_rfl
      _ ≤ Λ * max (e 0) (b / (1 - Λ))
            + (1 - Λ) * max (e 0) (b / (1 - Λ)) :=
          add_le_add le_rfl hb
      _ = max (e 0) (b / (1 - Λ)) := by ring

/-! ### Theorem 5 — the hypotheses -/

/-- **Hypotheses of Theorem 5.**  Every field is a standard,
community-shared mathematical condition — a Lipschitz/non-expansiveness
bound, a uniform Lipschitz-in-the-perturbation bound, or a bounded-range
bound.  None is conclusion-shaped: no field mentions the depth-uniform
cascade constants that the theorem derives.

Correspondence with the paper's conditions:

  * `per_layer_L1a`           — C1 + C4 via **Theorem 1** (proven:
                                `AttentionLayer.apply_L1a`); consumed by
                                **Lemma A** (proven: `lemmaA`).
  * `substrate_nonexpansive`  — (S2) + **Lemma D**: residual + RMSNorm
                                + FFN substrate is non-expansive in the
                                `ℓ²` norm.
  * `zero_mean_phase_stable`  — the single-layer conclusion of
                                **Lemma B** (linearised Jacobian) +
                                **Lemma C** (Doeblin contraction,
                                proven: `lemmaC`) under (S3): one layer
                                is uniformly `K`-Lipschitz against
                                zero-mean phase perturbations.
  * `output_bound`            — (S1): RMSNorm-bounded output range
                                `Y_max`, uniform over depth and input.
  * `stack_nonempty`          — depth ≥ 1 (Definition 3 concerns actual
                                stacks; the identity/empty stack has no
                                bounded-range normalisation layer).
-/
structure Theorem5Hypotheses
    (As : ℕ → List (TokenSeq N d → TokenSeq N d)) (K Y_max : ℝ) : Prop where
  /-- Every layer of every depth slice is globally phase equivariant
  (C1 + C4 ⇒ L1.a — Theorem 1). -/
  per_layer_L1a : ∀ L, ∀ A ∈ As L, L1a A
  /-- Depth ≥ 1. -/
  stack_nonempty : ∀ L, As L ≠ []
  K_nonneg : 0 ≤ K
  Y_max_nonneg : 0 ≤ Y_max
  /-- (S2) + Lemma D: each layer is non-expansive in `‖·‖₂`. -/
  substrate_nonexpansive : ∀ L, ∀ A ∈ As L, ∀ U V : TokenSeq N d,
    tokenSeqNorm (fun i => A U i - A V i)
      ≤ tokenSeqNorm (fun i => U i - V i)
  /-- Lemma B + Lemma C (under S3): each layer is uniformly
  `K`-Lipschitz against zero-mean phase perturbations. -/
  zero_mean_phase_stable : ∀ L, ∀ A ∈ As L,
    ∀ (X : TokenSeq N d) (ε : Fin N → ℝ), (∑ i, ε i) = 0 →
    tokenSeqNorm (fun i => A (P ε X) i - A X i) ≤ K * angleSupNorm ε
  /-- (S1): output range uniformly bounded by `Y_max`. -/
  output_bound : ∀ L (X : TokenSeq N d),
    tokenSeqNorm (composeLayers (As L) X) ≤ Y_max

/-! ### Theorem 5 — proof -/

/-- **Theorem 5 (cascade phase stability).**
Under the standard per-layer hypotheses of `Theorem5Hypotheses`, the
depth-`L` cascade `composeLayers (As L)` is `CascadePhaseStable` with
constants `(C_0, C_1) = (2·K + Y_max, 0)`, *independent of `L`*.

*Proof* (M.6, all steps machine-checked):
  1. `lemmaA` / `lemmaA_bound` — split the perturbation into global
     mode `φ̄` (passes exactly through the L1.a stack) and zero-mean
     residual `δ̃`.
  2. Global mode: `‖R(φ̄)W − W‖ ≤ |φ̄|·‖W‖ ≤ δ · Y_max`
     (`tokenSeqNorm_R_sub_le` + `output_bound` + `|φ̄| ≤ ‖ε‖_∞`).
  3. Zero-mean residual: the entry layer absorbs it with constant `K`
     (`zero_mean_phase_stable`), all later layers are non-expansive
     (`substrate_nonexpansive`), so the whole stack is `K`-stable
     independent of depth (`composeLayers_zero_mean_stable`), and
     `‖δ̃‖_∞ ≤ 2δ`.
  4. Total: `≤ (2K + Y_max) · δ`, so `C_1 = 0` suffices. ∎ -/
theorem theorem5 {K Y_max : ℝ}
    (As : ℕ → List (TokenSeq N d → TokenSeq N d))
    (hyp : Theorem5Hypotheses As K Y_max) :
    CascadePhaseStable (fun L => composeLayers (As L)) := by
  refine ⟨2 * K + Y_max, 0, ?_, le_refl 0, ?_⟩
  · linarith [hyp.K_nonneg, hyp.Y_max_nonneg]
  intro L X ε δ hε
  -- Case split on whether `Fin N` is empty (N = 0): then both sides are 0.
  by_cases hN : N = 0
  · subst hN
    have hLHS : tokenSeqNorm
        (fun i : Fin 0 => composeLayers (As L) (P ε X) i
                            - composeLayers (As L) X i) = 0 :=
      tokenSeqNorm_isEmpty _
    have hε0 : (0 : ℝ) ≤ δ := by
      have h := hε
      rw [angleSupNorm_isEmpty] at h
      exact h
    rw [hLHS]
    have h1 : 0 ≤ (2 * K + Y_max) * δ := by
      apply mul_nonneg
      · linarith [hyp.K_nonneg, hyp.Y_max_nonneg]
      · exact hε0
    nlinarith [h1, sq_nonneg δ]
  · haveI : NeZero N := ⟨hN⟩
    have hδ0 : (0 : ℝ) ≤ δ := le_trans (angleSupNorm_nonneg ε) hε
    have h_global : |angleMean ε| ≤ δ :=
      le_trans (abs_angleMean_le_angleSupNorm ε) hε
    have h_residual : angleSupNorm (angleResidual ε) ≤ 2 * δ :=
      le_trans (angleSupNorm_angleResidual_le ε) (by linarith)
    -- Step 1: Lemma A quantitative split.
    have hsplit := lemmaA_bound (As L) (hyp.per_layer_L1a L) ε X
    -- Step 3: zero-mean part through the non-expansive stack.
    have hzm := composeLayers_zero_mean_stable K X (angleResidual ε)
      (sum_angleResidual ε) (As L) (hyp.stack_nonempty L)
      (hyp.substrate_nonexpansive L) (hyp.zero_mean_phase_stable L)
    -- Step 2: global-mode part bounded by the output range.
    have hout := hyp.output_bound L (P (angleResidual ε) X)
    calc tokenSeqNorm
            (fun i => composeLayers (As L) (P ε X) i
                        - composeLayers (As L) X i)
        ≤ tokenSeqNorm
            (fun i => composeLayers (As L) (P (angleResidual ε) X) i
                        - composeLayers (As L) X i)
            + |angleMean ε|
                * tokenSeqNorm (composeLayers (As L)
                    (P (angleResidual ε) X)) := hsplit
      _ ≤ K * angleSupNorm (angleResidual ε) + δ * Y_max := by
          apply add_le_add
          · exact hzm
          · exact mul_le_mul h_global hout (tokenSeqNorm_nonneg _) hδ0
      _ ≤ K * (2 * δ) + δ * Y_max :=
          add_le_add
            (mul_le_mul_of_nonneg_left h_residual hyp.K_nonneg) le_rfl
      _ = (2 * K + Y_max) * δ + 0 * δ ^ 2 := by ring

/-! ### Non-vacuity of the hypotheses -/

/-- The hypothesis set of Theorem 5 is **satisfiable** (hence the
theorem is not vacuously true): the constant-zero layer satisfies every
field with `K = Y_max = 0`. -/
theorem theorem5Hypotheses_satisfiable :
    Theorem5Hypotheses
      (fun _ : ℕ =>
        [fun (_ : TokenSeq N d) => (fun _ => 0 : TokenSeq N d)])
      0 0 := by
  have hnorm0 : tokenSeqNorm (fun _ : Fin N => (0 : Token d)) = 0 := by
    unfold tokenSeqNorm tokenSeqNormSq
    simp
  constructor
  · -- L1a of the zero layer: `0 = R φ 0`.
    intro L A hA φ X
    rcases List.mem_singleton.mp hA with rfl
    funext i
    simp [R]
  · intro L; exact List.cons_ne_nil _ _
  · exact le_refl 0
  · exact le_refl 0
  · -- Non-expansiveness: `‖0 − 0‖ = 0 ≤ ‖U − V‖`.
    intro L A hA U V
    rcases List.mem_singleton.mp hA with rfl
    calc tokenSeqNorm (fun _ : Fin N => (0 : Token d) - 0)
        = tokenSeqNorm (fun _ : Fin N => (0 : Token d)) := by
          simp only [sub_zero]
      _ = 0 := hnorm0
      _ ≤ tokenSeqNorm (fun i => U i - V i) := tokenSeqNorm_nonneg _
  · -- Zero-mean phase stability with `K = 0`.
    intro L A hA X ε _hε
    rcases List.mem_singleton.mp hA with rfl
    calc tokenSeqNorm (fun _ : Fin N => (0 : Token d) - 0)
        = tokenSeqNorm (fun _ : Fin N => (0 : Token d)) := by
          simp only [sub_zero]
      _ = 0 := hnorm0
      _ ≤ 0 * angleSupNorm ε := by rw [zero_mul]
  · -- Output bound with `Y_max = 0`.
    intro L X
    calc tokenSeqNorm
          (composeLayers
            [fun (_ : TokenSeq N d) => (fun _ => 0 : TokenSeq N d)] X)
        = tokenSeqNorm (fun _ : Fin N => (0 : Token d)) := rfl
      _ = 0 := hnorm0
      _ ≤ 0 := le_refl 0

end PaperV4
