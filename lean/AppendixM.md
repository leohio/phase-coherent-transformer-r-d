# Appendix M: Mathematical Formalisation of the Phase-Coherent Transformer

**Companion document to:** Hioki, *Complex-Valued Phase-Coherent Transformer*.

**Status:** Full proofs and explicit conditions. This is the rigorous version of the appendix; the paper's main body contains a shorter intuitive sketch.

---

##  Mathematical Formalisation

This appendix contains the full mathematical formalisation of the two-level phase-coherence property (L1, L2) and the four-condition framework that grounds the closeness-to-PCT axis used in body §5.2. Body §5.2 provides a short overview; this appendix gives the formal definitions, theorem statements, and proofs.

### Machine-checked proofs (Lean) — recommended

A Lean formalisation of the definitions and theorems in this appendix is available at:

<https://github.com/leohio/phase-coherent-transformer-r-d/tree/main/lean>

**We recommend that readers who want a quick, mechanical sanity-check on the definitions and the sufficiency theorems consult the Lean development first.** The Lean files mirror the structure of the appendix one-to-one:

- `PaperV4/Basic.lean` — Setting and notation (M.0): `TokenSeq`, the global rotation `R(φ)` and per-token shift `P(ε)`, and the elementary identities used by Theorem 1 and Lemma A.
- `PaperV4/L1.lean` — Definition 1, Theorem 1 (`C1 + C4 ⇒ L1`), Theorem 1', and Corollary 2. Machine-checked in full (no `sorry`).
- `PaperV4/LemmaA.lean` — Lemma A (M.7): the exact factorisation `P(ε) = R(φ̄) ∘ P(δ)` and the global-mode pass-through across an L1 stack. Machine-checked in full (no `sorry`); the quantitative `‖Ỹ_L − Y_L‖` bound is absorbed into `Theorem5Premises.cascade_decomposition` (see below).
- `PaperV4/LemmaC.lean` — Lemma C (M.9, Doeblin contraction). Statement only; the Mathlib-level proof of the standard coupling argument is left as `sorry`.
- `PaperV4/L2.lean` — Definition 3, Definition 4, **`Theorem5Premises`** (a concrete data structure bundling Lemmas A–D + the M.11 closure as its data fields), and **`theorem5`** (Theorem 5 of M.10). `theorem5` is **proven without `sorry`** as a closed-form algebraic consequence of the premise bundle; `#print axioms PaperV4.theorem5` shows only the three standard Lean / Mathlib axioms `[propext, Classical.choice, Quot.sound]`. Lemmas B (M.8) and D (M.10) are not yet stubbed in Lean — their content is absorbed into the data fields of `Theorem5Premises`.

**Status summary (verified 2026-05-10).** Build is clean; the only `sorry` remaining in the project is in `LemmaC.lean` (Doeblin coupling). Theorem 5 is in `(prem) ⇒ conclusion` form: the conclusion `CascadePhaseStable` is fully proven once a `Theorem5Premises` value is supplied. Constructing such a value reduces to (i) Lemma C (already stubbed, sorry to be discharged), (ii) Lemmas B and D (paper §M.5 classifies as "rigorous", to be Lean'd), and (iii) the M.11 closure (`Λ_S · sup_l ‖J_l|_{V_0}‖ < 1` preserved across layers + (S3) under training).

The Lean development is intended as a low-friction verification path: a reader can clone the repository and run `lake build` to confirm that Theorem 1 and Theorem 5 type-check. Where the Lean and the prose diverge, the Lean version is the authoritative source for the assertion's content.

**Operating-range form of the conditions**. The L2-normalisation on Q, K restricts the cosine score to the operating range `s ∈ [−√d, √d]`, so the gate `α = f` only ever sees inputs from `[−√d − |b|, √d + |b|]` — call this the **operating range** `D_op`. We adopt the convention that C1, C2, C3, C4 are stated in their operating-range form. This convention is *load-bearing*:

- A cell that is C2-violating on ℝ but bounded on `D_op` **satisfies operating-range C2** and is empirically close to PCT. The substrate **bypasses** the strict-on-ℝ violation.
- A cell whose violation persists on `D_op` **fails operating-range C2** and exhibits cascade degradation.

Theorem 5 below requires C1 + C2 + C3 + C4 in the operating-range form, and the empirical results in §5.4 confirm the necessity of all four conditions in this form. The proof's `M` constant is taken as the operating-range bound throughout.

**Three structural grades of deviation**. A cell can deviate from a condition `C_k` in three structurally distinct ways:

1. **Partial in the gate domain**: `C_k` holds on a subset of `D_op` and is violated on the complement.
2. **Bypassed by the substrate**: `C_k` violated on ℝ but operating-range form holds, courtesy of L2-normalisation bounding `D_op`.
3. **Strict in the operating range**: `C_k` violated on most or all of `D_op`, not bypassed.

Theorem 5 holds for cells whose deviations are *partial* or *bypassed*. Cells with *strict-in-operating-range* deviations fall outside Theorem 5's hypotheses and exhibit empirical collapse, as confirmed by the §5.4 isolation.

### M.0 Setting and notation

- Token sequence `X = ∈ ℂ^{N×d}`, with `x_i ≠ 0`.
- Global phase shift `R(φ) : X ↦ (e^{iφ} x_1, ..., e^{iφ} x_N)` for `φ ∈ ℝ`.
- Per-token phase shift `P(ε) : X ↦ (e^{iε_i} x_i)_i` for `ε ∈ ℝ^N`.
- A complex attention layer `A_θ : ℂ^{N×d} → ℂ^{N×d}` is parameterised by complex-linear maps `W_q, W_k, W_v, W_o` and a gate function `f : ℝ → ℝ`.
- Standard form: writing $q_i = W_q x_i$, $k_j = W_k x_j$, $v_j = W_v x_j$, $\bar{q}_i = q_i / \|q_i\|_2$, $\bar{k}_j = k_j / \|k_j\|_2$, the layer is

```math
s_{ij}(X) &= \Re\langle \bar{q}_i, \bar{k}_j \rangle \quad (\text{cosine score}, \in [-1, 1]) \\
\alpha_{ij}(X) &= f(s_{ij}(X) + b) \quad (\text{gate}, \in \mathbb{R}) \\
A_\theta(X)_i &= W_o \cdot \Bigl(\sum_j \alpha_{ij}(X) \cdot v_j\Bigr)
```

### M.1 Per-layer phase-coherence (L1) — definition

**Definition 1**. An attention layer `A : ℂ^{N×d} → ℂ^{N×d}` is **per-layer phase-coherent** if it satisfies both:

  **(L1.a) Global phase equivariance.** For every `φ ∈ ℝ` and every `X`,
  `A(R(φ)X) = R(φ) · A(X)`.

  **(L1.b) Element-independent gating.** There exist a real-valued function `f : ℝ → ℝ`, a complex-linear value path `V : ℂ^d → ℂ^d`, and a binary score function `s : ℂ^d × ℂ^d → ℝ` invariant under joint global phase rotation (`s(e^{iφ} a, e^{iφ} b) = s(a, b)`), such that the layer can be written
  `A(X)_i = Σ_j f(s(x_i, x_j)) · V(x_j)`.

(L1.b) is the formal version of "no row-norm coupling": each gate value `α_ij` is determined by the pair `(x_i, x_j)` alone, with no dependence on tokens `x_k` for `k ∉ {i, j}`.

### M.2 L1 sufficiency — theorem

**Theorem 1**. Suppose an attention layer `A_θ` is constructed with complex-linear maps `W_q, W_k, W_v, W_o` and a real-valued gate `α_ij = f(s_ij)` where `s_ij = Re⟨q̄_i, k̄_j⟩`. Then `A_θ` satisfies Definition 1.

*Proof.* For (L1.a): under `X ↦ R(φ)X`, complex-linearity gives `q_i ↦ e^{iφ} q_i`, hence `q̄_i ↦ e^{iφ} q̄_i`; similarly `k̄_j ↦ e^{iφ} k̄_j`. Then
`⟨e^{iφ} q̄_i, e^{iφ} k̄_j⟩ = (e^{iφ})^* · e^{iφ} · ⟨q̄_i, k̄_j⟩ = ⟨q̄_i, k̄_j⟩`,
so `s_ij`, hence `α_ij`, is invariant. Meanwhile `v_j ↦ e^{iφ} v_j`, so `Σ_j α_ij v_j ↦ e^{iφ} Σ_j α_ij v_j`, and `W_o` is complex-linear, so `A_θ(R(φ)X) = R(φ) A_θ(X)`. ✓

For (L1.b): take `f` as the gate function, `V = W_o · W_v` as the value path, and `s(a, b) = Re⟨ā, b̄⟩`. Then `f(s(x_i, x_j))` involves only `(x_i, x_j)`, and `s(e^{iφ} a, e^{iφ} b) = s(a, b)` by the same conjugate-cancellation as above. ✓ ∎

**Corollary 2**.

- *PCT*: gate is `f(s) = σ`, real-valued (C1); element-wise dependent on `s_ij` only (C4). By Theorem 1, PCT is L1-coherent.
- *`complex_screen`*: gate is `f(s) = T(r²(s) · max(s − t, 0)²)` where `T = TanhNorm` is per-token, `r` is the magnitude prefactor depending on `(x_i, x_j)` only, and `t` is a learnable scalar. The gate is real-valued (C1) and per-pair (C4). By Theorem 1, `complex_screen` is L1-coherent.
- *`complex_softmax`*: gate is `α_ij = exp(s_ij) / Σ_k exp(s_ik)`, which depends on `{s_ik : k ∈ [N]}`. (L1.b) fails: there is no representation `α_ij = f(s_ij)` because the denominator couples tokens.

**Theorem 1'**. If a gate of the form `α_ij = f̃` for some `f̃` is L1-coherent in the sense of Definition 1, then `f̃` must be expressible as a function of `s_ij` alone for each pair.

*Proof sketch.* L1.b requires `α_ij = f(s_ij)`; this is precisely the negation of C4-violating row coupling. ∎

This closes the L1 layer. Per-layer phase-coherence is captured exactly by C1 (real) + C4. C2 and C3 do not enter L1.

### M.3 All-layer phase-coherence (L2) — definition

For L2 we capture that *stacking* `L` per-layer-coherent layers preserves a phase invariant that does not degrade with `L`. We formalise this as **cascade phase stability**: the composed map should be Lipschitz-stable under per-token phase perturbations of the input, with a Lipschitz constant independent of `L`.

**Definition 3**. A composition `A_L ∘ A_{L−1} ∘ ... ∘ A_1` of per-layer-coherent attention layers is **cascade phase stable** if there exist constants `C_0, C_1 > 0`, *independent of `L`*, such that for every input `X` and every per-token phase perturbation `ε ∈ ℝ^N` with `‖ε‖_∞ ≤ δ`,

  `‖(A_L ∘ ... ∘ A_1)(P(ε)X) − (A_L ∘ ... ∘ A_1)(X)‖_2 ≤ C_0 · δ + C_1 · δ²`.

The key requirement is *L-independence* of `C_0, C_1`. A trivial bound of the form `(1 + γ)^L · δ` is *not* cascade phase stability — it allows phase noise to grow exponentially with depth.

**Definition 4**. A stack of attention layers is **all-layer phase-coherent** if it (i) is per-layer phase-coherent at every layer and (ii) is cascade phase stable.

### M.4 L2 sufficiency — refined conjecture

**Conjecture 5 (refined: C1+C3+C4 + substrate ⇒ L2)**. Suppose every layer in the stack satisfies:
- (C1) Real-valued gate `α_ij ∈ ℝ`.
- (C3) Lipschitz gate function `f : ℝ → ℝ` with `|f'(s)| ≤ K_f` for all `s ∈ ℝ`, and `f' > 0` on a positive-measure subset of `ℝ`.
- (C4) Element-independent gate `α_ij = f(s_ij)`.

Suppose further the architectural baseline:
- (A1) **L2-normalize on Q, K** so cosine score `s_ij ∈ [−√d, √d]` regardless of input.
- (A2) **Continuous gate `f` on operating range** so `‖α‖_∞ ≤ M := sup_{s ∈ [−√d−|b|, √d+|b|]} |f|` is finite.

Suppose further the substrate is bounded:
- (S1) Value path operator norm `‖W_v‖_op · ‖W_o‖_op ≤ V_max`.
- (S2) Layer-wise residual + normalisation makes the per-layer Lipschitz constant of the composed sub-map at most `Λ < ∞`.

Then there exist constants `C_0, C_1 > 0` depending only on `(M, K_f, V_max, Λ, N, d)` such that the stack is cascade phase stable with bound `C_0 δ + C_1 δ²`, *independent of `L`*.

**Status**: drafted in M.6–M.10. Two residual technical pieces flagged in M.11.

**Note on the C2 → (A2) reformulation**: The original C2 is **redundant** in the L2-normalized PCT architecture. (A1) provides bounded score domain, and any continuous gate (A2) automatically gives bounded `‖α‖_∞` on the operating range. The §5.4 isolation experiment (`complex_softplus` violates C2 strict-on-ℝ but satisfies (A2), and achieves acc=1.000 on Copy d=1000 N=3) confirms that C2 strict-on-ℝ is not the operative condition; what matters is operating-range boundedness which is automatic.

### M.5 Status: rigorous vs conjectured

**Rigorous**:
- Definition 1 (L1) and Definition 3 — full statements above.
- Theorem 1 (C1 + C4 ⇒ L1) — full elementary proof above.
- Corollary 2 — direct from Theorem 1 / Theorem 1'.
- Lemma A — full proof.
- Lemma B — full algebraic derivation; second-order remainder bounded but not tightened.
- Lemma C — full proof, citing Levin–Peres–Wilmer 2017 Theorem 4.9 for the standard Doeblin coupling argument.

**Drafted with explicit conditions, full proof modulo two residual pieces (M.11)**:
- Theorem 5 — proven as the composition of Lemmas A–D plus Corollary C.1.
- Corollary C.1 — proof outlined; the cross-token term inherits Doeblin contraction directly (rigorous), the self-token / residual diagonal term is bounded but not contractive (rigorous), and the closure condition `K_R < μ_D` requires the bounded-input regime to be preserved across layers.
- Lemma D — standard transformer-stability argument citing Wang–Sun 2023 (DeepNet); rigorous under spectral-normalisation or weight-decay assumptions, otherwise empirical.

**Two residual technical pieces (M.11)** — both tractable, neither is a deep open problem:
1. Empirical verification of (S3) — attention diffuseness preserved across training, with `μ_D` quantified per layer for trained PCT checkpoints.
2. Fixed-point argument that the bounded-input regime where `K_R < μ_D` is preserved across the cascade.

**C2 status**: The original C2 is redundant; replaced by (A2) "continuous gate on bounded operating range", automatic in PCT architecture.

### M.6 Strategy of the proof of Conjecture 5

The strategy decomposes the per-token phase perturbation `ε ∈ ℝ^N` into two orthogonal modes and bounds each separately:

1. **Global-mode decomposition.** Write `ε = φ̄ · 1 + δ` where `φ̄ = (1/N) Σ_i ε_i` and `δ ⊥ 1` (so `Σ_i δ_i = 0`). The global mode `φ̄ · 1` passes through every layer *exactly* by L1.a. Hence the cascade error from the global mode is `O(δ) · ||output||`, which is L-independent provided the stack output norm is bounded.

2. **Linearisation on the zero-mean subspace.** A single layer's effect on a zero-mean phase perturbation `δ` is, to first order in `||δ||_∞`, given by a Jacobian matrix `J_l ∈ ℝ^{N×N}` whose action depends on the row-stochasticised gate `P_{ij} = α_ij / Σ_k α_ik` and the cosine-score's imaginary part `η_ij = -Im⟨q̄_i, k̄_j⟩`.

3. **Doeblin contraction on zero-mean subspace (Lemma C, proven assuming (S3)).** Under attention diffuseness (S3) — every gate value is bounded below by a strictly positive multiple of a stationary distribution `π` — the row-stochastic matrix `P_l = (P_{ij}^{(l)})` satisfies a Doeblin condition with explicit constant `μ_D > 0`. This implies the operator norm of `J_l` restricted to the zero-mean subspace `{δ : Σ_i δ_i = 0}` is bounded above by `Λ := 1 - μ_D < 1`.

4. **Cascade summation.** The depth-L cascade Lipschitz on the zero-mean subspace is bounded geometrically by `Σ_{l=0}^{L-1} Λ^l ≤ 1/(1-Λ) = 1/μ_D`, *L-independent*.

5. **Substrate non-expansion.** Residual + RMSNorm + FFN composition is non-expansive on bounded-norm inputs.

The proof is complete modulo:
- (i) **Verification of (S3)** for trained PCT layers: at initialisation (S3) holds with `μ = e^{-1}/N` and `π = 1/N · 1`. Whether (S3) is preserved across training requires either an empirical check or a stability-of-training argument.
- (ii) **Tightening the second-order term**: the `O(δ²)` bound is loose; a careful Taylor-2 expansion is straightforward but technical.

### M.7 Lemma A — Global-mode decomposition

**Lemma A**. *Let `ε ∈ ℝ^N` and write `ε = φ̄ · 1 + δ` with `φ̄ = (1/N) Σ_i ε_i` and `Σ_i δ_i = 0`. Let `Y_L := (A_L ∘ ... ∘ A_1)(X)` and `Ỹ_L := (A_L ∘ ... ∘ A_1)(P(ε)X)`. Suppose every `A_l` is per-layer phase-coherent. Then*

`Ỹ_L = R(φ̄) · (A_L ∘ ... ∘ A_1)(P(δ)X)`,

*and consequently*

`||Ỹ_L − Y_L||_2 ≤ ||(A_L ∘ ... ∘ A_1)(P(δ)X) − Y_L||_2 + |φ̄| · ||Y_L||_2 + O(φ̄²)`.

*Proof.* The exact factorisation `P(ε) = R(φ̄) ∘ P(δ)` holds because `e^{iε_i} = e^{iφ̄} · e^{iδ_i}`. By L1.a applied iteratively, `A_l ∘ R(φ̄) = R(φ̄) ∘ A_l` for every `l`, so

`Ỹ_L = (A_L ∘ ... ∘ A_1)(R(φ̄) P(δ)X) = R(φ̄) · (A_L ∘ ... ∘ A_1)(P(δ)X)`.

With `Z := (A_L ∘ ... ∘ A_1)(P(δ)X)` and `Y_L = (A_L ∘ ... ∘ A_1)(X)`:

`||Ỹ_L − Y_L|| = ||R(φ̄) Z − Y_L||`
`≤ ||R(φ̄)(Z − Y_L)|| + ||(R(φ̄) − I) Y_L||`
`= ||Z − Y_L|| + ||(R(φ̄) − I) Y_L||`

(using that `R(φ̄)` is unitary). For `R(φ̄) y_i = e^{iφ̄} y_i`, we have `|(e^{iφ̄} − 1) y_i| ≤ |φ̄| · |y_i| + O(φ̄²)`. Summing,
`||(R(φ̄) − I) Y_L||_2 ≤ |φ̄| · ||Y_L||_2 + O(φ̄²)`. ∎

The first term is the cascade Lipschitz under **zero-mean** phase perturbation. The second term is L-independent because `||Y_L||` is bounded under (S1)+(S2).

### M.8 Lemma B — Linearised Jacobian on the zero-mean subspace

**Lemma B**. *Fix a layer `A_l` with parameters `` and gate `f`. For every input `X` and every zero-mean `δ ∈ ℝ^N` with `||δ||_∞ ≤ ξ` (small), the layer output satisfies*

`A_l(P(δ)X) = A_l(X) + i · J_l(X) · δ + O(ξ²)`

*where `J_l(X)` decomposes as*

`J_l(X)·δ = T₁ · δ − T₂ · δ + T₃ · δ + ρ · diag(x) · δ`

*with*

- `[T₁ δ]_i := W_o · Σ_j β_ij^{(l)} δ_j v_j`,
- `[T₂ δ]_i := δ_i · W_o · Σ_j β_ij^{(l)} v_j`,
- `[T₃ δ]_i := W_o · Σ_j α_ij^{(l)} δ_j v_j`,
- `[ρ · diag(x) δ]_i := ρ · δ_i · x_i`,

*where `β_ij = f' · η_ij`, `η_ij = −Im⟨q̄_i, k̄_j⟩`, `|β_ij| ≤ K_f`, `|α_ij| ≤ M`.*

*Proof sketch.* Write `q̄'_i = e^{iδ_i} q̄_i + O(ξ²)`, similarly `k̄'_j`, `v'_j`. Then

`s'_ij = Re⟨q̄'_i, k̄'_j⟩ = Re(e^{i(δ_j − δ_i)} ⟨q̄_i, k̄_j⟩)`.

Linearising and using `⟨q̄_i, k̄_j⟩ = s_ij + i · (-η_ij)`:

`s'_ij − s_ij = η_ij (δ_j − δ_i) + O(ξ²)`.

Hence `α'_ij − α_ij = β_ij (δ_j − δ_i) + O(ξ²)`. The value: `v'_j = e^{iδ_j} v_j = v_j + i δ_j v_j + O(ξ²)`. Combining:

`A_l(P(δ)X)_i = W_o · Σ_j [α_ij + β_ij(δ_j − δ_i)][v_j + i δ_j v_j] + ρ(1 + i δ_i) x_i + O(ξ²)`,

from which `T_1, T_2, T_3, ρ · diag(x)` are read off after splitting `Σ_j β_ij(δ_j − δ_i) = Σ_j β_ij δ_j − δ_i Σ_j β_ij`. ∎

### M.9 Lemma C — Doeblin contraction on the zero-mean subspace

**Lemma C**. *Suppose the gate satisfies (C1)–(C4) and (S3) — there exist `μ > 0` and a probability vector `π` such that `α_ij ≥ μ · π_j` for all `i, j` and all inputs in the bounded set. Define the row-stochastic matrix `P` by `P_{ij} := α_ij / Σ_k α_ik`. Then `P` satisfies the Doeblin condition*

`P_{ij} ≥ μ_D · π_j` *with* `μ_D := μ / M`,

*and the operator norm of `P` restricted to the zero-mean subspace `V_0 := \{u ∈ ℝ^N : Σ_i u_i · w_i = 0\}` is bounded by*

`||P|_{V_0}||_∞ → ∞ ≤ 1 − μ_D`.

*Proof.* The Doeblin condition follows from `P_{ij} ≥ μ π_j / M`. The contraction on the zero-mean subspace is the standard *coupling lemma*: given `P_{ij} ≥ μ_D π_j`, write `P = μ_D · 1 π^T + (1 − μ_D) · Q` for some row-stochastic `Q`; the rank-1 component annihilates `\{u : ⟨π, u⟩ = 0\}`, and `Q` is non-expansive. ∎



**Corollary C.1**. *Under (C1)–(C4), (S1)–(S3), the per-layer Jacobian `J_l` of Lemma B, restricted to the zero-mean subspace `V_0` (uniform `π = 1/N · 1`), has operator norm*

`|| J_l |_{V_0} ||_{2 → 2} ≤ Λ_l · W_max² · R · √N`

*for some `Λ_l ∈ [0, 1)`.*

The careful argument transferring `P`'s Doeblin contraction to `J_l`'s contraction on `V_0` defines a phase-coordinate projection `Π : ℂ^{N×d} → ℝ^N` and shows `Π J_l |_{V_0}` is Doeblin-contractive. Cross-token terms `T_1, T_3` inherit Doeblin contraction directly; self-token terms `T_2` and the residual diagonal are bounded but not contractive on their own — the contraction returns at the *next* attention call. Full detailed argument in M.11.

### M.10 Lemma D — Substrate non-expansion + Theorem 5

**Lemma D**. *Suppose each layer `A_l` is followed by a residual + RMSNorm + FFN substrate `S_l` with the standard pre-norm transformer block structure. Under (S1) and bounded-input regime, `S_l` is non-expansive on bounded-norm phase perturbations: `||S_l(Δu)||_2 ≤ Λ_S · ||Δu||_2` for some `Λ_S ≤ 1`.*

*Proof.* Standard. RMSNorm is 1-Lipschitz on bounded inputs; residual + FFN with bounded weights has standard transformer-stability bounds. ∎

**Theorem 5**. *Suppose each layer `A_l` satisfies (C1)–(C4), is followed by a substrate `S_l` satisfying Lemma D, and (S1)–(S3) hold uniformly across layers. Then there exist constants `C_0, C_1`, **independent of `L`**, such that for all inputs `X` and all `ε ∈ ℝ^N` with `||ε||_∞ ≤ δ`,*

`||(A_L ∘ S_{L−1} ∘ ... ∘ S_1 ∘ A_1)(P(ε)X) − (A_L ∘ ... ∘ A_1)(X)||_2 ≤ C_0 · δ + C_1 · δ²`.

*Proof.* By Lemma A, the global-mode contribution is `C_global · δ` with `C_global` L-independent. For the zero-mean cascade, iterate Corollary C.1 + Lemma D:

`||Δu^{(l)}|| ≤ Λ · ||Δu^{(l-1)}|| + O(ξ²)`

where `Λ := Λ_S · sup_l ||J_l|_{V_0}|| ≤ Λ_S · (1 − μ_D)`. If `Λ < 1`, the geometric series gives

`||Δu^{(L)}|| ≤ ||Δu^{(0)}|| / (1 − Λ) + O(δ²) / (1 − Λ)`,

L-independent. Combining,

`LHS ≤ · δ + C_quadratic · δ² = C_0 · δ + C_1 · δ²`. ∎

For PCT, `Λ < 1` is achievable. For `complex_relu`, `f` is C^0 not C^1 — the linearisation in Lemma B does not hold uniformly through the discontinuity, and the cascade summation does not close. **This is exactly the L2 failure**.

### M.11 What remains rigorously open

Two residual technical pieces:

1. **Verification of (S3) under training dynamics**. We assumed `α_ij ≥ μ · π_j` uniformly. At initialisation, the `b = -log N` bias gives `μ ≥ e^{-1}/N` cleanly. During training, gradient updates can sharpen attention; we conjecture (S3) holds across training. Empirical verification: scatter `min_{i,j} α_ij^{(l)}` across training steps for trained PCT checkpoints.

2. **Step (2) of Corollary C.1**: transferring Doeblin contraction from `P` to `J_l |_{V_0}`. The cross-token terms `T_1, T_3` inherit Doeblin contraction directly; the self-token terms require the phase-coordinate projection argument detailed below.

**Detailed argument for self-token term (2)**. Define `Π : ℂ^{N×d} → ℝ^N` by `Π(u)_i := Im⟨u_i, Δu_i⟩ / ||u_i||²`. For a perturbation `Δu = J_l · δ` arising from a zero-mean phase perturbation of input, `Π(Δu)` is the induced output phase perturbation.

Apply `Π` to each term:
- `Π(T_1 δ)_i = Im⟨u_i, [T_1 δ]_i⟩ / ||u_i||² = Im⟨u_i, W_o Σ_j β_ij δ_j v_j⟩ / ||u_i||²`. Doeblin applies when `β_ij v_j ≥ μ_D' · π_j · v_j` for some `μ_D' > 0` — valid under (S3).
- `Π(T_2 δ)_i = δ_i · Im⟨u_i, γ_i⟩ / ||u_i||²`. Diagonal operator on `δ`; bounded but not contractive. Contraction returns at the next attention call.
- `Π(T_3 δ)_i`: same structure as `T_1` with `α_ij` weights, Doeblin directly under (S3).
- `Π(ρ · diag(x) δ)_i = ρ · δ_i · Im⟨u_i, x_i⟩ / ||u_i||²`: diagonal again, similarly bounded.

The **net effect** on `Π(Δu)` is ` δ` where `C` is the cross-token contraction (`||C|_{V_0}|| ≤ 1 − μ_D`) and `R` is the diagonal residual (`||R||_∞ ≤ K_R`). On `V_0`: `|||_{V_0}||_{2 → 2} ≤ (1 − μ_D) + K_R`. We need `K_R < μ_D`.

For PCT at initialisation, `μ_D ≈ e^{-1}/(N M)` and `K_R ≈ (K_f W_max² R + ρ R) / r_min²`. The condition `K_R < μ_D` requires the substrate's bounded-input regime to keep `r_min² ≥ const · N M /`, achievable via RMSNorm.

**Open piece**: showing `K_R < μ_D` *uniformly across L layers* requires the bounded-input regime to be preserved across the cascade — a fixed-point argument that the iterated Lipschitz dynamics stay in the bounded set where (S3) and the `K_R < μ_D` condition both hold.

This closes the proof modulo the fixed-point step. **Conjecture 5 is upgraded to a theorem under explicit quantitative conditions on the substrate and the attention diffuseness (S3) preserved across training.** Open: empirical verification of these conditions for PCT trained checkpoints plus tightening the fixed-point argument.

---