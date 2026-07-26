# Path-X: complex screening + phase-coherent recurrence (PCR)

This folder contains the model, code, and mathematical documentation for the
Path-X result of the PCT paper (§6): to our knowledge the **first genuinely
complex-valued neural network to solve Path-X** — the 16,384-token Long Range
Arena connectivity task on which every softmax Transformer scores at chance.

**Result** (rule-compliant: raw 1D token sequence in, one binary label out; no
2D structure, no auxiliary supervision, no handcrafted features; deterministic
full test-split sweep, n = 20,000):

| | Path-X test acc |
|---|---|
| **complex screening + PCR (N = 3 seeds)** | **92.71 ± 0.89** |
| best seed (weights in [`model/`](model/)) | 93.50 |

| Reference scale | |
|---|---|
| Transformer / Reformer / Performer / Linformer / BigBird / Luna-256 | chance (≈50) |
| S4-v1 | 88.10 |
| DSS | 89.72 |
| S4D-LegS | 91.95 |
| S4D-Inv | 92.80 |
| **ours** | **92.71 ± 0.89** |
| MEGA-chunk | 93.81 |
| LRU | 94.20 |
| S4 (S4-LegS) | 96.35 |
| MEGA | 97.98 |
| S5 | 98.58 |

Prior solvers are real-valued architectures that confine complex numbers to a
state-space kernel (S4/S4D/S5/LRU) or use none (MEGA). Here the representation
itself flows as a complex signal through both the recurrence and the attention
score/value paths (~94% of parameters are complex-valued; the transport and
matching paths are 100% complex — the input-dependent gates, norms, and
readout are real *by design*, per PCT's C1–C4).

**Files**

- [`code/pcr_screening.py`](code/pcr_screening.py) — model code, ported verbatim from the bench trainer; self-contained (torch only)
- [`code/load_example.py`](code/load_example.py) — load the checkpoint and classify
- [`model/pytorch_model.pt`](model/pytorch_model.pt) — best-seed weights (state dict, 2,011,660 params); also on the [Hugging Face Hub](https://huggingface.co/complexedleo/pcr-screening-pathx)
- [`model/config.json`](model/config.json) — exact architecture + optimizer config
- [`pathx_section_standalone.pdf`](pathx_section_standalone.pdf) — the paper's Path-X section as a standalone PDF

---

## 1. The task, and why it is a *transport* problem

Path-X asks whether two marked circles in a 128×128 line drawing are connected
by a dashed path. Under LRA rules the image arrives as a **flat sequence of
16,384 pixel tokens** — two pixels that are vertically adjacent in 2D are 128
tokens apart in 1D. Deciding connectivity means composing thousands of
long-range "hops" of stride ≈128, with the discriminative evidence (a few
hundred faint path pixels, or one small gap anywhere along the curve) diluted
across the sequence.

Selection alone — attention deciding *what* to read — cannot solve this: the
model must first *carry* information across positional offsets it has never
been told exist. That is a **transport** problem, and it is exactly the
component the PCT framework assigns to input-independent phase rotation.

## 2. Phase-coherent recurrence (PCR)

### 2.1 The recurrence

Each of the 6 blocks applies, per state channel $n \in \{1,\dots,N\}$
($N=256$ per direction, bidirectional):

$$h_t \;=\; \lambda \odot h_{t-1} \;+\; \gamma \odot (B x_t), \qquad
y_t \;=\; \mathrm{Re}\!\left[C\, h_t\right] + D \odot x_t$$

with $\lambda, \gamma \in \mathbb{C}^N$ diagonal, $B \in \mathbb{C}^{d\times N}$,
$C \in \mathbb{C}^{N\times d}$ (stored as $(\mathrm{re},\mathrm{im})$ pairs).
This is the LRU recipe (Orvieto et al. 2023): a *linear time-invariant* (LTI)
diagonal system, so the entire sequence can be computed as a convolution.

### 2.2 Stable exponential parameterisation

$$\lambda \;=\; \exp\!\big({-\exp(\nu) + i\theta}\big),\qquad
|\lambda| = e^{-e^{\nu}} < 1 \;\text{ for all } \nu \in \mathbb{R}.$$

Learning $\nu$ (log–log magnitude) and $\log\theta$ decouples the decay rate
from the rotation frequency and keeps the system unconditionally stable —
gradient steps can never push an eigenvalue outside the unit disc.

**Ring initialisation.** $|\lambda|^2 \sim \mathcal{U}[r_{\min}^2, r_{\max}^2]$
with $[r_{\min}, r_{\max}] = [0.999, 0.9999]$: at sequence length $L = 16{,}384$,
$0.999^{16384} \approx 7\times10^{-8}$ while $0.9999^{16384} \approx 0.19$, so
the init population spans memories from "a few hundred steps" to "the whole
sequence".

### 2.3 $\gamma$-normalisation

A near-unit-circle eigenvalue integrates a long window; without compensation
the state variance blows up as $\sum_{k\ge0}|\lambda|^{2k} = (1-|\lambda|^2)^{-1}$.
Setting

$$\gamma \;=\; \sqrt{1-|\lambda|^2}$$

gives every channel unit response energy
($\gamma^2 \sum_k |\lambda|^{2k} = 1$), so channels with wildly different
timescales coexist at one learning rate.

### 2.4 FFT form

Because the system is LTI, unrolling gives a causal convolution with kernel

$$k_t \;=\; \gamma\,\lambda^{t}\quad (t = 0,\dots,L-1), \qquad
h = k * (Bx),$$

computed exactly in $O(L\log L)$ by zero-padded FFT of length $2L$ (linear —
not circular — convolution). The backward direction uses independent
parameters on the reversed sequence; summing the two directions'
$C$-projections is algebraically identical to concatenation followed by a
single linear map, since
$[C_f\; C_b]\,[h_f; h_b] = C_f h_f + C_b h_b$.

### 2.5 Why the *phase* is the position mechanism

After $\Delta$ steps a unit of input is multiplied by

$$\lambda^{\Delta} \;=\; |\lambda|^{\Delta}\, e^{\,i\theta\Delta}.$$

The phase advances **linearly in the offset** $\Delta$, uniformly for every
token — an input-independent rotary code, the continuous-time analogue of
RoPE, realised by the dynamics instead of being bolted onto attention. A bank
of channels with different $\theta$ is a learned Fourier basis over offsets:
downstream real-linear readouts of $\mathrm{Re}[C h]$ can form
interference patterns that peak at specific relative offsets (e.g. the ±128
stride of the pixel grid) — *constructive* where phases align, *destructive*
elsewhere. This is precisely PCT's multi-layer phase coherence transplanted
into a recurrence: **transport happens in phase space; nothing input-dependent
touches the phase along the way.**

Two single-variable ablations pin this down (paper §6):

- **Phase necessity.** Fix $\theta = 0$ (real eigenvalues; an S4D-Real
  analogue, everything else identical): the model **never leaves chance**.
  Without rotation, $\lambda^\Delta$ is a monotone decay — all offsets look
  alike up to scale, and no positional geometry can form.
- **Phase bandwidth ⇒ generalisation.** Initial band
  $\theta \in [0, \pi/50]$ (slowest oscillation period $\ge 100$ steps):
  training accuracy reaches $0.999$ while **test stays at 0.53** — the model
  *memorises* 160k training images through low-frequency signatures but forms
  no reusable transport. Widening to $\theta \in [0, \pi/10]$ (periods down to
  20 steps) recovers genuine generalisation (0.92+ test). Phase is required in
  a graded way: none ⇒ cannot learn; too narrow ⇒ can only memorise.

The second ablation also separates **"ignition"** (train loss leaving
$\ln 2$) from generalisation — on this task the two can diverge completely,
so all reported numbers are deterministic full sweeps of held-out splits.

## 3. Complex screening attention (the selection half)

Every second/fourth block inserts the PCT paper's non-competing screening
gate, operating on the same complex signal (chunked to windows of 1024; the
recurrence carries information *between* chunks):

1. **Complex projections** $q,k,v,g \in \mathbb{C}^{d_h}$ per head (stored as
   $(\mathrm{re},\mathrm{im})$ pairs).
2. **Phase-aligned score.** With $\bar q = q/\lVert q\rVert$,
   $\bar k = k/\lVert k\rVert$ ($\mathbb{C}$-norms):
   $$s_{ij} \;=\; \mathrm{Re}\,\langle \bar q_i, \bar k_j\rangle \;\in\; [-1, 1].$$
   The score is maximal when the two tokens' phase patterns align — matching
   by constructive interference.
3. **Trim-and-square gate — no softmax, no row normalisation:**
   $$\alpha_{ij} \;=\; r^2\,\mathrm{ReLU}(s_{ij} - t)^2,\qquad
   r = e^{s_r}+1,\;\; t = 1 - 1/r,$$
   with one learnable width $s_r$ per head. Each pair $(i,j)$ is admitted or
   rejected **on its own absolute relevance**: tokens do not compete for a
   probability budget (C1), the gate is real-valued and smooth (C2), never
   flips sign (C3), and is element-independent (C4).
4. **Aggregation and magnitude control.** $u_i = \sum_j \alpha_{ij} v_j$
   (complex, un-normalised), then TanhNorm
   $u \mapsto \tanh(\lVert u\rVert)\,u/\lVert u\rVert$ — a *radial* squash that
   bounds magnitude while leaving phase untouched.
5. **modReLU output gate.** $g \mapsto \mathrm{ReLU}(|g|+b)\,\tfrac{g}{|g|}$,
   then the complex Hadamard product $u \odot g$ — again gating magnitude
   only, preserving phase geometry.

Division of labour: **PCR supplies position (where information travels),
screening supplies selection (what is read once it arrives).** In the original
screening layer the positional slot was a fixed cosine softmask; here that
slot is filled by a *learned, phase-coherent* mechanism instead. The hybrid's
N=1 comparison against PCR-only at matched budget improved the best-seed
result (93.50 vs 92.54 ± 0.28), consistent with selection adding on top of
transport rather than substituting for it.

## 4. What stays real, and why

The PCT conditions C1–C4 assign every *input-dependent* nonlinearity to the
real domain: the GLU gates after each recurrence, all normalisation
statistics, and the classification head. Phase is only ever (a) rotated by
input-independent dynamics, (b) compared via inner products, or (c) scaled by
non-negative reals. Anti-phase deletion — an input-dependent operation that
can flip or erase phase — is exactly the deviation that the paper's 6-cell
comparison identifies as the dominant failure axis, and its recurrence-side
echo is the $\theta=0$ ablation above.

## 5. Training recipe (as released)

| | |
|---|---|
| optimiser | AdamW, base lr $4.5\times10^{-4}$, wd 0.05 |
| recurrence params $(\nu,\theta)$ | lr × 1/3, **no** weight decay |
| $B, C$ | lr × 1/3, with weight decay |
| schedule | warmup 2,500 → hold → linear decay from 200k to ×0.1 |
| steps / batch | 250,000 / 32 (seq len 16,384) |
| eval | deterministic full-split sweep only (no sampled eval for decisions) |

Full config in [`model/config.json`](model/config.json). Training used the
bench trainer in [`../code/`](../code/); the model classes here are a verbatim
extraction and load the released checkpoint with `strict=True`.

## 6. Reproduce / use

```bash
cd pathx
python code/load_example.py   # loads model/pytorch_model.pt, runs a forward pass
```

Data: LRA Path-X (`pathfinder128`, 160k/20k/20k split), pixels flattened
row-major to 16,384 uint8 tokens.
