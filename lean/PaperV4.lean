/-
Root module for the Lean 4 formalisation of `paper/outline_v4.md`, Appendix M.

The submodules below mirror the section numbering of Appendix M:

  PaperV4.Basic   — M.0  Setting and notation (Token, TokenSeq, R(φ), P(ε))
  PaperV4.L1      — M.1, M.2  Definition 1 + Theorem 1 (C1 + C4 ⇒ L1)
  PaperV4.LemmaA  — M.7  Lemma A (global-mode decomposition)
  PaperV4.L2      — M.3, M.4, M.6, M.10  Definitions 3–4, quantitative
                    Lemma A bound, geometric cascade recurrence, and
                    Theorem 5 (proven from standard per-layer
                    hypotheses: non-expansiveness, uniform zero-mean
                    phase stability, bounded output range)
  PaperV4.LemmaC  — M.9  Lemma C (Doeblin contraction; proven)

Status legend used in the file headers:
  [PROVEN]      — full Lean proof, no `sorry`
  [STATEMENT]   — only the statement is given; the proof is `sorry`

As of 2026-08-05 the project contains NO `sorry` and NO bespoke
premises: every declaration depends only on the three standard
Lean / Mathlib axioms [propext, Classical.choice, Quot.sound], and
every hypothesis of every theorem is a standard mathematical condition
(linearity, row-stochasticity, Doeblin minorisation, Lipschitz /
non-expansiveness, uniform Lipschitz phase stability, bounded range).
The hypothesis bundle of Theorem 5 is satisfiable
(`theorem5Hypotheses_satisfiable`), so nothing holds vacuously.
-/

import PaperV4.Basic
import PaperV4.L1
import PaperV4.LemmaA
import PaperV4.LemmaC
import PaperV4.L2
