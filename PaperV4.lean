/-
Root module for the Lean 4 formalisation of `paper/outline_v4.md`, Appendix M.

The submodules below mirror the section numbering of Appendix M:

  PaperV4.Basic   — M.0  Setting and notation (Token, TokenSeq, R(φ), P(ε))
  PaperV4.L1      — M.1, M.2  Definition 1 + Theorem 1 (C1 + C4 ⇒ L1)
  PaperV4.LemmaA  — M.7  Lemma A (global-mode decomposition)
  PaperV4.L2      — M.3, M.4, M.6, M.10  Definitions 3–4, Theorem 5 (statements)
  PaperV4.LemmaC  — M.9  Lemma C (Doeblin contraction; statement)

Status legend used in the file headers:
  [PROVEN]      — full Lean proof, no `sorry`
  [PROVEN-MOD]  — proof relies on a small named auxiliary `sorry`
                  (each such gap is documented inline)
  [STATEMENT]   — only the statement is given; the proof is `sorry`
                  (mirrors the body's "drafted modulo …" markers)
-/

import PaperV4.Basic
import PaperV4.L1
import PaperV4.LemmaA
import PaperV4.LemmaC
import PaperV4.L2
