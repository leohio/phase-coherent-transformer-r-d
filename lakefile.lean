import Lake
open Lake DSL

package paperv4 where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "master"

@[default_target]
lean_lib PaperV4 where
  globs := #[.andSubmodules `PaperV4]
