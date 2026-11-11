# Vendored NestedSamplers.jl

This is a vendored, in-tree copy of the Nereus fork of NestedSamplers.jl.

- Source: https://github.com/jvines/NestedSamplers.jl
- Branch: fix/high-dim-stability
- Commit: 64b0ff7248e86d6ae5d2ff255f4dfc94af259c41
  (feat(bounds,proposals): MLFriends region bound + HSlice reflective slice)
- Vendored: 2026-07-15

Carries the fork-only `Bounds.MLFriends` + `Proposals.HSlice` that
`src/samplers/nested.jl` uses (`bounds=:mlfriends`, `proposal=:hslice`).
Upstream (TuringLang/NestedSamplers.jl) is unmaintained; this in-tree copy
replaces the former sibling-clone path dep so the build is self-contained and
portable. To update: pull the fork, re-copy `src/` + `Project.toml`, bump the
commit above.

Changed here since the commit above, so carry them over on an update:

- `sample_parallel` takes `callback`, called after the bootstrap step and after
  each batch; Nereus's `sample_nested` uses it for its progress bar.
- `Slice`, `RSlice`, `RWalk` and `RStagger` chain their steps: each slice or
  walk step starts where the previous one ended. The commit above starts every
  step from the copied live point, so the new point is one step from it
  whatever `slices`/`walks` say.

License: MIT (see LICENSE) — original authors retained.
