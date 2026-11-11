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
- `RWalk` and `RStagger` count a step out of the unit cube as a rejected step
  instead of drawing again until one lands inside.
- The bounds are refit once `since_update` (likelihood calls since the last
  fit) reaches `update_interval`, as in dynesty. The commit above refits only
  when `since_update` is an exact multiple of it, which a count that grows by a
  whole proposal at a time almost never is, so the first bound stayed for the
  whole run.
- A live point found outside the bound refits the bound and stays the walk's
  start (`Bounds.bound_containing`); the commit above drew another live point,
  so points outside a stale bound were chosen less often than the rest.

License: MIT (see LICENSE) — original authors retained.
