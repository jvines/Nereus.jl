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
- `sample_parallel` chooses the K walks' start points itself, serially, among
  the live points at or above the batch's threshold L_(K), and keeps the
  refit and the start when a start is outside the bound. The commit above
  refit inside each walk for that walk alone and drew another start, so the
  live points outside the bound were chosen less often for as long as the
  bound lasted, which with the refit check above was the whole run: log Z
  came out 1.5–2.7 below pt_emcee's on an RM obliquity fit. Either fix alone
  removes that.
- `HSlice` is a reflective (Hamiltonian) slice sampler: trajectories that
  reflect off the likelihood constraint about the gradient where a step lands
  outside it, the new point drawn uniformly from a trajectory's inside points.
  The commit above sliced along directions built from the likelihood gradient
  at the current point, which depend on the state, so the move was not
  reversible.
- `RWalk` and `RStagger` take exactly `walks` steps of one size, changed
  between chains only; a chain with no step accepted stays at its start. The
  commit above walked on until a step was accepted, shrank the step inside a
  chain that kept failing, and RStagger rescaled its step after every step, so
  the moves did not leave the constrained distribution unchanged. RStagger's
  walk is now RWalk's.
- `Rejection` draws from the whole bound (`proposal_bound`), each ellipsoid
  by its volume and overlaps counted once. The commit above handed it the one
  ellipsoid around a random live point, as the walks are handed, so each
  ellipsoid was drawn as often as live points sit in it.
- `Nested(...; bootstrap)` expands the bounds by dynesty's bootstrap rule
  (`Bounds.fit(rng, B, points; enlarge, bootstrap)`,
  `Bounds.bootstrap_expansion`), by default 5 resamples for `Rejection` and
  none for the walks and slices. A MultiEllipsoid fitted to the live points and
  enlarged 1.25 in volume missed 12–22% of the constrained region of an RM
  obliquity fit, and `:unif` put log Z 2 nats high. `Rejection` counts only its
  likelihood calls against `maxiter`: the expanded bounds can lie mostly
  outside the unit cube while the region is large. While it searches for a
  point it refits the bounds every `update_interval` calls (`draw_uniform`), as
  dynesty does, so an expansion that came out far too large is not drawn from
  until `maxiter` runs out.
- A MultiEllipsoid is split by dynesty's BIC rule (`bic = true`, the default
  now) rather than when its children halve its volume. The volume-halving rule
  cut a connected region into many ellipsoids of 2d+ points each: the
  bootstrap expansion could not make those cover it (1% of a 10-d ball stayed
  outside, against 0.04%), and a walk or slice, whose metric is the ellipsoid
  holding its start, depended on where it started; on a 10-d RM obliquity fit
  (15 ellipsoids against 1–5) log Z came out 0.35–0.5 nats above pt_emcee's.

License: MIT (see LICENSE) — original authors retained.
