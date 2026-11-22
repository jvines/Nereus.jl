# Changelog

Changes that matter to someone fitting with Nereus. Older releases are in the
git history (`git log --first-parent develop`).

## Unreleased

### Added

- **CCF line components (`data.ccf`).** The CCFs of a spectroscopic blend are
  fitted directly as a sum of Gaussian components, each static (centre γ) or
  tied to a planet's orbit (centre γ + K·[cos(f + ω) + e cos ω], from that
  planet's own P, time anchor, e and ω, K of either sign). Per component:
  γ, K (tied only), FWHM, depth. Per epoch, or shared: continuum, depth scale
  (the reference epoch's is 1, which removes the depth/scale degeneracy), white
  jitter. Optional per-pixel errors; NaN pixels are dropped; velocity grids may
  differ between epochs. It is an ordinary target term: every engine, evidence
  estimator and checkpoint sees it, and it combines with RVs, photometry,
  transit times and astrometry through the shared orbit. Julia: `CCFSet`,
  `CCFComponent`, `ccf_log_likelihood`, `ccf_model`,
  `ccf_component_velocities`, `load_ccf_json`. Job config: `data.ccf`.
  Docs: docs/src/ccf.md. Checked against an independent least-squares analysis
  of NOI-101822's FEROS CCFs (`test/test_ccf_parity_noi101822.jl`).
- `jitter_names` includes the CCF jitters, so a CCF jitter at zero is not
  flagged as railed.

### Notes

- The CCF likelihood treats the pixels of a CCF as independent. CCFs are
  oversampled, so CCF-only credible intervals are lower bounds.
- `ad_backend = :Enzyme` refuses a target with CCFs (it would drop the term);
  ForwardDiff differentiates it.
- Checkpoints written before this release still resume: data without CCFs
  hashes as before.
