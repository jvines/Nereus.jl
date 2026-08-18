# Noise model type hierarchy.
#
# Three pipeline stages (composition rules):
#
#   Stage 1 (MeanModifier): additive corrections to the mean RV model.
#           Compose freely — any number can be active simultaneously.
#
#   Stage 2 (SequentialNoise): sequential correlation applied to residuals.
#           Multiple allowed (AR + MA = ARMA). Orders are fixed at
#           construction — not trans-dimensional. Trans-dim toggles
#           components on/off, not order selection.
#           MUTUALLY EXCLUSIVE with Stage 3.
#
#   Stage 3 (CovarianceNoise): covariance structure for the likelihood.
#           At most one active. WhiteNoise is the implicit default.
#           MUTUALLY EXCLUSIVE with Stage 2 — GPs absorb correlated
#           noise, making AR/MA redundant.
#
# References:
#   Tuomi et al. 2013, A&A 551, A79 — AR/MA exponential-decay models
#   Foreman-Mackey et al. 2017, AJ 154, 220 — Celerite GP kernels
#   Dumusque et al. 2012, Nature 491, 207 — activity indicator decorrelation
#   Diaz et al. 2016, A&A 585, A134 — activity-dependent jitter (concept)
#   Feng et al. 2016, MNRAS 461, 2440 — activity-dependent jitter (RJ model, Eq. 16)

abstract type NoiseModel end
abstract type MeanModifier <: NoiseModel end
abstract type SequentialNoise <: NoiseModel end
abstract type CovarianceNoise <: NoiseModel end

#   Stage 3b (AdditiveCovariance): a low-rank / diagonal correction ADDED
#           on top of whatever base covariance the channel already has
#           (white default, or one CovarianceNoise GP). Composes freely
#           with the base and with each other via Woodbury — Σ = B + FΛFᵀ
#           factored through the base's own solve + logdet, so it never
#           trips the "at most one CovarianceNoise" rule. These are the
#           parametric intermediate models between white and GP: nightly
#           calibration offsets, marginalized rotation-harmonic blocks.
abstract type AdditiveCovariance <: NoiseModel end

"""
    ActivityDecorrelation(; indicators, per_instrument=true)

Linear decorrelation against activity indicators (Dumusque+ 2012).
Adds `C_j[inst] * indicator_j(t)` to the mean model. Indicators
are time-series like log R'HK, S-index, BIS, FWHM.
Stage 1 — composes freely with everything.
"""
struct ActivityDecorrelation <: MeanModifier
    indicators::Vector{String}
    per_instrument::Bool
    derivative::Bool   # FF'-style — add a Cdot_<ind>_<ins> coefficient on
                       # the time derivative of each indicator. Captures
                       # the rotation-phase-shifted activity-RV
                       # correlation that a single coefficient on the
                       # instantaneous indicator misses (Aigrain, Pont &
                       # Zucker 2012, MNRAS 419, 3147; Rajpaul+ 2015).
    # Optional name tag appended to this model's coefficients
    # (`C_<ind>_<ins>_<label>`). Lets two ActivityDecorrelation models
    # coexist in ONE layout with DISJOINT parameters — the setup for a
    # trans-dim comparison of distinct decorrelation hypotheses, e.g. plain
    # linear (`derivative=false`) vs FF′ (`derivative=true`) in a
    # `noise_exclusion_groups` entry: the occupancy then MEASURES whether the
    # activity-RV coupling is instantaneous or needs the derivative term.
    # Empty (default) ⇒ bare `C_<ind>_<ins>` names, fully back-compatible.
    label::String
end

ActivityDecorrelation(; indicators::Vector{String},
                       per_instrument::Bool=true,
                       derivative::Bool=false,
                       label::String="") =
    ActivityDecorrelation(indicators, per_instrument, derivative, label)

"""Suffix appended to an ActivityDecorrelation's coefficient names so two
models can coexist with disjoint params. Empty label ⇒ bare names."""
_ad_suffix(m::ActivityDecorrelation) = isempty(m.label) ? "" : "_" * m.label

"""
    ActivityJitter(; indicator="log_rhk")

Activity-dependent jitter (Diaz+ 2016, A&A 585, A134;
Feng+ 2016, MNRAS 461, 2440, Eq. 16):
σ_J(t) = σ_base + σ_act * indicator(t).
Replaces constant per-instrument jitter with time-varying jitter
that scales with an activity indicator.
Stage 1 — composes freely with everything.
"""
struct ActivityJitter <: MeanModifier
    indicator::String
end

ActivityJitter(; indicator::String="log_rhk") = ActivityJitter(indicator)

"""
    IndicatorFloor(; channels=[:bis,:fwhm,:halpha,:logrhk])

Null model for the activity-indicator channels, scoring the indicator
data block so a trans-dim noise SELECTION that includes `ActivityGP` is
well-posed: AGP scores the JOINT p(RV, indicators), every other noise
model scores only p(RV), so without a floor the trans-dim chain compares
different datasets and its occupancy is not a model posterior. With the
floor always active, an AGP-INACTIVE state scores p(RV|model)·p(y_I|floor)
and an AGP-ACTIVE state scores p(RV,y_I|GP) — the SAME data either way,
so the occupancy is a valid P(M|D). The floor self-gates: skips any
channel currently covered by an active (joint) ActivityGP. No effect on
any config without an ActivityGP toggle.

`kernel` sets the indicator model:
- `:white` — iid Gaussian about zero, `y_{I,c,i} ~ N(0, σ_floor_c² + err_i²)`.
  ONE param `ind_floor_<ch>` per channel.
- `:qp` — a per-channel quasi-periodic GP (the SAME kernel family AGP uses
  for its indicator block), with SHARED `(ind_floor_period, _lambda_e,
  _lambda_p)` and per-channel `(ind_floor_<ch>_amp, _<ch>_jit)`.

⚠ `:white` is a STRAWMAN null for activity indicators, which are temporally
CORRELATED. AGP models that correlation with a GP and the white floor cannot,
so AGP banks the indicator-correlation evidence (HD 18599: ≈ +82 nats) for
free — that advantage is irrelevant to the RV/planet science but leaks into
the AD↔AGP occupancy (drags AGP up to a spurious ~15%). `:qp` gives the floor
the same correlation-capturing power, so the indicator-evidence terms ≈ cancel
and the occupancy reflects the RV-conditional comparison (AD wins). Use `:qp`
for any AD-vs-AGP selection on real, correlated indicators. NOT a
covariance/sequential model — it adds the indicator data block, orthogonal
to the RV noise.
"""
struct IndicatorFloor <: NoiseModel
    channels::Vector{Symbol}
    kernel::Symbol
end

IndicatorFloor(; channels::Vector{Symbol} = [:bis, :fwhm, :halpha, :logrhk],
                 kernel::Symbol = :white) = IndicatorFloor(channels, kernel)

"""
    MAModel(; order=1, per_instrument=false, channel=:rv)

Moving average noise model with exponential decay
(Tuomi+ 2013, A&A 551, A79). Parameters: (ω_j, β_j) per order j.
ω controls correlation strength, β controls timescale.

`channel` (default `:rv`) selects which observation block the model
acts on; `:phot` applies the same ARMA process to the photometric
residuals. `per_instrument=true` gives each instrument on the channel
its own coefficients and constrains the sequential correlation to
within-instrument cadences only.

Stage 2 — can coexist with ARModel (AR+MA = ARMA).
Mutually exclusive with a global Stage 3 GP on the same channel.
Order is fixed at construction, not trans-dimensional.
"""
struct MAModel <: SequentialNoise
    order::Int
    per_instrument::Bool
    channel::Symbol
end

MAModel(; order::Int=1, per_instrument::Bool=false, channel::Symbol=:rv) =
    MAModel(order, per_instrument, channel)

"""
    ARModel(; order=1, per_instrument=false, channel=:rv)

Autoregressive noise model with exponential decay
(Tuomi+ 2013, A&A 551, A79). Parameters: (φ_j, α_j) per order j.
φ controls correlation strength, α controls timescale.

`channel` (default `:rv`) selects which observation block the model
acts on. `per_instrument=true` gives each instrument on the channel
its own coefficients and constrains the sequential correlation to
within-instrument cadences only.

Stage 2 — can coexist with MAModel (AR+MA = ARMA).
Mutually exclusive with a global Stage 3 GP on the same channel.
Order is fixed at construction, not trans-dimensional.
"""
struct ARModel <: SequentialNoise
    order::Int
    per_instrument::Bool
    channel::Symbol
end

ARModel(; order::Int=1, per_instrument::Bool=false, channel::Symbol=:rv) =
    ARModel(order, per_instrument, channel)

# Each CovarianceNoise subtype carries two scoping fields:
#   - `channel::Symbol` (`:rv` default, or `:phot`): which observation
#     block the GP attaches to.
#   - `instruments::Vector{String}` (default `String[]`, meaning "all
#     instruments on this channel"): if non-empty, the GP only acts on
#     observations whose instrument name is in this list. Multiple GPs
#     with disjoint `instruments` sets can coexist on the same channel
#     (per-instrument GPs); the remaining instruments fall back to
#     white noise.
#
# Parameter naming: `_gp_suffix(nm)` (in `noise/param_names.jl`) builds
# a single suffix combining channel and instrument list, so each GP gets
# its own hyperparameter slots in the layout.
#
# Convention: per channel, you may have either one GP with empty
# `instruments` (the "global" GP, applies to all observations on that
# channel) OR any number of GPs with non-empty `instruments` whose sets
# are pairwise disjoint. SequentialNoise (AR/MA) is still mutually
# exclusive with a global GP on the same channel.

"""
    CeleriteSHO(; channel=:rv, instruments=String[])

Simple Harmonic Oscillator GP kernel
(Foreman-Mackey+ 2017, AJ 154, 220).
Parameters: gp_S0, gp_omega0, gp_Q. O(n) evaluation via
celerite semi-separable solver.

`instruments` (default `String[]`): if empty, the GP is global on
`channel`; if non-empty, only observations on the listed instruments
contribute, and other GPs may cover the remaining instruments.

Stage 3 — see module docstring for composition rules.
"""
struct CeleriteSHO <: CovarianceNoise
    channel::Symbol
    instruments::Vector{String}
end

CeleriteSHO(; channel::Symbol = :rv, instruments::Vector{String} = String[]) =
    CeleriteSHO(channel, instruments)

"""
    CeleriteRotation(; channel=:rv, instruments=String[])

Rotation GP kernel — celerite2 formulation (Foreman-Mackey 2018,
RNAAS 2, 31). Two SHO terms at P_rot and P_rot/2 (fundamental +
first harmonic), each with its own quality factor.

Parameters:
  gp_sigma  — overall amplitude of the GP
  gp_period — primary rotation period (days)
  gp_Q0     — quality factor of the secondary mode (P/2)
  gp_dQ     — Q1 − Q2, difference in quality factors
  gp_f      — fractional amplitude of the secondary mode (0 ≤ f ≤ 1)

This is the **physically motivated** parameterization where amplitude
and quality factors are separated. Use when you want each harmonic's
coherence to vary independently.

Pass `instruments=["HARPS", ...]` to restrict this GP to a subset of
instruments on `channel` (per-instrument GP); empty means global.

Stage 3 — see module docstring for composition rules.
"""
struct CeleriteRotation <: CovarianceNoise
    channel::Symbol
    instruments::Vector{String}
end

CeleriteRotation(; channel::Symbol = :rv, instruments::Vector{String} = String[]) =
    CeleriteRotation(channel, instruments)

"""
    CeleriteRotationFM17(; channel=:rv, instruments=String[])

Rotation GP kernel — original Foreman-Mackey+ 2017 (AJ 154, 220)
formulation, as used in astroEMPEROR. One real exponential + one
complex (damped cosine) term sharing a common decay timescale:

    k(τ) = a (1+f)/(2+f) · exp(−τ/τ_decay)
         + a /  (2+f)    · exp(−τ/τ_decay) · cos(2π τ / P)

Parameters:
  gp_log_amp       — log of overall amplitude `a`
  gp_log_timescale — log of decay timescale `τ_decay` (days)
  gp_log_period    — log of rotation period `P` (days)
  gp_log_factor    — log of mixture factor `f` (≥ 0)

This is the **historical** parameterization used in the original
celerite paper. Mathematically distinct from `CeleriteRotation`
above: only one decay timescale shared between both modes, and no
separate quality factor for the harmonic.

Pass `instruments=["HARPS", ...]` to restrict this GP to a subset of
instruments on `channel` (per-instrument GP); empty means global.

Stage 3 — see module docstring for composition rules.
"""
struct CeleriteRotationFM17 <: CovarianceNoise
    channel::Symbol
    instruments::Vector{String}
end

CeleriteRotationFM17(; channel::Symbol = :rv,
                       instruments::Vector{String} = String[]) =
    CeleriteRotationFM17(channel, instruments)

"""
    MaternGP(; channel=:rv, instruments=String[])

Short-memory Matérn-3/2 GP on the RV residuals — the short-memory-class
representative, distinct from the oscillatory `CeleriteSHO` and the
rotation kernels. Uses the EXACT semiseparable (S+LEAF) Matérn-3/2
(`SSMatern32`, rank-2, midpoint-centred generators), NOT the cusped
celerite2 Matérn32Term approximation. Scored through the O(N) semiseparable
solver (a single-series `α=1, β=0` case of the multiseries machinery).

Parameters:
  matern_sigma — GP amplitude σ (`k(0)=σ²`)
  matern_rho   — length scale ρ (days)

Composes with the parametric `AdditiveCovariance` corrections (NightlyOffset,
HarmonicBlock) via a semiseparable-base Woodbury.

Stage 3 — see module docstring for composition rules.
"""
struct MaternGP <: CovarianceNoise
    channel::Symbol
    instruments::Vector{String}
end

MaternGP(; channel::Symbol = :rv, instruments::Vector{String} = String[]) =
    MaternGP(channel, instruments)

"""
    ActivityGP(; channels=[:bis, :fwhm], use_derivative=true,
                 instruments=String[])

Multivariate activity GP ([Rajpaul+ 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.2269R/abstract)):
a single latent Gaussian process `G(t)` (quasi-periodic kernel) drives
the RV and a list of activity indicators jointly via per-channel
linear combinations of `G(t)` and its time derivative `dG/dt`:

    ΔRV(t)   = Vc · G(t) + Vr · dG/dt
    BIS(t)   = Bc · G(t) + Br · dG/dt        (when :bis ∈ channels)
    FWHM(t)  = Fc · G(t) + Fr · dG/dt        (when :fwhm ∈ channels)
    logR'HK = Lc · G(t)                       (when :logrhk ∈ channels; no dG/dt)
    Hα(t)   = Hc · G(t) + Hr · dG/dt         (when :halpha ∈ channels)

The joint data covariance is built from kernel blocks `k_GG`, `k_GdotG`,
`k_dotGG`, `k_dotGdotG` (analytic derivatives of the quasi-periodic
kernel) weighted by the channel coefficients. Captures rotation-phase-
shifted activity-RV coupling that BIS-only `ActivityDecorrelation`
+ FF′ ([Aigrain+ 2012](https://ui.adsabs.harvard.edu/abs/2012MNRAS.419.3147A/abstract))
cannot reach.

Parameters (per active ActivityGP). `G(t)` is UNIT-VARIANCE per
Rajpaul+ 2015 — there is deliberately NO kernel amplitude parameter;
all scale lives in the per-channel couplings (a free amplitude is an
exact non-identifiability against them):
  gp_act_period  — rotation period `P_rot` (days)
  gp_act_lambda_e — exponential decay length scale (days)
  gp_act_lambda_p — periodic length scale (dimensionless)
  Vc, Vr         — RV coefficients on G and dG/dt
  Bc, Br         — BIS coefficients (when :bis active)
  Fc, Fr         — FWHM coefficients (when :fwhm active)
  Lc             — logR'HK coefficient (when :logrhk active)
  Hc, Hr         — Hα coefficients (when :halpha active)

`use_derivative = false` zeros out the dG/dt coefficients globally —
useful when matching the FF′ simplification.

`instruments` restricts the GP to specific RV instruments (same
convention as the celerite kernels); empty means global.

Stage 3 — coexists with `ActivityDecorrelation` / FF′ as a third
option; the trans-dim machinery picks between them on log Z.
"""
struct ActivityGP <: CovarianceNoise
    channels::Vector{Symbol}
    use_derivative::Bool
    instruments::Vector{String}
    # When true, the likelihood returns the CONDITIONAL Gaussian
    # `log p(RV | indicators)` instead of the joint
    # `log p(RV, indicators)`.
    #
    # ⚠ DIAGNOSTIC ONLY — NEVER use this as a model-selection / log-Z objective.
    # The conditional drops the indicator marginal log p(y_I | θ), the
    # only term anchoring the kernel + couplings to the indicator data.
    # Provably (Σ_cond ⪰ the RV noise floor) the model can then buy
    # unbounded conditional sharpness for free: couplings rail, λ_p
    # rails, the planet is absorbed into the conditional mean, and
    # "evidence" inflates by hundreds of nats (HD 18599 post-mortem
    # 2026-06-12). For honest model comparison use the CHAIN RULE:
    #   log Z_cond = log Z(joint run) − log Z(indicators_only run),
    # which equals log p(y_R | y_I) with hyperparams anchored by
    # p(θ | y_I) and is directly comparable to ActivityDecorrelation's
    # evidence.
    marginalize_indicators::Bool
    # When true, the likelihood scores ONLY the indicator block
    # log p(y_I | θ) — the RV path continues through the standard
    # white-noise/celerite machinery untouched. This is the second
    # term of the chain rule above: run a 0-planet fit with
    # `indicators_only = true` and subtract its log Z (minus the
    # trivially-factoring white-RV part, which cancels in the
    # difference of two such runs) from the joint run's.
    indicators_only::Bool
    # Latent-kernel family for the shared activity process G(t).
    #   :qp_dense (default) — the quasi-periodic latent kernel scored through
    #     the existing dense / Woodbury-low-rank builders. Behaviour unchanged.
    #   :matern32 / :sho / :es / :mep / :esp — an O(N·r²) SEMISEPARABLE latent
    #     kernel (src/noise/multiseries_gp.jl). Same Rajpaul (a·G + b·Ġ)
    #     couplings, but the joint marginal is factored in linear time via the
    #     S+LEAF LDLᵀ solver — the only tractable path at decade-baseline,
    #     10³⁺-point cadences. The QP hyperparameters map as amp→σ, λe→ρ,
    #     P→P, λp→η (η halved for :mep/:esp — the ÷2 convention trap).
    latent_kernel::Symbol
    # Solver backend for the covariance factorization. :auto uses the
    # semiseparable solver whenever `latent_kernel` selects a semiseparable
    # family (it always applies there), else the low-rank/dense QP path.
    # :dense / :lowrank / :semiseparable force a specific builder (diagnostics).
    backend::Symbol
end

ActivityGP(; channels::Vector{Symbol} = [:bis, :fwhm],
             use_derivative::Bool = true,
             instruments::Vector{String} = String[],
             marginalize_indicators::Bool = false,
             indicators_only::Bool = false,
             latent_kernel::Symbol = :qp_dense,
             backend::Symbol = :auto) =
    ActivityGP(channels, use_derivative, instruments,
               marginalize_indicators, indicators_only,
               latent_kernel, backend)

"""
    ErrorScale(; instruments=String[])

Per-instrument multiplicative error-scale factor: `σ_i² → f_ins²·σ_i²`,
one parameter `errscale_<ins>` per RV instrument (or per listed
instrument; empty = all RV instruments).

This is a **distinct hypothesis** from additive jitter — "the pipeline's
quoted errors are miscalibrated" (a multiplicative rescale of the FORMAL
error) versus "there is an extra, constant noise source" (an additive
`σ_jit²`). Motivated by the CDP finding that archival formal errors are
off by up to two orders of magnitude. The two are partially degenerate on
homoscedastic errors but separable on heteroscedastic ones: `f` scales
each point proportionally to its quoted error, jitter adds a flat floor.

White-class treatment. When active for an instrument it **replaces** that
instrument's additive jitter (`var = f²·σ_formal²`, no `+σ_jit²`) — so it
is the alternative, not an addition. Put it in a `noise_exclusion_groups`
entry with any other white treatment so the trans-dim menu carries at most
one. Composes with a base GP: `f²σ_i²` feeds the GP's diagonal too.

Stage 1 — applied at the variance-build step (like `ActivityJitter`).
"""
struct ErrorScale <: MeanModifier
    instruments::Vector{String}
end

ErrorScale(; instruments::Vector{String} = String[]) = ErrorScale(instruments)

"""
    NightlyOffset(; channel=:rv, instruments=String[], gap=0.5)

Nightly shared calibration offsets (grouped random effects): every point
in one observing night of one instrument shares a draw
`δ_night ~ N(0, σ_night²)`, marginalized analytically → a per-night
**rank-one block** added to the covariance. One parameter
`night_sigma_<ins>` per instrument.

The parametric little sibling of the phase-2 S+LEAF LEAF calibration
block — it fills the calibration/systematics class now, at O(N) via
Woodbury, without waiting for the block-sparse machinery. A "night" is
`gap`-clustered (a maximal run of an instrument's epochs with consecutive
gaps < `gap` days, default 0.5) — physical time structure, not observed-
epoch adjacency, so the model's marginal is invariant under subsampling
(unlike a cadence-conditioned MA(1)).

Stage 3b (`AdditiveCovariance`) — composes on top of the white default or
a base GP.
"""
struct NightlyOffset <: AdditiveCovariance
    channel::Symbol
    instruments::Vector{String}
    gap::Float64
end

NightlyOffset(; channel::Symbol = :rv, instruments::Vector{String} = String[],
                gap::Float64 = 0.5) = NightlyOffset(channel, instruments, gap)

"""
    HarmonicBlock(; channel=:rv, instruments=String[], nharm=3)

Marginalized rotation-harmonic block: a phase-COHERENT activity signal at
the rotation period `P_rot` plus `nharm−1` harmonics
(`P_rot, P_rot/2, …, P_rot/nharm`), with the cos/sin amplitudes given
zero-mean unit-variance Gaussian priors and **marginalized analytically**
→ a low-rank covariance addition `Σ += F Fᵀ`.

Linear-Gaussian realization of "shared phase, per-instrument amplitude":
the marginalized latent `(a_k, b_k) ~ N(0, I)` per harmonic is SHARED
across instruments (→ one coherent phase), while a free per-instrument
amplitude `harm_amp_<ins>` scales it (→ chromatic strength). The design
column for point `i`, harmonic `k` is `A_{ins(i)}·cos(2πk t_i/P)` (and
the sin counterpart); marginalizing `(a_k,b_k)` integrates the phase out.

Parameters: shared `harm_period` (P_rot) + per-instrument `harm_amp_<ins>`.
`nharm` (fundamental + harmonics) is fixed at construction. The global
`σ_amp × A_m` scale degeneracy is fixed by folding `σ_amp≡1` into `A_m`.

This is the coherent-activity hypothesis (Boisse-style; structurally the
DAEDALUS Fourier object) — put it in a `noise_exclusion_groups` entry with
`CeleriteRotation`, and the trans-dim choice between them MEASURES whether
the star's activity is phase-coherent over the baseline or phase-wandering.

Stage 3b (`AdditiveCovariance`).
"""
struct HarmonicBlock <: AdditiveCovariance
    channel::Symbol
    instruments::Vector{String}
    nharm::Int
    # Explicit frequencies in cycles per day. EMPTY (the default) keeps the
    # rotation-harmonic behaviour above: `nharm` harmonics of a fitted
    # `harm_period`. NON-EMPTY switches the block to a fixed external comb and
    # `harm_period` is then neither used nor allocated.
    #
    # The external case is for a host whose variability is a known set of
    # coherent oscillation modes -- a gamma Dor or delta Sct pulsator -- whose
    # frequencies have been measured elsewhere over a long baseline. Those are
    # determined far better (~1e-5 /d over years of photometry) than a transit
    # fit could ever constrain them, so they are fixed; the amplitudes and
    # phases stay marginalised, which is the point. Pre-subtracting a fitted
    # mode model instead would treat it as exact and understate the transit
    # depth uncertainty.
    freqs::Vector{Float64}
end

HarmonicBlock(; channel::Symbol = :rv, instruments::Vector{String} = String[],
                nharm::Int = 3, freqs::Vector{Float64} = Float64[]) =
    HarmonicBlock(channel, instruments,
                  isempty(freqs) ? nharm : length(freqs), freqs)

"""
    StudentT(; channel=:rv)

Heavy-tailed Student-t measurement likelihood — the outlier-class
representative. Each RV residual is scored as `r_i ~ t_ν(0, σ_i²)` (an
independent Student-t with `ν` degrees of freedom and per-point scale
`σ_i²` = the usual formal + jitter/error-scale variance) instead of a
Gaussian. One parameter `studentt_nu` (`ν → ∞` recovers the Gaussian;
small `ν` fattens the tails to absorb outliers without a hard clip).

⚠ WHITE-BRANCH ONLY, enforced hard. A Student-t likelihood is not a
covariance — it destroys the Gaussian (semiseparable) marginalization
that every GP / AdditiveCovariance path relies on, and the scale-mixture
augmentation that would restore it adds N latent scales (rejected for a
trans-dim sampler). So StudentT must NEVER combine with a
CovarianceNoise / AdditiveCovariance / SequentialNoise: construction
rejects it (non-trans-dim), and any such state scores −Inf at eval
(trans-dim — put them in a `noise_exclusion_groups` entry). It is the
third mutually-exclusive white treatment alongside jitter and
`ErrorScale`.

A direct `NoiseModel` (not one of the covariance stages): it changes the
likelihood family, orthogonal to the covariance classes.
"""
struct StudentT <: NoiseModel
    channel::Symbol
end

StudentT(; channel::Symbol = :rv) = StudentT(channel)

# Channel-coefficient param naming. logR'HK has no derivative term.
const _ACTIVITY_GP_COEFFS = Dict(
    :rv     => (:Vc, :Vr),
    :bis    => (:Bc, :Br),
    :fwhm   => (:Fc, :Fr),
    :logrhk => (:Lc, nothing),
    :halpha => (:Hc, :Hr),
)

"""Channel of a noise model — `:rv` or `:phot`. Defaults to `:rv` for
all subtypes; only `CovarianceNoise` currently honors per-channel
routing in the likelihood (see `transit_log_likelihood`)."""
noise_channel(nm::NoiseModel) = hasfield(typeof(nm), :channel) ? getfield(nm, :channel) : :rv

"""Instrument-name list a `CovarianceNoise` GP is restricted to.
Empty (default) means the GP is global on its channel."""
noise_instruments(nm::NoiseModel) =
    hasfield(typeof(nm), :instruments) ? getfield(nm, :instruments) : String[]

# noise_param_names and validate_noise_models are in noise/param_names.jl
# (loaded after model.jl to access InstrumentConfig)
