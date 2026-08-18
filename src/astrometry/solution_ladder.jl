# The astrometric solution ladder: 5-, 7- and 9-parameter models, and their
# EXACT marginal likelihoods.
#
# Gaia's non-single-star pipeline picks a solution type with a cascade of
# frequentist cuts (DR3: trigger on RUWE > 1.4, try 9p first, accept on
# s9 > 12 and F2 < 25, else fall back to 7p, else orbital) and publishes the
# winner as fact. Three consequences bite end users: a source on a boundary is
# reported as one type with no uncertainty propagated; the thresholds are tuned
# to control catalogue-level false positives, not to give per-source
# probabilities; and "the 7p sample" is a population defined by a cut, so any
# inference on it inherits the cut.
#
# The models differ only in how many linear terms of the along-scan abscissa
# model are free:
#
#   w_i = da*·sinψ + dd·cosψ + plx·p_i                      (5p, + proper motion)
#         + mu_a*·sinψ·dt + mu_d·cosψ·dt
#         + acc_a*·sinψ·dt²/2 + acc_d·cosψ·dt²/2            (7p)
#         + jrk_a*·sinψ·dt³/6 + jrk_d·cosψ·dt³/6            (9p)
#
# THE POINT OF THIS FILE. Every one of these is linear-Gaussian in ALL of its
# parameters, so its marginal likelihood is available in CLOSED FORM — no
# sampling, no thermodynamic integration, no estimator error bar. Only a model
# with a Keplerian in it needs MCMC. So a per-source posterior over the ladder
# costs three matrix factorisations, which is what makes it affordable at
# catalogue scale rather than a demonstration on a handful of stars.
#
# THE THING THAT MUST NOT BE GOT WRONG. `iad_log_likelihood` marginalises the
# catalogue correction under FLAT priors. That is fine for orbit fitting, where
# the resulting constant cancels. It is fatal for model COMPARISON, because the
# constant no longer cancels between models with different numbers of columns —
# improper priors give undefined Bayes factors (Lindley's paradox, arriving on
# schedule). So the acceleration and jerk coefficients need PROPER, normalised,
# physically motivated priors, and the prior width is a real modelling decision
# that changes the answer. `default_ladder_prior` derives it from a companion's
# reflex rather than pulling a number out of the air.

using LinearAlgebra: cholesky, Symmetric, logdet, issuccess, ldiv!

export astrom_design, astrom_logZ, astrom_chi2_reduced, astrom_solution,
       ladder_probabilities, default_ladder_prior, ASTROM_PARAM_NAMES

"""
    astrom_design(iad, order) -> Matrix{Float64}

Along-scan design matrix, `n_transit × order`, for `order ∈ (5, 7, 9)`.

Columns follow the abscissa model in the file header. `pm_factor` carries the
time baseline from the catalogue reference epoch in Julian years, so the
acceleration and jerk columns are its square and cube with the 1/2 and 1/6 from
the Taylor expansion folded in — that scaling is what makes the coefficients
mas/yr² and mas/yr³ rather than arbitrary.
"""
function astrom_design(iad::IADData, order::Int)
    order in (5, 7, 9) ||
        throw(ArgumentError("astrometric solution order must be 5, 7 or 9 " *
                            "(got $order); 2p and 6p are catalogue products " *
                            "that do not arise here"))
    n = n_iad(iad)
    X = Matrix{Float64}(undef, n, order)
    @inbounds for j in 1:n
        s, c = sincos(iad.psi[j])
        dt   = iad.pm_factor[j]
        X[j, 1] = s
        X[j, 2] = c
        X[j, 3] = iad.parallax_factor[j]
        X[j, 4] = s * dt
        X[j, 5] = c * dt
        if order >= 7
            X[j, 6] = s * dt^2 / 2
            X[j, 7] = c * dt^2 / 2
        end
        if order >= 9
            X[j, 8] = s * dt^3 / 6
            X[j, 9] = c * dt^3 / 6
        end
    end
    return X
end

"""
    default_ladder_prior(iad, order; a0_max = 5.0, accel_yr = 10.0) -> Vector

Prior standard deviations for the `order` linear coefficients, in the units each
carries. NOT a shrug: the Bayes factors between ladder rungs depend on these,
so they have to mean something.

- position (mas) and parallax (mas): `a0_max`, the largest reflex amplitude
  considered plausible for the source. Loose — these are nuisance offsets the
  catalogue already constrains.
- proper motion (mas/yr): `a0_max` per year of baseline is the scale at which a
  linear drift is indistinguishable from a mis-set catalogue PM.
- acceleration (mas/yr²) and jerk (mas/yr³): derived from a companion whose
  reflex would sweep `a0_max` over `accel_yr` years. A companion with period
  much longer than the mission produces exactly a slow curvature, so the prior
  width is "how curved could a real long-period companion make this", which is
  the hypothesis being tested.

Widening these makes the higher rungs harder to prefer (an Occam penalty), so
this is the single most consequential choice in the ladder comparison and
belongs in any sensitivity analysis rather than a footnote.
"""
function default_ladder_prior(iad::IADData, order::Int;
                              a0_max::Real = 5.0, accel_yr::Real = 10.0)
    σ = Float64[a0_max, a0_max, a0_max, a0_max, a0_max]
    if order >= 7
        push!(σ, 2 * a0_max / accel_yr^2)
        push!(σ, 2 * a0_max / accel_yr^2)
    end
    if order >= 9
        push!(σ, 6 * a0_max / accel_yr^3)
        push!(σ, 6 * a0_max / accel_yr^3)
    end
    return σ
end

"""
    astrom_logZ(iad, order; prior_sigma, residual) -> log Z

EXACT log marginal likelihood of the `order`-parameter astrometric solution.
No sampling. No estimator error.

For `w = Xq + ε`, `ε ~ N(0, W⁻¹)` with `W = diag(1/σᵢ²)`, and a proper prior
`q ~ N(0, S)` with `S = diag(prior_sigma²)`:

    log Z = -n/2·log(2π) + 1/2·log|W|
            - 1/2·(wᵀWw - vᵀM⁻¹v)
            - 1/2·log|S| - 1/2·log|M|          with M = XᵀWX + S⁻¹, v = XᵀWw

The `log|S|` and `log|M|` terms are the Occam factor, and they are exactly what
a flat prior throws away — which is why `iad_log_likelihood`'s marginalisation,
correct for orbit fitting, cannot be used to compare rungs.

`residual` optionally replaces the abscissae (e.g. after subtracting a
Keplerian), so the same routine serves the "is the leftover curvature
Keplerian or generic" comparison.
"""
function astrom_logZ(iad::IADData, order::Int;
                     prior_sigma::AbstractVector{<:Real} =
                         default_ladder_prior(iad, order),
                     residual::Union{Nothing,AbstractVector{<:Real}} = nothing)
    n = n_iad(iad)
    n > order || throw(ArgumentError(
        "need more transits ($n) than free parameters ($order)"))
    length(prior_sigma) == order || throw(ArgumentError(
        "prior_sigma has $(length(prior_sigma)) entries for order $order"))
    all(>(0), prior_sigma) || throw(ArgumentError("prior_sigma must be positive"))

    X = astrom_design(iad, order)
    y = residual === nothing ? iad.abscissa : Float64.(residual)
    length(y) == n || throw(ArgumentError("residual length $(length(y)) ≠ $n"))
    w = @. 1 / (iad.abscissa_err^2)

    yWy = 0.0
    @inbounds for j in 1:n; yWy += w[j] * y[j]^2; end
    Xw = X .* w                       # n × order, rows weighted
    A  = X' * Xw                      # XᵀWX
    v  = Xw' * y                      # XᵀWy

    S⁻¹ = 1 ./ (prior_sigma .^ 2)
    M   = A + Diagonal(S⁻¹)
    ch  = cholesky(Symmetric(M); check = false)
    issuccess(ch) || return -Inf      # degenerate design (missing plx/pm factors)

    quad     = dot(v, ch \ v)
    log_detW = -2 * sum(log, iad.abscissa_err)
    log_detS = 2 * sum(log, prior_sigma)

    return -0.5 * n * log(2π) + 0.5 * log_detW -
            0.5 * (yWy - quad) - 0.5 * log_detS - 0.5 * logdet(ch)
end

"""
    astrom_chi2_reduced(iad, order; prior_sigma, residual) -> chi2 / dof

Goodness of fit of the MAP solution at this rung.

REQUIRED alongside `ladder_probabilities`, which normalises over the rungs you
supplied and therefore CANNOT say whether any of them is adequate. On the real
DR4 pre-release, Gaia BH3 reports P(9p) = 1.000 with log Z going from -1.8e6 to
-3.3e5 — the ladder faithfully picking the least hopeless of three hopeless
models, because a 33 M_sun companion produces an orbit no polynomial in time
describes. Without this number that source looks like a confident 9p detection.

chi2/dof near 1 means the rung describes the data. Much greater than 1 means
none of the ladder does and the source needs an orbital solution.
"""
function astrom_chi2_reduced(iad::IADData, order::Int;
        prior_sigma::AbstractVector{<:Real} = default_ladder_prior(iad, order),
        residual::Union{Nothing,AbstractVector{<:Real}} = nothing)
    n = n_iad(iad)
    X = astrom_design(iad, order)
    y = residual === nothing ? iad.abscissa : Float64.(residual)
    w = @. 1 / (iad.abscissa_err^2)
    A = X' * (X .* w); v = (X .* w)' * y
    M = A + Diagonal(1 ./ (prior_sigma .^ 2))
    ch = cholesky(Symmetric(M); check = false)
    issuccess(ch) || return Inf
    q = ch \ v
    r = y .- X * q
    return sum(w .* r .^ 2) / max(n - order, 1)
end


"""
    astrom_solution(iad, order; prior_sigma, residual) -> NamedTuple

The COMPLETE posterior of an `order`-parameter astrometric solution, in closed
form. Not a fit — a solve.

These models are linear-Gaussian, so the posterior is exactly a multivariate
Gaussian and everything about it is available from one factorisation:

    q   = M⁻¹v            posterior mean = MAP = the global maximum
    cov = M⁻¹             full covariance, correlations included
    with M = XᵀWX + S⁻¹,  v = XᵀWy

There is no optimiser, no starting guess, no convergence to check, and no
question of local versus global optima: a linear-Gaussian posterior is unimodal
by construction. The multimodality that makes astrometry hard (period aliases,
the i/Omega reflection) lives entirely in the orbital rung, which is exactly the
rung that is not linear and does need sampling.

Returns `(names, q, sigma, cov, chi2_reduced, log_z, order)`. Parameter order
matches `astrom_design`: offsets, parallax, proper motion, then acceleration
(7p) and jerk (9p).

Units: offsets and parallax mas; proper motion mas/yr; acceleration mas/yr²;
jerk mas/yr³ — set by the dt scaling of the design columns.
"""
function astrom_solution(iad::IADData, order::Int;
        prior_sigma::AbstractVector{<:Real} = default_ladder_prior(iad, order),
        residual::Union{Nothing,AbstractVector{<:Real}} = nothing)
    n = n_iad(iad)
    n > order || throw(ArgumentError(
        "need more transits ($n) than free parameters ($order)"))
    length(prior_sigma) == order || throw(ArgumentError(
        "prior_sigma has $(length(prior_sigma)) entries for order $order"))

    X = astrom_design(iad, order)
    y = residual === nothing ? iad.abscissa : Float64.(residual)
    w = @. 1 / (iad.abscissa_err^2)
    Xw = X .* w
    M  = X' * Xw + Diagonal(1 ./ (prior_sigma .^ 2))
    v  = Xw' * y
    ch = cholesky(Symmetric(M); check = false)
    issuccess(ch) || throw(ArgumentError(
        "design is rank-deficient at order $order — parallax_factor and " *
        "pm_factor must be supplied, not left at their zero defaults"))

    q   = ch \ v
    cov = inv(ch)
    r   = y .- X * q
    return (names = ASTROM_PARAM_NAMES[1:order], q = q,
            sigma = sqrt.(diag(cov)), cov = cov,
            chi2_reduced = sum(w .* r .^ 2) / max(n - order, 1),
            log_z = astrom_logZ(iad, order; prior_sigma = prior_sigma,
                                residual = residual),
            order = order)
end

"Parameter names in design-column order."
const ASTROM_PARAM_NAMES = ["dra_cosdec", "ddec", "parallax",
                            "pmra_cosdec", "pmdec",
                            "accel_ra", "accel_dec",
                            "jerk_ra", "jerk_dec"]

"""
    ladder_probabilities(iad; orders, prior, model_prior, residual) -> NamedTuple

Posterior probability of each rung of the solution ladder, from exact
evidences. This is the quantity Gaia's cascade cannot report: a per-source
P(model | data) instead of a hard type assignment with no uncertainty.

Returns `(orders, log_z, prob, best, chi2_reduced, adequate)`.

`prob` is normalised over the rungs supplied, so it answers "which of THESE" and
says NOTHING about whether any of them fits. `chi2_reduced` and `adequate` are
the guard: `adequate = false` means every rung is a bad fit and the source needs
an orbital solution, however confident `prob` looks. Read them together or the
ladder will hand you confident nonsense on exactly the sources that matter most.
"""
function ladder_probabilities(iad::IADData;
        orders = (5, 7, 9),
        prior = nothing,
        model_prior = nothing,
        residual::Union{Nothing,AbstractVector{<:Real}} = nothing)
    ords = collect(orders)
    lz = Float64[]
    for o in ords
        ps = prior === nothing ? default_ladder_prior(iad, o) : prior(o)
        push!(lz, astrom_logZ(iad, o; prior_sigma = ps, residual = residual))
    end
    lp = model_prior === nothing ? zeros(length(ords)) :
         [log(model_prior(o)) for o in ords]
    tot = lz .+ lp
    m = maximum(tot)
    p = exp.(tot .- m); p ./= sum(p)
    best = ords[argmax(p)]
    c2 = [astrom_chi2_reduced(iad, o;
              prior_sigma = prior === nothing ? default_ladder_prior(iad, o) : prior(o),
              residual = residual) for o in ords]
    return (orders = ords, log_z = lz, prob = p, best = best,
            chi2_reduced = c2, adequate = minimum(c2) < 2.0)
end
