# Doppler tomography: the planet's shadow in the stellar line profile.
#
# A rotating star's absorption line is broadened because one limb approaches and
# the other recedes, so each velocity across the profile maps to a strip on the
# visible disc. A transiting planet hides one strip, removing its contribution and
# leaving a bump in the profile that travels across it during transit. The
# trajectory of that bump encodes λ and b directly.
#
# Why this exists in Nereus: for a rapidly rotating star the RM in RADIAL
# VELOCITY collapses the whole profile into one number and is easily swamped —
# on a bright A9V γ Dor host the pulsations produce RV signals comparable
# to the RM itself, and per-night RM fits returned confident, mutually exclusive
# obliquities. The line profile keeps the spatial information the RV throws away.
#
# TWO NORMALISATION TRAPS, both of which produced confident wrong answers before
# being caught by injection tests. They are the reason this module normalises the
# way it does, and the reason `tomogram_injection_test` exists at all:
#
#   * dividing each profile by its PEAK converts a change in line DEPTH (which
#     pulsations cause) into a change in line SHAPE, pinned at a fixed velocity.
#     It manufactured a −9σ feature that no injected λ could displace.
#   * dividing by the EQUIVALENT WIDTH removes the signal: a Doppler shadow IS a
#     localised loss of equivalent width. A 50× injection returned zero.
#
# Correct treatment: subtract a linear baseline fitted to the far wings, and do
# not renormalise. Injected λ then recovers to within ~1°.
#
# Conventions follow rm.jl, so λ means the same thing here, in the RM, and in the
# gravity-darkened transit model.

using Statistics
using Random: MersenneTwister, randperm, randn, rand
using LinearAlgebra: Symmetric, eigen, dot, BLAS, BlasInt
using FFTW: fft, ifft, fftshift, ifftshift, fftfreq
using AffineInvariantMCMC

"""
    ccf_profile(λ_obs, flux, mask_λ, mask_w, vgrid; berv = 0.0) -> Vector

Cross-correlate one spectrum against a line mask, returning the mean line profile
sampled on `vgrid` [km/s]. `mask_λ`/`mask_w` are line positions and weights (depths).

The mask MUST match the star: correlating a solar (G2) mask against an A-type
star produces a profile the fit cannot describe — on one A-type target the HARPS pipeline
returned CCF widths of 49–331 km/s with 0 of 30 exposures plausible.

The mask must also be DE-BLENDED. At vsini ≈ 25 km/s the lines are ~40 km/s wide,
so a dense mask piles neighbouring lines on top of each other and the profile
becomes a pedestal rather than a line. Require isolation of ≳2× the rotational
width when building the mask.

`λ_obs` and `mask_λ` must be in the SAME frame — air or vacuum. Mixing them costs
c·(n−1) ≈ 83 km/s at 5500 Å, which is large enough to look like a real velocity.
"""
function ccf_profile(λ_obs::AbstractVector, flux::AbstractVector,
                     mask_λ::AbstractVector, mask_w::AbstractVector,
                     vgrid::AbstractVector; berv::Real = 0.0)
    c = 299_792.458
    ccf  = zeros(Float64, length(vgrid))
    wsum = zeros(Float64, length(vgrid))
    lo, hi = first(λ_obs), last(λ_obs)
    @inbounds for k in eachindex(vgrid)
        v = vgrid[k] + berv
        s = w = 0.0
        for i in eachindex(mask_λ)
            λ = mask_λ[i] * (1 + v / c)
            (λ <= lo || λ >= hi) && continue
            j = searchsortedfirst(λ_obs, λ)
            (j <= 1 || j > length(λ_obs)) && continue
            t = (λ - λ_obs[j-1]) / (λ_obs[j] - λ_obs[j-1])
            f = flux[j-1] * (1 - t) + flux[j] * t
            isfinite(f) || continue
            s += mask_w[i] * (1 - f)          # flux deficit = absorption
            w += mask_w[i]
        end
        ccf[k]  = s
        wsum[k] = w
    end
    return @. ifelse(wsum > 0, ccf / max(wsum, 1e-12), 0.0)
end

"""
    tomogram_residuals(profiles, vgrid, in_transit; vsys = 0.0, bervs = nothing)
        -> (grid, R)

Build the residual map: each profile minus the mean OUT-OF-TRANSIT profile, with
every profile shifted to the stellar rest frame. `profiles` is (n_exposure × n_v).

Baseline removal ONLY — a linear fit to the far wings (|v − vsys| > 45 km/s) is
subtracted and nothing is renormalised. See the header for why peak- and
EW-normalisation both fail.
"""
function tomogram_residuals(profiles::AbstractMatrix, vgrid::AbstractVector,
                            in_transit::AbstractVector{Bool};
                            vsys::Real = 0.0, bervs = nothing,
                            norm_depth::Bool = true,
                            grid = range(-60, 60; length = 241))
    n = size(profiles, 1)
    g = collect(grid)
    X = Matrix{Float64}(undef, n, length(g))
    for i in 1:n
        v = collect(vgrid)
        y = collect(@view profiles[i, :])
        ok = isfinite.(y)
        vv, yy = v[ok], y[ok]
        wing = abs.(vv .- vsys) .> 45
        if count(wing) > 10                       # linear continuum from the wings
            A = hcat(ones(count(wing)), vv[wing])
            cf = A \ yy[wing]
            yy = yy .- (cf[1] .+ cf[2] .* vv)
        end
        shift = bervs === nothing ? vsys : (vsys - bervs[i])
        X[i, :] = [_interp1(vv .- shift, yy, x) for x in g]
    end
    oot = .!in_transit
    μ = vec(mean(view(X, oot, :); dims = 1))
    R = X .- μ'
    # Scale to unit line depth. Without this, pooling weights nights by however
    # deep their CCF happens to be — a DRS CCF and a mask CCF built by
    # accumulating line depth differ by a large factor — so the deepest night
    # dominates the sum for a reason that has nothing to do with its S/N.
    if norm_depth
        dep = maximum(μ)
        R ./= max(dep, 1e-9)
    end
    return g, R
end

"""
    tomogram_pulsation_filter(R, hours; vsini, T14, k = 1.6) -> (Rf, frac_removed)

Remove non-radial pulsation power from a residual map by 2-D Fourier slope
separation (Johnson et al. 2015, WASP-33).

A feature drifting across the line profile at dv/dt = s puts its power along
kt = -s·kv, so pulsations and the planet trail separate by SLOPE even when their
temporal frequencies are unresolved within a single night — and they usually are:
a 5.6 h night resolves 4.3 c/d while γ Dor modes sit near 1–3 c/d. That is why
this is done in the 2-D domain rather than by pre-whitening at known frequencies,
and why an external frequency comb (`HarmonicBlock(freqs = ...)`) is the wrong
tool for it.

The keep-band is set by the PLANET, not by the pulsations: `2·vsini/T14` is the
fastest a shadow can cross the profile (the λ ≈ 0 case, a full ±vsini sweep), and
the cut sits at `k` times that. So the shadow is inside the band at every λ by
construction and the filter cannot preferentially suppress the aligned hypothesis
and manufacture a polar answer. Verify it on your own data with
`tomogram_injection_test` run THROUGH the filter — if filtering ate the signal,
injected λ stops coming back.

`hours` is the time of each exposure relative to mid-transit, in hours.
"""
function tomogram_pulsation_filter(R::AbstractMatrix, hours::AbstractVector,
                                   grid::AbstractVector;
                                   vsini::Real, T14::Real, k::Real = 1.6)
    nt, nv = size(R)
    (nt < 4 || nv < 4) && return copy(R), 0.0
    dt = median(diff(sort(collect(hours))))
    dv = median(diff(collect(grid)))
    (isfinite(dt) && dt > 0 && isfinite(dv) && dv > 0) || return copy(R), 0.0
    return _slope_filter(R, dt, dv, k * 2 * vsini / T14)
end

function _slope_filter(R::AbstractMatrix, dt::Real, dv::Real, scut::Real)
    nt, nv = size(R)
    F  = fftshift(fft(R))
    kt = fftshift(fftfreq(nt, 1 / dt))
    kv = fftshift(fftfreq(nv, 1 / dv))
    p0 = sum(abs2, F)
    @inbounds for i in 1:nt, j in 1:nv
        slope = abs(kv[j]) > 1e-9 ? -kt[i] / kv[j] : Inf
        w = isinf(slope) ? 1.0 : clamp(scut / max(abs(slope), 1e-9), 0.0, 1.0)
        F[i, j] *= 0.5 * (1 - cos(π * w))
    end
    removed = p0 > 0 ? 1 - sum(abs2, F) / p0 : 0.0
    return real(ifft(ifftshift(F))), removed
end

function _interp1(x, y, xq)
    (xq <= first(x) || xq >= last(x)) && return 0.0
    j = searchsortedfirst(x, xq)
    (j <= 1 || j > length(x)) && return 0.0
    t = (xq - x[j-1]) / (x[j] - x[j-1])
    return y[j-1] * (1 - t) + y[j] * t
end

"""
    shadow_track(t, Tc, P, a_Rs, inc, λ, vsini) -> Vector

Sub-planet velocity as a function of time: where the shadow sits in the line
profile. NaN outside transit. Uses the same geometry as `planet_sky_position`.

λ ≈ 0 sweeps the profile from −vsini to +vsini; λ ≈ ±90° sits at ∓b·vsini and
barely moves, which is why near-polar geometries are the hardest to detect and
why pooling multiple transits matters.
"""
function shadow_track(t::AbstractVector, Tc::Real, P::Real, a_Rs::Real,
                      inc::Real, λ::Real, vsini::Real)
    out = fill(NaN, length(t))
    sλ, cλ = sincos(λ)
    @inbounds for i in eachindex(t)
        ph = 2π * (t[i] - Tc) / P
        x  = a_Rs * sin(ph)
        y  = a_Rs * cos(ph) * cos(inc)
        z  = a_Rs * cos(ph) * sin(inc)
        (hypot(x, y) < 1 && z > 0) || continue
        out[i] = vsini * (x * cλ - y * sλ)
    end
    return out
end

"""
    tomogram_matched_filter(R, grid, t, Tc, P, a_Rs, inc, vsini; λs, weight)
        -> (λ_grid, score)

Matched filter of the residual map against the predicted shadow track, over a grid
of λ. `weight` is an inverse-variance weight (use 1/var of the out-of-transit
residuals); pass one per night when pooling.

POOLING. The shadow track depends only on the geometry and is therefore identical
on every transit, while the pulsation pattern depends on pulsation phase and
differs between nights. Stacking adds the planet coherently and averages the
pulsations down. On a γ Dor host a single night gave nothing; three pooled
transits gave a clear detection.

SIGNIFICANCE. Do NOT use the formal error on this statistic. The residuals are
dominated by coherent pulsations, so a Gaussian likelihood with independent points
overstates S/N by an order of magnitude (it returned 21σ on data whose real
significance was marginal). Calibrate with `tomogram_null_distribution` instead.
"""
function tomogram_matched_filter(R::AbstractMatrix, grid::AbstractVector,
                                 t::AbstractVector, Tc::Real, P::Real,
                                 a_Rs::Real, inc::Real, vsini::Real;
                                 λs = range(-π, π; length = 181),
                                 weight::Real = 1.0, σ_line::Real = vsini / 2.5)
    λv = collect(λs)
    s  = zeros(Float64, length(λv))
    for (k, λ) in enumerate(λv)
        tr = shadow_track(t, Tc, P, a_Rs, inc, λ, vsini)
        num = den = 0.0
        @inbounds for i in eachindex(t)
            isfinite(tr[i]) || continue
            for j in eachindex(grid)
                m = -exp(-0.5 * ((grid[j] - tr[i]) / σ_line)^2)
                num += weight * m * R[i, j]
                den += weight * m * m
            end
        end
        s[k] = den > 0 ? num / sqrt(den) : 0.0
    end
    return λv, s
end

# Matched-filter numerator and denominator, kept separate so several nights can
# be pooled COHERENTLY: Σnum / sqrt(Σden), not Σ(num/sqrt(den)). Summing
# per-night normalised statistics weights every night equally regardless of how
# many in-transit exposures it has or how well it constrains the track.
function _mf_numden(R::AbstractMatrix, grid::AbstractVector, t::AbstractVector,
                    Tc::Real, P::Real, a_Rs::Real, inc::Real, vsini::Real,
                    λv::AbstractVector; σ_line::Real = vsini / 2.5)
    num = zeros(Float64, length(λv))
    den = zeros(Float64, length(λv))
    for (k, λ) in enumerate(λv)
        tr = shadow_track(t, Tc, P, a_Rs, inc, λ, vsini)
        n = d = 0.0
        @inbounds for i in eachindex(t)
            isfinite(tr[i]) || continue
            for j in eachindex(grid)
                m = -exp(-0.5 * ((grid[j] - tr[i]) / σ_line)^2)
                n += m * R[i, j]
                d += m * m
            end
        end
        num[k] = n; den[k] = d
    end
    return num, den
end

"""
    tomogram_null_distribution(R, grid, t, ...; n = 400, rng) -> Vector

Empirical null for the matched-filter statistic, by permuting the exposures in
time. This destroys the shadow's coherent track while preserving the pulsation
amplitude distribution and the correlated structure that makes formal errors
meaningless. Compare the observed peak against this, not against sqrt(chi2).

A second, stronger control is to re-run the filter with `Tc` displaced by a few
hours so the assumed transit falls on out-of-transit baseline: same data, same
pulsations, no planet.
"""
function tomogram_null_distribution(R::AbstractMatrix, grid::AbstractVector,
                                    t::AbstractVector, Tc::Real, P::Real,
                                    a_Rs::Real, inc::Real, vsini::Real;
                                    n::Int = 400, rng = Random.default_rng(),
                                    kwargs...)
    out = Vector{Float64}(undef, n)
    idx = collect(1:size(R, 1))
    for k in 1:n
        Random.shuffle!(rng, idx)
        _, s = tomogram_matched_filter(view(R, idx, :), grid, t, Tc, P,
                                       a_Rs, inc, vsini; kwargs...)
        out[k] = maximum(s)
    end
    return out
end

"""
    tomogram_injection_test(profiles, vgrid, t, in_transit, Tc, P, a_Rs, inc,
                            vsini, rr; λ_true, u1, u2) -> (λ_recovered, score)

Inject a synthetic shadow of the physically correct amplitude at known `λ_true`,
push it through the identical pipeline, and report what comes back.

THIS IS A GATE, NOT A DIAGNOSTIC. Run it before believing any measurement. Both
normalisation bugs described in the header passed every internal sanity check and
were caught only here — in one of them, injecting λ = 0°, −37° and −90° all
returned the same answer, which is the signature of an analysis reporting its own
artefact rather than the data.

The injected shadow removes a fraction (Rp/R★)²·I(μ)/⟨I⟩ of the stellar flux at
the sub-planet velocity, i.e. the same area the transit removes from the light
curve.
"""
function tomogram_injection_test(profiles::AbstractMatrix, vgrid::AbstractVector,
                                 t::AbstractVector, in_transit::AbstractVector{Bool},
                                 Tc::Real, P::Real, a_Rs::Real, inc::Real,
                                 vsini::Real, rr::Real;
                                 λ_true::Real = 0.0, u1::Real = 0.3, u2::Real = 0.3,
                                 vsys::Real = 0.0, bervs = nothing, kwargs...)
    σ_line = vsini / 2.5
    tr = shadow_track(t, Tc, P, a_Rs, inc, λ_true, vsini)
    inj = copy(profiles)
    g0, R0 = tomogram_residuals(profiles, vgrid, in_transit; vsys, bervs)
    ew = sum(vec(mean(view(R0, .!in_transit, :); dims = 1)))    # scale reference
    @inbounds for i in eachindex(t)
        isfinite(tr[i]) || continue
        ph = 2π * (t[i] - Tc) / P
        x  = a_Rs * sin(ph); y = a_Rs * cos(ph) * cos(inc)
        μ  = sqrt(max(1 - (x^2 + y^2), 0.0))
        I  = (1 - u1 * (1 - μ) - u2 * (1 - μ)^2) / (1 - u1/3 - u2/6)
        A  = rr^2 * I
        for j in eachindex(vgrid)
            inj[i, j] -= A * exp(-0.5 * ((vgrid[j] - vsys - tr[i]) / σ_line)^2)
        end
    end
    g, Ri = tomogram_residuals(inj, vgrid, in_transit; vsys, bervs)
    λv, s = tomogram_matched_filter(Ri .- R0, g, t, Tc, P, a_Rs, inc, vsini; kwargs...)
    return λv[argmax(s)], maximum(s)
end


"""
    tomogram_pooled(nights; λs, vsini, P, a_Rs, inc, n_null, rng) -> (λ, stat, p, λgrid, s_tot)

Pooled shadow detection across several transit nights.

This is the entry point that matters. A single night of an A star does not have
the signal: on a γ Dor host three separate nights peaked tens of degrees
apart, none of them meaningful. Pooling works because the planet's track
depends only on the geometry and is therefore the SAME angle every night, while
the pulsations that dominate each night's residuals are at a different phase on
each — so the shadow adds coherently and the noise does not.

`nights` is a vector of NamedTuples with fields
`(profiles, vgrid, times, bervs, Tc)`. Each night keeps its own transit centre;
the matched filter is evaluated on a common λ grid and the statistics summed.

TWO THINGS THAT WILL SILENTLY RUIN THIS, both learned the hard way:

  * `bervs` is not optional in practice. Nights separated by weeks differ in
    barycentric velocity by tens of km/s — 13.4 km/s between two nights of one
    such campaign, half the v sin i — so without it the tracks are offset from
    each other and the pooled peak is meaningless.
  * every night's profiles must have the SAME SIGN. A DRS CCF is a dip
    (`ccf/continuum`) while a mask-CCF built by accumulating line depth peaks;
    mixing the two makes one night subtract from the others. Convert dips with
    `(continuum - ccf)/continuum` before pooling.

Significance is a permutation null, not a formal error: see
`tomogram_null_distribution`.

LEGACY STATISTIC — prefer `tomogram_bayes` + `shadow_bayes_factor`. This is a
matched filter with a permutation null, so it reports a p-value; the Bayesian
path reports a Savage-Dickey Bayes factor on the shadow amplitude against
alpha = 0, which is what a Bayesian analysis should quote and which has the
right sanity property built in (an uninformative posterior returns logBF = 0
exactly). Still load-bearing: `fit_tomography`, the run_job "tomography" kind
and the six `tomogram.*` feature ops all route through here, so retiring it is
a MIGRATION of the public API onto tomogram_bayes, not a deletion.
"""
function tomogram_pooled(nights;
                         P::Real, a_Rs::Real, inc::Real, vsini::Real,
                         vsys::Real = 0.0, T14::Real, filter_pulsations::Bool = true,
                         λs = range(-π, π; length = 721),
                         n_null::Int = 300, rng = Random.default_rng())
    λv = collect(λs)
    Rs = Any[]; meta = Any[]; gref = nothing
    for nt in nights
        in_transit = @. abs(nt.times - nt.Tc) < T14 / 2
        g, R = tomogram_residuals(nt.profiles, nt.vgrid, in_transit;
                                  vsys = vsys, bervs = nt.bervs)
        if filter_pulsations
            hours = (nt.times .- nt.Tc) .* 24
            R, _ = tomogram_pulsation_filter(R, hours, g; vsini = vsini,
                                             T14 = T14 * 24)
        end
        gref = g
        push!(Rs, R); push!(meta, (nt.times, nt.Tc))
    end

    function pooled_score(maps)
        num = zeros(Float64, length(λv)); den = zeros(Float64, length(λv))
        for (R, (t, Tc)) in zip(maps, meta)
            n, d = _mf_numden(R, gref, t, Tc, P, a_Rs, inc, vsini, λv)
            num .+= n; den .+= d
        end
        # RETURNED RAW. An earlier version divided by `std(s)` to express the
        # peak in units of the scan's own spread, and that silently destroyed
        # the permutation test below.
        #
        # `std(s)` is not a noise scale. The shadow is broad in lambda -- angles
        # near the truth match the track partially -- so a real signal lifts a
        # wide region of the scan and raises its dispersion along with its peak.
        # Dividing one by the other therefore cancels the signal: on an injected
        # shadow at S/N 91, permuting the frames dropped the raw peak 2.53x and
        # the divisor 2.50x, leaving 3.027 against a null of 2.993. The p-value
        # was pinned near 0.5 at every signal strength ever tested.
        #
        # `den` is already the right normaliser and is what remains: it is the
        # template's own norm, built from the transit geometry alone with no
        # data in it, so it does not move when the frames are shuffled and the
        # comparison against the null means what it says.
        return @. ifelse(den > 0, num / sqrt(max(den, 1e-30)), 0.0)
    end

    tot = pooled_score(Rs)
    # Signed, NOT abs. The shadow is a flux DEFICIT and the template is negative,
    # so a real shadow drives the statistic positive. Taking abs() would let a
    # strongly anti-correlated λ — an emission-like feature, or the mirrored
    # track — win the argmax and be reported as a detection.
    i = argmax(tot)
    nullmax = Float64[]
    for _ in 1:n_null
        push!(nullmax, maximum(pooled_score([R[randperm(rng, size(R, 1)), :]
                                             for R in Rs])))
    end
    pval = isempty(nullmax) ? NaN : mean(nullmax .>= tot[i])
    return λv[i], tot[i], pval, λv, tot
end

# ===========================================================================
#  Bayesian shadow fit
# ===========================================================================
#
# `tomogram_matched_filter` returns a point estimate and nothing else, and its
# own docstring warns that the formal error on that statistic is meaningless
# because the residuals are pulsation-dominated rather than white. Everything
# built on top of it inherits that: on one real γ Dor target, four different
# frequentist recipes for an interval on λ disagreed by more than an order of
# magnitude in width, and the three that were coverage-tested all missed the
# nominal 68 per cent — one far too narrow, two far too wide. (One of the four
# was the scatter of per-night point estimates, which was never tested and is
# not an interval in the first place.)
#
# The reason is that all of them describe the noise OUTSIDE the likelihood —
# by permuting, bootstrapping or calibrating a threshold — while the estimator
# itself still assumes independent pixels. This section instead puts the
# correlation INSIDE the likelihood, so the posterior on λ is wide because the
# noise model says it should be, not because a threshold was rescaled after
# the fact.
#
# NOISE MODEL. A residual tomogram is correlated along both axes and for
# different reasons: along velocity because the CCF is oversampled with respect
# to the instrumental profile and because a non-radial mode perturbs a whole
# region of the line at once, and along time because the modes are coherent
# over many hours (γ Dor periods are 0.3–1 d, longer than a transit, so within
# one night a mode is a smooth drift rather than an oscillation). A separable
# covariance
#
#     Σ = K_t ⊗ K_v + σ² I
#
# captures both without pretending to know the mode spectrum. It is also the
# structure that makes the exact likelihood cheap: eigendecomposing the two
# factors separately costs O(n_t³ + n_v³) instead of O((n_t n_v)³), which for a
# 37 × 57 map is microseconds rather than minutes.
#
# WHY THE GP DOES NOT SIMPLY EAT THE SHADOW. It partly does, and that is the
# point — the resulting posterior is wider than the white-noise one for exactly
# that reason. What stops it eating the shadow entirely is that the map
# includes the out-of-transit exposures, where no shadow exists, so the
# hyperparameters are constrained by data the planet cannot have touched; and
# that the shadow is confined to a one-parameter track through the (v, t)
# plane, which is a far stronger shape constraint than "smooth in both axes".
#
# CAVEAT, deliberately left in. `tomogram_residuals` subtracts the mean
# out-of-transit profile, which correlates the rows at the O(1/n_oot) level, and
# Σ above does not model that. Rather than complicate the covariance, the
# assumption is tested where it matters: `tomogram_coverage_test` injects a
# known λ, runs this whole fit, and counts how often the credible interval
# contains the truth. If the projection mattered, coverage would not come out
# at the nominal level.

_matern32(r::Real, ℓ::Real) = (x = sqrt(3.0) * abs(r) / ℓ; (1 + x) * exp(-x))

function _kern(x::AbstractVector, ℓ::Real)
    n = length(x)
    # Element type follows the inputs so a ForwardDiff length scale passes
    # through; for Float64 inputs this is the Float64 matrix it always was.
    K = Matrix{promote_type(eltype(x), typeof(ℓ), Float64)}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        K[i, j] = _matern32(x[i] - x[j], ℓ)
    end
    return K
end

"""
    white_map_loglike(r, σ) -> Float64

Gaussian log-likelihood for a residual map with NO correlated structure — the
null against which `kron_gp_loglike` is compared.

Exists so the GP can be a MODEL-COMPARISON axis rather than an amplitude that
shrinks toward zero. Those are different questions: "did the GP amplitude go to
zero" is answered by the posterior, "is a GP warranted at all" needs the
evidence with the term present versus absent, and only the second is a Bayes
factor. The parameter vector is deliberately unchanged when the GP is off — the
unused hyperparameters carry their priors, which integrate to 1 and so leave
the marginal likelihood untouched, keeping the two configurations directly
comparable (the same reasoning the `use_tomogram` switch already relies on).
"""
function white_map_loglike(r::AbstractMatrix, σ::Real)
    σ > 0 || return -Inf
    n = length(r)
    return -0.5 * sum(abs2, r) / σ^2 - n * log(σ) - 0.5 * n * log(2π)
end

"""
    kron_gp_loglike(r, hours, grid, ℓ_t, ℓ_v, amp, σ) -> Float64

Exact Gaussian log-likelihood of the residual matrix `r` under
`Σ = K_t ⊗ K_v + σ² I`, with unit-amplitude Matérn-3/2 factors in time and
`amp²`-scaled Matérn-3/2 in velocity (only the product of the two amplitudes is
identifiable, so the time factor carries none).

Eigendecomposing each factor once turns the solve into an elementwise divide:
with `K_t = U_t D_t U_t'` and `K_v = U_v D_v U_v'`, `Z = U_t' r U_v` gives
`r' Σ⁻¹ r = Σ_ik Z_ik² / (D_t[i] D_v[k] + σ²)` and the log-determinant is the
matching sum of logs.
"""
function kron_gp_loglike(r::AbstractMatrix, hours::AbstractVector,
                         grid::AbstractVector, ℓ_t::Real, ℓ_v::Real,
                         amp::Real, σ::Real)
    return kron_gp_loglike(r, _kern(hours, ℓ_t), (amp^2) .* _kern(grid, ℓ_v), σ)
end

"""
    kron_gp_loglike(r, K_t, K_v, σ) -> Float64

Separable Kronecker GP on a residual map, given the two covariance matrices
directly: cov(r) = K_t ⊗ K_v + σ²I.

Taking K_t as a MATRIX rather than building it from a fixed Matérn length scale
is what lets the temporal kernel come from the noise menu. The velocity axis
stays a smoothness kernel because it is instrumental — the line profile is
oversampled and correlated by the instrumental profile, which is not a competing
physical hypothesis. The time axis is: whether the pulsations are a damped
oscillator, a rotation kernel or short-memory Matérn is exactly the question
the menu exists to answer, and it should not be answered by whoever wrote the
tomography module.

The eigendecomposition is what makes this affordable: a dense (n_t·n_v)² solve
on a 30 × 200 map is 6000² and hopeless, whereas two small eigendecompositions
plus an elementwise pass is nothing.
"""
function kron_gp_loglike(r::AbstractMatrix, K_t::AbstractMatrix,
                         K_v::AbstractMatrix, σ::Real)
    Et = eigen(Symmetric(Matrix(K_t)))
    Ev = eigen(Symmetric(Matrix(K_v)))
    dt = max.(Et.values, 0.0)
    dv = max.(Ev.values, 0.0)
    Z  = Et.vectors' * r * Ev.vectors
    σ2 = σ^2
    σ2 > 0 || return -Inf
    ll = -0.5 * length(Z) * log(2π)
    @inbounds for k in axes(Z, 2), i in axes(Z, 1)
        d = dt[i] * dv[k] + σ2
        ll -= 0.5 * (Z[i, k]^2 / d + log(d))
    end
    return ll
end

"""
    celerite_kernel_dense(τgrid, ar, cr, ac, bc, cc, dc) -> Matrix

Dense covariance from celerite coefficients:
k(τ) = Σ_j a_j exp(-c_j τ) + Σ_j exp(-c_j τ)(a_j cos d_j τ + b_j sin d_j τ).

Dense on purpose. celerite's O(N) solver applies to a 1-D series; here the
temporal kernel is one factor of a Kronecker product and has to be handed over
as a matrix for the eigendecomposition. n_exposures per transit night is tens,
so a dense build is free — this would be the wrong call for a full RV series.
"""
function celerite_kernel_dense(t::AbstractVector, ar, cr, ac, bc, cc, dc)
    n = length(t)
    TT = promote_type(eltype(t), eltype(ar), eltype(cr), eltype(ac), eltype(bc),
                      eltype(cc), eltype(dc), Float64)
    return _celerite_kernel_dense!(Matrix{TT}(undef, n, n), t, ar, cr, ac, bc, cc, dc)
end

function _celerite_kernel_dense!(K::AbstractMatrix, t::AbstractVector, ar, cr, ac, bc,
                                 cc, dc)
    n = length(t)
    TT = eltype(K)
    @inbounds for j in 1:n, i in 1:n
        τ = abs(t[i] - t[j])
        v = zero(TT)
        for q in eachindex(ar); v += ar[q] * exp(-cr[q] * τ); end
        for q in eachindex(ac)
            v += exp(-cc[q] * τ) * (ac[q] * cos(dc[q] * τ) + bc[q] * sin(dc[q] * τ))
        end
        K[i, j] = v
    end
    return K
end

"""
    shadow_map(t, Tc, P, a_Rs, inc, λ, vsini, grid, σ_line, rr, u1, u2) -> Matrix

The planet's shadow as it appears in a residual tomogram: a Gaussian of width
`σ_line` centred on the sub-planet velocity, scaled by the fraction of the
stellar flux the planet covers at that moment. Zero outside transit.

The overall scale is left to a free amplitude in the fit rather than predicted.
A depth-normalised residual map is in units of the disc-integrated line depth,
and converting a blocked flux fraction into a bump height in those units needs
the LOCAL line profile — its depth and width at the sub-planet point, set by
the mask, the instrumental profile and centre-to-limb variation. That factor is
instrument-specific and not known to better than tens of per cent, so it is
fitted per night as a nuisance and not interpreted.
"""
function shadow_map(t::AbstractVector, Tc::Real, P::Real, a_Rs::Real,
                    inc::Real, λ::Real, vsini::Real, grid::AbstractVector,
                    σ_line::Real; rr::Real = 0.116, u1::Real = 0.23,
                    u2::Real = 0.15)
    M  = zeros(Float64, length(t), length(grid))
    tr = shadow_track(t, Tc, P, a_Rs, inc, λ, vsini)
    norm = 1 - u1 / 3 - u2 / 6
    @inbounds for i in eachindex(t)
        isfinite(tr[i]) || continue
        ph = 2π * (t[i] - Tc) / P
        x  = a_Rs * sin(ph)
        y  = a_Rs * cos(ph) * cos(inc)
        r  = hypot(x, y)
        μ  = sqrt(max(1 - r^2, 0.0))
        A  = rr^2 * (1 - u1 * (1 - μ) - u2 * (1 - μ)^2) / norm
        for k in eachindex(grid)
            M[i, k] = -A * exp(-0.5 * ((grid[k] - tr[i]) / σ_line)^2)
        end
    end
    return M
end

# Parameter vector for `tomogram_bayes`:
#
#   [1] λ (rad)  [2] v sin i (km/s)  [3] b  [4] a/R★
#   then six per night: α, σ_line (km/s), log₁₀ amp, log₁₀ ℓ_v (km/s),
#                       log₁₀ ℓ_t (h), log₁₀ σ_jit
#
# The shadow amplitude and line width are per night because the instruments
# differ in resolution and in the mask used to build the CCF, and the
# depth normalisation therefore lands on a different scale for each. Sharing
# them would let the best-resolved night set the template width for the worst.
#
# α has a UNIFORM prior reaching down to zero rather than a log-uniform one, so
# "this night shows no shadow" stays inside the prior support. With a log
# parametrisation a night with nothing in it drags α → −∞ and its likelihood
# surface stops being informative about anything.
#
# SHARED AMPLITUDE. With `shared_α = true` the per-night α is replaced by a
# single one in slot 5 and each night keeps its own σ_line. That model is worth
# having for one specific purpose: it makes "there is no shadow" a single point,
# α = 0, so the Savage-Dickey density ratio at that point is a Bayes factor for
# shadow against no shadow — a detection statistic computed from the same
# posterior that gives λ, rather than from a separate permutation experiment
# with its own assumptions. Estimating a posterior density at the corner of a
# three-dimensional support is not reliable enough to do the same with per-night
# amplitudes. The nights are already scaled to unit line depth, so a common α is
# defensible; what genuinely differs between them, the local line width, stays
# per night.
const _TOMO_NPAR_NIGHT = 6

_tomo_npar(n_nights::Int; shared_α::Bool = false) =
    shared_α ? 5 + 5 * n_nights : 4 + _TOMO_NPAR_NIGHT * n_nights

"""
    tomogram_logpost(θ, nights, hours, P; priors...) -> Float64

Log-posterior for the Bayesian shadow fit. See the section header for the noise
model and `tomogram_bayes` for the driver.
"""
function tomogram_logpost(θ::AbstractVector, nights::Vector{TomoNight},
                          hours::Vector{Vector{Float64}}, P::Real;
                          vsini_mu::Real, vsini_sd::Real,
                          b_mu::Real, b_sd::Real,
                          a_mu::Real, a_sd::Real,
                          rr::Real, u1::Real, u2::Real,
                          α_max::Real = 20.0, shared_α::Bool = false,
                          σline_lo::Real = 2.0, σline_hi::Real = 25.0,
                          use_gp::Bool = true)
    # λ is PERIODIC, so it is wrapped and never rejected. A hard bound at ±π is a
    # wall on a circle: a walker that steps across it is rejected and parks at
    # the edge, which on real data produced a spurious population near ±180° and a
    # 3σ interval that wrapped the whole circle. Wrapping lets the ensemble cross
    # freely; `circular_summary` wraps again at analysis time.
    λ = mod(θ[1] + π, 2π) - π
    vsini, b, a_Rs = θ[2], θ[3], θ[4]
    (vsini > 0 && a_Rs > 1 && 0 <= b < a_Rs) || return -Inf

    lp  = -0.5 * ((vsini - vsini_mu) / vsini_sd)^2
    lp += -0.5 * ((b - b_mu) / b_sd)^2
    lp += -0.5 * ((a_Rs - a_mu) / a_sd)^2
    inc = acos(b / a_Rs)
    if shared_α
        (0 <= θ[5] <= α_max) || return -Inf
    end

    for (j, nt) in enumerate(nights)
        o = shared_α ? 5 + (j - 1) * 5 : 4 + (j - 1) * _TOMO_NPAR_NIGHT
        α  = shared_α ? θ[5] : θ[o+1]
        sh = shared_α ? 0 : 1                  # per-night blocks carry α first
        σl = θ[o+sh+1]
        la, lv, lt, ls = θ[o+sh+2], θ[o+sh+3], θ[o+sh+4], θ[o+sh+5]
        (0 <= α <= α_max) || return -Inf
        (σline_lo <= σl <= σline_hi) || return -Inf
        (-6 <= la <= 0) || return -Inf
        (0.0 <= lv <= log10(60.0)) || return -Inf
        (log10(0.05) <= lt <= log10(12.0)) || return -Inf
        (-6 <= ls <= 0) || return -Inf

        M = shadow_map(nt.t, nt.Tc, P, a_Rs, inc, λ, vsini, nt.grid, σl;
                       rr = rr, u1 = u1, u2 = u2)
        r = nt.R .- α .* M
        lp += use_gp ? kron_gp_loglike(r, hours[j], nt.grid, 10.0^lt, 10.0^lv,
                                       10.0^la, 10.0^ls) :
                       white_map_loglike(r, 10.0^ls)
        isfinite(lp) || return -Inf
    end
    return lp
end

"""
    tomogram_bayes(nights; P, vsini, b, a_Rs, rr, u1, u2, ...) -> NamedTuple

Sample the joint posterior of λ and the nuisances over one or more transits.
`vsini`, `b` and `a_Rs` are `(mean, sd)` pairs, propagated as Gaussian priors so
the geometry uncertainty enters λ rather than being asserted away.

Returns `(; chain, λ, logp, names)` with `λ` in degrees, wrapped to (−180, 180].

Walkers start with λ spread uniformly around the full circle. That is deliberate:
the shadow track for λ and for some reflections of it can be similar when the
transit is short compared with the crossing time, and an ensemble started in one
basin will report a confident answer without ever having visited the other.
"""
function tomogram_bayes(nights::Vector{TomoNight};
                        P::Real, vsini::Tuple{<:Real,<:Real},
                        b::Tuple{<:Real,<:Real}, a_Rs::Tuple{<:Real,<:Real},
                        rr::Real = 0.116, u1::Real = 0.23, u2::Real = 0.15,
                        n_walkers::Int = 260, n_steps::Int = 6000,
                        n_burn::Int = 3000, thin::Int = 5, seed::Int = 20260810,
                        α_max::Real = 20.0, shared_α::Bool = false,
                        use_gp::Bool = true)
    hours = [(nt.t .- nt.Tc) .* 24 for nt in nights]
    ndim  = _tomo_npar(length(nights); shared_α = shared_α)
    lp(θ) = tomogram_logpost(θ, nights, hours, P;
                             vsini_mu = vsini[1], vsini_sd = vsini[2],
                             b_mu = b[1], b_sd = b[2],
                             a_mu = a_Rs[1], a_sd = a_Rs[2],
                             rr = rr, u1 = u1, u2 = u2, α_max = α_max,
                             shared_α = shared_α, use_gp = use_gp)

    rng = MersenneTwister(seed)
    x0  = Matrix{Float64}(undef, ndim, n_walkers)
    for w in 1:n_walkers
        ok = false
        for _ in 1:400
            θ = Vector{Float64}(undef, ndim)
            θ[1] = -π + 2π * rand(rng)
            θ[2] = vsini[1] + vsini[2] * randn(rng)
            θ[3] = clamp(b[1] + b[2] * randn(rng), 0.0, 0.99)
            θ[4] = a_Rs[1] + a_Rs[2] * randn(rng)
            shared_α && (θ[5] = 0.2 + 1.6 * rand(rng))
            for j in 1:length(nights)
                o  = shared_α ? 5 + (j - 1) * 5 : 4 + (j - 1) * _TOMO_NPAR_NIGHT
                sh = shared_α ? 0 : 1
                σ0 = std(nights[j].R)
                shared_α || (θ[o+1] = 0.2 + 1.6 * rand(rng))
                θ[o+sh+1] = 5.0 + 6.0 * rand(rng)
                θ[o+sh+2] = log10(σ0) + 0.5 * randn(rng)
                θ[o+sh+3] = log10(8.0) + 0.2 * randn(rng)
                θ[o+sh+4] = log10(1.5) + 0.2 * randn(rng)
                θ[o+sh+5] = log10(σ0) + 0.5 * randn(rng)
            end
            if isfinite(lp(θ))
                x0[:, w] = θ; ok = true; break
            end
        end
        ok || error("could not initialise walker $w with a finite posterior")
    end

    chain, lls = AffineInvariantMCMC.sample(lp, n_walkers, x0, n_steps, thin;
                                            rng = rng)
    keep = (n_burn ÷ thin + 1):size(chain, 3)
    flat = reshape(chain[:, :, keep], ndim, :)
    lpv  = vec(lls[:, keep])
    λdeg = rad2deg.(vec(flat[1, :]))
    λdeg = @. mod(λdeg + 180, 360) - 180

    names = ["lambda", "vsini", "b", "a_Rs"]
    shared_α && push!(names, "alpha")
    per = shared_α ? ("sigma_line", "log_amp", "log_lv", "log_lt", "log_jit") :
                     ("alpha", "sigma_line", "log_amp", "log_lv", "log_lt",
                      "log_jit")
    for nt in nights, s in per
        push!(names, "$(s)_$(nt.tag)")
    end
    return (; chain = flat, λ = λdeg, logp = lpv, names)
end

"""
    circular_summary(λdeg; level = 0.6827) -> (; mode, mean, lo, hi, width)

Point estimate and shortest credible interval for an angular posterior.

λ lives on a circle, so an ordinary percentile interval is wrong whenever the
posterior has weight near ±180°: the 16th and 84th percentiles of the wrapped
samples then bracket the EMPTY part of the circle and report a 300° "interval"
for a well-localised angle. This rotates to the circular mean, takes the
shortest interval containing `level` of the samples there, and rotates back.

`width` is the full width of that interval in degrees, which is the honest thing
to quote for a broad posterior — for a 1σ interval on a well-behaved unimodal
posterior it is ≈ 2σ.
"""
function circular_summary(λdeg::AbstractVector; level::Real = 0.6827)
    θ = deg2rad.(λdeg)
    c, s = mean(cos.(θ)), mean(sin.(θ))
    μ = atan(s, c)
    d = sort(@. mod(rad2deg(θ - μ) + 180, 360) - 180)   # centred residuals
    n = length(d)
    k = max(1, round(Int, level * n))
    best, i0 = Inf, 1
    for i in 1:(n - k + 1)
        w = d[i + k - 1] - d[i]
        w < best && (best = w; i0 = i)
    end
    wrap(x) = mod(x + 180, 360) - 180
    lo, hi = wrap(rad2deg(μ) + d[i0]), wrap(rad2deg(μ) + d[i0 + k - 1])
    # posterior mode from a circular histogram, as the interval can be
    # asymmetric enough that the mean is not a representative point estimate
    edges = -180:2.0:180
    cnt = zeros(Int, length(edges) - 1)
    for x in λdeg
        cnt[clamp(searchsortedlast(edges, x), 1, length(cnt))] += 1
    end
    md = 0.5 * (edges[argmax(cnt)] + edges[argmax(cnt)+1])
    return (; mode = md, mean = wrap(rad2deg(μ)), lo = lo, hi = hi, width = best)
end

"""
    shadow_bayes_factor(α_samples; α_max = 20.0, bw = nothing) -> (; logBF, BF, p0)

Bayes factor for "a shadow is present" against "none is", by the Savage-Dickey
density ratio at α = 0.

The two models are nested — no shadow is exactly α = 0 — so the ratio of
marginal likelihoods equals the prior density at that point divided by the
posterior density there. With a uniform prior on [0, α_max] the prior density is
1/α_max and only the posterior needs estimating.

CHECK CONVERGENCE BEFORE BELIEVING THIS. The ratio depends entirely on the
posterior density at one edge of the support, so it is set by whichever draws
sit in the extreme tail — and in an affine-invariant ensemble that is exactly
where walkers stuck in a distant local mode end up. On real data it has
returned ln B ≈ 0 for an amplitude posterior well separated from zero, because
a few of 260 walkers were trapped in a distant λ mode where the amplitude
collapses to zero, and
were still leaking out at the end of the chain (the low-amplitude fraction
halved across the kept samples). Any statistic resting on 0.2 per cent of the
draws will do that. Verify that the fraction near zero is stationary across the
chain, and that no walker spends most of its life there, before quoting it.

Where it is trustworthy, it beats a permutation p-value: it is a statement made
by the same posterior that reports λ, under the same noise model. A permutation
null answers a different and weaker question — whether the map has any coherent
structure at all — and on pulsating stars it answers that far too
enthusiastically, because the pulsations are themselves coherent structure.

The density at zero is estimated from a REFLECTED kernel estimate. α is bounded
below, so an ordinary kernel puts half its mass at the boundary outside the
support and underestimates the density there by about a factor of two, which
would inflate the Bayes factor by the same factor.
"""
function shadow_bayes_factor(α::AbstractVector; α_max::Real = 20.0,
                             bw::Union{Nothing,Real} = nothing)
    n = length(α)
    h = bw === nothing ? 0.9 * min(std(α), iqr_(α) / 1.34) * n^(-0.2) : float(bw)
    h > 0 || return (; logBF = NaN, BF = NaN, p0 = NaN)
    # Reflecting a sample at x about the boundary places its mirror at −x, whose
    # kernel contributes exp(−x²/2h²) at zero — the same as the original's. So
    # the reflected density at zero is exactly twice the naive one.
    acc = 0.0
    @inbounds for x in α
        acc += 2 * exp(-0.5 * (x / h)^2)
    end
    p0 = acc / (n * h * sqrt(2π))
    # Savage-Dickey: BF(no shadow / shadow) = posterior(0) / prior(0), so the
    # evidence FOR a shadow is the reciprocal, prior(0) / posterior(0) with
    # prior(0) = 1/α_max. An uninformative posterior returns the prior, giving
    # p0 = 1/α_max and logBF = 0 exactly, which is the check to run on noise.
    logBF = -log(α_max) - log(p0)
    return (; logBF = logBF, BF = exp(logBF), p0 = p0)
end

iqr_(x) = (s = sort(collect(x)); quantile(s, 0.75) - quantile(s, 0.25))

# =====================================================================
# Framework-native tomographic likelihood
# =====================================================================
#
# `tomogram_bayes` and friends run their own sampler over a bespoke theta
# vector. That was fine as a standalone study but it means the tomographic term
# cannot see anything else Nereus knows: no noise menu, no trans-dim, no shared
# lambda with the RM velocities, no joint fit with the RV or the transit.
#
# This is the same model evaluated against a `Theta`, so tomography becomes just
# another likelihood term alongside `rv_log_likelihood` and
# `transit_log_likelihood`. lambda and v sin i come from the SHARED slots
# (`lambda_k`, `v_sin_i_star`) — which is what makes a joint RM + tomography fit
# a joint fit rather than two fits averaged.

"""
    tomogram_log_likelihood(theta, data) -> ll

Log-likelihood of every residual map in `data.tomo`. Returns 0 when there is no
tomography, so it costs nothing for a normal fit.

Geometry (λ, v sin i, b, a/R★, R_p/R★, the ephemeris and the eccentricity) is
read from the layout and is therefore SHARED with the RM velocities and the
transit. Each map n is a shadow plus Gaussian noise of covariance

    Σ_n = K_t,n ⊗ K_v(ℓ_v,n) + σ_n² I,

with private per-night parameters: the shadow amplitude α (`tomo_alpha_<tag>`,
or one shared `tomo_alpha`), the local line width σ_line, the velocity
correlation length ℓ_v (Matérn-3/2) and the white term σ_n (`tomo_jit_<tag>`).
The temporal factor K_t,n -- including its amplitude -- comes from the noise
menu: the active `channel = :tomo` covariance model that covers the night
(scoped by `instruments = [tag]`, or one global model for every night). No
active model ⇒ white, `σ_n² I`.

The shadow follows the FITTED ephemeris: night n is centred on the transit of
the model orbit nearest its nominal `Tc`, so the maps and the velocities see
the same Tc and P whether they are fixed or free. For a circular orbit the
geometry is computed relative to that transit exactly as `shadow_map` does;
for an eccentric one from `planet_sky_position`.

Differentiable with ForwardDiff (the Kronecker term has an analytic
derivative; see `kron_gp_loglike`), so gradient samplers can use it.
"""
function tomogram_log_likelihood(theta::Theta{T}, data::Data) where {T}
    nights = data.tomo
    isempty(nights) && return zero(T)

    layout = theta.params.layout
    sys    = layout.systemic
    cfg    = theta.params.config
    modes  = cfg.planet_modes

    # λ from the first planet with an obliquity -- the slot the RM velocities
    # and gravity darkening read. Without one there is nothing to fit.
    k = findfirst(has_obliquity, modes)
    k === nothing && return zero(T)
    λ = planet_lambda(theta, k)
    vsini_ms = system_vsini(theta)
    vsini_ms > 0 || return convert(T, -Inf)
    vsini = vsini_ms / 1000                       # maps are in km/s

    b_raw, rr_raw = planet_b_rr(theta, k)
    b, rr = cfg.parametrization.geom === :r1r2 ?
        (b_raw * (1 - rr_raw) / 2, b_raw * (1 + rr_raw) / 2) : (b_raw, rr_raw)
    P = planet_P(theta, k)
    a_Rs = planet_a_Rs(theta, k, P)
    (isfinite(a_Rs) && a_Rs > 1) || return convert(T, -Inf)
    (0 <= b < a_Rs) || return convert(T, -Inf)
    e, ω = planet_e_w(theta, k)
    (0 <= e < 1) || return convert(T, -Inf)
    Tp, Tc0 = _planet_tp_tc(theta, k, data.t_ref, P, e, ω)
    circular = e == 0
    inc = acos(b / a_Rs)                          # circular-orbit inclination

    u1, u2 = spectroscopic_ld(theta)
    isfinite(u1) || return convert(T, -Inf)

    ll = zero(T)
    @inbounds for (j, nt) in enumerate(nights)
        α  = theta.values[sys.tomo_alpha[j]]
        σl = theta.values[sys.tomo_sigma_line[j]]
        (α >= 0 && σl > 0) || return convert(T, -Inf)

        M = if circular
            # This night's transit of the model ephemeris. Written as Tc0 + E·P
            # so that a fixed ephemeris reproduces a night's Tc = Tc0 + E·P to
            # the last bit.
            E   = round((nt.Tc - _fd_value(Tc0)) / _fd_value(P))
            Tcn = Tc0 + E * P
            _shadow_map_circular(nt.t, Tcn, P, a_Rs, inc, λ, vsini, nt.grid, σl,
                                 rr, u1, u2)
        else
            _shadow_map_orbit(nt.t, Tp, P, e, ω, b, a_Rs, λ, vsini, nt.grid, σl,
                              rr, u1, u2)
        end
        r = nt.R .- α .* M
        σw = sys.tomo_jit[j] > 0 ? theta.values[sys.tomo_jit[j]] :
                                   convert(T, _tomo_white_sigma(nt))
        σw > 0 || return convert(T, -Inf)
        Kt = _tomo_temporal_kernel(theta, nt)
        if Kt === nothing
            ll += white_map_loglike(r, σw)
        else
            ℓv = theta.values[sys.tomo_ell_v[j]]
            ℓv > 0 || return convert(T, -Inf)
            ll += _kron_ll(r, Kt, _kern(nt.grid, ℓv), σw)
        end
        isfinite(ll) || return convert(T, -Inf)
    end
    return ll
end

"""
    tomogram_log_likelihood(theta, data, ws) -> ll

`tomogram_log_likelihood(theta, data)` evaluated in the scratch of a sampler
slot (`ws::PTWorkspace`, or a `TomoWorkspace`): the same model and the same
arithmetic, bit for bit, without the ~0.9 MB the allocating path builds per
call on the NGTS-33 maps. Not for concurrent use: one workspace per task, as
for the RV buffers in the same `PTWorkspace`. Element types other than
Float64 (ForwardDiff) take the allocating path.
"""
function tomogram_log_likelihood(theta::Theta{T}, data::Data, ws) where {T}
    T === Float64 || return tomogram_log_likelihood(theta, data)
    isempty(data.tomo) && return zero(T)
    return _tomogram_ll_ws(theta, data, _tomo_workspace!(ws, theta.params, data.tomo))
end

_tomo_workspace!(tw::TomoWorkspace, params, nights) =
    (tw.params === params && tw.nights_src === nights) ? tw :
    (tw.params = params; tw.nights_src = nights; tw.nights = _tomo_night_works(nights); tw)

function _tomo_workspace!(ws::PTWorkspace, params, nights)
    tw = ws.tomo
    tw === nothing && (tw = ws.tomo = TomoWorkspace(params, nights, _tomo_night_works(nights)))
    return _tomo_workspace!(tw, params, nights)
end

TomoWorkspace(params, nights::Vector{TomoNight}) =
    TomoWorkspace(params, nights, _tomo_night_works(nights))

function _tomo_night_works(nights::Vector{TomoNight})
    return [begin
                nt_, nv = size(nt.R)
                TomoNightWork(zeros(nt_, nv), zeros(nt_, nv), zeros(nt_, nt_), zeros(nv, nv),
                              SymEigenWork(nt_), SymEigenWork(nv), zeros(nt_, nv),
                              zeros(nt_, nv), zeros(nt_), zeros(nv))
            end for nt in nights]
end

function _tomogram_ll_ws(theta::Theta{Float64}, data::Data, tw::TomoWorkspace)
    T = Float64
    nights = data.tomo
    layout = theta.params.layout
    sys    = layout.systemic
    cfg    = theta.params.config
    modes  = cfg.planet_modes

    # Decoding exactly as `tomogram_log_likelihood(theta, data)`.
    k = findfirst(has_obliquity, modes)
    k === nothing && return zero(T)
    λ = planet_lambda(theta, k)
    vsini_ms = system_vsini(theta)
    vsini_ms > 0 || return convert(T, -Inf)
    vsini = vsini_ms / 1000

    b_raw, rr_raw = planet_b_rr(theta, k)
    b, rr = cfg.parametrization.geom === :r1r2 ?
        (b_raw * (1 - rr_raw) / 2, b_raw * (1 + rr_raw) / 2) : (b_raw, rr_raw)
    P = planet_P(theta, k)
    a_Rs = planet_a_Rs(theta, k, P)
    (isfinite(a_Rs) && a_Rs > 1) || return convert(T, -Inf)
    (0 <= b < a_Rs) || return convert(T, -Inf)
    e, ω = planet_e_w(theta, k)
    (0 <= e < 1) || return convert(T, -Inf)
    Tp, Tc0 = _planet_tp_tc(theta, k, data.t_ref, P, e, ω)
    circular = e == 0
    inc = acos(b / a_Rs)

    u1, u2 = spectroscopic_ld(theta)
    isfinite(u1) || return convert(T, -Inf)

    ll = zero(T)
    @inbounds for (j, nt) in enumerate(nights)
        w  = tw.nights[j]
        α  = theta.values[sys.tomo_alpha[j]]
        σl = theta.values[sys.tomo_sigma_line[j]]
        (α >= 0 && σl > 0) || return convert(T, -Inf)

        if circular
            E   = round((nt.Tc - Tc0) / P)
            Tcn = Tc0 + E * P
            _shadow_map_circular!(w.M, nt.t, Tcn, P, a_Rs, inc, λ, vsini, nt.grid, σl,
                                  rr, u1, u2)
        else
            _shadow_map_orbit!(w.M, nt.t, Tp, P, e, ω, b, a_Rs, λ, vsini, nt.grid, σl,
                               rr, u1, u2)
        end
        r = w.r
        r .= nt.R .- α .* w.M
        σw = sys.tomo_jit[j] > 0 ? theta.values[sys.tomo_jit[j]] :
                                   convert(T, _tomo_white_sigma(nt))
        σw > 0 || return convert(T, -Inf)
        if !_tomo_temporal_kernel!(w.Kt, theta, nt)
            ll += white_map_loglike(r, σw)
        else
            ℓv = theta.values[sys.tomo_ell_v[j]]
            ℓv > 0 || return convert(T, -Inf)
            _kern!(w.Kv, nt.grid, ℓv)
            ll += _kron_gp_loglike!(w, r, σw)
        end
        isfinite(ll) || return convert(T, -Inf)
    end
    return ll
end

# `_kern`, into K.
function _kern!(K::Matrix{Float64}, x::Vector{Float64}, ℓ::Float64)
    n = length(x)
    @inbounds for j in 1:n, i in 1:n
        K[i, j] = _matern32(x[i] - x[j], ℓ)
    end
    return K
end

# `_tomo_temporal_kernel`, into K: true when an active model covering this
# night built it, false for white.
function _tomo_temporal_kernel!(K::Matrix{Float64}, theta::Theta{Float64}, nt::TomoNight)
    cfg = theta.params.config
    for (i, m) in enumerate(cfg.noise_models)
        (m isa CovarianceNoise && noise_channel(m) === :tomo) || continue
        insts = noise_instruments(m)
        (isempty(insts) || nt.tag in insts) || continue
        is_noise_model_active(theta, i) || continue
        _tomo_Kt!(K, theta, m, nt.t) && return true
    end
    return false
end

function _tomo_Kt!(K::Matrix{Float64}, theta::Theta, m::MaternGP, t)
    s = _gp_suffix(m)
    σ = _tomo_p(theta, "matern_sigma$s"); ρ = _tomo_p(theta, "matern_rho$s")
    (σ === nothing || ρ === nothing || σ <= 0 || ρ <= 0) && return false
    σ2 = σ^2
    _kern!(K, t, ρ)
    K .= σ2 .* K
    return true
end

function _tomo_Kt!(K::Matrix{Float64}, theta::Theta, m::CeleriteSHO, t)
    s = _gp_suffix(m)
    lS = _tomo_p(theta, "gp_log_S0$s"); lQ = _tomo_p(theta, "gp_log_Q$s")
    lw = _tomo_p(theta, "gp_log_omega0$s")
    (lS === nothing || lQ === nothing || lw === nothing) && return false
    ar, cr, ac, bc, cc, dc = sho_coefficients(exp(lS), exp(lQ), exp(lw))
    _celerite_kernel_dense!(K, t, ar, cr, ac, bc, cc, dc)
    return true
end

function _tomo_Kt!(K::Matrix{Float64}, theta::Theta, m::CovarianceNoise, t)
    Kt = _tomo_Kt(theta, m, t)
    Kt === nothing && return false
    copyto!(K, Kt)
    return true
end

# `kron_gp_loglike(r, Kt, Kv, σ)` in the night's scratch. The eigensystems come
# from the same LAPACK call `eigen(Symmetric(·))` makes, with the same
# workspace sizes, and the product is formed in the order `Ut' * r * Uv`
# chooses for these shapes, (Ut' r) Uv, by the same two `gemm` calls.
function _kron_gp_loglike!(w::TomoNightWork, r::Matrix{Float64}, σ::Float64)
    # Anything `eigen` would throw for: the allocating path, which throws it.
    (_sym_eigen!(w.et, w.Kt) && _sym_eigen!(w.ev, w.Kv)) ||
        return kron_gp_loglike(r, w.Kt, w.Kv, σ)
    dt, dv = w.dt, w.dv
    @inbounds for i in eachindex(dt); dt[i] = max(w.et.W[i], 0.0); end
    @inbounds for k in eachindex(dv); dv[k] = max(w.ev.W[k], 0.0); end
    BLAS.gemm!('T', 'N', 1.0, w.et.Z, r, 0.0, w.Y)
    BLAS.gemm!('N', 'N', 1.0, w.Y, w.ev.Z, 0.0, w.Z)
    Z = w.Z
    σ2 = σ^2
    σ2 > 0 || return -Inf
    ll = -0.5 * length(Z) * log(2π)
    @inbounds for k in axes(Z, 2), i in axes(Z, 1)
        d = dt[i] * dv[k] + σ2
        ll -= 0.5 * (Z[i, k]^2 / d + log(d))
    end
    return ll
end

"""
    _sym_eigen!(ew::SymEigenWork, K) -> Bool

Eigenvalues (`ew.W`) and eigenvectors (`ew.Z`) of the symmetric matrix whose
upper triangle is that of `K`, by the `dsyevr` call `eigen(Symmetric(K))`
makes (`LAPACK.syevr!('V', 'A', 'U', ...)`) with the `lwork`/`liwork` its
workspace query returns -- so the same bits. `K` is not modified. False, with
nothing computed, where `syevr!` would throw (a non-finite entry in the
triangle, LAPACK reporting failure).
"""
function _sym_eigen!(ew::SymEigenWork, K::Matrix{Float64})
    n = ew.n
    A = ew.A
    @inbounds for j in 1:n, i in 1:j
        v = K[i, j]
        isfinite(v) || return false
        A[i, j] = v
    end
    if ew.lwork == 0
        _syevr_call!(ew, BlasInt(-1), BlasInt(-1)) == 0 || return false
        lwork = BlasInt(real(ew.work[1])); liwork = ew.iwork[1]
        resize!(ew.work, lwork); resize!(ew.iwork, liwork)
        ew.lwork = lwork; ew.liwork = liwork
    end
    return _syevr_call!(ew, ew.lwork, ew.liwork) == 0
end

# One dsyevr call, exactly as LinearAlgebra.LAPACK.syevr! makes it.
function _syevr_call!(ew::SymEigenWork, lwork::BlasInt, liwork::BlasInt)
    n = ew.n
    if lwork < 0
        length(ew.work) >= 1 || resize!(ew.work, 1)
        length(ew.iwork) >= 1 || resize!(ew.iwork, 1)
    end
    m = Ref{BlasInt}()
    info = Ref{BlasInt}()
    ccall((BLAS.@blasfunc(dsyevr_), BLAS.libblastrampoline), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ref{Float64}, Ref{Float64},
           Ref{BlasInt}, Ref{BlasInt}, Ref{Float64}, Ptr{BlasInt},
           Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
           Ref{BlasInt}, Clong, Clong, Clong),
          'V', 'A', 'U', n,
          ew.A, max(1, n), 0.0, 0.0,
          0, 0, -1.0, m,
          ew.W, ew.Z, max(1, n), ew.isuppz,
          ew.work, lwork, ew.iwork, liwork,
          info, 1, 1, 1)
    (lwork < 0 || m[] == n) || return BlasInt(-999)
    return info[]
end

_fd_value(x::Real) = x
_fd_value(x::ForwardDiff.Dual) = _fd_value(ForwardDiff.value(x))

# (Tp, Tc) of planet k from its time anchor.
function _planet_tp_tc(theta::Theta, k::Int, t_ref, P, e, ω)
    ta = planet_time_anchor(theta, k)
    tm = theta.params.config.parametrization.time
    if tm === :Tc
        return tc_to_tp(ta, P, e, ω), ta
    end
    Tp = tm === :Mo ? t_ref - ta * P / (2π) : ta
    return Tp, tp_to_tc(Tp, P, e, ω)
end

# The shadow for a circular orbit: `shadow_map`'s arithmetic, step for step,
# generic in the element type.
function _shadow_map_circular(t, Tc, P, a_Rs, inc, λ, vsini, grid, σ_line, rr, u1, u2)
    TT = promote_type(typeof(Tc), typeof(P), typeof(a_Rs), typeof(inc), typeof(λ),
                      typeof(vsini), typeof(σ_line), typeof(rr), typeof(u1), Float64)
    M = Matrix{TT}(undef, length(t), length(grid))
    return _shadow_map_circular!(M, t, Tc, P, a_Rs, inc, λ, vsini, grid, σ_line, rr, u1, u2)
end

# In place, into an n_t x n_v matrix (zeroed here).
function _shadow_map_circular!(M, t, Tc, P, a_Rs, inc, λ, vsini, grid, σ_line, rr, u1, u2)
    fill!(M, zero(eltype(M)))
    sλ, cλ = sincos(λ)
    norm = 1 - u1 / 3 - u2 / 6
    @inbounds for i in eachindex(t)
        ph = 2π * (t[i] - Tc) / P
        x  = a_Rs * sin(ph)
        y  = a_Rs * cos(ph) * cos(inc)
        z  = a_Rs * cos(ph) * sin(inc)
        (hypot(x, y) < 1 && z > 0) || continue
        tr = vsini * (x * cλ - y * sλ)
        r  = hypot(x, y)
        μ  = sqrt(max(1 - r^2, zero(r)))
        A  = rr^2 * (1 - u1 * (1 - μ) - u2 * (1 - μ)^2) / norm
        for q in eachindex(grid)
            M[i, q] = -A * exp(-0.5 * ((grid[q] - tr) / σ_line)^2)
        end
    end
    return M
end

# The shadow for an eccentric orbit: positions from `planet_sky_position`, the
# same geometry the RM anomaly uses.
function _shadow_map_orbit(t, Tp, P, e, ω, b, a_Rs, λ, vsini, grid, σ_line, rr, u1, u2)
    TT = promote_type(typeof(Tp), typeof(P), typeof(e), typeof(ω), typeof(b),
                      typeof(a_Rs), typeof(λ), typeof(vsini), typeof(σ_line),
                      typeof(rr), typeof(u1), Float64)
    M = Matrix{TT}(undef, length(t), length(grid))
    return _shadow_map_orbit!(M, t, Tp, P, e, ω, b, a_Rs, λ, vsini, grid, σ_line, rr, u1, u2)
end

function _shadow_map_orbit!(M, t, Tp, P, e, ω, b, a_Rs, λ, vsini, grid, σ_line, rr, u1, u2)
    fill!(M, zero(eltype(M)))
    sλ, cλ = sincos(λ)
    norm = 1 - u1 / 3 - u2 / 6
    @inbounds for i in eachindex(t)
        x, y, z = planet_sky_position(t[i], P, e, ω, Tp, b, a_Rs)
        r = hypot(x, y)
        (r < 1 && z > 0) || continue
        tr = vsini * (x * cλ - y * sλ)
        μ  = sqrt(max(1 - r^2, zero(r)))
        A  = rr^2 * (1 - u1 * (1 - μ) - u2 * (1 - μ)^2) / norm
        for q in eachindex(grid)
            M[i, q] = -A * exp(-0.5 * ((grid[q] - tr) / σ_line)^2)
        end
    end
    return M
end

"""
Per-map white scatter, from the out-of-transit-subtracted map. Only used when a
`Params` has no `tomo_jit_<tag>` slot, which `_build_layout` always allocates
now; kept for a layout built by older code.
"""
_tomo_white_sigma(nt::TomoNight) = max(std(nt.R), 1e-8)


"""
    _tomo_temporal_kernel(theta, night) -> Matrix | nothing

Dense temporal covariance for one residual map, built from the
`channel = :tomo` covariance model of the noise menu that is ACTIVE and covers
this night: one scoped to it (`instruments = [night.tag]`) or a global one
(`instruments = []`). `nothing` means none is active, i.e. white -- the null.

Reading `is_noise_model_active` is what makes this trans-dimensional: the chain
turns kernels on and off and the map likelihood follows, so the occupancy
answers "which description do the pulsations prefer" with the same calibrated
machinery used for RV noise, rather than the kernel being whatever the module
author picked. Scoped per night, each night chooses for itself: the pulsation
amplitude a night happens to catch is not shared between epochs a year apart.

Time is in DAYS, as everywhere else in Nereus. The kernel carries the
amplitude; the velocity factor is unit-amplitude.
"""
function _tomo_temporal_kernel(theta::Theta{T}, nt::TomoNight) where {T}
    cfg = theta.params.config
    for (i, m) in enumerate(cfg.noise_models)
        (m isa CovarianceNoise && noise_channel(m) === :tomo) || continue
        insts = noise_instruments(m)
        (isempty(insts) || nt.tag in insts) || continue
        # `is_noise_model_active` is the canonical check: true in fixed-dim,
        # true for untracked indices, and the mask otherwise. Rolling this by
        # hand gets the fixed-dim and untracked cases wrong.
        is_noise_model_active(theta, i) || continue
        K = _tomo_Kt(theta, m, nt.t)
        K === nothing || return K
    end
    return nothing
end

_tomo_p(theta, name) = begin
    idx = get(theta.params.layout.name_to_idx, name, 0)
    idx == 0 ? nothing : theta.values[idx]
end

"Build K_t for a specific kernel family. Returns nothing if its params are absent."
function _tomo_Kt(theta::Theta, m::MaternGP, t)
    s = _gp_suffix(m)
    σ = _tomo_p(theta, "matern_sigma$s"); ρ = _tomo_p(theta, "matern_rho$s")
    (σ === nothing || ρ === nothing || σ <= 0 || ρ <= 0) && return nothing
    return (σ^2) .* _kern(t, ρ)
end

function _tomo_Kt(theta::Theta, m::CeleriteSHO, t)
    s = _gp_suffix(m)
    lS = _tomo_p(theta, "gp_log_S0$s"); lQ = _tomo_p(theta, "gp_log_Q$s")
    lw = _tomo_p(theta, "gp_log_omega0$s")
    (lS === nothing || lQ === nothing || lw === nothing) && return nothing
    # NATURAL log, like every SHO in Nereus (gp.jl) and like their priors. This
    # decoded 10^x while the default priors were natural-log boxes, so the
    # kernel actually used had S0, Q and ω0 raised to the power log(10).
    ar, cr, ac, bc, cc, dc = sho_coefficients(exp(lS), exp(lQ), exp(lw))
    return celerite_kernel_dense(t, ar, cr, ac, bc, cc, dc)
end

_tomo_Kt(theta::Theta, m::CovarianceNoise, t) = nothing

# ---------------------------------------------------------------------
# The Kronecker likelihood under ForwardDiff
# ---------------------------------------------------------------------
#
# `kron_gp_loglike` rests on two symmetric eigendecompositions, which have no
# ForwardDiff rule. The log-likelihood itself is smooth, though, and its
# derivative needs only the decompositions of the VALUES. With Σ = K_v ⊗ K_t
# + σ² I acting on a map X as K_t X K_v + σ² X, α = Σ⁻¹ r (the matrix A) and
# D_ik = d_t,i d_v,k + σ² in the joint eigenbasis,
#
#     dℓ = −⟨A, dr⟩ + ½ [⟨A, dK_t A K_v⟩ + ⟨A, K_t A dK_v⟩ + 2σ dσ ‖A‖²]
#          − ½ [Σ_ik (U_tᵀ dK_t U_t)_ii d_v,k / D_ik
#               + Σ_ik d_t,i (U_vᵀ dK_v U_v)_kk / D_ik + 2σ dσ Σ_ik 1/D_ik],
#
# the first bracket from the quadratic form and the second from log|Σ|. No
# eigenvector derivatives appear, so (near-)degenerate eigenvalues are no
# problem. Each partial costs two matrix products per factor.

_kron_ll(r, Kt, Kv, σ) = kron_gp_loglike(r, Kt, Kv, σ)

function _kron_ll(r::AbstractMatrix, Kt::AbstractMatrix, Kv::AbstractMatrix,
                  σ::Real)
    D = promote_type(eltype(r), eltype(Kt), eltype(Kv), typeof(σ))
    D <: ForwardDiff.Dual || return kron_gp_loglike(r, Kt, Kv, σ)
    return _kron_ll_dual(convert(Matrix{D}, r), convert(Matrix{D}, Kt),
                         convert(Matrix{D}, Kv), convert(D, σ))
end

function _kron_ll_dual(r::Matrix{D}, Kt::Matrix{D}, Kv::Matrix{D},
                       σ::D) where {D<:ForwardDiff.Dual}
    ForwardDiff.valtype(D) === Float64 || throw(ArgumentError(
        "the Doppler-tomography likelihood supports first derivatives " *
        "(ForwardDiff over Float64) only; got $(D)"))
    N  = ForwardDiff.npartials(D)
    val = ForwardDiff.value
    rv, Ktv, Kvv, σv = val.(r), val.(Kt), val.(Kv), val(σ)
    σ2 = σv^2
    σ2 > 0 || return D(-Inf)
    Et = eigen(Symmetric(Ktv)); Ev = eigen(Symmetric(Kvv))
    dt = max.(Et.values, 0.0); dv = max.(Ev.values, 0.0)
    Ut, Uv = Et.vectors, Ev.vectors
    Z  = Ut' * rv * Uv
    Dm = dt .* dv' .+ σ2
    ll = -0.5 * length(Z) * log(2π)
    @inbounds for q in eachindex(Z)
        ll -= 0.5 * (Z[q]^2 / Dm[q] + log(Dm[q]))
    end
    A   = Ut * (Z ./ Dm) * Uv'
    AKv = A * Kvv
    KtA = Ktv * A
    invD = 1 ./ Dm
    wt = invD * dv            # Σ_k d_v,k / D_ik
    wv = invD' * dt           # Σ_i d_t,i / D_ik
    sA2, sID = sum(abs2, A), sum(invD)
    g = zeros(N)
    dR  = similar(rv); dKt = similar(Ktv); dKv = similar(Kvv)
    for p in 1:N
        @inbounds for q in eachindex(r);  dR[q]  = ForwardDiff.partials(r[q], p);  end
        @inbounds for q in eachindex(Kt); dKt[q] = ForwardDiff.partials(Kt[q], p); end
        @inbounds for q in eachindex(Kv); dKv[q] = ForwardDiff.partials(Kv[q], p); end
        dσ = ForwardDiff.partials(σ, p)
        gp = -dot(A, dR)
        quad = 2σv * dσ * sA2
        ldet = 2σv * dσ * sID
        if any(!iszero, dKt)
            quad += dot(A, dKt * AKv)
            ldet += dot(vec(sum(Ut .* (dKt * Ut); dims = 1)), wt)
        end
        if any(!iszero, dKv)
            quad += dot(A, KtA * dKv)
            ldet += dot(vec(sum(Uv .* (dKv * Uv); dims = 1)), wv)
        end
        g[p] = gp + 0.5 * quad - 0.5 * ldet
    end
    return D(ll, ForwardDiff.Partials{N,Float64}(Tuple(g)))
end
