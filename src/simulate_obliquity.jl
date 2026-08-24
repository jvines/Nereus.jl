# Forward simulation of a tomographic night and of an RM velocity night.
#
# The purpose is proposal-grade sensitivity work: choose a geometry, a cadence
# and a noise level, and get back data objects that the REAL analysis path
# consumes without adaptation. `simulate_tomogram` returns the field names
# `tomogram_pooled` expects and `simulate_rm_night` returns an `RMNight`, so a
# simulated night and an observed one are interchangeable downstream. Anything
# that only works on simulated data is a way of certifying a pipeline that will
# not be run on the sky.
#
# The shadow itself is NOT re-derived here. `shadow_map` is called to produce
# it, so the simulator cannot drift away from the model the fit assumes: if the
# shadow definition changes, both move together and a recovery test stays
# honest. The simulator's own job is only the three things `shadow_map` does
# not know about — where the line sits in the observed frame, how deep it is,
# and what the noise looks like.
#
# UNITS. `sigma_pixel` is per velocity pixel in units of the disc-integrated
# line depth, which is the frame `tomogram_residuals(..., norm_depth = true)`
# leaves its output in and therefore the frame `shadow_map`'s amplitude
# (rr^2 times a limb-darkening factor) already lives in. That makes the noise
# argument directly comparable between simulated and observed nights: measure
# the out-of-transit scatter of a real residual map and pass the number
# straight in. For scale: a mid-F pulsator observed at R~115k in 632 s frames
# sits near 0.066.
#
# WHITE NOISE IS AN UPPER BOUND, not an expectation. On a hot star the
# out-of-transit residuals are dominated by non-radial pulsations, which are
# coherent and do not average down as sqrt(N). A real night of such a star can
# predict ~3.6 sigma under the white assumption and still yield no individual
# detection, with only the pooled multi-night result carrying the signal. Use
# `noise_frames` to inject into real residual frames whenever a defensible
# number is needed, and treat the white-noise mode as the best case that
# bounds it from above.

"""
    simulate_tomogram(; P, Tc, a_Rs, λ, vsini, rr, T14, sigma_pixel, b or inc, ...)

Simulate one night of line profiles containing a planet shadow.

Returns a NamedTuple with fields `(tag, profiles, vgrid, times, bervs, Tc,
in_transit, truth)`. The first six are exactly what `tomogram_pooled` takes for
one night, so a vector of these drops straight in. For `tomogram_bayes`, build
the residual map first:

    s = simulate_tomogram(; ...)
    g, R = tomogram_residuals(s.profiles, s.vgrid, s.in_transit; bervs = s.bervs)
    night = TomoNight(s.tag, s.times, R, g, s.Tc)

# Geometry
`P` days, `Tc` BJD, `a_Rs` in stellar radii, `λ` radians, `vsini` km/s, `rr` the
radius ratio, `T14` the full transit duration in HOURS. Give either `b` (impact
parameter, from which `inc` follows) or `inc` in radians.

# Sampling
`cadence` seconds and `span` hours around `Tc`; `span` defaults to `2*T14`,
which puts as many frames out of transit as in it. The out-of-transit frames
are not optional padding — `tomogram_residuals` builds its reference profile
from them, and a night with too few produces a reference contaminated by the
shadow it is meant to reveal.

The returned `in_transit` flag is a NOMINAL `T14` mask, which is what a real
analysis has to use. The shadow itself is gated on the planet centre lying on
the disc — a shorter interval — so the first and last flagged frames can
legitimately carry no signal. That is not a sampling bug.

# Line shape
`sigma_line` is the width of the shadow, defaulting to `vsini/2.5` to match
`tomogram_injection_test`. `sigma_prof` is the width of the disc-integrated
profile and defaults to the ARoME relation `hypot(0.5503*vsini, sigma_line)`,
so the simulated CCF has the width a rotator of this `vsini` actually shows.

# Noise
`sigma_pixel` per pixel in unit-line-depth units. `sigma_ip` optionally
correlates it over a Gaussian of that width in km/s before rescaling to the
requested per-pixel dispersion; CCF pixels sampled finer than the instrumental
profile are correlated in reality, and pure white noise overstates how much a
matched filter gains by summing across the line. `noise_frames`, if given, is
an (n_frame x n_v) matrix in the same units added on top — the hook for
injecting real pulsating residuals instead of a Gaussian fiction.

`truth` carries the injected geometry back, plus `snr_white`, the matched-filter
significance in the white-noise limit. Compare a recovered significance against
that number, never against zero.
"""
function simulate_tomogram(; P::Real, Tc::Real, a_Rs::Real, λ::Real,
                           vsini::Real, rr::Real, T14::Real,
                           sigma_pixel::Real,
                           b::Union{Nothing,Real} = nothing,
                           inc::Union{Nothing,Real} = nothing,
                           cadence::Real = 600.0,
                           span::Union{Nothing,Real} = nothing,
                           sigma_line::Union{Nothing,Real} = nothing,
                           sigma_prof::Union{Nothing,Real} = nothing,
                           u1::Real = 0.23, u2::Real = 0.15,
                           vsys::Real = 0.0, berv::Real = 0.0,
                           vgrid = range(-80, 80; length = 321),
                           sigma_ip::Real = 0.0,
                           line_depth::Real = 1.0,
                           noise_frames = nothing,
                           tag::AbstractString = "SIM",
                           rng = Random.default_rng())
    (b === nothing) == (inc === nothing) &&
        throw(ArgumentError("give exactly one of `b` or `inc`"))
    i_orb = inc === nothing ? acos(clamp(b / a_Rs, -1, 1)) : float(inc)
    σ_l   = sigma_line === nothing ? vsini / 2.5 : float(sigma_line)
    σ_p   = sigma_prof === nothing ? hypot(0.5503 * vsini, σ_l) : float(sigma_prof)
    T14d  = T14 / 24
    spanh = span === nothing ? 2 * T14 : float(span)

    n = max(2, round(Int, spanh * 3600 / cadence) + 1)
    times = collect(range(Tc - spanh / 48, Tc + spanh / 48; length = n))
    in_transit = abs.((times .- Tc) .* 24) .<= T14 / 2

    v = collect(vgrid)
    centre = vsys - berv                    # where the line sits in the observed frame

    # The shadow, from the same function the fit uses. Its `grid` argument is in
    # the stellar rest frame, so hand it the observed grid shifted to that frame
    # and the returned map is already aligned with the profile built below.
    M = shadow_map(times, Tc, P, a_Rs, i_orb, λ, vsini, v .- centre, σ_l;
                   rr = rr, u1 = u1, u2 = u2)

    profiles = Matrix{Float64}(undef, n, length(v))
    @inbounds for i in 1:n
        for j in eachindex(v)
            profiles[i, j] = line_depth *
                (exp(-0.5 * ((v[j] - centre) / σ_p)^2) + M[i, j])
        end
    end

    if sigma_pixel > 0
        E = randn(rng, n, length(v))
        if sigma_ip > 0
            E = _smooth_rows(E, v, sigma_ip)
        end
        profiles .+= (line_depth * sigma_pixel) .* E
    end
    if noise_frames !== nothing
        size(noise_frames) == size(profiles) ||
            throw(DimensionMismatch("noise_frames is $(size(noise_frames)), " *
                                    "profiles are $(size(profiles))"))
        profiles .+= line_depth .* noise_frames
    end

    snr = sigma_pixel > 0 ? sqrt(sum(abs2, M) / sigma_pixel^2) : Inf
    truth = (; λ = λ, λ_deg = rad2deg(λ), vsini = vsini, rr = rr,
             b = a_Rs * cos(i_orb), inc = i_orb, a_Rs = a_Rs, P = P,
             sigma_line = σ_l, sigma_prof = σ_p, sigma_pixel = sigma_pixel,
             n_in_transit = count(in_transit), n_frames = n,
             amp_max = maximum(abs, M), snr_white = snr)

    return (; tag = String(tag), profiles = profiles, vgrid = v, times = times,
            bervs = fill(float(berv), n), Tc = float(Tc),
            in_transit = in_transit, truth = truth)
end

# Correlate white noise along the velocity axis, then restore unit dispersion.
# Smoothing lowers the per-pixel variance by the kernel's norm; rescaling keeps
# `sigma_pixel` meaning what the caller thinks it means, so the only change is
# the correlation structure.
function _smooth_rows(E::AbstractMatrix, v::AbstractVector, σ::Real)
    dv = length(v) > 1 ? abs(v[2] - v[1]) : 1.0
    half = max(1, ceil(Int, 4 * σ / dv))
    k = [exp(-0.5 * (d * dv / σ)^2) for d in -half:half]
    k ./= sqrt(sum(abs2, k))            # preserves variance of white input
    nv = size(E, 2)
    out = zeros(eltype(E), size(E))
    @inbounds for i in axes(E, 1), j in 1:nv
        acc = 0.0
        for (m, d) in enumerate(-half:half)
            jj = j + d
            1 <= jj <= nv && (acc += k[m] * E[i, jj])
        end
        out[i, j] = acc
    end
    return out
end

"""
    simulate_rm_night(; P, Tc, a_Rs, λ, vsini, rr, T14, sigma_rv, sigma0, b or inc, ...)

Simulate one night of in-transit radial velocities containing the RM anomaly.

Returns `(; night, truth)` where `night` is an `RMNight` ready for
`joint_obliquity_fit`. The anomaly comes from `rm_anomaly` (ARoME, Boué et al.
2013), the orbital term is the circular Keplerian through transit,
`-K sin(2π(t-Tc)/P)`, and `gamma` is added as the systemic offset.

`vsini` is in km/s here for consistency with `simulate_tomogram`, and converted
internally; `sigma0`, `K`, `gamma` and `sigma_rv` are all in m/s. `sigma0` is
the observed out-of-transit CCF dispersion and defaults to the ARoME-consistent
`hypot(0.5503*vsini, vsini/2.5)` — for real data it must be MEASURED, which is
why `RMNight` carries it rather than deriving it.

For v sin i above roughly 20 km/s this channel carries much less information
than the tomogram: the anomaly is diluted across a broad line and a CCF centroid
is a poor summary of a profile whose shape is what changed. Simulate both before
deciding which one the proposal asks time for.
"""
function simulate_rm_night(; P::Real, Tc::Real, a_Rs::Real, λ::Real,
                           vsini::Real, rr::Real, T14::Real,
                           sigma_rv::Real,
                           b::Union{Nothing,Real} = nothing,
                           inc::Union{Nothing,Real} = nothing,
                           sigma0::Union{Nothing,Real} = nothing,
                           K::Real = 0.0, gamma::Real = 0.0,
                           cadence::Real = 600.0,
                           span::Union{Nothing,Real} = nothing,
                           u1::Real = 0.23, u2::Real = 0.15,
                           tag::AbstractString = "SIM",
                           rng = Random.default_rng())
    (b === nothing) == (inc === nothing) &&
        throw(ArgumentError("give exactly one of `b` or `inc`"))
    i_orb = inc === nothing ? acos(clamp(b / a_Rs, -1, 1)) : float(inc)
    vs_ms = vsini * 1000
    σ0    = sigma0 === nothing ? hypot(0.5503 * vs_ms, vs_ms / 2.5) : float(sigma0)
    spanh = span === nothing ? 2 * T14 : float(span)

    n = max(2, round(Int, spanh * 3600 / cadence) + 1)
    t = collect(range(Tc - spanh / 48, Tc + spanh / 48; length = n))

    anom = rm_anomaly(t, Tc, P, a_Rs, i_orb, λ, vs_ms, σ0; rr = rr, u1 = u1, u2 = u2)
    orb  = @. -K * sin(2π * (t - Tc) / P)
    rv   = gamma .+ orb .+ anom .+ sigma_rv .* randn(rng, n)
    err  = fill(float(sigma_rv), n)

    in_transit = abs.((t .- Tc) .* 24) .<= T14 / 2
    snr = sigma_rv > 0 ? sqrt(sum(abs2, anom) / sigma_rv^2) : Inf
    truth = (; λ = λ, λ_deg = rad2deg(λ), vsini = vsini, rr = rr,
             b = a_Rs * cos(i_orb), inc = i_orb, a_Rs = a_Rs, P = P,
             sigma0 = σ0, K = K, gamma = gamma, sigma_rv = sigma_rv,
             amp_max = maximum(abs, anom), n_in_transit = count(in_transit),
             n_frames = n, snr_white = snr)

    return (; night = RMNight(String(tag), t, rv, err, σ0, float(Tc)),
            truth = truth)
end
