# Joint obliquity fit: the Doppler shadow and the RM velocities in one posterior.
#
# SUPERSEDED by the framework path (src/obliquity.jl + tomogram_log_likelihood).
# It works, and one study's production runs were done with it, which is the only
# reason it is still here. It carries its own theta vector, priors and celerite
# call and drives AffineInvariantMCMC directly, so it is cut off from the noise
# menu, trans-dim, sampler choice, evidence estimators, and shared parameters
# with an RV or transit fit of the same system.
#
# DELETE once the in-flight runs are reproduced through obliquity_data /
# obliquity_params. RMNight is the only thing other code still needs from here;
# move it to tomography_data.jl beside TomoNight when this goes.
#
# The two observables are different functionals of the same line profiles. The
# tomogram keeps the resolved profile; the radial velocity is its first moment.
# Fitting them together is the same move as a joint RV + bisector fit, which is
# standard practice for separating activity from orbital motion, and it is
# justified on the same grounds: two metrics computed from one cross-correlation
# function carry different information about the same star.
#
# WHAT IS SHARED AND WHAT IS NOT. λ, v sin i, b and a/R★ are common to both
# terms — they are properties of the system, and the whole point of the joint
# fit is that both observables constrain them. Everything else is separate,
# because the two observables see the noise differently: the maps carry a
# per-night shadow amplitude, local line width and a separable Kronecker GP;
# the velocities carry a per-night offset, jitter and a damped-oscillator GP.
# The RM amplitude is PREDICTED by ARoME from v sin i, r/R★ and the limb
# darkening, with nothing free to rescale it.
#
# WHAT TO WATCH. A radial velocity is a functional of the profile, so the two
# likelihoods do share photons, and how much depends on how the velocities were
# extracted. Velocities that are the centroid of the very CCF the tomogram is
# built from share almost everything; velocities from an independent reduction
# (template matching over orders the CCF mask never touches) share much less.
# `use_tomogram` and passing a subset of `rm_nights` let a caller measure the
# size of that effect instead of arguing about it: fit the tomogram alone, the
# velocities alone, and both, and compare. On a γ Dor host the velocities did not
# tighten λ, and including the velocities that share a CCF with the tomogram
# moved the peak by ~3°, which is the signature to look for.
#
# λ IS PERIODIC AND IS WRAPPED, NEVER BOUNDED. A hard rejection outside ±π is a
# wall on a circle: a walker stepping across it is rejected and parks at the
# edge. On real data that produced a spurious population near ±180° and a 3σ
# interval that wrapped the whole circle before it was found.

using Statistics
using Random: MersenneTwister, randn, rand
using AffineInvariantMCMC

"""
    RMNight(tag, t, rv, err, σ0, Tc)

One night of in-transit radial velocities for the joint fit. `rv` and `err` are
in m/s, `t` in BJD, `Tc` that night's mid-transit time.

`σ0` is the dispersion in m/s of a Gaussian fitted to the mean OUT-OF-TRANSIT
cross-correlation function. It must be MEASURED, not derived from v sin i:
Boué et al. (2013) are explicit that their expression holds when σ0 is the
observed CCF width, and deriving it from v sin i reintroduces the approximation
the formulation exists to avoid.
"""
struct RMNight
    tag::String
    t::Vector{Float64}
    rv::Vector{Float64}
    err::Vector{Float64}
    σ0::Float64
    Tc::Float64
end

"""
    rm_anomaly(t, Tc, P, a_Rs, inc, λ, vsini, σ0; rr, u1, u2) -> Vector

The ARoME radial-velocity anomaly (Boué et al. 2013, Eq. 15) for one night, in
m/s, zero outside transit. `vsini` and `σ0` in m/s.

The sub-planet line width follows from the CCF width rather than being fitted:
a rotationally broadened profile fitted by a Gaussian has σ ≈ 0.5503 v sin i
convolved with the local line, so β_p² = σ0² − (0.5503 v sin i)². It is
recomputed per call because v sin i is sampled.
"""
function rm_anomaly(t::AbstractVector, Tc::Real, P::Real, a_Rs::Real, inc::Real,
                    λ::Real, vsini::Real, σ0::Real; rr::Real = 0.116,
                    u1::Real = 0.23, u2::Real = 0.15)
    βp = sqrt(max(σ0^2 - (0.5503 * vsini)^2, 4.0e6))
    norm = 1 - u1 / 3 - u2 / 6
    out = zeros(Float64, length(t))
    @inbounds for i in eachindex(t)
        x, y, z = planet_sky_position(t[i] - Tc, P, 0.0, π/2, 0.0,
                                      a_Rs * cos(inc), a_Rs)
        (hypot(x, y) < 1 && z > 0) || continue
        μ  = sqrt(max(1 - x^2 - y^2, 0.0))
        Δf = rr^2 * (1 - u1 * (1 - μ) - u2 * (1 - μ)^2) / norm
        out[i] = rm_signal_arome(x, y, Δf, vsini, λ, σ0, βp)
    end
    return out
end

# Parameter vector for the joint fit:
#
#   [1] λ (rad, wrapped)  [2] v sin i (km/s)  [3] b  [4] a/R★
#   then 6 per tomographic night, exactly as `tomogram_logpost` expects
#   then, if any RM nights are supplied, [K] and 5 per RM night:
#       γ (m/s), log₁₀ S0, log₁₀ Q, log₁₀ ω0 (rad/day), log₁₀ jitter
#
# The tomographic block is always present in the vector even when
# `use_tomogram = false`, so that the three configurations a caller compares
# share an identical parameter space and walker initialisation. With the
# tomogram off those parameters simply carry their priors.
_joint_npar(n_tomo::Int, n_rm::Int) =
    4 + _TOMO_NPAR_NIGHT * n_tomo + (n_rm > 0 ? 1 + 5 * n_rm : 0)

"""
    joint_obliquity_logpost(θ, tomo_nights, hours, rm_nights, P; priors...)

Log-posterior for the joint shadow-plus-velocity fit. See the file header for
what is shared between the two terms and what is not.
"""
function joint_obliquity_logpost(θ::AbstractVector,
                                 tomo_nights::Vector{TomoNight},
                                 hours::Vector{Vector{Float64}},
                                 rm_nights::Vector{RMNight}, P::Real;
                                 vsini_mu::Real, vsini_sd::Real,
                                 b_mu::Real, b_sd::Real,
                                 a_mu::Real, a_sd::Real,
                                 K_mu::Real = 0.0, K_sd::Real = Inf,
                                 rr::Real, u1::Real, u2::Real,
                                 α_max::Real = 20.0,
                                 use_tomogram::Bool = true,
                                 use_tomo_gp::Bool = true,
                                 use_rv_gp::Bool = true)
    λ = mod(θ[1] + π, 2π) - π
    vsini, b, a_Rs = θ[2], θ[3], θ[4]
    (vsini > 0 && a_Rs > 1 && 0 <= b < a_Rs) || return -Inf

    lp = if use_tomogram
        tomogram_logpost(θ, tomo_nights, hours, P;
                         vsini_mu = vsini_mu, vsini_sd = vsini_sd,
                         b_mu = b_mu, b_sd = b_sd, a_mu = a_mu, a_sd = a_sd,
                         rr = rr, u1 = u1, u2 = u2, α_max = α_max,
                         use_gp = use_tomo_gp)
    else
        -0.5 * ((vsini - vsini_mu) / vsini_sd)^2 -
        0.5 * ((b - b_mu) / b_sd)^2 - 0.5 * ((a_Rs - a_mu) / a_sd)^2
    end
    isfinite(lp) || return -Inf
    isempty(rm_nights) && return lp

    off = 4 + _TOMO_NPAR_NIGHT * length(tomo_nights)
    K   = θ[off + 1]
    isfinite(K_sd) && (lp += -0.5 * ((K - K_mu) / K_sd)^2)
    inc = acos(b / a_Rs)

    for (q, n) in enumerate(rm_nights)
        o = off + 1 + (q - 1) * 5
        γ, lS, lQ, lw, lj = θ[o+1], θ[o+2], θ[o+3], θ[o+4], θ[o+5]
        (-2 <= lS <= 12) || return -Inf
        (log10(0.2) <= lQ <= 2.0) || return -Inf
        (log10(0.2) <= lw <= log10(60.0)) || return -Inf    # rad/day
        (-1 <= lj <= 3.5) || return -Inf

        kep = @. -K * sin(2π * (n.t - n.Tc) / P)
        rm  = rm_anomaly(n.t, n.Tc, P, a_Rs, inc, λ, vsini * 1e3, n.σ0;
                         rr = rr, u1 = u1, u2 = u2)
        r = n.rv .- kep .- rm .- γ
        v = n.err .^ 2 .+ (10.0^lj)^2
        if use_rv_gp
            ar, cr, ac, bc, cc, dc = sho_coefficients(10.0^lS, 10.0^lQ, 10.0^lw)
            lp += celerite_loglike(n.t, r, v, ar, cr, ac, bc, cc, dc)
        else
            # White null for the velocities: jitter only, no damped oscillator.
            # Same rationale as white_map_loglike — a Bayes factor needs the
            # term ABSENT, not merely shrunk.
            lp += -0.5 * sum(@. r^2 / v + log(v) + log(2π))
        end
        isfinite(lp) || return -Inf
    end
    return lp
end

"""
    joint_obliquity_fit(tomo_nights, rm_nights; P, vsini, b, a_Rs, ...) -> NamedTuple

Sample the joint posterior. `vsini`, `b`, `a_Rs` and `K` are `(mean, sd)` pairs.
Pass `rm_nights = RMNight[]` for the tomogram alone, or `use_tomogram = false`
for the velocities alone; the parameter space is identical in all three cases so
the results are directly comparable.

Returns `(; chain, λ, logp, names)` with `λ` in degrees, wrapped to (−180, 180].
Walkers start with λ spread uniformly around the full circle.
"""
function joint_obliquity_fit(tomo_nights::Vector{TomoNight},
                             rm_nights::Vector{RMNight};
                             P::Real, vsini::Tuple{<:Real,<:Real},
                             b::Tuple{<:Real,<:Real},
                             a_Rs::Tuple{<:Real,<:Real},
                             K::Tuple{<:Real,<:Real} = (0.0, Inf),
                             rr::Real = 0.116, u1::Real = 0.23, u2::Real = 0.15,
                             use_tomogram::Bool = true,
                             use_tomo_gp::Bool = true,
                             use_rv_gp::Bool = true,
                             n_walkers::Int = 300, n_steps::Int = 8000,
                             n_burn::Int = 4000, thin::Int = 5,
                             seed::Int = 20260813, α_max::Real = 20.0)
    Base.depwarn("joint_obliquity_fit is superseded by the framework path " *
        "(obliquity_data/obliquity_params + a standard sampler), which reaches the " *
        "noise menu, trans-dim and evidence estimators this cannot see.",
        :joint_obliquity_fit)
    hours = [(nt.t .- nt.Tc) .* 24 for nt in tomo_nights]
    ndim  = _joint_npar(length(tomo_nights), length(rm_nights))
    lp(θ) = joint_obliquity_logpost(θ, tomo_nights, hours, rm_nights, P;
                                    vsini_mu = vsini[1], vsini_sd = vsini[2],
                                    b_mu = b[1], b_sd = b[2],
                                    a_mu = a_Rs[1], a_sd = a_Rs[2],
                                    K_mu = K[1], K_sd = K[2],
                                    rr = rr, u1 = u1, u2 = u2,
                                    α_max = α_max, use_tomogram = use_tomogram,
                                    use_tomo_gp = use_tomo_gp,
                                    use_rv_gp = use_rv_gp)

    rng = MersenneTwister(seed)
    x0  = Matrix{Float64}(undef, ndim, n_walkers)
    off = 4 + _TOMO_NPAR_NIGHT * length(tomo_nights)
    for w in 1:n_walkers
        ok = false
        for _ in 1:800
            θ = Vector{Float64}(undef, ndim)
            θ[1] = -π + 2π * rand(rng)
            θ[2] = vsini[1] + vsini[2] * randn(rng)
            θ[3] = clamp(b[1] + b[2] * randn(rng), 0.0, 0.99)
            θ[4] = a_Rs[1] + a_Rs[2] * randn(rng)
            for (j, nt) in enumerate(tomo_nights)
                o = 4 + (j - 1) * _TOMO_NPAR_NIGHT
                σ0 = std(nt.R)
                θ[o+1] = 0.2 + 1.6 * rand(rng)
                θ[o+2] = 5.0 + 6.0 * rand(rng)
                θ[o+3] = log10(σ0) + 0.5 * randn(rng)
                θ[o+4] = log10(8.0) + 0.2 * randn(rng)
                θ[o+5] = log10(1.5) + 0.2 * randn(rng)
                θ[o+6] = log10(σ0) + 0.5 * randn(rng)
            end
            if !isempty(rm_nights)
                θ[off+1] = isfinite(K[2]) ? K[1] + K[2]*randn(rng) : K[1]
                for (q, n) in enumerate(rm_nights)
                    o = off + 1 + (q - 1) * 5
                    θ[o+1] = median(n.rv) + 50 * randn(rng)
                    θ[o+2] = log10(var(n.rv)) + 0.3 * randn(rng)
                    θ[o+3] = log10(1.0) + 0.2 * randn(rng)
                    θ[o+4] = log10(6.0) + 0.2 * randn(rng)
                    θ[o+5] = log10(30.0) + 0.3 * randn(rng)
                end
            end
            if isfinite(lp(θ)); x0[:, w] = θ; ok = true; break; end
        end
        ok || error("could not initialise walker $w with a finite posterior")
    end

    chain, lls = AffineInvariantMCMC.sample(lp, n_walkers, x0, n_steps, thin;
                                            rng = rng)
    keep = (n_burn ÷ thin + 1):size(chain, 3)
    flat = reshape(chain[:, :, keep], ndim, :)
    λdeg = rad2deg.(vec(flat[1, :]))
    λdeg = @. mod(λdeg + 180, 360) - 180

    names = ["lambda", "vsini", "b", "a_Rs"]
    for nt in tomo_nights, s in ("alpha", "sigma_line", "log_amp", "log_lv",
                                 "log_lt", "log_jit")
        push!(names, "$(s)_$(nt.tag)")
    end
    if !isempty(rm_nights)
        push!(names, "K")
        for n in rm_nights, s in ("gamma", "log_S0", "log_Q", "log_w0", "log_jit")
            push!(names, "$(s)_rv_$(n.tag)")
        end
    end
    return (; chain = flat, λ = λdeg, logp = vec(lls[:, keep]), names)
end

"""
    rm_velocity_fit(rm_nights; P, vsini, b, a_Rs, K, ...) -> NamedTuple

Fit the Rossiter-McLaughlin anomaly in RADIAL VELOCITY alone, with the stellar
pulsations carried by a damped-harmonic-oscillator Gaussian process per night
and the RM amplitude predicted by ARoME.

This is `joint_obliquity_fit` with the tomographic term switched off, and it is
kept as its own entry point because it is the measurement most papers make: it
is what a spectroscopic transit gives you without a resolved line profile.

The amplitude is deliberately NOT free. A scale factor on the RM term will
happily absorb pulsation power and return a confident, unphysical answer; with
the amplitude fixed by v sin i, r/R★ and the limb darkening, a conflict between
the data and the model shows up as a wide or multimodal posterior on λ, which is
the honest failure mode. On a γ Dor host expect exactly that: the velocities
give a bimodal λ whose two solutions differ by less than half the residual
scatter.
"""
function rm_velocity_fit(rm_nights::Vector{RMNight}; kwargs...)
    Base.depwarn("rm_velocity_fit is superseded by the framework path; RM " *
        "velocities already flow through rv_log_likelihood, so a Params built " *
        "by obliquity_params gets the noise menu per night.", :rm_velocity_fit)
    return joint_obliquity_fit(TomoNight[], rm_nights; use_tomogram = false,
                               kwargs...)
end
