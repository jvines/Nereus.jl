# Framework-native obliquity fitting: RM velocities and/or Doppler tomography.
#
# The bespoke path (`obliquity_joint.jl`) carries its own theta vector, its own
# priors and its own celerite call, and drives AffineInvariantMCMC directly.
# That is self-contained but it is also cut off from everything else Nereus
# knows — no noise menu, no trans-dimensional model selection, no choice of
# sampler, no evidence estimators, no shared parameters with an RV or transit
# fit of the same system.
#
# This builds the SAME model as a `Params`, so all of that applies. The pieces
# were already there: `_decode_rm_state` is called from `rv_log_likelihood`
# (likelihood.jl), so RM velocities have always flowed through the standard
# noise machinery; and `tomogram_log_likelihood` now adds the maps. What was
# missing was an entry point that assembles them.
#
# ONE RM NIGHT = ONE INSTRUMENT. That is not a hack to reuse plumbing: each
# night genuinely needs its own systemic offset and its own jitter (different
# night, different conditions, often a different pipeline), which is exactly
# what an instrument is in this codebase. It also means the noise menu applies
# PER NIGHT, so one pulsating night can select a different noise model from a
# quiet one instead of forcing a single compromise.
#
# GEOMETRY WITHOUT PHOTOMETRY. RM modes require PM by construction (the kernel
# needs the transit geometry), but `t_phot` may be empty: the transit term then
# contributes zero and b, r/R★ and rho_s are constrained by their priors, which
# is what an obliquity fit with a published ephemeris actually wants. Supply
# those as NormalPriors from the discovery paper.

export RMNight, obliquity_data, obliquity_params

"""
    obliquity_data(rm_nights; tomo_nights, t_ref) -> (Data, Vector{String})

Assemble RM velocity nights (and optionally tomographic maps) into a `Data`,
returning it with the instrument names in order. Each night becomes its own
instrument — see the file header for why that is the right structure and not a
convenience.
"""
function obliquity_data(rm_nights::Vector{RMNight};
                        tomo_nights::Vector{TomoNight} = TomoNight[],
                        t_ref::Union{Nothing,Real} = nothing)
    isempty(rm_nights) && isempty(tomo_nights) &&
        throw(ArgumentError("obliquity_data: no RM nights and no tomography"))
    names = String[]; t = Float64[]; rv = Float64[]; err = Float64[]; inst = Int[]
    for (i, n) in enumerate(rm_nights)
        nm = isempty(n.tag) ? "night$i" : n.tag
        nm in names && throw(ArgumentError(
            "duplicate RM night tag `$nm` — tags become instrument names and " *
            "must be unique, or their offsets and jitters collide"))
        push!(names, nm)
        append!(t, n.t); append!(rv, n.rv); append!(err, n.err)
        append!(inst, fill(i, length(n.t)))
    end
    if isempty(rm_nights)
        # Tomography-only: Data still needs an RV column, so give it one point
        # of infinite error at the reference epoch. It contributes nothing.
        push!(names, "none")
        t = [tomo_nights[1].Tc]; rv = [0.0]; err = [1e9]; inst = [1]
    end
    d = Data(t_rv = t, rv = rv, rv_err = err, rv_inst = inst,
             tomo = tomo_nights,
             t_ref = t_ref === nothing ? sum(t)/length(t) : Float64(t_ref))
    return d, names
end

"""
    obliquity_params(data, inst_names; P, b, a_Rs, rr, M_s, R_s, priors,
                     noise_models, transdim_noise, arome) -> Params

Build the `Params` for an obliquity fit. `b`, `a_Rs` and `rr` are
`(mean, sd)` tuples — the published transit solution — because an obliquity fit
normally has no light curve of its own to constrain them.

Pass `noise_models` from `default_noise_menu(data)` and `transdim_noise = true`
to let the data choose the noise description per night rather than assuming one.
"""
function obliquity_params(data::Data, inst_names::Vector{String};
                          P::Real, Tc::Real,
                          b::Tuple{<:Real,<:Real},
                          a_Rs::Tuple{<:Real,<:Real},
                          rr::Tuple{<:Real,<:Real} = (0.1, 0.02),
                          vsini::Tuple{<:Real,<:Real} = (10_000.0, 5_000.0),
                          M_s::Real = 1.0, R_s::Real = 1.0,
                          use_rho_s::Bool = true,
                          arome::Bool = false,
                          noise_models::Vector{<:NoiseModel} = NoiseModel[],
                          transdim_noise::Bool = false,
                          priors::Dict{String,PriorSpec} = Dict{String,PriorSpec}())
    mode = arome ? RVPM_RM_A : RVPM_RM
    pr = copy(priors)
    # Geometry from the published solution. Narrow, because the obliquity fit
    # is not trying to re-derive the transit — it is asking where the shadow
    # crosses, given it.
    get!(pr, "P_k1",   NormalPrior(Float64(P), 1e-6 * P, 0.0, 10P))
    # Time is parametrised as mean anomaly at t_ref by default, not as Tc, so
    # the ephemeris is pinned through `Mo_k1` rather than a transit-time slot.
    # A NormalPrior on Mo of width 2*pi*sigma_Tc/P carries the published
    # ephemeris uncertainty into the right coordinate.
    if !haskey(pr, "Mo_k1")
        # Transit is at true anomaly f = pi/2 - omega, NOT at mean anomaly 0,
        # so Tc -> Mo must go through the library conversions. Rolling it by
        # hand as 2*pi*(Tc - t_ref)/P puts the planet a quarter-orbit off, no
        # RM point lands in transit, and the anomaly is then zero for EVERY
        # lambda — a silent null rather than an error.
        Tp = tc_to_tp(Float64(Tc), Float64(P), 0.0, 0.0)
        Mo = mod(tp_to_mo(Tp, Float64(P), data.t_ref), 2π)
        σ_Mo = min(2π * 0.01 / Float64(P), 0.5)
        pr["Mo_k1"] = NormalPrior(Mo, σ_Mo, 0.0, 2π)
    end
    get!(pr, "b_k1",   NormalPrior(Float64(b[1]), Float64(b[2]), 0.0, 1.0))
    get!(pr, "rr_k1",  NormalPrior(Float64(rr[1]), Float64(rr[2]), 0.0, 0.5))
    get!(pr, "v_sin_i_star",
         NormalPrior(Float64(vsini[1]), Float64(vsini[2]), 100.0, 300_000.0))
    # lambda stays UNIFORM on the full circle by default. An obliquity fit that
    # starts from a prior centred on zero is not measuring the obliquity.
    get!(pr, "lambda_k1", UniformPrior(-π, π))

    # a_Rs is a CONSTRAINT, not decoration. Without use_rho_s it is fully
    # determined by (M_s, R_s, P) through Kepler's third law and the tuple the
    # caller passed would be silently ignored — so the published a/R* and its
    # uncertainty would vanish from the fit while appearing in the call. Convert
    # it to the stellar-density prior it actually is. rho_s is in SOLAR units.
    par = ParametrizationConfig(use_rho_s = use_rho_s)
    if use_rho_s && !haskey(pr, "rho_s")
        ρ  = _a_Rs_to_rho_s(Float64(a_Rs[1]), Float64(P))
        ρhi = _a_Rs_to_rho_s(Float64(a_Rs[1] + a_Rs[2]), Float64(P))
        ρlo = _a_Rs_to_rho_s(max(Float64(a_Rs[1] - a_Rs[2]), 1.001), Float64(P))
        σρ = max((ρhi - ρlo) / 2, 1e-4 * ρ)
        pr["rho_s"] = NormalPrior(ρ, σρ, max(ρ - 5σρ, 1e-4), ρ + 5σρ)
    end

    return Params(max_kplanet = 1, planet_modes = [mode],
                  instruments = InstrumentConfig(rv = inst_names, pm = String[]),
                  data = data, priors = pr, stability = :none,
                  parametrization = par,
                  M_s = Float64(M_s), R_s = Float64(R_s),
                  noise_models = Vector{NoiseModel}(noise_models),
                  transdim_noise = transdim_noise)
end


"Invert `rho_s_to_a_Rs`. rho_s in SOLAR units, P in days."
_a_Rs_to_rho_s(a_Rs::Real, P::Real) = a_Rs^3 / rho_s_to_a_Rs(1.0, P)^3
