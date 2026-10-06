# Framework-native obliquity fitting: RM velocities and/or Doppler tomography.
#
# An obliquity fit is an ordinary Nereus target -- `Params` + `Data` +
# `NereusTarget` -- run by any Nereus sampler. The pieces:
#
#   * RM velocities flow through `rv_log_likelihood`: `_decode_rm_state`
#     (rm.jl) adds the anomaly to the in-transit points, with the standard
#     noise machinery on the residuals.
#   * Residual maps flow through `tomogram_log_likelihood` (tomography.jl),
#     with per-night nuisances and a temporal kernel from the noise menu.
#   * λ, v sin i and the transit geometry are single slots read by both, which
#     is what makes a joint fit joint.
#
# This file assembles them: `obliquity_data` builds the `Data`,
# `obliquity_params` the `Params` in the standard configuration, and
# `obliquity_noise_models` / `obliquity_noise_menu` the per-night noise.
#
# ONE RM NIGHT = ONE INSTRUMENT. That is not a hack to reuse plumbing: each
# night genuinely needs its own systemic offset and its own jitter (different
# night, different conditions, often a different pipeline), which is exactly
# what an instrument is in this codebase. It also means the noise applies PER
# NIGHT, so one pulsating night can carry a different noise model from a
# quiet one instead of forcing a single compromise.
#
# GEOMETRY WITHOUT PHOTOMETRY. RM modes require PM by construction (the kernel
# needs the transit geometry), but `t_phot` may be empty: the transit term then
# contributes zero and b, r/R★ and a/R★ are set by their priors, which is what
# an obliquity fit with a published transit solution actually wants.

export RMNight, obliquity_data, obliquity_params, obliquity_noise_models,
       obliquity_noise_menu

"""
    obliquity_data(rm_nights; tomo_nights, t_ref) -> (Data, Vector{String})

Assemble RM velocity nights (and optionally tomographic maps) into a `Data`,
returning it with the RV instrument names in order. Each night becomes its own
instrument -- see the file header for why that is the right structure and not
a convenience. With no RM nights the `Data` holds the maps alone and the name
list is empty.

`t_ref` defaults to the mean RV epoch (or the median map epoch without
velocities); it only anchors the `Mo` time parametrization.
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
    tr = t_ref !== nothing ? Float64(t_ref) :
         !isempty(t) ? sum(t) / length(t) : nothing
    # Maps alone are data in their own right: no dummy velocity is added.
    d = isempty(rm_nights) ?
        Data(tomo = tomo_nights, t_ref = tr) :
        Data(t_rv = t, rv = rv, rv_err = err, rv_inst = inst,
             tomo = tomo_nights, t_ref = tr)
    return d, names
end

"""
    obliquity_noise_models(rv_names, tomo_tags; rv_noise = :sho,
                           tomo_noise = :matern) -> Vector{NoiseModel}

The standard per-night noise of an obliquity fit: on every RM night (RV
instrument in `rv_names`) a damped-harmonic-oscillator GP scoped to that night
(`CeleriteSHO(channel = :rv, instruments = [tag])`, parameters
`gp_log_S0_<tag>`, `gp_log_Q_<tag>`, `gp_log_omega0_<tag>`), and on every
residual map a Matérn-3/2 temporal kernel scoped to that map
(`MaternGP(channel = :tomo, instruments = [tag])`, parameters
`matern_sigma_tomo_<tag>` -- the amplitude A of the Kronecker GP -- and
`matern_rho_tomo_<tag>`, its time scale in days).

Per night because the pulsation phase and the amplitude a given night happens
to sample are not shared between epochs a year apart. `rv_noise = :white` and
`tomo_noise = :white` drop the correlated term (jitter / σ_n only);
`tomo_noise = :sho` uses an oscillator in time instead of the Matérn.
"""
function obliquity_noise_models(rv_names::AbstractVector{<:AbstractString},
                                tomo_tags::AbstractVector{<:AbstractString};
                                rv_noise::Symbol = :sho,
                                tomo_noise::Symbol = :matern)
    rv_noise in (:sho, :matern, :white) || throw(ArgumentError(
        "rv_noise must be :sho, :matern or :white; got :$rv_noise"))
    tomo_noise in (:matern, :sho, :white) || throw(ArgumentError(
        "tomo_noise must be :matern, :sho or :white; got :$tomo_noise"))
    nms = NoiseModel[]
    for tag in rv_names
        rv_noise === :sho    && push!(nms, CeleriteSHO(channel = :rv, instruments = [String(tag)]))
        rv_noise === :matern && push!(nms, MaternGP(channel = :rv, instruments = [String(tag)]))
    end
    for tag in tomo_tags
        tomo_noise === :matern && push!(nms, MaternGP(channel = :tomo, instruments = [String(tag)]))
        tomo_noise === :sho    && push!(nms, CeleriteSHO(channel = :tomo, instruments = [String(tag)]))
    end
    return nms
end

"""
    obliquity_noise_menu(rv_names, tomo_tags) -> (noise_models, toggleable, exclusion_groups)

Trans-dimensional noise selection for an obliquity fit, PER NIGHT. Each RM
night chooses between white jitter, a damped oscillator and a Matérn-3/2; each
residual map between white σ_n, a Matérn-3/2 and an oscillator in time. The
two kernels of a night are one exclusion group (at most one active), and
"none active" is the white null. Pass `noise_models` to `Params(...;
transdim_noise = true)` (or `obliquity_params(...; noise_models,
transdim_noise = true)`) and the other two to `TransDimConfig(noise = true,
...)`.
"""
function obliquity_noise_menu(rv_names::AbstractVector{<:AbstractString},
                              tomo_tags::AbstractVector{<:AbstractString})
    toggleable = NoiseModel[]
    groups = Vector{NoiseModel}[]
    for tag in rv_names
        g = NoiseModel[CeleriteSHO(channel = :rv, instruments = [String(tag)]),
                       MaternGP(channel = :rv, instruments = [String(tag)])]
        append!(toggleable, g); push!(groups, g)
    end
    for tag in tomo_tags
        g = NoiseModel[MaternGP(channel = :tomo, instruments = [String(tag)]),
                       CeleriteSHO(channel = :tomo, instruments = [String(tag)])]
        append!(toggleable, g); push!(groups, g)
    end
    return (noise_models = copy(toggleable), toggleable = toggleable,
            exclusion_groups = groups)
end

# A geometry argument is either a fixed value or a (mean, sd) Gaussian.
_obl_fixed_or_normal(x::Real, lo, hi) = FixedPrior(Float64(x))
function _obl_fixed_or_normal(x::Tuple{<:Real,<:Real}, lo, hi)
    μ, σ = Float64(x[1]), Float64(x[2])
    σ > 0 || throw(ArgumentError("a (mean, sd) prior needs sd > 0; got $x"))
    return NormalPrior(μ, σ, lo(μ, σ), hi(μ, σ))
end
_obl_fixed_or_normal(x::AbstractVector{<:Real}, lo, hi) =
    length(x) == 1 ? _obl_fixed_or_normal(x[1], lo, hi) :
    length(x) == 2 ? _obl_fixed_or_normal((x[1], x[2]), lo, hi) :
    throw(ArgumentError("expected a value or (mean, sd); got $x"))

_obl_center(x::Real) = Float64(x)
_obl_center(x) = Float64(x[1])

"""
    obliquity_params(data, rv_names; P, Tc, b, a_Rs, rr, vsini, K, ...) -> Params

Build the `Params` of an obliquity fit -- RM velocities, Doppler shadow, or
both -- in the STANDARD CONFIGURATION, every part of which is an option.

# Geometry and ephemeris
Each of `P`, `Tc`, `b`, `a_Rs`, `rr`, `vsini` (m/s) and `K` (m/s) is either a
number, which FIXES the parameter, or a `(mean, sd)` pair, which gives it a
Gaussian prior. An obliquity fit normally has no light curve of its own, so
the transit solution comes from a published fit and enters with that fit's
uncertainty, or fixed. `K = nothing` leaves the RV semi-amplitude on the
framework default.

- The ephemeris is parametrized by the transit time (`time = :Tc`), so a
  published `Tc ± σ` is a prior on the quantity it was measured as. Every RM
  night and every residual map is placed on the transit of THIS ephemeris
  nearest it, so the two observables can never see different Tc.
- a/R★ is sampled directly (`a_Rs_param = :a_Rs`, the parameter `a_Rs_k1`),
  so a published a/R★ ± σ is a Gaussian on a/R★ itself. `:rho_s` instead
  samples the stellar density (the published a/R★ is converted to an
  approximately equivalent density prior, the behaviour before this option
  existed); `:kepler` derives a/R★ from `M_s`, `R_s` and P.
- `ecc = 0.0` (default) fixes a circular orbit; `ecc = :free` samples it.
- λ (`lambda_k1`) is uniform on the full circle and WRAPPED (`lambda_prior =
  :wrapped`, a `WrappedUniformPrior(-π, π)`): no sampler sees a wall at the
  seam. `:bounded` restores `UniformPrior(-π, π)`.

# The RM anomaly
- `sigma0`: the MEASURED out-of-transit CCF width σ₀ (m/s) of each RM night --
  a `Dict(tag => σ0)` or the `RMNight`s themselves. With it the anomaly is the
  ARoME kernel (Boué+2013) with each night's own σ₀ and the sub-planet width
  derived, β_p² = σ₀² − (0.5503 v sin i)², so the amplitude is fixed by
  v sin i, R_p/R★ and the limb darkening with nothing free to rescale it.
  Without it, `arome = true` gives ARoME with one free (σ_ccf, β_p) pair, and
  the default is the flux-weighted kernel (`RVPM_RM`).
- `beta_p_floor` (m/s, default 2000): lower limit on the derived β_p, the width
  of the local line (see `ObliquityConfig`).
- `occultation` (`:disc` default, or `:point`): the occulted flux fraction --
  the exact disc overlap, or a point planet at its centre (see
  `ObliquityConfig`).
- `ld = (u1, u2)`: quadratic limb darkening of the spectroscopic band, fixed
  (override `u1_spec`/`u2_spec` in `priors` to fit them). Default: none, i.e.
  the first photometric band's, or a uniform disc.

# The Doppler shadow
- `shared_alpha = true` fits one shadow amplitude for all maps.
- Per-map priors (standard): α U(0, 20), σ_line U(2, 25) km/s, ℓ_v logU(1, 60)
  km/s, σ_n logU(1e-6, 1), Kronecker amplitude A logU(1e-6, 1), ℓ_t logU(0.05,
  12) h -- `matern_rho_tomo_<tag>` is in days.

# Noise
`noise_models = nothing` (default) builds the standard per-night noise
(`obliquity_noise_models`; `rv_noise`, `tomo_noise` select it). RM-night
priors (standard): S0 logU(1e-2, 1e12), Q logU(0.2, 100), ω0 logU(0.2, 60)
rad/d, jitter logU(0.1, 3162) m/s; each night's offset is uniform on its mean
± the Nereus γ rule (`_gamma_default_bounds`). Pass `noise_models` explicitly
(with `transdim_noise = true` for a menu from `obliquity_noise_menu`) to
choose otherwise.

`priors` overrides any prior by name, last. `pm_names` names photometric
instruments when the `Data` also carries a light curve.
"""
function obliquity_params(data::Data, inst_names::Vector{String};
                          P, Tc,
                          b, a_Rs,
                          rr = (0.1, 0.02),
                          vsini = (10_000.0, 5_000.0),
                          K = nothing,
                          ecc = 0.0,
                          lambda_prior::Symbol = :wrapped,
                          sigma0 = nothing,
                          beta_p_floor::Real = DEFAULT_BETA_P_FLOOR,
                          occultation::Symbol = :disc,
                          ld = nothing,
                          shared_alpha::Bool = false,
                          rv_noise::Symbol = :sho,
                          tomo_noise::Symbol = :matern,
                          noise_models::Union{Nothing,Vector{<:NoiseModel}} = nothing,
                          transdim_noise::Bool = false,
                          pm_names::Vector{String} = String[],
                          a_Rs_param::Symbol = :a_Rs,
                          use_rho_s::Union{Nothing,Bool} = nothing,
                          M_s::Real = 1.0, R_s::Real = 1.0,
                          arome::Union{Nothing,Bool} = nothing,
                          priors::AbstractDict = Dict{String,PriorSpec}())
    # --- back-compatible keywords -----------------------------------------
    use_rho_s === true && (a_Rs_param = :rho_s)
    a_Rs_param in (:a_Rs, :rho_s, :kepler) || throw(ArgumentError(
        "a_Rs_param must be :a_Rs, :rho_s or :kepler; got :$a_Rs_param"))
    lambda_prior in (:wrapped, :bounded) || throw(ArgumentError(
        "lambda_prior must be :wrapped or :bounded; got :$lambda_prior"))
    (ecc == 0.0 || ecc === :free) || throw(ArgumentError(
        "ecc must be 0.0 (circular, fixed) or :free; got $ecc"))

    s0 = sigma0 === nothing ? Dict{String,Float64}() :
         sigma0 isa AbstractVector{RMNight} ?
            Dict{String,Float64}(n.tag => n.σ0 for n in sigma0) :
            Dict{String,Float64}(String(k) => Float64(v) for (k, v) in sigma0)
    use_arome = arome === nothing ? !isempty(s0) : arome
    (!isempty(s0) && !use_arome) && throw(ArgumentError(
        "sigma0 is only read by the ARoME kernel; it cannot be combined with arome = false"))
    has_rv_data = !isempty(inst_names)
    mode = !has_rv_data ? PM_DT : (use_arome ? RVPM_RM_A : RVPM_RM)

    tomo_tags = [nt.tag for nt in data.tomo]
    nms = noise_models === nothing ?
        obliquity_noise_models(inst_names, tomo_tags; rv_noise, tomo_noise) :
        Vector{NoiseModel}(noise_models)

    obl = ObliquityConfig(sigma0 = s0, beta_p_floor = beta_p_floor,
                          occultation = occultation, spec_ld = ld !== nothing,
                          shared_alpha = shared_alpha)

    # --- priors: the standard configuration, then the caller's -------------
    pr = Dict{String,PriorSpec}()
    Pc = _obl_center(P)
    pr["P_k1"] = _obl_fixed_or_normal(P, (μ, σ) -> max(μ - 50σ, 0.0), (μ, σ) -> μ + 50σ)
    # Transit time as the time anchor. A Normal no wider than a quarter period
    # each side, so its window can never hold two copies of the orbit.
    pr["Tc_k1"] = _obl_fixed_or_normal(Tc, (μ, σ) -> μ - min(20σ, Pc / 4),
                                           (μ, σ) -> μ + min(20σ, Pc / 4))
    pr["b_k1"]  = _obl_fixed_or_normal(b,  (μ, σ) -> 0.0, (μ, σ) -> 2.0)
    pr["rr_k1"] = _obl_fixed_or_normal(rr, (μ, σ) -> 0.0, (μ, σ) -> 0.5)
    pr["v_sin_i_star"] = _obl_fixed_or_normal(vsini, (μ, σ) -> 100.0,
                                              (μ, σ) -> 300_000.0)
    if K !== nothing
        has_rv_data || throw(ArgumentError("K is given but there are no RM nights"))
        pr["K_k1"] = _obl_fixed_or_normal(K, (μ, σ) -> 0.0, (μ, σ) -> μ + 50σ)
    end
    if ecc == 0.0
        pr["sesinw_k1"] = FixedPrior(0.0)
        pr["secosw_k1"] = FixedPrior(0.0)
    end
    pr["lambda_k1"] = lambda_prior === :wrapped ? WrappedUniformPrior(-π, π) :
                                                  UniformPrior(-π, π)
    if a_Rs_param === :a_Rs
        pr["a_Rs_k1"] = _obl_fixed_or_normal(a_Rs, (μ, σ) -> 1.0, (μ, σ) -> μ + 50σ)
    elseif a_Rs_param === :rho_s
        # a/R★ ± σ converted to a stellar-density prior (rho_s in SOLAR units),
        # truncated at ±5σ -- the pre-option behaviour.
        aμ = _obl_center(a_Rs)
        aσ = a_Rs isa Real ? 0.0 : Float64(a_Rs[2])
        ρ  = _a_Rs_to_rho_s(aμ, Pc)
        if aσ == 0
            pr["rho_s"] = FixedPrior(ρ)
        else
            ρhi = _a_Rs_to_rho_s(aμ + aσ, Pc)
            ρlo = _a_Rs_to_rho_s(max(aμ - aσ, 1.001), Pc)
            σρ  = max((ρhi - ρlo) / 2, 1e-4 * ρ)
            pr["rho_s"] = NormalPrior(ρ, σρ, max(ρ - 5σρ, 1e-4), ρ + 5σρ)
        end
    end
    for tag in inst_names
        # RM nights: the paper-standard oscillator and jitter boxes, only for
        # the parameters that exist (a caller's own noise models may differ).
        pr["sigma_$tag"] = LogUniformPrior(0.1, 10^3.5)
    end
    for m in nms
        (m isa CeleriteSHO && m.channel === :rv) || continue
        sfx = _gp_suffix(m)
        pr["gp_log_S0$sfx"]     = UniformPrior(log(1e-2), log(1e12))
        pr["gp_log_Q$sfx"]      = UniformPrior(log(0.2), log(100.0))
        pr["gp_log_omega0$sfx"] = UniformPrior(log(0.2), log(60.0))
    end
    for tag in tomo_tags
        pr["tomo_ell_v_$tag"] = LogUniformPrior(1.0, 60.0)
    end
    if ld !== nothing
        length(ld) == 2 || throw(ArgumentError("ld must be (u1, u2); got $ld"))
        pr["u1_spec"] = FixedPrior(Float64(ld[1]))
        pr["u2_spec"] = FixedPrior(Float64(ld[2]))
    end
    for (k, v) in priors
        pr[String(k)] = v
    end

    par = ParametrizationConfig(time = :Tc, use_rho_s = a_Rs_param === :rho_s,
                                sample_a_Rs = a_Rs_param === :a_Rs)
    return Params(max_kplanet = 1, planet_modes = [mode],
                  instruments = InstrumentConfig(rv = inst_names, pm = pm_names),
                  data = data, priors = pr, stability = :none,
                  parametrization = par,
                  M_s = Float64(M_s), R_s = Float64(R_s),
                  noise_models = nms, transdim_noise = transdim_noise,
                  obliquity = obl)
end


"Invert `rho_s_to_a_Rs`. rho_s in SOLAR units, P in days."
_a_Rs_to_rho_s(a_Rs::Real, P::Real) = a_Rs^3 / rho_s_to_a_Rs(1.0, P)^3
