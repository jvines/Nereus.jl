# Auto-generated default priors from data bounds.
# Follows EMPEROR conventions: all priors are data-driven, user overrides.

using Statistics: std, mean, median

"""
    _rv_activity_scatter(data, instruments, only=nothing) -> Float64

Max per-instrument RV std (floored at 10 m/s) — the activity/jitter scale, as
opposed to `rv_max = max|absolute RV|` (the systemic velocity, ~km/s for
absolute-RV data). Amplitude-like priors (GP σ, celerite amp) must scale with
THIS, not rv_max, or their ceilings become non-constraining (e.g. a 31 km/s
absolute RV → a 93 km/s GP-amplitude ceiling).
"""
# Robust (MAD-based) per-instrument RV scatter. Uses the median absolute
# deviation, NOT std: a handful of activity/instrumental outliers inflate std
# by factors of a few, which loosens every GP amplitude cap (∝ 3·σ_act) enough
# for the GP to become a free interpolator (HD 18599: outlier-inflated σ let a
# bounded SHO still reach 34 m/s and eat the planet). MAD is outlier-resistant.
_mad_scatter(x) = isempty(x) ? 0.0 : 1.4826 * median(abs.(x .- median(x)))
function _rv_activity_scatter(data, instruments, only::Union{Nothing,AbstractVector} = nothing)
    stds = Float64[]
    for idx in 1:length(instruments.rv_names)
        only === nothing || instruments.rv_names[idx] in only || continue
        mask = data.rv_inst .== idx
        count(mask) > 1 && push!(stds, _mad_scatter(data.rv[mask]))
    end
    isempty(stds) ? max(_mad_scatter(data.rv), 10.0) : max(maximum(stds), 10.0)
end

"""
Activity scatter seen by the instruments a noise model actually covers.

A per-instrument GP declared `instruments=["FEROS"]` must have its amplitude
ceiling set by FEROS, not by whichever instrument in the dataset happens to be
the loudest — otherwise adding a noisy instrument to a fit silently loosens the
prior on a GP that never touches it, and two fits that should be identical are
not. When the model covers everything (`instruments` empty), this is the
dataset-wide scatter as before.
"""
_model_activity_scatter(m, data, instruments) =
    _rv_activity_scatter(data, instruments,
        (hasproperty(m, :instruments) && m.instruments !== nothing &&
         !isempty(m.instruments)) ? m.instruments : nothing)

"""
    default_priors(config, data) -> Dict{String, PriorSpec}

Generate sensible default priors for all model parameters based on
the data and model configuration. Follows EMPEROR conventions.

The user can override any prior by passing it in the `priors` kwarg
to `Params()`.
"""
function default_priors(config::ParamsConfig, data::Data)
    dic = Dict{String, PriorSpec}()
    parametrization = config.parametrization
    instruments = config.instruments
    has_rv = length(instruments.rv_names) > 0
    has_pm = length(instruments.pm_names) > 0
    # Astrometry-bearing fit? In RVAS/RVPMAS the Keplerian RV over a short
    # observed arc of a long-period orbit is a large near-constant offset
    # (can be ~km/s) that `gamma` must absorb on top of the (arbitrary,
    # relative) RV zero-point — so the RV-only assumption gamma ≈ data_mean
    # breaks and the default `mean ± 3·scatter` bound excludes the correct
    # wide-orbit mode (HD 159062 B needs gamma ≈ 1480 m/s). Widen gamma then.
    has_astrom = any(f -> hasproperty(data, f) && getproperty(data, f) !== nothing,
                     (:relastrom, :hgca, :iad, :gost, :g23h, :gaia_dr3))

    # --- Data-derived bounds ---
    if has_rv && length(data.t_rv) > 0
        rv_max = max(maximum(abs, data.rv), 10.0)   # floor at 10 m/s
        # RV scatter: max per-instrument std (not global, which includes offsets)
        rv_scatter = _rv_activity_scatter(data, instruments)
        t_min_rv = minimum(data.t_rv)
        t_max_rv = maximum(data.t_rv)
        baseline_rv = max(t_max_rv - t_min_rv, 100.0)
        dt_rv = diff(sort(data.t_rv))
        min_cadence_rv = length(dt_rv) > 0 ? minimum(dt_rv[dt_rv .> 0]) : 1.0
    else
        rv_max = 100.0
        rv_scatter = 100.0
        baseline_rv = 1000.0
        min_cadence_rv = 1.0
        t_min_rv = 0.0
        t_max_rv = 1000.0
    end
    if has_pm && length(data.t_phot) > 0
        t_min_pm = minimum(data.t_phot)
        t_max_pm = maximum(data.t_phot)
        baseline_pm = max(t_max_pm - t_min_pm, 100.0)
        dt_pm = diff(sort(data.t_phot))
        min_cadence_pm = length(dt_pm) > 0 ? minimum(dt_pm[dt_pm .> 0]) : 1.0 / (60 * 24)
    else
        t_min_pm = 0.0
        t_max_pm = 1000.0
        baseline_pm = 1000.0
        min_cadence_pm = 1.0 / (60 * 24)
    end

    baseline = max(baseline_rv, baseline_pm, 100.0)  # floor at 100 days
    t_min_cand = min(length(data.t_rv) > 0 ? t_min_rv : Inf,
                     length(data.t_phot) > 0 ? t_min_pm : Inf)
    t_max_cand = max(length(data.t_rv) > 0 ? t_max_rv : -Inf,
                     length(data.t_phot) > 0 ? t_max_pm : -Inf)
    # Fallback if no actual data
    t_min = isfinite(t_min_cand) ? t_min_cand : 0.0
    t_max = isfinite(t_max_cand) ? t_max_cand : 1000.0

    # --- n_p ---
    dic["n_p"] = FixedPrior(Float64(config.max_kplanet))

    # --- Planetary parameters ---
    for k in 1:config.max_kplanet
        modes = config.planet_modes[k]

        # First slot: P (days) under :K_driven / :M_sec_driven, or
        # a (AU) under :a_driven (orvara convention).
        if parametrization.mass === :a_driven
            # Log-flat prior on a, 0.5–1000 AU (orvara default).
            dic["a_k$k"] = LogUniformPrior(0.5, 1000.0)
        else
            dic["P_k$k"] = LogUniformPrior(0.1, 3 * baseline)
        end

        # Second slot: K (m/s) under :K_driven, M_sec (M_sun)
        # under :M_sec_driven and :a_driven.
        #
        # SB2 binary: TWO amplitudes K_A, K_B (m/s), always sampled
        # directly regardless of the global mass parametrization. Broad
        # data-driven ceiling — the binary reflex is typically ≫ a planet
        # K (often km/s), and rv_scatter already pools the double-lined
        # RV spread. Floored at 1 km/s so a modest-amplitude binary is
        # still covered.
        if SB_SOURCE in modes
            K_upper = max(3 * rv_scatter, 1000.0)
            K_knee  = min(100.0, K_upper / 2)
            dic["K_A_k$k"] = ModJeffreysPrior(K_knee, K_upper)
            dic["K_B_k$k"] = ModJeffreysPrior(K_knee, K_upper)
        elseif RV_SOURCE in modes
            if parametrization.mass === :K_driven
                K_upper = max(3 * rv_scatter, 200.0)
                dic["K_k$k"] = ModJeffreysPrior(min(100.0, K_upper / 2), K_upper)
            else  # :M_sec_driven or :a_driven — broad LogUniform on companion mass
                # 1 Earth mass to 0.5 M_sun: covers terrestrials → BDs → low-mass stars.
                dic["M_sec_k$k"] = LogUniformPrior(3.0e-6, 0.5)
            end
        end

        # Eccentricity parametrization
        if parametrization.ew === :sesinw
            dic["sesinw_k$k"] = UniformPrior(-1.0, 1.0)
            dic["secosw_k$k"] = UniformPrior(-1.0, 1.0)
        elseif parametrization.ew === :esinw
            dic["esinw_k$k"] = UniformPrior(-1.0, 1.0)
            dic["ecosw_k$k"] = UniformPrior(-1.0, 1.0)
        else  # :ew
            dic["ecc_k$k"] = NormalPrior(0.0, 0.3, 0.0, 1.0)
            dic["w_k$k"] = UniformPrior(-π, π)
        end

        # Time anchor
        if parametrization.time === :Mo
            dic["Mo_k$k"] = UniformPrior(0.0, 2π)
        elseif parametrization.time === :Tp
            dic["Tp_k$k"] = UniformPrior(t_min, t_max)
        else  # :Tc
            dic["Tc_k$k"] = UniformPrior(t_min, t_max)
        end

        # Transit geometry (PM planets only)
        if PM_SOURCE in modes
            if parametrization.geom === :b_rr
                dic["rr_k$k"] = UniformPrior(0.001, 1.0)
                dic["b_k$k"] = UniformPrior(0.0, 1.0)
            else  # :r1r2
                dic["r1_k$k"] = UniformPrior(0.0, 1.0)
                dic["r2_k$k"] = UniformPrior(0.0, 1.0)
            end

            # a/R* per planet — deferred until per-planet a_Rs is in the layout
            # if !parametrization.use_rho_s
            #     dic["a_k$k"] = LogUniformPrior(0.1, 100.0)
            # end
        end

        # Astrometry geometry (AS planets only)
        # EMPEROR II / orvara conventions: isotropic prior on
        # orbital orientation, i.e. cos(i) ~ Uniform(-1, 1) which
        # translates to p(i) = sin(i)/2 on [0, π]. Critical for
        # astrometric fits — without it the chain can find face-on
        # degenerate modes.
        #
        # For the full RV+PM+AS combination (RVPMASBlock) the
        # inclination is *derived* from the transit impact parameter
        # `b` and is not a sampled slot — skip the inc prior; the
        # astrometric isotropy is already enforced by the b-prior plus
        # the geometric relation.
        if AS_SOURCE in modes
            if !(RV_SOURCE in modes && PM_SOURCE in modes)
                dic["inc_k$k"] = SinePrior()
            end
            dic["Omega_k$k"] = UniformPrior(0.0, 2π)
        end
    end

    # --- Stellar density ---
    if parametrization.use_rho_s
        dic["rho_s"] = LogUniformPrior(0.001, 100.0)
    end

    # --- RV instrumental (sharing-aware, per-instrument bounds) ---
    # Use per-instrument mean + scatter for gamma bounds (handles mixed
    # zero points: differential + absolute RVs in the same dataset).
    # K and sigma scale with the RV scatter, not the absolute offset.
    sh = config.sharing
    for (label, idxs) in _instrument_groups(instruments.rv_names,
                                             get(sh, :gamma, nothing))
        # γ-marginalized path: the per-instrument systemic offset is
        # integrated analytically in closed form (see
        # `_rv_ll_gamma_marginalized`), so it is NOT a sampled parameter.
        # Freeze the slot at 0.0 — the marginalized likelihood ignores
        # the stored value; the conditional point estimate is reported
        # post-hoc via `conditional_gamma`.
        if parametrization.marginalize_gamma
            dic["gamma_$label"] = FixedPrior(0.0)
            continue
        end
        # Per-instrument: center on mean, widen by 3× scatter
        inst_rv = Float64[]
        for idx in idxs
            mask = data.rv_inst .== idx
            append!(inst_rv, data.rv[mask])
        end
        if length(inst_rv) > 1
            mu = mean(inst_rv)
            spread = max(3 * std(inst_rv), 3 * rv_scatter, 100.0)
            has_astrom && (spread = max(spread, 3.0e4))   # absorb orbital reflex (RVAS); ±30 km/s covers a≳1 AU stellar companions
            dic["gamma_$label"] = UniformPrior(mu - spread, mu + spread)
        elseif length(inst_rv) == 1
            mu = inst_rv[1]
            spread = max(3 * rv_scatter, 100.0)
            has_astrom && (spread = max(spread, 3.0e4))
            dic["gamma_$label"] = UniformPrior(mu - spread, mu + spread)
        else
            dic["gamma_$label"] = UniformPrior(-3 * rv_max, 3 * rv_max)
        end
    end
    for (label, _) in _instrument_groups(instruments.rv_names,
                                          get(sh, :sigma, nothing))
        # Vines+ 2023 Table 7: σ_RV ~ N(5, 5²). Truncated to (0, 50] m/s
        # since σ must be positive and the upper bound is pushed beyond
        # the 5 m/s mode by enough sigmas that data drives the tail.
        # The previous ModJeffreysPrior peaked near 0 and (per Vines'
        # comment about ARMA absorbing signal at low cadence) added
        # mass to noise-eats-signal modes that the paper avoided.
        dic["sigma_$label"] = NormalPrior(5.0, 5.0, 0.0, 50.0)
    end

    # --- PM instrumental (sharing-aware) ---
    for (label, _) in _instrument_groups(instruments.pm_names,
                                          get(sh, :pm_offset, nothing))
        dic["offset_$label"] = NormalPrior(0.0, 0.005, -1.0, 1.0)
    end
    for (label, _) in _instrument_groups(instruments.pm_names,
                                          get(sh, :pm_jitter, nothing))
        # Photometry jitter is in FLUX units (around 1.0), not m/s.
        # Typical TESS / Kepler post-detrending scatter ~ 1e-4 to 1e-2.
        # ModJeffreys with knee at 1e-4 transitions from ~uniform below
        # to Jeffreys above; upper bound 1e-1 catches the noisy stars.
        dic["jitter_$label"] = ModJeffreysPrior(1e-4, 1e-1)
    end
    # Third-light dilution D = L_diluting / L_total (per photometric band).
    # Default 0 (no third light) for a normal fit. When the system has an
    # SB2 binary, the circumprimary planet's transit IS diluted by the
    # non-transited secondary — free the dilution so the depth isn't
    # biased. It stays per-PM-instrument (band-specific); the user can
    # override with a CCF-informed prior (L_B/L_A → D transformed to the
    # transit band). This is the transit-band twin of the astrometric
    # `f_light`; the two are separate params because L_B/L_A is
    # wavelength-dependent.
    has_sb_binary = any(has_sb, config.planet_modes)
    for (label, _) in _instrument_groups(instruments.pm_names,
                                          get(sh, :pm_dilution, nothing))
        dic["dilution_$label"] = has_sb_binary ? UniformPrior(0.0, 0.9) :
                                                  FixedPrior(0.0)
    end

    # --- Astrometry: parallax (mas) and primary mass (M_sun) ---
    # If any planet has AS as a data source, we have `plx` and `M_pri`
    # slots in the layout. The default plx prior is Gaussian centered
    # on the HGCA parallax. The default M_pri prior is a FixedPrior at
    # config.M_s (back-compat with the pre-sampling-M_pri code); the
    # user passes a NormalPrior to enable propagation of stellar-mass
    # uncertainty into the companion mass.
    if any(AS_SOURCE in m for m in config.planet_modes)
        if data.hgca !== nothing
            σ = max(data.hgca.plx_err, 1e-3)
            μ = data.hgca.plx
            lo = max(0.001, μ - 10 * σ)
            hi = μ + 10 * σ
            dic["plx"] = NormalPrior(μ, σ, lo, hi)
        else
            dic["plx"] = LogUniformPrior(0.1, 1000.0)
        end
        # Default: fix M_pri at config.M_s. Override via priors[].
        if !isnan(config.M_s)
            dic["M_pri"] = FixedPrior(config.M_s)
        else
            # No stellar mass set: very broad prior (user must override)
            dic["M_pri"] = LogUniformPrior(0.05, 5.0)
        end
    end

    # --- Trend ---
    if config.trend_order >= 1
        dic["dvdt"] = UniformPrior(-1.0, 1.0)
    end
    if config.trend_order >= 2
        dic["d2vdt2"] = UniformPrior(-0.01, 0.01)
    end

    # --- TTV (per-transit free offsets, per planet with TTV source) ---
    for k in 1:config.max_kplanet
        has_ttv(config.planet_modes[k]) || continue
        n_ttv_k = get(config.ttv_n_transits, k, 0)
        n_ttv_k > 0 || continue
        # σ_ttv defaults to ~1 day, bounded by ±max(0.5 P_prior, 1 day).
        # Users tighten with NormalPrior(0, minutes, ±tight_bound) when
        # they have a-priori knowledge of TTV amplitude.
        sigma_ttv = 1.0   # days
        bound     = 10.0  # days
        for i in 1:n_ttv_k
            dic["ttv_k$(k)_t$i"] = NormalPrior(0.0, sigma_ttv, -bound, bound)
        end
    end

    # Doppler tomography, per residual map.
    #
    # alpha is the shadow amplitude in units of the map's own normalisation,
    # so its scale is set by how the CCFs were built, not by the star. It is
    # bounded BELOW at zero: a negative shadow is an emission feature, and
    # letting alpha go negative lets the mirrored track win the fit and be
    # reported as a detection (the same failure the signed peak selection in
    # tomogram_pooled exists to prevent). The upper bound matches the
    # alpha_max default that shadow_bayes_factor's Savage-Dickey ratio is
    # normalised against — change one and you must change the other or the
    # Bayes factor is silently rescaled.
    #
    # sigma_line is the LOCAL line width (intrinsic ⊗ instrumental), not
    # v·sin i. Deriving it from v·sin i is the same mistake as deriving
    # sigma_ccf that way above.
    for nt in (data === nothing ? TomoNight[] : data.tomo)
        dic["tomo_alpha_$(nt.tag)"]      = UniformPrior(0.0, 20.0)
        dic["tomo_sigma_line_$(nt.tag)"] = UniformPrior(2.0, 25.0)   # km/s
        # Velocity correlation length: at least a CCF bin, at most the line.
        dic["tomo_ell_v_$(nt.tag)"]      = LogUniformPrior(0.5, 60.0)  # km/s
    end

    # --- Obliquity: RM / tomography (:RM*) and gravity darkening (:GD) ---
    # Both measure λ and both need V·sin(i_*), so the gate must match the
    # layout's. Gating on RM alone left a :GD-only model with allocated slots
    # and no priors -- which fails loudly at build time, but only because
    # `_build_layout` checks; it is the same mismatch that would silently
    # mis-gate anything not checked.
    if any(has_any_rm(m) || has_gd(m) for m in config.planet_modes)
        # Stellar V·sin(i_*): LogUniform spanning slow rotators (0.5
        # km/s, typical M dwarf v_sin_i floor) to very fast rotators
        # (100 km/s, A-star regime). User overrides with NormalPrior
        # when spectroscopic v_sin_i is measured.
        dic["v_sin_i_star"] = LogUniformPrior(500.0, 100_000.0)   # m/s
        # Stellar inclination for gravity darkening. Uniform in [0, pi/2]: an
        # ISOTROPIC prior would be uniform in cos(i_star), but that piles prior
        # mass at the equator-on end where the gravity-darkening signal vanishes,
        # so it biases toward a non-detection. Uniform in the angle is the honest
        # weakly-informative choice here; use cos-uniform explicitly if an
        # isotropic population prior is wanted.
        if any(has_gd(m) for m in config.planet_modes)
            dic["i_star"] = UniformPrior(0.0, pi / 2)
        end
        # The per-band `gd_beta_<INST>` defaults are set in `_build_layout`,
        # where the photometric instrument names are known. Bolometric (0.25)
        # unless the user supplies a band-corrected value from `gd_beta_band`.

        # ARoME CCF kernel line widths (Boué+2013 Eq. 15), only allocated when
        # some planet uses :RM_A. BOTH should normally be overridden with a
        # NormalPrior from the data: σ₀ is the Gaussian dispersion of the
        # co-added out-of-transit CCF, β_p the local line width (intrinsic ⊗
        # instrumental). Deriving σ₀ from v·sin i instead of measuring it is what
        # produces Boué+2013's quoted v·sin i ≲ 20 km/s validity ceiling.
        if any(has_rm_a(m) for m in config.planet_modes)
            dic["sigma_ccf"] = LogUniformPrior(500.0, 100_000.0)  # m/s
            dic["beta_p"]    = LogUniformPrior(200.0,  50_000.0)  # m/s
        end

        for k in 1:config.max_kplanet
            if has_any_rm(config.planet_modes[k]) || has_gd(config.planet_modes[k])
                # Sky-projected obliquity uniform on [-π, π]. User
                # tightens to NormalPrior(0, π/8) for aligned systems.
                dic["lambda_k$k"] = UniformPrior(-π, π)
            end
        end
    end

    # --- SB2 luminous-companion photocenter (f_light) ---
    # Secondary light fraction L_B/(L_A+L_B) in the astrometric band.
    # Only present when an SB2 binary also has astrometry. Broad default
    # [0, 0.5): a main-sequence secondary is under-luminous for its mass
    # (f_light < f_mass), but the user informs this from the CCF flux
    # ratio + a color transform to the astrometric band.
    if any(m -> has_sb(m) && has_as(m), config.planet_modes)
        dic["f_light"] = UniformPrior(0.0, 0.5)
    end

    # --- Noise models ---
    for nm in config.noise_models
        _default_noise_priors!(dic, nm, instruments, data,
                                rv_max, min_cadence_rv)
    end

    return dic
end

# ActivityDecorrelation coefficient priors — SCALE-ADAPTIVE (automatic).
#
# NOTE since indicators became normalised at Data construction (median-subtracted,
# divided by their RMS per instrument — Vines et al. 2023 Table 7 footnote), the
# regressor has unit RMS and C is directly the m/s amplitude of the activity term.
# The adaptive bound below is therefore belt-and-braces rather than load-bearing:
# it still derives ±κ·(RV scatter / indicator scatter) per instrument, which now
# evaluates to roughly κ·σ_RV/0.67 on normalised input.
#
# THIS BOUND IS A SCIENCE CHOICE, not housekeeping. On HD 18599 the BIS sits at
# P_rot/2 ~ P_orb, so a coefficient large enough to reach the measured BIS-RV
# slope (-0.83) absorbs the planet: bounded near the published |C| <= 3 the fit
# returns K = 11.9 m/s, bounded at the RV amplitude it returns 5.8. Vines et al.
# Table 9 reports |C| <= 2.1, i.e. roughly 10x tighter than κ=3 allows here, and
# says so in words: "a straightforward decorrelation is not possible since it
# would remove the candidate signal". κ=3 mirrors EMPEROR's "scale indicator to
# the RV peak, C in (-1,1)" and is deliberately permissive; pass an explicit
# prior when the indicator is near-degenerate with the orbital period.
function _default_noise_priors!(dic, m::ActivityDecorrelation,
                                 instruments, data, rv_max, min_cadence)
    suf = _ad_suffix(m); κ = 3.0
    _rvsc(iidx) = (msk = data.rv_inst .== iidx;
                   count(msk) > 1 ? max(_mad_scatter(data.rv[msk]), 1.0) : 10.0)
    _rvsc_all() = max(_mad_scatter(data.rv), 1.0)
    _bound(vals, sc) = (isc = _mad_scatter(filter(isfinite, vals)); κ * sc / (isc > 0 ? isc : 1.0))
    _bound_inst(vals, iidx) = _bound(vals[data.rv_inst .== iidx], _rvsc(iidx))
    dv(ind) = (isdefined(data, :indicator_derivs) && haskey(data.indicator_derivs, ind)) ?
              data.indicator_derivs[ind] : data.indicators[ind]
    if m.per_instrument
        for (ins_idx, ins) in enumerate(instruments.rv_names)
            for ind in m.indicators
                haskey(data.indicators, ind) || continue
                ind_vals = data.indicators[ind]
                inst_mask = data.rv_inst .== ins_idx
                count(i -> inst_mask[i] && isfinite(ind_vals[i]), eachindex(ind_vals)) > 0 || continue
                b = _bound_inst(ind_vals, ins_idx)
                dic["C_$(ind)_$(ins)$suf"] = UniformPrior(-b, b)
                if m.derivative
                    bd = _bound_inst(dv(ind), ins_idx)
                    dic["Cdot_$(ind)_$(ins)$suf"] = UniformPrior(-bd, bd)
                end
            end
        end
    else
        for ind in m.indicators
            haskey(data.indicators, ind) || continue
            isempty(filter(isfinite, data.indicators[ind])) && continue
            b = _bound(data.indicators[ind], _rvsc_all())
            dic["C_$(ind)$suf"] = UniformPrior(-b, b)
            if m.derivative
                bd = _bound(dv(ind), _rvsc_all())
                dic["Cdot_$(ind)$suf"] = UniformPrior(-bd, bd)
            end
        end
    end
end

function _default_noise_priors!(dic, m::ActivityJitter,
                                 instruments, data, rv_max, min_cadence)
    for ins in instruments.rv_names
        # Key on indicator too — must match noise_param_names(::ActivityJitter)
        # so the prior lands on the right layout slot when ≥2 jitter models
        # (one per indicator) target the same instrument.
        dic["jit_base_$(m.indicator)_$(ins)"] = UniformPrior(0.0, 3 * rv_max)
        dic["jit_act_$(m.indicator)_$(ins)"]  = UniformPrior(-50.0, 50.0)
    end
end

function _default_noise_priors!(dic, m::IndicatorFloor,
                                 instruments, data, rv_max, min_cadence)
    _chan_scale(ch) = begin
        name = String(ch)
        if data !== nothing && haskey(data.indicators, name) &&
           !isempty(data.indicators[name])
            v = data.indicators[name]
            max(1.4826 * median(abs.(v .- median(v))), 1e-3)
        else
            1.0
        end
    end
    if m.kernel === :qp
        # Quasi-periodic GP floor — SAME kernel-hyperparameter scales as
        # ActivityGP (so the floor's indicator evidence is comparable to
        # AGP's indicator marginal and the AD↔AGP terms ≈ cancel).
        dic["ind_floor_period"]   = LogUniformPrior(1.0, 365.0)
        dic["ind_floor_lambda_e"] = LogUniformPrior(1.0, 1000.0)
        dic["ind_floor_lambda_p"] = LogUniformPrior(0.25, 10.0)
        for ch in m.channels
            scale = _chan_scale(ch)
            # GP amplitude (channel units) + a white jitter floor.
            dic["ind_floor_$(ch)_amp"] = LogUniformPrior(0.05 * scale, 20.0 * scale)
            dic["ind_floor_$(ch)_jit"] = LogUniformPrior(1e-3 * scale, 5.0 * scale)
        end
        return
    end
    # :white — per-channel white floor σ, scaled by the channel's robust
    # spread (brackets "white scatter at the data's own level").
    for ch in m.channels
        dic["ind_floor_$(ch)"] = LogUniformPrior(0.05 * _chan_scale(ch), 20.0 * _chan_scale(ch))
    end
end

function _default_noise_priors!(dic, m::MAModel,
                                 instruments, data, rv_max, min_cadence)
    s = _channel_suffix(m.channel)
    chan_inst = m.channel === :rv ? instruments.rv_names : instruments.pm_names
    # Channel-specific β prior:
    #   :rv   → U(0, 10) d, matching Vines+ 2023 Table 7. Capping β at
    #           ~2× a 4-d planet period prevents the MA term from
    #           absorbing low-frequency Keplerian power on sparsely
    #           sampled RVs (the paper notes ARMA underperforms at low
    #           cadence for exactly this reason, §4.2).
    #   :phot → LogUniform(min_cadence, 365 d), retained for high-cadence
    #           photometry where ARMA actually has the temporal
    #           resolution to fit short-timescale correlated noise.
    if m.channel === :rv
        # Tuomi+ 2013 (A&A 551, A79): the MA timescale β has a LOG-uniform prior
        # spanning ~1 min … 1 yr — most prior mass at SHORT (red-noise) scales,
        # data pulls it to ~hours (τ Ceti best-fit ~1 h). The previous
        # UniformPrior(0,10) d was uniform-ON-DAYS (median 5 d!), piling prior
        # mass at multi-day timescales that SPAN the planet periods → the MA
        # correlated points across the planet and absorbed it. Match Tuomi (and
        # the :phot branch below): log-uniform from the RV cadence floor to 1 yr.
        dt_rv = length(data.t_rv) > 1 ? diff(sort(data.t_rv)) : Float64[]
        min_cad = (length(dt_rv) > 0 && any(dt_rv .> 0)) ?
                  minimum(dt_rv[dt_rv .> 0]) : min_cadence
        beta_prior = LogUniformPrior(min_cad, 365.0)
    else
        if length(data.t_phot) > 0
            dt_pm = diff(sort(data.t_phot))
            min_cad = length(dt_pm) > 0 && any(dt_pm .> 0) ?
                      minimum(dt_pm[dt_pm .> 0]) : 1.0 / (60 * 24)
        else
            min_cad = min_cadence
        end
        beta_prior = LogUniformPrior(min_cad, 365.0)
    end
    if m.per_instrument
        for ins in chan_inst
            for j in 1:m.order
                dic["ma_omega_$(j)_$(ins)$s"] = UniformPrior(-1.0, 1.0)
                dic["ma_beta_$(j)_$(ins)$s"]  = beta_prior
            end
        end
    else
        for j in 1:m.order
            dic["ma_omega_$j$s"] = UniformPrior(-1.0, 1.0)
            dic["ma_beta_$j$s"]  = beta_prior
        end
    end
end

function _default_noise_priors!(dic, m::ARModel,
                                 instruments, data, rv_max, min_cadence)
    s = _channel_suffix(m.channel)
    chan_inst = m.channel === :rv ? instruments.rv_names : instruments.pm_names
    # Same channel-specific α policy as MAModel — see comment there. Tuomi log-
    # uniform timescale (cadence floor … 1 yr), NOT uniform-on-days (which piled
    # mass at planet-period timescales and let AR absorb Keplerian power).
    if m.channel === :rv
        dt_rv = length(data.t_rv) > 1 ? diff(sort(data.t_rv)) : Float64[]
        min_cad = (length(dt_rv) > 0 && any(dt_rv .> 0)) ?
                  minimum(dt_rv[dt_rv .> 0]) : min_cadence
        alpha_prior = LogUniformPrior(min_cad, 365.0)
    else
        if length(data.t_phot) > 0
            dt_pm = diff(sort(data.t_phot))
            min_cad = length(dt_pm) > 0 && any(dt_pm .> 0) ?
                      minimum(dt_pm[dt_pm .> 0]) : 1.0 / (60 * 24)
        else
            min_cad = min_cadence
        end
        alpha_prior = LogUniformPrior(min_cad, 365.0)
    end
    if m.per_instrument
        for ins in chan_inst
            for j in 1:m.order
                dic["ar_phi_$(j)_$(ins)$s"]   = UniformPrior(-1.0, 1.0)
                dic["ar_alpha_$(j)_$(ins)$s"] = alpha_prior
            end
        end
    else
        for j in 1:m.order
            dic["ar_phi_$j$s"]   = UniformPrior(-1.0, 1.0)
            dic["ar_alpha_$j$s"] = alpha_prior
        end
    end
end

function _default_noise_priors!(dic, m::CeleriteSHO,
                                 instruments, data, rv_max, min_cadence)
    s = _gp_suffix(m)
    if m.channel === :phot
        # Photometry: fractional-flux scale; keep broad but period-floored so it
        # can't interpolate cadence-to-cadence.
        dic["gp_log_S0$s"] = UniformPrior(-25.0, 0.0)
        dic["gp_log_Q$s"] = UniformPrior(log(0.5), 5.0)
        dic["gp_log_omega0$s"] = UniformPrior(log(2π / 50), log(2π / 0.1))
        return
    end
    # RV: an UNBOUNDED SHO collapses into a point-to-point free interpolator
    # (k(0)=S0·ω0·Q; a short ω0 + large S0 manufactures a coherent wiggle at any
    # period and soaks outliers — the HD 18599 K≈23 artifact). Bound it like the
    # rotation kernel: (a) period ∈ [1, 300] d (no sub-day interpolation), (b)
    # Q > ½ (oscillatory only — short-memory is MaternGP's slot, per the menu),
    # (c) variance k(0) ≤ (3·σ_act)² across the WHOLE (ω0,Q) box, by capping
    # S0 ≤ (3σ)²/(ω0_max·Q_max) ⇒ S0·ω0·Q ≤ (3σ)² everywhere.
    σ = _model_activity_scatter(m, data, instruments)
    Pmin, Pmax, Qmin, Qmax = 1.0, 300.0, 0.5, 10.0
    ω0min, ω0max = 2π / Pmax, 2π / Pmin
    dic["gp_log_omega0$s"] = UniformPrior(log(ω0min), log(ω0max))
    dic["gp_log_Q$s"] = UniformPrior(log(Qmin), log(Qmax))
    S0cap = (3σ)^2 / (ω0max * Qmax)
    dic["gp_log_S0$s"] = UniformPrior(log(S0cap) - 12.0, log(S0cap))
end

# MaternGP: short-memory Matérn-3/2 RV GP. Amplitude anchored on the activity
# scatter (like the rotation GP), length scale sub-day to a season.
function _default_noise_priors!(dic, m::MaternGP,
                                 instruments, data, rv_max, min_cadence)
    s = _gp_suffix(m)
    σ_max = m.channel === :phot ? 1.0 : 3 * _model_activity_scatter(m, data, instruments)
    dic["matern_sigma$s"] = LogUniformPrior(1e-4, σ_max)
    # ρ floored at 1 d: a sub-day length scale lets the Matérn interpolate
    # point-to-point (same free-interpolator failure as an unbounded SHO).
    dic["matern_rho$s"] = LogUniformPrior(1.0, 100.0)
end

function _default_noise_priors!(dic, m::CeleriteRotation,
                                 instruments, data, rv_max, min_cadence)
    s = _gp_suffix(m)
    # GP amplitude scales with the activity scatter, NOT rv_max (= |absolute RV|);
    # see _rv_activity_scatter. Using rv_max made σ_max ~3×(systemic velocity) and
    # the GP a free interpolator/offset-absorber on absolute-RV data.
    σ_max = m.channel === :phot ? 1.0 : 3 * _model_activity_scatter(m, data, instruments)
    dic["gp_sigma$s"] = LogUniformPrior(1e-4, σ_max)
    dic["gp_period$s"] = LogUniformPrior(1.0, 365.0)
    dic["gp_Q0$s"] = LogUniformPrior(0.1, 10.0)
    dic["gp_dQ$s"] = UniformPrior(0.0, 5.0)
    dic["gp_f$s"] = UniformPrior(0.01, 0.99)
end

function _default_noise_priors!(dic, m::CeleriteRotationFM17,
                                 instruments, data, rv_max, min_cadence)
    s = _gp_suffix(m)
    # Photometry-channel scale: amplitude is fractional flux, ranging
    # roughly 1e-4 (very quiet) to 0.1 (active K dwarf). RV-channel anchors
    # on the activity scatter, NOT rv_max (= |absolute RV|); see
    # _rv_activity_scatter / the CeleriteRotation σ_max note above.
    log_amp_hi = m.channel === :phot ? log(0.1) :
                 log(3 * _model_activity_scatter(m, data, instruments) + 1e-3)
    log_amp_lo = m.channel === :phot ? -12.0    : -5.0
    dic["gp_log_amp$s"] = UniformPrior(log_amp_lo, log_amp_hi)
    dic["gp_log_timescale$s"] = UniformPrior(log(1.0), log(1000.0))
    dic["gp_log_period$s"] = UniformPrior(log(1.0), log(365.0))
    dic["gp_log_factor$s"] = UniformPrior(-5.0, 5.0)
end

# StudentT: degrees of freedom ν. Log-uniform over [2, 100]: ν≈2 is very
# heavy-tailed (finite-variance boundary), ν≈100 is indistinguishable from
# Gaussian, so the prior spans "outlier-dominated" to "effectively normal".
function _default_noise_priors!(dic, m::StudentT,
                                 instruments, data, rv_max, min_cadence)
    dic["studentt_nu"] = LogUniformPrior(2.0, 100.0)
end

# ErrorScale: multiplicative error-scale factor f (σ² → f²σ²), one per
# instrument. Log-uniform centred on 1 so the prior is scale-symmetric;
# [0.1, 10] in f = ±2 dex in variance, covering the CDP finding of archival
# formal errors mis-scaled by up to two orders of magnitude.
function _default_noise_priors!(dic, m::ErrorScale,
                                 instruments, data, rv_max, min_cadence)
    insts = isempty(m.instruments) ? instruments.rv_names : m.instruments
    for ins in insts
        dic["errscale_$(ins)"] = LogUniformPrior(0.1, 10.0)
    end
end

# NightlyOffset: per-instrument nightly-offset scatter σ_night (RV units),
# log-uniform up to a few × the activity scatter (same anchor as the rotation
# GP amplitude — a calibration offset can't reasonably exceed the RV variability).
function _default_noise_priors!(dic, m::NightlyOffset,
                                 instruments, data, rv_max, min_cadence)
    s = _channel_suffix(m.channel)
    σ_max = m.channel === :phot ? 1.0 : 3 * _model_activity_scatter(m, data, instruments)
    chan_inst = m.channel === :rv ? instruments.rv_names : instruments.pm_names
    insts = isempty(m.instruments) ? chan_inst : m.instruments
    for ins in insts
        dic["night_sigma_$(ins)$s"] = LogUniformPrior(1e-3, σ_max)
    end
end

# HarmonicBlock: shared rotation period P_rot (same 1–365 d prior as the
# rotation GPs, and directly comparable when the two are in an exclusion group)
# + per-instrument amplitude A (RV units, anchored on the activity scatter).
function _default_noise_priors!(dic, m::HarmonicBlock,
                                 instruments, data, rv_max, min_cadence)
    s = _channel_suffix(m.channel)
    σ_max = m.channel === :phot ? 1.0 : 3 * _model_activity_scatter(m, data, instruments)
    # No rotation period to fit when the comb is supplied externally.
    isempty(m.freqs) && (dic["harm_period$s"] = LogUniformPrior(1.0, 365.0))
    chan_inst = m.channel === :rv ? instruments.rv_names : instruments.pm_names
    insts = isempty(m.instruments) ? chan_inst : m.instruments
    for ins in insts
        dic["harm_amp_$(ins)$s"] = LogUniformPrior(1e-3, σ_max)
    end
end

# ActivityGP (Rajpaul+ 2015): kernel hyperparameters + per-channel
# coupling coefficients on G and dG/dt. Amplitude is fixed at 1 by
# convention (degenerate with channel coefficients) — left sampled
# but with a tight LogUniform around unity. Channel coefficients use
# wide priors centred on zero so the data drives sign/magnitude.
function _default_noise_priors!(dic, m::ActivityGP,
                                 instruments, data, rv_max, min_cadence)
    s = _gp_suffix(m)
    # Kernel hyperparameters. NO amplitude: G(t) is unit-variance per
    # Rajpaul+ 2015 — scale belongs to the channel couplings (see
    # noise_param_names(::ActivityGP)).
    dic["gp_act_period$s"]   = LogUniformPrior(1.0, 365.0)
    # λ_e (exponential decay): days; sub-day to multi-rotation.
    dic["gp_act_lambda_e$s"] = LogUniformPrior(1.0, 1000.0)
    # λ_p (periodic length): dimensionless, O(0.3 – 1) for smooth
    # quasi-periodic activity. The floor MUST stay well above the
    # free-interpolator regime: at λ_p ≈ 0.02 the sin² kernel supports
    # ~hour-wide structure that can absorb/alias a planet (measured on
    # HD 18599: posterior railed at the old 1e-2 floor with K biased to
    # ~2× published).
    dic["gp_act_lambda_p$s"] = LogUniformPrior(0.25, 10.0)

    # Channel couplings. BOTH the G and dG/dt coefficients are sampled
    # as AMPLITUDES in the channel's own data units: G(t) is
    # unit-variance, and the likelihood divides the derivative
    # coefficient by std(Ġ) = sqrt(1/λe² + π²/(P²λp²)) at evaluation.
    # Sampling the raw Ġ coefficient instead rides the unpenalized
    # (λe, λp, Vr) soft ridge — Var(Ġ) → 0 as λ_p rails high, Vr → ∞
    # compensating at fixed RV power (HD 18599 post-mortem: Vr ±1000).
    if !m.indicators_only
        Vc_scale = 5 * rv_max + 1.0
        dic["Vc$s"] = UniformPrior(-Vc_scale, Vc_scale)
        if m.use_derivative
            dic["Vr$s"] = UniformPrior(-Vc_scale, Vc_scale)
        end
    end

    # Indicator channels — scale per indicator from data MAD. If
    # indicator data is missing at prior-build time, fall back to a
    # broad default.
    for ch in m.channels
        ch === :rv && continue
        name = String(ch)
        mad = if data !== nothing && haskey(data.indicators, name) &&
                 !isempty(data.indicators[name])
            vals = data.indicators[name]
            max(1.4826 * median(abs.(vals .- median(vals))), 1e-3)
        else
            10.0 / 3
        end
        scale = 3 * mad
        cg, cd = _ACTIVITY_GP_COEFFS[ch]
        dic[string(cg, s)] = UniformPrior(-scale, scale)
        if m.use_derivative && cd !== nothing
            dic[string(cd, s)] = UniformPrior(-scale, scale)
        end
        # Per-channel indicator jitter: fitted floor on Σ_II's diagonal
        # (reported activity-index errors are routinely underestimated;
        # an unfloored near-singular Σ_II lets the conditional mean
        # over-track the indicators). Scaled by the reported errors
        # when available, by the channel MAD otherwise.
        med_err = if data !== nothing && haskey(data.indicator_errs, name) &&
                     !isempty(data.indicator_errs[name])
            e = filter(x -> isfinite(x) && x > 0, data.indicator_errs[name])
            isempty(e) ? mad / 10 : median(e)
        else
            mad / 10
        end
        dic["gp_act_jit_$(ch)$s"] = LogUniformPrior(0.05 * med_err,
                                                     30.0 * med_err)
    end
end
