# Science-output tables — the numerical deliverable of a Nereus run.
#
# Builds model-CONDITIONED, UNIT-tagged parameter statistics (median + 1σ
# asymmetric + 3σ CI) for fitted params, ready to serialize into the return
# JSON (the exoautomata API contract) and the multi-format tables. See the
# `project_output_package_spec` memory.
#
# Conditioning is the key correctness point: in a trans-dim chain a component's
# params are stale junk when that component is inactive, so every per-component
# param is summarized over ONLY the draws where its component is active, and the
# active fraction (occupancy) is reported alongside.

using Statistics: median, quantile, mean
using Random: MersenneTwister, randn
using Printf: @printf
import JSON3

# --- unit registry -----------------------------------------------------
# base param name → (unit, is_angle_in_radians). Angles are converted to deg.
const _SCI_UNITS = Dict{String, Tuple{String, Bool}}(
    "P" => ("d", false),          "K" => ("m/s", false),
    "sigma_tt" => ("d", false),   # transit-time jitter (`sigma_tt_k<n>`), days
    "K_A" => ("m/s", false),      "K_B" => ("m/s", false),   # SB2 amplitudes
    "sesinw" => ("", false),      "secosw" => ("", false),
    "esinw" => ("", false),       "ecosw" => ("", false),
    "ecc" => ("", false),         "e" => ("", false),
    "w" => ("deg", true),         "omega" => ("deg", true),
    "Mo" => ("deg", true),        "M0" => ("deg", true),
    "phic" => ("deg", true),      # transit phase 2π(Tc − t_ref_c)/P
    # Ω and λ are sampled in radians like ω and Mo, and were published as raw
    # radians. Only the exact bases match (sci_param_unit takes everything
    # before `_k<n>` as the base), so `gp_act_lambda_e`, `ind_floor_lambda_p`
    # and the ARMA `ma_omega_<j>` coefficients never pick these up.
    "Omega" => ("deg", true),     "lambda" => ("deg", true),
    "Tp" => ("BJD", false),       "Tc" => ("BJD", false),
    "b" => ("", false),           "rr" => ("", false),
    "r1" => ("", false),          "r2" => ("", false),
    "a" => ("AU", false),
    # SAMPLED IN RADIANS, reported in degrees -- so the flag must be `true`.
    # It was `false`, which means "call it deg and never convert", and every
    # table published the raw radians under a `deg` heading: a gaia4 fit wrote
    # `inc_k1  2.114  deg` for an inclination of 121.11 deg. The same flag
    # drives the prior-rail check (the CI is compared against the user's prior
    # in the units it is reported in), so that was comparing deg against rad.
    # `inc_deg` below is a DERIVED key already in degrees and stays `false`.
    "inc" => ("deg", true),       "i" => ("deg", true),
    "plx" => ("mas", false),      # parallax, sampled directly in astrometry
    "M_sec" => ("M_sun", false),  # companion mass, sampled in astrometry
    # The fitted stellar density is SAMPLED IN SOLAR UNITS, ρ★/ρ☉ = (M/M☉)/(R/R☉)³
    # (`rho_s_to_a_Rs`), and was labelled g/cm3: a 0.532 ρ☉ star (0.750 g/cm³)
    # was published as 0.532 g/cm³, 29% low. No conversion here, because the
    # prior-rail check compares the CI with the user's prior in these units;
    # the g/cm³ value is the derived `rho_star_transit_k<n>`.
    "rho_s" => ("rho_sun", false),
    # The circular-equivalent density ρ★ g³ a transit measures, sampled under
    # density = :rho_circ in the same solar units, and the coordinate η of the
    # orbit on its ellipse of constant g (src/density.jl): not an angle of the
    # orbit, so it stays in radians.
    "rho_circ" => ("rho_sun", false), "eta" => ("rad", false),
    "q1" => ("", false),          "q2" => ("", false),  # Kipping LD
    "a_Rs" => ("", false),        # a/R★ when sampled directly (`sample_a_Rs`)
    # derived
    "msini" => ("M_earth", false),"mass" => ("M_jup", false),
    "radius" => ("R_jup", false), "rho" => ("g/cm3", false),
    "Teq" => ("K", false),        "insol" => ("S_earth", false),
    "P_yr" => ("yr", false),
    # SB2 binary derived (from the two amplitudes + inclination)
    "mass_ratio_q" => ("", false),      # q = M_B/M_A = K_A/K_B (dimensionless)
    "M_A_sini3" => ("M_sun", false),    "M_B_sini3" => ("M_sun", false),
    "M_A" => ("M_sun", false),          "M_B" => ("M_sun", false),
    "inc_deg" => ("deg", false),        "P_days" => ("d", false),
    # RV trend: dvdt·(t − t_ref) + d2vdt2·(t − t_ref)², t in days
    "dvdt" => ("m/s/day", false),      "d2vdt2" => ("m/s/day^2", false),
)

# True when an instrument/noise param's trailing instrument token names a
# PHOTOMETRIC (PM) instrument, not an RV one. Instrument names can contain
# underscores (e.g. `HARPS_DRS`), so match against the known name lists by
# suffix rather than splitting on `_`.
function _is_phot_param(s::AbstractString, params::Params)
    for nm in params.config.instruments.pm_names
        endswith(s, "_" * nm) && return true
    end
    return false
end

# Dimensionless parameters carry the EMPTY unit string "": e, √e sinω, b,
# R_p/R★, q1/q2, flux offsets and dilutions, coefficients of a normalised time,
# shape factors of the CCF and the tomography maps. "" is a classification, not
# a gap: every name Nereus can emit resolves through `_sci_unit_lookup` below,
# and one that does not is a bug (test/test_science_table_labels.jl walks every
# model family).

# Fitted stellar / system-level slots without a planet suffix.
const _SCI_SYSTEM_UNITS = Dict{String, Tuple{String, Bool}}(
    "M_pri" => ("M_sun", false),          # primary mass, sampled in astrometry
    # RM: v sin i★ and the two ARoME widths are in m/s (src/default_priors.jl)
    "v_sin_i_star" => ("m/s", false),
    "sigma_ccf" => ("m/s", false),        "beta_p" => ("m/s", false),
    # Gravity darkening's stellar inclination: a polar angle SAMPLED IN RADIANS
    # on [0, π] (src/circular.jl). It was published as raw radians, no unit.
    "i_star" => ("deg", true),
    "f_light" => ("", false),             # L_B/(L_A+L_B), astrometric band
    "tomo_alpha" => ("", false),          # shared Doppler-shadow amplitude
    "u1_spec" => ("", false),             "u2_spec" => ("", false),
    "studentt_nu" => ("", false),         # Student-t degrees of freedom
)

# Ordered (prefix, unit) rules for the instrument- and map-scoped slots whose
# names carry a tag after the prefix. First match wins.
const _SCI_PREFIX_UNITS = (
    # astrometric jitters, mas
    ("sigma_as_", "mas"), ("sigma_iad_", "mas"),
    # CCF line components (src/ccf_data.jl): centres, amplitudes and widths on
    # the km/s velocity grid; trend γ̇·(t − t_ref) + ½γ̈·(t − t_ref)², t in days;
    # depths, continua, per-epoch scales and jitters in units of the normalised
    # CCF.
    ("ccf_gdot_", "km/s/day"), ("ccf_gddot_", "km/s/day^2"),
    ("ccf_gamma_", "km/s"), ("ccf_K_", "km/s"), ("ccf_fwhm_", "km/s"),
    ("ccf_depth_", ""), ("ccf_cont_", ""), ("ccf_scale_", ""), ("ccf_jit_", ""),
    # Doppler tomography, per map: local line width and velocity correlation
    # length in km/s; shadow amplitude and white term in the map's own units
    # (unit line depth).
    ("tomo_sigma_line_", "km/s"), ("tomo_ell_v_", "km/s"),
    ("tomo_alpha_", ""), ("tomo_jit_", ""),
    ("gd_beta_", ""),                      # gravity-darkening exponent, per band
    # photometric baseline: polynomial in a NORMALISED time, relative flux
    ("phot_c", ""), ("dilution_", ""),
    ("q1_", ""), ("q2_", ""), ("u1_", ""), ("u2_", ""),   # limb darkening
)

# Unit of a noise-model parameter, by the model that owns it. `y` is the unit of
# the channel the model acts on: m/s for RVs, "" for relative flux and for the
# tomography residual maps. Activity indicators are normalised to unit RMS per
# instrument by default (`Data(normalize_indicators = true)`, src/data.jl), so
# an indicator-side amplitude is dimensionless, an RV coefficient on one is in
# m/s and one on its time derivative in m/s·d. The log-parametrised celerite
# terms carry the log of their quantity's unit: S0 is a PSD, [y]²·d with ω0 in
# rad/d; the FM17 amplitude is a variance.
function _noise_param_unit(s::AbstractString, nm::NoiseModel)
    rv = noise_channel(nm) === :rv
    y = rv ? "m/s" : ""
    for (p, u) in (
            ("gp_log_S0", rv ? "ln(m2/s2*day)" : "ln(day)"),
            ("gp_log_omega0", "ln(1/day)"), ("gp_log_Q", ""),
            ("gp_log_amp", rv ? "ln(m2/s2)" : ""),
            ("gp_log_timescale", "ln(day)"), ("gp_log_period", "ln(day)"),
            ("gp_log_factor", ""),
            ("gp_sigma", y), ("gp_period", "d"),
            ("gp_Q0", ""), ("gp_dQ", ""), ("gp_f", ""),
            ("matern_sigma", y), ("matern_rho", "d"),
            ("gp_act_period", "d"), ("gp_act_lambda_e", "d"),
            ("gp_act_lambda_p", ""), ("gp_act_jit_", ""),
            ("Vc", "m/s"), ("Vr", "m/s"),
            ("Bc", ""), ("Br", ""), ("Fc", ""), ("Fr", ""),
            ("Lc", ""), ("Hc", ""), ("Hr", ""),
            ("Cdot_", "m/s*day"), ("C_", "m/s"),
            ("jit_base_", "m/s"), ("jit_act_", "m/s"),
            ("ind_floor_period", "d"), ("ind_floor_lambda_e", "d"),
            ("ind_floor_lambda_p", ""), ("ind_floor_", ""),
            ("ma_omega_", ""), ("ma_beta_", "d"),
            ("ar_phi_", ""), ("ar_alpha_", "d"),
            ("errscale_", ""), ("night_sigma_", y),
            ("harm_period", "d"), ("harm_amp_", y),
            ("studentt_nu", ""))
        startswith(s, p) && return (u, false)
    end
    return nothing
end

# `(unit, sampled_in_radians)` for a name, or `nothing` when no rule knows it.
function _sci_unit_lookup(name::AbstractString, params::Params)
    s = String(name)
    m = match(r"^(.*)_k(\d+)$", s)
    if m !== nothing && haskey(_SCI_UNITS, m.captures[1])
        return _SCI_UNITS[m.captures[1]]
    end
    haskey(_SCI_UNITS, s) && return _SCI_UNITS[s]
    haskey(_SCI_SYSTEM_UNITS, s) && return _SCI_SYSTEM_UNITS[s]
    occursin(_TTV_SLOT_NAME, s) && return ("d", false)  # free TTV offset (days)
    # The instrument slots, by the layout's own index vectors rather than by
    # name: a shared group is labelled `HARPS+FEROS` / `TESS+K2`, and a name
    # suffix test read `jitter_TESS+K2` as an RV jitter in m/s. RV offsets and
    # jitters are in m/s; photometric offsets, jitters, dilutions and baseline
    # coefficients are relative flux.
    L = params.layout
    idx = get(L.name_to_idx, s, 0)
    if idx != 0
        sy = L.systemic
        (idx in sy.rv_gamma || idx in sy.rv_sigma) && return ("m/s", false)
        (idx in sy.pm_offset || idx in sy.pm_jitter || idx in sy.pm_dilution ||
         any(v -> idx in v, sy.pm_trend)) && return ("", false)
    end
    for (p, u) in _SCI_PREFIX_UNITS
        startswith(s, p) && return (u, false)
    end
    for nm in params.config.noise_models
        s in noise_param_names(nm, params.config.instruments) || continue
        r = _noise_param_unit(s, nm)
        r === nothing || return r
    end
    # A name outside this layout: its spelling still says RV or photometric.
    phot = _is_phot_param(s, params)
    startswith(s, "gamma_")  && return (phot ? "" : "m/s", false)
    startswith(s, "sigma_")  && return (phot ? "" : "m/s", false)
    startswith(s, "jitter_") && return ("", false)      # photometric jitter
    startswith(s, "offset_") && return ("", false)      # photometric offset
    return nothing
end

"""
    sci_param_unit(name, params) -> (unit, sampled_in_radians)

Unit of a chain parameter as every output reports it (the science tables'
`unit` column and summary JSON, the LaTeX/ECSV/`.dat` writers, the figure axis
labels), and whether the chain holds it in radians, which those outputs
convert to degrees. `""` is dimensionless. A name no rule knows also comes back
`""`; `sci_unit_known` tells the two apart.
"""
function sci_param_unit(name::AbstractString, params::Params)
    r = _sci_unit_lookup(name, params)
    return r === nothing ? ("", false) : r
end

"""Whether `name` has an explicit unit rule (dimensionless counts as one)."""
sci_unit_known(name::AbstractString, params::Params) =
    _sci_unit_lookup(name, params) !== nothing

"""
    chain_param_unit(name, params) -> Union{String, Nothing}

Unit of the RAW chain values of `name`: what a draw holds, so a sampled angle is
`"rad"` where the tables report `"deg"`. The same rules as `sci_param_unit` (one
source, resolved from the built target's instrument channels), and what the fit
summary sends the client and `chains.nc` stores as the `units` attribute of every
variable. `nothing` for a name no rule knows, or for a `params` that has no
layout to resolve it against; callers then send no unit rather than a guess.
"""
function chain_param_unit(name::AbstractString, params)
    r = try
        _sci_unit_lookup(name, params)
    catch
        nothing
    end
    r === nothing && return nothing
    return r[2] ? "rad" : r[1]
end

# --- per-parameter active mask (trans-dim conditioning) ----------------
# Draws over which `name` is DEFINED: its planet/noise component active. Returns
# `nothing` for always-defined params (γ, jitter, planet params under fixed Nₚ).
function sci_active_mask(chains, params::Params, name::AbstractString)
    allnames = names(chains, :parameters)
    pidx = _param_planet_index(String(name), length(params.layout.planet_blocks))
    if pidx !== nothing
        col = Symbol("planet_active_$pidx")
        col in allnames && return _draws(chains, col) .> 0.5
    end
    nidx = _noise_model_index_for_param(String(name), params)
    if nidx !== nothing
        col = Symbol("noise_active_$nidx")
        col in allnames && return _draws(chains, col) .> 0.5
    end
    return nothing
end

# Conditioned ParamStats + metadata for one parameter. Angle params are
# converted rad→deg. Returns (stats, unit, occupancy, n_used) or nothing if the
# component is essentially never active (<1% — not meaningfully constrained).
function sci_param_entry(chains, params::Params, name::AbstractString)
    v = _draws(chains, Symbol(name))
    mask = sci_active_mask(chains, params, name)
    occ = mask === nothing ? 1.0 : mean(mask)
    samp = mask === nothing ? v : v[mask]
    (isempty(samp) || occ < 0.01) && return nothing
    unit, is_ang = sci_param_unit(name, params)
    # rad2deg ONLY. This used to be rad2deg(mod2pi(x)), and mod2pi re-splits a
    # contiguous posterior at 0/360: it already broke ω ≈ 0 for w_k under :ew,
    # whose prior is (-π, π), and it would undo the seam re-chart of
    # src/circular.jl for any angle whose median sits near 0. A circular
    # parameter arrives contiguous with its median inside the user's window
    # (recenter_circular! ran before anything reads the chains). Any other
    # angle must not be wrapped at all: its walls are real (a non-uniform or
    # multi-period prior) or the likelihood is not periodic in it (Mo under
    # TTVs), so its values are already the right ones.
    # `samp` is ours (a copy of the column, its masked or converted copy), so
    # ParamStats! may sort it.
    is_ang && (samp = rad2deg.(samp))
    return (stats = ParamStats!(samp), unit = unit, occupancy = occ,
            n_used = length(samp))
end

# JSON-able dict for one parameter entry (1σ asymmetric + 3σ CI + unit + occ).
function sci_entry_dict(e)
    s = e.stats
    return Dict{String, Any}(
        "value"    => s.best,
        "err_lo"   => s.unc_lo,            # 1σ asymmetric (median − 16th)
        "err_hi"   => s.unc_hi,            # 1σ asymmetric (84th − median)
        "ci3"      => [s.ci3[1], s.ci3[2]],# 3σ credible interval [lo, hi]
        "unit"     => e.unit,
        "occupancy"=> e.occupancy,         # active fraction (P(component | D))
        "n_used"   => e.n_used,
    )
end

# A param "rails" against its prior when its 3σ CI piles up at a bound (within 1%
# of the prior span) — i.e. the prior, not the data, sets that edge → the value is
# prior-dominated and must not be read as a measurement.
#
# `shift` (see `log_scale_shift`) puts the span on the scale the prior is flat on.
# Measured linearly, every period under 30 d on LogUniform(0.1, 3000) railed.
#
# `floor_ok` exempts the lower bound: a jitter can be zero, so a CI down to its
# floor is a value, not a prior-dominated one (see `jitter_names`).
function _railed(ci3, lo, hi, shift = nothing; floor_ok::Bool = false)
    (isfinite(lo) && isfinite(hi) && hi > lo) || return (false, "")
    if shift !== nothing && lo + shift > 0
        g(x) = log(max(x, lo) + shift)
        ci3, lo, hi = (g(ci3[1]), g(ci3[2])), g(lo), g(hi)
    end
    tol = 0.01 * (hi - lo)
    !floor_ok && (ci3[1] - lo) < tol && return (true, "lower")
    (hi - ci3[2]) < tol && return (true, "upper")
    return (false, "")
end

# --- fitted-parameter table --------------------------------------------
"""
    science_fitted(chains, params) -> (entries, conditioning)

Model-conditioned, unit-tagged stats for every fitted parameter in the winning
model: planet params for planets 1..modal-Nₚ (conditioned on each planet
active), noise params for the active noise model(s), and the always-defined
instrument params. `entries` is an ordered Vector of `name => dict`;
`conditioning` records what the table is conditioned on (modal Nₚ, active noise
models) — WITHOUT which a trans-dim "K = …" is meaningless.
"""
function science_fitted(chains, params::Params)
    allnames = names(chains, :parameters)
    max_kp = length(params.layout.planet_blocks)

    # winning model: modal Nₚ + which noise models are active >50%
    modal_np = :n_planets in allnames ? _mode_int(_draws(chains, :n_planets)) : max_kp
    active_noise = String[]
    for (i, nm) in enumerate(params.config.noise_models)
        col = Symbol("noise_active_$i")
        col in allnames || continue
        mean(_draws(chains, col) .> 0.5) > 0.5 &&
            push!(active_noise, _noise_model_label(nm))
    end

    entries = Vector{Pair{String, Dict{String, Any}}}()
    circ = circular_names(params)
    cinc = cyclic_inclination_names(params)
    jit = jitter_names(params)
    restr = restricted_angles(params)
    # The winning model's live SLOTS, not `1:modal_np`: after a death in the
    # middle of the slot list the live planet sits in a later slot, and the
    # count-based cut dropped it from the table (see `_winning_planet_slots`).
    live = Set(_winning_planet_slots(chains, params))
    for name in reported_param_names(params)
        Symbol(name) in allnames || continue
        # skip planet params for slots that are not in the winning model
        pidx = _param_planet_index(name, max_kp)
        (pidx !== nothing && !(pidx in live)) && continue
        e = sci_param_entry(chains, params, name)
        e === nothing && continue
        d = sci_entry_dict(e)
        # Flag params whose 3σ CI rails against a prior bound. A full-circle
        # angle has no bound to rail against -- its window is a chart whose seam
        # was moved off the posterior -- so it is never flagged. Every other
        # angle keeps real walls (an arc, a non-uniform prior, Mo under TTVs) and
        # is checked against the user's prior in the degrees its CI is reported
        # in. This used to skip every angle, which was harmless while Ω and λ
        # were not registered as angles; now that they are, a narrow-arc λ prior
        # would have silently lost its flag. A jitter can be zero: only its
        # upper bound is checked. A cyclic inclination has no bound at all
        # (src/circular.jl): a CI reaching 0 or 180 deg is a face-on or
        # retrograde orbit the sampler passed through, not a prior wall.
        # Nor do ρ_circ and η (src/density.jl): their box is the sampler's, and
        # the prior's walls are on ρ★ and the e-coordinates, checked as such.
        _, is_ang = sci_param_unit(name, params)
        if !(name in circ) && !(name in cinc) && !is_density_reparam_name(name)
            ps = params.config.priors[name]
            lo, hi = bounds(ps)
            is_ang && ((lo, hi) = (rad2deg(lo), rad2deg(hi)))
            rail, side = _railed(d["ci3"], lo, hi,
                                 is_ang ? nothing : log_scale_shift(ps);
                                 floor_ok = name in jit)
            d["railed"] = rail
            rail && (d["railed_bound"] = side)
            # An angle whose prior covers only part of its range: the rail is
            # that restriction, which the user may have meant.
            rail && haskey(restr, name) && (d["railed_note"] =
                "the prior restricts this angle to part of its full range " *
                "$(restr[name]); the bound is the prior's, not the data's")
        end
        # A free TTV offset: which transit it is (src/ttv.jl).
        ep = ttv_name_epoch(params.config, name)
        ep === nothing || (d["epoch"] = ep)
        push!(entries, name => d)
    end

    conditioning = Dict{String, Any}(
        "n_planets" => modal_np,
        "planet_slots" => sort!(collect(live)),
        "active_noise_models" => active_noise,
        "note" => "stats conditioned on each component being active; " *
                  "occupancy field = P(component active | data)",
    )
    ttv_rows = [k => v["epoch"] for (k, v) in entries if haskey(v, "epoch")]
    if !isempty(ttv_rows)
        conditioning["ttv_slot_epochs"] = ["$k=$e" for (k, e) in ttv_rows]
        conditioning["ttv_epoch_note"] = TTV_EPOCH_NOTE
    end
    _spin_sense_notes!(entries, conditioning, params,
                       name -> name == "i_star" || startswith(name, "lambda_k"))
    return entries, conditioning
end

# The rotation-sense note (`SPIN_SENSE_NOTE`) on the rows it concerns and in
# the table's conditioning, when the data leave the sense open (`spin_flip`).
function _spin_sense_notes!(entries, conditioning, params::Params, concerns)
    spin_flip(params) === nothing && return nothing
    hit = false
    for (k, d) in entries
        concerns(k) || continue
        d["spin_sense_note"] = SPIN_SENSE_NOTE
        hit = true
    end
    hit && (conditioning["spin_sense"] = SPIN_SENSE_NOTE)
    return nothing
end

# --- derived-parameter table (with stellar-uncertainty propagation) ----
# compute_derived key → unit. Angles already in deg, dimensionless = "".
const _DERIVED_UNITS = Dict{String, String}(
    "ecc" => "",            "omega_deg" => "deg",
    "a_au" => "AU",         "a_Rs" => "",
    "rr" => "",             "b" => "",            "inc_deg" => "deg",
    "msini_jup" => "M_jup", "msini_earth" => "M_earth",
    "mass_jup" => "M_jup",  "mass_earth" => "M_earth",
    "radius_jup" => "R_jup","radius_earth" => "R_earth",
    "density_gcc" => "g/cm3","Teq" => "K",
    "S_Wm2" => "W/m2",      "S_earth" => "S_earth",
    "TSM" => "",            "ESM" => "",          "P_yr" => "yr",
    # period when not sampled (:a_driven), derived K, orbit as an angle on the sky
    "P_days" => "d",        "K" => "m/s",
    "a_mas" => "mas",       "alpha_star_uas" => "μas",
    # transit-geometry observables (see compute_derived)
    "T14" => "hr",          "T23" => "hr",
    "depth_ppm" => "ppm",   "rho_star_transit" => "g/cm3",
    "grazing_frac" => "",
    # companion argument of periastron, sky separation and position angle
    "omega_p_deg" => "deg", "sep_mas" => "mas", "pa_deg" => "deg",
    # system rows (no planet suffix): RV-trend bounds, stellar M and log g
    "trend_P_min" => "d",   "trend_K_min" => "m/s",
    "trend_Msini_min" => "M_jup", "trend_Mc_min_torres" => "M_jup",
    "M_s_rho" => "M_sun",   "logg_s" => "dex",
    # the epochs the other time anchors sample: transit time, periastron time,
    # mean anomaly at t_ref; and the secondary conjunction after Tc
    "Tc" => "BJD",          "Tp" => "BJD",        "Mo_deg" => "deg",
    "Tsec" => "BJD",
    # spin-orbit angles (compute_derived reports them in degrees). They had no
    # entry, so every obliquity table published λ, |λ|, i★ and ψ without a unit.
    "lambda_deg" => "deg",  "lambda_abs_deg" => "deg",
    "i_star_deg" => "deg",  "psi_deg" => "deg",
    # planet surface gravity G·M_p/R_p², companion-to-star mass ratio, and the
    # geometric transit probability of a planet with no transit in the model
    "g_p" => "m/s2",        "mass_ratio" => "",   "p_transit" => "",
    # SB2 binary (from the two amplitudes, + the inclination with astrometry).
    # The masses had no entry and were published without their M_sun.
    "mass_ratio_q" => "",
    "M_A_sini3" => "M_sun", "M_B_sini3" => "M_sun",
    "M_A" => "M_sun",       "M_B" => "M_sun",
    "a_sini_au" => "AU",
)

# Unit of a compute_derived key, or `nothing` for a key no rule knows. The
# per-LD-group physical coefficients are keyed `u1_<label>`/`u2_<label>`.
function _derived_unit_lookup(key::AbstractString)
    haskey(_DERIVED_UNITS, key) && return _DERIVED_UNITS[key]
    (startswith(key, "u1_") || startswith(key, "u2_")) && return ""
    return nothing
end
derived_unit(key::AbstractString) = something(_derived_unit_lookup(key), "")

_planet_active_mask(chains, k) = (col = Symbol("planet_active_$k");
    col in names(chains, :parameters) ? _draws(chains, col) .> 0.5 : nothing)

"""
    science_derived(chains, params; M_s, R_s, T_eff, …) -> (entries, conditioning)

Model-conditioned, unit-tagged stats for DERIVED params (e, ω, m·sin i, a, ρ,
T_eq, …), computed per-draw via `compute_derived`. Eccentricity/ω are computed
per draw (NOT reconstructed from median sesinw/secosw — that collapses to e≈0).

**Stellar uncertainty is propagated**: M★/R★/T_eff are drawn per sample from
their uncertainties so derived error bars aren't fake-tight. If only point
estimates are available, a CONSERVATIVE fractional error is assumed
(`ms_frac`/`rs_frac`/`teff_abs`) — never zero. The assumed σ's are returned in
`conditioning["stellar"]` for transparency in run_info.
"""
function science_derived(chains, params::Params;
                          M_s=nothing, R_s=nothing, T_eff=nothing,
                          J_mag=nothing, K_mag=nothing, Ab=0.0,
                          sigma_M_s=nothing, sigma_R_s=nothing, sigma_T_eff=nothing,
                          ms_frac=0.10, rs_frac=0.05, teff_abs=150.0,
                          seed::Int=20260620, t_ref=nothing, data=nothing,
                          distance_pc=nothing, trend_sep_arcsec=nothing,
                          separation_epoch=nothing)
    cfg = params.config
    M_s === nothing && !isnan(cfg.M_s) && (M_s = cfg.M_s)
    R_s === nothing && hasproperty(cfg, :R_s) && !isnan(cfg.R_s) && (R_s = cfg.R_s)
    n = size(chains, 1) * size(chains, 3)
    rng = MersenneTwister(seed)
    stellar = Dict{String, Any}()
    function _draw(val, sig, frac, floor_, key)
        val === nothing && return nothing
        s = sig === nothing ? frac * abs(val) : sig
        stellar[key] = Dict("value" => val, "sigma" => s,
                            "assumed" => sig === nothing)
        return max.(val .+ s .* randn(rng, n), floor_)
    end
    ms_v  = _draw(M_s,  sigma_M_s,  ms_frac, 0.01,  "M_s")
    rs_v  = _draw(R_s,  sigma_R_s,  rs_frac, 0.01,  "R_s")
    tf_v  = _draw(T_eff, sigma_T_eff, 0.0,   100.0, "T_eff")
    tf_v === nothing && T_eff !== nothing && (tf_v = fill(Float64(T_eff), n))
    if T_eff !== nothing && sigma_T_eff === nothing && tf_v !== nothing
        tf_v = max.(Float64(T_eff) .+ teff_abs .* randn(rng, n), 100.0)
        stellar["T_eff"] = Dict("value" => T_eff, "sigma" => teff_abs, "assumed" => true)
    end

    allnames = names(chains, :parameters)
    modal_np = :n_planets in allnames ? _mode_int(_draws(chains, :n_planets)) :
               cfg.max_kplanet
    live = Set(_winning_planet_slots(chains, params))   # not 1:modal_np

    entries = Vector{Pair{String, Dict{String, Any}}}()
    rho_first = nothing        # first transiting planet's rho_star_transit
    # One planet slot at a time, in slot order: compute_derived holds some 25
    # vectors the length of the chain per slot, dropped once summarised.
    for kk in 1:cfg.max_kplanet
        kk in live || continue
        for dp in compute_derived(chains, params;
                                  M_s=ms_v, R_s=rs_v, T_eff=tf_v,
                                  J_mag=J_mag, K_mag=K_mag, Ab=Ab, planets = kk:kk,
                                  t_ref = t_ref, separation_epoch = separation_epoch)
            rho_first === nothing && (rho_first = _first_rho_transit([dp]))
            k = parse(Int, split(dp.name, "_")[end])     # "planet_1" → 1
            k in live || continue
            mask = _planet_active_mask(chains, k)
            # add P in years alongside the compute_derived outputs
            vals = copy(dp.values)
            Psym = Symbol("P_k$k")
            if Psym in allnames
                vals["P_yr"] = _draws(chains, Psym) ./ 365.25
            elseif haskey(vals, "P_days")
                vals["P_yr"] = vals["P_days"] ./ 365.25   # P derived (:a_driven)
            end
            for (key, samp) in vals
                s = mask === nothing ? copy(samp) : samp[mask]
                (isempty(s) || any(!isfinite, extrema(s))) && continue
                unit = derived_unit(key)
                # Every value is finite here, and `s` is our copy: sorted in place.
                e = (stats=ParamStats!(s), unit=unit,
                     occupancy = mask === nothing ? 1.0 : mean(mask),
                     n_used = mask === nothing ? length(samp) : count(mask))
                push!(entries, "$(key)_k$k" => sci_entry_dict(e))
            end
        end
    end
    # System and stellar rows (RV-trend bounds, M_s_rho, logg_s): one set, no
    # planet suffix, over every draw (not conditioned on a planet being active).
    sysv = compute_derived_system(chains, params; M_s=ms_v, R_s=rs_v, data=data,
                                  distance_pc=distance_pc,
                                  trend_sep_arcsec=trend_sep_arcsec,
                                  rho_star=rho_first)
    for (key, samp) in sort!(collect(sysv); by=first)
        s = filter(isfinite, samp)
        isempty(s) && continue
        e = (stats=ParamStats!(s), unit=get(_DERIVED_UNITS, key, ""),
             occupancy = 1.0, n_used = length(s))
        push!(entries, key => sci_entry_dict(e))
    end
    conditioning = Dict{String, Any}("n_planets" => modal_np,
                                     "planet_slots" => sort!(collect(live)),
                                     "stellar" => stellar,
        "note" => "derived per-draw with stellar (M★/R★/T_eff) uncertainty " *
                  "propagated; conditioned on planet active")
    # ψ is folded to the axis angle by compute_derived; i★ and λ (and |λ|,
    # which the flip sends to 180° - |λ|) are the sampler's representative.
    _spin_sense_notes!(entries, conditioning, params,
                       k -> any(p -> startswith(k, p),
                                ("psi_deg_k", "i_star_deg_k", "lambda_deg_k",
                                 "lambda_abs_deg_k")))
    return entries, conditioning
end

# --- model-selection table ---------------------------------------------
"""
    science_model_selection(chains, params) -> Dict

Trans-dim model posterior from ONE chain: occupancy P(M|D) per toggleable noise
model and per planet-count, with Bayes factor vs the modal model (within a chain,
BF_ij = occupancy_i / occupancy_j at equal model priors). Empty `Dict()` if the
chain has no trans-dim columns (a fixed-dim run — model selection is not defined).
"""
function science_model_selection(chains, params::Params)
    allnames = names(chains, :parameters)
    out = Dict{String, Any}()
    # noise-model occupancy
    noise = Dict{String, Float64}()
    for (i, nm) in enumerate(params.config.noise_models)
        col = Symbol("noise_active_$i")
        col in allnames || continue
        # Disambiguate rather than overwrite: two models of the same type (e.g.
        # a per-instrument GP on two instruments) share a label, and keying the
        # occupancy Dict on it would drop all but the last.
        lbl = _noise_model_label(nm)
        haskey(noise, lbl) && (lbl = "$(lbl)#$(i)")
        noise[lbl] = mean(_draws(chains, col) .> 0.5)
    end
    if !isempty(noise)
        best = maximum(values(noise))
        out["noise_models"] = Dict(k => Dict("occupancy" => v,
            "bayes_factor_vs_best" => v / max(best, eps())) for (k, v) in noise)
    end
    # planet-count occupancy
    if :n_planets in allnames
        np = round.(Int, _draws(chains, :n_planets))
        counts = Dict{Int, Float64}()
        for j in np; counts[j] = get(counts, j, 0.0) + 1; end
        n = length(np)
        out["n_planets"] = Dict(string(j) => c / n for (j, c) in counts)
    end
    return out
end

# =====================================================================
# Multi-format writers (json / csv / ecsv / dat / tex)
# =====================================================================
const _SCI_COLS = ("parameter", "value", "err_lo", "err_hi", "ci3_lo", "ci3_hi",
                   "unit", "occupancy")
_sci_row(name, v) = (name, v["value"], v["err_lo"], v["err_hi"],
                     v["ci3"][1], v["ci3"][2], v["unit"], v["occupancy"])

# Display rounding: 2 sig figs on the smaller error, value matched (human formats).
function _fmt3(value, lo, hi)
    errs = filter(x -> isfinite(x) && x > 0, [abs(lo), abs(hi)])
    isempty(errs) && return (string(round(value, sigdigits=4)), "0", "0")
    d = clamp(1 - floor(Int, log10(minimum(errs))), -6, 8)
    f(x) = !isfinite(x) ? "nan" :
           d <= 0 ? string(round(Int, round(x; digits=d))) : string(round(x; digits=d))
    return (f(value), f(abs(lo)), f(abs(hi)))
end

# Greek-word → LaTeX command, for rendering instrument/noise param subscripts as
# proper math (Jose's rule: γ not "gamma"). Lower-cased lookup.
const _TEX_GREEK = Dict("lambda"=>"\\lambda", "sigma"=>"\\sigma", "omega"=>"\\omega",
                        "alpha"=>"\\alpha", "beta"=>"\\beta", "phi"=>"\\phi",
                        "gamma"=>"\\gamma", "rho"=>"\\rho", "tau"=>"\\tau")

# Checked BEFORE the lowercased lookup. Without it `Omega_k1` -- the longitude
# of the ascending node -- was lowercased to "omega" and published as
# $\omega$, which is the argument of periastron. Two different angles, both in
# the same table. Only the capitals LaTeX actually defines: there is no
# \Alpha, because a capital alpha is the letter A.
const _TEX_GREEK_CAP = Dict("Omega"=>"\\Omega", "Lambda"=>"\\Lambda",
                            "Gamma"=>"\\Gamma", "Delta"=>"\\Delta",
                            "Phi"=>"\\Phi", "Psi"=>"\\Psi", "Sigma"=>"\\Sigma",
                            "Theta"=>"\\Theta", "Pi"=>"\\Pi", "Xi"=>"\\Xi")

# Render an underscore-separated suffix (e.g. an instrument name "HARPS_POST" or
# an indicator "fwhm_AD") as a math subscript body: greek tokens become commands,
# the rest roman, joined by thin spaces (no raw underscores → no broken math).
_texsub(s) = join(map(t -> get(_TEX_GREEK_CAP, String(t),
                              get(_TEX_GREEK, lowercase(t), "\\mathrm{$t}")),
                      split(String(s), "_")), "\\,")

# Generic math label when there is no dedicated symbol (gp_*, ar_*, ind_floor_*,
# rotation coeffs): consistent math styling, greek where applicable — never \texttt.
_texlabel(s) = "\$" * _texsub(s) * "\$"

# LaTeX symbol for a parameter name. Planet/derived params get dedicated symbols;
# γ/σ/jitter and activity-decorrelation C coefficients get greek/C with a roman
# subscript; everything else falls back to clean math (`_texlabel`), not \texttt.
function _texsym(name::AbstractString)
    s = String(name)
    m = match(r"^(.*)_k(\d+)$", s)
    if m !== nothing
        base, k = m.captures[1], m.captures[2]
        sym = Dict("P"=>"P", "K"=>"K", "ecc"=>"e", "omega_deg"=>"\\omega", "omega_p_deg"=>"\\omega_p",
                   "sep_mas"=>"\\rho", "pa_deg"=>"\\mathrm{PA}",
                   "Mo"=>"M_0", "phic"=>"\\phi_c", "Tp"=>"T_p", "Tc"=>"T_c", "sesinw"=>"\\sqrt{e}\\sin\\omega",
                   "secosw"=>"\\sqrt{e}\\cos\\omega", "msini_earth"=>"M\\sin i",
                   "msini_jup"=>"M\\sin i", "mass_jup"=>"M_p", "mass_earth"=>"M_p",
                   "a_au"=>"a", "a_Rs"=>"a/R_\\star", "inc_deg"=>"i", "b"=>"b",
                   # Sampled names. The fitted table uses these, not the
                   # derived spellings above, and was rendering
                   # $\mathrm{a}\,\mathrm{k1}$ / $\mathrm{inc}\,\mathrm{k1}$.
                   "a"=>"a", "inc"=>"i", "M_sec"=>"M_{\\rm sec}",
                   "Omega"=>"\\Omega", "lambda"=>"\\lambda", "w"=>"\\omega",
                   "K_A"=>"K_{\\rm A}", "K_B"=>"K_{\\rm B}",
                   "rr"=>"R_p/R_\\star", "radius_jup"=>"R_p", "radius_earth"=>"R_p",
                   "density_gcc"=>"\\rho_p", "Teq"=>"T_{\\rm eq}",
                   "S_earth"=>"S", "S_Wm2"=>"S", "P_yr"=>"P", "P_days"=>"P",
                   "a_mas"=>"a", "alpha_star_uas"=>"\\alpha_\\star",
                   "rho_circ"=>"\\rho_{\\rm circ}", "eta"=>"\\eta",
                   "Tsec"=>"T_{\\rm sec}", "g_p"=>"g_p", "mass_ratio"=>"M_p/M_\\star",
                   "p_transit"=>"p_{\\rm tr}", "a_sini_au"=>"a\\sin i",
                   "lambda_deg"=>"\\lambda", "lambda_abs_deg"=>"|\\lambda|",
                   "psi_deg"=>"\\psi", "i_star_deg"=>"i_\\star",
                   "M_A"=>"M_{\\rm A}", "M_B"=>"M_{\\rm B}",
                   "M_A_sini3"=>"M_{\\rm A}\\sin^3 i", "M_B_sini3"=>"M_{\\rm B}\\sin^3 i",
                   "mass_ratio_q"=>"q", "Mo_deg"=>"M_0")
        # brace the base so symbols carrying their own subscript (a/R_\star, …)
        # don't make an invalid double subscript when we append _{k}.
        return haskey(sym, base) ? "\${$(sym[base])}_{$k}\$" : _texlabel(s)
    end
    s == "plx"   && return "\$\\varpi\$"
    s == "rho_s" && return "\$\\rho_\\star\$"
    s == "M_s_rho" && return "\$M_\\star(\\rho_\\star)\$"
    s == "logg_s" && return "\$\\log g_\\star\$"
    s == "trend_K_min" && return "\$K_{\\rm min,trend}\$"
    s == "trend_P_min" && return "\$P_{\\rm min,trend}\$"
    s == "trend_Msini_min" && return "\$(M\\sin i)_{\\rm min,trend}\$"
    s == "trend_Mc_min_torres" && return "\$M_{c,\\rm min}^{\\rm Torres}\$"
    s == "i_star" && return "\$i_\\star\$"
    s == "M_pri" && return "\$M_{\\rm pri}\$"
    s == "v_sin_i_star" && return "\$v\\sin i_\\star\$"
    (mm = match(r"^gamma_(.+)$", s))            !== nothing && return "\$\\gamma_{$(_texsub(mm.captures[1]))}\$"
    (mm = match(r"^(?:sigma|jitter)_(.+)$", s)) !== nothing && return "\$\\sigma_{$(_texsub(mm.captures[1]))}\$"
    (mm = match(r"^C_(.+)$", s))                !== nothing && return "\$C_{$(_texsub(mm.captures[1]))}\$"
    return _texlabel(s)
end
_texesc(s) = replace(String(s), "_" => "\\_")

# Unit string → its two renderings: (LaTeX text mode, for the .tex tables;
# math mode without the dollars, for the figure labels). ONE table for both, so
# a unit added for one output path cannot be missing from the other; every unit
# the registries above emit has an entry (test/test_science_table_labels.jl).
const _UNIT_FORMS = Dict{String, Tuple{String, String}}(
    "" => ("", ""),
    "m/s" => ("m\\,s\$^{-1}\$", "\\mathrm{m\\,s^{-1}}"),
    "m/s/day" => ("m\\,s\$^{-1}\$\\,d\$^{-1}\$", "\\mathrm{m\\,s^{-1}\\,d^{-1}}"),
    "m/s/day^2" => ("m\\,s\$^{-1}\$\\,d\$^{-2}\$", "\\mathrm{m\\,s^{-1}\\,d^{-2}}"),
    "m/s*day" => ("m\\,s\$^{-1}\$\\,d", "\\mathrm{m\\,s^{-1}\\,d}"),
    "m/s2" => ("m\\,s\$^{-2}\$", "\\mathrm{m\\,s^{-2}}"),
    "km/s" => ("km\\,s\$^{-1}\$", "\\mathrm{km\\,s^{-1}}"),
    "km/s/day" => ("km\\,s\$^{-1}\$\\,d\$^{-1}\$", "\\mathrm{km\\,s^{-1}\\,d^{-1}}"),
    "km/s/day^2" => ("km\\,s\$^{-1}\$\\,d\$^{-2}\$", "\\mathrm{km\\,s^{-1}\\,d^{-2}}"),
    "ln(day)" => ("ln(d)", "\\mathrm{ln(d)}"),
    "ln(1/day)" => ("ln(d\$^{-1}\$)", "\\mathrm{ln(d^{-1})}"),
    "ln(m2/s2)" => ("ln(m\$^{2}\$\\,s\$^{-2}\$)", "\\mathrm{ln(m^{2}\\,s^{-2})}"),
    "ln(m2/s2*day)" => ("ln(m\$^{2}\$\\,s\$^{-2}\$\\,d)", "\\mathrm{ln(m^{2}\\,s^{-2}\\,d)}"),
    "M_earth" => ("\$M_\\oplus\$", "M_\\oplus"),
    "M_jup" => ("\$M_{\\rm Jup}\$", "M_{\\mathrm{Jup}}"),
    "M_sun" => ("\$M_\\odot\$", "M_\\odot"),
    "R_earth" => ("\$R_\\oplus\$", "R_\\oplus"),
    "R_jup" => ("\$R_{\\rm Jup}\$", "R_{\\mathrm{Jup}}"),
    "S_earth" => ("\$S_\\oplus\$", "S_\\oplus"),
    "rho_sun" => ("\$\\rho_\\odot\$", "\\rho_\\odot"),
    "W/m2" => ("W\\,m\$^{-2}\$", "\\mathrm{W\\,m^{-2}}"),
    "g/cm3" => ("g\\,cm\$^{-3}\$", "\\mathrm{g\\,cm^{-3}}"),
    "deg" => ("deg", "\\mathrm{deg}"), "rad" => ("rad", "\\mathrm{rad}"),
    "mas" => ("mas", "\\mathrm{mas}"), "μas" => ("\$\\mu\$as", "\\mu\\mathrm{as}"),
    "AU" => ("AU", "\\mathrm{AU}"), "d" => ("d", "\\mathrm{d}"),
    "hr" => ("hr", "\\mathrm{hr}"), "yr" => ("yr", "\\mathrm{yr}"), "dex" => ("dex", "\\mathrm{dex}"),
    "K" => ("K", "\\mathrm{K}"), "BJD" => ("BJD", "\\mathrm{BJD}"),
    "ppm" => ("ppm", "\\mathrm{ppm}"),
)
_texunit(u) = (f = get(_UNIT_FORMS, String(u), nothing);
               f === nothing ? _texesc(u) : f[1])

function _write_csv(path, entries; delim=",")
    open(path, "w") do io
        println(io, join(_SCI_COLS, delim))
        for (k, v) in entries
            println(io, join(_sci_row(k, v), delim))
        end
    end
end

# ECSV 1.0 (astropy-readable): typed-column YAML header + space-delimited body,
# table meta carries the conditioning. Units live in a per-row `unit` column
# because each parameter has its own unit (ECSV column-units can't vary by row).
function _write_ecsv(path, entries, meta)
    open(path, "w") do io
        println(io, "# %ECSV 1.0")
        println(io, "# ---")
        println(io, "# datatype:")
        for (c, dt) in zip(_SCI_COLS,
                ("string","float64","float64","float64","float64","float64","string","float64"))
            println(io, "# - {name: $c, datatype: $dt}")
        end
        # only scalar/string/vector meta in the ECSV YAML line — nested Dicts
        # (e.g. the stellar block) aren't valid inline YAML here; they live in
        # the JSON table. Keeps the ECSV header astropy-parseable.
        mline = join(["$k: $(_yamlval(v))" for (k, v) in meta
                      if !(v isa AbstractDict)], ", ")
        println(io, "# meta: {", mline, "}")
        println(io, "# schema: astropy-2.0")
        println(io, join(_SCI_COLS, " "))
        for (k, v) in entries
            r = _sci_row(k, v)
            cells = [i == 7 && r[i] == "" ? "\"\"" : string(r[i]) for i in 1:length(r)]
            println(io, join(cells, " "))
        end
    end
end
_yamlval(v) = v isa AbstractString ? "\"$v\"" :
              v isa AbstractVector ? "[" * join(string.(v), ", ") * "]" : string(v)

# What a free TTV offset's `epoch` counts, for the tables that show it.
const TTV_EPOCH_NOTE = "epoch of a ttv_k<p>_t<I> offset: the transit it holds, " *
                       "counted from the planet's first observed transit (0)"

function _write_dat(path, entries, title; note = nothing)
    open(path, "w") do io
        println(io, "# ", title)
        note === nothing || println(io, "# * ", note)
        any(haskey(v, "epoch") for (_, v) in entries) &&
            println(io, "# ", TTV_EPOCH_NOTE)
        @printf(io, "# %-22s %16s %14s %16s %-8s %8s\n",
                "parameter", "value", "err_lo", "err_hi", "unit", "occ")
        for (k, v) in entries
            vs, lo, hi = _fmt3(v["value"], v["err_lo"], v["err_hi"])
            rail = (get(v, "railed", false) ? "  railed" : "") *
                   (haskey(v, "spin_sense_note") ? "  *" : "") *
                   (haskey(v, "epoch") ? "  epoch $(v["epoch"])" : "")
            @printf(io, "%-24s %16s %14s %16s %-8s %8.3f%s\n",
                    k, vs, "-"*lo, "+"*hi, v["unit"], v["occupancy"], rail)
        end
    end
end

# `SPIN_SENSE_NOTE` in TeX; any other note as it is.
_texnote(note) = note == SPIN_SENSE_NOTE ?
    "rotation sense not constrained by the data: " *
    "\$(i_\\star, \\lambda) \\equiv (180^\\circ - i_\\star, \\lambda + 180^\\circ)\$, " *
    "\$\\psi \\equiv 180^\\circ - \\psi\$" : note

function _write_tex(path, entries, caption; note = nothing)
    any_rail = any(get(v, "railed", false) for (_, v) in entries)
    any_spin = note !== nothing && any(haskey(v, "spin_sense_note") for (_, v) in entries)
    open(path, "w") do io
        println(io, "\\begin{table}")
        println(io, "\\centering")
        println(io, "\\caption{", caption, "}")
        println(io, "\\begin{tabular}{lcl}")
        println(io, "\\hline\\hline")
        println(io, "Parameter & Value & Unit \\\\")
        println(io, "\\hline")
        for (k, v) in entries
            vs, lo, hi = _fmt3(v["value"], v["err_lo"], v["err_hi"])
            dag = get(v, "railed", false) ? "\\textsuperscript{\$\\dagger\$}" : ""
            any_spin && haskey(v, "spin_sense_note") &&
                (dag *= "\\textsuperscript{\$\\ddagger\$}")
            ep = haskey(v, "epoch") ? " (epoch $(v["epoch"]))" : ""
            println(io, _texsym(k), ep, " & ",
                    "\$$(vs)^{+$(hi)}_{-$(lo)}\$", dag, " & ", _texunit(v["unit"]), " \\\\")
        end
        println(io, "\\hline")
        println(io, "\\end{tabular}")
        any_rail && println(io, "{\\footnotesize \$^{\\dagger}\$ prior-dominated: " *
                                "the 3\$\\sigma\$ CI rails against a prior bound; " *
                                "treat as a limit, not a measurement.}")
        any_spin && println(io, "{\\footnotesize \$^{\\ddagger}\$ ",
                            _texnote(note), ".}")
        any(haskey(v, "epoch") for (_, v) in entries) &&
            println(io, "{\\footnotesize ", "epoch: the transit a TTV offset holds, counted from the planet's first " *
                    "observed transit (0).}")
        println(io, "\\end{table}")
    end
end

"""
    write_science_table(dir, name, entries, meta; formats, title) -> Dict(format=>path)

Write one science table (fitted / derived) in the requested `formats`
(:json,:csv,:ecsv,:dat,:tex). JSON/CSV/ECSV keep full precision; DAT/TeX use
2-sig-fig-on-error display rounding. Returns a manifest of format → path.
"""
function write_science_table(dir::AbstractString, name::AbstractString,
                              entries, meta::AbstractDict;
                              formats=(:json, :csv, :ecsv, :dat, :tex),
                              title::AbstractString=name)
    mkpath(dir)
    paths = Dict{String, String}()
    for fmt in formats
        p = joinpath(dir, "$(name).$(fmt)")
        if fmt === :json
            obj = Dict("meta" => meta, "parameters" => Dict(k => v for (k, v) in entries))
            open(p, "w") do io; JSON3.write(io, obj; allow_inf=true); end
        elseif fmt === :csv;  _write_csv(p, entries)
        elseif fmt === :ecsv; _write_ecsv(p, entries, meta)
        elseif fmt === :dat;  _write_dat(p, entries, title; note = get(meta, "spin_sense", nothing))
        elseif fmt === :tex;  _write_tex(p, entries, title; note = get(meta, "spin_sense", nothing))
        else; continue
        end
        paths[String(fmt)] = p
    end
    return paths
end

# =====================================================================
# Return-JSON contract assembly (the exoautomata API deliverable)
# =====================================================================
"""Provenance for `run_info`: which build of Nereus produced this result.

NOT `git rev-parse`. That ran in the PROCESS's working directory, which is
wherever the caller's notebook started and has nothing to do with where Nereus
lives. From a runtime bundle there is no repository, so it printed
`fatal: not a git repository` to stderr on every fit -- past the try/catch,
straight to the user's console. Worse, from inside ANY other checkout it
succeeded and stamped that repository's commit as Nereus's provenance: a
plausible-looking hash that means something else entirely, on a table destined
for a paper.

The package knows its own version, and a bundle ships BUILD_INFO.txt beside
the source with the exact commit it was built from. Both are read here; the
commit is reported when it is genuinely knowable and omitted when it is not.
"""
function _build_provenance()
    info = Dict{String, Any}()
    try
        info["version"] = string(pkgversion(@__MODULE__))
    catch
    end
    try
        bi = normpath(joinpath(@__DIR__, "..", "BUILD_INFO.txt"))
        if isfile(bi)
            for line in eachline(bi)
                k, _, v = partition_kv(line)
                k in ("commit", "describe", "built_utc", "platform") || continue
                isempty(v) || (info[k] = v)
            end
        end
    catch
    end
    return info
end

"""`"commit:      abc123"` -> `("commit", "abc123")`."""
function partition_kv(line::AbstractString)
    i = findfirst(':', line)
    i === nothing && return ("", "", "")
    return (strip(line[1:i-1]), ":", strip(line[i+1:end]))
end

# The priors AS THE USER WROTE THEM, from `config.priors`. The layout is what the
# samplers evaluate, and for a full-circle angle its window may have been moved
# so the seam sits off the posterior (src/circular.jl): the same density over a
# different chart. Reporting that would hand the user bounds they never wrote --
# Mo_k1 on [-3.1, 3.2] for a posterior at 0.05 under their U(0, 2π). Circular
# parameters are marked, since their bounds are a seam and a value at one of
# them is not railed.
function _prior_block(params::Params)
    out = Dict{String, Any}()
    circ = circular_names(params)
    cinc = cyclic_inclination_names(params)
    for name in reported_param_names(params)
        pr = params.config.priors[name]
        lo, hi = bounds(pr)
        d = Dict{String, Any}(
            "type" => string(nameof(typeof(pr))),
            "lower" => lo, "upper" => hi)
        # ρ_circ / η: the box the sampler moves in; their prior is the one on ρ★
        # and the e-coordinates, carried through the change of variables.
        is_density_reparam_name(name) && (d["carried"] = true)
        # The topology it was sampled on (src/circular.jl): a full circle
        # (its window a chart), or a polar angle reflected at its poles.
        name in circ && (d["circular"] = true)
        name in cinc && (d["polar"] = true)
        out[name] = d
    end
    return out
end

function _provenance(data::Data, params::Params)
    inst = params.config.instruments
    prov = Dict{String, Any}()
    if hasproperty(data, :t_rv) && !isempty(data.t_rv)
        names_rv = inst.rv_names
        nper = Dict(names_rv[i] => count(==(i), data.rv_inst) for i in eachindex(names_rv))
        prov["rv"] = Dict("n_obs" => length(data.t_rv),
            "baseline_days" => maximum(data.t_rv) - minimum(data.t_rv),
            "n_per_instrument" => nper)
    end
    if hasproperty(data, :t_phot) && !isempty(data.t_phot)
        prov["phot"] = Dict("n_obs" => length(data.t_phot),
            "baseline_days" => maximum(data.t_phot) - minimum(data.t_phot))
    end
    return prov
end

"""
    science_summary(out_dir, chains, params, data; n_walkers, result, formats, …) -> Dict

Assemble the full return-JSON contract for a run (the exoautomata API
deliverable) AND write the science tables to `out_dir/tables/`. The stellar
keywords (`M_s`, `R_s`, `T_eff`, `J_mag`, `K_mag`, `Ab` and the 1σ's
`sigma_M_s`, `sigma_R_s`, `sigma_T_eff`) go to `science_derived`, which draws
M★/R★/T_eff per sample from them; a σ left `nothing` is assumed there. Returns a Dict
with: `fitted`, `derived`, `model_selection`, `run_info` (convergence R̂/ESS,
priors, data provenance, git hash, sampler), `tables` (format→path manifest),
and an empty `figures` manifest for the plotting layer to fill (logical_name →
local path; exoautomata uploads to S3 and rewrites). Nereus stays S3-agnostic.
"""
function science_summary(out_dir::AbstractString, chains, params::Params, data::Data;
                          n_walkers::Union{Nothing, Int}=nothing,
                          result=nothing, formats=(:json, :csv, :ecsv, :dat, :tex),
                          M_s=nothing, R_s=nothing, T_eff=nothing,
                          J_mag=nothing, K_mag=nothing, Ab=0.0,
                          sigma_M_s=nothing, sigma_R_s=nothing, sigma_T_eff=nothing,
                          distance_pc=nothing, trend_sep_arcsec=nothing,
                          separation_epoch=nothing)
    fit_e, fit_c = science_fitted(chains, params)
    der_e, der_c = science_derived(chains, params; M_s=M_s, R_s=R_s, T_eff=T_eff,
                                    J_mag=J_mag, K_mag=K_mag, Ab=Ab,
                                    sigma_M_s=sigma_M_s, sigma_R_s=sigma_R_s,
                                    sigma_T_eff=sigma_T_eff, t_ref=data.t_ref, data=data,
                                    distance_pc=distance_pc,
                                    trend_sep_arcsec=trend_sep_arcsec,
                                    separation_epoch=separation_epoch)
    modsel = science_model_selection(chains, params)

    tdir = joinpath(out_dir, "tables")
    tables = Dict{String, Any}(
        "fitted"  => write_science_table(tdir, "fitted",  fit_e, fit_c; formats=formats),
        "derived" => write_science_table(tdir, "derived", der_e, der_c; formats=formats))

    # convergence (active-conditional R̂/ESS) — suppress its console print
    conv = Dict{String, Any}("assessed" => false)
    if n_walkers !== nothing
        try
            r = convergence_report(chains, n_walkers; model_params=params, io=devnull)
            conv = Dict{String, Any}("assessed" => true, "pass" => r.pass,
                "worst_rhat" => r.worst_rhat, "worst_rhat_param" => string(r.worst_rhat_param),
                "min_ess" => r.min_ess, "min_ess_param" => string(r.min_ess_param),
                "n_fail" => r.n_fail)
        catch e
            conv = Dict{String, Any}("assessed" => false, "error" => sprint(showerror, e))
        end
    end

    run_info = Dict{String, Any}(
        "nereus" => _build_provenance(),
        "convergence" => conv,
        "priors" => _prior_block(params),
        "data_provenance" => _provenance(data, params),
        "stellar" => der_c["stellar"])
    if result !== nothing
        run_info["log_evidence"] = hasfield(typeof(result), :log_evidence) ?
            Float64(getfield(result, :log_evidence)) : nothing
        run_info["n_evals"] = hasfield(typeof(result), :n_evals) ?
            Int(getfield(result, :n_evals)) : nothing
    end

    return Dict{String, Any}(
        "status" => "ok",
        "fitted"  => Dict("conditioning" => fit_c,
                          "parameters" => Dict(k => v for (k, v) in fit_e)),
        "derived" => Dict("conditioning" => der_c,
                          "parameters" => Dict(k => v for (k, v) in der_e)),
        "model_selection" => modsel,
        "run_info" => run_info,
        "tables" => tables,
        "figures" => Dict{String, Any}())   # filled by the plotting layer
end
