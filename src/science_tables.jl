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
    "K_A" => ("m/s", false),      "K_B" => ("m/s", false),   # SB2 amplitudes
    "sesinw" => ("", false),      "secosw" => ("", false),
    "esinw" => ("", false),       "ecosw" => ("", false),
    "ecc" => ("", false),         "e" => ("", false),
    "w" => ("deg", true),         "omega" => ("deg", true),
    "Mo" => ("deg", true),        "M0" => ("deg", true),
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
    "rho_s" => ("g/cm3", false),  # fitted stellar density
    "q1" => ("", false),          "q2" => ("", false),  # Kipping LD
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

# Unit + angle flag for any chain parameter name. Planet params carry a
# `_k<n>` suffix; instrument/noise params are matched by prefix.
function sci_param_unit(name::AbstractString, params::Params)
    s = String(name)
    m = match(r"^(.*)_k(\d+)$", s)
    if m !== nothing && haskey(_SCI_UNITS, m.captures[1])
        return _SCI_UNITS[m.captures[1]]
    end
    haskey(_SCI_UNITS, s) && return _SCI_UNITS[s]
    # γ/jitter on a PHOTOMETRIC instrument live in relative flux (dimensionless),
    # not m/s — disambiguate by which instrument the param belongs to.
    # Astrometric jitter is in mas, and it is spelled `sigma_as_<imager>` —
    # so it has to be caught BEFORE the generic `sigma` prefix below, which
    # would otherwise publish milliarcseconds as m/s in every table.
    startswith(s, "sigma_as_") && return ("mas", false)
    phot = _is_phot_param(s, params)
    startswith(s, "gamma")  && return (phot ? "" : "m/s", false)   # systemic velocity
    startswith(s, "sigma")  && return (phot ? "" : "m/s", false)   # jitter
    startswith(s, "jitter") && return (phot ? "" : "m/s", false)
    (startswith(s, "C_") || startswith(s, "ind_floor")) && return ("", false)
    startswith(s, "dilution") && return ("", false)     # flux dilution factor
    startswith(s, "phot_c")   && return ("", false)     # phot baseline-trend coeff
    startswith(s, "q1_") && return ("", false)          # Kipping LD (per group)
    startswith(s, "q2_") && return ("", false)
    startswith(s, "u1_") && return ("", false)          # physical LD (per group)
    startswith(s, "u2_") && return ("", false)
    return ("", false)                                  # unknown → dimensionless
end

# --- per-parameter active mask (trans-dim conditioning) ----------------
# Draws over which `name` is DEFINED: its planet/noise component active. Returns
# `nothing` for always-defined params (γ, jitter, planet params under fixed Nₚ).
function sci_active_mask(chains, params::Params, name::AbstractString)
    allnames = names(chains, :parameters)
    pidx = _param_planet_index(String(name), length(params.layout.planet_blocks))
    if pidx !== nothing
        col = Symbol("planet_active_$pidx")
        col in allnames && return vec(Array(chains[col])) .> 0.5
    end
    nidx = _noise_model_index_for_param(String(name), params)
    if nidx !== nothing
        col = Symbol("noise_active_$nidx")
        col in allnames && return vec(Array(chains[col])) .> 0.5
    end
    return nothing
end

# Conditioned ParamStats + metadata for one parameter. Angle params are
# converted rad→deg. Returns (stats, unit, occupancy, n_used) or nothing if the
# component is essentially never active (<1% — not meaningfully constrained).
function sci_param_entry(chains, params::Params, name::AbstractString)
    v = vec(Array(chains[Symbol(name)]))
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
    is_ang && (samp = rad2deg.(samp))
    return (stats = ParamStats(samp), unit = unit, occupancy = occ,
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
    modal_np = :n_planets in allnames ? _mode_int(vec(Array(chains[:n_planets]))) : max_kp
    active_noise = String[]
    for (i, nm) in enumerate(params.config.noise_models)
        col = Symbol("noise_active_$i")
        col in allnames || continue
        mean(vec(Array(chains[col])) .> 0.5) > 0.5 &&
            push!(active_noise, _noise_model_label(nm))
    end

    entries = Vector{Pair{String, Dict{String, Any}}}()
    circ = circular_names(params)
    jit = jitter_names(params)
    # The winning model's live SLOTS, not `1:modal_np`: after a death in the
    # middle of the slot list the live planet sits in a later slot, and the
    # count-based cut dropped it from the table (see `_winning_planet_slots`).
    live = Set(_winning_planet_slots(chains, params))
    for name in params.layout.unfrozen_names
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
        # upper bound is checked.
        _, is_ang = sci_param_unit(name, params)
        if !(name in circ)
            ps = params.config.priors[name]
            lo, hi = bounds(ps)
            is_ang && ((lo, hi) = (rad2deg(lo), rad2deg(hi)))
            rail, side = _railed(d["ci3"], lo, hi,
                                 is_ang ? nothing : log_scale_shift(ps);
                                 floor_ok = name in jit)
            d["railed"] = rail
            rail && (d["railed_bound"] = side)
        end
        push!(entries, name => d)
    end

    conditioning = Dict{String, Any}(
        "n_planets" => modal_np,
        "planet_slots" => sort!(collect(live)),
        "active_noise_models" => active_noise,
        "note" => "stats conditioned on each component being active; " *
                  "occupancy field = P(component active | data)",
    )
    return entries, conditioning
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
    # transit-geometry observables (see compute_derived)
    "T14" => "hr",          "T23" => "hr",
    "depth_ppm" => "ppm",   "rho_star_transit" => "g/cm3",
    "grazing_frac" => "",
    # per-instrument physical limb-darkening (keys are u1_<label>/u2_<label>;
    # those resolve to "" via the get-default, these bare entries are for
    # completeness/lookup robustness)
    "u1" => "",             "u2" => "",
)

_planet_active_mask(chains, k) = (col = Symbol("planet_active_$k");
    col in names(chains, :parameters) ? vec(Array(chains[col])) .> 0.5 : nothing)

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
                          seed::Int=20260620)
    cfg = params.config
    M_s === nothing && !isnan(cfg.M_s) && (M_s = cfg.M_s)
    R_s === nothing && hasproperty(cfg, :R_s) && !isnan(cfg.R_s) && (R_s = cfg.R_s)
    n = length(vec(Array(chains[:, 1, :])))
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

    derived_vec = compute_derived(chains, params;
                                   M_s=ms_v, R_s=rs_v, T_eff=tf_v,
                                   J_mag=J_mag, K_mag=K_mag, Ab=Ab)

    allnames = names(chains, :parameters)
    modal_np = :n_planets in allnames ? _mode_int(vec(Array(chains[:n_planets]))) :
               cfg.max_kplanet
    live = Set(_winning_planet_slots(chains, params))   # not 1:modal_np

    entries = Vector{Pair{String, Dict{String, Any}}}()
    for dp in derived_vec
        k = parse(Int, split(dp.name, "_")[end])     # "planet_1" → 1
        k in live || continue
        mask = _planet_active_mask(chains, k)
        # add P in years alongside the compute_derived outputs
        vals = copy(dp.values)
        Psym = Symbol("P_k$k")
        Psym in allnames && (vals["P_yr"] = vec(Array(chains[Psym])) ./ 365.25)
        for (key, samp) in vals
            m = mask === nothing ? trues(length(samp)) : mask
            s = samp[m]
            (isempty(s) || any(!isfinite, extrema(s))) && continue
            unit = get(_DERIVED_UNITS, key, "")
            e = (stats=ParamStats(filter(isfinite, s)), unit=unit,
                 occupancy = mask === nothing ? 1.0 : mean(mask), n_used=count(m))
            push!(entries, "$(key)_k$k" => sci_entry_dict(e))
        end
    end
    conditioning = Dict{String, Any}("n_planets" => modal_np,
                                     "planet_slots" => sort!(collect(live)),
                                     "stellar" => stellar,
        "note" => "derived per-draw with stellar (M★/R★/T_eff) uncertainty " *
                  "propagated; conditioned on planet active")
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
        noise[lbl] = mean(vec(Array(chains[col])) .> 0.5)
    end
    if !isempty(noise)
        best = maximum(values(noise))
        out["noise_models"] = Dict(k => Dict("occupancy" => v,
            "bayes_factor_vs_best" => v / max(best, eps())) for (k, v) in noise)
    end
    # planet-count occupancy
    if :n_planets in allnames
        np = round.(Int, vec(Array(chains[:n_planets])))
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
        sym = Dict("P"=>"P", "K"=>"K", "ecc"=>"e", "omega_deg"=>"\\omega",
                   "Mo"=>"M_0", "Tp"=>"T_p", "Tc"=>"T_c", "sesinw"=>"\\sqrt{e}\\sin\\omega",
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
                   "S_earth"=>"S", "S_Wm2"=>"S", "P_yr"=>"P")
        # brace the base so symbols carrying their own subscript (a/R_\star, …)
        # don't make an invalid double subscript when we append _{k}.
        return haskey(sym, base) ? "\${$(sym[base])}_{$k}\$" : _texlabel(s)
    end
    s == "plx"   && return "\$\\varpi\$"
    s == "rho_s" && return "\$\\rho_\\star\$"
    (mm = match(r"^gamma_(.+)$", s))            !== nothing && return "\$\\gamma_{$(_texsub(mm.captures[1]))}\$"
    (mm = match(r"^(?:sigma|jitter)_(.+)$", s)) !== nothing && return "\$\\sigma_{$(_texsub(mm.captures[1]))}\$"
    (mm = match(r"^C_(.+)$", s))                !== nothing && return "\$C_{$(_texsub(mm.captures[1]))}\$"
    return _texlabel(s)
end
_texesc(s) = replace(String(s), "_" => "\\_")

# Unit string → LaTeX (proper symbols/exponents instead of escaped ASCII).
const _TEX_UNIT = Dict(
    "m/s"=>"m\\,s\$^{-1}\$", "M_earth"=>"\$M_\\oplus\$", "M_jup"=>"\$M_{\\rm Jup}\$",
    "R_jup"=>"\$R_{\\rm Jup}\$", "R_earth"=>"\$R_\\oplus\$", "S_earth"=>"\$S_\\oplus\$",
    "W/m2"=>"W\\,m\$^{-2}\$", "g/cm3"=>"g\\,cm\$^{-3}\$", "deg"=>"deg",
    "AU"=>"AU", "d"=>"d", "yr"=>"yr", "K"=>"K", "BJD"=>"BJD", "mas"=>"mas", ""=>"")
_texunit(u) = get(_TEX_UNIT, String(u), _texesc(u))

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

function _write_dat(path, entries, title)
    open(path, "w") do io
        println(io, "# ", title)
        @printf(io, "# %-22s %16s %14s %16s %-8s %8s\n",
                "parameter", "value", "err_lo", "err_hi", "unit", "occ")
        for (k, v) in entries
            vs, lo, hi = _fmt3(v["value"], v["err_lo"], v["err_hi"])
            rail = get(v, "railed", false) ? "  railed" : ""
            @printf(io, "%-24s %16s %14s %16s %-8s %8.3f%s\n",
                    k, vs, "-"*lo, "+"*hi, v["unit"], v["occupancy"], rail)
        end
    end
end

function _write_tex(path, entries, caption)
    any_rail = any(get(v, "railed", false) for (_, v) in entries)
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
            println(io, _texsym(k), " & ",
                    "\$$(vs)^{+$(hi)}_{-$(lo)}\$", dag, " & ", _texunit(v["unit"]), " \\\\")
        end
        println(io, "\\hline")
        println(io, "\\end{tabular}")
        any_rail && println(io, "{\\footnotesize \$^{\\dagger}\$ prior-dominated: " *
                                "the 3\$\\sigma\$ CI rails against a prior bound; " *
                                "treat as a limit, not a measurement.}")
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
        elseif fmt === :dat;  _write_dat(p, entries, title)
        elseif fmt === :tex;  _write_tex(p, entries, title)
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
    for name in params.layout.unfrozen_names
        pr = params.config.priors[name]
        lo, hi = bounds(pr)
        d = Dict{String, Any}(
            "type" => string(nameof(typeof(pr))),
            "lower" => lo, "upper" => hi)
        name in circ && (d["circular"] = true)
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
deliverable) AND write the science tables to `out_dir/tables/`. Returns a Dict
with: `fitted`, `derived`, `model_selection`, `run_info` (convergence R̂/ESS,
priors, data provenance, git hash, sampler), `tables` (format→path manifest),
and an empty `figures` manifest for the plotting layer to fill (logical_name →
local path; exoautomata uploads to S3 and rewrites). Nereus stays S3-agnostic.
"""
function science_summary(out_dir::AbstractString, chains, params::Params, data::Data;
                          n_walkers::Union{Nothing, Int}=nothing,
                          result=nothing, formats=(:json, :csv, :ecsv, :dat, :tex),
                          M_s=nothing, R_s=nothing, T_eff=nothing,
                          J_mag=nothing, K_mag=nothing)
    fit_e, fit_c = science_fitted(chains, params)
    der_e, der_c = science_derived(chains, params; M_s=M_s, R_s=R_s, T_eff=T_eff,
                                    J_mag=J_mag, K_mag=K_mag)
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
