# Posterior distribution plots for the science-output package (Jose's spec):
#   posteriors/raw         — lp-vs-param skyline; discarded=cyan, HPD=pink,
#                            median=purple vline, max=red vline. Full sample.
#   posteriors/parameters  — HPD-only points, all cyan; median/max vlines.
#   posteriors/histograms  — HPD histogram + fitted normal + annotated moments
#                            (astroEMPEROR emperors_canvas style).
#
# "HPD" = the joint credible region (`_credible_region_pool`) for planet params;
# the central `credmass` marginal region for shared/noise/instrument params.

using LaTeXStrings: latexstring   # CairoMakie re-exports the type, not this fn

# matplotlib tab colors
const _TAB = (cyan = "#17becf", pink = "#e377c2", purple = "#9467bd", red = "#d62728")

# ---- LaTeX axis labels (symbol + unit; greek words → LaTeX) -----------
# Symbol words are matched per UNDERSCORE-SEPARATED TOKEN, never by substring:
# a substring pass rewrote "rho_s" to "\rho_s" and then buried the command in
# \mathrm{}, and it would mangle any name merely *containing* "mu"/"nu".
const _MATHTOK = Dict("alpha"=>"\\alpha", "beta"=>"\\beta", "gamma"=>"\\gamma",
                      "delta"=>"\\delta", "lambda"=>"\\lambda", "sigma"=>"\\sigma",
                      "omega"=>"\\omega", "tau"=>"\\tau", "phi"=>"\\phi",
                      "rho"=>"\\rho", "mu"=>"\\mu", "nu"=>"\\nu", "psi"=>"\\psi",
                      "eta"=>"\\eta", "theta"=>"\\theta", "star"=>"\\star",
                      "Omega"=>"\\Omega", "Lambda"=>"\\Lambda", "Phi"=>"\\Phi",
                      "Psi"=>"\\Psi", "Theta"=>"\\Theta")
# indicator-name abbreviations (applied before \mathrm rendering)
const _ABBR = ("bisector_span"=>"BIS", "fwhm"=>"FWHM", "halpha"=>"Halpha",
               "log_rhk"=>"logRHK", "logrhk"=>"logRHK", "bis"=>"BIS")
_abbr(s) = (o = String(s); for (k,v) in _ABBR; o = replace(o, k => v); end; o)
# instrument/indicator token → math-safe \mathrm content (underscores → thin space)
_mtxt(s) = replace(_abbr(s), "_" => "\\,")

# One underscore-separated token → LaTeX: symbol words become commands, single
# letters stay italic variables, multi-letter words are upright text.
_textok(t) = get(_MATHTOK, t, length(t) == 1 ? String(t) : "\\mathrm{$t}")

# `head_{sub,sub,…}` for an underscore-separated name, with every subscript in
# ONE brace group. The braces are the whole point: a base carrying its own
# underscore ("M_sec", "K_A") used to emit `M_sec_{1}` / `K_A_{1}` — a double
# subscript that MathTeXEngine garbles (M_s·e·c_1) or refuses to parse at all.
function _sym_sub(name, extra::AbstractString = "")
    toks = split(_abbr(String(name)), '_'; keepempty = false)
    isempty(toks) && return "\\mathrm{?}"
    subs = String[_textok(t) for t in toks[2:end]]
    isempty(extra) || push!(subs, extra)
    isempty(subs) && return _textok(toks[1])
    return _textok(toks[1]) * "_{" * join(subs, ",") * "}"
end

# LaTeX math symbol (no $) for a parameter name.
function _sci_sym(name)
    s = String(name)
    m = match(r"^(.*)_k(\d+)$", s)
    if m !== nothing
        base, k = m.captures[1], m.captures[2]
        base == "P"      && return "P_{$k}"
        base == "K"      && return "K_{$k}"
        base == "sesinw" && return "\\sqrt{e}\\,\\sin\\omega_{$k}"
        base == "secosw" && return "\\sqrt{e}\\,\\cos\\omega_{$k}"
        base == "esinw"  && return "e\\,\\sin\\omega_{$k}"
        base == "ecosw"  && return "e\\,\\cos\\omega_{$k}"
        base == "Mo"     && return "M_{0,$k}"
        base in ("ecc","e")               && return "e_{$k}"
        base in ("inc","inc_deg","i")     && return "i_{$k}"
        base in ("omega","omega_deg","w") && return "\\omega_{$k}"
        base == "Tp" && return "T_{\\mathrm{p},$k}"
        base == "Tc" && return "T_{\\mathrm{c},$k}"
        base == "b"  && return "b_{$k}"
        base == "rr" && return "(R_p/R_\\star)_{$k}"
        return _sym_sub(base, k)
    end
    for (pre, sym) in (("gamma","\\gamma"), ("sigma","\\sigma"), ("jitter","\\sigma"))
        startswith(s, pre*"_") &&
            return "$(sym)_{\\mathrm{$(_mtxt(s[length(pre)+2:end]))}}"
    end
    startswith(s, "C_") && return "C_{\\mathrm{$(_mtxt(s[3:end]))}}"
    if startswith(s, "ind_floor_")
        r = s[11:end]
        r == "period"   && return "P_{\\mathrm{floor}}"
        r == "lambda_e" && return "\\lambda_{e}"
        r == "lambda_p" && return "\\lambda_{p}"
        endswith(r,"_amp") && return "A_{\\mathrm{$(_mtxt(r[1:end-4]))}}"
        endswith(r,"_jit") && return "\\sigma_{\\mathrm{$(_mtxt(r[1:end-4]))}}"
        return "\\mathrm{$(_mtxt(r))}"
    end
    s == "rho_s"        && return "\\rho_\\star"       # fitted stellar density
    s == "v_sin_i_star" && return "v\\sin i_\\star"
    return _sym_sub(s)
end

# LaTeX rendering of a unit string.
function _sci_unit_latex(u)
    u == "" && return ""
    m = Dict("m/s"=>"\\mathrm{m\\,s^{-1}}", "d"=>"\\mathrm{d}", "deg"=>"\\mathrm{deg}",
             "BJD"=>"\\mathrm{BJD}", "AU"=>"\\mathrm{AU}", "yr"=>"\\mathrm{yr}",
             "K"=>"\\mathrm{K}", "g/cm3"=>"\\mathrm{g\\,cm^{-3}}",
             "M_earth"=>"M_\\oplus", "M_jup"=>"M_{\\mathrm{Jup}}",
             "M_sun"=>"M_\\odot", "mas"=>"\\mathrm{mas}",
             "R_earth"=>"R_\\oplus", "R_jup"=>"R_{\\mathrm{Jup}}",
             "S_earth"=>"S_\\oplus", "W/m2"=>"\\mathrm{W\\,m^{-2}}")
    return get(m, u, "\\mathrm{$(_mtxt(u))}")
end

# Single-line axis label `$symbol$ (unit)` — for x-axes.
function _sci_label(name, params)
    sym = _sci_sym(name)
    u = _sci_unit_latex(first(sci_param_unit(name, params)))
    return latexstring(u == "" ? sym : sym * "\\;(" * u * ")")
end

# Two-line y-axis label (symbol over unit). MathTeXEngine can't line-break a
# LaTeXString, so the y-label is a UNICODE string (greek γ/σ/ω, digit subscripts
# — same intent as LaTeX, just newline-able) with the unit on the next line.
const _SUBD = Dict('0'=>'₀','1'=>'₁','2'=>'₂','3'=>'₃','4'=>'₄',
                   '5'=>'₅','6'=>'₆','7'=>'₇','8'=>'₈','9'=>'₉')
_subn(s) = join(get(_SUBD, c, c) for c in string(s))
const _GREEK_U = Dict("alpha"=>"α", "beta"=>"β", "gamma"=>"γ", "delta"=>"δ",
                      "lambda"=>"λ", "sigma"=>"σ", "omega"=>"ω", "tau"=>"τ",
                      "phi"=>"φ", "rho"=>"ρ", "mu"=>"μ", "nu"=>"ν", "psi"=>"ψ",
                      "eta"=>"η", "theta"=>"θ", "star"=>"⋆",
                      "Omega"=>"Ω", "Lambda"=>"Λ", "Phi"=>"Φ", "Psi"=>"Ψ",
                      "Theta"=>"Θ")
# Unicode twin of `_sym_sub`: same per-token lookup, subscript tokens joined by
# a space (the "σ HARPS" convention already used below) so no raw `_` is shown.
function _sym_sub_u(name)
    toks = split(_abbr(String(name)), '_'; keepempty = false)
    isempty(toks) && return "?"
    return join((get(_GREEK_U, t, String(t)) for t in toks), " ")
end
# Unicode twin of `_mtxt` for instrument/indicator text: underscores become
# spaces (as `_mtxt` makes them thin spaces) and NO greek substitution happens,
# so an instrument that happens to be called "PHI" is left alone.
_utxt(s) = replace(_abbr(String(s)), "_" => " ")

function _sci_sym_u(name)
    s = String(name)
    m = match(r"^(.*)_k(\d+)$", s)
    if m !== nothing
        base, k = m.captures[1], m.captures[2]; ks = _subn(k)
        base == "P"      && return "P" * ks
        base == "K"      && return "K" * ks
        base == "sesinw" && return "√e·sinω" * ks
        base == "secosw" && return "√e·cosω" * ks
        base == "esinw"  && return "e·sinω" * ks
        base == "ecosw"  && return "e·cosω" * ks
        base == "Mo"     && return "M₀," * ks
        base in ("ecc","e")               && return "e" * ks
        base in ("inc","inc_deg","i")     && return "i" * ks
        base in ("omega","omega_deg","w") && return "ω" * ks
        base == "Tp" && return "Tₚ," * ks
        base == "Tc" && return "Tc," * ks
        base == "b"  && return "b" * ks
        base == "rr" && return "Rₚ/R⋆ " * ks
        return _sym_sub_u(base) * ks
    end
    for (pre, sym) in (("gamma","γ"), ("sigma","σ"), ("jitter","σ"))
        startswith(s, pre*"_") && return sym * " " * _utxt(s[length(pre)+2:end])
    end
    startswith(s, "C_") && return "C(" * replace(_utxt(s[3:end]), " " => ",") * ")"
    if startswith(s, "ind_floor_")
        r = s[11:end]
        r == "period"   && return "P floor"
        r == "lambda_e" && return "λₑ"
        r == "lambda_p" && return "λₚ"
        endswith(r,"_amp") && return "A(" * _utxt(r[1:end-4]) * ")"
        endswith(r,"_jit") && return "σ(" * _utxt(r[1:end-4]) * ")"
        return _sym_sub_u(r)
    end
    s == "rho_s"        && return "ρ⋆"
    s == "i_star"       && return "i⋆"
    s == "v_sin_i_star" && return "v sin i⋆"
    return _sym_sub_u(s)
end

function _sci_ylabel(name, params)
    sym = _sci_sym_u(name)
    u = first(sci_param_unit(name, params))
    return u == "" ? sym : sym * "\n(" * u * ")"
end

# Chain values of `name` in the unit its axis label states. The labels above
# append sci_param_unit's unit, which is "deg" for angles, but the values were
# plotted straight from the chain -- radians under a degree label for Mo and ω,
# and now for Ω and λ too. rad2deg only, never mod2pi: the draws arrive
# contiguous (recenter_circular! ran before any plot), and a wrap would split a
# posterior near 0 back into two lobes at 0 and 360.
function _sci_values(chains, name, params)
    a = Array(chains[Symbol(name)])
    return last(sci_param_unit(name, params)) ? rad2deg.(a) : a
end

# Map each sampled name → owning planet slot (0 = shared/noise/instrument).
function _param_owner(params)
    owner = Dict{String, Int}()
    layout = params.layout
    for (k, blk) in enumerate(layout.planet_blocks)
        for slot in planet_slot_indices(blk)
            uf = findfirst(==(slot), layout.unfrozen_idx)
            uf === nothing && continue
            owner[layout.unfrozen_names[uf]] = k
        end
    end
    return owner
end

# Thin a chain to ≤ max_draws flat samples (even stride) for plotting — the
# posterior/histogram plots need the marginal distribution, not all ~1.9M draws.
# Rebuilds a single-chain Chains; the per-param + active-flag columns are kept.
function _thin_for_plot(chains, max_draws)
    syms = names(chains, :parameters)
    n = length(vec(Array(chains[syms[1]])))
    n <= max_draws && return chains
    idx = round.(Int, range(1, n; length = max_draws))
    mat = Matrix{Float64}(undef, length(idx), length(syms))
    for (j, s) in enumerate(syms)
        mat[:, j] = vec(Array(chains[s]))[idx]
    end
    return MCMCChains.Chains(mat, syms)
end

# Set of all noise-model parameter names (AD coefs, GP/MA/AR, indicator floor).
function _noise_param_set(params)
    s = Set{String}()
    for nm in params.config.noise_models,
        p in noise_param_names(nm, params.config.instruments)
        push!(s, p)
    end
    return s
end

# Group folder for a parameter: planets / instrumental / noise / other.
function _param_group(name, owner, noise_set)
    n = String(name)
    get(owner, n, 0) > 0 && return "planets"
    n in noise_set && return "noise"
    (startswith(n,"gamma") || startswith(n,"sigma") || startswith(n,"jitter")) &&
        return "instrumental"
    return "other"
end

# Indices (into the flat chain) that lie in the HPD region for parameter `name`.
function _hpd_indices(chains, params, name, owner, credmass)
    k = get(owner, String(name), 0)
    if k > 0
        return Set(_credible_region_pool(chains, params, k; credmass = credmass))
    end
    v = vec(Array(chains[Symbol(name)]))
    m = median(v); s = 1.4826 * median(abs.(v .- m)); s = s > 0 ? s : 1.0
    d2 = ((v .- m) ./ s) .^ 2
    return Set(findall(<=(quantile(d2, credmass)), d2))
end

# Sampled, non-fixed parameter names (optionally restricted to one planet).
function _sci_post_names(chains, params, owner; planet=nothing)
    cn = Set(names(chains, :parameters))
    nm = [n for n in params.layout.unfrozen_names if Symbol(n) in cn]
    nm = [n for n in nm if (v = vec(Array(chains[Symbol(n)])); maximum(v) - minimum(v) > 1e-12)]
    planet !== nothing && (nm = [n for n in nm if get(owner, n, 0) == planet])
    return nm
end

"""
    plot_posteriors_raw(chains, params; credmass=0.85, output, …) -> Dict

One lp-vs-parameter skyline per sampled param: discarded draws in tab:cyan, HPD
draws in tab:pink, a tab:purple vline at the (HPD) median and a tab:red vline at
the max-lp value. Writes to `output/posteriors/raw/`.
"""
function plot_posteriors_raw(chains, params; credmass::Real=0.85,
                              output::Union{Nothing,String}=nothing, fmt::Symbol=:png,
                              save_pdf::Bool=false, max_points::Int=120_000,
                              max_draws::Int=50_000, figsize=(900,420), planet=nothing)
    chains = _thin_for_plot(chains, max_draws)
    cn = Set(names(chains, :parameters))
    (:lp in cn) || throw(ArgumentError("plot_posteriors_raw requires an :lp column"))
    lp = vec(Array(chains[:lp]))
    owner = _param_owner(params)
    names_ = _sci_post_names(chains, params, owner; planet=planet)
    noise_set = _noise_param_set(params)
    outdir = output === nothing ? nothing : joinpath(output, "posteriors", "raw")
    figs = Dict{String, Figure}()
    with_theme(nereus_theme()) do
        for nm in names_
            v = vec(_sci_values(chains, nm, params))
            hpd = _hpd_indices(chains, params, nm, owner, credmass)
            allidx = collect(1:length(v))
            length(allidx) > max_points &&
                (allidx = allidx[round.(Int, range(1, length(allidx); length=max_points))])
            disc = [i for i in allidx if !(i in hpd)]
            keep = [i for i in allidx if i in hpd]
            hpd_v = v[collect(hpd)]
            med = median(hpd_v)
            vmax = v[argmax(lp)]
            fig = Figure(size=figsize)
            ax = Axis(fig[1,1]; xlabel=_sci_label(nm, params), ylabel="log P")
            isempty(disc) || scatter!(ax, v[disc], lp[disc]; color=(_TAB.cyan,0.35),
                                      markersize=3, strokewidth=0, rasterize=2)
            scatter!(ax, v[keep], lp[keep]; color=(_TAB.pink,0.5),
                     markersize=3, strokewidth=0, rasterize=2)
            vlines!(ax, [med];  color=_TAB.purple, linewidth=1.8, label="median")
            vlines!(ax, [vmax]; color=_TAB.red,    linewidth=1.8, label="max")
            axislegend(ax; position=:rb, framevisible=false, labelsize=11)
            figs[nm] = fig
            if outdir !== nothing
                gd = joinpath(outdir, _param_group(nm, owner, noise_set)); mkpath(gd)
                _save_plot(joinpath(gd, "raw_$(nm).$fmt"), fig; save_pdf=save_pdf, px_per_unit=2)
            end
        end
    end
    return figs
end

"""
    plot_posteriors_parameters(chains, params; credmass=0.85, output, …) -> Dict

HPD-only lp-vs-parameter skyline, all points tab:cyan, with median (purple) and
max (red) vlines. Writes to `output/posteriors/parameters/`.
"""
function plot_posteriors_parameters(chains, params; credmass::Real=0.85,
                              output::Union{Nothing,String}=nothing, fmt::Symbol=:png,
                              save_pdf::Bool=false, max_points::Int=120_000,
                              max_draws::Int=50_000, figsize=(900,420), planet=nothing)
    chains = _thin_for_plot(chains, max_draws)
    cn = Set(names(chains, :parameters))
    (:lp in cn) || throw(ArgumentError("plot_posteriors_parameters requires an :lp column"))
    lp = vec(Array(chains[:lp]))
    owner = _param_owner(params)
    names_ = _sci_post_names(chains, params, owner; planet=planet)
    noise_set = _noise_param_set(params)
    outdir = output === nothing ? nothing : joinpath(output, "posteriors", "parameters")
    figs = Dict{String, Figure}()
    with_theme(nereus_theme()) do
        for nm in names_
            v = vec(_sci_values(chains, nm, params))
            hpd = collect(_hpd_indices(chains, params, nm, owner, credmass))
            length(hpd) > max_points &&
                (hpd = hpd[round.(Int, range(1, length(hpd); length=max_points))])
            med = median(v[hpd]); vmax = v[argmax(lp)]
            fig = Figure(size=figsize)
            ax = Axis(fig[1,1]; xlabel=_sci_label(nm, params), ylabel="log P")
            scatter!(ax, v[hpd], lp[hpd]; color=(_TAB.cyan,0.45),
                     markersize=3, strokewidth=0, rasterize=2)
            vlines!(ax, [med];  color=_TAB.purple, linewidth=1.8, label="median")
            vlines!(ax, [vmax]; color=_TAB.red,    linewidth=1.8, label="max")
            axislegend(ax; position=:rb, framevisible=false, labelsize=11)
            figs[nm] = fig
            if outdir !== nothing
                gd = joinpath(outdir, _param_group(nm, owner, noise_set)); mkpath(gd)
                _save_plot(joinpath(gd, "param_$(nm).$fmt"), fig; save_pdf=save_pdf, px_per_unit=2)
            end
        end
    end
    return figs
end

"""
    plot_posteriors_histograms(chains, params; credmass=0.85, output, …) -> Dict

HPD-region histogram per param with a fitted normal overlaid and the distribution
moments (mean, std, skew, kurtosis) annotated — astroEMPEROR `emperors_canvas`
style. Writes to `output/posteriors/histograms/`.
"""
function plot_posteriors_histograms(chains, params; credmass::Real=0.85,
                              output::Union{Nothing,String}=nothing, fmt::Symbol=:png,
                              save_pdf::Bool=false, max_draws::Int=50_000,
                              figsize=(720,520), planet=nothing)
    chains = _thin_for_plot(chains, max_draws)
    owner = _param_owner(params)
    names_ = _sci_post_names(chains, params, owner; planet=planet)
    noise_set = _noise_param_set(params)
    outdir = output === nothing ? nothing : joinpath(output, "posteriors", "histograms")
    figs = Dict{String, Figure}()
    with_theme(nereus_theme()) do
        for nm in names_
            v = vec(_sci_values(chains, nm, params))
            hpd = collect(_hpd_indices(chains, params, nm, owner, credmass))
            x = v[hpd]
            length(x) < 10 && continue
            μ = mean(x); σ = std(x)
            z = σ > 0 ? (x .- μ) ./ σ : zeros(length(x))
            skew = mean(z .^ 3); kurt = mean(z .^ 4) - 3
            fig = Figure(size=figsize)
            ax = Axis(fig[1,1]; xlabel=_sci_label(nm, params), ylabel="density")
            hist!(ax, x; bins=40, normalization=:pdf,
                  color=(_TAB.cyan,0.55), strokewidth=0)
            if σ > 0
                xs = range(minimum(x), maximum(x); length=200)
                pdf = @. exp(-0.5*((xs-μ)/σ)^2) / (σ*sqrt(2π))
                lines!(ax, xs, pdf; color=_TAB.pink, linewidth=2.2)
            end
            vlines!(ax, [median(x)]; color=_TAB.purple, linewidth=1.8)
            txt = "μ = $(round(μ,sigdigits=5))\nσ = $(round(σ,sigdigits=4))\n" *
                  "skew = $(round(skew,sigdigits=3))\nkurt = $(round(kurt,sigdigits=3))"
            text!(ax, 0.97, 0.97; text=txt, align=(:right,:top), space=:relative,
                  fontsize=13, color=:black)
            figs[nm] = fig
            if outdir !== nothing
                gd = joinpath(outdir, _param_group(nm, owner, noise_set)); mkpath(gd)
                _save_plot(joinpath(gd, "hist_$(nm).$fmt"), fig; save_pdf=save_pdf, px_per_unit=2)
            end
        end
    end
    return figs
end

"""
    plot_traces_grouped(chains, params; output, …) -> Dict

Trace plots GROUPED into one wide multi-panel figure per group:
  - one per planet  (`trace_planet_K<k>` — P, K, sesinw, secosw, Mo stacked),
  - instrument params (γ, jitter), noise-model params, and any other nuisance.
Wider-than-tall panels stacked vertically. Writes to `output/traces/`.
"""
function plot_traces_grouped(chains, params; output::Union{Nothing,String}=nothing,
                              fmt::Symbol=:png, save_pdf::Bool=false,
                              n_walkers::Union{Nothing,Int}=nothing)
    cn = Set(names(chains, :parameters))
    layout = params.layout
    owner = _param_owner(params)
    noise_set = Set{String}()
    for nm in params.config.noise_models, p in noise_param_names(nm, params.config.instruments)
        push!(noise_set, p)
    end
    _instr(n) = startswith(n,"gamma") || startswith(n,"sigma") || startswith(n,"jitter")
    # group → ordered param list (preserve layout order)
    groups = Pair{String, Vector{String}}[]
    gidx = Dict{String, Int}()
    function _push(g, n)
        if !haskey(gidx, g); push!(groups, g => String[]); gidx[g] = length(groups); end
        push!(groups[gidx[g]].second, n)
    end
    for n in layout.unfrozen_names
        Symbol(n) in cn || continue
        k = get(owner, n, 0)
        g = k > 0 ? "planet_K$k" : (n in noise_set ? "noise" : _instr(n) ? "instrument" : "other")
        _push(g, n)
    end
    outdir = output === nothing ? nothing : joinpath(output, "traces")
    outdir !== nothing && mkpath(outdir)
    # Per-WALKER series for one param. Native multi-chain storage → one per
    # chain column; a FLAT trans-dim chain (walker = fast index) → de-interleave
    # by `n_walkers`. Each walker is drawn separately with low alpha so the
    # overlap grays out and you can see where the walkers converge — NOT the
    # walkers flattened into one jumpy line.
    function _walker_series(name)
        a = _sci_values(chains, name, params)   # degrees for angles, as labelled
        if ndims(a) == 2 && size(a, 2) > 1
            return [collect(view(a, :, c)) for c in 1:size(a, 2)]
        end
        v = vec(a)
        if n_walkers !== nothing && n_walkers > 1 && length(v) % n_walkers == 0
            return [v[w:n_walkers:length(v)] for w in 1:n_walkers]   # walker = fast index
        end
        return [v]
    end
    function _draw!(ax, name)
        series = _walker_series(name)
        nw = length(series)
        α = nw == 1 ? 0.85 : clamp(6.0 / nw, 0.02, 0.35)
        for s in series
            xs = length(s) > 1500 ? (1:cld(length(s),1500):length(s)) : (1:length(s))
            lines!(ax, collect(xs), s[xs]; color=(:black, α), linewidth=0.4,
                   rasterize = nw > 20 ? 2 : false)
        end
    end
    figs = Dict{String, Figure}()
    with_theme(nereus_theme()) do
        for (g, ns) in groups
            isempty(ns) && continue
            np = length(ns)
            fig = Figure(size=(950, max(180, 150 * np)))
            axs = Axis[]
            for (i, n) in enumerate(ns)
                ax = Axis(fig[i, 1]; ylabel=_sci_ylabel(n, params),
                          xlabel=(i == np ? "Iteration" : ""))
                # hide ALL x-decorations (incl. ticks) on upper panels — the
                # protruding ticks were the gap; rowgap=0 then truly glues them.
                i < np && hidexdecorations!(ax; grid=false)
                _draw!(ax, n)
                push!(axs, ax)
            end
            linkxaxes!(axs...)               # shared iteration axis
            rowgap!(fig.layout, 0)           # glued panels, no vertical space
            figs[g] = fig
            outdir !== nothing && _save_plot(joinpath(outdir, "trace_$(g).$fmt"), fig;
                                             save_pdf=save_pdf, px_per_unit=2)
        end
    end
    return figs
end
