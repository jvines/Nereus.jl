# Plotting utilities: phase folding, CI computation, binning, model evaluation.

using Statistics: mean, median, std, quantile

"""
    _save_plot(filename, fig; save_pdf=false, save_kwargs...)

Save `fig` to `filename`. If `save_pdf` is `true` AND `filename` ends
in `.png`, also save a `.pdf` companion alongside it (same basename).
PDF companion is OFF by default; the user controls this via the
`output.save_pdf` flag in a `run_job` config or the `save_pdf` kwarg
on Python wrappers (`fit_rv`, `fit_transdim`, `run_job`).

Extra `save_kwargs` (e.g., `px_per_unit=3` for PNG raster density)
are passed to the primary `save` call. They are NOT applied to the
PDF companion — CairoMakie's PDF backend is vector-format-native and
ignores `px_per_unit` anyway.

Returns the primary `filename`.
"""
function _save_plot(filename::AbstractString, fig;
                     save_pdf::Bool=false, save_kwargs...)
    save(filename, fig; save_kwargs...)
    if save_pdf
        base, ext = splitext(filename)
        if lowercase(ext) == ".png"
            save(base * ".pdf", fig)
        end
    end
    return filename
end

"""
    _best_legend_corner(xs, ys; frac=0.35) -> Symbol

Pick the axis corner (`:lt`, `:rt`, `:lb`, `:rb`) with the fewest data
points inside a `frac × frac` box of the axis area — used for in-axis
legend placement that doesn't sit on top of data. Falls back to `:rt`
if `xs` / `ys` are empty or non-finite.
"""
function _best_legend_corner(xs::AbstractVector, ys::AbstractVector;
                              frac::Real = 0.35)
    isempty(xs) && return :rt
    xs_f = filter(isfinite, xs)
    ys_f = filter(isfinite, ys)
    (isempty(xs_f) || isempty(ys_f)) && return :rt
    xlo, xhi = extrema(xs_f)
    ylo, yhi = extrema(ys_f)
    xw, yw = xhi - xlo, yhi - ylo
    (xw <= 0 || yw <= 0) && return :rt
    xL = xlo + frac * xw
    xR = xhi - frac * xw
    yB = ylo + frac * yw
    yT = yhi - frac * yw
    cnt = Dict{Symbol,Int}(:lt => 0, :rt => 0, :lb => 0, :rb => 0)
    @inbounds for i in eachindex(xs)
        x, y = xs[i], ys[i]
        (isfinite(x) && isfinite(y)) || continue
        if x <= xL && y >= yT
            cnt[:lt] += 1
        elseif x >= xR && y >= yT
            cnt[:rt] += 1
        elseif x <= xL && y <= yB
            cnt[:lb] += 1
        elseif x >= xR && y <= yB
            cnt[:rb] += 1
        end
    end
    return argmin(cnt)
end

"""
    _mode_int_local(vals) -> Int

Mode (most-frequent integer value) of a samples vector. Used to pick
the "winning" N_p configuration for plotting.
"""
function _mode_int_local(vals::AbstractVector)
    counts = Dict{Int, Int}()
    for x in vals
        k = round(Int, x)
        counts[k] = get(counts, k, 0) + 1
    end
    best_k, best_n = 0, 0
    for (k, n) in counts
        if n > best_n
            best_k, best_n = k, n
        end
    end
    return best_k
end

"""
    winning_td_state(chains, params, active_idx) -> TransDimState

Build a `TransDimState` reflecting the winning planet count
(`n_planets_active = modal_np`) and the modal noise-active pattern
(>50% threshold per noise model) over the supplied sample subset.
Returns a fresh `TransDimState` suitable for attaching to a
`Theta{Float64}(params; td=...)`.
"""
function winning_td_state(chains, params, active_idx,
                            modal_np::Int)
    max_kp = params.config.max_kplanet
    n_noise = length(params.config.noise_models)
    td_state = TransDimState(; max_planets=max_kp, n_noise=n_noise)
    for k in 1:min(modal_np, max_kp)
        activate_planet!(td_state, k)
    end
    chain_names = Set(names(chains, :parameters))
    for nm_idx in 1:n_noise
        col = Symbol("noise_active_$nm_idx")
        if col in chain_names
            noise_samp = vec(Array(chains[col]))[active_idx]
            isempty(noise_samp) && continue
            if count(noise_samp .== 1.0) / length(noise_samp) > 0.5
                td_state.noise_active[nm_idx] = true
            end
        else
            # Non-toggleable noise model: always active
            td_state.noise_active[nm_idx] = true
        end
    end
    return td_state
end

"""
    winning_config_idx(chains, params, active_idx, td_state) -> Vector{Int}

Restrict `active_idx` to the samples whose ACTIVE noise-model configuration
matches the winning (modal) configuration `td_state`.

In a trans-dim chain the global max-lp draw is *systematically* the most
flexible model: a period-commensurate GP maximises likelihood by injecting
coherent power at the orbital period, so it dominates max-lp while LOSING the
occupancy vote. Feeding `set_theta_best_lp!` the unconditioned pool therefore
plots the model the occupancy DISFAVOURS — a "the occupancy says ErrorScale but
the fold is an obvious GP" mismatch. Matching the modal active-set first makes
the plotted best-lp draw consistent with the reported winning model.

Exact match on every toggleable noise model (non-toggleable/always-on models
carry no `noise_active_*` column, so impose no constraint). Falls back to the
unfiltered `active_idx` if the matched subset is empty.
"""
function winning_config_idx(chains, params, active_idx, td_state)
    chain_names = Set(names(chains, :parameters))
    n_noise = length(params.config.noise_models)
    keep = trues(length(active_idx))
    matched_any = false
    for nm_idx in 1:n_noise
        col = Symbol("noise_active_$nm_idx")
        col in chain_names || continue          # always-on model → no constraint
        matched_any = true
        want = td_state.noise_active[nm_idx] ? 1.0 : 0.0
        samp = vec(Array(chains[col]))[active_idx]
        @. keep &= (samp == want)
    end
    matched_any || return active_idx
    sub = active_idx[keep]
    return isempty(sub) ? active_idx : sub
end

"""
    phase_fold(t, P, t0) -> phases

Phase-fold times around period P with reference epoch t0.
Returns phases in [-0.5, 0.5).
"""
function phase_fold(t::AbstractVector, P::Real, t0::Real)
    ph = mod.((t .- t0) ./ P .+ 0.5, 1.0) .- 0.5
    return ph
end

"""
    compute_rv_model_on_grid(theta, data, t_grid; include_trend=false, gamma_inst=nothing)
        -> model_on_grid

Evaluate the RV model at times `t_grid` using parameters in `theta`.

By default returns the **Keplerian-only** signal (no γ, no trend). For
time-series plots, set `include_trend=true` to add `dvdt*(t-t_ref) +
d2vdt2*(t-t_ref)^2`. If `gamma_inst` is set, also adds the systemic γ
for that instrument index. Activity decorrelation is **not** included
(it depends on per-cadence indicator values, not on time).
"""
function compute_rv_model_on_grid(theta, data, t_grid;
                                     include_trend::Bool=false,
                                     gamma_inst::Union{Nothing,Int}=nothing,
                                     comp::Int=1)
    # `comp` selects the SB2 stellar component the model curve is drawn
    # for: 1 = primary (planet + K_A), 2 = secondary (−K_B binary only).
    # For a normal single-star fit `comp` is irrelevant (all blocks add
    # K·geom for comp==1 and the file's callers leave it at 1).
    parametrization = theta.params.config.parametrization
    t_ref = data.t_ref
    two_pi = 2π

    # Decode planets
    p_idx = planet_indices(theta)
    n_rv = 0
    for k in p_idx
        has_K(theta.params.layout.planet_blocks[k]) && (n_rv += 1)
    end

    Ps = Float64[]; Ks = Float64[]; cBs = Float64[]; es = Float64[]
    ws = Float64[]; Tps = Float64[]
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_K(block) || continue
        P = planet_P(theta, k)
        if block isa SB2Block
            K = planet_K_A(theta, k); cB = -planet_K_B(theta, k)
        else
            K = planet_K(theta, k); cB = 0.0
        end
        e, w = planet_e_w(theta, k)
        (e < 0 || e >= 1) && continue
        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tp = t_ref - ta * P / two_pi
        elseif parametrization.time === :Tp
            Tp = ta
        else
            Tp = tc_to_tp(ta, P, e, w)
        end
        push!(Ps, P); push!(Ks, K); push!(cBs, cB); push!(es, e)
        push!(ws, w); push!(Tps, Tp)
    end

    # Evaluate on grid
    model = zeros(length(t_grid))
    for j in eachindex(Ps)
        for (i, t) in enumerate(t_grid)
            M = two_pi * (t - Tps[j]) / Ps[j]
            E = kepler_solve(M, es[j])
            f = true_anomaly(E, es[j])
            geom = cos(f + ws[j]) + es[j] * cos(ws[j])
            model[i] += _comp_rv(geom, comp, Ks[j], cBs[j])
        end
    end

    if include_trend
        trend_order = theta.params.config.trend_order
        if trend_order >= 1
            dvdt = rv_dvdt(theta)
            for i in eachindex(t_grid)
                model[i] += dvdt * (t_grid[i] - t_ref)
            end
        end
        if trend_order >= 2
            d2vdt2 = rv_d2vdt2(theta)
            for i in eachindex(t_grid)
                dt = t_grid[i] - t_ref
                model[i] += d2vdt2 * dt * dt
            end
        end
    end

    if gamma_inst !== nothing
        g = rv_gamma(theta, gamma_inst)
        for i in eachindex(t_grid)
            model[i] += g
        end
    end

    return model
end

"""
    compute_rv_model_planet(theta, data, t_grid, planet_idx; comp=1) -> model

Evaluate just one planet's Keplerian signal on t_grid, for stellar
component `comp` (1 = primary A, 2 = secondary B). For a normal planet
`comp=2` returns zeros (the circumprimary planet is invisible in the
secondary's lines). For the SB2 binary block `comp=1` gives +K_A and
`comp=2` gives −K_B (anti-phase).
"""
function compute_rv_model_planet(theta, data, t_grid, planet_k::Int; comp::Int=1)
    parametrization = theta.params.config.parametrization
    t_ref = data.t_ref
    two_pi = 2π

    block = theta.params.layout.planet_blocks[planet_k]
    has_K(block) || return zeros(length(t_grid))

    P = planet_P(theta, planet_k)
    KA, cB = block isa SB2Block ?
        (planet_K_A(theta, planet_k), -planet_K_B(theta, planet_k)) :
        (planet_K(theta, planet_k), 0.0)
    e, w = planet_e_w(theta, planet_k)
    (e < 0 || e >= 1) && return zeros(length(t_grid))

    ta = planet_time_anchor(theta, planet_k)
    if parametrization.time === :Mo
        Tp = t_ref - ta * P / two_pi
    elseif parametrization.time === :Tp
        Tp = ta
    else
        Tp = tc_to_tp(ta, P, e, w)
    end

    model = zeros(length(t_grid))
    for (i, t) in enumerate(t_grid)
        M = two_pi * (t - Tp) / P
        E = kepler_solve(M, e)
        f = true_anomaly(E, e)
        geom = cos(f + w) + e * cos(w)
        model[i] += _comp_rv(geom, comp, KA, cB)
    end
    return model
end

"""
    _planet_Tp(theta, data, planet_k) -> Float64

Epoch of mean anomaly M=0 (periastron for e>0, a consistent reference for
e≈0) for planet `k`. Used as the phase-fold reference: folding around this
per-orbit epoch — rather than a distant fixed epoch like `data.t_ref`, which
sits thousands of cycles from conjunction for a short-period planet — cancels
the period-over-baseline phase smear that otherwise decoheres the CI band.
"""
function _planet_Tp(theta, data, planet_k::Int)
    parametrization = theta.params.config.parametrization
    P = planet_P(theta, planet_k)
    e, w = planet_e_w(theta, planet_k)
    ta = planet_time_anchor(theta, planet_k)
    if parametrization.time === :Mo
        return data.t_ref - ta * P / (2π)
    elseif parametrization.time === :Tp
        return ta
    else
        return tc_to_tp(ta, P, e, w)
    end
end

"""
    _planet_Tc(theta, data, planet_k) -> time of inferior conjunction

The constrained orbital epoch — fold here, NOT at periastron (`_planet_Tp`).
Periastron Tp is set by ω (and Mo), which are individually ill-determined at
low-to-moderate e (they are anti-correlated: only the mean longitude λ = Mo + ω,
i.e. the conjunction phase, is constrained). Folding per-draw on Tp jitters the
reference by the full ω scatter → the per-draw RV curves decohere → the
pointwise median washes flat and the band frays. Conjunction Tc = tp_to_tc(Tp)
is fixed by λ, so all draws phase-align and the fold is coherent.
"""
function _planet_Tc(theta, data, planet_k::Int)
    P = planet_P(theta, planet_k)
    e, w = planet_e_w(theta, planet_k)
    return tp_to_tc(_planet_Tp(theta, data, planet_k), P, e, w)
end

"""
    compute_transit_model_on_grid(theta, data, t_grid, ins_idx; planet=nothing) -> model

Evaluate the transit photometric model `f(t)` on `t_grid` for instrument
`ins_idx`. Returns the multi-planet product of (offset + Π transit_flux).

If `planet=k` is given, evaluate only planet `k`'s transit (no offset
contribution from other planets); useful for phase-folded plots that
isolate one planet.

Returns `1.0 + offset + (transit dips)` per cadence — exactly what the
photometric likelihood compares against `data.flux`.
"""
function compute_transit_model_on_grid(theta, data, t_grid, ins_idx::Int;
                                          planet::Union{Nothing, Int}=nothing)
    parametrization = theta.params.config.parametrization
    t_ref = data.t_ref
    two_pi = 2π

    # Decode all transit-active planets (or just one if `planet` given)
    p_idx_list = planet_indices(theta)
    if planet !== nothing
        p_idx_list = [planet]
    end
    n_transit = 0
    for k in p_idx_list
        block = theta.params.layout.planet_blocks[k]
        has_geometry(block) || continue
        n_transit += 1
    end

    n_grid = length(t_grid)
    model = ones(n_grid)

    # Per-instrument offset / LD
    layout = theta.params.layout
    systemic = layout.systemic
    offset = pm_offset(theta, ins_idx)
    q1 = theta.values[systemic.ld_q1[ins_idx]]
    q2 = theta.values[systemic.ld_q2[ins_idx]]
    u1, u2 = kipping_q_to_u(q1, q2)

    # Baseline continuum (offset + any trend) and third-light dilution — match the
    # likelihood so a trend/diluted fit plots a consistent model. (Exposure
    # supersampling is omitted: the overlay is already a fine-grid smooth curve.)
    dil = pm_dilution(theta, ins_idx)
    trend = Float64[theta.values[s] for s in systemic.pm_trend[ins_idx]]
    inv_th = theta.params.config.phot_trend_order > 0 ? _phot_inv_t_half(data) : 0.0

    if n_transit == 0
        for i in 1:n_grid
            model[i] = _phot_continuum(Float64(t_grid[i]), t_ref, inv_th, offset, trend)
        end
        return model
    end

    Ps   = Vector{Float64}(undef, n_transit)
    es   = Vector{Float64}(undef, n_transit)
    ws   = Vector{Float64}(undef, n_transit)
    Tps  = Vector{Float64}(undef, n_transit)
    rrs  = Vector{Float64}(undef, n_transit)
    bs   = Vector{Float64}(undef, n_transit)
    a_Rs = Vector{Float64}(undef, n_transit)
    transits = Vector{Bool}(undef, n_transit)

    use_rho = parametrization.use_rho_s
    rho_val = use_rho ? rho_s(theta) : 0.0

    j = 0
    for k in p_idx_list
        block = theta.params.layout.planet_blocks[k]
        has_geometry(block) || continue
        j += 1
        Ps[j] = planet_P(theta, k)
        e, w = planet_e_w(theta, k)
        if e < 0 || e >= 1
            return fill(NaN, n_grid)
        end
        es[j] = e
        ws[j] = w

        ta = planet_time_anchor(theta, k)
        Tps[j] = parametrization.time === :Mo ?
                  (t_ref - ta * Ps[j] / two_pi) :
                  parametrization.time === :Tp ? ta :
                  tc_to_tp(ta, Ps[j], e, w)

        b_raw, rr_raw = planet_b_rr(theta, k)
        if parametrization.geom === :r1r2
            rrs[j] = b_raw * (1 + rr_raw) / 2
            bs[j]  = b_raw * (1 - rr_raw) / 2
        else
            bs[j]  = b_raw
            rrs[j] = rr_raw
        end

        if use_rho
            a_Rs[j] = rho_s_to_a_Rs(rho_val, Ps[j])
        else
            M_s = theta.params.config.M_s
            R_s = theta.params.config.R_s
            if !isnan(M_s) && !isnan(R_s) && R_s > 0
                P_s = Ps[j] * 86400.0
                GM = 1.3271244e26 * M_s
                a_cm = cbrt(GM * P_s^2 / (4 * π^2))
                a_Rs[j] = a_cm / (R_s * 6.9570e10)
            else
                return fill(NaN, n_grid)
            end
        end

        transits[j] = bs[j] < 1 + rrs[j]
    end

    @inbounds for i in 1:n_grid
        t = t_grid[i]
        tprod = 1.0
        for jj in 1:n_transit
            transits[jj] || continue
            z = sky_separation(t, Ps[jj], es[jj], ws[jj], Tps[jj],
                                bs[jj], a_Rs[jj])
            tprod *= transit_flux(z, rrs[jj], u1, u2)
        end
        cont = _phot_continuum(Float64(t), t_ref, inv_th, offset, trend)
        model[i] = cont * ((1.0 - dil) * tprod + dil)
    end
    return model
end

# Number of FLAT draws in a Chains object: every plotting consumer
# flattens columns with vec(Array(...)), which is step-major over
# (n_steps × n_chains). Index pools MUST live in this flat space —
# `1:size(chains, 1)` only spans the FIRST WALKER of an in-memory
# multi-walker chain (measured: an HD 18599 AGP fold rendered a stuck
# walker-1 theta with γ ≈ 0 because the bf-cutoff pool, thresholded by
# the GLOBAL lp max but scanned over walker 1 only, came back empty).
_n_flat_draws(chains) = size(chains, 1) * size(chains, 3)

"""
    _top_lp_draw_pool(chains; bf_cutoff=10.0) -> Vector{Int}

EMPEROR best-fit cluster filter: keep samples whose log-posterior is
within `ln(bf_cutoff)` of the maximum — the convention is the FLAT
cut, NOT a percentile and NOT dimension-scaled. An earlier version
multiplied the cut by d/2 ("typical-set correction"); on a
30-parameter chain that turned the 2.3-nat cut into 34.5 nats and let
the high-e tail of the posterior pollute every band and fold the
filter exists to protect (the d/2 reasoning is exactly backwards: the
filter's PURPOSE is to exclude the typical set and show the best-fit
cluster). Falls back to all indices if `:lp` isn't in the chain or the
cluster is empty.
"""
function _top_lp_draw_pool(chains; bf_cutoff::Real=10.0)
    n_flat = _n_flat_draws(chains)
    chain_names = Set(names(chains, :parameters))
    (:lp in chain_names) || return collect(1:n_flat)
    lp = vec(Array(chains[:lp]))
    finite_mask = isfinite.(lp)
    any(finite_mask) || return collect(1:n_flat)
    lp_max = maximum(lp[finite_mask])
    threshold = lp_max - log(bf_cutoff)
    pool = findall(i -> finite_mask[i] && lp[i] >= threshold, 1:n_flat)
    return isempty(pool) ? collect(1:n_flat) : pool
end

"""
    _credible_region_pool(chains, params, planet; credmass=0.95) -> Vector{Int}

Draw pool for a planet's predictive band = the JOINT credible region of that
planet's parameters, NOT an lp cutoff. Acts on ALL of planet `k`'s sampled
parameters present in the chain — `P, K, sesinw, secosw` (RV) AND `b, rr, …`
(transit/PM) — so it is the single default mechanism for RV, PM and RVPM folds.
Keeps the central `credmass` fraction of draws by robust squared distance:

    d²(draw) = Σ_param ((x − median) / (1.4826·MAD))²,   keep draws with d² ≤ q_credmass

- Linear params (P, K, sesinw, secosw, b, rr, a constrained Tc, …) enter directly.
  Because `e = sesinw² + secosw²`, the (sesinw, secosw) terms penalise high
  eccentricity, so the prior-driven e→1 tail (sharp periastron spikes that,
  pointwise-percentiled, wash the median flat and fray the band) is excluded
  WITHOUT an lp threshold or an eccentricity prior.
- The orbital-phase anchor (`Mo`, or `Tp` under that parametrisation) WRAPS, so
  it enters as (cos, sin) — a genuinely unconstrained phase then contributes a
  near-constant (no spurious gating), while a constrained one discriminates.

Falls back to all draws if no planet-`k` params are in the chain. Fixed
(zero-spread) params are skipped.
"""
function _credible_region_pool(chains, params, planet::Int; credmass::Real=0.95)
    n_flat = _n_flat_draws(chains)
    allnames = String.(names(chains, :parameters))
    Psym = Symbol("P_k$(planet)")
    Pv = Psym in Symbol.(allnames) ? vec(Array(chains[Psym])) : nothing
    timekind = params.config.parametrization.time
    anchor = timekind === :Mo ? "Mo" : timekind === :Tp ? "Tp" : ""  # wrapping anchors

    _zsq(c) = begin
        m = median(c); s = 1.4826 * median(abs.(c .- m))
        s > 0 ? ((c .- m) ./ s) .^ 2 : nothing
    end

    dist2 = zeros(Float64, n_flat)
    used = 0
    for nm in allnames
        mch = match(r"^(.*)_k(\d+)$", nm)
        mch === nothing && continue
        parse(Int, mch.captures[2]) == planet || continue
        base = mch.captures[1]
        c = vec(Array(chains[Symbol(nm)]))
        length(c) == n_flat || continue
        if base == anchor                     # wrapping phase anchor → (cos, sin)
            ang = base == "Mo" ? c : (Pv === nothing ? c : 2π .* (c ./ Pv))
            for comp in (cos.(ang), sin.(ang))
                z = _zsq(comp); z === nothing && continue
                @. dist2 += z;
            end
            used += 1
        else                                   # linear param
            z = _zsq(c); z === nothing && continue
            @. dist2 += z
            used += 1
        end
    end
    used == 0 && return collect(1:n_flat)
    thr = quantile(dist2, clamp(Float64(credmass), 0.01, 1.0))
    pool = findall(<=(thr), dist2)
    return isempty(pool) ? collect(1:n_flat) : pool
end

"""
    compute_ci_bands(chains, params, data, t_grid;
                      planet=nothing, n_draws=1000, bf_cutoff=10.0)
      -> (median, lo1, hi1, lo2, hi2, lo3, hi3)

Compute credibility interval bands from posterior samples. EMPEROR
convention: draws come only from the best-fit cluster — samples whose
`:lp` is within `ln(bf_cutoff)` of the maximum log-posterior.
If `planet` is specified, computes single-planet Keplerian only.
"""
function compute_ci_bands(chains, params, data, t_grid;
                           planet::Union{Nothing, Int}=nothing,
                           n_draws::Int=1000,
                           bf_cutoff::Real=10.0,
                           fold_phase::Union{Nothing, AbstractVector}=nothing,
                           fold_t0::Real=0.0)
    pool = _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff)
    n_samples = length(pool)
    draw_idx = pool[rand(1:n_samples, min(n_draws, n_samples))]
    # Phase-fold mode: evaluate each draw on the phase grid using THAT DRAW's
    # own period (t = t0 + φ·P_draw) so every draw is a clean single-cycle
    # curve in phase. Evaluating on a fixed reference-period time grid instead
    # lets draws whose period differs from the reference complete ≠1 cycle over
    # the window → the per-draw curves decohere → kinky median + frayed band
    # whenever the period posterior is broad. Requires planet !== nothing.
    fold = fold_phase !== nothing && planet !== nothing
    n_grid = fold ? length(fold_phase) : length(t_grid)

    models = Matrix{Float64}(undef, length(draw_idx), n_grid)
    valid = trues(length(draw_idx))

    chain_names = Set(names(chains, :parameters))
    for (i, idx) in enumerate(draw_idx)
        theta = Theta{Float64}(params)
        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue
            samp_val = vec(Array(chains[sym]))[idx]
            set_param!(theta, name, samp_val)
        end

        if fold
            # Reference each draw to ITS OWN inferior-CONJUNCTION epoch (the
            # constrained orbital phase λ = Mo + ω), NOT periastron. Periastron
            # is set by ω alone, which is ill-determined at low/moderate e, so a
            # per-draw periastron fold jitters the reference by the full ω
            # scatter and decoheres the band; conjunction is fixed by λ, so all
            # draws phase-align at φ=0. (This ALSO cancels the period-over-
            # baseline phase smear, since each draw uses its own P.)
            P_draw  = planet_P(theta, planet)
            Tc_draw = _planet_Tc(theta, data, planet)
            t_eval  = Tc_draw .+ collect(fold_phase) .* P_draw
            m = compute_rv_model_planet(theta, data, t_eval, planet)
        elseif planet === nothing
            m = compute_rv_model_on_grid(theta, data, t_grid)
        else
            m = compute_rv_model_planet(theta, data, t_grid, planet)
        end
        # Reject samples with NaN/Inf model values
        if any(!isfinite, m)
            valid[i] = false
            models[i, :] .= 0.0
        else
            models[i, :] = m
        end
    end
    models = models[valid, :]
    if isempty(models)                  # every draw non-finite → return NaN bands
        nan = fill(NaN, n_grid)         # (band! skips NaN) rather than erroring
        return (; median=nan, lo1=nan, hi1=nan, lo2=nan, hi2=nan, lo3=nan, hi3=nan)
    end

    # Nested 1/2/3σ bands = EMPIRICAL percentiles of the actual computed curves
    # (Gaussian-equivalent 15.87/84.13, 2.28/97.72, 0.13/99.87 %). This is the
    # honest posterior-predictive band: it captures the genuine ASYMMETRY of an
    # eccentric RV curve, which a mean±kσ Gaussian band erases. Smoothness comes
    # from enough draws (the caller passes a large `n_draws` on a coarse phase
    # grid), NOT from a Gaussian assumption — the outer percentile is MC-noisy
    # only when undersampled.
    pq(p) = vec(mapslices(x -> quantile(x, p), models; dims=1))
    return (; median = pq(0.5),
            lo1 = pq(0.1587), hi1 = pq(0.8413),
            lo2 = pq(0.0228), hi2 = pq(0.9772),
            lo3 = pq(0.0013), hi3 = pq(0.9987))
end

"""
    compute_transit_ci_bands(chains, params, data, t_grid, ins_idx;
                              planet=nothing, n_draws=1000)

Same shape as `compute_ci_bands` but evaluates the transit photometric
model on `t_grid` for instrument `ins_idx`. If `planet=k`, only that
planet's transit is computed (no contribution from other planets).
"""
function compute_transit_ci_bands(chains, params, data, t_grid,
                                     ins_idx::Int;
                                     planet::Union{Nothing, Int}=nothing,
                                     n_draws::Int=1000,
                                     bf_cutoff::Real=10.0,
                                     fold_phase::Union{Nothing, AbstractVector}=nothing)
    pool = _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff)
    n_samples = length(pool)
    draw_idx = pool[rand(1:n_samples, min(n_draws, n_samples))]
    n_grid = fold_phase === nothing ? length(t_grid) : length(fold_phase)

    models = Matrix{Float64}(undef, length(draw_idx), n_grid)
    valid = trues(length(draw_idx))

    time_kind = params.config.parametrization.time
    chain_names = Set(names(chains, :parameters))
    for (i, idx) in enumerate(draw_idx)
        theta = Theta{Float64}(params)
        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue
            samp_val = vec(Array(chains[sym]))[idx]
            set_param!(theta, name, samp_val)
        end

        # Phase-aligned bands: with `fold_phase` (and `planet`), evaluate each
        # draw at ITS OWN ephemeris t0_d + phase×P_d so the transits stack at
        # phase 0. A common time grid turns posterior Tc/P spread into a
        # full-depth full-width "bathtub" band (at a fixed phase, some draws
        # are in-transit and some are not — the pointwise quantiles then span
        # the whole transit), which buries the shape uncertainty.
        t_eval = if fold_phase !== nothing && planet !== nothing
            P_d  = planet_P(theta, planet)
            ta_d = planet_time_anchor(theta, planet)
            t0_d = (time_kind === :Tc || time_kind === :Tp) ? ta_d : data.t_ref
            t0_d .+ fold_phase .* P_d
        else
            t_grid
        end
        m = compute_transit_model_on_grid(theta, data, t_eval, ins_idx;
                                            planet=planet)
        if any(!isfinite, m)
            valid[i] = false
            models[i, :] .= 0.0
        else
            models[i, :] = m
        end
    end
    models = models[valid, :]
    if isempty(models)                  # every draw non-finite → return NaN bands
        nan = fill(NaN, n_grid)         # (band! skips NaN) rather than erroring
        return (; median=nan, lo1=nan, hi1=nan, lo2=nan, hi2=nan, lo3=nan, hi3=nan)
    end

    # Nested 1/2/3σ bands = EMPIRICAL percentiles of the actual computed curves
    # (Gaussian-equivalent 15.87/84.13, 2.28/97.72, 0.13/99.87 %). This is the
    # honest posterior-predictive band: it captures the genuine ASYMMETRY of an
    # eccentric RV curve, which a mean±kσ Gaussian band erases. Smoothness comes
    # from enough draws (the caller passes a large `n_draws` on a coarse phase
    # grid), NOT from a Gaussian assumption — the outer percentile is MC-noisy
    # only when undersampled.
    pq(p) = vec(mapslices(x -> quantile(x, p), models; dims=1))
    return (; median = pq(0.5),
            lo1 = pq(0.1587), hi1 = pq(0.8413),
            lo2 = pq(0.0228), hi2 = pq(0.9772),
            lo3 = pq(0.0013), hi3 = pq(0.9987))
end

"""
    set_theta_best_lp!(theta, chains, params, active_idx) -> theta

Fill `theta`'s unfrozen params from the MAX-LOG-POSTERIOR sample within
`active_idx` (falling back to per-parameter marginal medians when the chain
has no `:lp`). The max-lp sample is the EMPEROR best-fit convention and the
correct REFERENCE MODEL for plot overlays: per-parameter marginal medians
are jointly inconsistent (circular angles, sesinw/secosw, correlated
rho_s/b/rr) and produce overlay curves that fall outside the predictive
bands and mis-subtract nuisance signals (measured: WASP-47 trans-dim folds,
HD 159062 relastrom).
"""
function set_theta_best_lp!(theta, chains, params, active_idx)
    chain_names = Set(names(chains, :parameters))
    if :lp in chain_names
        lp = vec(Array(chains[:lp]))
        ibest = active_idx[argmax(@view lp[active_idx])]
        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue
            set_param!(theta, name, vec(Array(chains[sym]))[ibest])
        end
    else
        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue
            set_param!(theta, name, median(vec(Array(chains[sym]))[active_idx]))
        end
    end
    return theta
end

"""
    bin_data(x, y, yerr, n_bins) -> (x_bin, y_bin, yerr_bin)

Bin data into `n_bins` equal-width bins with inverse-variance weighting.
"""
function bin_data(x, y, yerr, n_bins::Int)
    edges = range(minimum(x), maximum(x); length=n_bins + 1)
    x_bin = Float64[]
    y_bin = Float64[]
    yerr_bin = Float64[]

    for i in 1:n_bins
        mask = (x .>= edges[i]) .& (x .< edges[i + 1])
        count(mask) == 0 && continue
        w = 1.0 ./ yerr[mask] .^ 2
        sw = sum(w)
        push!(x_bin, sum(x[mask] .* w) / sw)
        push!(y_bin, sum(y[mask] .* w) / sw)
        push!(yerr_bin, 1.0 / sqrt(sw))
    end

    return x_bin, y_bin, yerr_bin
end

"""
    bin_phasefold(x, y; target=15, nbins=nothing, xlo=minimum(x), xhi=maximum(x),
                  min_count=5, robust=true) -> (x_bin, y_bin, yerr_bin)

Robust phase-fold binning for DISPLAY overlays. Bins span `[xlo, xhi]`
with the bin count **scaled to occupancy** (~`target` cadences per bin,
clamped to `[8, 80]`) unless `nbins` is given explicitly. Each bin is
summarised at its **mean x** by the **median y** with a robust standard
error `1.4826·MAD/√n`.

This is the standard for transit phase-fold overlays. It fixes the
failure mode of fixed-count, weighted-/plain-mean binning (`bin_data`):
on shallow or ultra-short-period transits a bin holds only a handful of
cadences, so a single outlier yanks the whole binned point and a fixed
60-bin grid leaves bins near-empty. The median estimator is outlier-
resistant, occupancy scaling keeps each point well-populated, and the
returned error makes a noisy bin *read* as noisy. Pass `robust=false`
for a plain mean / standard-error of the mean.
"""
function bin_phasefold(x, y; target::Int = 15,
                        nbins::Union{Nothing,Int} = nothing,
                        xlo::Real = minimum(x), xhi::Real = maximum(x),
                        min_count::Int = 5, robust::Bool = true)
    # Occupancy basis = points INSIDE [xlo, xhi]: callers may pass the full
    # phase vector with a narrow window, and sizing bins off the total count
    # starves every bin below min_count (measured: 2983 cadences → 80 bins
    # over a ±0.025 window holding ~150 points → zero bins survive).
    n = count(xi -> xlo <= xi <= xhi, x)
    n == 0 && return (Float64[], Float64[], Float64[])
    nb = nbins === nothing ? clamp(round(Int, n / max(target, 1)), 8, 80) : nbins
    edges = range(xlo, xhi; length = nb + 1)
    x_bin = Float64[]; y_bin = Float64[]; yerr_bin = Float64[]
    for k in 1:nb
        mask = (x .>= edges[k]) .& (x .< edges[k + 1])
        c = count(mask); c < min_count && continue
        xv = @view x[mask]; yv = @view y[mask]
        if robust
            med = median(yv)
            push!(y_bin, med)
            push!(yerr_bin, 1.4826 * median(abs.(yv .- med)) / sqrt(c))
        else
            mu = mean(yv)
            push!(y_bin, mu)
            push!(yerr_bin, std(yv) / sqrt(c))
        end
        push!(x_bin, mean(xv))
    end
    return x_bin, y_bin, yerr_bin
end
