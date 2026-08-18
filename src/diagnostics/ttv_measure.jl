# TTV measurement utilities — extract per-transit Tc and N-body
# posterior envelopes from sampled Nereus chains.
#
# Public API:
#   measure_per_transit_tcs(...)  → 1-D LL-profile per-transit Tc fit
#   ttvc_envelope(...)            → TTVFaster-based posterior envelope
#
# Both feed `plot_ttv_oc` to produce the canonical O−C diagram with
# data points + N-body envelope in one publication-quality figure.

using Statistics: median, quantile
using Random: MersenneTwister
using TTVFaster: Planet_plane_hk, compute_ttv!

const _MJ_PER_MS_TTV = 9.5458e-4   # M_Jup / M_Sun
const _MIN_PER_DAY   = 1440.0

"""
    measure_per_transit_tcs(data, theta_ref, params;
                            tcs_predicted, half_window=0.5,
                            grid_min=0.4, ngrid=801, planet_k=1)
        -> NamedTuple

Measure the photometric Tc of each transit by profiling the photometric
log-likelihood over a 1-D grid of Tc offsets while holding all other
parameters at `theta_ref`. The fit uses only the phot points within
`half_window` days of each predicted transit center.

Returns `(cycle_idx, tc_obs, tc_lo, tc_hi, mle_dt, lo_dt, hi_dt)` where
`*_dt` are the offsets from `tcs_predicted` in days (suitable for an
O−C plot).

# Arguments
- `data::Data` — photometric data
- `theta_ref::Theta` — reference parameters (typically posterior median)
- `params::Params` — model layout
- `tcs_predicted::Vector{Float64}` — linear-ephemeris transit centers
- `half_window::Float64` — window half-width [d]; phot points within
  this distance of each `tc_predicted` are used.
- `grid_min::Float64` — Tc-offset grid spans ±`grid_min` [d]
- `ngrid::Int` — grid resolution
- `planet_k::Int` — planet index (1-based) whose Tc is being measured
"""
function measure_per_transit_tcs(data::Data, theta_ref::Theta, params::Params;
                                   tcs_predicted::Vector{Float64},
                                   half_window::Float64 = 0.5,
                                   grid_min::Float64 = 0.4,
                                   ngrid::Int = 801,
                                   planet_k::Int = 1)
    layout = params.layout
    config = params.config

    # Decode planet-k's shape params from theta_ref
    P_k  = theta_ref.values[layout.name_to_idx["P_k$(planet_k)"]]
    Tc_k = theta_ref.values[layout.name_to_idx["Tc_k$(planet_k)"]]
    se_k = theta_ref.values[layout.name_to_idx["sesinw_k$(planet_k)"]]
    sc_k = theta_ref.values[layout.name_to_idx["secosw_k$(planet_k)"]]
    e_k  = hypot(se_k, sc_k)^2
    w_k  = atan(se_k, sc_k)
    b_k  = theta_ref.values[layout.name_to_idx["b_k$(planet_k)"]]
    rr_k = theta_ref.values[layout.name_to_idx["rr_k$(planet_k)"]]

    # a/R*: use ρ_s if parametrized
    if config.parametrization.use_rho_s
        rho = theta_ref.values[layout.name_to_idx["rho_s"]]
        a_Rs = rho_s_to_a_Rs(rho, P_k)
    else
        # Fall back to M_s + R_s
        M_s = config.M_s; R_s = config.R_s
        P_s = P_k * 86400.0
        GM  = 1.3271244e26 * M_s
        a_cm = cbrt(GM * P_s^2 / (4π^2))
        a_Rs = a_cm / (R_s * 6.9570e10)
    end

    # Per-instrument LD, offset, jitter
    n_pm = length(config.instruments.pm_names)
    lds      = Vector{QuadLimbDark}(undef, n_pm)
    offsets  = Vector{Float64}(undef, n_pm)
    jitters  = Vector{Float64}(undef, n_pm)
    for ix in 1:n_pm
        q1 = theta_ref.values[layout.systemic.ld_q1[ix]]
        q2 = theta_ref.values[layout.systemic.ld_q2[ix]]
        u1, u2 = kipping_q_to_u(q1, q2)
        lds[ix]     = QuadLimbDark([u1, u2])
        offsets[ix] = pm_offset(theta_ref, ix)
        jitters[ix] = pm_jitter(theta_ref, ix)
    end

    n_tr = length(tcs_predicted)
    cycle_idx = collect(0:(n_tr - 1))
    tc_obs = Vector{Float64}(undef, n_tr)
    tc_lo  = Vector{Float64}(undef, n_tr)
    tc_hi  = Vector{Float64}(undef, n_tr)

    # Pre-sort phot points by time (assumed sorted; verify cheaply)
    issorted(data.t_phot) || error("measure_per_transit_tcs requires sorted data.t_phot")

    # Grid of Tc offsets
    dts_grid = range(-grid_min, grid_min; length = ngrid)

    for i in 1:n_tr
        tc_pred = tcs_predicted[i]
        # Phot points in window
        lo_idx = searchsortedfirst(data.t_phot, tc_pred - half_window)
        hi_idx = searchsortedlast(data.t_phot, tc_pred + half_window)
        if hi_idx < lo_idx
            tc_obs[i] = tc_pred
            tc_lo[i]  = tc_pred - 1/_MIN_PER_DAY
            tc_hi[i]  = tc_pred + 1/_MIN_PER_DAY
            continue
        end

        # Evaluate LL on the grid
        lls = Vector{Float64}(undef, ngrid)
        @inbounds for (g, dt) in enumerate(dts_grid)
            tc_g = tc_pred + dt
            Tp_g = tc_to_tp(tc_g, P_k, e_k, w_k)
            ll = 0.0
            @inbounds for j in lo_idx:hi_idx
                t = data.t_phot[j]
                z = sky_separation(t, P_k, e_k, w_k, Tp_g, b_k, a_Rs)
                inst_j = data.phot_inst[j]
                ld = lds[inst_j]
                f  = transit_flux(ld, z, rr_k)
                mf = (1.0 + offsets[inst_j]) * f
                var_j = data.flux_err[j]^2 + jitters[inst_j]^2
                resid = data.flux[j] - mf
                ll += -(log(2π * var_j) + resid * resid / var_j) / 2
            end
            lls[g] = ll
        end

        # MLE and 1σ from Δlog L = 0.5
        gmax = argmax(lls)
        ll_max = lls[gmax]
        threshold = ll_max - 0.5
        # Scan left/right from peak for the bound crossings
        glo = gmax; while glo > 1 && lls[glo - 1] >= threshold; glo -= 1; end
        ghi = gmax; while ghi < ngrid && lls[ghi + 1] >= threshold; ghi += 1; end
        # Linear interpolate to the threshold for sub-grid precision
        function interp(g_in::Int, dir::Int)
            g_out = g_in + dir
            if g_out < 1 || g_out > ngrid; return dts_grid[g_in]; end
            l_in  = lls[g_in]
            l_out = lls[g_out]
            l_in == l_out && return dts_grid[g_in]
            frac = (threshold - l_in) / (l_out - l_in)
            return dts_grid[g_in] + frac * (dts_grid[g_out] - dts_grid[g_in])
        end
        dt_lo = glo > 1 ? interp(glo, -1) : dts_grid[1]
        dt_hi = ghi < ngrid ? interp(ghi, +1) : dts_grid[ngrid]

        tc_obs[i] = tc_pred + dts_grid[gmax]
        tc_lo[i]  = tc_pred + dt_lo
        tc_hi[i]  = tc_pred + dt_hi
    end

    mle_dt = tc_obs .- tcs_predicted
    lo_dt  = tc_lo  .- tcs_predicted
    hi_dt  = tc_hi  .- tcs_predicted
    return (cycle_idx = cycle_idx,
            tc_obs = tc_obs, tc_lo = tc_lo, tc_hi = tc_hi,
            mle_dt = mle_dt, lo_dt = lo_dt, hi_dt = hi_dt)
end


"""
    ttvc_envelope(chains, params; tcs_observed, planet_a_k=1, planet_b_k=2,
                  M_s=nothing, n_draws=400, seed=42, jmax=5)
        -> NamedTuple{(:med, :lo, :hi)}

Reconstruct the TTV-C (TTVFaster) N-body envelope on `tcs_observed`
from posterior draws in `chains`. Computes the 16/50/84 percentile
TTV-on-planet-a across draws, in days. Returns offsets from the
linear-ephemeris transit centers.
"""
function ttvc_envelope(chains, params::Params;
                        tcs_observed::Vector{Float64},
                        planet_a_k::Int = 1, planet_b_k::Int = 2,
                        M_s::Union{Float64, Nothing} = nothing,
                        n_draws::Int = 400, seed::Int = 42,
                        jmax::Int = 5)
    M_s_use = M_s === nothing ? params.config.M_s : M_s
    isnan(M_s_use) && error("ttvc_envelope: M_s not set in params and not supplied")

    chain_n = size(chains)[1]
    chain_c = size(chains)[3]
    total = chain_n * chain_c
    n_use = min(n_draws, total)
    inds = rand(MersenneTwister(seed), 1:total, n_use)

    n_tr = length(tcs_observed)
    samples = Matrix{Float64}(undef, n_use, n_tr)

    pname_set = Set(string.(names(chains, :parameters)))

    function _get(theta_vals, nm::AbstractString)
        # Helper: pull a parameter value from theta_vals or fall back to layout fixed
        idx = params.layout.name_to_idx[nm]
        return theta_vals[idx]
    end

    for (k, idx) in enumerate(inds)
        chain_idx = ((idx - 1) ÷ chain_n) + 1
        iter_idx  = ((idx - 1) % chain_n) + 1
        theta_k = Theta{Float64}(params)
        set_n_p!(theta_k, params.config.max_kplanet)
        for nm in keys(params.layout.name_to_idx)
            nm in pname_set || continue
            theta_k.values[params.layout.name_to_idx[nm]] =
                chains[iter_idx, Symbol(nm), chain_idx]
        end

        P_a  = _get(theta_k.values, "P_k$(planet_a_k)")
        Tc_a = _get(theta_k.values, "Tc_k$(planet_a_k)")
        K_a  = _get(theta_k.values, "K_k$(planet_a_k)")
        se_a = _get(theta_k.values, "sesinw_k$(planet_a_k)")
        sc_a = _get(theta_k.values, "secosw_k$(planet_a_k)")
        b_a  = _get(theta_k.values, "b_k$(planet_a_k)")
        e_a  = hypot(se_a, sc_a)^2
        w_a  = atan(se_a, sc_a)
        Tp_a = tc_to_tp(Tc_a, P_a, e_a, w_a)
        Tc1_a = Tp_a + P_a / 4

        # a/R* for planet a
        if params.config.parametrization.use_rho_s
            rho = _get(theta_k.values, "rho_s")
            a_Rs_a = rho_s_to_a_Rs(rho, P_a)
        else
            P_s = P_a * 86400.0
            GM = 1.3271244e26 * M_s_use
            a_Rs_a = cbrt(GM * P_s^2 / (4π^2)) / (params.config.R_s * 6.9570e10)
        end
        inc_a = acosd(b_a / a_Rs_a)
        mp_a  = msini(M_s_use, K_a, P_a, e_a) / sind(inc_a)
        mr_a  = mp_a * 1.898e30 / 1.989e33 / M_s_use

        P_b   = _get(theta_k.values, "P_k$(planet_b_k)")
        Tc_b  = _get(theta_k.values, "Tc_k$(planet_b_k)")
        K_b   = _get(theta_k.values, "K_k$(planet_b_k)")
        se_b  = _get(theta_k.values, "sesinw_k$(planet_b_k)")
        sc_b  = _get(theta_k.values, "secosw_k$(planet_b_k)")
        e_b   = hypot(se_b, sc_b)^2
        w_b   = atan(se_b, sc_b)
        # Use a's inclination (coplanar assumption) for non-transiting b
        mp_b  = msini(M_s_use, K_b, P_b, e_b)
        mr_b  = mp_b * 1.898e30 / 1.989e33 / M_s_use

        # TTVFaster has 0/0 at exactly e=0
        e_a_f = max(e_a, 1e-4)
        e_b_f = max(e_b, 1e-4)
        pl_a = Planet_plane_hk(mr_a, P_a, Tc1_a,
                                e_a_f*cos(w_a), e_a_f*sin(w_a))
        pl_b = Planet_plane_hk(mr_b, P_b, Tc_b,
                                e_b_f*cos(w_b), e_b_f*sin(w_b))

        # Pair: inner first
        inner, outer = P_a < P_b ? (pl_a, pl_b) : (pl_b, pl_a)
        n_lo_o = ceil(Int, (minimum(tcs_observed) - outer.trans0) / outer.period)
        n_hi_o = floor(Int, (maximum(tcs_observed) - outer.trans0) / outer.period)
        times_outer = [outer.trans0 + n*outer.period for n in n_lo_o:n_hi_o]
        ttv_inner_v = zeros(Float64, length(tcs_observed))
        ttv_outer_v = zeros(Float64, length(times_outer))
        try
            if P_a < P_b
                compute_ttv!(jmax, inner, outer, tcs_observed, times_outer,
                              ttv_inner_v, ttv_outer_v)
                samples[k, :] .= ttv_inner_v
            else
                compute_ttv!(jmax, inner, outer, times_outer, tcs_observed,
                              ttv_outer_v, ttv_inner_v)
                samples[k, :] .= ttv_inner_v
            end
        catch
            samples[k, :] .= 0.0
        end
        @inbounds for j in 1:n_tr
            isnan(samples[k, j]) && (samples[k, j] = 0.0)
        end
    end

    med = vec(median(samples; dims = 1))
    lo  = [quantile(samples[:, j], 0.16) for j in 1:n_tr]
    hi  = [quantile(samples[:, j], 0.84) for j in 1:n_tr]
    return (med = med, lo = lo, hi = hi)
end
