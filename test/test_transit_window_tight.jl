# The transit window (the cadences whose flux is computed) was narrowed from
# 2P(1+k)/(π a_R), about four circular half-durations, to a bound from the
# orbit (`_transit_window_halfwidth`) wherever that provably skipped only
# cadences whose flux is exactly 1, which kept the likelihood bit-identical.
# Since the old window was found to cut real transits short at high e
# (test_transit_window_superset.jl), the window is the bound alone. What this
# file checks holds for it:
#
#   1. data-free: for random orbits (e up to 0.9999), no cadence outside the
#      bound has the planet in front of the star with z < 1 + k, and every
#      cadence the bound drops from the old window has flux exactly 1;
#   2. workspace likelihood: === the old window's value on random prior draws,
#      wherever the old window held the whole transit;
#   3. non-workspace paths (supersampled exposures, TTV): the per-cadence
#      transit product is === with the window and with no window at all.
using Test
using Nereus
using Random
using Nereus: _transit_window_halfwidth, _set_transit_windows!,
              _phot_sparse_refresh, _phot_chunk_loglik, _phot_transit_product,
              _phot_trend_cache, QuadLimbDark, transit_flux, tp_to_tc, tc_to_tp,
              kepler_solve, kipping_q_to_u

include(joinpath(@__DIR__, "fixtures", "transit_gate_target.jl"))

# z and sin(ω+f) exactly as `_phot_sparse_refresh` computes them.
function tw_zsep(t, P, e, ω, Tp, b, aR)
    ome2 = 1 - e * e; s1 = sqrt(ome2); sω, cω = sincos(ω)
    cos_i = b * ((1 + e * sω) / ome2) / aR; si2 = 1 - cos_i * cos_i
    M = 2π * (t - Tp) / P; E = iszero(e) ? M : kepler_solve(M, e); sE, cE = sincos(E)
    d = 1 - e * cE; cf = (cE - e) / d; sf = s1 * sE / d
    roa = ome2 / (1 + e * cf); swf = sω * cf + cω * sf
    return aR * roa * sqrt(max(1 - si2 * swf * swf, 0.0)), swf
end
tw_fold(t, Tc, P) = (Δ = t - Tc; Δ - P * round(Δ / P))
tw_old(P, k, aR) = 2 * P / π * (1 + k) / aR

# Random orbit, half of them at high e. Times in BJD.
function tw_orbit(rng)
    P  = exp(log(0.3) + rand(rng) * (log(300) - log(0.3)))
    e  = rand(rng) < 0.15 ? 0.0 : rand(rng) < 0.5 ? 1 - 10^(-4 * rand(rng)) : rand(rng)
    ω  = 2π * rand(rng) - π
    k  = exp(log(1e-3) + rand(rng) * (log(0.5) - log(1e-3)))
    aR = exp(log(1.5) + rand(rng) * (log(300) - log(1.5)))
    b  = (1 + k) * rand(rng)
    Tp = 2_460_000.0 + 50 * rand(rng)
    return (; P, e, ω, k, aR, b, Tp, Tc = tp_to_tc(Tp, P, e, ω))
end

@testset "narrower transit window" begin
    ld = QuadLimbDark([0.4, 0.2])

    @testset "superset, data-free" begin
        rng = MersenneTwister(2026)
        n_bound = 0; n_tight = 0; n_out = 0; n_band = 0
        bad_bound = 0; bad_band = 0
        for _ in 1:20_000
            o = tw_orbit(rng)
            e = min(o.e, 0.99989)
            (0.0 <= e < 1) || continue
            ts = max(abs(o.Tc), abs(o.Tp))
            hw = _transit_window_halfwidth(o.P, e, o.ω, o.k, o.aR, ts)
            old = tw_old(o.P, o.k, o.aR)
            g = min(old, hw)
            isfinite(hw) && (n_bound += 1)
            g < old && (n_tight += 1)
            for j in 1:60
                # whole orbit, just outside the bound, and the dropped band
                Δ = j <= 20 ? (rand(rng) - 0.5) * o.P :
                    j <= 40 ? (isfinite(hw) ? hw * (1 + 10.0^(-9 * rand(rng))) : 0.0) :
                              g + (old - g) * rand(rng)
                Δ = rand(rng) < 0.5 ? -Δ : Δ
                t = o.Tc + Δ + o.P * rand(rng, -400:400)
                Δf = abs(tw_fold(t, o.Tc, o.P))
                z, swf = tw_zsep(t, o.P, e, o.ω, o.Tp, o.b, o.aR)
                if Δf > hw
                    n_out += 1
                    (z < 1 + o.k && swf > 0) && (bad_bound += 1)
                end
                if g < Δf <= old
                    n_band += 1
                    (swf <= 0 || transit_flux(ld, z, o.k) === 1.0) || (bad_band += 1)
                end
            end
        end
        @info "window superset" n_bound n_tight n_out n_band
        @test n_bound > 5_000 && n_tight > 5_000 && n_band > 100_000
        @test bad_bound == 0     # nothing in front of the star outside the bound
        @test bad_band == 0      # every dropped cadence has flux exactly 1 (or is behind)
    end

    @testset "no bound where none exists" begin
        @test _transit_window_halfwidth(4.0, 0.5, 0.3, 0.1, 2.0, 0.0) == Inf   # s >= 1
        @test _transit_window_halfwidth(4.0, 0.99995, 0.3, 0.1, 1e6, 0.0) == Inf
        @test _transit_window_halfwidth(4.0, -0.1, 0.3, 0.1, 10.0, 0.0) == Inf
        @test _transit_window_halfwidth(4.0, NaN, 0.3, 0.1, 10.0, 0.0) == Inf
        @test _transit_window_halfwidth(4.0, 0.1, 0.3, 0.1, 0.0, 0.0) == Inf
        @test _transit_window_halfwidth(-4.0, 0.1, 0.3, 0.1, 10.0, 0.0) == Inf
        # circular: asin((1+k)/aR) of phase, i.e. the b = 0 half-duration
        hw = _transit_window_halfwidth(4.0, 0.0, 0.3, 0.1, 10.0, 0.0)
        @test hw ≈ asin(0.11) / (2π) * 4.0 rtol = 1e-6     # plus the margins
        @test hw < tw_old(4.0, 0.1, 10.0) / 3
    end

    @testset "workspace likelihood === old window" begin
        tg = transit_gate_target()
        data = tg.data; n = length(data.t_phot)
        th = Nereus.Theta{Float64}(tg.params); ws = transit_gate_ws(tg)
        n_pm = 2; sys = tg.params.layout.systemic
        n_cmp = 0; n_narrow = 0
        for v in transit_gate_points(tg, 400; seed = 31)
            th.values .= v
            val = Nereus.transit_log_likelihood(th, data, ws)
            P, e, ω = ws.transit_Ps[1], ws.transit_es[1], ws.transit_ws[1]
            Tp, b, aR, rr = ws.transit_Tps[1], ws.transit_bs[1], ws.transit_a_Rs[1],
                            ws.transit_rrs[1]
            (isfinite(val) && 0 <= e < 1 && b < 1 + rr) || continue
            Tc = tp_to_tc(Tp, P, e, ω); old = tw_old(P, rr, aR)
            idx = [i for i in 1:n if abs(tw_fold(data.t_phot[i], Tc, P)) <= old]
            # skip draws where the old window cut the transit short
            any(i -> !(i in idx) && (zs = tw_zsep(data.t_phot[i], P, e, ω, Tp, b, aR);
                                     zs[1] < 1 + rr && zs[2] > 0), 1:n) && continue
            length(ws.transit_in_idx[1]) < length(idx) && (n_narrow += 1)
            lds = [QuadLimbDark(collect(kipping_q_to_u(th.values[sys.ld_q1[ix]],
                                                       th.values[sys.ld_q2[ix]]))) for ix in 1:n_pm]
            fc = ones(1, n)
            _phot_sparse_refresh(1, length(idx), 1, data, idx, P, e, ω, Tp, b, aR, rr, lds, fc)
            ref = _phot_chunk_loglik(1, n, data, 1, [true], fc,
                                     [Nereus.pm_offset(th, ix) for ix in 1:n_pm],
                                     [Nereus.pm_jitter(th, ix) for ix in 1:n_pm],
                                     [Nereus.pm_dilution(th, ix) for ix in 1:n_pm],
                                     data.t_ref, 0.0, _phot_trend_cache(th, n_pm), 2π)
            @test val === ref
            n_cmp += 1
        end
        @info "workspace likelihood vs old window" n_cmp n_narrow
        @test n_cmp > 150 && n_narrow > 50
    end

    for (exposure, ttv) in ((true, false), (false, true))
        @testset "per-cadence product, exposure = $exposure, ttv = $ttv" begin
            tg = transit_gate_target(; exposure, ttv)
            data = tg.data; n = length(data.t_phot)
            th = Nereus.Theta{Float64}(tg.params)
            sys = tg.params.layout.systemic
            n_super = Nereus._phot_n_super(data)
            @test (n_super > 1) == exposure
            n_cmp = 0; n_narrow = 0
            for v in transit_gate_points(tg, 60; seed = 41)
                th.values .= v
                P = Nereus.planet_P(th, 1); e, ω = Nereus.planet_e_w(th, 1)
                (0 <= e < 1) || continue
                Tp = tc_to_tp(Nereus.planet_time_anchor(th, 1), P, e, ω)
                b, rr = Nereus.planet_b_rr(th, 1)
                b < 1 + rr || continue
                aR = Nereus.rho_s_to_a_Rs(Nereus.rho_s(th), P)
                Tc = tp_to_tc(Tp, P, e, ω)
                _, ttv_state = Nereus._decode_ttv_state(th, Nereus.planet_indices(th))
                r_for_j = [ttv ? 1 : 0]
                old = [tw_old(P, rr, aR)]
                # the likelihood's window: pad 0 (each cadence adds its own
                # reach) and the cos i term
                tight = _set_transit_windows!([Inf], 1, [true], r_for_j, ttv_state, [P], [e], [ω],
                                              [rr], [aR], [Tc], [Tp], 0.0, [b])
                tight[1] < old[1] && (n_narrow += 1)
                lds = [QuadLimbDark(collect(kipping_q_to_u(th.values[sys.ld_q1[ix]],
                                                           th.values[sys.ld_q2[ix]]))) for ix in 1:2]
                same = true
                for i in 1:n
                    texp = isempty(data.exposure_times) ? 0.0 : data.exposure_times[i]
                    ld = lds[data.phot_inst[i]]
                    pr(w) = _phot_transit_product(data.t_phot[i], texp, ld, 1, [true], [P], [e],
                                                  [ω], [Tp], [b], [aR], [rr], [Tc], w, r_for_j,
                                                  ttv_state, n_super)
                    same &= pr(tight) === pr([Inf])
                end
                @test same
                n_cmp += 1
            end
            @info "per-cadence product" exposure ttv n_cmp n_narrow
            @test n_cmp > 20
            @test n_narrow > 5
        end
    end
end
