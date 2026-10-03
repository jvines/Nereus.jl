# The transit window (cadences whose flux is computed) used a half-width of
# 2P(1+k)/(π a_R). That is not an upper bound on the transit: at high e it cut
# real transits short. For P = 4.137 d, k = 0.031, a_R = 12.7, b = 0.3,
# ω = -π/2 it dropped 4.4 % of the in-transit cadences at e = 0.90 and 21 % at
# e = 0.93; at a_R = 3 the loss starts near e = 0.8. The window is now the
# bound from the orbit alone (`_transit_window_halfwidth`; every cadence where
# the orbit gives no bound), plus half the longest exposure when supersampling
# and the largest |δt| of a TTV planet.
#
#   1. superset, data-free: for random orbits and times, a cadence outside the
#      window has flux exactly 1 (z >= 1 + k, or the planet behind the star);
#   2. the case above, in the likelihood: the old window loses cadences, the
#      new one none, and the workspace likelihood equals the model with no
#      window at all;
#   3. random prior draws: the likelihood equals the no-window model at every
#      draw, and differs from the old window's value exactly at the draws where
#      the old window dropped a cadence with the planet in front and z < 1 + k.
using Test
using Nereus
using Random
using Nereus: _transit_window_halfwidth, _set_transit_windows!, _phot_sparse_refresh,
              _phot_chunk_loglik, _phot_transit_product, _phot_trend_cache, QuadLimbDark,
              transit_flux, tp_to_tc, kepler_solve, kipping_q_to_u

include(joinpath(@__DIR__, "fixtures", "transit_gate_target.jl"))

function sup_zsep(t, P, e, ω, Tp, b, aR)    # as `_phot_sparse_refresh`
    ome2 = 1 - e * e; s1 = sqrt(ome2); sω, cω = sincos(ω)
    cos_i = b * ((1 + e * sω) / ome2) / aR; si2 = 1 - cos_i * cos_i
    M = 2π / P * (t - Tp); E = kepler_solve(M, e); sE, cE = sincos(E)
    d = 1 - e * cE; cf = (cE - e) / d; sf = s1 * sE / d
    roa = ome2 / (1 + e * cf); swf = sω * cf + cω * sf
    return aR * roa * sqrt(max(1 - si2 * swf * swf, 0.0)), swf
end
sup_fold(t, Tc, P) = (Δ = t - Tc; Δ - P * round(Δ / P))
sup_old(P, k, aR) = 2 * P / π * (1 + k) / aR

# Workspace log L of a one-planet, one-instrument light curve, and the same
# model with every cadence evaluated (`win = Inf`) or only those in a window.
function sup_case(t, P, e, ω, k, aR, b; ld = QuadLimbDark([0.4, 0.2]))
    n = length(t)
    Tc = TG_BJD0 + 0.5; Tp = Nereus.tc_to_tp(Tc, P, e, ω)
    # flux: the model itself plus a fixed ripple, so every cadence matters
    flux = [1 + 1e-4 * sin(37.0 * i) for i in 1:n]
    data = Data(; t_phot = t, flux = flux, flux_err = fill(1e-4, n), phot_inst = ones(Int, n))
    params = Params(; max_kplanet = 1, planet_modes = [PM_ONLY],
                      instruments = InstrumentConfig(pm = ["TESS"]), data = data,
                      parametrization = ParametrizationConfig(time = :Tc, use_rho_s = true),
                      priors = Dict{String, PriorSpec}("P_k1" => UniformPrior(0.99P, 1.01P),
                                                       "Tc_k1" => UniformPrior(Tc - 0.1, Tc + 0.1)),
                      stability = :none)
    th = Nereus.Theta{Float64}(params)
    rho = aR^3 / (Nereus.rho_s_to_a_Rs(1.0, P))^3
    vals = Dict("P_k1" => P, "Tc_k1" => Tc, "b_k1" => b, "rr_k1" => k, "rho_s" => rho,
                "sesinw_k1" => sqrt(e) * sin(ω), "secosw_k1" => sqrt(e) * cos(ω),
                "q1_TESS" => 0.36, "q2_TESS" => 0.33, "offset_TESS" => 0.0, "jitter_TESS" => 1e-4)
    for nm in params.layout.unfrozen_names
        Nereus.set_param!(th, nm, get(vals, nm, 0.0))
    end
    ws = Nereus.PTWorkspace(params, 1, 0; n_obs = 0, n_phot = n)
    val = Nereus.transit_log_likelihood(th, data, ws)
    P_, e_, ω_, Tp_ = ws.transit_Ps[1], ws.transit_es[1], ws.transit_ws[1], ws.transit_Tps[1]
    b_, aR_, k_ = ws.transit_bs[1], ws.transit_a_Rs[1], ws.transit_rrs[1]
    Tc_ = tp_to_tc(Tp_, P_, e_, ω_)
    lds = [QuadLimbDark(collect(kipping_q_to_u(Nereus.get_param(th, "q1_TESS"),
                                               Nereus.get_param(th, "q2_TESS"))))]
    function ref(win)
        idx = [i for i in 1:n if abs(sup_fold(t[i], Tc_, P_)) <= win]
        fc = ones(1, n)
        _phot_sparse_refresh(1, length(idx), 1, data, idx, P_, e_, ω_, Tp_, b_, aR_, k_, lds, fc)
        return _phot_chunk_loglik(1, n, data, 1, [true], fc, [Nereus.pm_offset(th, 1)],
                                  [Nereus.pm_jitter(th, 1)], [Nereus.pm_dilution(th, 1)],
                                  data.t_ref, 0.0, _phot_trend_cache(th, 1), 2π)
    end
    intr = [i for i in 1:n if (z = sup_zsep(t[i], P_, e_, ω_, Tp_, b_, aR_); z[1] < 1 + k_ && z[2] > 0)]
    old = sup_old(P_, k_, aR_)
    lost_old = count(i -> abs(sup_fold(t[i], Tc_, P_)) > old, intr)
    lost_new = length(setdiff(intr, ws.transit_in_idx[1]))
    return (; val, ref_all = ref(Inf), ref_old = ref(old), n_in = length(intr), lost_old, lost_new)
end

@testset "transit window holds the whole transit" begin
    ld = QuadLimbDark([0.4, 0.2])

    @testset "superset, data-free" begin
        rng = MersenneTwister(77)
        n_win = 0; n_out = 0; n_near = 0; bad = 0
        for _ in 1:30_000
            P  = exp(log(0.3) + rand(rng) * (log(300) - log(0.3)))
            e  = rand(rng) < 0.15 ? 0.0 : rand(rng) < 0.6 ? 0.99989 * (1 - 10^(-4 * rand(rng))) :
                 0.99989 * rand(rng)
            ω  = 2π * rand(rng) - π
            k  = exp(log(1e-3) + rand(rng) * (log(0.5) - log(1e-3)))
            aR = exp(log(1.5) + rand(rng) * (log(1000) - log(1.5)))
            b  = (1 + k) * rand(rng)
            Tp = TG_BJD0 + 50 * rand(rng); Tc = tp_to_tc(Tp, P, e, ω)
            hw = _transit_window_halfwidth(P, e, ω, k, aR, max(abs(Tc), abs(Tp)))
            isfinite(hw) || continue
            n_win += 1
            for j in 1:40
                Δ = j <= 15 ? (rand(rng) - 0.5) * P :                     # anywhere
                    j <= 30 ? hw * (1 + 10.0^(-10 * rand(rng))) :         # just outside
                              hw + (sup_old(P, k, aR) - hw) * rand(rng)   # old-window band
                Δ = rand(rng) < 0.5 ? -Δ : Δ
                t = Tc + Δ + P * rand(rng, -1000:1000)
                abs(sup_fold(t, Tc, P)) > hw || continue
                n_out += 1; j > 15 && j <= 30 && (n_near += 1)
                z, swf = sup_zsep(t, P, e, ω, Tp, b, aR)
                (swf <= 0 || transit_flux(ld, z, k) === 1.0) || (bad += 1)
            end
        end
        @info "window superset" n_win n_out n_near
        @test n_win > 15_000 && n_out > 400_000 && n_near > 200_000
        @test bad == 0
        # no bound -> every cadence
        @test _transit_window_halfwidth(4.0, 0.95, 0.0, 0.1, 12.7, 0.0) == Inf     # s = 1.7
    end

    @testset "high-e case the old window cut short" begin
        P, k, aR, b, ω = 4.13747, 0.031, 12.7, 0.3, -π / 2
        t = collect(range(TG_BJD0 + 0.5 - 0.6, TG_BJD0 + 0.5 + 0.6; step = 1 / 1440))
        for (e, lost) in ((0.80, false), (0.90, true), (0.93, true))
            r = sup_case(t, P, e, ω, k, aR, b)
            @test r.n_in > 50
            @test (r.lost_old > 0) == lost          # the old window's defect
            @test r.lost_new == 0                   # the new window holds the transit
            @test r.val === r.ref_all
            @test (r.val !== r.ref_old) == lost
            lost && @info "e = $e: old window lost $(r.lost_old) of $(r.n_in) in-transit cadences, Δ log L = $(r.val - r.ref_old)"
        end
        # small a/R*: the loss starts at lower e
        r = sup_case(collect(range(TG_BJD0 - 0.6, TG_BJD0 + 1.6; step = 1 / 1440)),
                     2.0, 0.8, -π / 2, 0.1, 3.0, 0.3)
        @test r.lost_old > 0 && r.lost_new == 0 && r.val === r.ref_all
    end

    @testset "non-workspace product: window loses nothing" begin
        # A cadence the old window dropped now carries its dip; supersampled too.
        P, k, aR, b, ω, e = 4.13747, 0.031, 12.7, 0.3, -π / 2, 0.93
        Tp = TG_BJD0; Tc = tp_to_tc(Tp, P, e, ω)
        for (texp, ns) in ((0.0, 1), (1800 / 86_400, 30))
            win = _set_transit_windows!([Inf], 1, [true], [0], nothing, [P], [e], [ω], [k], [aR],
                                        [Tc], [Tp], texp / 2)
            old = [sup_old(P, k, aR)]
            n_fixed = 0; same = true
            for t in range(Tc - 0.5, Tc + 0.5; length = 4001)
                pr(w) = _phot_transit_product(t, texp, ld, 1, [true], [P], [e], [ω], [Tp], [b],
                                              [aR], [k], [Tc], w, [0], nothing, ns)
                same &= pr(win) === pr([Inf])
                n_fixed += pr(old) !== pr([Inf])
            end
            @test same
            @test n_fixed > 10
        end
    end

    @testset "random prior draws" begin
        n_A_total = 0
        for (exposure, ttv) in ((false, false), (true, false), (false, true))
            tg = transit_gate_target(; exposure, ttv)
            data = tg.data; nd = length(data.t_phot)
            th = Nereus.Theta{Float64}(tg.params); ws = transit_gate_ws(tg)
            sys = tg.params.layout.systemic
            n_cmp = 0; n_A = 0; n_A_changed = 0; n_other_changed = 0
            for v in transit_gate_points(tg, 300; seed = 71)
                th.values .= v
                P = Nereus.planet_P(th, 1); e, ω = Nereus.planet_e_w(th, 1)
                (0 <= e < 1) || continue
                Tp = Nereus.tc_to_tp(Nereus.planet_time_anchor(th, 1), P, e, ω)
                b, k = Nereus.planet_b_rr(th, 1)
                b < 1 + k || continue
                aR = Nereus.rho_s_to_a_Rs(Nereus.rho_s(th), P)
                Tc = tp_to_tc(Tp, P, e, ω)
                lds = [QuadLimbDark(collect(kipping_q_to_u(th.values[sys.ld_q1[ix]],
                                                           th.values[sys.ld_q2[ix]]))) for ix in 1:2]
                n_super = Nereus._phot_n_super(data)
                _, ttv_state = Nereus._decode_ttv_state(th, Nereus.planet_indices(th))
                rj = [ttv ? 1 : 0]
                win = _set_transit_windows!([Inf], 1, [true], rj, ttv_state, [P], [e], [ω], [k],
                                            [aR], [Tc], [Tp],
                                            n_super > 1 ? Nereus._phot_max_exposure(data) / 2 : 0.0)
                old = [sup_old(P, k, aR)]
                A = false; same = true
                for i in 1:nd
                    texp = isempty(data.exposure_times) ? 0.0 : data.exposure_times[i]
                    pr(w) = _phot_transit_product(data.t_phot[i], texp, lds[data.phot_inst[i]], 1,
                                                  [true], [P], [e], [ω], [Tp], [b], [aR], [k], [Tc],
                                                  w, rj, ttv_state, n_super)
                    p_all = pr([Inf])
                    same &= pr(win) === p_all
                    A |= pr(old) !== p_all
                end
                @test same
                # likelihood: the workspace value equals the no-window model
                val = Nereus.transit_log_likelihood(th, data, ws)
                if !exposure && !ttv
                    fc_all = ones(1, nd); fc_old = ones(1, nd)
                    _phot_sparse_refresh(1, nd, 1, data, collect(1:nd), P, e, ω, Tp, b, aR, k, lds, fc_all)
                    io = [i for i in 1:nd if abs(sup_fold(data.t_phot[i], Tc, P)) <= old[1]]
                    _phot_sparse_refresh(1, length(io), 1, data, io, P, e, ω, Tp, b, aR, k, lds, fc_old)
                    ll(fc) = _phot_chunk_loglik(1, nd, data, 1, [true], fc,
                                                [Nereus.pm_offset(th, ix) for ix in 1:2],
                                                [Nereus.pm_jitter(th, ix) for ix in 1:2],
                                                [Nereus.pm_dilution(th, ix) for ix in 1:2],
                                                data.t_ref, 0.0, _phot_trend_cache(th, 2), 2π)
                    @test val === ll(fc_all)
                    changed = val !== ll(fc_old)
                    @test changed == A
                    n_A_changed += A && changed
                    n_other_changed += !A && changed
                end
                n_cmp += 1; n_A += A
            end
            @info "random draws" exposure ttv n_cmp n_A n_A_changed n_other_changed
            @test n_cmp > 150
            n_A_total += n_A
        end
        @test n_A_total >= 1      # the old window cut some draw's transit short
    end
end
