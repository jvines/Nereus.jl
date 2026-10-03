# The sky separation z depends on sin²(ω+f) only, so it is as small at the
# occultation (planet BEHIND the star, sin(ω+f) < 0) as at the transit. On an
# eccentric orbit the occultation can come within the transit window, and the
# model then put a transit-shaped dip there. A cadence with the planet behind
# the star now has flux exactly 1 (the gravity-darkened model already did this
# through its line-of-sight coordinate).
#
# What changes is exactly those cadences:
#   * per cadence, for the workspace refresh and for the non-workspace transit
#     product (instantaneous and supersampled), the flux is 1 when the planet is
#     behind and bit-identical to before when it is in front;
#   * on random prior draws, the workspace likelihood differs from the old
#     model (same window, no sign test) exactly at the draws with an in-window
#     cadence behind the star with z < 1 + k, and equals the corrected model
#     everywhere. Draws where the old window misses part of a real transit are
#     left to test_transit_window_superset.jl.
using Test
using Nereus
using Random
using Nereus: _phot_sparse_refresh, _phot_chunk_loglik, _phot_transit_product,
              _phot_trend_cache, QuadLimbDark, transit_flux, tp_to_tc, tc_to_tp,
              kepler_solve, kipping_q_to_u, _sky_separation_signed

include(joinpath(@__DIR__, "fixtures", "transit_gate_target.jl"))

function occ_zsep(t, P, e, ω, Tp, b, aR)    # as `_phot_sparse_refresh`
    ome2 = 1 - e * e; s1 = sqrt(ome2); sω, cω = sincos(ω)
    cos_i = b * ((1 + e * sω) / ome2) / aR; si2 = 1 - cos_i * cos_i
    M = 2π / P * (t - Tp); E = kepler_solve(M, e); sE, cE = sincos(E)
    d = 1 - e * cE; cf = (cE - e) / d; sf = s1 * sE / d
    roa = ome2 / (1 + e * cf); swf = sω * cf + cω * sf
    return aR * roa * sqrt(max(1 - si2 * swf * swf, 0.0)), swf
end
occ_fold(t, Tc, P) = (Δ = t - Tc; Δ - P * round(Δ / P))

@testset "occultation is not a transit" begin
    ld = QuadLimbDark([0.4, 0.2])
    # e = 0.8, ω = 0, b = 0.3: periastron falls between the two conjunctions,
    # which are 0.215 d apart; the occultation crosses the disc 0.3 R* from its
    # centre, partly inside the old window (±0.228 d).
    P, e, ω, k, b, aR = 4.137, 0.8, 0.0, 0.1, 0.3, 12.7
    Tp = 2_460_000.3; Tc = tp_to_tc(Tp, P, e, ω)
    t = collect(range(Tc - P / 2, Tc + P / 2; length = 40_001))
    n = length(t)

    @testset "workspace refresh, per cadence" begin
        data = Data(; t_phot = t, flux = ones(n), flux_err = fill(1e-3, n),
                      phot_inst = ones(Int, n))
        fc = ones(1, n)
        _phot_sparse_refresh(1, n, 1, data, collect(1:n), P, e, ω, Tp, b, aR, k, [ld], fc)
        n_behind_in = 0; n_front_in = 0; ok = true
        for i in 1:n
            z, swf = occ_zsep(t[i], P, e, ω, Tp, b, aR)
            if swf > 0
                ok &= fc[1, i] === transit_flux(ld, z, k)
                n_front_in += z < 1 + k
            else
                ok &= fc[1, i] === 1.0
                n_behind_in += z < 1 + k
            end
        end
        @test ok
        @test n_behind_in > 100 && n_front_in > 100   # both conjunctions on the disc
        @test minimum(fc) < 0.99
        # the occultation lies inside the old window
        old = 2P / π * (1 + k) / aR
        @test any(i -> abs(occ_fold(t[i], Tc, P)) <= old &&
                       (zs = occ_zsep(t[i], P, e, ω, Tp, b, aR); zs[1] < 1 + k && zs[2] <= 0), 1:n)
    end

    @testset "non-workspace transit product, per cadence" begin
        texp = 1800 / 86_400
        ok_i = true; ok_s = true; n_behind = 0
        for ti in t[1:7:end]
            prod_i = _phot_transit_product(ti, 0.0, ld, 1, [true], [P], [e], [ω], [Tp], [b],
                                           [aR], [k], [Tc], [Inf], [0], nothing, 1)
            z, sw = _sky_separation_signed(ti, P, e, ω, Tp, b, aR)
            @test z === sky_separation(ti, P, e, ω, Tp, b, aR)
            ok_i &= prod_i === (sw > 0 ? transit_flux(ld, z, k) : 1.0)
            n_behind += (sw <= 0 && z < 1 + k)
            # supersampled: 30 sub-samples, each with the same rule
            prod_s = _phot_transit_product(ti, texp, ld, 1, [true], [P], [e], [ω], [Tp], [b],
                                           [aR], [k], [Tc], [Inf], [0], nothing, 30)
            fsum = 0.0
            for s in 1:30
                zs, sws = _sky_separation_signed(ti + ((2s - 31) / 60) * texp, P, e, ω, Tp, b, aR)
                fsum += sws > 0 ? transit_flux(ld, zs, k) : 1.0
            end
            ok_s &= prod_s === 1.0 * (fsum / 30)
        end
        @test ok_i && ok_s
        @test n_behind > 10
    end

    @testset "likelihood changes exactly where an occultation was in the window" begin
        tg = transit_gate_target()
        data = tg.data; nd = length(data.t_phot)
        th = Nereus.Theta{Float64}(tg.params); ws = transit_gate_ws(tg)
        sys = tg.params.layout.systemic
        n_cmp = 0; n_B = 0; n_A = 0
        for v in transit_gate_points(tg, 800; seed = 61)
            th.values .= v
            val = Nereus.transit_log_likelihood(th, data, ws)
            Pj, ej, ωj = ws.transit_Ps[1], ws.transit_es[1], ws.transit_ws[1]
            Tpj, bj, aRj, kj = ws.transit_Tps[1], ws.transit_bs[1], ws.transit_a_Rs[1],
                               ws.transit_rrs[1]
            (isfinite(val) && 0 <= ej < 1 && bj < 1 + kj) || continue
            Tcj = tp_to_tc(Tpj, Pj, ej, ωj); old = 2Pj / π * (1 + kj) / aRj
            lds = [QuadLimbDark(collect(kipping_q_to_u(th.values[sys.ld_q1[ix]],
                                                       th.values[sys.ld_q2[ix]]))) for ix in 1:2]
            f_old = ones(1, nd); f_fix = ones(1, nd)       # old model / corrected model
            A = false; B = false
            for i in 1:nd
                z, swf = occ_zsep(data.t_phot[i], Pj, ej, ωj, Tpj, bj, aRj)
                inwin = abs(occ_fold(data.t_phot[i], Tcj, Pj)) <= old
                A |= (!inwin && z < 1 + kj && swf > 0)
                B |= (inwin && z < 1 + kj && swf <= 0)
                fl = transit_flux(lds[data.phot_inst[i]], z, kj)
                inwin && (f_old[1, i] = fl)
                (inwin && swf > 0) && (f_fix[1, i] = fl)
            end
            n_A += A
            A && continue
            ll(fc) = _phot_chunk_loglik(1, nd, data, 1, [true], fc,
                                        [Nereus.pm_offset(th, ix) for ix in 1:2],
                                        [Nereus.pm_jitter(th, ix) for ix in 1:2],
                                        [Nereus.pm_dilution(th, ix) for ix in 1:2],
                                        data.t_ref, 0.0, _phot_trend_cache(th, 2), 2π)
            @test val === ll(f_fix)
            @test (val !== ll(f_old)) == B
            n_B += B; n_cmp += 1
        end
        @info "occultation in the old window" n_cmp n_B n_A
        @test n_cmp > 400 && n_B >= 5
    end
end
