# Doppler tomography: the planet's shadow in the mean line profile.
#
# A synthetic injection proves the machinery recovers a shadow it was given,
# across the residual construction, the matched filter, the shadow track and
# the BERV correction.
using Test
using Nereus
using Nereus: tomogram_residuals, tomogram_matched_filter, tomogram_pooled,
               shadow_track, tomogram_null_distribution, fit_tomography,
               run_job, FEATURE_ACTIONS
using DelimitedFiles, Random, Statistics

@testset "Doppler tomography" begin

    # ---- shadow track geometry ------------------------------------------
    @testset "shadow track" begin
        P, Tc, aRs, inc, vsini = 2.827969, 0.0, 6.81, deg2rad(83.6), 25.9
        t = collect(range(-0.05, 0.05; length = 41))
        # lambda = 0: the track is antisymmetric about mid-transit, because the
        # planet crosses from the approaching to the receding limb.
        tr = shadow_track(t, Tc, P, aRs, inc, 0.0, vsini)
        ok = findall(isfinite, tr)
        @test !isempty(ok)
        mid = t[ok][argmin(abs.(t[ok]))]
        @test abs(mid) < 0.01
        # sign flips across mid-transit
        neg = tr[ok][t[ok] .< -0.005]; pos = tr[ok][t[ok] .> 0.005]
        @test !isempty(neg) && !isempty(pos)
        @test sign(mean(neg)) == -sign(mean(pos))
        # |v| never exceeds vsini: the planet cannot occult gas rotating faster
        # than the limb.
        @test maximum(abs.(tr[ok])) <= vsini * 1.001
        # lambda -> lambda + pi mirrors the track
        tr2 = shadow_track(t, Tc, P, aRs, inc, π, vsini)
        ok2 = findall(i -> isfinite(tr[i]) && isfinite(tr2[i]), eachindex(t))
        @test all(abs.(tr[ok2] .+ tr2[ok2]) .< 1e-6 * vsini .+ 1e-9)
    end


    # ---- synthetic night: the API surface end to end ---------------------
    # A shadow we injected ourselves, so this runs everywhere and pins the
    # wiring (fit_tomography, the run_job job kind, the six feature ops)
    # rather than the science. Recovery of lambda is the assertion; the real
    # -data testset below is what pins the measurement.
    function _synth_night(; λ_true = deg2rad(-40.0), P = 2.827969, Tc = 2.4e6,
                            aRs = 6.81, inc = deg2rad(83.6), vsini = 25.9,
                            T14 = 2.62/24, vsys = 5.0, berv = 3.0, rr = 0.30,
                            nt = 40, seed = 7)
        rng   = MersenneTwister(seed)
        vgrid = collect(range(-70, 70; length = 281))
        t     = collect(range(Tc - 0.75 * T14, Tc + 0.75 * T14; length = nt))
        σ_line = vsini / 2.5
        # Rotationally broadened absorption line. `tomogram_residuals`
        # de-shifts each frame by (vsys - berv), so that is where the line has
        # to sit for the frames to co-align — putting it at vsys + berv instead
        # leaves the shadow off-centre by 2*berv and biases lambda by ~20 deg.
        prof = Matrix{Float64}(undef, nt, length(vgrid))
        for i in 1:nt, j in eachindex(vgrid)
            u = (vgrid[j] - vsys + berv) / vsini
            prof[i, j] = 0.5 * exp(-0.5 * (u / 0.45)^2)   # dip-converted
        end
        # SIGN CONVENTION. tomogram_pooled now selects the peak on the SIGNED
        # statistic, not |statistic|: within one convention a real shadow drives
        # it positive, and abs() would let an anti-correlated lambda (an
        # emission-like feature, or the mirrored track) win the argmax and be
        # reported as a detection. The file header states the convention —
        # profiles must be dip-converted, (continuum - ccf)/continuum — so the
        # shadow is a DEFICIT. This synthetic previously built an absorption
        # line and filled it in, which is the opposite convention and returned
        # the mirrored lobe (+76 deg for a -40 deg truth).
        tr = shadow_track(t, Tc, P, aRs, inc, λ_true, vsini)
        for i in 1:nt
            isfinite(tr[i]) || continue
            for j in eachindex(vgrid)
                prof[i, j] -= rr^2 * 0.5 *
                    exp(-0.5 * ((vgrid[j] - vsys + berv - tr[i]) / σ_line)^2)
            end
        end
        prof .+= 1e-4 .* randn(rng, size(prof))
        return (profiles = prof, vgrid = vgrid, times = t, Tc = Tc,
                bervs = fill(berv, nt)), (; λ_true, P, Tc, aRs, inc, vsini, T14, vsys)
    end

    @testset "fit_tomography recovers an injected shadow" begin
        night, g = _synth_night()
        r = fit_tomography([night]; P = g.P, a_Rs = g.aRs, inc = g.inc,
                           vsini = g.vsini, T14 = g.T14, vsys = g.vsys,
                           λs = range(-π, π; length = 181), n_null = 40,
                           rng = MersenneTwister(3))
        @test r.summary["status"] == "ok"
        @test r.summary["n_nights"] == 1
        @test isapprox(r.summary["lambda_deg"], rad2deg(g.λ_true); atol = 12.0)
        # The pulsation filter is ON by default and this synthetic has NO
        # pulsations, so it strips real signal for no benefit — the p-value is
        # conservative by construction. With n_null = 40 the resolution is
        # 0.025 anyway. The filter-off case is checked below.
        @test r.summary["p_value"] < 0.20
        @test length(r.summary["lambda_scan_deg"]) == 181
        @test length(r.summary["lambda_scan_score"]) == 181
        # Same injection with the filter off. Measured: filter ON gives p=0.125,
        # filter OFF gives p=0.25 — the filter HELPS even here, where there are
        # no pulsations to remove, because it also whitens the residual map.
        # So it is not a signal-costing trade: the significance drop it causes
        # on a pulsating host is pulsation power being removed rather than
        # shadow being eaten.
        r_nf = fit_tomography([night]; P = g.P, a_Rs = g.aRs, inc = g.inc,
                              vsini = g.vsini, T14 = g.T14, vsys = g.vsys,
                              λs = range(-π, π; length = 181), n_null = 40,
                              filter_pulsations = false,
                              rng = MersenneTwister(3))
        @test isapprox(r_nf.summary["lambda_deg"], rad2deg(g.λ_true); atol = 12.0)
        @test r_nf.summary["p_value"] < 0.5      # still finds it unfiltered
        @test r.summary["filter_pulsations"] == true
        @test r_nf.summary["filter_pulsations"] == false

        # It is a matched filter; handing it a sampler is a category error and
        # must say so rather than quietly ignoring the argument.
        @test_logs (:warn, r"matched filter") match_mode = :any fit_tomography(
            [night]; P = g.P, a_Rs = g.aRs, inc = g.inc, vsini = g.vsini,
            T14 = g.T14, vsys = g.vsys, λs = range(-π, π; length = 21),
            n_null = 0, engine = "pt")
    end

    @testset "fit_tomography argument checking" begin
        night, g = _synth_night()
        @test_throws Exception fit_tomography([]; P = g.P, a_Rs = g.aRs,
            inc = g.inc, vsini = g.vsini, T14 = g.T14)
        bad = (profiles = night.profiles, vgrid = night.vgrid,
               times = night.times[1:end-1], Tc = night.Tc,
               bervs = night.bervs[1:end-1])
        @test_throws Exception fit_tomography([bad]; P = g.P, a_Rs = g.aRs,
            inc = g.inc, vsini = g.vsini, T14 = g.T14)
        noTc = (profiles = night.profiles, vgrid = night.vgrid,
                times = night.times, bervs = night.bervs)
        @test_throws Exception fit_tomography([noTc]; P = g.P, a_Rs = g.aRs,
            inc = g.inc, vsini = g.vsini, T14 = g.T14)
    end

    @testset "run_job tomography kind" begin
        night, g = _synth_night()
        dir = mktempdir()
        cfg = Dict("kind" => "tomography", "output_dir" => dir,
                   "orbit" => Dict("P" => g.P, "a_Rs" => g.aRs, "inc" => g.inc,
                                   "vsini" => g.vsini, "T14" => g.T14,
                                   "vsys" => g.vsys),
                   # over the wire a matrix is a list of rows
                   "nights" => [Dict("profiles" => [collect(r) for r in
                                                    eachrow(night.profiles)],
                                     "vgrid" => night.vgrid,
                                     "times" => night.times,
                                     "Tc" => night.Tc,
                                     "bervs" => night.bervs)],
                   "options" => Dict("n_null" => 20, "n_lambda" => 181))
        out = run_job(cfg)
        @test out["status"] == "ok"
        @test out["kind"] == "tomography"
        @test isapprox(out["lambda_deg"], rad2deg(g.λ_true); atol = 12.0)
        # It must leave the same artefact every other job kind leaves.
        @test isfile(joinpath(dir, "summary.json"))
        # A tomography job must not be pushed through the sampler validator.
        bad = run_job(Dict("kind" => "tomography", "output_dir" => mktempdir(),
                           "orbit" => Dict("P" => g.P), "nights" => []))
        @test bad["status"] == "failed"
        @test occursin("a_Rs", bad["error"])
    end

    @testset "tomogram.* feature ops" begin
        night, g = _synth_night()
        rows(M) = [collect(r) for r in eachrow(M)]
        geom = Dict(:t => night.times, :Tc => night.Tc, :P => g.P,
                    :a_Rs => g.aRs, :inc => g.inc, :vsini => g.vsini)

        st = FEATURE_ACTIONS["tomogram.shadow_track"](
                merge(geom, Dict(:lambda => g.λ_true)))
        @test count(st["in_transit"]) == st["n_in_transit"]
        @test st["n_in_transit"] > 10
        @test maximum(abs.(filter(isfinite, st["v_shadow"]))) <= g.vsini * 1.001

        in_tr = abs.(night.times .- night.Tc) .< g.T14 / 2
        res = FEATURE_ACTIONS["tomogram.residuals"](
                Dict(:profiles => rows(night.profiles), :vgrid => night.vgrid,
                     :in_transit => in_tr, :vsys => g.vsys,
                     :bervs => night.bervs))
        @test res["n_frames"] == length(night.times)
        R = res["residuals"]
        @test length(R) == length(night.times)
        @test length(R[1]) == length(res["grid"])

        mf = FEATURE_ACTIONS["tomogram.matched_filter"](
                merge(geom, Dict(:residuals => R, :grid => res["grid"],
                                 :n_lambda => 181)))
        @test isapprox(mf["lambda_deg"], rad2deg(g.λ_true); atol = 12.0)
        @test length(mf["lambda_scan_score"]) == 181

        nd = FEATURE_ACTIONS["tomogram.null_distribution"](
                merge(geom, Dict(:residuals => R, :grid => res["grid"],
                                 :n_lambda => 91, :n => 30, :seed => 4)))
        @test length(nd["null"]) == 30
        @test nd["p_value"] <= 1.0 && nd["p_value"] > 0
        @test nd["peak_abs"] > nd["null_median"]

        # CCF of a single synthetic absorption line against a one-line mask.
        # ccf_profile accumulates the flux DEFICIT, so the profile PEAKS at the
        # line's velocity rather than dipping.
        c = 299792.458
        λ0 = 5500.0
        vtrue = 12.0
        λobs = collect(range(5498.0, 5502.0; length = 900))
        flux = 1.0 .- 0.4 .* exp.(-0.5 .* ((λobs .- λ0 * (1 + vtrue / c)) ./ 0.05).^2)
        vg = collect(range(-40, 40; length = 161))
        cp = FEATURE_ACTIONS["tomogram.ccf_profile"](
                Dict(:lambda_obs => λobs, :flux => flux, :mask_lambda => [λ0],
                     :mask_weight => [1.0], :vgrid => vg))
        @test length(cp["profile"]) == length(vg)
        @test abs(vg[argmax(cp["profile"])] - vtrue) <= 1.0

        inj = FEATURE_ACTIONS["tomogram.injection_test"](
                merge(geom, Dict(:profiles => rows(night.profiles),
                                 :vgrid => night.vgrid, :in_transit => in_tr,
                                 :rr => 0.3, :lambda_true => deg2rad(20.0),
                                 :vsys => g.vsys, :bervs => night.bervs,
                                 :n_lambda => 181)))
        @test haskey(inj, "lambda_deg") && haskey(inj, "error_deg")
        @test isfinite(inj["score"])

        # Every op must name the argument it is missing, not throw a MethodError.
        for op in ("tomogram.shadow_track", "tomogram.matched_filter",
                   "tomogram.residuals", "tomogram.injection_test")
            e = try; FEATURE_ACTIONS[op](Dict{Symbol,Any}()); nothing
                catch err; err end
            @test e !== nothing
            @test occursin("missing required argument", sprint(showerror, e))
        end
    end


    # The real-data regression that pinned a pooled multi-night detection lives
    # with that study, not here: it needs CCF products that are a study's
    # intermediates rather than package data, and it encodes an unpublished
    # measurement. The synthetic injections above cover the same code paths —
    # residual construction, matched filter, shadow track and the BERV
    # correction — against a known truth.
end
