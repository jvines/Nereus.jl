using Test
using Statistics
using LinearAlgebra
using Random
using Nereus   # find_segments, detrend_notch, detrend_locor (internals via Nereus._locor_*)

@testset "_locor_cycles" begin
    t = collect(0.0:0.1:2.9)
    P_rot = 1.0
    t0 = 0.0

    res = Nereus._locor_cycles(t, P_rot, t0)
    phase = res.phase
    cycle_id = res.cycle_id

    # lengths match input
    @test length(phase) == length(t)
    @test length(cycle_id) == length(t)

    # all phases in [0, 1)
    @test all(0.0 .<= phase .< 1.0)

    # cycle_id: 0 for t in [0,1), 1 for [1,2), 2 for [2,3)
    for (ti, ci) in zip(t, cycle_id)
        if 0.0 <= ti < 1.0
            @test ci == 0
        elseif 1.0 <= ti < 2.0
            @test ci == 1
        elseif 2.0 <= ti < 3.0
            @test ci == 2
        end
    end
end

@testset "_rcomb_cycle" begin
    # Five clean cycles of the same sinusoid → cycle 2 is perfectly
    # predictable as a linear combination of the others.
    n_pts = 30
    phases = collect(range(0.0, 1.0; length = n_pts + 1))[1:end-1]  # 0:1/30:1-eps

    make_cycle(ph) = 1.0 .+ 0.02 .* sin.(2π .* ph)
    median_normalize(f) = f ./ median(f)

    cycles_phase = [copy(phases) for _ in 0:4]
    cycles_flux = [median_normalize(make_cycle(phases)) for _ in 0:4]

    target_phase = cycles_phase[3]   # cycle k=2 (1-based index 3)
    target_flux = cycles_flux[3]

    donor_phases = [cycles_phase[i] for i in (1, 2, 4, 5)]
    donor_fluxes = [cycles_flux[i] for i in (1, 2, 4, 5)]

    baseline = Nereus._rcomb_cycle(target_phase, target_flux, donor_phases, donor_fluxes)

    @test length(baseline) == n_pts
    rms = sqrt(mean((baseline .- target_flux) .^ 2))
    @test rms < 1e-3
end

@testset "_rcomb_cycle_clipped" begin
    # Five clean cycles of the same sinusoid (deterministic phase grid),
    # cycle k=2 (1-based index 3) as target.
    n_pts = 30
    phases = collect(range(0.0, 1.0; length = n_pts + 1))[1:end-1]

    make_cycle(ph) = 1.0 .+ 0.02 .* sin.(2π .* ph)
    median_normalize(f) = f ./ median(f)

    # --- clean reference: clean target, clean donors, plain _rcomb_cycle ---
    clean_flux = [median_normalize(make_cycle(phases)) for _ in 0:4]
    clean_target_phase = copy(phases)
    clean_target_flux = clean_flux[3]
    clean_donor_phases = [copy(phases) for i in (1, 2, 4, 5)]
    clean_donor_fluxes = [clean_flux[i] for i in (1, 2, 4, 5)]
    clean_baseline = Nereus._rcomb_cycle(clean_target_phase, clean_target_flux,
                                  clean_donor_phases, clean_donor_fluxes)

    # --- contaminated case ---
    # Inject a box dip into ONE donor (cycle 1) over phase span [0.10, 0.20).
    # Inject a transit dip into the TARGET over a DIFFERENT span [0.60, 0.70).
    raw_flux = [make_cycle(phases) for _ in 0:4]

    donor1_mask = (phases .>= 0.10) .& (phases .< 0.20)
    raw_flux[1][donor1_mask] .-= 0.05

    target_transit_mask = (phases .>= 0.60) .& (phases .< 0.70)
    raw_flux[3][target_transit_mask] .-= 0.05

    cyc_flux = [median_normalize(f) for f in raw_flux]

    target_phase = copy(phases)
    target_flux = cyc_flux[3]
    donor_phases = [copy(phases) for i in (1, 2, 4, 5)]
    donor_fluxes = [cyc_flux[i] for i in (1, 2, 4, 5)]

    res = Nereus._rcomb_cycle_clipped(target_phase, target_flux, donor_phases, donor_fluxes;
                               niter = 3, clip_sigma = 3.5)
    baseline = res.baseline
    outlier = res.outlier

    @test length(baseline) == n_pts
    @test length(outlier) == n_pts

    # (a) the target's injected-transit points are flagged as outliers
    @test all(outlier[target_transit_mask])

    # (b) over NON-transit target points the recovered baseline matches the
    #     clean baseline to small RMS — the single contaminated donor did not
    #     distort the result.
    keep = .!target_transit_mask
    rms = sqrt(mean((baseline[keep] .- clean_baseline[keep]) .^ 2))
    @test rms < 5e-3
end

@testset "_locor_segment! normal" begin
    # Single contiguous fast-rotator segment, P_rot = 0.8 d, ~12 d baseline
    # (≥10 cycles), 10-min cadence, pure sinusoid + tiny noise.
    P_rot = 0.8
    t0 = 0.0
    cadence = 10 / (60 * 24)        # 10 min in days
    t = collect(0.0:cadence:12.0)
    noise = 1e-4
    rng_seed = 1234
    # deterministic tiny noise
    n = length(t)
    pert = noise .* sin.(100.0 .* t .+ 0.5)   # small, structured, deterministic
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot) .+ pert
    flux_err = fill(noise, n)

    flux_detrended = similar(flux)
    baseline = similar(flux)
    outlier = falses(n)
    cycle_id_out = fill(-1, n)
    used_notch = falses(n)

    Nereus._locor_segment!(flux_detrended, baseline, outlier, cycle_id_out, used_notch,
                    t, flux, flux_err, 1:n, P_rot, t0, nothing, 3, 3.5, 0.5)

    @test all(.!used_notch)
    @test all(isfinite, flux_detrended)
    # detrended flux should be flat ≈ 1
    @test abs(median(flux_detrended) - 1.0) < 1e-3
    @test std(flux_detrended) < 50 * noise
end

@testset "_locor_segment! fallback" begin
    # Only ~2 cycles → length(ucyc) < 3 → notch fallback.
    P_rot = 0.8
    t0 = 0.0
    cadence = 10 / (60 * 24)
    t = collect(0.0:cadence:1.6)
    n = length(t)
    noise = 1e-4
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot)
    flux_err = fill(noise, n)

    flux_detrended = similar(flux)
    baseline = similar(flux)
    outlier = falses(n)
    cycle_id_out = fill(-1, n)
    used_notch = falses(n)

    Nereus._locor_segment!(flux_detrended, baseline, outlier, cycle_id_out, used_notch,
                    t, flux, flux_err, 1:n, P_rot, t0, nothing, 3, 3.5, 0.5)

    @test all(used_notch)
    @test all(isfinite, flux_detrended)
end

@testset "detrend_locor headline" begin
    # Synthetic fast rotator: P_rot = 0.8 d, 10-min cadence, ~12 d baseline,
    # with a box transit injected mid-baseline. LOCoR should flatten the
    # rotation while preserving the transit dip.
    P_rot_true = 0.8
    cad = 10 / 60 / 24            # 10 min in days
    t = collect(0.0:cad:12.0)
    n = length(t)

    rng = MersenneTwister(20260523)
    noise = 3e-4 .* randn(rng, n)
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot_true) .+ noise

    # Box transit: depth 0.01 over a ~2.5 hr window mid-baseline.
    t_mid = 6.0
    half_dur = (2.5 / 24) / 2
    in_transit = (t .>= t_mid - half_dur) .& (t .<= t_mid + half_dur)
    flux[in_transit] .*= (1 - 0.01)

    flux_err = fill(3e-4, n)

    res = Nereus.detrend_locor(t, flux, flux_err; P_rot = 0.8)

    # output field is `trend` (unified across detrenders), not `baseline`
    @test hasproperty(res, :trend)
    @test !hasproperty(res, :baseline)

    # (d) output arrays all length n
    @test length(res.flux_detrended) == n
    @test length(res.trend) == n
    @test length(res.phase) == n
    @test length(res.cycle_id) == n
    @test length(res.outlier) == n
    @test length(res.used_notch) == n

    # (c) P_rot_used echoes the explicit P_rot
    @test res.P_rot_used == 0.8

    @test all(isfinite, res.flux_detrended)

    # (a) out-of-transit detrended flux is flat ≈ 1
    oot = .!in_transit .& .!res.outlier
    @test abs(median(res.flux_detrended[oot]) - 1.0) < 1e-2
    @test std(res.flux_detrended[oot]) < 5e-3

    # (b) the injected transit survives (at least half the 1% depth)
    @test minimum(res.flux_detrended[in_transit]) < 1 - 0.005
end

@testset "detrend_locor auto P_rot" begin
    # Light check that the auto-P_rot path runs end-to-end and returns sane
    # output. We do NOT assert an exact value, nor even finiteness: the ACF
    # (`find_rotation_period`) is tuned for slow rotators and returns NaN on
    # this short-baseline fast rotator, which `detrend_locor` then handles via
    # the whole-LC notch fallback. The contract here is: it runs, the period
    # echoed is whatever the ACF returned, and all output arrays stay length n.
    P_rot_true = 0.8
    cad = 10 / 60 / 24
    t = collect(0.0:cad:12.0)
    n = length(t)
    rng = MersenneTwister(7)
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot_true) .+ 3e-4 .* randn(rng, n)
    flux_err = fill(3e-4, n)

    res = Nereus.detrend_locor(t, flux, flux_err)
    # auto P_rot is whatever the ACF produced (finite when it locks on, NaN
    # when it cannot — both are valid; the latter triggers the notch fallback).
    @test res.P_rot_used === find_rotation_period(t, flux, flux_err).P_rot
    @test all(isfinite, res.flux_detrended)
    @test length(res.flux_detrended) == n
    @test length(res.phase) == n
    @test length(res.cycle_id) == n
end

@testset "sector_id passthrough" begin
    # Build a 2-sector synthetic: two time chunks separated by a >0.5 d gap.
    # Sector 1: ~12 d baseline starting at 0; sector 2: a separate chunk
    # offset by a big gap so find_segments (gap-based) splits them, while
    # sector_id is the *user's* per-cadence tag — independent metadata.
    P_rot_true = 0.8
    cad = 10 / 60 / 24
    t1 = collect(0.0:cad:12.0)
    t2 = collect(20.0:cad:32.0)   # >0.5 d gap before t2 (sector boundary)
    t = vcat(t1, t2)
    n1 = length(t1); n2 = length(t2); n = length(t)
    sid = [fill(1, n1); fill(2, n2)]

    rng = MersenneTwister(424242)
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot_true) .+ 3e-4 .* randn(rng, n)
    flux_err = fill(3e-4, n)

    # transit_mask for savgol (needs nothing special; pass `nothing` default).
    mask = falses(n)  # for savgol passthrough variant

    # --- detrend_locor ---
    res_l = Nereus.detrend_locor(t, flux, flux_err; P_rot = 0.8, sector_id = sid)
    @test res_l.sector_id == sid
    @test res_l.sector_id isa Vector{Int}
    @test length(res_l.sector_id) == n
    # Slicing the detrended result by sector 2 recovers exactly the sector-2
    # cadences in order.
    @test res_l.flux_detrended[res_l.sector_id .== 2] == res_l.flux_detrended[(n1+1):n]
    # default omitted → nothing
    res_l_def = Nereus.detrend_locor(t, flux, flux_err; P_rot = 0.8)
    @test res_l_def.sector_id === nothing
    # length mismatch → ArgumentError
    @test_throws ArgumentError Nereus.detrend_locor(t, flux, flux_err;
        P_rot = 0.8, sector_id = sid[1:end-1])

    # --- detrend_notch ---
    res_n = Nereus.detrend_notch(t, flux, flux_err; sector_id = sid)
    @test res_n.sector_id == sid
    @test res_n.sector_id isa Vector{Int}
    @test length(res_n.sector_id) == n
    @test res_n.flux_detrended[res_n.sector_id .== 2] == res_n.flux_detrended[(n1+1):n]
    res_n_def = Nereus.detrend_notch(t, flux, flux_err)
    @test res_n_def.sector_id === nothing
    @test_throws ArgumentError Nereus.detrend_notch(t, flux, flux_err;
        sector_id = sid[1:end-1])

    # --- detrend_savgol ---
    res_s = Nereus.detrend_savgol(t, flux, flux_err; window_length = 101,
        sector_id = sid)
    @test res_s.sector_id == sid
    @test res_s.sector_id isa Vector{Int}
    @test length(res_s.sector_id) == n
    @test res_s.flux_detrended[res_s.sector_id .== 2] == res_s.flux_detrended[(n1+1):n]
    res_s_def = Nereus.detrend_savgol(t, flux, flux_err; window_length = 101)
    @test res_s_def.sector_id === nothing
    @test_throws ArgumentError Nereus.detrend_savgol(t, flux, flux_err;
        window_length = 101, sector_id = sid[1:end-1])
end

@testset "detrend_locor transit_mask commensurate" begin
    # Inject-recovery proof for the cleanmask defense against LOCoR destroying
    # transits whose orbital period is commensurate with the rotation period.
    #
    # P_orb = 2 * P_rot = 1.6 d → the transit recurs every 2 rotation cycles
    # at the SAME within-rotation phase. Donors therefore contain the dip at
    # exactly the target's transit phase; mask-free LOCoR learns it into the
    # baseline and absorbs it. The transit_mask excludes in-transit points from
    # both the donor curves and the target LSQ fit, so the baseline becomes the
    # clean rotational extrapolation and the dip is preserved.
    P_rot = 0.8
    cad = 10 / 60 / 24                  # 10 min in days
    t = collect(0.0:cad:16.0)           # ~16 d baseline, 20 rotation cycles
    n = length(t)

    rng = MersenneTwister(20260524)
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot) .+ 1e-4 .* randn(rng, n)

    # Commensurate transit: P_orb = 1.6 d, depth 1%, duration ~2 hr.
    P_orb = 2 * P_rot                    # 1.6 d
    depth = 0.01
    half_dur = (2.0 / 24) / 2            # ~2 hr full duration
    t0_transit = 1.0                     # first transit center
    tmask = falses(n)
    k = 0
    while t0_transit + k * P_orb <= t[end]
        tc = t0_transit + k * P_orb
        tmask .|= (t .>= tc - half_dur) .& (t .<= tc + half_dur)
        k += 1
    end
    @test any(tmask)
    flux[tmask] .*= (1 - depth)

    flux_err = fill(1e-4, n)

    # --- Run A: mask-free → transit should be ATTENUATED (absorbed). ---
    res_free = Nereus.detrend_locor(t, flux, flux_err; P_rot = 0.8)
    recovered_free = 1 - median(res_free.flux_detrended[tmask])
    # LOCoR ate most of it: recovered depth well under half the injected depth.
    @test recovered_free < 0.5 * depth

    # --- Run B: with transit_mask → transit should be RECOVERED. ---
    res_mask = Nereus.detrend_locor(t, flux, flux_err; P_rot = 0.8, transit_mask = tmask)
    recovered_mask = 1 - median(res_mask.flux_detrended[tmask])
    @test recovered_mask >= 0.7 * depth

    # length-mismatch transit_mask → ArgumentError
    @test_throws ArgumentError Nereus.detrend_locor(t, flux, flux_err;
        P_rot = 0.8, transit_mask = tmask[1:end-1])
end

@testset "detrend_rotation routing" begin
    # Fast-rotator synthetic with an injected box transit (same recipe as the
    # headline test) — used to exercise the auto-selector's routing logic.
    P_rot_true = 0.8
    cad = 10 / 60 / 24
    t = collect(0.0:cad:12.0)
    n = length(t)
    rng = MersenneTwister(20260524)
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot_true) .+ 3e-4 .* randn(rng, n)
    t_mid = 6.0
    half_dur = (2.5 / 24) / 2
    in_transit = (t .>= t_mid - half_dur) .& (t .<= t_mid + half_dur)
    flux[in_transit] .*= (1 - 0.01)
    flux_err = fill(3e-4, n)

    # :locor with explicit P_rot → LOCoR ran (LOCoR-only fields present).
    r_locor = Nereus.detrend_rotation(t, flux, flux_err; method = :locor, P_rot = 0.8)
    @test hasproperty(r_locor, :cycle_id)
    @test hasproperty(r_locor, :P_rot_used)

    # :notch → notch ran (notch-only field present).
    r_notch = Nereus.detrend_rotation(t, flux, flux_err; method = :notch)
    @test hasproperty(r_notch, :delta_bic)

    # :auto with P_rot below threshold (passed explicitly to bypass the ACF) → LOCoR.
    r_auto_fast = Nereus.detrend_rotation(t, flux, flux_err; method = :auto, P_rot = 0.8)
    @test hasproperty(r_auto_fast, :cycle_id)

    # :auto with P_rot above threshold → notch.
    r_auto_slow = Nereus.detrend_rotation(t, flux, flux_err; method = :auto, P_rot = 10.0)
    @test hasproperty(r_auto_slow, :delta_bic)

    # invalid method → ArgumentError.
    @test_throws ArgumentError Nereus.detrend_rotation(t, flux, flux_err; method = :banana)
end

@testset "find_transits :locor" begin
    # Fast-rotator synthetic with an injected box transit. Exercises the
    # :locor detrend branch wired into find_transits, on a deliberately small
    # period grid so it stays fast.
    P_rot_true = 0.8
    cad = 10 / 60 / 24
    t = collect(0.0:cad:12.0)
    n = length(t)
    rng = MersenneTwister(20260524)
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot_true) .+ 3e-4 .* randn(rng, n)
    t_mid = 6.0
    half_dur = (2.5 / 24) / 2
    in_transit = (t .>= t_mid - half_dur) .& (t .<= t_mid + half_dur)
    flux[in_transit] .*= (1 - 0.01)
    flux_err = fill(3e-4, n)

    res = Nereus.find_transits(t, flux, flux_err;
        detrend = :locor, detrend_kwargs = (P_rot = 0.8,),
        n_periods = 200, period_min = 0.5, period_max = 5.0)

    # The :locor branch ran and returned flattened flux of the right shape.
    @test all(isfinite, res.flux_detrended)
    @test length(res.flux_detrended) == n

    # An unknown detrend symbol still throws, naming the valid set.
    err = try
        Nereus.find_transits(t, flux, flux_err; detrend = :bogus,
            n_periods = 200, period_min = 0.5, period_max = 5.0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(":locor", err.msg)
end

@testset "find_transits output: epoch t0, duration, snr" begin
    # Clean single-planet synthetic (no detrend needed) so BLS reliably
    # recovers the planet, letting us check the OUTPUT contract:
    #   * t0 is a real mid-transit BJD inside the baseline (NOT a phase)
    #   * durations is reported and physical (0 < dur < period)
    #   * snr field present (renamed from `scores`)
    P_true = 4.137; dur_d = 0.10; depth = 0.01; t0_inj = 1.5
    cad = 10 / 60 / 24
    t = collect(0.0:cad:27.0); n = length(t)
    rng = MersenneTwister(2026)
    flux = ones(n) .+ 5e-4 .* randn(rng, n)
    for k in 0:6
        tc = t0_inj + k * P_true
        @. flux[abs(t - tc) < dur_d/2] -= depth
    end
    ferr = fill(5e-4, n)

    res = Nereus.find_transits(t, flux, ferr; detrend = :none, method = :bls,
        period_min = 0.5, period_max = 30.0, n_periods = 10_000)

    @test !isempty(res.periods)
    @test abs(res.periods[1] - P_true) / P_true < 0.01      # fundamental, not a harmonic
    # t0 is a REAL epoch in the baseline, not a phase in [0,P)
    @test t[1] <= res.t0s[1] <= t[end]
    # folding the reported epoch lands on an injected transit phase
    ph = mod(res.t0s[1] - t0_inj, P_true) / P_true
    @test min(ph, 1 - ph) < 0.02
    # duration present + physical, roughly the injected 0.1 d
    @test haskey(res, :durations)
    @test 0 < res.durations[1] < res.periods[1]
    @test abs(res.durations[1] - dur_d) < 0.05
    # snr field present (formerly `scores`) and positive
    @test haskey(res, :snr)
    @test res.snr[1] > 0
    @test !haskey(res, :scores)
end
