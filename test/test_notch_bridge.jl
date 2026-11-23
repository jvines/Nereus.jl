# Notch detrender: the trend across excluded cadences (pass-1 dips, known
# transits) must be an interpolation of clean data on both sides, so a transit
# keeps its injected depth in `flux ./ trend`.
#
# Regression for TOI-4939 (TIC 149109125) TESS S103: the ephemeris-free notch
# (window 0.5 d) deepened a 6.5 ppt transit to 8.7 ppt. A false-positive
# pass-1 dip right after egress left a 20-min clean island between two
# excluded runs; that island's per-window poly2 was extrapolated across the
# second run, and the chord bridging the transit was anchored on it.

using Test, Random, Statistics
using Nereus

# 2-min cadence, a 1-d gap mid-baseline (two segments), rotation of amplitude
# `amp` with its first harmonic, trapezoid transits (10 % ingress).
function _bridge_synth(seed; σ = 1.5e-3, amp = 8e-3, prot = 3.1, depth = 6e-3,
                       P = 4.3, t0 = 1.7, T14 = 3.5 / 24, tmax = 27.0)
    rng = MersenneTwister(seed)
    t = collect(0.0:(2 / 1440):tmax)
    t = t[.!((t .> 13.2) .& (t .< 14.2))]
    ph = mod.(t .- t0 .+ P / 2, P) .- P / 2
    ing = 0.1 * T14
    shape = [abs(x) >= T14 / 2 ? 0.0 : abs(x) <= T14 / 2 - ing ? 1.0 :
             (T14 / 2 - abs(x)) / ing for x in ph]
    base = 1 .+ amp .* sin.(2π .* t ./ prot) .+
           0.3amp .* sin.(2π .* t ./ (prot / 2) .+ 1)
    f = base .* (1 .- depth .* shape) .+ σ .* randn(rng, length(t))
    return (; t, f, e = fill(σ, length(t)), ph, T14, depth, P, t0, base)
end

# Per-transit depth: core (|φ| < 0.4 T14) against the adjacent baseline
# (0.6-1.5 T14), one value per transit with enough coverage.
function _bridge_depths(s, fd)
    ep = round.(Int, (s.t .- s.t0) ./ s.P)
    out = Float64[]
    for k in unique(ep)
        inn = (ep .== k) .& (abs.(s.ph) .< 0.4s.T14)
        oot = (ep .== k) .& (abs.(s.ph) .> 0.6s.T14) .& (abs.(s.ph) .< 1.5s.T14)
        (count(inn) > 20 && count(oot) > 20) || continue
        push!(out, 1 - mean(fd[inn]) / mean(fd[oot]))
    end
    return out
end

@testset "notch: excluded runs bridged by interpolation" begin
    # Exact poly2 baseline, no noise: excluded cadences (a transit run, a
    # 20-min clean island, a second run) and the island itself start with a
    # garbage trend; the bridge must restore the baseline exactly.
    t = collect(0.0:(2 / 1440):2.0)
    base = @. 1.0 + 0.004 * (t - 1.0) - 0.006 * (t - 1.0)^2
    flux = copy(base)
    err = fill(1e-3, length(t))
    excl = falses(length(t))
    excl[(t .>= 0.80) .& (t .<= 0.95)] .= true          # "transit"
    excl[(t .>= 0.964) .& (t .<= 1.25)] .= true         # false-positive run
    island = .!excl .& (t .> 0.95) .& (t .< 0.964)
    @test count(island) > 3
    trend = copy(base)
    trend[excl .| island] .= 2.0                         # extrapolation garbage
    fd = flux ./ trend
    segs = Nereus.find_segments(t)
    Nereus._bridge_trend!(trend, fd, t, flux, err, excl, segs, 0.25, 5)
    @test maximum(abs.(trend .- base)) < 1e-9
    @test fd ≈ flux ./ trend

    # Excluded run at a segment start: one-sided, the straight line of the
    # clean flank continued across it, never a poly2 blow-up.
    excl2 = t .< 0.15
    trend2 = copy(base); trend2[excl2] .= -5.0
    fd2 = flux ./ trend2
    Nereus._bridge_trend!(trend2, fd2, t, flux, err, excl2, segs, 0.25, 5)
    @test all(isfinite, trend2) && all(trend2 .> 0)
    @test maximum(abs.(diff(diff(trend2[excl2])))) < 1e-12   # a straight line
    @test maximum(abs.(trend2 .- base)) < 2e-3           # baseline curvature only

    # Nothing excluded on a side ⇒ an ordinary (edge) window, left untouched.
    trend3 = copy(base) .+ 1e-6
    fd3 = flux ./ trend3
    Nereus._bridge_trend!(trend3, fd3, t, flux, err, falses(length(t)), segs, 0.25, 5)
    @test trend3 == base .+ 1e-6
end


@testset "notch: transit depth preserved, ephemeris-free and masked" begin
    # Per-transit depth errors over 5 noise realisations (~30 transits per
    # path). White-noise floor 0.20 ppt; before the fix the rms was ~0.54 ppt
    # without an ephemeris and ~0.40 ppt with a transit_mask, with single
    # transits off by 1.5-2.7 ppt.
    floor = 1.5e-3 * sqrt(1 / 84 + 1 / 189)
    err = Dict(k => Float64[] for k in (:free, :mask, :mask_ns))
    trend_dev = 0.0
    for seed in 1:5
        s = _bridge_synth(seed)
        # Ephemeris-free (pass-1 dips + refine), window 0.5 d.
        r = detrend_notch(s.t, s.f, s.e; window = 0.5)
        append!(err[:free], _bridge_depths(s, r.flux_detrended) .- s.depth)
        # Ephemeris supplied: transit_mask from P, T0 and T14 (±0.75 T14),
        # window 3 T14, with and without the notch search.
        m = abs.(s.ph) .< 0.75s.T14
        for (k, ns) in ((:mask, true), (:mask_ns, false))
            rm = detrend_notch(s.t, s.f, s.e; window = 3s.T14, transit_mask = m,
                               notch_search = ns)
            append!(err[k], _bridge_depths(s, rm.flux_detrended) .- s.depth)
            trend_dev = max(trend_dev, maximum(abs.(rm.trend[m] .- s.base[m])))
        end
    end
    rms(x) = sqrt(mean(x .^ 2))
    for k in (:free, :mask, :mask_ns)
        @test length(err[k]) >= 25
        @test abs(mean(err[k])) < 0.15e-3                  # no bias
        @test maximum(abs.(err[k])) < 5floor              # no single bad transit
    end
    @test rms(err[:free]) < 1.8floor
    @test rms(err[:mask]) < 1.8floor
    @test rms(err[:mask_ns]) < 1.8floor
    @test trend_dev < 1.5e-3            # masked trend follows the true baseline
end

@testset "notch: transit next to a second dip (TOI-4939 S103)" begin
    # A 6 ppt transit followed, after a short clean stretch, by a shallow
    # multi-hour dip that the search also flags: the configuration that
    # deepened TOI-4939's transit. Before the fix the trend in transit was off
    # by 2-4 ppt here.
    for (gaph, durh) in ((0.3, 5.0), (0.5, 6.0)), seed in 1:4
        rng = MersenneTwister(seed)
        t = collect(0.0:(2 / 1440):10.0)
        T14 = 3.5 / 24
        tc = 5.0 + 0.1seed
        base = @. 1 + 0.008 * sin(2π * t / 3.1)
        x = abs.(t .- tc)
        shape = [a >= T14 / 2 ? 0.0 : a <= 0.4T14 ? 1.0 : (T14 / 2 - a) / (0.1T14) for a in x]
        dip = (t .> tc + T14 / 2 + gaph / 24) .& (t .< tc + T14 / 2 + (gaph + durh) / 24)
        f = base .* (1 .- 6e-3 .* shape .- 2e-3 .* dip) .+ 1e-3 .* randn(rng, length(t))
        r = detrend_notch(t, f, fill(1e-3, length(t)); window = 0.5)
        c = x .< 0.4T14
        @test abs(mean(r.trend[c] ./ base[c]) - 1) < 1.3e-3
    end
end

@testset "savgol: masked runs anchored on averaged flanks" begin
    # The masked-run interpolation used to anchor on the single flux point
    # bracketing each run, so that point's noise went straight into the
    # trend across the transit (per-transit depth rms 0.68 ppt here, against
    # 0.30 ppt with averaged anchors; white-noise floor 0.20 ppt).
    errs = Float64[]
    for seed in 1:10
        s = _bridge_synth(seed; tmax = 13.0)
        m = abs.(s.ph) .< 0.75s.T14
        r = detrend_savgol(s.t, s.f, s.e; window_length = 301, transit_mask = m)
        append!(errs, _bridge_depths(s, r.flux_detrended) .- s.depth)
    end
    @test length(errs) >= 20
    @test sqrt(mean(errs .^ 2)) < 0.45e-3

    # Anchors: a constant series with one masked run is reproduced exactly;
    # a linear series too (anchor = mean value at the mean index).
    y = collect(1.0:20.0); m = falses(20); m[8:11] .= true
    @test Nereus._interp_masked(y, m) ≈ y
    @test Nereus._interp_masked(fill(2.0, 20), m) == fill(2.0, 20)
    m2 = falses(20); m2[1:3] .= true                    # edge run: flat carry
    @test all(Nereus._interp_masked(y, m2)[1:3] .== mean(y[4:6]))
end
