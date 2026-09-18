# Non-finite-input guards on the Gaia DR4 epoch-astrometry path.
#
# `_gep_build_source` used to filter only on the along-scan abscissa and its
# error. A transit whose `obs_time_bary_corr` (or `parallax_factor_al`, or a
# CCD's `scan_pos_angle`) was NULL therefore pushed NaN straight into
# IADData.t and IADData.pm_factor, where it surfaced much later as a
# non-finite log-likelihood — indistinguishable from a bad model.
#
# The DR4_RC3 pre-release never triggers this (its NaN-barycentric rows also
# have NaN abscissae), so these tests build the pathological columns directly.

@testset "Gaia DR4 epoch: non-finite guards" begin
    NCCD = 9
    t0   = Int64(157_800_000_000_000_000)      # ~2015.0, in ns since 2010.0 TCB
    dt   = Int64(4_500_000_000)                # ~4.5 s between CCDs

    # one transit block: (bary_corr, parallax_factor_al, scan angles)
    function block!(cols, tid, bc, pAL; bad_angle::Int = 0)
        push!(cols[:transit_id], Int64(tid))
        push!(cols[:obs_time_tcb], Int64[t0 + tid * Int64(10_000_000_000_000) + (j-1)*dt
                                         for j in 1:NCCD])
        push!(cols[:obs_time_bary_corr], Float32(bc))
        push!(cols[:parallax_factor_al], Float32(pAL))
        push!(cols[:scan_pos_angle],
              Float64[j == bad_angle ? NaN : 61.0 + 0.001j for j in 1:NCCD])
        push!(cols[:centroid_pos_al], Float64[0.1 * j for j in 1:NCCD])
        push!(cols[:centroid_pos_error_al], Float32[0.2 for _ in 1:NCCD])
        push!(cols[:used_by_agis_al], UInt8[UInt8('T') for _ in 1:NCCD])
        return cols
    end

    cols = Dict{Symbol, Vector{Any}}(
        k => Any[] for k in (:transit_id, :obs_time_tcb, :obs_time_bary_corr,
                             :parallax_factor_al, :scan_pos_angle,
                             :centroid_pos_al, :centroid_pos_error_al,
                             :used_by_agis_al))

    block!(cols, 1, 1.2e5,  0.31)              # clean                     -> 9 kept
    block!(cols, 2, NaN,    0.32)              # NULL barycentric corr     -> 0 kept
    block!(cols, 3, 1.4e5,  0.33; bad_angle=4) # one NULL scan angle       -> 8 kept
    block!(cols, 4, 1.5e5,  NaN)               # NULL parallax factor      -> 0 kept

    iad = @test_logs (:warn, r"non-finite epoch"i) match_mode = :any begin
        Nereus._gep_build_source(cols, collect(1:4))
    end

    # the pathological blocks are dropped, not silently NaN-ed through
    @test length(iad.t) == 2 * NCCD - 1
    for v in (iad.t, iad.abscissa, iad.abscissa_err, iad.psi,
              iad.parallax_factor, iad.pm_factor)
        @test all(isfinite, v)
    end

    # kept epochs are EXACTLY those of blocks 1 and 3; blocks 2 and 4 contribute
    # nothing (before the fix they contributed NaN, one per CCD)
    epoch(tid, j, bc) = Nereus._GEP_MJD_ORIGIN +
        (Float64(t0 + tid * Int64(10_000_000_000_000) + (j-1)*dt) + Float64(Float32(bc))) *
        Nereus._GEP_NS_TO_DAY
    want = vcat([epoch(1, j, 1.2e5) for j in 1:NCCD],
                [epoch(3, j, 1.4e5) for j in 1:NCCD if j != 4])
    @test sort(iad.t) ≈ sort(want)
    gone = vcat([epoch(2, j, 1.2e5) for j in 1:NCCD],
                [epoch(4, j, 1.5e5) for j in 1:NCCD])
    @test !any(t -> any(g -> isapprox(t, g; atol = 1e-9), gone), iad.t)
    @test count(≈(deg2rad(61.004); atol = 1e-12), iad.psi) == 1   # block 3's CCD 4 gone
    @test sort(unique(round.(iad.parallax_factor; digits = 6))) ==
          round.(Float64[Float32(0.31), Float32(0.33)]; digits = 6)

    # epoch and pm_factor stay consistent: both derive from the same corrected time
    @test all(isapprox.((iad.t .- Nereus._GEP_MJD_ORIGIN) ./ 365.25 .+
                        (Nereus._GEP_TCB_ORIGIN_JYEAR - Nereus._GEP_DR4_REF_JYEAR),
                        iad.pm_factor; atol = 1e-9))
end

@testset "IADData rejects non-finite input" begin
    n  = 5
    ok = (t = collect(56_000.0:56_004.0), abscissa = fill(0.5, n),
          abscissa_err = fill(0.2, n), psi = fill(1.0, n),
          parallax_factor = fill(0.3, n), pm_factor = fill(-2.5, n))

    @test Nereus.IADData(; ok...) isa Nereus.IADData        # baseline passes

    for fld in (:t, :abscissa, :abscissa_err, :psi, :parallax_factor, :pm_factor)
        bad = merge(ok, (; fld => (v = copy(getfield(ok, fld)); v[3] = NaN; v)))
        @test_throws ArgumentError Nereus.IADData(; bad...)
        inf = merge(ok, (; fld => (v = copy(getfield(ok, fld)); v[1] = Inf; v)))
        @test_throws ArgumentError Nereus.IADData(; inf...)
    end
end
