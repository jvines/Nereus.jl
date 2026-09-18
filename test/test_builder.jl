# Tests for the build_target convenience constructor.

@testset "build_target" begin

    # Tiny synthetic RV data, used across most test cases.
    _rvdat() = (t      = [55000.0, 55100.0, 55200.0, 55300.0],
                rv     = [10.0, -5.0, 8.0, -3.0],
                rv_err = [1.0, 1.0, 1.0, 1.0])

    @testset "RV-only K-driven, 1 planet, 1 instrument" begin
        target = build_target(
            M_pri = 1.0,
            planets = (b = (
                P      = LogUniformPrior(10.0, 1000.0),
                K      = LogUniformPrior(1.0, 100.0),
                sesinw = UniformPrior(-1.0, 1.0),
                secosw = UniformPrior(-1.0, 1.0),
                Mo     = UniformPrior(0.0, 2π),
            ),),
            rv = (HIRES = (data = _rvdat(), sigma = LogUniformPrior(0.1, 10.0)),),
        )
        @test target isa NereusTarget
        @test target.params.config.parametrization.mass === :K_driven
        @test target.params.config.M_s == 1.0
        @test "P_k1" in target.params.layout.unfrozen_names
        @test "K_k1" in target.params.layout.unfrozen_names
        @test "sigma_HIRES" in target.params.layout.unfrozen_names
        @test "gamma_HIRES" in target.params.layout.unfrozen_names
        @test target.params.config.instruments.rv_names == ["HIRES"]
        # M_pri-as-Real → FixedPrior; not in unfrozen for RV_ONLY
        @test !("M_pri" in target.params.layout.unfrozen_names)
    end

    @testset "RVAS a-driven, with relAST + HGCA + Gaussian M_pri" begin
        relast = RelAstromData(
            t = [55000.0, 55365.0],
            ra_off = [10.0, 12.0], dec_off = [-5.0, -3.0],
            ra_err = [1.0, 1.0], dec_err = [1.0, 1.0],
        )
        hgca = HGCAData(
            epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra   = (5.0, 4.95, 4.9),
            pmdec  = (-3.0, -3.05, -3.1),
            sigma_pmra  = (0.2, 0.2, 0.2),
            sigma_pmdec = (0.2, 0.2, 0.2),
            plx = 25.0, plx_err = 0.05, hip_id = 1,
        )
        target = build_target(
            M_pri = NormalPrior(0.856, 0.014),       # Gaussian → sampled
            planets = (b = (
                a      = LogUniformPrior(1.0, 100.0),
                M_sec  = LogUniformPrior(0.001, 0.5),
                sesinw = UniformPrior(-1.0, 1.0),
                secosw = UniformPrior(-1.0, 1.0),
                Mo     = UniformPrior(0.0, 2π),
                inc    = SinePrior(),
                Omega  = UniformPrior(0.0, 2π),
            ),),
            rv = (HIRES = (data = _rvdat(), sigma = LogUniformPrior(0.1, 30.0)),),
            relAST = relast,
            hgca   = hgca,
        )
        @test target.params.config.parametrization.mass === :a_driven
        @test "a_k1" in target.params.layout.unfrozen_names
        @test "M_sec_k1" in target.params.layout.unfrozen_names
        @test "M_pri" in target.params.layout.unfrozen_names    # Gaussian → sampled
        @test "plx" in target.params.layout.unfrozen_names      # default-priored from HGCA
        @test target.data.relastrom === relast
        @test target.data.hgca === hgca
    end

    @testset "Two planets, two instruments" begin
        target = build_target(
            M_pri = 1.0,
            planets = (
                b = (P=LogUniformPrior(10.0, 1000.0),
                     K=LogUniformPrior(1.0, 100.0),
                     sesinw=UniformPrior(-1.0, 1.0),
                     secosw=UniformPrior(-1.0, 1.0),
                     Mo=UniformPrior(0.0, 2π)),
                c = (P=LogUniformPrior(1000.0, 10000.0),
                     K=LogUniformPrior(1.0, 100.0),
                     sesinw=UniformPrior(-1.0, 1.0),
                     secosw=UniformPrior(-1.0, 1.0),
                     Mo=UniformPrior(0.0, 2π)),
            ),
            rv = (HIRES = (data = _rvdat(), sigma = LogUniformPrior(0.1, 10.0)),
                  HARPS = (data = _rvdat(), sigma = LogUniformPrior(0.1, 10.0))),
        )
        @test "P_k1" in target.params.layout.unfrozen_names
        @test "P_k2" in target.params.layout.unfrozen_names
        @test "sigma_HIRES" in target.params.layout.unfrozen_names
        @test "sigma_HARPS" in target.params.layout.unfrozen_names
        @test "gamma_HIRES" in target.params.layout.unfrozen_names
        @test "gamma_HARPS" in target.params.layout.unfrozen_names
        @test target.params.config.instruments.rv_names == ["HIRES", "HARPS"]
        @test length(target.data.t_rv) == 8     # 4 epochs × 2 instruments
    end

    @testset "Real numbers become FixedPrior automatically" begin
        target = build_target(
            M_pri = 0.856,                       # Real
            planets = (b = (
                P      = LogUniformPrior(10.0, 1000.0),
                K      = LogUniformPrior(1.0, 100.0),
                sesinw = 0.0,                    # Real → FixedPrior
                secosw = 0.0,
                Mo     = UniformPrior(0.0, 2π),
            ),),
            rv = (HIRES = (data = _rvdat(), sigma = LogUniformPrior(0.1, 10.0)),),
        )
        # Frozen params are NOT in unfrozen_names
        @test !("sesinw_k1" in target.params.layout.unfrozen_names)
        @test !("secosw_k1" in target.params.layout.unfrozen_names)
    end

    @testset "Auto-detect M_sec_driven parametrization" begin
        target = build_target(
            M_pri = 1.0,
            planets = (b = (
                P      = LogUniformPrior(10.0, 1000.0),
                M_sec  = LogUniformPrior(0.001, 0.5),
                sesinw = UniformPrior(-1.0, 1.0),
                secosw = UniformPrior(-1.0, 1.0),
                Mo     = UniformPrior(0.0, 2π),
            ),),
            rv = (HIRES = (data = _rvdat(), sigma = LogUniformPrior(0.1, 10.0)),),
        )
        @test target.params.config.parametrization.mass === :M_sec_driven
    end

    @testset "Errors" begin
        @test_throws ArgumentError build_target()
        @test_throws ArgumentError build_target(
            M_pri = 1.0,
            planets = (b = (Mo = UniformPrior(0.0, 2π),),),  # no mass param
            rv = (HIRES = (data = _rvdat(),),),
        )
        # `a` declared but no `M_sec`:
        @test_throws ArgumentError build_target(
            M_pri = 1.0,
            planets = (b = (a = LogUniformPrior(1.0, 100.0),
                             sesinw = UniformPrior(-1.0, 1.0),
                             secosw = UniformPrior(-1.0, 1.0),
                             Mo = UniformPrior(0.0, 2π)),),
            rv = (HIRES = (data = _rvdat(),),),
        )
        # rv block missing :data:
        @test_throws ArgumentError build_target(
            M_pri = 1.0,
            planets = (b = (P=LogUniformPrior(10.0, 1000.0),
                             K=LogUniformPrior(1.0, 100.0),
                             sesinw=UniformPrior(-1.0, 1.0),
                             secosw=UniformPrior(-1.0, 1.0),
                             Mo=UniformPrior(0.0, 2π)),),
            rv = (HIRES = (sigma = LogUniformPrior(0.1, 10.0),),),
        )
    end
end
