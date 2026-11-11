# Per-instrument along-scan jitter on the IAD likelihood (`sigma_iad_<name>`),
# and the within-transit outlier rule (`clip_transit_outliers`).
#
# The jitter turns W = diag(1/σ²) into a function of a parameter. The
# marginalisation of the catalogue solution stays exact, but log det A and
# Σ log σ_eff stop being constants -- so the check that matters is against an
# INDEPENDENT evaluation of the same marginal: q as an explicit Gaussian
# nuisance with a huge prior width, integrated out in closed form on the dense
# n×n covariance, in BigFloat. That never forms the normal equations at all.

using Random: MersenneTwister
import LinearAlgebra

# A Gaia-like single source: `n_tr` field-of-view transits of `n_ccd` CCDs,
# 5 s apart, the scan angle drifting 3.6e-6 rad per CCD as it does in DR4,
# with an orbit, a parallax and an optional proper motion in the abscissae.
# Noise is drawn at √(σ² + s_true²): s_true is the excess the jitter is for.
function _jit_iad(; n_tr = 24, n_ccd = 9, σ = 0.1, s_true = 0.0, pm = (0.0, 0.0),
                  plx = 20.0, orb = nothing, M_sec = 0.0, seed = 7, t0 = 57000.0,
                  inst = 1)
    rng = MersenneTwister(seed)
    t = Float64[]; psi = Float64[]; a = Float64[]; plxf = Float64[]; pmf = Float64[]
    for k in 1:n_tr
        tk = t0 + 1000.0 * (k - 0.5) / n_tr + 0.37 * k
        ψ0 = 2π * rand(rng)
        pf = sin(2π * (tk - t0) / 365.25 - ψ0)
        for c in 1:n_ccd
            tj = tk + (c - 1) * 5.0 / 86400
            ψ  = ψ0 + (c - 1) * 3.6e-6
            yr = (tj - (t0 + 500.0)) / 365.25
            Δα = pm[1] * yr; Δδ = pm[2] * yr
            if orb !== nothing
                ra, de = Nereus.star_reflex_offset(orb, tj, M_sec)
                Δα += ra; Δδ += de
            end
            push!(t, tj); push!(psi, ψ); push!(plxf, pf); push!(pmf, yr)
            push!(a, Δα * sin(ψ) + Δδ * cos(ψ) + plx * pf +
                     sqrt(σ^2 + s_true^2) * randn(rng))
        end
    end
    return IADData(t = t, abscissa = a, abscissa_err = fill(σ, length(t)), psi = psi,
                   parallax_factor = plxf, pm_factor = pmf)
end

const _JIT_ORB = (P = 3 * 365.25, e = 0.3, ω = 0.5, Ω = 1.2, i = deg2rad(60), M = 1.0,
                  M_sec = 0.02, tp = 57100.0, plx = 20.0)
_jit_orbit() = Nereus.build_orbit(_JIT_ORB.P, _JIT_ORB.e, _JIT_ORB.ω, _JIT_ORB.Ω,
                                  _JIT_ORB.i, _JIT_ORB.M, _JIT_ORB.M_sec, _JIT_ORB.tp,
                                  _JIT_ORB.plx)

function _jit_params(iad; iad_names = String[], gaia = false)
    kw = Dict{Symbol, Any}(:iad => iad)
    if gaia
        gost = GOSTData(t = collect(range(57000.0, 58000.0, length = 30)),
                        psi = mod.(2π .* (1:30) ./ 7, 2π),
                        parallax_factor = sin.(2π .* (1:30) ./ 365.25))
        kw[:gost] = gost
        cov = zeros(5, 5); for k in 1:5; cov[k, k] = (k <= 2 ? 0.05 : 0.02)^2; end
        # What DR3 published: the fit through its own window's transits.
        Δq = gost_5param_fit(_jit_orbit(), gost, _JIT_ORB.M_sec; t_ref = jyear_to_mjd(2016.0),
                             window = Nereus.GAIA_DR3_WINDOW)
        kw[:gaia_dr3] = GaiaDR3Data(params = (Δq[1], Δq[2], _JIT_ORB.plx + Δq[3], Δq[4], Δq[5]),
                                    cov = cov, t_ref = jyear_to_mjd(2016.0))
    end
    data = Data(; t_rv = [57000.0, 57300.0], rv = [10.0, -10.0], rv_err = [1.0, 1.0], kw...)
    params = Params(max_kplanet = 1, planet_modes = [RVAS],
        instruments = InstrumentConfig(rv = ["X"], iad = iad_names), data = data,
        stability = :none, M_s = _JIT_ORB.M,
        parametrization = ParametrizationConfig(time = :Mo, mass = :a_driven),
        priors = Dict{String, PriorSpec}(
            "n_p" => FixedPrior(1.0), "a_k1" => LogUniformPrior(0.5, 100.0),
            "M_sec_k1" => LogUniformPrior(0.001, 0.5),
            "sesinw_k1" => UniformPrior(-1.0, 1.0), "secosw_k1" => UniformPrior(-1.0, 1.0),
            "Mo_k1" => UniformPrior(0.0, 2π), "inc_k1" => SinePrior(),
            "Omega_k1" => UniformPrior(0.0, 2π), "sigma_X" => LogUniformPrior(0.1, 10.0),
            "M_pri" => FixedPrior(_JIT_ORB.M)))
    th = Theta(params)
    a = ((_JIT_ORB.M + _JIT_ORB.M_sec) * (_JIT_ORB.P / Nereus.KEPLER_YEAR_DAYS)^2)^(1/3)
    set_param!(th, "n_p", 1); set_param!(th, "a_k1", a)
    set_param!(th, "M_sec_k1", _JIT_ORB.M_sec)
    set_param!(th, "sesinw_k1", sqrt(_JIT_ORB.e) * sin(_JIT_ORB.ω))
    set_param!(th, "secosw_k1", sqrt(_JIT_ORB.e) * cos(_JIT_ORB.ω))
    set_param!(th, "Mo_k1", 2π * (57150.0 - _JIT_ORB.tp) / _JIT_ORB.P)
    set_param!(th, "inc_k1", _JIT_ORB.i); set_param!(th, "Omega_k1", _JIT_ORB.Ω)
    set_param!(th, "plx", _JIT_ORB.plx); set_param!(th, "gamma_X", 0.0)
    set_param!(th, "sigma_X", 1.0)
    return data, params, th
end

# The marginal of the abscissae over a flat-prior catalogue solution q, the
# slow way: q ~ N(0, λ²I), r ~ N(Xq, Σ) integrates to r ~ N(0, Σ + λ²XXᵀ), and
# the flat prior is the λ → ∞ limit times (2πλ²)^(n_q/2). Dense, BigFloat,
# no normal equations. Relative error ~ |q̂|²/λ².
function _dense_marginal(theta, data; λ = big(1e6))
    iad = data.iad
    n = n_iad(iad); n_inst = n_iad_inst(iad); n_q = Nereus._iad_n_q(n_inst)
    plx = astrom_plx(theta)
    _, orbs, M_secs = Nereus._iad_active_orbits(theta, astrom_M_pri(theta), plx, data.t_ref)
    r = Vector{Float64}(undef, n)
    Nereus._iad_residuals!(r, iad, orbs, M_secs, plx)
    pos = Nereus._iad_pos_cols(n_inst)
    setprecision(BigFloat, 256) do
        X = zeros(BigFloat, n, n_q)
        Σ = zeros(BigFloat, n, n)
        for j in 1:n
            m = iad.inst[j]; s, c = sincos(big(iad.psi[j])); pmf = big(iad.pm_factor[j])
            X[j, pos[m]] = s; X[j, pos[m] + 1] = c; X[j, 3] = s * pmf; X[j, 4] = c * pmf
            Σ[j, j] = big(iad.abscissa_err[j])^2 + big(iad_jitter(theta, m))^2
        end
        C = LinearAlgebra.Symmetric(Σ + λ^2 * X * X')
        F = LinearAlgebra.cholesky(C)
        rb = big.(r)
        quad = rb' * (F \ rb)
        ld = 2 * sum(log, LinearAlgebra.diag(F.U))
        ll = -quad / 2 - ld / 2 - n / 2 * log(2 * big(π)) + n_q / 2 * log(2 * big(π) * λ^2)
        Float64(ll)
    end
end

@testset "IAD jitter + transit outliers" begin

    orb = _jit_orbit()
    iad1 = _jit_iad(; s_true = 0.15, orb = orb, M_sec = _JIT_ORB.M_sec)

    @testset "off by default" begin
        _, p, _ = _jit_params(iad1)
        @test isempty(p.layout.systemic.iad_jitter)
        @test !any(startswith("sigma_iad_"), p.layout.names)
        @test n_iad_jitters(p.config.instruments) == 0
        # Three-argument positional form still builds (callers predating it).
        @test InstrumentConfig(["A"], String[], String[]).iad_names == String[]
    end

    @testset "layout, accessor, default prior, units, jitter set" begin
        _, p, th = _jit_params(iad1; iad_names = ["G"])
        @test "sigma_iad_G" in p.layout.names
        @test p.layout.systemic.iad_jitter == [p.layout.name_to_idx["sigma_iad_G"]]
        set_param!(th, "sigma_iad_G", 0.3)
        @test iad_jitter(th, 1) == 0.3
        @test iad_jitter(th, 2) == 0.0              # no such instrument: zero
        # Default prior: ModJeffreys from 0, knee at the median reported error,
        # cap at ten times it -- the same shape as `sigma_as_*`.
        k = findfirst(==("sigma_iad_G"), p.layout.unfrozen_names)
        pr = p.layout.unfrozen_priors[k]
        @test pr.dist isa Nereus.ModJeffreys
        @test pr.lo == 0.0
        @test pr.dist.knee ≈ median(iad1.abscissa_err)
        @test pr.hi ≈ 10 * median(iad1.abscissa_err)
        # A jitter railing at zero is a value, not a pathology: it is in the
        # set fit_health / the science tables / the MAP exempt at the lower bound.
        @test "sigma_iad_G" in Nereus.jitter_names(p)
        @test Nereus.sci_param_unit("sigma_iad_G", p) == ("mas", false)
    end

    @testset "configuration errors" begin
        # Positional: one entry per instrument of the data.
        @test_throws ArgumentError _jit_params(iad1; iad_names = ["A", "B"])
        two = merge_iad(iad1, _jit_iad(; seed = 9, t0 = 58200.0))
        @test_throws ArgumentError _jit_params(two; iad_names = ["A"])
        # A jitter with no IAD to put it on.
        @test_throws ArgumentError Params(max_kplanet = 1, planet_modes = [RV_ONLY],
            instruments = InstrumentConfig(rv = ["X"], iad = ["G"]),
            data = Data(t_rv = [1.0, 2.0, 3.0], rv = [0.0, 1.0, 0.0], rv_err = ones(3)),
            stability = :none, M_s = 1.0)
    end

    @testset "jitter at zero is the unjittered likelihood, bit for bit" begin
        for gaia in (false, true)
            d0, _, th0 = _jit_params(iad1; gaia = gaia)
            d1, _, th1 = _jit_params(iad1; iad_names = ["G"], gaia = gaia)
            set_param!(th1, "sigma_iad_G", 0.0)
            ll0 = iad_log_likelihood(th0, d0)
            @test isfinite(ll0)
            @test iad_log_likelihood(th1, d1) === ll0
            set_param!(th1, "sigma_iad_G", 0.2)
            @test iad_log_likelihood(th1, d1) != ll0
        end
    end

    @testset "matches the dense explicit-nuisance marginal" begin
        d, _, th = _jit_params(iad1; iad_names = ["G"])
        for s in (0.0, 0.05, 0.15, 0.6)
            set_param!(th, "sigma_iad_G", s)
            @test iad_log_likelihood(th, d) ≈ _dense_marginal(th, d) atol = 1e-7
        end
        # Off the orbit too: the jitter weights the orbit residuals, not a
        # constant.
        set_param!(th, "sigma_iad_G", 0.15)
        set_param!(th, "inc_k1", deg2rad(40))
        @test iad_log_likelihood(th, d) ≈ _dense_marginal(th, d) atol = 1e-7

        # Data drawn with a 0.15 mas excess: the likelihood prefers a jitter
        # near it over none and over a large one.
        set_param!(th, "inc_k1", _JIT_ORB.i)
        ll(s) = (set_param!(th, "sigma_iad_G", s); iad_log_likelihood(th, d))
        @test ll(0.15) > ll(0.0)
        @test ll(0.15) > ll(0.6)
    end

    @testset "per instrument: only the opted-in one is inflated" begin
        two = merge_iad(iad1, _jit_iad(; seed = 9, t0 = 58200.0, σ = 0.3,
                                        orb = orb, M_sec = _JIT_ORB.M_sec))
        d0, _, th0 = _jit_params(two)
        d, p, th = _jit_params(two; iad_names = ["", "B"])
        @test p.layout.systemic.iad_jitter[1] == 0
        @test !("sigma_iad_" in p.layout.names)
        set_param!(th, "sigma_iad_B", 0.0)
        @test iad_log_likelihood(th, d) === iad_log_likelihood(th0, d0)
        set_param!(th, "sigma_iad_B", 0.4)
        @test iad_jitter(th, 1) == 0.0
        @test iad_log_likelihood(th, d) ≈ _dense_marginal(th, d) atol = 1e-7
        # and the effective errors used downstream say so
        σe = iad_sigma_eff(th, d.iad)
        @test σe[d.iad.inst .== 1] == d.iad.abscissa_err[d.iad.inst .== 1]
        @test σe[d.iad.inst .== 2] ≈ sqrt.(0.3^2 + 0.4^2) .* ones(count(==(2), d.iad.inst))
    end

    @testset "ForwardDiff flows through the jitter" begin
        d, p, th = _jit_params(iad1; iad_names = ["G"])
        js = p.layout.name_to_idx["sigma_iad_G"]
        ji = p.layout.name_to_idx["inc_k1"]
        f = v -> iad_log_likelihood(Theta(p, v), d)
        for s in (0.0, 0.12)
            v0 = copy(th.values); v0[js] = s
            g = ForwardDiff.gradient(f, v0)
            @test all(isfinite, g)
            for (idx, h) in ((js, 1e-6), (ji, 1e-7))
                vp = copy(v0); vp[idx] += h
                vm = copy(v0); vm[idx] -= h
                fd = (f(vp) - f(vm)) / (2h)
                # At s = 0 the likelihood is even in s, so its slope is zero;
                # the central difference is then O(h), not O(h²).
                @test g[idx] ≈ fd atol = 1e-4 * max(1.0, abs(fd))
            end
            s == 0.0 && @test g[js] == 0.0
        end
        # The joint Hipparcos IAD + Gaia DR3 path differentiates too.
        dg, pg, thg = _jit_params(iad1; iad_names = ["G"], gaia = true)
        vg = copy(thg.values); vg[pg.layout.name_to_idx["sigma_iad_G"]] = 0.1
        @test all(isfinite, ForwardDiff.gradient(v -> iad_log_likelihood(Theta(pg, v), dg), vg))
    end

    @testset "residuals and χ² downstream use the jittered errors" begin
        d, _, th = _jit_params(iad1; iad_names = ["G"])
        set_param!(th, "sigma_iad_G", 0.25)
        fit = Nereus._iad_oc(th, d)
        @test fit.σ ≈ sqrt.(iad1.abscissa_err .^ 2 .+ 0.25^2)
        # Σ(oc/σ_eff)² is the χ²_min the jittered likelihood minimised.
        n_q = Nereus._iad_n_q(1)
        A = zeros(n_q, n_q); v = zeros(n_q)
        r = Vector{Float64}(undef, n_iad(iad1))
        _, orbs, Ms = Nereus._iad_active_orbits(th, astrom_M_pri(th), astrom_plx(th), d.t_ref)
        Nereus._iad_residuals!(r, iad1, orbs, Ms, astrom_plx(th))
        rWr, _ = Nereus._iad_normal_equations!(A, v, iad1, r, iad1.pm_factor,
                                               Nereus._iad_pos_cols(1), [0.25^2])
        χ2, _ = Nereus._iad_solve(A, v, rWr, n_q)
        @test sum(abs2, fit.oc ./ fit.σ) ≈ χ2 rtol = 1e-9
        # Without a jitter slot the error bars are the reported ones.
        d0, _, th0 = _jit_params(iad1)
        @test Nereus._iad_oc(th0, d0).σ == iad1.abscissa_err
    end

    @testset "outliers within a transit" begin
        # A Barnard's-star proper motion: across-scan offsets of ~10⁴ mas make
        # the 3e-5 rad scan-angle drift inside a transit worth ~0.3 mas, i.e.
        # 3σ at the ends of a transit. Corrected, nothing is flagged.
        clean = _jit_iad(; σ = 0.1, pm = (-800.0, 10300.0), orb = orb, M_sec = _JIT_ORB.M_sec)
        c2, rep = clip_transit_outliers(clean; verbose = false)
        @test rep.n_rejected == 0
        @test rep.n_transits_tested == 24
        @test c2 === clean                          # nothing to remove: same object

        a = copy(clean.abscissa)
        # (1) a single CCD 1 mas (10σ) off in transit 3
        a[2 * 9 + 4] += 1.0
        # (2) transit 7 scattered across its CCDs, no majority agreeing
        a[6 * 9 .+ (1:9)] .+= [1.0, -0.8, 0.0, 2.0, -1.5, 0.05, 1.2, -2.2, 0.9]
        # (3) transit 11 shifted as a WHOLE by 2 mas: what an orbit does. The
        #     rule cannot see this, and must not.
        a[10 * 9 .+ (1:9)] .+= 2.0
        bad = IADData(t = clean.t, abscissa = a, abscissa_err = clean.abscissa_err,
                      psi = clean.psi, parallax_factor = clean.parallax_factor,
                      pm_factor = clean.pm_factor)
        out, rep = clip_transit_outliers(bad; verbose = false)
        whole = 6 * 9 .+ (1:9)
        @test sort(rep.index) == sort(vcat(2 * 9 + 4, collect(whole)))
        @test rep.n_rejected == 10
        @test rep.n_transits_rejected == 1
        @test all(rep.whole_transit[findall(in(whole), rep.index)])
        @test !rep.whole_transit[findfirst(==(2 * 9 + 4), rep.index)]
        k = findfirst(==(2 * 9 + 4), rep.index)
        @test rep.transit[k] == 3 && rep.ccd[k] == 4
        @test rep.deviation[k] ≈ 1.0 atol = 0.3
        @test rep.z[k] > 5
        @test n_iad(out) == n_iad(bad) - 10
        @test !any(i -> in(i, rep.index), 10 * 9 .+ (1:9))     # coherent shift kept
        # The input is never modified.
        @test bad.abscissa == a

        # A looser threshold catches no fewer, a tighter one no more.
        @test clip_transit_outliers(bad; nsigma = 3, verbose = false)[2].n_rejected >= 10
        @test clip_transit_outliers(bad; nsigma = 20, verbose = false)[2].n_rejected <= 10

        # Groups too small to vote, or spanning a change of scan direction
        # (Hipparcos records), are never tested.
        tiny = _jit_iad(; n_ccd = 2)
        @test clip_transit_outliers(tiny; verbose = false)[2].n_transits_tested == 0
        hipish = IADData(t = repeat(57000.0 .+ 30.0 .* (0:9); inner = 4),
                         abscissa = 5.0 .* randn(MersenneTwister(3), 40),
                         abscissa_err = fill(1.0, 40),
                         psi = repeat(2π .* rand(MersenneTwister(4), 10); inner = 4) .+
                               repeat([0.0, 0.004, 0.009, 0.013], 10),
                         parallax_factor = zeros(40), pm_factor = zeros(40))
        @test clip_transit_outliers(hipish; verbose = false)[2].n_transits_tested == 0
        @test_throws ArgumentError clip_transit_outliers(bad; min_ccds = 2)
        @test_throws ArgumentError clip_transit_outliers(bad; nsigma = 0)

        # Multi-instrument: indices refer to the merged container, and the
        # clean copy keeps both instruments' reference solutions.
        two = merge_iad(clean, bad)
        out2, rep2 = clip_transit_outliers(two; verbose = false)
        @test rep2.n_rejected == 10
        @test all(==(2), rep2.inst)
        @test n_iad_inst(out2) == 2
    end

    @testset "run_job: `jitter` and `clip_outliers` on a data.iad block" begin
        dat = _jit_iad(; σ = 0.1, orb = orb, M_sec = _JIT_ORB.M_sec)
        a = copy(dat.abscissa); a[5] += 3.0
        vals(x) = Dict{String, Any}("t" => dat.t, "abscissa" => x,
                                    "abscissa_err" => dat.abscissa_err, "psi" => dat.psi,
                                    "parallax_factor" => dat.parallax_factor,
                                    "pm_factor" => dat.pm_factor)
        rv = Dict{String, Any}("values" => Dict{String, Any}(
            "bjd" => [2457000.5, 2457100.5, 2457200.5], "rv" => [1.0, -1.0, 0.5],
            "rv_err" => [1.0, 1.0, 1.0], "instrument" => ["X", "X", "X"]))
        dcfg = Dict{String, Any}("rv" => rv, "iad" => [Dict{String, Any}(
            "instrument" => "G", "jitter" => true, "clip_outliers" => true,
            "values" => vals(a))])
        data, rvn, pmn, asn, iadn, outl = Nereus._build_data(dcfg)
        @test iadn == ["G"]
        @test length(outl) == 1 && outl[1].instrument == "G"
        @test outl[1].index == [5]
        @test n_iad(data.iad) == n_iad(dat) - 1
        cfg = Dict{String, Any}("model" => Dict{String, Any}(
            "max_kplanet" => 1, "planet_modes" => ["RVAS"],
            "parametrization" => Dict{String, Any}("mass" => "a_driven")),
            "priors" => Dict{String, Any}("M_pri" => Dict("type" => "fixed", "args" => [1.0])))
        p, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict("M_s" => 1.0)),
                                      rvn, pmn, asn, iadn)
        @test "sigma_iad_G" in p.layout.names
        # The summary block a client reads.
        s = Nereus._iad_outlier_summary(outl)
        @test s[1]["instrument"] == "G" && s[1]["n_rejected"] == 1

        # Off by default: no key, no slot, no clipping, data untouched.
        dcfg0 = Dict{String, Any}("rv" => rv, "iad" => Dict{String, Any}("values" => vals(a)))
        data0, _, _, _, iadn0, outl0 = Nereus._build_data(dcfg0)
        @test isempty(iadn0) && isempty(outl0)
        @test data0.iad.abscissa == a

        # Typos are errors, not models.
        bad = deepcopy(dcfg); bad["iad"][1]["jitter"] = "yes"
        @test_throws ArgumentError Nereus._build_data(bad)
        bad = deepcopy(dcfg); bad["iad"][1]["clip_outliers"] = -1
        @test_throws ArgumentError Nereus._build_data(bad)
        vcfg = Dict{String, Any}("output_dir" => "/nonexistent", "sampler" => Dict(),
                                 "model" => cfg["model"], "data" => bad)
        err = try; Nereus._validate_config(vcfg); nothing; catch e; e; end
        @test err isa ArgumentError && occursin("clip_outliers", err.msg)

        # fit_* specs: an already-built IADData carries no jitter request.
        iad, names, reps = Nereus._resolve_iad([dat, dat])
        @test n_iad_inst(iad) == 2 && names == ["", ""] && isempty(reps)
        @test Nereus._iad_block_name(Dict("catalogue" => "hipparcos", "hip" => 64426), 1) ==
              "HIP64426"
        @test Nereus._iad_block_name(Dict("catalogue" => "gaia_dr4", "source_id" => 12), 1) ==
              "GaiaDR4_12"
    end
end
