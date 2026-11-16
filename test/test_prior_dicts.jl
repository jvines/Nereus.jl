# Prior dicts, the form the Python client and the fit_* API send.

using Nereus, Test, Distributions

@testset "prior dicts: a normal takes optional bounds" begin
    free = Nereus._as_prior(Dict("type" => "normal", "mu" => 2.2, "sigma" => 0.02))
    @test free.lo == -Inf && free.hi == Inf

    # An inclination needs a bounded prior: the bare normal reaches below 0.
    inc = Nereus._as_prior(Dict("type" => "normal", "mu" => 2.2, "sigma" => 0.02,
                                "lo" => 0.0, "hi" => π))
    @test (inc.lo, inc.hi) == (0.0, Float64(π))
    @test inc.dist isa Distributions.Truncated

    half = Nereus._as_prior(Dict("type" => "normal", "mu" => 1.0, "sigma" => 0.5, "lo" => 0.0))
    @test (half.lo, half.hi) == (0.0, Inf)

    @test_throws ArgumentError Nereus._as_prior(
        Dict("type" => "normal", "mu" => 1.0, "sigma" => 0.5, "lo" => 2.0, "hi" => 1.0))
end

@testset "prior names: every alias reaches its canonical prior" begin
    A = Nereus.PRIOR_ALIASES
    # No spelling is claimed by two priors (the lookup Dict would keep one silently).
    @test sum(length, values(A)) == length(Nereus._PRIOR_BY_ALIAS)
    for (c, names) in pairs(A), n in names
        @test Nereus.canonical_prior(n) === c
    end
    # Case, separators and a trailing "Prior" are ignored.
    for (s, c) in ("Gaussian" => :normal, "NORM" => :normal, "NormalPrior" => :normal,
                   "log_uniform" => :loguniform, "Log-Uniform" => :loguniform,
                   "LogUniformPrior" => :loguniform, "Jeffreys" => :loguniform,
                   "truncated_normal" => :truncnorm, "Truncated Gaussian" => :truncnorm,
                   "ModJeffreysPrior" => :modjeffreys, "WrappedUniformPrior" => :wrappeduniform,
                   "SinePrior" => :sine, "BetaPrior" => :beta, "FixedPrior" => :fixed,
                   "Dirac" => :fixed, "top-hat" => :uniform, :flat => :uniform)
        @test Nereus.canonical_prior(s) === c
    end
    err = try Nereus.canonical_prior("lorentzian"); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("gaussian", err.msg) && occursin("lorentzian", err.msg)
    # The job-config builder has a constructor for every canonical name.
    @test Set(keys(Nereus._PRIOR_TYPES)) == Set(keys(A))
end

@testset "prior dicts: aliases build the same prior" begin
    P(d...) = Nereus._as_prior(Dict{String,Any}(d...))
    same(a, b) = typeof(a) == typeof(b) && (a.lo, a.hi) == (b.lo, b.hi) &&
                 logpdf(a, 0.3) == logpdf(b, 0.3)
    ref = P("type" => "normal", "mu" => 0.0, "sigma" => 1.0)
    for t in ("gaussian", "gauss", "norm", "Normal", "NormalPrior")
        @test same(P("type" => t, "mu" => 0.0, "sigma" => 1.0), ref)
    end
    @test same(P("type" => "flat", "lo" => 0.0, "hi" => 1.0),
               P("type" => "uniform", "lo" => 0.0, "hi" => 1.0))
    @test same(P("type" => "jeffreys", "lo" => 0.1, "hi" => 10.0),
               LogUniformPrior(0.1, 10.0))
    @test P("type" => "dirac", "value" => 2.5) |> is_fixed
    tn = P("type" => "truncated_normal", "mu" => 0.0, "sigma" => 1.0, "lo" => 0.0)
    @test (tn.lo, tn.hi) == (0.0, Inf) && tn.dist isa Distributions.Truncated
    # A truncnorm with no bounds is refused, not left untruncated.
    @test_throws ArgumentError P("type" => "truncnorm", "mu" => 0.0, "sigma" => 1.0)
    # Types the job config always had, now reachable from a prior dict too.
    @test same(P("type" => "beta", "a" => 0.867, "b" => 3.03), BetaPrior(0.867, 3.03))
    @test same(P("type" => "modjeffreys", "knee" => 1.0, "upper" => 100.0),
               ModJeffreysPrior(1.0, 100.0))
    w = P("type" => "wrapped", "lo" => -π, "hi" => π)
    # Re-pinned: a WrappedUniform is no longer `is_wrapped` (that now means a
    # polar angle folded at its poles); it is the circular Uniform, walled at
    # its chart edges and re-cut (src/circular.jl).
    @test w.dist isa Nereus.WrappedUniform && !Nereus.is_wrapped(w)
    @test P("type" => "isotropic").dist isa Nereus.Sine
    @test_throws ArgumentError P("type" => "lorentzian", "mu" => 0.0)
    @test_throws ArgumentError P("mu" => 0.0, "sigma" => 1.0)
end

@testset "job config: prior type aliases" begin
    cfg(t, args) = Dict{String,Any}("x" => Dict{String,Any}("type" => t, "args" => args))
    ref = Nereus._build_priors(cfg("NormalPrior", [0.0, 1.0]))["x"]
    for t in ("normal", "gaussian", "norm", "Gauss")
        p = Nereus._build_priors(cfg(t, [0.0, 1.0]))["x"]
        @test typeof(p) == typeof(ref) && logpdf(p, 0.3) == logpdf(ref, 0.3)
    end
    @test Nereus._build_priors(cfg("flat", [0.0, 2.0]))["x"].hi == 2.0
    @test Nereus._build_priors(cfg("log_uniform", [1.0, 10.0]))["x"].dist isa LogUniform
    tn = Nereus._build_priors(cfg("truncnorm", [0.0, 1.0, 0.0, 3.0]))["x"]
    @test (tn.lo, tn.hi) == (0.0, 3.0)
    @test_throws ArgumentError Nereus._build_priors(cfg("truncnorm", [0.0, 1.0]))
    @test_throws ArgumentError Nereus._build_priors(cfg("lorentzian", [0.0, 1.0]))

    ext = Nereus._build_external_prior(Dict{String,Any}("quantity" => "ecc",
        "prior" => Dict{String,Any}("type" => "gaussian", "args" => [0.0, 0.3])))
    @test ext.quantity === :ecc && ext.prior.dist isa Normal
end

@testset "prior dicts: fit_rm / fit_ttv / fit_binary take nothing or prior objects" begin
    # These three build a job config. The Python client sends `nothing` when it
    # has no priors and prior objects when it has some; both used to fail.
    @test Nereus._cfg_priors(nothing) == Dict{String,Any}()
    cfg = Nereus._cfg_priors(Dict(:P_k1 => Nereus.UniformPrior(9.0, 11.0),
                                  "e_k1" => 0.0,
                                  "K_k1" => Dict("type" => "UniformPrior",
                                                 "args" => [0.0, 50.0])))
    @test cfg["P_k1"].dist isa Uniform && cfg["e_k1"].dist isa Nereus.Fixed
    @test cfg["K_k1"] isa AbstractDict          # job-config form passes through
    built = Nereus._build_priors(cfg)
    @test built["P_k1"] === cfg["P_k1"] && built["e_k1"].dist isa Nereus.Fixed
    @test built["K_k1"].dist isa Uniform && built["K_k1"].hi == 50.0
end

@testset "fit_rm / fit_ttv / fit_binary route the other fit_* keywords into the config" begin
    cfg() = Dict{String,Any}("star" => Dict{String,Any}(),
                             "model" => Dict{String,Any}("max_kplanet" => 1))
    c = Nereus._cfg_route_kw!(cfg(), "fit_ttv",
                              (M_s = 1.0, R_s = 0.9, T_eff = 5700.0, stability = :none,
                               ttv_backend = :ttvfaster, time_anchor = :phic,
                               parametrization = :a_driven))
    @test c["star"] == Dict("M_s" => 1.0, "R_s" => 0.9, "T_eff" => 5700.0)
    @test c["model"]["stability"] == "none" && c["model"]["ttv_backend"] == "ttvfaster"
    @test c["model"]["parametrization"] == Dict("time" => "phic", "mass" => "a_driven")
    e = try Nereus._cfg_route_kw!(cfg(), "fit_ttv", (density = :rho_star,)); nothing
        catch err; err end
    @test e isa ArgumentError && occursin("no keyword `density`", e.msg) &&
          occursin("run_job", e.msg)
    c2 = cfg(); c2["star"]["M_s"] = 1.0
    @test_throws ArgumentError Nereus._cfg_route_kw!(c2, "fit_binary", (M_s = 1.1,))
end

@testset "fit_rm: the default time anchor applies" begin
    # fit_rm passed time = "Tc", so its Tc window held several copies of a
    # short-period orbit; the other fits take the default anchor (φ_c)
    @test !haskey(Nereus._RM_PARAMETRIZATION, "time")
    @test Nereus._rm_parametrization(:M_sec_driven) ==
          merge(Nereus._RM_PARAMETRIZATION, Dict("mass" => "M_sec_driven"))
    @test Nereus._rm_parametrization(Dict(:time => "Tc", :mass => "K_driven")) ==
          Dict("time" => "Tc", "mass" => "K_driven")
    cfg = Dict{String,Any}("star" => Dict{String,Any}(),
        "model" => Dict{String,Any}("parametrization" =>
                                    Nereus._rm_parametrization(Nereus._RM_PARAMETRIZATION)))
    Nereus._cfg_route_kw!(cfg, "fit_rm", (time_anchor = :Tc,))
    @test cfg["model"]["parametrization"]["time"] == "Tc"
    @test !haskey(Nereus._RM_PARAMETRIZATION, "time")      # the default is not changed
end
