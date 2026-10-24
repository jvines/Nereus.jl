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
    @test Nereus.is_wrapped(w)
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
