# Runner dispatch tests — exercise `_dispatch_sampler` through the
# JSON-native config path (Strings for Symbols, Ints for Floats, Arrays
# for Tuples) and verify each sampler's result tuple is extracted with
# the right field names.
#
# These guard a class of bugs the per-sampler unit tests CANNOT catch,
# because they live in the runner glue, not the samplers: e.g. the `pa`
# / `smc` branches read `res.log_z` / `res.mean_acceptance` when
# PAResult actually exposes `log_evidence` / `acceptance`, and JSON
# delivers `mutation_kernel`/`bounds`/`proposal` as Strings, not
# Symbols. Every failure in this file mirrors a real run_job crash.

using Test
using Random
using Nereus
using MCMCChains

function _disp_target(seed::Int = 2027; n_obs::Int = 30)
    rng    = MersenneTwister(seed)
    t      = sort(rand(rng, n_obs) .* 40.0)
    rv     = 12.0 .* sin.(2π .* t ./ 8.0) .+ randn(rng, n_obs)
    data   = Data(; t_rv = t, rv = rv, rv_err = fill(1.0, n_obs))
    ic     = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                      instruments = ic, data = data, M_s = 1.0)
    target = NereusTarget(params, data; unconstrained = false)
    return target, data
end

# Build the cfg the way run_job sees it and dispatch.
_dispatch(name, kwargs, target, data) =
    Nereus._dispatch_sampler(
        Dict(:sampler => Dict(:name => name, :kwargs => kwargs)),
        target, data, 1)

@testset "runner dispatch — JSON-native kwarg types" begin
    target, data = _disp_target()

    @testset "pa — String mutation_kernel + Int floats + field extraction" begin
        # mutation_kernel as String (→Symbol), and step_scale/stretch_a
        # as Ints (→Float64) — both JSON-native types in one call.
        res = _dispatch("pa", Dict(:mutation_kernel => "stretch",
                                    :n_replicas => 40, :n_mcmc => 4,
                                    :max_steps => 30, :step_scale => 1,
                                    :stretch_a => 2, :show_progress => false),
                        target, data)
        @test res.chains isa MCMCChains.Chains
        @test isfinite(res.log_evidence)
        @test res.n_evals > 0
    end

    @testset "smc — JSON-array betas + PAResult field extraction" begin
        # betas arrives as a JSON3.Array (here a Vector); must be
        # accepted and normalized to Vector{Float64}.
        res = _dispatch("smc", Dict(:betas => [0.0, 0.5, 1.0],
                                     :n_replicas => 40, :n_mcmc => 4,
                                     :show_progress => false),
                        target, data)
        @test res.chains isa MCMCChains.Chains
        @test isfinite(res.log_evidence)
    end

    @testset "smc — runs without betas (default ladder)" begin
        # betas is only a step-count proxy and defaults; a config that
        # omits it must still run rather than UndefKeywordError.
        res = _dispatch("smc", Dict(:n_replicas => 40, :n_mcmc => 4,
                                     :show_progress => false),
                        target, data)
        @test res.chains isa MCMCChains.Chains
        @test isfinite(res.log_evidence)
    end

    @testset "nested + nested_ins — String bounds/proposal coerced centrally" begin
        for name in ("nested", "nested_ins")
            res = _dispatch(name, Dict(:bounds => "single",
                                        :proposal => "rwalk", :dlogz => 2),
                            target, data)
            @test res.chains isa MCMCChains.Chains
            @test isfinite(res.log_evidence)
        end
    end

    @testset "nested_dynamic — String bounds/proposal + Higson knobs" begin
        res = _dispatch("nested_dynamic",
                        Dict(:bounds => "single", :proposal => "rwalk",
                             :pfrac => 0, :dlogz_init => 2,
                             :n_live_init => 80, :n_live_batch => 50),
                        target, data)
        @test res.chains isa MCMCChains.Chains
        @test isfinite(res.log_evidence)
    end

    @testset "unsupported kwarg → clear ArgumentError, not MethodError" begin
        # NOTE: the original example (:enlarge) went stale — sample_nested
        # gained a real `enlarge` kwarg (nested.jl, MLFriends/high-dim work),
        # so it is now legitimately accepted. Use a name no sampler declares.
        @test_throws ArgumentError _dispatch("nested",
            Dict(:definitely_not_a_kwarg => 1.25), target, data)
    end
end

# Trans-dim samplers need a `transdim` block in the cfg and a target
# with max_kplanet ≥ 2. They exercise the same JSON-native kwarg path
# (String within_model → Symbol, Int init_scale/dlogz → Float64) through
# _dispatch_sampler, plus the td-block construction.
function _disp_target_td(seed::Int = 2027; n_obs::Int = 40)
    rng    = MersenneTwister(seed)
    t      = sort(rand(rng, n_obs) .* 60.0)
    rv     = 12.0 .* sin.(2π .* t ./ 8.0) .+ randn(rng, n_obs)
    data   = Data(; t_rv = t, rv = rv, rv_err = fill(1.0, n_obs))
    ic     = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 2, planet_modes = [RV_ONLY, RV_ONLY],
                      instruments = ic, data = data, M_s = 1.0)
    target = NereusTarget(params, data; unconstrained = false)
    return target, data
end

_dispatch_td(name, kwargs, target, data) =
    Nereus._dispatch_sampler(
        Dict(:sampler   => Dict(:name => name, :kwargs => kwargs),
             # Explicit empty arrays mirror what a JSON config sends
             # (`toggleable: []`) — empty JSON3.Array, not the typed
             # default. Must coerce to Vector{NoiseModel}.
             :transdim  => Dict(:max_kplanet => 2, :transdim_fraction => 0.4,
                                :toggleable => [], :noise_exclusion_groups => [])),
        target, data, 1)

@testset "runner dispatch — trans-dim samplers (td block + JSON kwargs)" begin
    target, data = _disp_target_td()

    @testset "moms — String within_model + Int floats" begin
        res = _dispatch_td("moms",
                           Dict(:within_model => "slice", :init_scale => 1,
                                :inclusion_prior => 0.5, :n_warmup => 150,
                                :n_samples => 300, :show_progress => false),
                           target, data)
        @test res.chains isa MCMCChains.Chains
        @test :n_planets in names(res.chains, :parameters)
        @test res.n_evals > 0
    end

    @testset "moms_ns — Int dlogz/init_scale" begin
        res = _dispatch_td("moms_ns",
                           Dict(:dlogz => 2, :init_scale => 1, :n_live => 40,
                                :n_mcmc => 4, :max_iter => 1500,
                                :show_progress => false),
                           target, data)
        @test res.chains isa MCMCChains.Chains
        @test :n_planets in names(res.chains, :parameters)
        # moms_ns is a nested sampler — its real log Z must be surfaced,
        # not discarded as NaN (regression: the dispatch used to capture
        # the whole (chains, logZ, strategy) tuple as `chains`).
        @test isfinite(res.log_evidence)
    end

    @testset "rjmcmc — Int initial_scale/target_accept" begin
        res = _dispatch_td("rjmcmc",
                           Dict(:initial_scale => 1, :target_accept => 1,
                                :n_warmup => 150, :n_samples => 300,
                                :show_progress => false),
                           target, data)
        @test res.chains isa MCMCChains.Chains
        @test :n_planets in names(res.chains, :parameters)
    end

    @testset "transdim_ptemcee — Int stretch_a/inclusion_prior" begin
        res = _dispatch_td("transdim_ptemcee",
                           Dict(:stretch_a => 2, :inclusion_prior => 0.5,
                                :n_temps => 4, :n_walkers => 20,
                                :n_steps => 200, :n_burnin => 100,
                                :show_progress => false),
                           target, data)
        @test res.chains isa MCMCChains.Chains
    end

    @testset "unsupported kwarg on a trans-dim sampler → ArgumentError" begin
        # transdim_fraction belongs in the `transdim` block, not
        # sampler.kwargs; within_step is not an rjmcmc kwarg. Both must
        # produce a clean ArgumentError, not a MethodError. (Regression:
        # the kwarg guard used to apply only to the nested family.)
        @test_throws ArgumentError _dispatch_td("rjmcmc",
            Dict(:transdim_fraction => 0.4, :within_step => 0.1),
            target, data)
    end
end

# ---------------------------------------------------------------------------
# use_rho_s wired through the run_job config path (ExoAutomata handoff).
# ρ⋆ is what transits actually constrain; ParametrizationConfig always had the
# field and the whole model/likelihood/prior stack dispatches on it, but the
# runner's config parse never read it — so it could never be enabled from a
# job config. These lock in the read AND the omitted-key backward compat.
@testset "use_rho_s through run_job config (ExoAutomata handoff)" begin
    n  = 400
    tp = collect(range(0.0, 20.0; length = n))
    dphot = Data(; t_phot = tp,
                   flux = 1.0 .+ 1e-4 .* randn(MersenneTwister(3), n),
                   flux_err = fill(3e-4, n), phot_inst = ones(Int, n))
    star = (; M_s = 1.0, R_s = 1.0)
    cfg = Dict{String, Any}(
        "model" => Dict{String, Any}(
            "max_kplanet"  => 1,
            "planet_modes" => ["PM_ONLY"],
            "parametrization" => Dict{String, Any}(
                "time" => "Tc", "geom" => "b_rr", "use_rho_s" => true)))
    p1, _, _ = Nereus._build_model(cfg, dphot, star, String[], ["TESS"])
    @test p1.config.parametrization.use_rho_s === true
    @test haskey(p1.layout.name_to_idx, "rho_s")     # shared ρ⋆ slot allocated
    @test "rho_s" in p1.layout.unfrozen_names        # sampled ⇒ auto prior attached

    # Negative mirror: omitted key ⇒ false ⇒ no ρ⋆ slot (exact old behavior).
    cfg2 = deepcopy(cfg)
    delete!(cfg2["model"]["parametrization"], "use_rho_s")
    p2, _, _ = Nereus._build_model(cfg2, dphot, star, String[], ["TESS"])
    @test p2.config.parametrization.use_rho_s === false
    @test !haskey(p2.layout.name_to_idx, "rho_s")
end
