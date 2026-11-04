# `burnin_hold` (src/samplers/burnin_hold.jl): the PT ensembles hold chosen
# parameters at one value, the same for every walker of every rung, for the first
# part of burn-in, then draw them from their priors walker by walker. By default
# that is every IAD jitter `sigma_iad_<instrument>` at 0, where the likelihood is
# the unjittered one: from the prior an IAD jitter otherwise absorbs the reflex
# before any walker finds the orbit (HD 114762 RV + Gaia DR4: every pt_emcee run
# ended 470-970 nats below it).
#
# Checked here, on runs of a few seconds: the keywords resolve and refuse what
# cannot be held; while held, every walker carries the value exactly (on
# pt_emcee's flat scale too, where exp(log(x + s)) - s need not round back to x)
# and its cached log L is the likelihood there; after the release the value
# spreads; a run stopped inside the hold and resumed across the release is the
# uninterrupted run bit for bit; for pt_emcee, transdim_pt_emcee and
# pt_whitening.
using Test
using Nereus
using Nereus: IADData, TransDimConfig, reset_transdim_caches!
using Random
using Serialization

# --- An astrometry-only source with a Gaia-like IAD and one jittered instrument.
function _bh_iad(; n_tr = 20, n_ccd = 8, σ = 0.1, plx = 20.0, seed = 3)
    rng = MersenneTwister(seed)
    t = Float64[]; psi = Float64[]; a = Float64[]; pf = Float64[]; pmf = Float64[]
    for k in 1:n_tr
        tk = 57000.0 + 1000.0 * (k - 0.5) / n_tr
        ψ0 = 2π * rand(rng)
        f = sin(2π * (tk - 57000.0) / 365.25 - ψ0)
        for c in 1:n_ccd
            push!(t, tk + (c - 1) * 5.0 / 86400); push!(psi, ψ0 + (c - 1) * 3.6e-6)
            push!(pf, f); push!(pmf, (tk - 57500.0) / 365.25)
            push!(a, plx * f + sqrt(σ^2 + 0.05^2) * randn(rng))
        end
    end
    return IADData(t = t, abscissa = a, abscissa_err = fill(σ, length(t)), psi = psi,
                   parallax_factor = pf, pm_factor = pmf)
end
const _BH_IAD = _bh_iad()
_bh_iad_target(; jitter = true) = build_target(
    planets = (b = (a = LogUniformPrior(0.5, 10.0), M_sec = LogUniformPrior(1e-3, 0.1),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi), inc = SinePrior(),
                    Omega = UniformPrior(0.0, 2pi)),),
    iad = _BH_IAD, iad_names = jitter ? ["G"] : String[], M_pri = 1.0,
    plx = NormalPrior(20.0, 0.1))

# --- A one-planet RV target, for a hold the caller chooses.
Random.seed!(5)
const _BH_N = 50
const _BH_T = sort!(300 .* rand(_BH_N))
const _BH_RV = 30.0 .* sin.(2π .* _BH_T ./ 5.7) .+ 2.0 .* randn(_BH_N)
_bh_rv_target() = build_target(
    planets = (b = (P = LogUniformPrior(5.0, 6.5), K = LogUniformPrior(5.0, 80.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _BH_T, rv = _BH_RV, rv_err = fill(2.0, _BH_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))

_cube(r) = r.chains.value.data
_ckstate(path) = open(deserialize, path).state
_col(r, nm) = vec(r.chains[Symbol(nm)].data)

@testset "burnin_hold" begin
    @testset "resolution" begin
        p = _bh_rv_target().params
        h = Nereus._burnin_hold(p, nothing, 0.5, 100)
        @test h.step == 0 && isempty(h.dims)          # no IAD jitter: nothing held
        @test Nereus._burnin_hold_fp(h, p) == (;)     # and the fingerprint unchanged
        h = Nereus._burnin_hold(p, Dict("sigma_SIM" => 2.0), 0.5, 100)
        @test h.step == 50 && p.layout.unfrozen_names[h.dims] == ["sigma_SIM"] &&
              h.values == [2.0]
        # JSON3 objects key by Symbol and deliver integers.
        @test Nereus._burnin_hold(p, Dict(:sigma_SIM => 2), 0.5, 100).values == [2.0]
        @test Nereus._burnin_hold(p, Dict("sigma_SIM" => 2.0), 0.0, 100).step == 0
        @test Nereus._burnin_hold(p, false, 0.5, 100).step == 0
        for bad in (Dict("sigma_X" => 1.0),       # not a parameter
                    Dict("sigma_SIM" => 20.0),    # outside its prior
                    Dict("Mo_k1" => 1.0),         # a full-circle angle: re-charted
                    "sigma_SIM")
            @test_throws ArgumentError Nereus._burnin_hold(p, bad, 0.5, 100)
        end
        @test_throws ArgumentError Nereus._burnin_hold(p, Dict("sigma_SIM" => 2.0), 1.5, 100)

        pj = _bh_iad_target().params
        @test Nereus.default_burnin_hold(pj) == Dict("sigma_iad_G" => 0.0)
        @test isempty(Nereus.default_burnin_hold(_bh_iad_target(jitter = false).params))
        # The node flip of this astrometry-only planet negates sesinw.
        tgi = _bh_iad_target()
        @test_throws ArgumentError sample_pt_emcee(tgi, tgi.data; n_temps = 2,
            n_walkers = 20, n_steps = 2, n_burnin = 2,
            burnin_hold = Dict("sesinw_k1" => 0.0), show_progress = false)
        # A trans-dim planet slot is not always there to hold.
        tg = _bh_rv_target()
        @test_throws ArgumentError sample_transdim_pt_emcee(tg, tg.data;
            td = TransDimConfig(max_kplanet = 1), n_temps = 2, n_walkers = 20,
            n_steps = 2, n_burnin = 2, burnin_hold = Dict("K_k1" => 10.0),
            show_progress = false)
    end

    @testset "pt_emcee: IAD jitter held at 0, then released" begin
        kw = (n_temps = 3, n_walkers = 30, n_burnin = 60, seed = 3,
              show_progress = false, bridge_headline = false)
        run(n; k...) = (tg = _bh_iad_target(); sample_pt_emcee(tg, tg.data; kw..., n_steps = n, k...))
        tg = _bh_iad_target()
        p = tg.params
        names = p.layout.unfrozen_names
        j = findfirst(==("sigma_iad_G"), names)
        knee = Nereus.log_scale_shift(p.layout.unfrozen_priors[j])

        full = run(120)
        path = joinpath(mktempdir(), "pt_emcee_state.jls")
        run(20; checkpoint = path)                    # inside the hold (30 steps)
        ck = _ckstate(path)
        @test all(==(log(knee)), ck.state[:, :, j])  # the walkers' flat-scale value
        # Each walker's cached log L is the likelihood with the jitter at 0
        # exactly, not at exp(log(knee)) - knee: evaluated as the sampler does.
        th = Theta{Float64}(p)
        ws = Nereus.PTWorkspace(p, p.config.max_kplanet, length(p.config.noise_models);
                                n_obs = length(tg.data.t_rv), n_phot = length(tg.data.t_phot))
        for (t, w) in ((1, 1), (2, 7), (3, 30))
            for (d, idx) in enumerate(p.layout.unfrozen_idx)
                s = Nereus.log_scale_shift(p.layout.unfrozen_priors[d])
                y = ck.state[t, w, d]
                th.values[idx] = d == j ? 0.0 : (s === nothing ? y : exp(y) - s)
            end
            @test ck.logL[t, w] == (Nereus.rv_log_likelihood(th, tg.data, ws) +
                                    Nereus.tomogram_log_likelihood(th, tg.data, ws)) +
                                   Nereus.transit_log_likelihood(th, tg.data, ws)
        end
        run(45; checkpoint = path, resume = true)     # across the release
        ext = run(120; checkpoint = path, resume = true)
        @test _cube(ext) == _cube(full)
        s = _col(full, "sigma_iad_G")
        @test length(unique(s)) > 100 && all(>=(0), s)

        # Off: the jitter moves from the start.
        run(20; checkpoint = path, burnin_hold = false)
        @test length(unique(_ckstate(path).state[:, :, j])) > 1
    end

    @testset "transdim_pt_emcee: a chosen hold, parked like an inactive slot" begin
        kw = (td = TransDimConfig(max_kplanet = 1), n_temps = 3, n_walkers = 20,
              n_burnin = 60, seed = 4, show_progress = false,
              burnin_hold = Dict("sigma_SIM" => 2.0))
        run(n; k...) = (reset_transdim_caches!(); tg = _bh_rv_target();
                        sample_transdim_pt_emcee(tg, tg.data; kw..., n_steps = n, k...))
        j = findfirst(==("sigma_SIM"), _bh_rv_target().params.layout.unfrozen_names)
        full = run(120)
        path = joinpath(mktempdir(), "transdim_state.jls")
        run(20; checkpoint = path)
        @test all(==(2.0), _ckstate(path).state[:, :, j])
        run(45; checkpoint = path, resume = true)
        ext = run(120; checkpoint = path, resume = true)
        @test _cube(ext) == _cube(full)
        @test length(unique(_col(full, "sigma_SIM"))) > 100
    end

    @testset "pt_whitening: the whitened swap leaves a held coordinate" begin
        # Whitened swaps start at step 10, inside the 30-step hold.
        kw = (n_temps = 3, n_walkers = 30, n_burnin = 60, seed = 5, warmup_swaps = 10,
              whiten_window = 10, whiten_refresh = 3, show_progress = false)
        run(n; k...) = (tg = _bh_iad_target(); sample_pt_whitening(tg, tg.data; kw..., n_steps = n, k...))
        j = findfirst(==("sigma_iad_G"), _bh_iad_target().params.layout.unfrozen_names)
        full = run(120)
        path = joinpath(mktempdir(), "pt_whitening_state.jls")
        run(25; checkpoint = path)
        @test all(==(0.0), _ckstate(path).state[:, :, j])
        run(45; checkpoint = path, resume = true)
        ext = run(120; checkpoint = path, resume = true)
        @test _cube(ext) == _cube(full)
        @test length(unique(_col(full, "sigma_iad_G"))) > 100
    end
end
