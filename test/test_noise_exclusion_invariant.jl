# No trans-dim move may leave two members of a noise exclusion group active.
#
# `default_noise_menu` puts ErrorScale, CeleriteRotation, ActivityDecorrelation
# and ActivityGP in ONE exclusion group: they are competing descriptions of the
# same excess scatter, and composing two of them double-models it. A real
# HD 18599 run (nereus v0.9.8, transdim_pt_emcee, planets + noise) spent a third
# of its cold draws in CeleriteRotation + AD. The leak was the planet<->AD swap:
# its planet->AD direction activated AD after checking only that AD itself was
# off, never whether another member of AD's group was on. Every other noise
# activation (birth, within-group swap, init) checked the groups.
using Test
using Random
using Serialization
using Nereus
using Nereus: Params, Data, InstrumentConfig, NereusTarget, TransDimConfig,
              NoiseModel, PlanetDataSources, ActivityDecorrelation, CeleriteRotation,
              ErrorScale, ActivityGP, default_noise_menu, propose_planet_ad_swap

# Largest number of simultaneously active members over the exclusion groups,
# for one active mask indexed like `noise_models`.
function _max_group_active(mask, groups, noise_models)
    worst = 0
    for g in groups
        n = 0
        for m in g
            j = findfirst(==(m), noise_models)
            (j === nothing || j > length(mask)) && continue
            n += mask[j] ? 1 : 0
        end
        worst = max(worst, n)
    end
    return worst
end

@testset "noise exclusion groups hold under every trans-dim move" begin
    # RV = a rotation signal that the indicators track, so a planet at P_rot and
    # AD are interchangeable explanations: the planet<->AD swap is accepted in
    # both directions, which is what exposes a leak in it.
    P_ROT, N, SIG = 9.0, 70, 1.0
    rng = MersenneTwister(20261124)
    t = sort(rand(rng, N)) .* 90.0
    act = 6.0 .* sin.(2π .* t ./ P_ROT)
    rv = act .+ SIG .* randn(rng, N)
    bis  = 0.5 .* act .+ 0.3 .* randn(rng, N)
    fwhm = 0.8 .* act .+ 0.3 .* randn(rng, N)
    data = Data(t_rv = t, rv = rv, rv_err = fill(SIG, N), rv_inst = ones(Int, N),
                indicators = Dict("bis" => bis, "fwhm" => fwhm),
                indicator_errs = Dict("bis" => fill(0.3, N), "fwhm" => fill(0.3, N)))

    menu = default_noise_menu(data)
    nms = menu.noise_models
    groups = menu.exclusion_groups
    rvmax = maximum(abs, rv)
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
        instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
        priors = Dict{String,PriorSpec}(
            "P_k1" => UniformPrior(2.0, 40.0),
            "gamma_I1" => UniformPrior(-3rvmax, 3rvmax),
            "sigma_I1" => LogUniformPrior(0.05, 12.0)),
        noise_models = nms, transdim_noise = true, stability = :none)

    idx(T) = findfirst(m -> m isa T, nms)
    i_es, i_rot, i_ad, i_agp = idx(ErrorScale), idx(CeleriteRotation),
                               idx(ActivityDecorrelation), idx(ActivityGP)
    @test all(!isnothing, (i_es, i_rot, i_ad, i_agp))
    # the four share one group (the menu's contract, which this test relies on)
    @test any(g -> all(i -> nms[i] in g, (i_es, i_rot, i_ad, i_agp)), groups)

    @testset "planet->AD swap never activates AD beside a group-mate" begin
        for other in (i_es, i_rot, i_agp)
            th = Nereus.Theta{Float64}(params)
            th.td = Nereus.TransDimState(max_planets = 1, n_noise = length(nms))
            Nereus.activate_planet!(th.td, 1)
            th.values[params.layout.name_to_idx["P_k1"]] = P_ROT
            Nereus.activate_noise!(th.td, other)
            # always-on members (IndicatorFloor) stay on, as in the sampler
            for j in eachindex(nms)
                nms[j] in menu.toggleable || Nereus.activate_noise!(th.td, j)
            end
            for trial in 1:40
                cand, lq = propose_planet_ad_swap(th, MersenneTwister(trial),
                                                  menu.toggleable; data = data,
                                                  exclusion_groups = groups)
                # with a group-mate on, neither direction is legal
                @test !isfinite(lq)
                @test _max_group_active(cand.td.noise_active, groups, nms) <= 1
            end
        end
    end

    @testset "planet->AD swap still fires when the group is empty" begin
        # guards against the fix disabling the move outright
        th = Nereus.Theta{Float64}(params)
        th.td = Nereus.TransDimState(max_planets = 1, n_noise = length(nms))
        Nereus.activate_planet!(th.td, 1)
        th.values[params.layout.name_to_idx["P_k1"]] = P_ROT
        for j in eachindex(nms)
            nms[j] in menu.toggleable || Nereus.activate_noise!(th.td, j)
        end
        n_ok = 0
        for trial in 1:40
            cand, lq = propose_planet_ad_swap(th, MersenneTwister(trial),
                                              menu.toggleable; data = data,
                                              exclusion_groups = groups)
            isfinite(lq) || continue
            n_ok += 1
            @test Nereus.is_noise_active(cand.td, i_ad)
            @test cand.td.n_planets_active == 0
            @test _max_group_active(cand.td.noise_active, groups, nms) <= 1
        end
        @test n_ok > 0
    end

    @testset "transdim_pt_emcee: no recorded draw or walker breaks a group" begin
        tgt = NereusTarget(params, data)
        td = TransDimConfig(; max_kplanet = 1, planets = true, noise = true,
                              toggleable = menu.toggleable,
                              noise_exclusion_groups = groups)
        ck = joinpath(mktempdir(), "ck.jls")
        r = Nereus.sample_transdim_pt_emcee(tgt, data; td = td, n_temps = 4,
                n_walkers = 16, n_steps = 500, n_burnin = 200, seed = 5,
                checkpoint = ck, show_progress = false)
        ch = r.chains; cn = names(ch, :parameters)
        cols = [Symbol("noise_active_$j") for j in eachindex(nms)]
        @test all(c -> c in cn, cols[1:length(menu.toggleable)])
        draws = [c in cn ? vec(Array(ch[c])) .> 0.5 : fill(true, size(ch, 1))
                 for c in cols]      # an always-on member has no column: on
        nbad = count(i -> _max_group_active([d[i] for d in draws], groups, nms) > 1,
                     1:size(ch, 1))
        @test nbad == 0
        # every walker at every temperature at the end of the run
        st = open(deserialize, ck).state
        @test all(tds -> _max_group_active(tds.noise_active, groups, nms) <= 1,
                  st.td_states)
    end
end
