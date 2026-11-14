# The within-model moves that rjmcmc, moms and pt share (src/samplers/rjmcmc.jl):
# the step scale `_prior_widths` takes from each prior, and the scale of the
# current density moms hands them.
#
# `_prior_widths` is the slice sampler's stepping-out width, fixed for the run,
# and the RWM's starting σ. For a ModJeffreys it used to be a tenth of
# sqrt(max(lo, 1e-10) * hi), with no knee: 1.4e-5 for the default K prior
# ModJeffreys(100, 200), so K stayed at its initial draw. moms handed the
# shared moves its spike-and-slab log prior, which they compared with the
# plain `log_prior` of their proposals.
using Test
using Nereus
using Random
using Statistics: std
using Nereus: ActivityDecorrelation, Data, InstrumentConfig, NereusTarget, NoiseModel,
              Params, PlanetDataSources, PriorSpec

Random.seed!(23)
const _WM_N = 40
const _WM_T = sort!(60 .* rand(_WM_N))
const _WM_ERR = 1.5
const _WM_RV = 5.0 .* sin.(2π .* _WM_T ./ 7.3) .+ _WM_ERR .* randn(_WM_N)

# One circular planet whose K carries the default-shaped ModJeffreys prior.
_wm_target() = build_target(;
    planets = (b = (P = UniformPrior(7.2, 7.4), K = ModJeffreysPrior(100.0, 200.0),
                    sesinw = FixedPrior(0.0), secosw = FixedPrior(0.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _WM_T, rv = _WM_RV, rv_err = fill(_WM_ERR, _WM_N)),
                 sigma = LogUniformPrior(0.1, 10.0)),))

_wm_col(ch, nm) = vec(ch[Symbol(nm)].data)

@testset "within-model step scales" begin
    tg = _wm_target()
    L = tg.params.layout
    w = Nereus._prior_widths(L)
    iK = findfirst(==("K_k1"), L.unfrozen_names)
    iσ = findfirst(n -> startswith(n, "sigma"), L.unfrozen_names)
    # ModJeffreys(knee, upper) is flat in log(K + knee): a tenth of the geometric
    # mean of knee and upper + knee. A LogUniform (no shift) keeps its value.
    @test w[iK] ≈ 0.1 * sqrt(100.0 * 300.0)
    @test w[iσ] ≈ 0.1 * sqrt(0.1 * 10.0)

    # K moves: its posterior sd is about σ √(2/N) = 0.34 m/s. With the old
    # width it was ~1e-3 (every step 1.4e-5).
    sd_fisher = _WM_ERR * sqrt(2 / _WM_N)
    td = TransDimConfig(max_kplanet = 1, planets = false)
    kw = (; td, n_samples = 1000, n_warmup = 500, seed = 7, show_progress = false)
    for (nm, run) in (("rjmcmc", () -> sample_rjmcmc(_wm_target(), _wm_target().data; kw...)[1]),
                      ("moms", () -> sample_moms(_wm_target(), _wm_target().data; kw...)[1]))
        sdK = std(_wm_col(run(), "K_k1"))
        @test 0.6 < sdK / sd_fisher < 1.6
    end
end

@testset "moms: the shared moves get log_prior, not the spike-and-slab prior" begin
    # With the planet set fixed no move changes a γ_k, so the inclusion prior
    # adds the same constant to every state and must change nothing: the slice,
    # RWM and noise moves see the same density whatever it is. Handed the
    # spike-and-slab value, they judged their proposals against a current
    # density shifted by log p_inc, and the chains differed.
    rng = MersenneTwister(20261111)
    n = 60
    t = sort(rand(rng, n)) .* 70.0
    act = sin.(2π .* t ./ 8.0)
    rv = act .+ 4.0 .* sin.(2π .* t ./ 7.3) .+ 1.5 .* randn(rng, n)
    data = Data(t_rv = t, rv = rv, rv_err = fill(1.5, n), rv_inst = ones(Int, n),
                indicators = Dict("bis" => act .+ 2.0 .* randn(rng, n),
                                  "bis2" => act .+ 2.0 .* randn(rng, n)),
                indicator_errs = Dict("bis" => fill(1.0, n), "bis2" => fill(1.0, n)))
    ad, rot = (ActivityDecorrelation(indicators = [s]) for s in ("bis", "bis2"))
    td = TransDimConfig(; max_kplanet = 1, planets = false, noise = true,
                          toggleable = NoiseModel[ad, rot],
                          noise_exclusion_groups = [NoiseModel[ad, rot]],
                          transdim_fraction = 0.3)
    target() = NereusTarget(Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
            instruments = InstrumentConfig(rv = ["I1"]), data = data,
            priors = Dict{String,PriorSpec}("P_k1" => UniformPrior(7.2, 7.4),
                                            "gamma_I1" => UniformPrior(-20.0, 20.0),
                                            "sigma_I1" => LogUniformPrior(0.2, 12.0)),
            noise_models = NoiseModel[ad, rot], transdim_noise = true,
            stability = :none), data)
    run(p_inc; kw...) = sample_moms(target(), data; td, n_warmup = 100, n_samples = 200,
                                    seed = 3, inclusion_prior = p_inc,
                                    show_progress = false, kw...)[1].value.data
    for wm in (:slice, :rwm)
        a = run(0.5; within_model = wm)
        @test any(c -> 0 < sum(c) < length(c), eachcol(a[:, end-1:end, 1]))  # toggles moved
        @test run(0.9; within_model = wm) == a
    end
end
