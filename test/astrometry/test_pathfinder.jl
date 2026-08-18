# Smoke test for Pathfinder warmup wrapper.

using Nereus, Statistics, Printf

# Tiny synthetic 1-planet RV-only fit
t_rv = collect(0.0:1.0:50.0)
P_true, K_true = 5.0, 30.0
rv = K_true .* cos.(2π .* t_rv ./ P_true) .+ randn(length(t_rv))
data = Data(t_rv=t_rv, rv=rv, rv_err=fill(1.0, length(t_rv)))
params = Params(
    max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["X"]),
    data = data, stability = :none,
    priors = Dict{String, PriorSpec}(
        "n_p"        => FixedPrior(1.0),
        "P_k1"       => LogUniformPrior(1.0, 50.0),
        "K_k1"       => LogUniformPrior(1.0, 100.0),
        "sesinw_k1"  => UniformPrior(-1.0, 1.0),
        "secosw_k1"  => UniformPrior(-1.0, 1.0),
        "Mo_k1"      => UniformPrior(0.0, 2π),
    ),
)
target = NereusTarget(params, data)

println("Running Pathfinder (4 runs, 50 draws)...")
result = pathfinder_init(target; n_runs=4, n_draws=50, seed=42)
@printf("Pathfinder draws: size %s\n", size(result.draws))
@printf("Pareto shape (k̂): %.3f  (good if < 0.5, bad if > 0.7)\n",
        result.psis_result.pareto_shape)
@printf("Fit distribution: %s\n", typeof(result.fit_distribution).name.wrapper)
# MultiPathfinder returns a MixtureModel of MvNormals
println("Mean of empirical draws (Nereus bounded space requires inverse-transform):")
draws = result.draws
println("  draws mean (unconstrained): ", round.(mean(draws, dims=2)[:], digits=3))
println("  draws std  (unconstrained): ", round.(std(draws, dims=2)[:], digits=3))
