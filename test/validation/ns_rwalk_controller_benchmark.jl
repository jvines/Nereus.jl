# Does the rwalk controller fix change FITTING OUTCOMES, not just kernel
# displacement? Inject a known 1-planet RV signal (P=8.0 d, K=12 m/s) and run
# sample_nested_ins with proposal=:rwalk across seeds. Report recovered period
# error and both log Z estimators. Run this against the fixed and the reverted
# controller and compare.
using Nereus, Random, Statistics, Printf, MCMCChains

const P_TRUE = 8.0
const K_TRUE = 12.0

function make_target(seed::Int; n_obs::Int = 30)
    rng    = MersenneTwister(seed)
    t      = sort(rand(rng, n_obs) .* 40.0)
    rv     = K_TRUE .* sin.(2π .* t ./ P_TRUE) .+ randn(rng, n_obs)
    data   = Data(; t_rv = t, rv = rv, rv_err = fill(1.0, n_obs))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                     instruments = InstrumentConfig(rv = ["SIM"]),
                     data = data, M_s = 1.0)
    return NereusTarget(params, data; unconstrained = false), data
end

pname(ch) = first(filter(n -> occursin("P", string(n)) &&
                              !occursin("Tp", string(n)), names(ch)))

label = length(ARGS) >= 1 ? ARGS[1] : "run"
Perrs, zns, zins, nevals = Float64[], Float64[], Float64[], Int[]

for seed in 1:8
    target, data = make_target(2000 + seed)
    res = sample_nested_ins(target, data; proposal = :rwalk, n_live = 100,
                             dlogz = 0.5, max_iter = 20000, seed = seed)
    P̂ = median(vec(Array(res.chains[pname(res.chains)])))
    push!(Perrs, abs(P̂ - P_TRUE))
    push!(zns,  res.log_z_ns)
    push!(zins, res.log_z_ins)
    push!(nevals, res.n_evals)
    @printf("  seed %2d   P̂=%8.4f  |ΔP|=%7.4f   logZ_ns=%9.3f  logZ_ins=%9.3f\n",
            seed, P̂, abs(P̂ - P_TRUE), res.log_z_ns, res.log_z_ins)
end

@printf("\n[%s]  median |ΔP| = %.4f d   |  frac within 1%% of truth = %.2f\n",
        label, median(Perrs), count(e -> e < 0.08, Perrs) / length(Perrs))
@printf("[%s]  median logZ_ns = %.3f   median logZ_ins = %.3f   |ns-ins| = %.3f\n",
        label, median(zns), median(zins), abs(median(zns) - median(zins)))
@printf("[%s]  median n_evals = %d\n", label, round(Int, median(nevals)))
