#!/usr/bin/env julia
# Functional check for the full-size noise-mask fix (commit 2d1b260).
#
# Config under test: noise_models = [MAModel (always-on), ARModel
# (toggleable)]. Pre-fix, rjmcmc/pt/moms/moms_ns sized the walker noise
# mask to length(toggleable)=1 while every consumer indexes by position in
# config.noise_models — so the always-on MA at config index 1 read
# noise_active[1]=false: silently INACTIVE, its hyperparams skipped by
# within-model moves (frozen spectators) and its prior/likelihood terms
# dropped. Post-fix the mask is full-size with non-toggleables forced
# active, so the MA params must MOVE in the chain.
#
# Also checks moms honors planets=false at init (pre-fix a fixed-planet
# noise-only run started — and stayed — with ZERO active planets).

using Nereus
using MCMCChains
using Statistics: std, mean
using Random

rng = MersenneTwister(7)
n   = 80
t   = sort(rand(rng, n)) .* 120.0
K, P = 12.0, 9.3
rv  = K .* sin.(2π .* t ./ P) .+ 2.5 .* randn(rng, n)
data = Data(t_rv = t, rv = rv, rv_err = fill(2.0, n), rv_inst = fill(1, n))

ma = MAModel(order = 1)      # always-on (NOT in toggleable)
ar = ARModel(order = 1)      # toggleable

priors = Dict{String, PriorSpec}(
    "n_p"       => FixedPrior(1.0),
    "P_k1"      => LogUniformPrior(5.0, 20.0),
    "K_k1"      => UniformPrior(0.0, 30.0),
    "sesinw_k1" => UniformPrior(-1.0, 1.0),
    "secosw_k1" => UniformPrior(-1.0, 1.0),
    "Mo_k1"     => UniformPrior(0.0, 2π),
    "gamma_I1"  => UniformPrior(-20.0, 20.0),
    "sigma_I1"  => LogUniformPrior(0.1, 10.0),
)
params = Params(;
    max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["I1"]),
    data = data, M_s = 1.0,
    parametrization = ParametrizationConfig(time = :Mo),
    priors = priors,
    noise_models = [ma, ar],
    transdim_noise = true,
)
target = NereusTarget(params, data)

td = TransDimConfig(;
    max_kplanet = 1,
    planets     = false,
    noise       = true,
    toggleable  = NoiseModel[ar],
)

failures = String[]

# ---- rjmcmc: always-on MA params must not be frozen spectators --------
chains, _ = sample_rjmcmc(target, data; td = td, n_samples = 4000,
                           n_warmup = 1500, seed = 11, show_progress = false)
for nm in (:ma_omega_1, :ma_beta_1)
    v = vec(Array(chains[nm]))
    s = std(v)
    ok = s > 1e-8
    println("rjmcmc $nm: std = $s  ", ok ? "MOVES ✓" : "FROZEN ✗")
    ok || push!(failures, "rjmcmc $nm frozen (always-on MA inactive)")
end
np_rj = vec(Array(chains[:n_planets]))
println("rjmcmc planets=false: mean n_planets = ", mean(np_rj),
        all(np_rj .== 1.0) ? "  ✓ fixed at 1" : "  ✗ NOT fixed")
all(np_rj .== 1.0) || push!(failures, "rjmcmc planets=false not honored")

# ---- moms: planets=false must start (and stay) all-active -------------
chm, _, _ = sample_moms(target, data; td = td, n_samples = 3000,
                         n_warmup = 1200, seed = 12, show_progress = false)
np = vec(Array(chm[:n_planets]))
println("moms planets=false: mean n_planets = ", mean(np),
        all(np .== 1.0) ? "  ✓ fixed at 1" : "  ✗ NOT fixed")
all(np .== 1.0) || push!(failures, "moms planets=false not honored")
for nm in (:ma_omega_1, :ma_beta_1)
    v = vec(Array(chm[nm]))
    s = std(v)
    ok = s > 1e-8
    println("moms $nm: std = $s  ", ok ? "MOVES ✓" : "FROZEN ✗")
    ok || push!(failures, "moms $nm frozen (always-on MA inactive)")
end

if isempty(failures)
    println("\n✅ noise-mask fix VERIFIED (rjmcmc + moms)")
else
    println("\n❌ FAILURES:")
    foreach(f -> println("  - ", f), failures)
    exit(1)
end
