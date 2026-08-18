#!/usr/bin/env julia
# Speed benchmark: trans-dim PT on 51 Peg with real sample counts.

using Nereus
using DelimitedFiles
using Random: randperm
using Statistics: median, mean, std, quantile
using Printf
using MCMCChains

# ---- Load data -------------------------------------------------------
datafile = joinpath(@__DIR__, "data", "51peg.csv")
raw = readdlm(datafile, ',', Any, '\n'; header=true)
data_mat = raw[1]

bjd      = Float64.(data_mat[:, 1])
rv       = Float64.(data_mat[:, 2])
rv_err   = Float64.(data_mat[:, 3])
inst_str = String.(data_mat[:, 4])

inst_names = sort!(unique(inst_str))
inst_map = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_map[s] for s in inst_str]

# Full data — no subsampling
println("51 Peg: $(length(bjd)) observations (full dataset)")
for (i, name) in enumerate(inst_names)
    mask = rv_inst .== i
    println("  [$i] $name: n=$(sum(mask))")
end

data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst)
ic = InstrumentConfig(rv=inst_names)

# ---- max_kplanet = 2 -------------------------------------------------
params = Params(;
    max_kplanet = 2,
    planet_modes = [RV_ONLY, RV_ONLY],
    instruments = ic,
    data = data,
)
target = NereusTarget(params, data; unconstrained=false)

td = TransDimConfig(
    max_kplanet = 2,
    birth_strategies = [PriorBirth(), InformedBirth()],
    birth_weights = [0.3, 0.7],
    transdim_fraction = 0.3,
)

println("\nModel: max_kplanet=2, $(n_unfrozen(params)) free params, $(length(bjd)) obs")

# ---- Benchmark 1: RJMCMC --------------------------------------------
for (n_s, n_w) in [(10_000, 5_000), (50_000, 10_000), (100_000, 20_000)]
    println("\n--- RJMCMC: $(n_w) warmup + $(n_s) samples ---")
    t0 = time()
    chains = sample_rjmcmc(target, data;
        td = td, n_samples = n_s, n_warmup = n_w, seed = 42)
    dt = time() - t0
    np = vec(Array(chains[:n_planets]))
    n0 = count(np .== 0); n1 = count(np .== 1); n2 = count(np .== 2)
    @printf("  Time: %.1f s  (%.0f samples/sec)\n", dt, n_s / dt)
    @printf("  N_p: 0=%d (%.1f%%)  1=%d (%.1f%%)  2=%d (%.1f%%)\n",
            n0, 100*n0/n_s, n1, 100*n1/n_s, n2, 100*n2/n_s)
end

# ---- Benchmark 2: PT trans-dim --------------------------------------
for (n_r, n_c) in [(10, 6), (12, 8), (14, 10)]
    n_expected = n_c > 0 ? 2^n_r : 0  # rough estimate
    println("\n--- PT trans-dim: $(n_r) rounds, $(n_c) chains (~$(n_expected) samples) ---")
    t0 = time()
    chains_pt, log_ev = sample_pt(target;
        td = td, n_rounds = n_r, n_chains = n_c, seed = 42,
        show_report = false)
    dt = time() - t0
    np = vec(Array(chains_pt[:n_planets]))
    n_total = length(np)
    n0 = count(np .== 0); n1 = count(np .== 1); n2 = count(np .== 2)
    @printf("  Time: %.1f s  (%d samples, %.0f samples/sec)\n", dt, n_total, n_total / dt)
    @printf("  N_p: 0=%d (%.1f%%)  1=%d (%.1f%%)  2=%d (%.1f%%)\n",
            n0, 100*n0/n_total, n1, 100*n1/n_total, n2, 100*n2/n_total)
    @printf("  log Z ≈ %.2f\n", log_ev)
end

println("\n" * "="^60)
println("Done.")
