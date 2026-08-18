#!/usr/bin/env julia
# Trans-dimensional fit of GJ 876: does RJMCMC recover N_p ≥ 2?
#
# GJ 876 — 4 known planets (Rivera+ 2010, Millholland+ 2018):
#   b: P ≈ 61.1 d, K ≈ 214 m/s, e ≈ 0.032  (giant, 2:1 resonance with c)
#   c: P ≈ 30.1 d, K ≈ 88 m/s,  e ≈ 0.256
#   d: P ≈ 1.94 d, K ≈ 6.6 m/s, e ≈ 0.21
#   e: P ≈ 124 d,  K ≈ 3.5 m/s, e ≈ 0.055  (marginal detection)
#
# Data: 270 HARPS observations (pre + post fiber upgrade).
# Keplerian model (no N-body) — b+c resonance interaction ignored.
#
# Expected: N_p ≥ 2 strongly favored (b and c are loud).
# N_p = 3 likely (d at K ~ 6 m/s vs HARPS ~1 m/s precision).
# N_p = 4 marginal (e at K ~ 3.5 m/s).

using Nereus
using DelimitedFiles
using Statistics: median, mean, std
using Printf
using MCMCChains

# ---- Load data -------------------------------------------------------
datafile = joinpath(@__DIR__, "data", "gj876.csv")
raw = readdlm(datafile, ',', Any, '\n'; header=true)
data_mat = raw[1]

bjd      = Float64.(data_mat[:, 1])
rv       = Float64.(data_mat[:, 2])
rv_err   = Float64.(data_mat[:, 3])
inst_str = String.(data_mat[:, 4])

inst_names = sort!(unique(inst_str))
inst_map = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_map[s] for s in inst_str]

println("GJ 876 trans-dim: $(length(bjd)) observations")
for (i, name) in enumerate(inst_names)
    mask = rv_inst .== i
    println("  [$i] $name: n=$(sum(mask)), " *
            "rms=$(@sprintf("%.1f", std(rv[mask]))) m/s, " *
            "mean_err=$(@sprintf("%.2f", mean(rv_err[mask]))) m/s")
end

# ---- Build model -----------------------------------------------------
data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst)
ic = InstrumentConfig(rv=inst_names)

# max_kplanet=5 — room for the 4 real planets + 1 spurious to test rejection
params = Params(;
    max_kplanet = 5,
    planet_modes = fill(RV_ONLY, 5),
    instruments = ic,
    data = data,
    M_s = 0.334,    # Rivera+ 2010
)

target = NereusTarget(params, data; unconstrained=false)

println("\nModel: max_kplanet=5, $(n_unfrozen(params)) total params")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- Trans-dim config ------------------------------------------------
td = TransDimConfig(
    max_kplanet = 5,
    birth_strategies = [PriorBirth(), InformedBirth()],
    birth_weights = [0.3, 0.7],
    transdim_fraction = 0.3,
)

# ---- Run PT trans-dim ------------------------------------------------
println("\nStarting PT trans-dim: 15 rounds, 20 chains")
t0 = time()
chains, log_ev, pt_evals = sample_pt(target;
    td = td,
    n_rounds = 15,
    n_chains = 20,
    seed = 42,
    show_report = true,
)
dt = time() - t0
@printf("PT done in %.1f seconds (%d likelihood evals)\n", dt, pt_evals)

# ---- Save chains first (before anything that might crash) ------------
save_chains(joinpath("/private/tmp", "gj876_chains.nc"), chains, params; data=data)
println("Chains saved to /private/tmp/gj876_chains.nc")

# ---- Results ---------------------------------------------------------
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)

println("\n" * "="^60)
println("  GJ 876 — Trans-dimensional PT results")
println("="^60)

print_transdim_summary(chains, params; td=td, M_s=0.334)
save_transdim_summary(chains, params, outdir;
                       starname="GJ876", td=td, log_evidence=log_ev, M_s=0.334)

# ---- Plots -----------------------------------------------------------
plotdir = "/private/tmp/nereus_gj876"

println("\nGenerating plots...")
plot_rv_timeseries(chains, params, data; output=plotdir)
for k in 1:min(4, params.config.max_kplanet)
    plot_rv_phasefold(chains, params, data; planet=k, output=plotdir)
end
plot_histograms(chains, params; output=plotdir)

println("Plots saved to $plotdir/")
println("\n" * "="^60)
