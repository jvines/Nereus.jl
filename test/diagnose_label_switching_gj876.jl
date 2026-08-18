#!/usr/bin/env julia
# Real-chain validation of the label_switching diagnostic.
#
# GJ 876 is the right target for it: four known planets spanning the whole
# confidence gradient the diagnostic is about —
#   b: P ≈ 61.1 d, K ≈ 214 m/s   (dominant)
#   c: P ≈ 30.1 d, K ≈ 88  m/s   (dominant)
#   d: P ≈ 1.94 d, K ≈ 6.6 m/s   (middle)
#   e: P ≈ 124 d,  K ≈ 3.5 m/s   (marginal)
# — run trans-dim at max_kplanet=5, so there is a spare slot for the
# permutation degeneracy to actually use.
#
# The prediction under test (paper/second_quantization_exchangeable_slots.md):
# switching needs a component PRESENT but NOT DOMINANT, so it should land on
# the middle-confidence planets and NOT on b/c (monopolised) or e (absent
# most of the time). This script measures it instead of asserting it.

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

# Julia block-buffers stdout when it is redirected to a file, so progress
# markers go to stderr (unbuffered) and stdout is flushed explicitly.
# n_rounds / n_chains from ARGS so a cheap smoke run and the real run share
# one script.
# NB rounds are a DOUBLING schedule: total draws = 2(2^n_rounds − 1), so
# 12 rounds = 8190 draws and 15 rounds = 65534 (8× the work). Entropy is
# already stable at a few thousand occupied draws (Miller-Madow bias
# < 0.002 bit), so 12 is the sweet spot here.
#
# n_chains must stay ≤ 21: PTWorkspace preallocates the donor-population
# buffer at a hardcoded 20 (rjmcmc.jl:420) while pt.jl:659 builds an
# unclamped `view(ws.population, 1:ws.n_pop)`, so ≥22 chains BoundsErrors
# once enough replicas hold an active planet.
const N_ROUNDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
const N_CHAINS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
_progress(msg) = (println(stderr, "[ls] ", msg); flush(stderr))

_progress("threads = $(Threads.nthreads()), rounds = $N_ROUNDS, chains = $N_CHAINS")
println("GJ 876 label-switching validation: $(length(bjd)) observations")

data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst)
ic = InstrumentConfig(rv=inst_names)
params = Params(;
    max_kplanet = 5,
    planet_modes = fill(RV_ONLY, 5),
    instruments = ic,
    data = data,
    M_s = 0.334,    # Rivera+ 2010
)
target = NereusTarget(params, data; unconstrained=false)

td = TransDimConfig(
    max_kplanet = 5,
    birth_strategies = [PriorBirth(), InformedBirth()],
    birth_weights = [0.3, 0.7],
    transdim_fraction = 0.3,
)

_progress("starting PT trans-dim (max_kplanet=5) …")
t0 = time()
chains, log_ev, pt_evals = sample_pt(target;
    td = td,
    n_rounds = N_ROUNDS,
    n_chains = N_CHAINS,
    seed = 42,
    show_report = true,
)
@printf("PT done in %.1f s (%d likelihood evals), logZ = %.2f\n",
        time() - t0, pt_evals, log_ev)
_progress(@sprintf("PT done in %.1f s, logZ = %.2f — running diagnostic",
                   time() - t0, log_ev))

save_chains("/private/tmp/gj876_labelswitch_chains.nc", chains, params; data=data)

# ---- The diagnostic --------------------------------------------------
println("\n" * "="^70)
println("  BLIND MODE — components discovered from the chain itself")
println("="^70)
rep_blind = label_switching(chains)
print_label_switching(rep_blind)

println("\n" * "="^70)
println("  REFERENCE MODE — Rivera+ 2010 / Millholland+ 2018 orbits")
println("="^70)
rep_ref = label_switching(chains;
    reference_periods = [1.938, 30.088, 61.117, 124.26],
    labels            = ["d", "c", "b", "e"])
print_label_switching(rep_ref)

# ---- The prediction under test ---------------------------------------
println("\n" * "="^70)
println("  PREDICTION: switching lives on MIDDLE-confidence components")
println("="^70)
@printf("  %-8s %10s %12s %10s\n", "planet", "P(occ)", "H [bit]", "dup")
for c in sort(rep_ref.components; by = x -> x.entropy_bits)
    @printf("  %-8s %10.2f %12.2f %10.2f\n",
            c.label, c.p_occupied, c.entropy_bits, c.duplicate_frac)
end
worst = argmax(c -> c.entropy_bits, rep_ref.components)
@printf("\n  highest-entropy component: %s (P(occ)=%.2f, H=%.2f bit)\n",
        worst.label, worst.p_occupied, worst.entropy_bits)
println("  Prediction holds iff that is a present-but-not-dominant planet,")
println("  i.e. NOT b/c (P(occ)≈1 and monopolised) and NOT a near-absent one.")
println("\n  status = ", rep_ref.status, "  (blind: ", rep_blind.status, ")")
flush(stdout)
_progress("DONE — worst = $(worst.label), H = $(round(worst.entropy_bits; digits=2)) bit")
