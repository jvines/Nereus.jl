#!/usr/bin/env julia
# HD 18599 — post-hoc analysis of trans-dim chains.
#
# Loads `results/HD18599_RVPM_transdim/chains.nc` and reports:
#   1. Joint (N_p, noise-config) posterior, with each config labelled
#      against the matching Vines+ 2023 Run number (Table 8).
#   2. K_k1 conditional on each noise config — compared to the paper's
#      reported semi-amplitudes (Run 2 = 11±3 m/s, Run 1 ≈ 5.5 m/s).
#
# Toggleable noise order in the trans-dim run (see fit_HD18599_RVPM_transdim.jl):
#   noise_active_1 → AR(1)
#   noise_active_2 → MA(1)
#   noise_active_3 → ActivityDecorrelation(BIS)
#
# Vines+ 2023 Run map (Table 8): (AR, MA, Act) →
#   Run 1 = (1, 1, 0)   AR(1)MA(1)
#   Run 2 = (0, 0, 1)   activities       ← adopted, K = 11±3
#   Run 3 = (1, 1, 1)   AR(1)MA(1) + activities
#   Run 4 = (1, 0, 0)   AR(1)MA(0)
#   Run 5 = (0, 1, 0)   AR(0)MA(1)
#   Run 6 = (0, 0, 0)   pure Keplerian

using Nereus
using Statistics: median, std, quantile
using Printf
using MCMCChains

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const CHAIN_PATH = joinpath(REPO_ROOT, "Nereus.jl", "results",
                             "HD18599_RVPM_transdim", "chains.nc")

isfile(CHAIN_PATH) || error("No chains at $CHAIN_PATH — run the fit first.")

println("Loading chains from $CHAIN_PATH ...")
chains, _ = load_chains(CHAIN_PATH)

# Noise config: (AR, MA, Act) → run name + paper-K expectation
const RUN_MAP = Dict{Tuple{Int,Int,Int}, NamedTuple}(
    (1,1,0) => (run = "Run 1", desc = "AR(1)MA(1)",        K_paper = "≈ 5.5 m/s (½ × Run 2)"),
    (0,0,1) => (run = "Run 2", desc = "activities",         K_paper = "11 ± 3 m/s (ADOPTED)"),
    (1,1,1) => (run = "Run 3", desc = "AR(1)MA(1) + act",   K_paper = "≈ 11 m/s"),
    (1,0,0) => (run = "Run 4", desc = "AR(1)MA(0)",         K_paper = "≈ 11 m/s"),
    (0,1,0) => (run = "Run 5", desc = "AR(0)MA(1)",         K_paper = "≈ 11 m/s"),
    (0,0,0) => (run = "Run 6", desc = "pure Keplerian",     K_paper = "≈ 11 m/s"),
)

function classify(ar, ma, act)
    key = (Int(ar), Int(ma), Int(act))
    return get(RUN_MAP, key, (run = "—", desc = "(unmapped)", K_paper = "—"))
end

# Pull indicator columns
np    = vec(Array(chains[:, :n_planets, :]))
ar    = vec(Array(chains[:, :noise_active_1, :]))
ma    = vec(Array(chains[:, :noise_active_2, :]))
act   = vec(Array(chains[:, :noise_active_3, :]))
K     = vec(Array(chains[:, :K_k1, :]))

n_total = length(np)
@printf("\nLoaded %d posterior samples (%d chains × %d per chain)\n",
         n_total, size(chains, 3), size(chains, 1))
@printf("Modal N_p = %d   (P(N_p=0) = %.3f, P(N_p=1) = %.3f, P(N_p=2) = %.3f)\n",
         findmax([count(==(k), round.(Int, np)) for k in 0:2])[2] - 1,
         count(==(0), round.(Int, np)) / n_total,
         count(==(1), round.(Int, np)) / n_total,
         count(==(2), round.(Int, np)) / n_total)

println("\n" * "="^85)
println("  Joint (N_p, AR, MA, Act) posterior — paper-Run-mapped")
println("="^85)
@printf("  %-6s  %-3s %-3s %-3s  %-22s  %10s  %20s\n",
         "N_p", "AR", "MA", "Act", "Vines+ 2023 mapping", "P", "K_k1 paper expects")
println("  " * "-"^85)

# Joint key: (np, ar, ma, act)
joint = Dict{NTuple{4,Int}, Vector{Int}}()
for i in 1:n_total
    key = (round(Int, np[i]),
           round(Int, ar[i]),
           round(Int, ma[i]),
           round(Int, act[i]))
    push!(get!(joint, key, Int[]), i)
end
sorted = sort(collect(joint); by = x -> -length(x[2]))
for ((np_v, ar_v, ma_v, act_v), idxs) in sorted
    info = classify(ar_v, ma_v, act_v)
    @printf("  %-6d  %-3d %-3d %-3d  %-22s  %10.4f  %20s\n",
             np_v, ar_v, ma_v, act_v,
             "$(info.run): $(info.desc)",
             length(idxs) / n_total,
             info.K_paper)
end
println("  " * "-"^85)

# K_k1 conditional on each noise config (np = modal_np = 1 expected)
modal_np = 1
println("\n" * "="^85)
println("  K_k1 conditional on noise config (N_p = $modal_np only) — paper bar: 11 ± 3 m/s")
println("="^85)
@printf("  %-3s %-3s %-3s  %-22s  %8s  %22s  %14s\n",
         "AR", "MA", "Act", "Vines+ 2023 mapping",
         "n_samp", "K_med +upper -lower", "68% CI")
println("  " * "-"^85)

# Group by noise config among samples with N_p = modal_np. Report
# asymmetric 68% credible intervals (median + upper − lower) — K
# posteriors are skewed and median±std is misleading.
config_K = Dict{NTuple{3,Int}, Vector{Float64}}()
for i in 1:n_total
    round(Int, np[i]) == modal_np || continue
    key = (round(Int, ar[i]), round(Int, ma[i]), round(Int, act[i]))
    push!(get!(config_K, key, Float64[]), K[i])
end
config_sorted = sort(collect(config_K); by = x -> -length(x[2]))
for ((ar_v, ma_v, act_v), Ks) in config_sorted
    info = classify(ar_v, ma_v, act_v)
    K_med = median(Ks)
    K_lo  = quantile(Ks, 0.16)
    K_hi  = quantile(Ks, 0.84)
    @printf("  %-3d %-3d %-3d  %-22s  %8d  %7.2f +%5.2f -%5.2f  [%5.2f, %5.2f]\n",
             ar_v, ma_v, act_v,
             "$(info.run): $(info.desc)",
             length(Ks), K_med, K_hi - K_med, K_med - K_lo, K_lo, K_hi)
end
println("  " * "-"^85)

# Headline numbers vs paper bar
println("\n" * "="^85)
println("  Headline check vs Vines+ 2023")
println("="^85)
run2_idx = filter(i -> round(Int, np[i]) == modal_np &&
                       round(Int, ar[i]) == 0 &&
                       round(Int, ma[i]) == 0 &&
                       round(Int, act[i]) == 1, 1:n_total)
run1_idx = filter(i -> round(Int, np[i]) == modal_np &&
                       round(Int, ar[i]) == 1 &&
                       round(Int, ma[i]) == 1 &&
                       round(Int, act[i]) == 0, 1:n_total)
modal_cfg = config_sorted[1][1]
modal_info = classify(modal_cfg[1], modal_cfg[2], modal_cfg[3])

@printf("  Modal noise config in trans-dim posterior: %s — %s (P=%.3f)\n",
         modal_info.run, modal_info.desc,
         length(config_sorted[1][2]) / count(==(modal_np), round.(Int, np)))
@printf("  Paper picks Run 2 (activities only) with P_BIC ≫ others.\n")
@printf("  Match: %s\n",
         modal_info.run == "Run 2" ? "PASS — modal config matches paper" :
         "FAIL — paper modal config underweighted")
println()
function _asym(samples::Vector{Float64})
    m = median(samples)
    lo, hi = quantile(samples, 0.16), quantile(samples, 0.84)
    return (m, hi - m, m - lo, lo, hi)
end
if !isempty(run2_idx)
    K_run2 = K[run2_idx]
    m, up, dn, lo, hi = _asym(K_run2)
    @printf("  K | Run 2 (paper-modal): %.2f +%.2f -%.2f m/s  [68%%: %.2f, %.2f]  (paper: 11 ± 3)\n",
             m, up, dn, lo, hi)
    overlap = lo ≤ 14.0 && hi ≥ 8.0
    @printf("  68%% CI overlaps paper [8, 14]: %s\n", overlap ? "PASS" : "FAIL")
end
if !isempty(run1_idx)
    K_run1 = K[run1_idx]
    m, up, dn, lo, hi = _asym(K_run1)
    @printf("  K | Run 1 (ARMA only):   %.2f +%.2f -%.2f m/s  [68%%: %.2f, %.2f]  (paper: ≈ 5.5)\n",
             m, up, dn, lo, hi)
end
m, up, dn, lo, hi = _asym(K)
@printf("  Marginal K (all configs): %.2f +%.2f -%.2f m/s  [68%%: %.2f, %.2f]\n",
         m, up, dn, lo, hi)
println("="^85)
