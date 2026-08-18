# Post-run diagnostic for v13 — investigates why the trans-dim posterior
# locks on N_p=1 instead of recovering b/c/d/e (4 confirmed planets).
#
# Hypothesis tree:
#   A) Noise model overfit — RV jitter / activity / GP swallows the
#      planet signals. Diagnostic: RV residuals after N_p=1 MAP still
#      show clear b (4.16d), c (588d), d (9.03d), e (0.79d) periodogram
#      peaks; jitter posteriors are pinned at the upper prior bound.
#   B) Likelihood normalization bug — the perf-sprint cache code path
#      mis-scales transit_log_likelihood vs rv_log_likelihood, so
#      adding transit-detected planets doesn't pay off. Diagnostic:
#      phot residuals after N_p=1 MAP STILL show b (21,000 ppm depth)
#      and e (170 ppm) at literature periods.
#   C) Trans-dim proposal blind spots — JointInformedBirth not
#      proposing the right periods. Diagnostic: manually evaluate
#      log-density at literature N_p=4 → if higher than N_p=1 MAP,
#      the chain CAN'T be at equilibrium.
#
# Outputs to results/WASP47_joint_search/:
#   diag_rv_timeseries.pdf
#   diag_rv_phasefold_k1.pdf
#   diag_phot_timeseries.pdf
#   diag_phot_phasefold_k1.pdf
#   diag_rv_residual_periodogram.pdf
#   diag_phot_residual_periodogram.pdf
#   diag_likelihood_split.txt

# Re-include the run script in DIAG_ONLY mode to rebuild target / params /
# data / td without re-running sample_pt_warm.
ENV["WASP47_PT_DIAG_ONLY"] = "1"
include(joinpath(@__DIR__, "wasp47_pt_only.jl"))

using Nereus
using Printf
using Statistics: median, std, quantile, mean
using LombScargle: lombscargle, freqpower
using Random
using CairoMakie

const CHAIN_PATH = joinpath(OUT_DIR, "pt_warm_chains.nc")
isfile(CHAIN_PATH) ||
    error("Chain file not found: $CHAIN_PATH (did sample_pt_warm finish?)")

println("\n" * "="^70)
println("WASP-47 v13 diagnostic — N_p posterior + residual periodograms")
println("="^70)

chains, _meta = load_chains(CHAIN_PATH)

# ---------------------------------------------------------------------
# 1. N_p posterior shape
# ---------------------------------------------------------------------
np_vec = vec(Array(chains[:, :n_planets, :]))
n_total = length(np_vec)
np_dist = Dict(k => count(==(k), np_vec) / n_total for k in 0:4)
println("\nPosterior N_p distribution:")
for k in 0:4
    @printf("  N_p = %d: %.1f%%   (n=%d)\n",
            k, 100 * np_dist[k], count(==(k), np_vec))
end
modal_np = argmax([np_dist[k] for k in 0:4]) - 1
@printf("\nModal N_p = %d (%.1f%%)\n", modal_np, 100 * np_dist[modal_np])

# ---------------------------------------------------------------------
# 2. MAP-ish point at modal N_p
# ---------------------------------------------------------------------
# Chain doesn't store lp — recompute logL on a random modal subset
# to find a pseudo-MAP sample.
mask_modal = np_vec .== modal_np
modal_idx_in_np = findall(mask_modal)
isempty(modal_idx_in_np) &&
    error("No samples at modal N_p=$modal_np")

N_SUB = min(2000, length(modal_idx_in_np))
rng = MersenneTwister(42)
sub_idx = modal_idx_in_np[sortperm(rand(rng, length(modal_idx_in_np)))[1:N_SUB]]

function build_theta_at(target, chains, idx, n_active::Int)
    layout = target.params.layout
    td_local = TransDimState(max_planets = target.params.config.max_kplanet)
    for k in 1:n_active
        Nereus.activate_planet!(td_local, k)
    end
    th = Theta(target.params; td=td_local)
    for name in keys(layout.name_to_idx)
        sym = Symbol(name)
        sym in names(chains) || continue
        v = vec(Array(chains[:, sym, :]))[idx]
        isfinite(v) || continue
        slot = layout.name_to_idx[name]
        th.values[slot] = v
    end
    return th
end

println("Recomputing logL on $N_SUB random modal samples...")
function find_pseudo_map(target, chains, sub_idx, n_active)
    best_lp = -Inf
    best_idx = sub_idx[1]
    for idx in sub_idx
        th = build_theta_at(target, chains, idx, n_active)
        lp = rv_log_likelihood(th, target.data) +
             transit_log_likelihood(th, target.data)
        if isfinite(lp) && lp > best_lp
            best_lp = lp; best_idx = idx
        end
    end
    return best_idx, best_lp
end
map_idx_global, map_lp = find_pseudo_map(target, chains, sub_idx, modal_np)
@printf("Pseudo-MAP idx: %d, logL = %.4f\n", map_idx_global, map_lp)

# ---------------------------------------------------------------------
# 3. MAP planet params
# ---------------------------------------------------------------------
println("\nMAP planet parameters at modal N_p=$modal_np:")
for k in 1:max(modal_np, 1)
    P_k  = vec(Array(chains[:, Symbol("P_k$k"), :]))[map_idx_global]
    K_k  = vec(Array(chains[:, Symbol("K_k$k"), :]))[map_idx_global]
    sek  = vec(Array(chains[:, Symbol("sesinw_k$k"), :]))[map_idx_global]
    cek  = vec(Array(chains[:, Symbol("secosw_k$k"), :]))[map_idx_global]
    e_k  = sek^2 + cek^2
    @printf("  k%d: P=%.4f d   K=%.2f m/s   e=%.3f\n", k, P_k, K_k, e_k)
end

# Literature periods to overlay
const LIT_NAMES = (:e, :b, :d, :c)
const LIT_P     = (0.789593, 4.1591287, 9.030672, 588.4)

# ---------------------------------------------------------------------
# 4. Standard plots from Nereus
# ---------------------------------------------------------------------
println("\nGenerating canonical plots...")
mkpath(OUT_DIR)
try
    fig = plot_rv_timeseries(chains, target.params, target.data)
    save(joinpath(OUT_DIR, "diag_rv_timeseries.pdf"), fig)
    println("  ✓ diag_rv_timeseries.pdf")
catch e
    @warn "plot_rv_timeseries failed" exception=e
end

for k in 1:max(modal_np, 1)
    try
        fig = plot_rv_phasefold(chains, target.params, target.data; planet=k)
        save(joinpath(OUT_DIR, "diag_rv_phasefold_k$k.pdf"), fig)
        println("  ✓ diag_rv_phasefold_k$k.pdf")
    catch e
        @warn "plot_rv_phasefold k=$k failed" exception=e
    end
end

try
    fig = plot_pm_timeseries(chains, target.params, target.data)
    save(joinpath(OUT_DIR, "diag_phot_timeseries.pdf"), fig)
    println("  ✓ diag_phot_timeseries.pdf")
catch e
    @warn "plot_pm_timeseries failed" exception=e
end

for k in 1:max(modal_np, 1)
    try
        fig = plot_pm_phasefold(chains, target.params, target.data; planet=k)
        save(joinpath(OUT_DIR, "diag_phot_phasefold_k$k.pdf"), fig)
        println("  ✓ diag_phot_phasefold_k$k.pdf")
    catch e
        @warn "plot_pm_phasefold k=$k failed" exception=e
    end
end

# ---------------------------------------------------------------------
# 5. Build MAP theta and compute residual periodograms
# ---------------------------------------------------------------------
println("\nBuilding MAP theta + residuals...")

# Pull all unfrozen param names from layout, fetch their MAP values.
theta_map = build_theta_at(target, chains, map_idx_global, modal_np)

# Likelihood split
ll_rv    = rv_log_likelihood(theta_map, target.data)
ll_phot  = transit_log_likelihood(theta_map, target.data)
ll_total = ll_rv + ll_phot
println("\nLikelihood split at MAP (modal N_p=$modal_np):")
@printf("  rv_log_likelihood     = %.4f\n", ll_rv)
@printf("  transit_log_likelihood = %.4f\n", ll_phot)
@printf("  total                 = %.4f\n", ll_total)
open(joinpath(OUT_DIR, "diag_likelihood_split.txt"), "w") do io
    println(io, "WASP-47 v13 likelihood split at MAP (modal N_p=$modal_np)")
    @printf(io, "rv_log_likelihood:      %.4f\n", ll_rv)
    @printf(io, "transit_log_likelihood: %.4f\n", ll_phot)
    @printf(io, "total:                  %.4f\n", ll_total)
end

# Compute RV residuals and run weighted Lomb-Scargle
preds_rv, vars_rv = rv_predictions(theta_map, target.data)
rv_resid = target.data.rv .- preds_rv
weights  = 1 ./ vars_rv  # use full variance (data err + jitter) for LS weighting

freqs_rv = collect(range(1/1500, 1/0.4; length=20_000))
ls_rv = lombscargle(target.data.t_rv, rv_resid, weights;
                    frequencies = freqs_rv)
freqs_rv_out, powers_rv = freqpower(ls_rv)
periods_rv = 1 ./ freqs_rv_out

# Top peaks (anti-aliased)
sorted = sortperm(powers_rv; rev=true)
top_periods = Float64[]
println("\nTop RV-residual periodogram peaks:")
for i in sorted
    P = periods_rv[i]
    if all(abs(log10(P) - log10(P0)) > 0.02 for P0 in top_periods)
        push!(top_periods, P)
        @printf("  P = %.4f d   power = %.3f\n", P, powers_rv[i])
        length(top_periods) >= 10 && break
    end
end

# Match against literature
println("\nLit periods → closest residual peak:")
for (name, P_lit) in zip(LIT_NAMES, LIT_P)
    j = argmin(abs.(periods_rv .- P_lit))
    @printf("  %s (P=%7.4f d): peak P=%7.4f d, power=%.3f\n",
            name, P_lit, periods_rv[j], powers_rv[j])
end

# Plot RV residual periodogram
let fig = Figure(size=(1200, 360))
    ax = Axis(fig[1, 1], xlabel="Period (d)", ylabel="LS power",
              xscale=log10,
              title="RV residuals after N_p=$modal_np MAP fit")
    lines!(ax, periods_rv, powers_rv, color=:black, linewidth=0.7)
    for (name, P_lit) in zip(LIT_NAMES, LIT_P)
        vlines!(ax, [P_lit]; color=:red, linestyle=:dash, linewidth=1.0)
        text!(ax, P_lit, maximum(powers_rv) * 0.9; text=string(name),
              align=(:center, :bottom), color=:red, fontsize=10)
    end
    save(joinpath(OUT_DIR, "diag_rv_residual_periodogram.pdf"), fig)
    println("\n  ✓ diag_rv_residual_periodogram.pdf")
end

# ---------------------------------------------------------------------
# 6. Literature N_p=4 sanity check — manually set planets to b/c/d/e
# from literature and evaluate logL. If this is HIGHER than the chain's
# N_p=1 MAP, the chain failed to find the better mode; if LOWER, the
# chain is right and the noise/likelihood is genuinely happier with N_p=1.
# ---------------------------------------------------------------------
println("\n" * "-"^70)
println("Literature N_p=4 evaluation (b + c + d + e)")
println("-"^70)

td_lit = TransDimState(max_planets = target.params.config.max_kplanet)
Nereus.activate_planet!(td_lit, 1)  # b
Nereus.activate_planet!(td_lit, 2)  # d
Nereus.activate_planet!(td_lit, 3)  # e
Nereus.activate_planet!(td_lit, 4)  # c (RV-only k4 slot)
th_lit = Theta(target.params; td=td_lit)

# Copy MAP non-planet systemics first
for name in keys(target.params.layout.name_to_idx)
    sym = Symbol(name)
    sym in names(chains) || continue
    v = vec(Array(chains[:, sym, :]))[map_idx_global]
    isfinite(v) || continue
    th_lit.values[target.params.layout.name_to_idx[name]] = v
end

# Now overwrite planets with literature values.
# k1 = b, k2 = d, k3 = e (transiting RVPM), k4 = c (RV-only)
function _set_planet!(th, k, P, K, Tc; e=0.0, ω=0.0, b_imp=0.5, rr=0.10)
    layout = th.params.layout
    block = layout.planet_blocks[k]
    th.values[block.P] = P
    if hasproperty(block, :K)
        th.values[block.K] = K
    end
    if hasproperty(block, :Tc)
        th.values[block.Tc] = Tc
    end
    if hasproperty(block, :sesinw) && hasproperty(block, :secosw)
        sesinw, secosw = Nereus.ew_to_sesinw(e, ω)
        th.values[block.sesinw] = sesinw
        th.values[block.secosw] = secosw
    end
    if hasproperty(block, :b)
        th.values[block.b] = b_imp
    end
    if hasproperty(block, :r)
        th.values[block.r] = rr
    end
end

t1 = target.data.t_phot[1]
_set_planet!(th_lit, 1, 4.1591287, 140.84, t1 + 2.5; rr=0.103)   # b
_set_planet!(th_lit, 2, 9.030672,    4.96, t1 + 5.0; rr=0.030)   # d
_set_planet!(th_lit, 3, 0.789593,    4.55, t1 + 0.4; rr=0.013)   # e
_set_planet!(th_lit, 4, 588.4,      31.6,  t1 + 100.0)            # c (no Tc/rr)

ll_rv_lit   = rv_log_likelihood(th_lit, target.data)
ll_phot_lit = transit_log_likelihood(th_lit, target.data)
ll_tot_lit  = ll_rv_lit + ll_phot_lit
@printf("Literature N_p=4 logL split:\n")
@printf("  rv_log_likelihood:      %.4f  (vs MAP-N_p=1: %.4f, Δ=%+.4f)\n",
        ll_rv_lit, ll_rv, ll_rv_lit - ll_rv)
@printf("  transit_log_likelihood: %.4f  (vs MAP-N_p=1: %.4f, Δ=%+.4f)\n",
        ll_phot_lit, ll_phot, ll_phot_lit - ll_phot)
@printf("  total:                  %.4f  (vs MAP-N_p=1: %.4f, Δ=%+.4f)\n",
        ll_tot_lit, ll_total, ll_tot_lit - ll_total)
println("If Δ > 0 → chain failed to find the better mode (proposal/mixing bug)")
println("If Δ < 0 → noise model genuinely fits better than 4 planets (bug or prior issue)")

open(joinpath(OUT_DIR, "diag_likelihood_split.txt"), "a") do io
    println(io, "\n--- Literature N_p=4 evaluation ---")
    @printf(io, "rv_log_likelihood:      %.4f (Δ=%+.4f)\n",
            ll_rv_lit, ll_rv_lit - ll_rv)
    @printf(io, "transit_log_likelihood: %.4f (Δ=%+.4f)\n",
            ll_phot_lit, ll_phot_lit - ll_phot)
    @printf(io, "total:                  %.4f (Δ=%+.4f)\n",
            ll_tot_lit, ll_tot_lit - ll_total)
end

# Photometry residuals — recompute via full transit_log_likelihood
# infrastructure isn't exposed; skip residual periodogram for phot for now.
println("\n[skipping phot residual periodogram — needs phot_predictions helper]")

# Below block left as placeholder (will be skipped due to early return).
phot_pred  = zeros(length(target.data.t_phot))
phot_resid = zeros(length(target.data.t_phot))
phot_w     = ones(length(target.data.t_phot))
# Phot residual periodogram skipped (no phot_predictions helper)

println("\n" * "="^70)
println("Done. All plots in $OUT_DIR")
println("="^70)
