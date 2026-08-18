# Cleaner test: take v13 MAP-N_p=1 as-is, ADD c (P=588d, K=32) on top
# via the RV-only slot k4. This isolates "does c improve logL?" without
# disturbing the chain's optimized b.

ENV["WASP47_PT_DIAG_ONLY"] = "1"
include(joinpath(@__DIR__, "wasp47_pt_only.jl"))

using Nereus, Printf, Random, MCMCChains
chains, _ = load_chains(joinpath(OUT_DIR, "pt_warm_chains.nc"))

println("\n" * "="^70)
println("v13 MAP-N_p=1 + c added")
println("="^70)

# Find pseudo-MAP at N_p=1
np = vec(Array(chains[:, :n_planets, :]))
modal = findall(np .== 1)
N_SUB = min(2000, length(modal))
sub = modal[sortperm(rand(MersenneTwister(42), length(modal)))[1:N_SUB]]

function build_theta_at(target, chains, idx, n_active)
    layout = target.params.layout
    td = TransDimState(max_planets = target.params.config.max_kplanet)
    for k in 1:n_active
        Nereus.activate_planet!(td, k)
    end
    th = Theta(target.params; td=td)
    for name in keys(layout.name_to_idx)
        sym = Symbol(name)
        sym in names(chains) || continue
        v = vec(Array(chains[:, sym, :]))[idx]
        isfinite(v) && (th.values[layout.name_to_idx[name]] = v)
    end
    return th
end

best_lp = -Inf; best_idx = sub[1]
for idx in sub
    th = build_theta_at(target, chains, idx, 1)
    lp = rv_log_likelihood(th, target.data) + transit_log_likelihood(th, target.data)
    if isfinite(lp) && lp > best_lp
        global best_lp = lp; global best_idx = idx
    end
end

# Baseline: N_p=1 MAP
th_map = build_theta_at(target, chains, best_idx, 1)
ll_rv_map = rv_log_likelihood(th_map, target.data)
ll_phot_map = transit_log_likelihood(th_map, target.data)
@printf("Baseline N_p=1 MAP:\n  rv: %.4f\n  phot: %.4f\n  total: %.4f\n",
        ll_rv_map, ll_phot_map, ll_rv_map + ll_phot_map)

# Now activate k4 (RV-only) and set c parameters. k4's time anchor
# could be Mo, Tp, or Tc depending on parametrization. The parametrization
# config in this script is `time = :Tc`. But k4 is RVOnlyBlock — let's
# inspect what time slot it has.
block_k4 = target.params.layout.planet_blocks[4]
@printf("\nk4 block type: %s\n", typeof(block_k4))
@printf("k4 fields: %s\n", fieldnames(typeof(block_k4)))

# Activate k4 + set P, K, and time anchor to median(t_rv) (a sensible
# default — anchor doesn't matter much if K is small).
td_2 = TransDimState(max_planets = target.params.config.max_kplanet)
Nereus.activate_planet!(td_2, 1)
Nereus.activate_planet!(td_2, 4)
th_2 = Theta(target.params; td=td_2)
# Copy ALL values from th_map
th_2.values .= th_map.values

# Set c parameters in k4 slot
sys = target.params.layout
block = sys.planet_blocks[4]
th_2.values[block.P]      = 588.4
hasproperty(block, :K)      && (th_2.values[block.K] = 31.6)
hasproperty(block, :sesinw) && (th_2.values[block.sesinw] = 0.0)
hasproperty(block, :secosw) && (th_2.values[block.secosw] = 0.0)
# Time anchor — set to median(t_rv) which is well within prior support.
hasproperty(block, :Mo) && (th_2.values[block.Mo] = 0.0)
hasproperty(block, :Tp) && (th_2.values[block.Tp] = target.data.t_ref)
hasproperty(block, :Tc) && (th_2.values[block.Tc] = target.data.t_phot[1] + 100.0)

ll_rv_c = rv_log_likelihood(th_2, target.data)
ll_phot_c = transit_log_likelihood(th_2, target.data)
@printf("\nMAP + c (P=588.4, K=31.6, e=0):\n  rv: %.4f (Δ=%+.4f)\n  phot: %.4f (Δ=%+.4f)\n  total: %.4f (Δ=%+.4f)\n",
        ll_rv_c, ll_rv_c - ll_rv_map,
        ll_phot_c, ll_phot_c - ll_phot_map,
        ll_rv_c + ll_phot_c, (ll_rv_c + ll_phot_c) - (ll_rv_map + ll_phot_map))

# RVOnlyBlock uses `:t` for the generic time anchor.
println("\nSweeping c's time anchor `:t` for max RV logL...")
function sweep_t(th_2, block, target, t_grid)
    bll = -Inf
    bt  = t_grid[1]
    for t in t_grid
        th_2.values[block.t] = t
        ll = rv_log_likelihood(th_2, target.data)
        if isfinite(ll) && ll > bll
            bll = ll; bt = t
        end
    end
    return bt, bll
end
t_grid = collect(range(target.data.t_rv[1],
                          target.data.t_rv[1] + 588.4; length=300))
best_t, best_ll = sweep_t(th_2, block, target, t_grid)
@printf("  best t = %.4f, rv_logL = %.4f (Δ from MAP = %+.4f)\n",
        best_t, best_ll, best_ll - ll_rv_map)
th_2.values[block.t] = best_t

ll_rv_opt = rv_log_likelihood(th_2, target.data)
ll_phot_opt = transit_log_likelihood(th_2, target.data)
@printf("\nMAP + c (Tc/Tp/Mo optimized):\n  rv: %.4f (Δ=%+.4f)\n  phot: %.4f (Δ=%+.4f)\n  total: %.4f (Δ=%+.4f)\n",
        ll_rv_opt, ll_rv_opt - ll_rv_map,
        ll_phot_opt, ll_phot_opt - ll_phot_map,
        ll_rv_opt + ll_phot_opt, (ll_rv_opt + ll_phot_opt) - (ll_rv_map + ll_phot_map))

## Re-optimize γ_$inst per-instrument zero-points with c active.
## Hypothesis: chain's γ values absorbed c's per-instrument phase. If
## we let γ refit, the RV improvement from c should appear.
println("\nRe-optimizing γ_inst values with c active...")
sys = target.params.layout.systemic
n_inst = length(sys.rv_gamma)

# Per-instrument WLS optimum γ: γ_i = (Σ w_j (rv_j − model_j)) / (Σ w_j)
# where the sum is over observations belonging to instrument i and
# `model_j` excludes the gamma offset (so we use rv_predictions and
# subtract the current gamma contribution).
function optimize_gammas!(th, target)
    sys = th.params.layout.systemic
    # Get current model predictions
    preds, _vars = rv_predictions(th, target.data)
    # rv_predictions includes gamma. So: rv_signal = preds − γ_i_for_obs_i
    # Solve: best γ_i_new = γ_i_old + Σ w_j (rv_obs - preds) / Σ w_j
    n_obs = length(target.data.t_rv)
    sums_num = zeros(length(sys.rv_gamma))
    sums_den = zeros(length(sys.rv_gamma))
    for j in 1:n_obs
        ins  = target.data.rv_inst[j]
        σ²    = target.data.rv_err[j]^2 +
                   th.values[sys.rv_sigma[ins]]^2
        w    = 1 / σ²
        sums_num[ins] += w * (target.data.rv[j] - preds[j])
        sums_den[ins] += w
    end
    for ins in 1:length(sys.rv_gamma)
        sums_den[ins] > 0 || continue
        Δγ = sums_num[ins] / sums_den[ins]
        th.values[sys.rv_gamma[ins]] += Δγ
    end
    return th
end

# Apply to th_2 (MAP + c)
optimize_gammas!(th_2, target)
ll_rv_g = rv_log_likelihood(th_2, target.data)
ll_phot_g = transit_log_likelihood(th_2, target.data)
@printf("\nMAP + c (γ re-optimized):\n  rv: %.4f (Δ from MAP-N_p=1: %+.4f)\n  phot: %.4f (Δ: %+.4f)\n  total: %.4f (Δ: %+.4f)\n",
        ll_rv_g, ll_rv_g - ll_rv_map,
        ll_phot_g, ll_phot_g - ll_phot_map,
        ll_rv_g + ll_phot_g, (ll_rv_g + ll_phot_g) - (ll_rv_map + ll_phot_map))

# Show the γ shifts
println("\nγ_inst shifts after adding c:")
for (i, slot) in enumerate(target.params.layout.systemic.rv_gamma)
    γ_map  = th_map.values[slot]
    γ_new  = th_2.values[slot]
    @printf("  inst %d: %.3f → %.3f (Δ = %+.3f m/s)\n", i, γ_map, γ_new, γ_new - γ_map)
end

println("\n" * "="^70)
println("Done.")
println("="^70)
