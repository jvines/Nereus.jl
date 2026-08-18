# Literature N_p=4 evaluation with optimized Tc per planet.
#
# For each transiting planet (b, d, e) with literature P/K, scan Tc
# across one full period and pick the value that maximizes
# transit_log_likelihood (with the planet active, all others fixed at
# the v13 MAP-N_p=1 systemics). Then evaluate the joint model.
#
# This isolates the question: "given perfect b/c/d/e placement, does
# N_p=4 fit better than N_p=1 in the v13 chain's noise model?"

ENV["WASP47_PT_DIAG_ONLY"] = "1"
include(joinpath(@__DIR__, "wasp47_pt_only.jl"))

using Nereus
using Printf
using Statistics: median
using Random
using MCMCChains

const CHAIN_PATH = joinpath(OUT_DIR, "pt_warm_chains.nc")
chains, _ = load_chains(CHAIN_PATH)

println("\n" * "="^70)
println("Literature N_p=4 evaluation with optimized Tc")
println("="^70)

# 1. Find pseudo-MAP at modal N_p=1 (same as before)
np_vec = vec(Array(chains[:, :n_planets, :]))
mask_modal = np_vec .== 1
modal_idx = findall(mask_modal)
N_SUB = min(2000, length(modal_idx))
rng = MersenneTwister(42)
sub_idx = modal_idx[sortperm(rand(rng, length(modal_idx)))[1:N_SUB]]

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

best_lp = -Inf
best_idx = sub_idx[1]
for idx in sub_idx
    th = build_theta_at(target, chains, idx, 1)
    lp = rv_log_likelihood(th, target.data) +
         transit_log_likelihood(th, target.data)
    if isfinite(lp) && lp > best_lp
        global best_lp = lp; global best_idx = idx
    end
end
@printf("N_p=1 pseudo-MAP idx=%d, logL=%.4f\n", best_idx, best_lp)

# Save MAP systemics for re-use
function copy_systemics!(th_src, th_dst)
    layout = th_dst.params.layout
    sys = layout.systemic
    for f in fieldnames(typeof(sys))
        v = getfield(sys, f)
        if v isa Int
            v > 0 && (th_dst.values[v] = th_src.values[v])
        elseif v isa AbstractVector{Int}
            for s in v
                s > 0 && (th_dst.values[s] = th_src.values[s])
            end
        end
    end
end

th_map = build_theta_at(target, chains, best_idx, 1)

# 2. For each transiting planet, optimize Tc at lit P, K
function set_planet_lit!(th, k, P, K, Tc; e=0.0, ω=0.0, b_imp=0.5, rr=0.10)
    layout = th.params.layout
    block = layout.planet_blocks[k]
    th.values[block.P] = P
    hasproperty(block, :K) && (th.values[block.K] = K)
    hasproperty(block, :Tc) && (th.values[block.Tc] = Tc)
    if hasproperty(block, :sesinw) && hasproperty(block, :secosw)
        sesinw, secosw = Nereus.ew_to_sesinw(e, ω)
        th.values[block.sesinw] = sesinw
        th.values[block.secosw] = secosw
    end
    hasproperty(block, :b) && (th.values[block.b] = b_imp)
    hasproperty(block, :r) && (th.values[block.r] = rr)
end

function optimize_tc(target, th_template, k, P, K, e_val, ω_val, b_imp,
                     rr, Tc_lo, Tc_hi; n_grid=400)
    Tcs = collect(range(Tc_lo, Tc_hi; length=n_grid))
    lps = Float64[]
    for Tc in Tcs
        # Fresh theta with ONLY planet k active
        td_lit = TransDimState(max_planets = target.params.config.max_kplanet)
        Nereus.activate_planet!(td_lit, k)
        th = Theta(target.params; td=td_lit)
        copy_systemics!(th_template, th)
        set_planet_lit!(th, k, P, K, Tc; e=e_val, ω=ω_val,
                        b_imp=b_imp, rr=rr)
        lp = transit_log_likelihood(th, target.data)
        push!(lps, isfinite(lp) ? lp : -Inf)
    end
    j = argmax(lps)
    return Tcs[j], lps[j]
end

# Photometry start time
t1 = target.data.t_phot[1]
@printf("\nt_phot[1] = %.4f (MJD)\n", t1)
@printf("t_phot range: %.2f to %.2f d\n", t1, target.data.t_phot[end])

# Tc priors are UniformPrior(t1, t1+30) — but for the Tc OPTIMIZATION we
# can search the full prior support (t1 to t1+30) for each transiting
# planet. The optimal Tc just needs to be in [t1, t1 + P_planet) modulo
# period; for P=4.16, scan t1..t1+4.16; for P=9.03, t1..t1+9.03;
# for P=0.79, t1..t1+0.79. All inside the prior support.

# WASP-47 b
P_b = 4.1591287; K_b = 140.84
println("\n--- Optimizing Tc for b (P=$P_b) ---")
Tc_b, ll_b_opt = optimize_tc(target, th_map, 1, P_b, K_b, 0.0, 0.0, 0.5, 0.103,
                              t1, t1 + P_b)
@printf("  optimized Tc = %.4f  (transit_logL = %.4f)\n", Tc_b, ll_b_opt)

# WASP-47 d
P_d = 9.030672; K_d = 4.96
println("\n--- Optimizing Tc for d (P=$P_d) ---")
Tc_d, ll_d_opt = optimize_tc(target, th_map, 2, P_d, K_d, 0.0, 0.0, 0.5, 0.030,
                              t1, t1 + 9.0)  # cap at prior upper
@printf("  optimized Tc = %.4f  (transit_logL = %.4f)\n", Tc_d, ll_d_opt)

# WASP-47 e
P_e = 0.789593; K_e = 4.55
println("\n--- Optimizing Tc for e (P=$P_e) ---")
Tc_e, ll_e_opt = optimize_tc(target, th_map, 3, P_e, K_e, 0.0, 0.0, 0.5, 0.013,
                              t1, t1 + P_e)
@printf("  optimized Tc = %.4f  (transit_logL = %.4f)\n", Tc_e, ll_e_opt)

# 3. Build joint N_p=4 with optimized Tc and literature P/K for c (no Tc)
println("\n--- Joint N_p=4 with optimized Tc ---")
td_lit4 = TransDimState(max_planets = target.params.config.max_kplanet)
Nereus.activate_planet!(td_lit4, 1)
Nereus.activate_planet!(td_lit4, 2)
Nereus.activate_planet!(td_lit4, 3)
Nereus.activate_planet!(td_lit4, 4)
th_lit4 = Theta(target.params; td=td_lit4)
copy_systemics!(th_map, th_lit4)
set_planet_lit!(th_lit4, 1, P_b, K_b, Tc_b; rr=0.103)
set_planet_lit!(th_lit4, 2, P_d, K_d, Tc_d; rr=0.030)
set_planet_lit!(th_lit4, 3, P_e, K_e, Tc_e; rr=0.013)
set_planet_lit!(th_lit4, 4, 588.4, 31.6, t1 + 100.0)  # c — RV-only

ll_rv_lit  = rv_log_likelihood(th_lit4, target.data)
ll_phot_lit = transit_log_likelihood(th_lit4, target.data)
@printf("\nLiterature N_p=4 (with optimized Tc):\n")
@printf("  rv_log_likelihood       = %.4f\n", ll_rv_lit)
@printf("  transit_log_likelihood = %.4f\n", ll_phot_lit)
@printf("  total                   = %.4f\n", ll_rv_lit + ll_phot_lit)
@printf("\nDelta vs N_p=1 MAP:\n")
@printf("  rv:    Δ = %+.4f\n", ll_rv_lit - (-923.1739))
@printf("  phot:  Δ = %+.4f\n", ll_phot_lit - 62837.7686)
@printf("  total: Δ = %+.4f\n", (ll_rv_lit + ll_phot_lit) - 61914.5946)

# 4. Same but with eccentricity priors active — e fudged at 0.05 each (small)
println("\n--- Same with e=0.05 per planet ---")
th_e5 = Theta(target.params; td=td_lit4)
copy_systemics!(th_map, th_e5)
set_planet_lit!(th_e5, 1, P_b, K_b, Tc_b; e=0.05, rr=0.103)
set_planet_lit!(th_e5, 2, P_d, K_d, Tc_d; e=0.05, rr=0.030)
set_planet_lit!(th_e5, 3, P_e, K_e, Tc_e; e=0.05, rr=0.013)
set_planet_lit!(th_e5, 4, 588.4, 31.6, t1 + 100.0; e=0.05)

ll_rv_e5  = rv_log_likelihood(th_e5, target.data)
ll_phot_e5 = transit_log_likelihood(th_e5, target.data)
@printf("  rv:    %.4f   phot: %.4f   total: %.4f\n",
        ll_rv_e5, ll_phot_e5, ll_rv_e5 + ll_phot_e5)

println("\n" * "="^70)
println("Done.")
println("="^70)
