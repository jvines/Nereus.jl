#!/usr/bin/env julia
# Is the residual R̂≈1.43 on gp_act_* a real convergence gap or an INACTIVE-SLOT
# artifact? AGP is active only ~15% of draws; when inactive its gp_act_* params
# are stale unbounded junk. Compare the spread of gp_act_period when AGP is
# ACTIVE vs INACTIVE, and recompute R̂/ESS on the AGP-ACTIVE subset only.
using Nereus, MCMCChains, Statistics, Printf

ch, _ = Nereus.load_chains(joinpath(@__DIR__, "..", "..", "results",
                                      "HD18599_transdim_select", "chains.nc"))
cn = names(ch, :parameters)
@printf("loaded chain: %d draws × %d params\n", size(ch,1)*size(ch,3), length(cn))

agp_col = :noise_active_2          # AD=1, AGP=2 (driver toggle order)
agp = vec(Array(ch[agp_col])) .> 0.5
@printf("AGP active fraction = %.3f\n\n", mean(agp))

for p in (:gp_act_period, :gp_act_lambda_e, :gp_act_lambda_p, :Lc)
    p in cn || (@printf("%-18s : not in chain\n", p); continue)
    v = vec(Array(ch[p]))
    va = v[agp]; vi = v[.!agp]
    @printf("%-18s : ALL std=%.4g  | ACTIVE n=%d std=%.4g med=%.4g  | INACTIVE n=%d std=%.4g\n",
            p, std(v), length(va), (isempty(va) ? NaN : std(va)),
            (isempty(va) ? NaN : median(va)), length(vi), (isempty(vi) ? NaN : std(vi)))
end

# Recompute R̂/ESS for gp_act_period on AGP-ACTIVE draws only, per-walker.
# walker is the FAST index in the flattened cold chain (n_walkers_eff).
nw = 160
println("\n--- R̂/ESS on AGP-ACTIVE draws only (per-walker, gp_act_period) ---")
for p in (:gp_act_period, :gp_act_lambda_e, :gp_act_lambda_p)
    p in cn || continue
    v = vec(Array(ch[p])); a = agp
    nsteps = length(v) ÷ nw
    # rebuild per-walker columns, masking inactive to NaN then dropping
    # (approximate: take the longest common active length across walkers)
    cols = Vector{Vector{Float64}}()
    for w in 1:nw
        idx = w:nw:length(v)
        vals = v[idx][a[idx]]
        push!(cols, vals)
    end
    L = minimum(length.(cols))
    L < 50 && (@printf("%-18s : too few active draws/walker (min=%d)\n", p, L); continue)
    M = Array{Float64,3}(undef, L, nw, 1)
    for w in 1:nw; M[:,w,1] = cols[w][1:L]; end
    c2 = MCMCChains.Chains(M)
    er = MCMCChains.ess_rhat(c2)
    @printf("%-18s : active-only R̂=%.3f  ESS=%.0f  (L=%d/walker)\n",
            p, er[:,:rhat][1], er[:,:ess][1], L)
end
