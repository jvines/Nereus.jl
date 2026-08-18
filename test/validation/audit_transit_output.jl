#!/usr/bin/env julia
# Audit the TRANSIT output package: real HD 209458 b TESS s56 transit-only fit →
# science tables (transit derived: Rp/Rs, b, radius, inc, a/Rs, depth, LD) + the
# pm_phasefold / pm_timeseries plots. Exercises the transit branch of
# science_tables + the pm plots that the RV audit never touched.

using Nereus, MCMCChains, Statistics, Printf

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD209458")
lc = load_tess_lc(joinpath(DATADIR, "HD209458_tess_s56_lc.csv"))
const P_LIT = 3.52474859
const TC_LIT_BASE = 2452826.628521 - 2_457_000.0
n_tr = round(Int, (mean(lc.t) - TC_LIT_BASE) / P_LIT)
const TC_LIT = TC_LIT_BASE + n_tr * P_LIT
# Output-audit speed: keep only NEAR-TRANSIT points (|phase| < 0.04 ≈ ±3.4 hr),
# then bin to 10-min. The flat out-of-transit baseline (>95% of the sector)
# adds cost without constraining the transit shape we're auditing.
ph = mod.(lc.t .- TC_LIT .+ 0.5P_LIT, P_LIT) ./ P_LIT .- 0.5
keep = abs.(ph) .< 0.04
kt, kf, ke = lc.t[keep], lc.flux[keep], lc.flux_err[keep]
kph = ph[keep]
# Per-transit LOCAL baseline normalization BEFORE stitching/folding. Within the
# sector each transit window carries its own slow flux drift (stellar +
# TESS systematics); a single global `offset` can't absorb it, so stitching the
# windows raw leaves transit-to-transit residual structure. Fit a line to each
# transit's OUT-OF-TRANSIT flanks (|phase| > ~T14/2) and divide it out.
let ph_half = 0.019            # ~T14/2 in phase for HD209458 b (T14 ≈ 3.1 hr)
    cyc = round.(Int, (kt .- TC_LIT) ./ P_LIT)
    for c in unique(cyc)
        idx = findall(==(c), cyc)
        oot = idx[abs.(kph[idx]) .> ph_half]
        length(oot) < 4 && continue                       # too few flank points
        t0 = mean(kt[oot])
        coef = hcat(ones(length(oot)), kt[oot] .- t0) \ kf[oot]   # linear baseline
        base = coef[1] .+ coef[2] .* (kt[idx] .- t0)
        kf[idx] ./= base; ke[idx] ./= base
    end
end
flat_t, flat_flux, flat_flux_err = let bin_d = 10.0/(60*24)
    bid = floor.(Int, (kt .- minimum(kt)) ./ bin_d); ub = sort(unique(bid))
    tb=Float64[]; fb=Float64[]; eb=Float64[]
    for b in ub; i=findall(==(b),bid); push!(tb,mean(kt[i])); push!(fb,mean(kf[i]))
        push!(eb,mean(ke[i])/sqrt(length(i))); end
    tb,fb,eb
end
@printf("HD209458 TESS s56 → %d near-transit binned pts (of %d)\n", length(flat_t), length(lc.t))

target = build_target(M_s=1.148, R_s=1.203,
    planets=(b=(P=NormalPrior(P_LIT,0.005,3.40,3.65),
                Tc=NormalPrior(TC_LIT,0.05,TC_LIT-0.5,TC_LIT+0.5),
                sesinw=UniformPrior(-0.3,0.3), secosw=UniformPrior(-0.3,0.3),
                b=UniformPrior(0.0,1.0), rr=UniformPrior(0.05,0.20)),),
    phot=(TESS=(data=(t=flat_t,flux=flat_flux,flux_err=flat_flux_err),
                jitter=LogUniformPrior(1e-5,1e-2),
                offset=NormalPrior(0.0,1e-3,-0.01,0.01),
                q1=UniformPrior(0.0,1.0), q2=UniformPrior(0.0,1.0)),))
@printf("free params: %s\n", join(target.params.layout.unfrozen_names, ", "))

@printf("\nNUTS 2×(300+400)...\n"); t0=time()
chains = sample_nuts(target; n_chains=2, n_samples=400, n_warmup=300, seed=42, show_report=false, progress=false)
@printf("done %.1f min\n", (time()-t0)/60)

# --- generate the output package ---
out = joinpath(@__DIR__, "..", "..", "results", "HD209458_transit")
mkpath(out)
Nereus.save_chains(joinpath(out,"chains.nc"), chains, target.params; data=target.data)
summ = Nereus.science_summary(out, chains, target.params, target.data;
                               n_walkers=2, M_s=1.148, R_s=1.203, T_eff=6065.0)
pdir = joinpath(out,"plots"); mkpath(joinpath(pdir,"models"))
try; Nereus.plot_pm_phasefold(chains, target.params, target.data; output=pdir); @printf("✓ pm_phasefold\n"); catch e; @warn "pm_phasefold" exception=e; end
try; Nereus.plot_pm_timeseries(chains, target.params, target.data; output=pdir); @printf("✓ pm_timeseries\n"); catch e; @warn "pm_timeseries" exception=e; end

@printf("\n=== TRANSIT derived table (vs Knutson+07: b=0.50, Rp/Rs=0.1209, depth=1.46%%) ===\n")
for (k,v) in summ["derived"]["parameters"]
    @printf("  %-16s = %10.5g  [-%.4g/+%.4g] %s%s\n", k, v["value"], v["err_lo"], v["err_hi"],
            v["unit"], get(v,"railed",false) ? " RAILED" : "")
end
@printf("\ntables: %s\n", joinpath(out,"tables"))
@printf("plots:  %s\n", pdir)
