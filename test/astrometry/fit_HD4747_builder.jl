# HD 4747 B with the build_target convenience API — same physics as
# fit_HD4747_NUTS.jl but ~30 lines shorter at the model-definition step.

using Nereus
using MCMCChains
using Statistics: median, quantile
using Printf

DATADIR  = joinpath(@__DIR__, "..", "..", "..", "data", "HD4747")
HGCAFILE = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062", "HGCA_vEDR3.fits")

hgca   = load_hgca_row(HGCAFILE, 3850)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD4747_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD4747_relAST.txt"))

target = build_target(
    M_pri = 0.856,                                # fixed at Crepp+ 2016
    planets = (b = (
        a      = LogUniformPrior(1.0, 100.0),
        M_sec  = LogUniformPrior(0.001, 0.5),
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2π),
    ),),
    rv = (HIRES = (data = rvdat, sigma = LogUniformPrior(0.1, 30.0)),),
    relAST = relast,
    hgca   = hgca,
)
println("Free params: $(n_unfrozen(target.params))")
println("Unfrozen: ", join(target.params.layout.unfrozen_names, ", "))
println()

println("Running Pigeons PT: 12 rounds × 8 chains")
t0 = time()
chains, log_Z = sample_pt(target; n_rounds = 12, n_chains = 8, seed = 42, show_report = false)
@printf("Done in %.1f min, log Z = %.2f\n\n", (time()-t0)/60, log_Z)

# Posterior summary
a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
e_v   = ses.^2 .+ sec.^2
M_J   = M_sec .* 1047.57
P_yr  = [sqrt(a_v[j]^3 / (0.856 + M_sec[j])) for j in 1:length(a_v)]

println("Posterior medians:")
@printf("  a_AU      = %.2f [+%.2f, -%.2f]\n",
        quantile(a_v, 0.5), quantile(a_v, 0.84) - quantile(a_v, 0.5),
        quantile(a_v, 0.5) - quantile(a_v, 0.16))
@printf("  M_J       = %.1f [+%.1f, -%.1f]\n",
        quantile(M_J, 0.5), quantile(M_J, 0.84) - quantile(M_J, 0.5),
        quantile(M_J, 0.5) - quantile(M_J, 0.16))
@printf("  P_yr      = %.1f [+%.1f, -%.1f]\n",
        quantile(P_yr, 0.5), quantile(P_yr, 0.84) - quantile(P_yr, 0.5),
        quantile(P_yr, 0.5) - quantile(P_yr, 0.16))
@printf("  e         = %.3f\n", quantile(e_v, 0.5))
@printf("  i_deg     = %.1f\n", rad2deg(quantile(inc_v, 0.5)))

M_med = quantile(M_J, 0.5)
M_sig = (quantile(M_J, 0.84) - quantile(M_J, 0.16)) / 2
M_diff = abs(M_med - 65.3) / sqrt(M_sig^2 + 3.3^2)
@printf("\nM_companion: ours = %.1f ± %.1f M_J, Brandt = 65.3 ± 3.3\nTension: %.2f σ\n",
        M_med, M_sig, M_diff)
println(M_diff < 3.0 ? "✅ build_target API VALIDATED" :
                       "⚠ Tension > 3σ — investigate")
