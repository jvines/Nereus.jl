# C3-v2: re-run HD 159062 with broader P prior to test mode migration.

using Nereus
using LogDensityProblems
using MCMCChains
using Statistics: median, std, quantile
using Printf

println("=" ^ 70)
println("C3-v2: HD 159062 Pigeons PT with broader prior")
println("=" ^ 70)

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

const M_PRI = 0.81

params = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = M_PRI,
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        # Much broader P prior — 50 to 30000 yr in days
        "P_k1"         => LogUniformPrior(50.0 * 365.25, 30000.0 * 365.25),
        "K_k1"         => LogUniformPrior(50.0, 5000.0),
        "sesinw_k1"    => UniformPrior(-1.0, 1.0),
        "secosw_k1"    => UniformPrior(-1.0, 1.0),
        "Mo_k1"        => UniformPrior(0.0, 2π),
        "inc_k1"       => UniformPrior(0.01, π - 0.01),
        "Omega_k1"     => UniformPrior(0.0, 2π),
        "sigma_HIRES"  => LogUniformPrior(0.1, 20.0),
        # Sample M_pri (Hirsch+ 2019 / Brandt+ 2021: 0.81 ± 0.04 M_sun)
        "M_pri"        => NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)
target = NereusTarget(params, data)
println("Prior on P: 50–30000 yr (vs C3 original 800–8000 yr)")
println()

println("Running Pigeons PT: 12 rounds × 8 chains")
t0 = time()
chains, log_Z = sample_pt(target; n_rounds = 12, n_chains = 8, seed = 42, show_report = false)
elapsed = time() - t0
println(@sprintf("Done in %.1f min, log Z = %.2f", elapsed/60, log_Z))
println()

P_v   = vec(Array(chains[:, :P_k1, :]))
K_v   = vec(Array(chains[:, :K_k1, :]))
ses_v = vec(Array(chains[:, :sesinw_k1, :]))
sec_v = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
Ω_v   = vec(Array(chains[:, :Omega_k1, :]))
Mp_v  = vec(Array(chains[:, :M_pri, :]))

n = length(P_v)
M_sec = Vector{Float64}(undef, n)
a_au  = Vector{Float64}(undef, n)
e_arr = Vector{Float64}(undef, n)
for j in 1:n
    e_arr[j] = ses_v[j]^2 + sec_v[j]^2
    M_sec[j] = msec_from_K(K_v[j], P_v[j], e_arr[j], abs(sin(inc_v[j])), Mp_v[j])
    a_au[j]  = a_from_P(P_v[j], Mp_v[j] + M_sec[j])
end

# Period histogram (rough multimodality check)
P_yr = P_v ./ 365.25
println("P_yr posterior summary:")
@printf("  median = %.1f, mean = %.1f\n", quantile(P_yr, 0.5), sum(P_yr)/n)
@printf("  range:  min = %.1f, max = %.1f\n", minimum(P_yr), maximum(P_yr))
@printf("  16/50/84%% = %.1f / %.1f / %.1f\n",
        quantile(P_yr, 0.16), quantile(P_yr, 0.50), quantile(P_yr, 0.84))
println()

# Multi-mode detection: compute fraction of samples in different regions
n_short = sum(P_yr .< 500)
n_mid   = sum(500 .<= P_yr .< 1500)
n_long  = sum(1500 .<= P_yr .< 5000)
n_vlong = sum(P_yr .>= 5000)
@printf("  P < 500 yr:        %d samples (%.1f%%)\n", n_short, 100*n_short/n)
@printf("  500 ≤ P < 1500 yr: %d samples (%.1f%%)\n", n_mid,   100*n_mid/n)
@printf("  1500 ≤ P < 5000 yr:%d samples (%.1f%%)  ← Brandt's region\n", n_long,  100*n_long/n)
@printf("  P ≥ 5000 yr:       %d samples (%.1f%%)\n", n_vlong, 100*n_vlong/n)
println()

# M_sec summary
println("M_companion posterior summary:")
@printf("  16/50/84%% = %.4f / %.4f / %.4f M_sun\n",
        quantile(M_sec, 0.16), quantile(M_sec, 0.50), quantile(M_sec, 0.84))
@printf("  σ_M = %.4f M_sun  (orvara: 0.609 ± 0.024)\n",
        (quantile(M_sec, 0.84) - quantile(M_sec, 0.16))/2)
println()

println("a posterior:  16/50/84%% = $(round(quantile(a_au, 0.16), digits=1)) / $(round(quantile(a_au, 0.50), digits=1)) / $(round(quantile(a_au, 0.84), digits=1)) AU")
