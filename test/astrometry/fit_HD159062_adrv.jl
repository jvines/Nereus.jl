# HD 159062 with :a_driven parametrization (orvara convention).

using Nereus
using MCMCChains
using Statistics: median, quantile
using Printf

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
    parametrization = ParametrizationConfig(mass = :a_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"        => FixedPrior(1.0),
        # Sample a directly with broad log-flat prior — orvara convention.
        # Range covers 5 AU to 1500 AU, well around HD 159062's a~50–250 AU range.
        "a_k1"       => LogUniformPrior(5.0, 1500.0),
        "M_sec_k1"   => LogUniformPrior(0.05, 1.5),    # M-dwarf to low-mass star
        "sesinw_k1"  => UniformPrior(-1.0, 1.0),
        "secosw_k1"  => UniformPrior(-1.0, 1.0),
        "Mo_k1"      => UniformPrior(0.0, 2π),
        "inc_k1"     => SinePrior(),
        "Omega_k1"   => UniformPrior(0.0, 2π),
        "sigma_HIRES"=> LogUniformPrior(1e-3, 1e3),
        "M_pri"      => NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)
println("=" ^ 70)
println("HD 159062 :a_driven")
println("=" ^ 70)
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

target = NereusTarget(params, data)
t0 = time()
chains, log_Z = sample_pt(target; n_rounds=12, n_chains=8, seed=42, show_report=false)
println(@sprintf("Done in %.1f min, log Z = %.2f", (time()-t0)/60, log_Z))
println()

a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
Mp_v  = vec(Array(chains[:, :M_pri, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
ses_v = vec(Array(chains[:, :sesinw_k1, :]))
sec_v = vec(Array(chains[:, :secosw_k1, :]))
Ω_v   = vec(Array(chains[:, :Omega_k1, :]))
e_arr = ses_v.^2 .+ sec_v.^2
P_yr  = [sqrt(a_v[j]^3 / (Mp_v[j] + M_sec[j])) for j in 1:length(a_v)]

println("Posteriors:")
for (k, v) in pairs((a_au=a_v, P_yr=P_yr, M_pri=Mp_v, M_sec=M_sec,
                     e=e_arr, i_deg=rad2deg.(inc_v), Ω_deg=rad2deg.(Ω_v)))
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-10s = %12.4f  [+%.4f, -%.4f]\n", string(k), q50, q84-q50, q50-q16)
end
println()

# Period mode breakdown
n = length(P_yr)
n_short = sum(P_yr .< 500); n_mid = sum(500 .<= P_yr .< 1500)
n_long  = sum(1500 .<= P_yr .< 5000); n_vlong = sum(P_yr .>= 5000)
@printf("P-distribution: <500yr: %.1f%%, 500–1500yr: %.1f%%, 1500–5000yr: %.1f%% (Brandt), >5000yr: %.1f%%\n",
        100*n_short/n, 100*n_mid/n, 100*n_long/n, 100*n_vlong/n)

println()
println("Brandt+ 2021: M_sec = 0.6083 +0.0083/-0.0073 M_sun, a=207AU, P=2880yr, e=0.51, i=50.5°")
M_med = quantile(M_sec, 0.50); M_sig = (quantile(M_sec, 0.84) - quantile(M_sec, 0.16))/2
M_diff = abs(M_med - 0.6083) / sqrt(M_sig^2 + 0.008^2)
@printf("M_companion: %.4f ± %.4f M_sun → %.2fσ from Brandt\n", M_med, M_sig, M_diff)
