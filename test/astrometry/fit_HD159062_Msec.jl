# C3 with :M_sec_driven parametrization.

using Nereus
using LogDensityProblems
using MCMCChains
using Statistics: median, std, quantile
using Printf

println("=" ^ 70)
println("C3-Msec: HD 159062 — :M_sec_driven + sampled M_pri + broad P prior")
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
    parametrization = ParametrizationConfig(mass = :M_sec_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        "P_k1"         => LogUniformPrior(50.0 * 365.25, 30000.0 * 365.25),
        "M_sec_k1"     => LogUniformPrior(0.001, 2.0),  # 1 M_J → 2 M_sun
        "sesinw_k1"    => UniformPrior(-1.0, 1.0),
        "secosw_k1"    => UniformPrior(-1.0, 1.0),
        "Mo_k1"        => UniformPrior(0.0, 2π),
        "inc_k1"       => SinePrior(),
        "Omega_k1"     => UniformPrior(0.0, 2π),
        # Wide jitter matching orvara HD159062 config (1e-5 to 1e3 m/s)
        "sigma_HIRES"  => LogUniformPrior(1e-3, 1e3),
        "M_pri"        => NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)
target = NereusTarget(params, data)
t0 = time()
chains, log_Z = sample_pt(target; n_rounds=12, n_chains=8, seed=42, show_report=false)
println(@sprintf("Done in %.1f min, log Z = %.2f", (time()-t0)/60, log_Z))
println()

M_sec = vec(Array(chains[:, :M_sec_k1, :]))
P_v   = vec(Array(chains[:, :P_k1, :]))
Mp_v  = vec(Array(chains[:, :M_pri, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
ses_v = vec(Array(chains[:, :sesinw_k1, :]))
sec_v = vec(Array(chains[:, :secosw_k1, :]))
Ω_v   = vec(Array(chains[:, :Omega_k1, :]))

n = length(M_sec)
e_arr = ses_v.^2 .+ sec_v.^2
a_au = [a_from_P(P_v[j], Mp_v[j] + M_sec[j]) for j in 1:n]

println("Posteriors:")
for (k, v) in pairs((P_yr=P_v ./ 365.25, M_pri=Mp_v, M_sec=M_sec, a_au=a_au,
                     e=e_arr, i_deg=rad2deg.(inc_v), Ω_deg=rad2deg.(Ω_v)))
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-10s = %12.4f  [+%.4f, -%.4f]\n", string(k), q50, q84-q50, q50-q16)
end
println()

println("Brandt+ 2021: M_companion = 0.6083 +0.0083/-0.0073 M_sun, P=2880yr, a=207AU, e=0.51, i=50.5°")
println()

M_med = quantile(M_sec, 0.50); M_sig = (quantile(M_sec, 0.84) - quantile(M_sec, 0.16))/2
M_diff = abs(M_med - 0.6083) / sqrt(M_sig^2 + 0.008^2)
@printf("M_companion: %.4f ± %.4f M_sun → %.2fσ from Brandt\n", M_med, M_sig, M_diff)
println(M_diff < 3.0 ? "✅ within 3σ" : "⚠ tension > 3σ")
