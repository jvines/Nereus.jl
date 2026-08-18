# C2 with :M_sec_driven parametrization. Should break the K↔M_pri
# degeneracy that pushed M_pri to 1.19 M_sun in the K_driven run.

using Nereus
using LogDensityProblems
using MCMCChains
using Statistics: median, std, quantile
using Printf

println("=" ^ 70)
println("C2-Msec: HD 4747 — :M_sec_driven parametrization + M_pri sampling")
println("=" ^ 70)

DATADIR  = joinpath(@__DIR__, "..", "..", "..", "data", "HD4747")
HGCAFILE = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062", "HGCA_vEDR3.fits")
hgca   = load_hgca_row(HGCAFILE, 3850)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD4747_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD4747_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

const M_PRI_HD4747 = 0.82

params = Params(
    max_kplanet  = 1,
    planet_modes = [RVAS],
    instruments  = InstrumentConfig(rv = ["HIRES"]),
    data         = data,
    stability    = :none,
    M_s          = M_PRI_HD4747,
    parametrization = ParametrizationConfig(mass = :M_sec_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        "P_k1"         => LogUniformPrior(15.0 * 365.25, 80.0 * 365.25),
        # M_sec sampled directly: 1 M_J to 200 M_J range
        "M_sec_k1"     => LogUniformPrior(0.001, 0.2),
        "sesinw_k1"    => UniformPrior(-1.0, 1.0),
        "secosw_k1"    => UniformPrior(-1.0, 1.0),
        "Mo_k1"        => UniformPrior(0.0, 2π),
        "inc_k1"       => UniformPrior(0.01, π - 0.01),
        "Omega_k1"     => UniformPrior(0.0, 2π),
        # Wide jitter prior matching orvara config_HD4747.ini (1e-5 to 1e3 m/s)
        "sigma_HIRES"  => LogUniformPrior(1e-3, 1e3),
        "M_pri"        => NormalPrior(0.82, 0.04, 0.5, 1.2),
    ),
)
println("Model: $(n_unfrozen(params)) free params (mass = :M_sec_driven)")
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))
println()

target = NereusTarget(params, data)
println("Running Pigeons PT: 12 rounds × 8 chains")
t0 = time()
chains, log_Z = sample_pt(target; n_rounds=12, n_chains=8, seed=42, show_report=false)
elapsed = time() - t0
println(@sprintf("Done in %.1f min, log Z = %.2f", elapsed/60, log_Z))
println()

# Extract M_sec from the chain directly (it's sampled now)
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
P_v   = vec(Array(chains[:, :P_k1, :]))
Mp_v  = vec(Array(chains[:, :M_pri, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
ses_v = vec(Array(chains[:, :sesinw_k1, :]))
sec_v = vec(Array(chains[:, :secosw_k1, :]))
Ω_v   = vec(Array(chains[:, :Omega_k1, :]))

n = length(M_sec)
e_arr = ses_v.^2 .+ sec_v.^2
M_J  = M_sec .* 1047.57
a_au = [a_from_P(P_v[j], Mp_v[j] + M_sec[j]) for j in 1:n]

println("Posteriors:")
for (k, v) in pairs((P_yr=P_v ./ 365.25, M_pri=Mp_v, M_sec=M_sec, M_J=M_J, a_au=a_au,
                     e=e_arr, i_deg=rad2deg.(inc_v), Ω_deg=rad2deg.(Ω_v)))
    q16, q50, q84 = quantile(v, 0.16), quantile(v, 0.50), quantile(v, 0.84)
    @printf("  %-10s = %12.4f  [+%.4f, -%.4f]\n", string(k), q50, q84-q50, q50-q16)
end
println()

println("Brandt+ 2021 reference (HD 4747 B):")
println("  M_J      = 65.3 ± 3.3")
println("  P_yr     = 37.5 ± 1.6")
println("  a_AU     = 11.3 ± 0.4")
println("  e        = 0.74 ± 0.01")
println("  i_deg    = 49.5 ± 1.4")
println()

M_med = quantile(M_J, 0.50); M_sig = (quantile(M_J, 0.84) - quantile(M_J, 0.16))/2
M_diff = abs(M_med - 65.3) / sqrt(M_sig^2 + 3.3^2)
println("=" ^ 70)
@printf("M_companion: %.1f ± %.1f M_J vs Brandt 65.3 ± 3.3 → %.2fσ\n", M_med, M_sig, M_diff)
println(M_diff < 3.0 ? "✅ within 3σ of orvara" : "⚠ tension > 3σ")
