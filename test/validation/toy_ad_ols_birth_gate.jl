#!/usr/bin/env julia
# Gate for the OLS-INFORMED ActivityDecorrelation birth (proposals.jl
# `propose_noise_birth`/`propose_noise_death`, ad_ols_fit/_ad_mvn_logq).
#
# The old toy gate (toy_noise_model_selection.jl) used MA(1), a SINGLE
# toggleable model and a PRIOR-draw birth — n_inactive == n_active_after == 1,
# so the count-term sign cancels and the informed-birth density path is never
# exercised. This gate fixes both gaps:
#
#   * AD with the Keplerian amplitude fixed at 0  ⇒  mean = γ + Σ_k C_k I_k(t),
#     a LINEAR-GAUSSIAN model whose marginal likelihood is EXACT Bayesian
#     linear regression evidence.  Reference Z is analytic, not MC.
#   * Gaussian priors on γ and C  ⇒  closed form  y ~ N(0, σ²I + Φ S Φᵀ).
#   * Two variants:
#       NOFLOOR (default)  — toggleable {AD} only. Isolates the count-term
#                            sign + the informed-birth fwd/rev density.
#       FLOOR=1            — adds an always-on IndicatorFloor, replicating the
#                            HD 18599 joint run. Tests whether counting the
#                            floor in birth's `n_active_after` (but not death's
#                            `n_active`) breaks birth↔death symmetry.
#
# PASS: occupancy P(AD) ≈ σ(logZ_AD − logZ_null) within MC error across seeds.
# FAIL: the informed-birth Hastings ratio does not sample P(M|D).

using Nereus, MCMCChains, Statistics, Printf, Random
using LinearAlgebra

const FLOOR   = get(ENV, "FLOOR", "0") == "1"
const C_TRUE  = parse(Float64, get(ENV, "C_TRUE", "0.6"))
const N       = parse(Int, get(ENV, "N", "60"))
const SIG     = 2.0          # white RV noise (= rv_err, jitter fixed ~0)
const TAU_G   = 5.0          # γ prior sd
const TAU_C   = 1.0          # C prior sd

rng = MersenneTwister(20260615)
t   = sort(rand(rng, N)) .* 90.0
X   = randn(rng, N); X .-= mean(X)                 # one standardized indicator
y   = C_TRUE .* X .+ SIG .* randn(rng, N)          # γ_true = 0
σe  = SIG                                          # σ_eff (jitter fixed ~0)

# ---- EXACT analytic evidence (Bayesian linear regression) ----------------
# y ~ N(0, σ²I + Φ S Φᵀ).  null: Φ=[1], S=[τ_γ²].  AD: Φ=[1 X], S=diag(τ_γ²,τ_C²)
function logmvn0(y, Σ)
    C = cholesky(Symmetric(Σ)); n = length(y)
    -0.5 * (n*log(2π) + 2*sum(log, diag(C.U)) + dot(y, C \ y))
end
Σ_null = σe^2 .* Matrix(I, N, N) .+ TAU_G^2 .* (ones(N) * ones(N)')
Σ_ad   = Σ_null .+ TAU_C^2 .* (X * X')
logZ_null = logmvn0(y, Σ_null)
logZ_ad   = logmvn0(y, Σ_ad)
ΔlogZ     = logZ_ad - logZ_null
P_ad_true = 1 / (1 + exp(-ΔlogZ))                  # equal model prior (incl 0.5)
@printf("ANALYTIC: logZ_null=%.4f  logZ_AD=%.4f  ΔlogZ=%+.4f  →  P(AD)=%.4f\n",
        logZ_null, logZ_ad, ΔlogZ, P_ad_true)

# ---- Nereus model: AD on a no-Keplerian (K=0) RV target -----------------
inds_dict = Dict("bis" => X)
errs_dict = Dict("bis" => fill(1.0, N))            # only used by the floor
data = Data(t_rv=t, rv=y, rv_err=fill(SIG, N), rv_inst=ones(Int, N),
            indicators=inds_dict, indicator_errs=errs_dict)

ad  = ActivityDecorrelation(indicators=["bis"])
flr = IndicatorFloor(channels=[:bis])
noise_models = FLOOR ? NoiseModel[ad, flr] : NoiseModel[ad]

priors = Dict{String,PriorSpec}(
    "n_p"       => FixedPrior(1.0),
    "P_k1"      => FixedPrior(10.0),
    "K_k1"      => FixedPrior(0.0),                # ⇒ Keplerian ≡ 0, linear-Gaussian
    "sesinw_k1" => FixedPrior(0.0), "secosw_k1" => FixedPrior(0.0),
    "Mo_k1"     => FixedPrior(0.0),
    "gamma_I1"  => NormalPrior(0.0, TAU_G, -25.0, 25.0),
    "sigma_I1"  => FixedPrior(1e-3),               # jitter ~0  ⇒ σ_eff = SIG
    "C_bis_I1"  => NormalPrior(0.0, TAU_C, -10.0, 10.0))

params = Params(; max_kplanet=1, planet_modes=[RV_ONLY],
    instruments=InstrumentConfig(rv=["I1"]), data=data, M_s=1.0,
    parametrization=ParametrizationConfig(time=:Mo),
    priors=priors, noise_models=noise_models, transdim_noise=true, stability=:none)
target = NereusTarget(params, data)

td = TransDimConfig(; max_kplanet=1, planets=false, noise=true,
                      toggleable=NoiseModel[ad])

@printf("\nVARIANT: %s | exercising OLS-informed AD birth | n_free(AD on)=%d\n",
        FLOOR ? "FLOOR (AD + always-on IndicatorFloor)" : "NOFLOOR (AD only)",
        n_unfrozen(params))

pAD = Float64[]
for sd in (1, 2, 3, 4)
    res = sample_transdim_ptemcee(target, data; td=td, n_temps=12, n_walkers=80,
            n_steps=14000, n_burnin=5000, n_birth_tries=8, n_birth_refine=10,
            seed=sd, show_progress=false)
    a = vec(Array(res.chains[:noise_active_1])) .> 0.5
    p = mean(a); push!(pAD, p)
    @printf("  seed %d:  P(AD active) = %.4f   (toggles acc = %d)\n",
            sd, p, sum(res.noise_td_accepted))
end
p̄ = mean(pAD); spread = maximum(pAD) - minimum(pAD)
logit(p) = log(clamp(p,1e-4,1-1e-4) / (1 - clamp(p,1e-4,1-1e-4)))
@printf("\nP(AD) sampled = %.4f (4-seed mean, spread %.4f)\n", p̄, spread)
@printf("observed logit = %+.4f   predicted (ΔlogZ) = %+.4f   discrepancy = %.4f nats\n",
        logit(p̄), ΔlogZ, abs(logit(p̄) - ΔlogZ))
if abs(logit(p̄) - ΔlogZ) < 3*max(spread, 0.15) + 0.5
    println("\n✅ OLS-INFORMED AD BIRTH VALIDATED — occupancy = analytic P(M|D)")
else
    println("\n❌ FAIL — informed-birth occupancy ≠ analytic evidence ratio")
end
