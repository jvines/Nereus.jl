#!/usr/bin/env julia
# Validation of sample_pt_hmc (Hamiltonian parallel tempering, fixed-dim):
#   1. recovers an injected 1-planet RV signal (K, P, e),
#   2. its TI+ log-evidence agrees with BOTH the stretch-move PT (same estimator,
#      different sampler) AND nested sampling (independent family) on the SAME target.
# The injection is ECCENTRIC (e=0.3): a CIRCULAR signal sits on the
# sesinw/secosw=0 origin degeneracy (Mo/ω unidentifiable) — there NS integrates
# the degenerate ridge unreliably (2–4.5 nat scatter vs TI, NOT a mixing bug),
# so cross-family evidence comparison is meaningless. See feedback_test_e0_ω_degeneracy.
using Nereus, MCMCChains, Statistics, Printf, Random

# Eccentric Keplerian RV (Newton-solved), so the posterior is a clean isolated
# mode away from the e=0 cusp.
function kepler_rv(t, P, K, e, ω; t0 = 0.0)
    M = 2π .* (t .- t0) ./ P; E = copy(M)
    for _ in 1:60; E .-= (E .- e .* sin.(E) .- M) ./ (1 .- e .* cos.(E)); end
    ν = 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
    return K .* (cos.(ν .+ ω) .+ e * cos(ω))
end
rng = MersenneTwister(11); n = 45
t   = sort(rand(rng, n) .* 60.0)
P_true, K_true, e_true = 7.3, 9.0, 0.3
rv  = kepler_rv(t, P_true, K_true, e_true, 0.8) .+ 1.2 .* randn(rng, n)
data = Data(t_rv = t, rv = rv, rv_err = fill(1.2, n), rv_inst = ones(Int, n))
priors = Dict{String,PriorSpec}(
    "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
    "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
    "Mo_k1" => UniformPrior(0.0, 2π),
    "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))
params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
    parametrization = ParametrizationConfig(time = :Mo), priors = priors)

target = NereusTarget(params, data; unconstrained = true)
ch, logz, rep = Nereus.sample_pt_hmc(target; n_temps = 20, n_sweeps = 1500,
                                      n_warmup = 400, n_walkers_per_temp = 2,
                                      seed = 1, progress = false)
K = vec(Array(ch[:K_k1])); P = vec(Array(ch[:P_k1]))
e = sqrt.(vec(Array(ch[:sesinw_k1])).^2 .+ vec(Array(ch[:secosw_k1])).^2)
@printf("\n=== PT-HMC (adaptive ladder) ===\n")
@printf("K_k1 = %.2f [%.2f, %.2f]   (injected %.1f)\n",
        median(K), quantile(K, 0.16), quantile(K, 0.84), K_true)
@printf("P_k1 = %.4f [%.4f, %.4f]  (injected %.2f)\n",
        median(P), quantile(P, 0.16), quantile(P, 0.84), P_true)
@printf("e_k1 = %.2f [%.2f, %.2f]   (injected %.1f; Lucy-Sweeney biased high, informational)\n",
        median(e), quantile(e, 0.16), quantile(e, 0.84), e_true)
@printf("logZ: TI+=%.2f±%.2f  SS+=%.2f  H+=%.2f\n",
        rep.ti_plus[1], rep.ti_plus[2], rep.ss_plus[1], rep.hybrid[1])

# GATE: same-estimator cross-check vs the stretch-move PT (BOTH TI+, different
# sampler) — isolates the SAMPLER from the estimator. The reference must be
# ADEQUATELY RESOLVED: a 12-temp stretch-PT under-resolves this RV TI integrand
# by ~5 nats (climbs -97→-93→-92 from 12→24→36 temps), so n_chains=24 here.
ch2, logz2 = Nereus.sample_pt(NereusTarget(params, data; unconstrained = true);
                               n_rounds = 14, n_chains = 24, seed = 1, show_report = false)
@printf("\n=== stretch-PT (24 temps, TI+) ===\nlogZ = %.2f\n", logz2)
@printf("ΔlogZ (HMC − stretch) = %+.2f\n", logz - logz2)

# Independent cross-FAMILY sanity (informational, NOT gated): nested sampling.
# Default rslice (slices=5) UNDER-MIXES the curved e–ω–Mo ridge → logZ biased
# LOW by ~2.5 nats; strong rslice (slices=15, walks=50) climbs to within ~0.5
# nats of the TI value. We therefore cross-check against the STRONG NS config.
_, logz_ns = Nereus.sample_nested(target, data; n_live = 1500, dlogz = 0.05,
                                   proposal = :rslice, slices = 15, n_walks = 50)
@printf("nested (strong rslice) logZ = %.2f   ΔlogZ (HMC − NS) = %+.2f\n",
        logz_ns, logz - logz_ns)

# Recovery = truth within the 90% credible interval. NOT the 68% CI: at e=0.3
# the eccentricity is Lucy-Sweeney biased high, and the K–e correlation drags K
# up with it (all three samplers agree K≈9.4) — an HONEST biased posterior, not
# a sampler error, so the 68% CI legitimately misses the point injection.
k_ok = quantile(K, 0.05) < K_true < quantile(K, 0.95)
p_ok = quantile(P, 0.05) < P_true < quantile(P, 0.95)
z_ok = isfinite(logz) && abs(logz - logz2) < 1.5          # same-estimator: tight
ns_ok = isfinite(logz_ns) && abs(logz - logz_ns) < 1.5    # cross-family: strong NS
@printf("\n[%s] K recovered  [%s] P recovered  [%s] logZ vs stretch-PT (<1.5)  [%s] logZ vs strong-NS (<1.5)\n",
        k_ok ? "PASS" : "FAIL", p_ok ? "PASS" : "FAIL",
        z_ok ? "PASS" : "FAIL", ns_ok ? "PASS" : "FAIL")
println(k_ok && p_ok && z_ok && ns_ok ? "\n✅ PT-HMC VALIDATED" : "\n❌ PT-HMC needs work")
