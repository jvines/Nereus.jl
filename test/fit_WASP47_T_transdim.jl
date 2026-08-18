# WASP-47 — TESS s42 trans-dim T-only fit.
#
# Trans-dim setup: max_kplanet = 3, each slot tightly priored at one of
# the three known transit periods (b: 4.16d HJ, d: 9.03d Nep, e: 0.79d USP).
# RJMCMC explores N_p ∈ {0, 1, 2, 3}; posterior should strongly favor
# N_p = 3 (all three transits visible at TESS precision).
#
# Headline test for paper: real-data trans-dim recovery of multi-planet
# transit count.

using Nereus
using Statistics: median, std, mean, quantile
using Printf
using MCMCChains

println("=" ^ 70)
println("WASP-47 — TESS s42 trans-dim T-only")
println("=" ^ 70)

DATADIR = joinpath(@__DIR__, "..", "..", "data", "WASP47")
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"))

const Pb_LIT = 4.1591287
const Pd_LIT = 9.030672
const Pe_LIT = 0.789593
# Empirical Tc anchors (see fit_WASP47_T_only.jl comment).
const TCb_LIT = 2461.834
const TCd_LIT = 2458.373
const TCe_LIT = 2461.201

# Bin to 5-min cadence
flat = let bd = 5.0 / (60 * 24)
    bin_id = floor.(Int, (lc.t .- minimum(lc.t)) ./ bd)
    ub = sort(unique(bin_id))
    tt = Float64[]; ff = Float64[]; ee = Float64[]
    for b in ub
        idxs = findall(==(b), bin_id)
        push!(tt, mean(lc.t[idxs])); push!(ff, mean(lc.flux[idxs]))
        push!(ee, mean(lc.flux_err[idxs]) / sqrt(length(idxs)))
    end
    (t = tt, flux = ff, flux_err = ee)
end
println("Binned PDCSAP: $(length(flat.t)) pts, σ ≈ $(round(std(flat.flux), digits=6))")

# Build target with 3 planet slots, each priored on one candidate.
target = build_target(
    M_s = 1.04, R_s = 1.137,
    planets = (
        b = (
            P  = NormalPrior(Pb_LIT, 0.005, 4.10, 4.22),
            Tc = NormalPrior(TCb_LIT, 0.01, TCb_LIT - 0.1, TCb_LIT + 0.1),
            sesinw = UniformPrior(-0.15, 0.15),
            secosw = UniformPrior(-0.15, 0.15),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.05, 0.18),
        ),
        d = (
            P  = NormalPrior(Pd_LIT, 0.01, 8.95, 9.10),
            Tc = NormalPrior(TCd_LIT, 0.02, TCd_LIT - 0.2, TCd_LIT + 0.2),
            sesinw = UniformPrior(-0.15, 0.15),
            secosw = UniformPrior(-0.15, 0.15),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.005, 0.05),
        ),
        e = (
            P  = NormalPrior(Pe_LIT, 0.001, 0.78, 0.80),
            Tc = NormalPrior(TCe_LIT, 0.01, TCe_LIT - 0.05, TCe_LIT + 0.05),
            sesinw = UniformPrior(-0.05, 0.05),
            secosw = UniformPrior(-0.05, 0.05),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.005, 0.025),
        ),
    ),
    phot = (TESS = (
        data    = flat,
        # CRITICAL for trans-dim: jitter upper-bound must not exceed
        # transit depth, else the chain absorbs transit dips into
        # noise at N_p=0 and never escapes. Sector σ ≈ 0.0025 →
        # cap jitter at 1e-3 forces birth proposals to compete with
        # actual signal, not noise inflation.
        jitter  = LogUniformPrior(1e-5, 1e-3),
        offset  = NormalPrior(0.0, 1e-3, -0.01, 0.01),
        q1      = UniformPrior(0.0, 1.0),
        q2      = UniformPrior(0.0, 1.0),
    ),),
)
println("Free params ($(n_unfrozen(target.params))): max_kplanet = $(target.params.config.max_kplanet)")

# Trans-dim config — only PriorBirth (informed birth uses LS periodogram,
# which doesn't apply for transit search; would need BLS).
td = TransDimConfig(
    max_kplanet      = 3,
    planets          = true,
    noise            = false,
    birth_strategies = [PriorBirth()],
    birth_weights    = [1.0],
    transdim_fraction = 0.3,
)

println("\nRunning RJMCMC — 5000 warmup + 15000 samples ...")
t0 = time()
chains, n_evals = sample_rjmcmc(target, target.data;
                                td        = td,
                                n_samples = 15_000,
                                n_warmup  = 5_000,
                                seed      = 42)
elapsed = time() - t0
@printf("Done in %.1f min, %d likelihood evals\n\n", elapsed/60, n_evals)

# N_p posterior
np_chain = vec(Array(chains[:, :n_planets, :]))
println("N_p posterior:")
for k in 0:3
    frac = sum(np_chain .== k) / length(np_chain)
    @printf("  P(N_p = %d | data) = %.3f  (%d samples)\n", k, frac, sum(np_chain .== k))
end

# Conditional posteriors at N_p = 3
mask3 = np_chain .== 3
println("\nAt N_p = 3 ($(sum(mask3)) samples):")
for kp in 1:3
    P_v = vec(Array(chains[:, Symbol("P_k$kp"), :]))[mask3]
    if !isempty(P_v)
        @printf("  P_k%d   = %.5f d   [16/84: %.5f / %.5f]\n",
                kp, quantile(P_v, 0.5), quantile(P_v, 0.16), quantile(P_v, 0.84))
    end
end
