# Shared AD↔GP-Rot fixture. `include` this; do not copy it.
#
# This file exists because the same class of bug bit three times in one day.
# Each script that reconstructed "the same" dataset by hand got a DIFFERENT one:
#
#   * toy_ad_vs_gprot_swap.jl hardcoded a tracer level intended as a near-tie
#     that actually measures dlogZ = +13.7 — unmeasurable, so the gate reported
#     FAIL whatever the sampler did;
#   * substituting the sibling's level (2.0) into it flipped the gap's SIGN,
#     because the two scripts seed the TIME sampling differently;
#   * xcheck_gprot_evidence.jl rebuilt the data with MersenneTwister(20260619)
#     for the times instead of MersenneTwister(7) and measured dlogZ = -6.91
#     against this file's +1.62 — a comparison that looked like a contradiction
#     and was simply a different dataset.
#
# The evidence gap is an emergent property of the exact noise realisation, so
# ANY script comparing occupancy against a reference must use the identical
# data or it is not comparing anything.
using Nereus, MCMCChains, Statistics, Printf, Random

const N     = 70
const SIG   = 1.5
const P_ROT = 8.0
const TT    = (rng = MersenneTwister(7); sort(rand(rng, N)) .* 80.0)

function make_data(bis_noise; seed = 20260619)
    rng = MersenneTwister(seed)
    amp = 5.0 .* (1.0 .+ 0.3 .* sin.(2π .* TT ./ 40.0))
    act = amp .* (sin.(2π .* TT ./ P_ROT) .+ 0.4 .* sin.(4π .* TT ./ P_ROT .+ 0.5))
    rv  = act .+ SIG .* randn(rng, N)
    bis = act .+ bis_noise .* randn(rng, N)
    Data(t_rv = TT, rv = rv, rv_err = fill(SIG, N), rv_inst = ones(Int, N),
         indicators = Dict("bis" => bis),
         indicator_errs = Dict("bis" => fill(1.0, N)))
end

const AD  = ActivityDecorrelation(indicators = ["bis"])
const ROT = CeleriteRotation(channel = :rv)

# bis_noise = 2.0 is the level the scan settled on: a NON-SATURATED gap, small
# enough that the disfavoured member is visited often enough to measure.
const BIS_NOISE = 2.0

mkp(data, nms; td) = (rvmax = maximum(abs, data.rv);
    Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
        instruments = InstrumentConfig(rv = ["I1"]), data = data,
        parametrization = ParametrizationConfig(time = :Mo),
        priors = Dict{String,PriorSpec}(
            "gamma_I1" => UniformPrior(-3rvmax, 3rvmax),
            "sigma_I1" => LogUniformPrior(0.2, 12.0)),
        noise_models = nms, transdim_noise = td, stability = :none))

"""
    reference_dlogZ(data) -> (Δ, σ_est, σ_seed, detail)

Reference ΔlogZ(AD − GP-Rot) with an HONEST uncertainty.

`σ_seed`, the seed-to-seed scatter, is what these gates used to quote. It is the
wrong quantity: it measures reproducibility, not how well the estimator knows a
GP marginal likelihood. Measured side by side, GP-Rot's own reported error is
1.59 nats while its 3-seed scatter is 0.11 — a factor of ~14. Every gate that
flagged a ~1.6-nat "discrepancy" was inside 1 sigma of its own reference error
and did not know it.

`σ_est` combines the estimators' self-reported errors in quadrature, which is
the bar an occupancy discrepancy actually has to clear.
"""
function reference_dlogZ(data; nseed = 3, estimator = :hybrid)
    za = Float64[]; zr = Float64[]; ea = Float64[]; er = Float64[]
    for sd in 1:nseed
        for (nms, zs, es) in ((NoiseModel[AD], za, ea), (NoiseModel[ROT], zr, er))
            r = sample_ptemcee(NereusTarget(mkp(data, nms; td = false), data), data;
                    n_temps = 14, n_walkers = 60, n_steps = 9000, n_burnin = 3500,
                    seed = sd, init_strategy = :prior, show_progress = false)
            v, e = getfield(r.evidence, estimator)
            push!(zs, v); push!(es, e)
        end
    end
    Δ       = mean(za) - mean(zr)
    σ_est   = sqrt(mean(ea)^2 + mean(er)^2)
    σ_seed  = sqrt(std(za)^2 + std(zr)^2) / sqrt(nseed)
    return Δ, σ_est, σ_seed, (za = za, zr = zr, ea = ea, er = er)
end
