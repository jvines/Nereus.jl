# Synthetic photometry targets for the transit-window tests
# (test_transit_*.jl). Self-contained: the reference values the tests compare
# against are computed at test time, never stored.
#
#   * two instruments whose cadences are concatenated, not merged, so `t_phot` is
#     NOT sorted in time: a contiguous 2-min sector and three ground nights that
#     fall inside it;
#   * times in BJD, so the rounding of a time is that of a 2.46e6 number;
#   * a prior wide enough to reach every regime of the window: e from the whole
#     (sesinw, secosw) square, rho_s from 0.02 to 30 solar (a/R* about 2.5 to 30
#     at P = 4 d), b up to 1.3, rr up to 0.3.
#
# `exposure = true` gives the ground nights a 30-min exposure, which switches on
# finite-exposure supersampling and with it the non-workspace likelihood.
# `ttv = true` gives the planet free per-transit offsets (PM_TTV).

using Nereus
using Random

const TG_BJD0 = 2_460_000.0
const TG_TRUTH = (P = 4.137, Tc = TG_BJD0 + 1.31, b = 0.3, rr = 0.08, rho = 1.4)

function transit_gate_data(; exposure::Bool = false, seed::Int = 3)
    rng = MersenneTwister(seed)
    tA = collect(range(TG_BJD0, TG_BJD0 + 9.0; step = 2 / 1440))      # 2-min sector
    tB = Float64[]
    for c in (2.31, 5.12, 8.47)                                     # ground nights
        append!(tB, range(TG_BJD0 + c - 0.16, TG_BJD0 + c + 0.16; step = 5 / 1440))
    end
    t = vcat(tA, tB)
    inst = vcat(fill(1, length(tA)), fill(2, length(tB)))
    # A box-free smooth dip at the truth ephemeris: the flux values only have to
    # make the likelihood depend on the model, not to be a perfect transit.
    ph = @. mod(t - TG_TRUTH.Tc + TG_TRUTH.P / 2, TG_TRUTH.P) - TG_TRUTH.P / 2
    σ = [i == 1 ? 6e-4 : 1.2e-3 for i in inst]
    flux = @. 1.0 - 0.006 * exp(-0.5 * (ph / 0.04)^2) + σ * randn(rng)
    expo = exposure ? [i == 1 ? 120.0 : 1800.0 for i in inst] ./ 86_400 : Float64[]
    return Data(; t_phot = t, flux = flux, flux_err = σ, phot_inst = inst,
                  exposure_times = expo)
end

function transit_gate_target(; exposure::Bool = false, ttv::Bool = false, seed::Int = 3)
    data = transit_gate_data(; exposure, seed)
    ic = InstrumentConfig(pm = ["TESS", "GROUND"])
    priors = Dict{String, PriorSpec}(
        "P_k1"      => UniformPrior(3.9, 4.4),
        "Tc_k1"     => UniformPrior(TG_BJD0 + 0.9, TG_BJD0 + 1.7),
        "b_k1"      => UniformPrior(0.0, 1.3),
        "rr_k1"     => UniformPrior(0.005, 0.3),
        "sesinw_k1" => UniformPrior(-1.0, 1.0),
        "secosw_k1" => UniformPrior(-1.0, 1.0),
        "rho_s"     => LogUniformPrior(0.02, 30.0),
    )
    ttv_n = Dict{Int, Int}()
    if ttv
        ttv_n[1] = 3
        for i in 1:3
            priors["ttv_k1_t$i"] = UniformPrior(-0.08, 0.08)
        end
    end
    params = Params(; max_kplanet = 1, planet_modes = [ttv ? PM_TTV : PM_ONLY],
                      instruments = ic, data = data,
                      parametrization = ParametrizationConfig(time = :Tc, use_rho_s = true),
                      priors = priors, stability = :none, ttv_n_transits = ttv_n)
    return NereusTarget(params, data)
end

# `n` points drawn independently from the prior (bounded space), each a full
# `Theta` value vector, kept only when the log prior is finite (e < 1).
function transit_gate_points(target, n::Int; seed::Int = 11)
    rng = MersenneTwister(seed)
    params = target.params
    L = params.layout
    th = Nereus.Theta{Float64}(params)
    pts = Vector{Vector{Float64}}()
    while length(pts) < n
        for (j, idx) in enumerate(L.unfrozen_idx)
            th.values[idx] = Nereus.quantile(L.unfrozen_priors[j], rand(rng))
        end
        isfinite(Nereus.log_prior(th)) && push!(pts, copy(th.values))
    end
    return pts
end

transit_gate_ws(target) = Nereus.PTWorkspace(target.params, target.params.config.max_kplanet,
                                             length(target.params.config.noise_models);
                                             n_obs = length(target.data.t_rv),
                                             n_phot = length(target.data.t_phot))
