# The easy single-planet RV target of the pt_emcee default-settings tests:
# a 12.3 d, K = 12 m/s planet over a 1000 d baseline, as a run_job config with
# no parametrization and no sampler settings -- the reported case. Shared by
# test_pt_emcee_stranded.jl and test_transdim_pt_emcee_defaults.jl, which run in
# different CI shards (test/shards.jl) and so cannot share a variable.
function _easy_rv_cfg()
    rng = MersenneTwister(20260922)
    n = 60
    t = sort(2459000.0 .+ 1000.0 .* rand(rng, n))
    P, K, e, ω, tp = 12.3, 12.0, 0.1, 1.0, 2459003.0
    function _rv(ti)
        M = mod2pi(2π * (ti - tp) / P)
        E = M
        for _ in 1:50
            E -= (E - e * sin(E) - M) / (1 - e * cos(E))
        end
        ν = 2atan(sqrt((1 + e) / (1 - e)) * tan(E / 2))
        return K * (cos(ν + ω) + e * cos(ω))
    end
    rv = _rv.(t) .+ 2.0 .* randn(rng, n) .+ 5.0
    cfg = Dict{String,Any}(
        "seed" => 42,
        "output_dir" => mktempdir(),
        "data" => Dict("rv" => Dict("values" => Dict(
            "bjd" => t, "rv" => rv, "rv_err" => fill(2.0, n),
            "instrument" => fill("HARPS", n)))),
        "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"]),
        "sampler" => Dict("name" => "pt_emcee",
                          "kwargs" => Dict("show_progress" => false)),
        "output" => Dict("plots" => String[], "ppc" => false,
                         "detection_limits" => false, "loo" => false))
    return cfg
end
