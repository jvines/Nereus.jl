# The figures, compiled when Nereus precompiles rather than in the first fit
# that plots -- for a release build. A one-planet, two-instrument RV fit and a
# small chain around it go through the batch plot path (`_make_plots`, `auto`):
# the corner, the posterior and trace figures and the RV figures, and with
# them the Makie and Cairo machinery every other figure shares. In a fresh
# process the first plot set otherwise compiles for about two minutes and
# holds the compiler's memory on top of the figures'.
#
# It adds about a minute to every precompilation, so it is off unless the
# `precompile_plots` preference is set -- by the release bundle, in the
# environment it precompiles:
#
#     using Preferences, Nereus
#     set_preferences!(Nereus, "precompile_plots" => true)
#
# (a `LocalPreferences.toml` next to the project, `[Nereus]`
# `precompile_plots = true`, does the same). A change of the preference
# recompiles Nereus.
using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference

const _PRECOMPILE_PLOTS = @load_preference("precompile_plots", false)

if _PRECOMPILE_PLOTS
    @setup_workload let
        n = 24
        t = collect(range(0.0, 300.0; length = n))
        inst = [isodd(i) ? 1 : 2 for i in 1:n]
        data = Data(; t_rv = t, rv = zeros(n), rv_err = fill(2.0, n), rv_inst = inst)
        params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                           instruments = InstrumentConfig(rv = ["A", "B"]),
                           data = data, M_s = 1.0)
        vals = Dict("P_k1" => 11.3, "K_k1" => 25.0, "sesinw_k1" => 0.2,
                    "secosw_k1" => -0.1, "Mo_k1" => 0.7, "gamma_A" => 0.0,
                    "gamma_B" => 3.0, "sigma_A" => 2.0, "sigma_B" => 2.0)
        nm = params.layout.unfrozen_names
        rng = Random.MersenneTwister(1)
        n_iter, n_walk = 250, 8
        arr = zeros(n_iter, length(nm) + 1, n_walk)
        for w in 1:n_walk, i in 1:n_iter
            z2 = 0.0
            for (j, x) in enumerate(nm)
                z = randn(rng)
                z2 += z^2
                arr[i, j, w] = vals[x] + 0.01 * max(abs(vals[x]), 0.1) * z
            end
            arr[i, end, w] = -0.5 * z2
        end
        chains = MCMCChains.Chains(arr, vcat(Symbol.(nm), [:lp]))
        theta = Theta{Float64}(params)
        for (k, v) in vals
            set_param!(theta, k, v)
        end
        data.rv .= rv_predictions(theta, data)[1] .+ 2.0 .* randn(rng, n)
        cfg = Dict("output" => Dict("plots" => ["auto"], "show_progress" => false))
        @compile_workload begin
            dir = mktempdir()
            try
                Logging.with_logger(Logging.NullLogger()) do
                    _make_plots(cfg, chains, params, data, dir)
                end
            finally
                rm(dir; recursive = true, force = true)
            end
        end
        # The font probe's answer is the build machine's; probe again at run time.
        _SERIF_FAMILY[] = nothing
    end
end
