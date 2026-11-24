# The fit path, compiled when Nereus precompiles rather than in the first fit of
# every fresh process. A first `fit_rv` or `run_job` otherwise spends about two
# minutes compiling the target builder, the sampler, the post-fit stages (derived
# quantities, summary, diagnostics, tables) and the default outputs, whatever
# the run's size; a tiny fit through astronereus paid it too.
#
# The workload is what the daemon and the `fit_*` functions go through: a tiny
# RV fit through `fit_rv` -> `_finish` with `pt_emcee` at a handful of steps
# (post-fit included, with the outputs that fit writes by default), and `run_job`
# on a minimal config. It reads and writes only inside a temporary directory,
# touches no network, and every random stream is seeded.
#
# It is on by default. To switch it off (a faster `Pkg.precompile()` while
# developing, at the price of the first fit compiling), set the preference and
# let Nereus recompile:
#
#     using Preferences, Nereus
#     set_preferences!(Nereus, "precompile_fit" => false)
#
# or put `[Nereus]` / `precompile_fit = false` in a `LocalPreferences.toml` next
# to the project. The figures are a separate workload (`precompile_plots`,
# src/plotting/precompile.jl).
using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference

const _PRECOMPILE_FIT = @load_preference("precompile_fit", true)

# The workload's inputs. A function, as `_plot_precompile_fixture` is, so the
# suite can run exactly what precompilation runs.
function _fit_precompile_inputs()
    rng = Random.MersenneTwister(7)
    n = 24
    t = sort(2459000.0 .+ 200.0 .* rand(rng, n))
    rv = 10.0 .* sin.(2π .* (t .- 2459000.0) ./ 11.3) .+ 5.0 .+ 2.0 .* randn(rng, n)
    data = Dict("HARPS" => (t = t, rv = rv, rv_err = fill(2.0, n)))
    engine = Dict("engine" => "pt_emcee",
                  "options" => Dict("n_steps" => 40, "n_burnin" => 20, "n_walkers" => 12,
                                    "n_temps" => 3, "seed" => 1, "show_progress" => false))
    cfg = Dict{String, Any}(
        "seed" => 1,
        "data" => Dict("rv" => Dict("values" => Dict(
            "bjd" => t, "rv" => rv, "rv_err" => fill(2.0, n),
            "instrument" => fill("HARPS", n)))),
        "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"]),
        "sampler" => Dict("name" => "pt_emcee",
                          "kwargs" => Dict("show_progress" => false, "n_steps" => 40,
                                           "n_burnin" => 20, "n_walkers" => 12,
                                           "n_temps" => 3)))
    return (; data, engine, cfg)
end

if _PRECOMPILE_FIT
    @setup_workload begin
        fx = _fit_precompile_inputs()
        @compile_workload begin
            dir = mktempdir()
            try
                Logging.with_logger(Logging.NullLogger()) do
                    redirect_stdio(; stdout = devnull, stderr = devnull) do
                        fit_rv(fx.data; engine = fx.engine, output_dir = joinpath(dir, "fit_rv"))
                        cfg = copy(fx.cfg)
                        cfg["output_dir"] = joinpath(dir, "run_job")
                        run_job(cfg)
                    end
                end
            finally
                rm(dir; recursive = true, force = true)
            end
        end
    end
end
