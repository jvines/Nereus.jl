# The fit path, compiled when Nereus precompiles rather than in the first fit of
# every fresh process. A first `fit_rv`, `fit_transit`, `fit_astrometry`,
# `fit_joint` or `run_job` otherwise spends about two minutes compiling the
# target builder, the sampler, the post-fit stages (derived quantities, summary,
# diagnostics, tables) and the default outputs, whatever the run's size; a tiny
# fit through astronereus paid it too.
#
# The workload is what the daemon and the `fit_*` functions go through, each at
# a handful of steps of `pt_emcee`, post-fit included, with the outputs a fit
# writes by default:
#
#   - `fit_rv`: one planet, one instrument;
#   - `fit_transit`: a 300-point synthetic light curve (the transit model path);
#   - `fit_astrometry`: an inline HGCA block and a 40-scan inline IAD block, the
#     Hipparcos epoch modelled instantaneously (`hip_scans = false`), so nothing
#     is fetched;
#   - `fit_joint`: RV + transit;
#   - `run_job` on a minimal RV config.
#
# It reads and writes only inside a temporary directory, touches no network,
# and every random stream is seeded.
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

    # Transit and joint: a 3 d box transit over 9 d.
    rng2 = Random.MersenneTwister(11)
    P = 3.0
    tr = collect(range(0.0, 9.0; length = 300))
    ph = mod.(tr .+ 0.5P, P) .- 0.5P
    flux = [abs(x) < 0.07 ? 0.99 : 1.0 for x in ph] .+ 2e-4 .* randn(rng2, 300)
    phot = Dict("T" => (t = tr, flux = flux, flux_err = fill(2e-4, 300)))
    tt = collect(0.0:0.7:9.0)
    rvj = Dict("source" => "RV", "data" => Dict("A" => (
        t = tt, rv = 20.0 .* sin.(2π .* tt ./ P) .+ 1.5 .* randn(rng2, length(tt)),
        rv_err = fill(2.0, length(tt)))))
    phj = Dict("source" => "PM", "data" => phot)
    planets = (k1 = (P = UniformPrior(2.9, 3.1), K = UniformPrior(0.0, 100.0),
                     rr = UniformPrior(0.02, 0.3), b = UniformPrior(0.0, 1.0),
                     sesinw = 0.0, secosw = 0.0),)
    priors = Dict{String, PriorSpec}("P_k1" => UniformPrior(2.9, 3.1))
    engine_tr = Dict("engine" => "pt_emcee",
                     "options" => Dict("n_steps" => 100, "n_burnin" => 50, "n_walkers" => 16,
                                       "n_temps" => 3, "seed" => 1, "show_progress" => false))

    # Astrometry: HGCA values and Hipparcos-like scans, inline.
    cov = [0.0025 0.0; 0.0 0.0025]
    hgca = _parse_hgca_block(Dict(
        "epochs" => collect(mjd_epochs((1991.25, 2004.6, 2016.0))),
        "pmra" => [5.0, 4.95, 4.9], "pmdec" => [-3.0, -3.05, -3.1],
        "cov_ep" => [cov, cov, cov], "plx" => 25.0, "plx_err" => 0.05, "hip_id" => 0))
    na = 40
    ta = sort(48000.0 .+ 3000.0 .* rand(rng2, na))
    iad = _parse_iad_block(Dict("values" => Dict(
        "t" => ta, "abscissa" => 0.5 .* randn(rng2, na), "abscissa_err" => fill(0.5, na),
        "psi" => 2π .* rand(rng2, na),
        "parallax_factor" => 0.9 .* sin.(2π .* (ta .- 48000.0) ./ 365.25),
        "pm_factor" => (ta .- 49500.0) ./ 365.25)))
    return (; data, engine, cfg, phot, rvj, phj, planets, priors, engine_tr, hgca, iad)
end

# The workload's fits, in order. Called by the workload and by the suite.
function _fit_precompile_run(fx, dir)
    fit_rv(fx.data; engine = fx.engine, output_dir = joinpath(dir, "fit_rv"))
    fit_transit(fx.phot; time_anchor = :phic, priors = fx.priors, engine = fx.engine_tr,
                output_dir = joinpath(dir, "fit_transit"))
    fit_astrometry(; iad = fx.iad, hgca = fx.hgca, hip_scans = false, engine = fx.engine_tr,
                   output_dir = joinpath(dir, "fit_astrometry"))
    fit_joint(fx.rvj, fx.phj; planets = fx.planets, time_anchor = :phic,
              engine = fx.engine_tr, output_dir = joinpath(dir, "fit_joint"))
    cfg = copy(fx.cfg)
    cfg["output_dir"] = joinpath(dir, "run_job")
    run_job(cfg)
    return nothing
end

if _PRECOMPILE_FIT
    @setup_workload begin
        fx = _fit_precompile_inputs()
        @compile_workload begin
            dir = mktempdir()
            try
                Logging.with_logger(Logging.NullLogger()) do
                    redirect_stdio(; stdout = devnull, stderr = devnull) do
                        _fit_precompile_run(fx, dir)
                    end
                end
            finally
                rm(dir; recursive = true, force = true)
            end
        end
    end
end
