# The glob patterns `_dispatch_plot` publishes in `summary["plots"]`.
#
# Each branch of `_dispatch_plot` hand-writes the glob describing the files it
# just wrote. Nothing tied those strings to the writers, and five of them had
# drifted into naming files that no run has ever produced:
#
#   rv_phasefold          "models/rv_phased_K*.png"        (writer: RV_phasefold_K<k>)
#   pm_phasefold          "models/pm_phased_*_K*.png"      (writer: Transit_phasefold_K<k>_<ins>)
#   posteriors_raw        "posteriors/raw/*.png"           (writer: posteriors/raw/<group>/raw_<p>)
#   posteriors_parameters "posteriors/parameters/*.png"    (writer: .../<group>/param_<p>)
#   posteriors_histograms "posteriors/histograms/*.png"    (writer: .../<group>/hist_<p>)
#
# The first two were the wrong PREFIX, the last three the wrong DEPTH. All five
# were invisible because `summary["figures"]` is scanned off disk and stayed
# correct beside them.
#
# Two things are pinned here: the glob semantics (`*` must not cross `/`, which
# is the whole reason the depth bugs matched nothing), and every published
# pattern against a path list taken from a real run's figure tree.

using Nereus, Test

# Logical figure names exactly as `run_job` builds them: the path under
# <out_dir>/plots/ with the extension stripped. Taken from a real RV+astrometry
# run, plus the families that run did not exercise.
const _REAL_FIGS = [
    "models/rv_timeseries", "models/rv_components", "models/rv_sb2_timeseries",
    "models/RV_phasefold_K1", "models/RV_phasefold_K2",
    "models/Transit_phasefold_K1_TESS",
    "models/pm_timeseries_TESS",
    "models/rv_astrom_phasefold_K1", "models/orbit_skyplane_K1",
    "models/relastrom_timeseries_K1", "models/relastrom_residuals_K1",
    "models/iad_residuals", "models/epoch_astrometry_orbit_K1",
    "models/hgca_pm_residuals_K1",
    "models/g23h_residuals", "models/pm_anomaly_K1",
    "models/ttv_oc", "models/transit_overlay_K1", "models/rm_anomaly_K1",
    "corner", "activity_gp_latent", "activity_gp_decomposition",
    "transdim/occupancy",
    "traces/trace_planet_K1", "traces/trace_instrument", "traces/trace_other",
    "histograms/a_k1", "posteriors/plx", "posteriors/inference_plx",
    "posteriors/raw/planets/raw_M_sec_k1",
    "posteriors/raw/instrumental/raw_sigma_FIES",
    "posteriors/raw/other/raw_plx",
    "posteriors/parameters/planets/param_sesinw_k1",
    "posteriors/parameters/instrumental/param_sigma_FIES",
    "posteriors/parameters/other/param_plx",
    "posteriors/histograms/planets/hist_a_k1",
    "posteriors/histograms/instrumental/hist_gamma_NEID",
    "posteriors/histograms/other/hist_plx",
]

# Every pattern `_dispatch_plot` can return, as published in summary["plots"].
const _PUBLISHED = [
    "models/rv_timeseries.png", "models/rv_sb2_timeseries.png",
    "models/rv_components.png", "models/RV_phasefold_K*.png",
    "models/pm_timeseries_*.png", "models/Transit_phasefold_K*_*.png",
    "models/rv_astrom_phasefold_K*.png", "models/orbit_skyplane_K*.png",
    "models/relastrom_timeseries_K*.png", "models/relastrom_residuals_K*.png",
    "models/iad_residuals.png", "models/epoch_astrometry_orbit_K*.png",
    "models/hgca_pm_residuals_K*.png",
    "models/g23h_residuals.png", "models/pm_anomaly_K*.png", "models/ttv_oc.png",
    "models/transit_overlay_K*.png", "models/rm_anomaly_K*.png",
    "corner.png", "traces/*.png", "histograms/*.png", "posteriors/*.png",
    "transdim/occupancy.png", "posteriors/raw/*/*.png",
    "posteriors/parameters/*/*.png", "posteriors/histograms/*/*.png",
    "activity_gp_latent.png", "activity_gp_decomposition.png",
]

_matches(pat, path) = occursin(Nereus._glob_regex(replace(pat, r"\.png$" => "")), path)

@testset "summary[\"plots\"] patterns" begin

    # The semantics that made the depth bugs invisible.
    @testset "* does not cross a path separator" begin
        @test !_matches("posteriors/raw/*.png", "posteriors/raw/planets/raw_a_k1")
        @test  _matches("posteriors/raw/*/*.png", "posteriors/raw/planets/raw_a_k1")
        @test  _matches("traces/*.png", "traces/trace_planet_K1")
        @test !_matches("traces/*.png", "traces/planets/trace_K1")
        # a `*` must not swallow the rest of a longer name at the same level
        @test  _matches("models/pm_timeseries_*.png", "models/pm_timeseries_TESS")
        @test !_matches("models/rv_timeseries.png", "models/rv_timeseries_extra")
    end

    # Regex metacharacters in a pattern must be literals, not operators.
    @testset "literal characters are escaped" begin
        @test  _matches("models/ttv_oc.png", "models/ttv_oc")
        @test !_matches("models/ttv_oc.png", "models/ttvXoc")   # `.` is not "any"
    end

    # THE REGRESSION: every published pattern must name something real.
    @testset "every published pattern matches a real figure" begin
        unmatched = [p for p in _PUBLISHED
                     if !any(f -> _matches(p, f), _REAL_FIGS)]
        @test isempty(unmatched)
    end

    # The five that were broken, pinned individually so a regression is named.
    @testset "previously-stale patterns" begin
        @test _matches("models/RV_phasefold_K*.png", "models/RV_phasefold_K1")
        @test _matches("models/Transit_phasefold_K*_*.png",
                       "models/Transit_phasefold_K1_TESS")
        @test _matches("posteriors/raw/*/*.png", "posteriors/raw/other/raw_plx")
        @test _matches("posteriors/parameters/*/*.png",
                       "posteriors/parameters/planets/param_sesinw_k1")
        @test _matches("posteriors/histograms/*/*.png",
                       "posteriors/histograms/planets/hist_a_k1")
        # the exact strings that used to be published, still matching nothing
        for stale in ("models/rv_phased_K*.png", "models/pm_phased_*_K*.png",
                      "posteriors/raw/*.png", "posteriors/parameters/*.png",
                      "posteriors/histograms/*.png")
            @test !any(f -> _matches(stale, f), _REAL_FIGS)
        end
    end

    # The runtime guard that reports drift instead of publishing it silently.
    @testset "unmatched patterns are reported" begin
        figs = Dict{String, Any}(f => "/tmp/$f.png" for f in _REAL_FIGS)
        @test Nereus._warn_unmatched_plot_patterns(_PUBLISHED, figs) === nothing
        @test_logs (:warn,) Nereus._warn_unmatched_plot_patterns(
            ["posteriors/raw/*.png"], figs)
        # nothing rendered at all ⇒ nothing to check, and no warning
        @test_logs Nereus._warn_unmatched_plot_patterns(_PUBLISHED, Dict{String,Any}())
    end
end
