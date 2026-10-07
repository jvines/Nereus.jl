# Splitting the suite across processes. CI runs NEREUS_TEST_SHARD=i/n for
# i = 1..n in parallel (ci/run_tests.sh); unset, the default, runs everything,
# so `julia --project=. test/runtests.jl` is unchanged.
#
# A unit is a block of the suite that must run in one process, in order: a test
# file; files where a later one uses an earlier one's helpers; or the inline
# testsets of runtests.jl, where "foundation" and "trans-dim likelihood" draw
# from the same `_rng` stream and so stay together. `seconds` is its measured
# cost -- all the assignment needs: for the test files, the `unit time` lines of
# a five-shard run on the x86_64 CI runner (2026-10-06, release PR #18); the
# inline units keep older estimates, as they print no unit time. A unit missing
# a measurement counts as DEFAULT_SECONDS; a test file missing from this list is
# an error, so a new file cannot be skipped silently.
#
# Units are dealt longest-first to the least-loaded shard (LPT): deterministic,
# so a unit runs in the same shard on every run and every machine.
#
# SLOW_UNITS are full default-settings fits, each minutes on its own and 38% of
# the suite together. NEREUS_TEST_TIER=fast leaves them out (pull requests into
# develop); unset or "full" runs everything (release PRs, manual runs, local).

const DEFAULT_SECONDS = 5

const SLOW_UNITS = ("test_transdim_pt_emcee_defaults.jl", "test_activity_gp_lowrank.jl",
                    "test_obliquity_runjob.jl")

const _FAST_TIER = let t = get(ENV, "NEREUS_TEST_TIER", "full")
    t in ("fast", "full") || error("NEREUS_TEST_TIER must be \"fast\" or \"full\", got \"$t\"")
    t == "fast"
end

"""Whether unit `name` belongs to the tier being run."""
_in_tier(name) = !(_FAST_TIER && name in SLOW_UNITS)

const TEST_UNITS = [
    (name = "inline: foundation + trans-dim state/config/likelihood", seconds = 141, files = String[]),
    (name = "inline: photometry detrending", seconds = 159, files = String[]),
    (name = "inline: polynomial Stein discrepancy", seconds = 2, files = String[]),
    (name = "inline: sample_map physical-space mode", seconds = 49, files = String[]),
    (name = "inline: MoMS sampler", seconds = 1, files = String[]),
    (name = "inline: Daedalus sampler", seconds = 5, files = String[]),
    (name = "astrometry/test_data.jl", seconds = 1, files = ["astrometry/test_data.jl"]),
    (name = "astrometry/test_projection.jl", seconds = 2, files = ["astrometry/test_projection.jl"]),
    (name = "astrometry/test_reflex_kernel.jl", seconds = 7, files = ["astrometry/test_reflex_kernel.jl"]),
    (name = "astrometry/test_likelihood.jl", seconds = 2, files = ["astrometry/test_likelihood.jl"]),
    (name = "astrometry/test_sine_prior.jl", seconds = 1, files = ["astrometry/test_sine_prior.jl"]),
    (name = "astrometry/test_param_modes.jl", seconds = 2, files = ["astrometry/test_param_modes.jl"]),
    (name = "astrometry/test_m_pri.jl", seconds = 8, files = ["astrometry/test_m_pri.jl"]),
    (name = "astrometry/test_obs_prior.jl", seconds = 11, files = ["astrometry/test_obs_prior.jl"]),
    (name = "astrometry/test_ofti.jl", seconds = 19, files = ["astrometry/test_ofti.jl"]),
    (name = "astrometry/test_iad_gost.jl", seconds = 8, files = ["astrometry/test_iad_gost.jl"]),
    (name = "astrometry/test_iad_multi_instrument.jl", seconds = 10, files = ["astrometry/test_iad_multi_instrument.jl"]),
    (name = "astrometry/test_iad_marginalised_residuals.jl", seconds = 1, files = ["astrometry/test_iad_marginalised_residuals.jl"]),
    (name = "epoch astrometry orbit + trans-dim plot draws (shares a fixture)", seconds = 157, files = ["astrometry/test_epoch_astrometry_orbit.jl", "test_transdim_plot_draws.jl"]),
    (name = "astrometry/test_hgca_residual_plot.jl", seconds = 61, files = ["astrometry/test_hgca_residual_plot.jl"]),
    (name = "astrometry/test_astrometry_model_plot.jl", seconds = 90, files = ["astrometry/test_astrometry_model_plot.jl"]),
    (name = "astrometry/test_iad_multi_instrument_e2e.jl", seconds = 24, files = ["astrometry/test_iad_multi_instrument_e2e.jl"]),
    (name = "astrometry/test_gaia_epoch_guards.jl", seconds = 1, files = ["astrometry/test_gaia_epoch_guards.jl"]),
    (name = "test_builder.jl", seconds = 2, files = ["test_builder.jl"]),
    (name = "test_new_samplers.jl", seconds = 66, files = ["test_new_samplers.jl"]),
    (name = "test_runner_dispatch.jl", seconds = 137, files = ["test_runner_dispatch.jl"]),
    (name = "test_rm.jl", seconds = 9, files = ["test_rm.jl"]),
    (name = "test_rv_offset_priors.jl", seconds = 14, files = ["test_rv_offset_priors.jl"]),
    (name = "test_tomography.jl", seconds = 37, files = ["test_tomography.jl"]),
    (name = "test_tomography_framework.jl", seconds = 3, files = ["test_tomography_framework.jl"]),
    (name = "test_obliquity_framework.jl", seconds = 3, files = ["test_obliquity_framework.jl"]),
    (name = "test_tomo_noise_menu.jl", seconds = 1, files = ["test_tomo_noise_menu.jl"]),
    (name = "test_obliquity_workspace.jl", seconds = 39, files = ["test_obliquity_workspace.jl"]),
    (name = "test_obliquity_joint_framework.jl", seconds = 3, files = ["test_obliquity_joint_framework.jl"]),
    (name = "test_obliquity_target.jl", seconds = 65, files = ["test_obliquity_target.jl"]),
    (name = "test_obliquity_parity.jl", seconds = 12, files = ["test_obliquity_parity.jl"]),
    # Every sampler through run_job on an obliquity target; pt_hmc dominates.
    (name = "test_obliquity_runjob.jl", seconds = 487, files = ["test_obliquity_runjob.jl"]),
    (name = "test_as_coupling_mask.jl", seconds = 2, files = ["test_as_coupling_mask.jl"]),
    (name = "test_as_coupling_move.jl", seconds = 1, files = ["test_as_coupling_move.jl"]),
    (name = "test_obliquity_joint.jl", seconds = 104, files = ["test_obliquity_joint.jl"]),
    (name = "test_simulate_obliquity.jl", seconds = 24, files = ["test_simulate_obliquity.jl"]),
    (name = "test_gravity_darkening.jl", seconds = 25, files = ["test_gravity_darkening.jl"]),
    (name = "test_gd_workspace_likelihood.jl", seconds = 3, files = ["test_gd_workspace_likelihood.jl"]),
    (name = "test_informed_noise_birth.jl", seconds = 6, files = ["test_informed_noise_birth.jl"]),
    (name = "test_annealed_noise_birth.jl", seconds = 2, files = ["test_annealed_noise_birth.jl"]),
    (name = "test_solution_ladder.jl", seconds = 5, files = ["test_solution_ladder.jl"]),
    (name = "test_noise_swap_samplers.jl", seconds = 33, files = ["test_noise_swap_samplers.jl"]),
    (name = "test_ttv.jl", seconds = 5, files = ["test_ttv.jl"]),
    (name = "test_ppc.jl", seconds = 21, files = ["test_ppc.jl"]),
    (name = "test_detection_limits.jl", seconds = 37, files = ["test_detection_limits.jl"]),
    (name = "test_loo.jl", seconds = 10, files = ["test_loo.jl"]),
    (name = "test_fit_health.jl", seconds = 14, files = ["test_fit_health.jl"]),
    (name = "test_prior_dicts.jl", seconds = 2, files = ["test_prior_dicts.jl"]),
    (name = "test_relast_channel.jl", seconds = 5, files = ["test_relast_channel.jl"]),
    (name = "test_fit_summary_transdim.jl", seconds = 47, files = ["test_fit_summary_transdim.jl"]),
    # The two full default-settings fits, split so they run in different
    # shards. The trans-dim one is the single heaviest unit of the suite: LPT
    # gives it shard 1 to itself, and ci/run_tests.sh gives shard 1 more threads.
    (name = "test_pt_emcee_stranded.jl", seconds = 98, files = ["test_pt_emcee_stranded.jl"]),
    (name = "test_pt_emcee_resume.jl", seconds = 19, files = ["test_pt_emcee_resume.jl"]),
    (name = "test_pt_resume.jl", seconds = 43, files = ["test_pt_resume.jl"]),
    (name = "test_pt_hmc_resume.jl", seconds = 84, files = ["test_pt_hmc_resume.jl"]),
    (name = "test_pt_whitening_resume.jl", seconds = 9, files = ["test_pt_whitening_resume.jl"]),
    (name = "test_transdim_pt_emcee_resume.jl", seconds = 18, files = ["test_transdim_pt_emcee_resume.jl"]),
    (name = "test_rjmcmc_resume.jl", seconds = 10, files = ["test_rjmcmc_resume.jl"]),
    (name = "test_nuts_resume.jl", seconds = 159, files = ["test_nuts_resume.jl"]),
    (name = "test_ensemble_resume.jl", seconds = 7, files = ["test_ensemble_resume.jl"]),
    (name = "test_ess_resume.jl", seconds = 15, files = ["test_ess_resume.jl"]),
    (name = "test_moms_resume.jl", seconds = 16, files = ["test_moms_resume.jl"]),
    (name = "test_transdim_pt_emcee_defaults.jl", seconds = 641, files = ["test_transdim_pt_emcee_defaults.jl"]),
    (name = "test_circular.jl", seconds = 149, files = ["test_circular.jl"]),
    (name = "test_node_flip.jl", seconds = 90, files = ["test_node_flip.jl"]),
    (name = "test_lambda_slide.jl", seconds = 1, files = ["test_lambda_slide.jl"]),
    (name = "test_label_switching.jl", seconds = 6, files = ["test_label_switching.jl"]),
    (name = "test_pt_donor_buffer.jl", seconds = 1, files = ["test_pt_donor_buffer.jl"]),
    (name = "test_transdim_activity_columns.jl", seconds = 20, files = ["test_transdim_activity_columns.jl"]),
    (name = "test_activity_gp.jl", seconds = 19, files = ["test_activity_gp.jl"]),
    (name = "test_activity_gp_lowrank.jl", seconds = 532, files = ["test_activity_gp_lowrank.jl"]),
    (name = "test_indicator_floor_ws.jl", seconds = 6, files = ["test_indicator_floor_ws.jl"]),
    (name = "test_celerite_ws.jl", seconds = 4, files = ["test_celerite_ws.jl"]),
    (name = "test_rv_modifier_slots.jl", seconds = 4, files = ["test_rv_modifier_slots.jl"]),
    (name = "test_multiseries_gp.jl", seconds = 13, files = ["test_multiseries_gp.jl"]),
    (name = "test_multiseries_layout.jl", seconds = 6, files = ["test_multiseries_layout.jl"]),
    (name = "test_multiseries_typestab.jl", seconds = 6, files = ["test_multiseries_typestab.jl"]),
    (name = "test_parametric_noise.jl", seconds = 10, files = ["test_parametric_noise.jl"]),
    (name = "test_nightly_closed_form.jl", seconds = 6, files = ["test_nightly_closed_form.jl"]),
    (name = "test_harmonic_factor_index.jl", seconds = 1, files = ["test_harmonic_factor_index.jl"]),
    (name = "test_harmonic_external.jl", seconds = 1, files = ["test_harmonic_external.jl"]),
    (name = "test_transdim_caches.jl", seconds = 1, files = ["test_transdim_caches.jl"]),
    (name = "test_birth_death_reversibility.jl", seconds = 1, files = ["test_birth_death_reversibility.jl"]),
    (name = "test_alias_jump.jl", seconds = 1, files = ["test_alias_jump.jl"]),
    (name = "test_transdim_death_counters.jl", seconds = 7, files = ["test_transdim_death_counters.jl"]),
    (name = "test_locor.jl", seconds = 14, files = ["test_locor.jl"]),
    (name = "test_locor_io.jl", seconds = 5, files = ["test_locor_io.jl"]),
    (name = "test_lightcurve.jl", seconds = 3, files = ["test_lightcurve.jl"]),
    # The transit window and its cache (test/fixtures/transit_gate_target.jl).
    (name = "test_phot_n_super_cache.jl", seconds = 1, files = ["test_phot_n_super_cache.jl"]),
    (name = "test_transit_window_search.jl", seconds = 4, files = ["test_transit_window_search.jl"]),
    (name = "test_transit_window_tight.jl", seconds = 17, files = ["test_transit_window_tight.jl"]),
    (name = "test_transit_window_superset.jl", seconds = 38, files = ["test_transit_window_superset.jl"]),
    (name = "test_transit_occultation.jl", seconds = 22, files = ["test_transit_occultation.jl"]),
    (name = "test_transit_window_contact.jl", seconds = 27, files = ["test_transit_window_contact.jl"]),
    (name = "test_transit_flux.jl", seconds = 4, files = ["test_transit_flux.jl"]),
    # The bridge-sampling evaluator through the workspace likelihoods.
    (name = "test_bridge_workspace.jl", seconds = 14, files = ["test_bridge_workspace.jl"]),
    (name = "test_bridge_threading.jl", seconds = 3, files = ["test_bridge_threading.jl"]),
    # Evidence estimators that do not temper from the prior, and the Rajpaul
    # kernel gradient check: all three carried assertions but were not run
    # until they were listed.
    (name = "test_bridge_evidence.jl", seconds = 1, files = ["test_bridge_evidence.jl"]),
    (name = "test_reference_path_evidence.jl", seconds = 8, files = ["test_reference_path_evidence.jl"]),
    (name = "test_phot_determinism.jl", seconds = 1, files = ["test_phot_determinism.jl"]),
    (name = "test_phot_supersampling.jl", seconds = 1, files = ["test_phot_supersampling.jl"]),
    (name = "test_sampler_determinism.jl", seconds = 252, files = ["test_sampler_determinism.jl"]),
    (name = "test_bls_informed_phot.jl", seconds = 2, files = ["test_bls_informed_phot.jl"]),
    (name = "test_evidence_curved.jl", seconds = 6, files = ["test_evidence_curved.jl"]),
    (name = "test_evidence_headline.jl", seconds = 8, files = ["test_evidence_headline.jl"]),
    (name = "test_mode_laplace.jl", seconds = 6, files = ["test_mode_laplace.jl"]),
    (name = "verify_rajpaul_kernel_fd.jl", seconds = 1, files = ["verify_rajpaul_kernel_fd.jl"]),
    (name = "test_plot_corner_thinning.jl", seconds = 8, files = ["test_plot_corner_thinning.jl"]),
    (name = "test_plot_trace_thinning.jl", seconds = 3, files = ["test_plot_trace_thinning.jl"]),
    (name = "test_plot_labels.jl", seconds = 2, files = ["test_plot_labels.jl"]),
    (name = "test_science_table_labels.jl", seconds = 1, files = ["test_science_table_labels.jl"]),
    (name = "test_plot_patterns.jl", seconds = 1, files = ["test_plot_patterns.jl"]),
    (name = "test_ladder_history.jl", seconds = 80, files = ["test_ladder_history.jl"]),
    (name = "test_thread_slots.jl", seconds = 176, files = ["test_thread_slots.jl"]),
]

const _SHARD = let s = get(ENV, "NEREUS_TEST_SHARD", "")
    if isempty(s)
        nothing
    else
        m = match(r"^(\d+)/(\d+)$", s)
        m === nothing && error("NEREUS_TEST_SHARD must look like \"i/n\", got \"$s\"")
        i, n = parse(Int, m[1]), parse(Int, m[2])
        1 <= i <= n || error("NEREUS_TEST_SHARD: shard $i of $n does not exist")
        (i, n)
    end
end

"""Shard (1-based) each unit of the tier runs in, for `n` shards: longest-first to
the least loaded. Units outside the tier get no shard."""
function _assign_shards(n::Int)
    load = zeros(n)
    shard_of = Dict{String, Int}()
    units = [u for u in TEST_UNITS if _in_tier(u.name)]
    order = sortperm([u.seconds for u in units]; rev = true, alg = MergeSort)  # stable
    for j in order
        k = argmin(load)
        shard_of[units[j].name] = k
        load[k] += units[j].seconds
    end
    return shard_of, load
end

const _SHARD_OF = _SHARD === nothing ? nothing : first(_assign_shards(_SHARD[2]))

"""Whether unit `name` runs in this process."""
_in_shard(name) = _in_tier(name) && (_SHARD === nothing || _SHARD_OF[name] == _SHARD[1])

# Every test file must belong to a unit: a file added to test/ and not to this
# list would otherwise never run.
let listed = Set(f for u in TEST_UNITS for f in u.files)
    present = Set(relpath(joinpath(r, f), @__DIR__) for (r, _, fs) in walkdir(@__DIR__)
                  for f in fs if startswith(f, "test_") && endswith(f, ".jl"))
    push!(present, "verify_rajpaul_kernel_fd.jl")
    missing_ = setdiff(filter(f -> !occursin("validation", f), present), listed)
    isempty(missing_) || error("test files not in TEST_UNITS (test/shards.jl), so never " *
                               "run: " * join(sort!(collect(missing_)), ", "))
end

if _SHARD !== nothing
    let (shard_of, load) = _assign_shards(_SHARD[2])
        names_ = sort!([u.name for u in TEST_UNITS if get(shard_of, u.name, 0) == _SHARD[1]])
        tier = _FAST_TIER ? "fast tier, $(length(SLOW_UNITS)) slow units left out" : "full suite"
        @info "test shard $(_SHARD[1])/$(_SHARD[2]) ($tier): $(length(names_)) units, ~$(round(Int, load[_SHARD[1]])) s measured" units = names_
    end
end
