module Nereus

include("constants.jl")
include("kepler.jl")
include("orbit.jl")
include("transit.jl")
include("priors.jl")
include("packed_priors.jl")
include("astrometry/data.jl")  # RelAstromData, HGCAData (no deps; needed by Data struct)
include("tomography_data.jl") # TomoNight (needed by Data)
include("data.jl")           # Data struct (no deps on model.jl)
include("noise/types.jl")    # abstract types + concrete structs
include("model.jl")          # ParamsConfig, ParamsLayout, Params (needs Data for constructor)
include("noise/param_names.jl")  # noise_param_names (needs InstrumentConfig)
include("default_priors.jl")    # default_priors (needs model.jl + priors.jl + Data)
include("params_constructor.jl") # Params() constructor (needs default_priors + Data)
include("transdim/state.jl")
include("transdim/config.jl")
include("parameters.jl")
include("transdim/proposals.jl")
include("transdim/informed_phot.jl")  # BLS + M-R + K helpers for JointInformedBirth
include("transdim/birth_strategies.jl")
include("noise/activity.jl")  # needs Theta from parameters.jl
include("noise/arma.jl")
include("noise/gp.jl")       # needs Theta; defines gp_log_likelihood
include("noise/activity_gp.jl")  # Rajpaul+ 2015 multivariate-GP kernel + LL
include("noise/multiseries_gp.jl")  # O(N·r²) semiseparable multi-series activity GP (needs sho_coefficients)
include("noise/parametric_noise.jl")  # ErrorScale / NightlyOffset / HarmonicBlock / MaternGP (needs celerite_solve, _gp_suffix)
include("noise/noise_menu.jl")        # default_noise_menu (needs all noise structs)
include("derived.jl")        # stability + derived quantities (needs constants, transit, model)
include("astrometry/projection.jl")  # PlanetOrbits-backed sky projection (needs Theta + parameters.jl)
include("astrometry/likelihood.jl")  # rel + HGCA log-likelihoods (needs projection + Theta)
include("astrometry/loaders.jl")     # FITS / orvara-format loaders for catalogs
include("astrometry/fetchers.jl")    # ESA bundle + GOST scraper (cached)
include("astrometry/gaia_epoch.jl")  # Gaia DR4 epoch-astrometry reader → IADData (needs IADData)
include("astrometry/solution_ladder.jl")  # exact 5/7/9-param evidences (needs IADData)
include("likelihood.jl")     # calls gp_log_likelihood, check_stability, astrom_log_likelihood
include("transit_likelihood.jl")
include("reporting.jl")
include("science_tables.jl")  # model-conditioned, unit-tagged science tables (needs reporting.jl)
include("transforms.jl")
include("target.jl")
include("builder.jl")     # high-level build_target convenience over Params + Data + NereusTarget
include("loaders_external.jl")   # TESS LC + Vizier RV CSV loaders (data prep happens in Python sidecar; Julia just ingests)
include("progress.jl")          # ProgressBar utility used by every sampler
include("juliacall_compat.jl") # _is_under_juliacall + @maybe_threaded — gate every Julia threading construct so production workers don't deadlock under the Python GIL
include("samplers/nuts.jl")
include("samplers/rjmcmc.jl")
include("samplers/moms.jl")     # MoMS sampler — needs rjmcmc helpers
include("samplers/moms_ns.jl")  # MoMS-NS trans-dim nested sampler
include("samplers/evidence.jl")  # reddemcee-style TI+/SS+/H+ — used by PT
include("samplers/bridge.jl")           # posterior-only evidence (bridge sampling)
include("samplers/reference_path.jl")   # evidence via a path from a fitted reference
include("samplers/pt.jl")
include("samplers/pt_hmc.jl")  # Hamiltonian parallel tempering (fixed-dim, differentiable)
include("samplers/nested.jl")
include("samplers/ensemble.jl")
include("samplers/map.jl")
include("samplers/ess.jl")
include("samplers/ofti.jl")
include("samplers/pathfinder.jl")
include("samplers/ptemcee.jl") # parallel-tempered ensemble (Vousden+ 2016)
include("samplers/transdim_ptemcee.jl") # trans-dim PT + MoMS variable selection
include("samplers/population_annealing.jl") # sequential-MC tempered sampler (Hukushima & Iba 2003)
include("samplers/nested_ins.jl")            # importance nested sampling (Feroz+ 2013)
include("samplers/nested_dynamic.jl")        # dynamic nested sampling (Higson+ 2019)
include("samplers/smc.jl")                   # SMC sampler interface (Del Moral+ 2006); delegates to sample_pa
include("samplers/pt_whitening.jl")          # NF-coupled PT with diagonal-Gaussian flow (Arbel+ 2021 lite)
include("io.jl")
include("diagnostics/psd.jl")  # Polynomial Stein Discrepancy (needs NereusTarget)
include("diagnostics/ttv_measure.jl")  # per-transit Tc + TTV-C envelope from chains
include("preprocessing/detrend.jl")
include("preprocessing/notch.jl")
include("preprocessing/find_transits.jl")
include("preprocessing/transit_window.jl")  # near-transit masking (native cadence, not binning)
include("preprocessing/rotation_period.jl")
include("preprocessing/locor.jl")  # LOCoR detrender (needs find_segments, find_rotation_period, detrend_notch)
include("preprocessing/lightcurve.jl")  # LightCurve container + clean_lightcurve (needs all detrend_*)
include("periodogram/periodogram.jl")
include("diagnostics/ppc.jl")  # posterior predictive checks (needs Theta + GLSPgram + rv_predictions)
include("diagnostics/detection_limits.jl")  # Bayesian K_lim(P) from posterior chains
include("diagnostics/loo.jl")               # PSIS-LOO / WAIC via chain replay
include("diagnostics/fit_health.jl")        # assess_fit — post-fit "silent-wrong" guard (needs MCMCChains; MAPResult is duck-typed)
include("diagnostics/label_switching.jl")   # per-component slot-permutation entropy for exchangeable-slot runs (needs MCMCChains)
include("diagnostics/coherence.jl")         # SHO-Q coherence discriminant (candidate-vetting test 4; needs celerite primitives)
include("diagnostics/candidate_gate.jl")    # candidate promotion gate (conjunction of vetoes; needs periodogram + coherence)
include("plotting/theme.jl")
include("plotting/utils.jl")
include("plotting/rv_plots.jl")
include("plotting/pm_plots.jl")
include("plotting/diagnostics.jl")
include("plotting/detrend_plots.jl")
include("plotting/astrom_plots.jl")
include("plotting/ttv_plots.jl")
include("plotting/ppc_plots.jl")
include("plotting/detection_limits_plots.jl")
include("plotting/posterior_plots.jl")
include("plotting/posterior_science_plots.jl")  # raw/parameters/histograms posteriors
include("plotting/transdim_plots.jl")
include("plotting/activity_gp_plots.jl")
include("preprocessing/rotation_period_plots.jl")
include("rm.jl")                   # Rossiter-McLaughlin (Hirano+ 2011 lite)
include("gravity_darkening.jl")    # von Zeipel gravity-darkened transits (Barnes 2009)
include("tomography.jl")           # Doppler tomography: the planet shadow in the line profile
include("obliquity_joint.jl")      # joint shadow + RM-velocity obliquity fit (bespoke sampler)
include("simulate_obliquity.jl")   # forward simulation of tomographic and RM nights
include("obliquity.jl")            # framework-native obliquity fitting (Params/Theta + noise menu)
include("plotting/rm_plots.jl")    # RM in-transit RV anomaly figure
include("ttv.jl")                  # Per-transit free time offsets (TTV-A)
include("ttv_nbody.jl")            # N-body-predicted TTVs via TTVFaster (TTV-C)
include("ttv_nbody_full.jl")       # full ODE backend via NbodyGradient.jl
include("runner.jl")               # JSON/dict-driven batch entry point
include("api.jl")                  # split-by-functionality public API (fit_*)
include("features.jl")             # capability registry behind the API

# Public API — everything else lives in submodules or is internal.
export
    # Doppler tomography (src/tomography.jl)
    ccf_profile, tomogram_residuals, shadow_track, tomogram_matched_filter,
    tomogram_null_distribution, tomogram_injection_test, tomogram_pooled,
    tomogram_pulsation_filter, TomoNight, kron_gp_loglike, shadow_map,
    tomogram_logpost, tomogram_bayes, circular_summary, shadow_bayes_factor,
    tomogram_log_likelihood, white_map_loglike,
    # Joint obliquity fit (src/obliquity_joint.jl)
    RMNight, rm_anomaly, joint_obliquity_logpost, joint_obliquity_fit,
    rm_velocity_fit,
    # Forward simulation (src/simulate_obliquity.jl)
    simulate_tomogram, simulate_rm_night,
    # Kepler / orbit
    kepler_solve, kepler_solve!,
    true_anomaly,
    rv_keplerian, rv_keplerian_sum,
    sesinw_to_ew, esinw_to_ew,
    ew_to_sesinw, ew_to_esinw,
    mo_to_tp, tp_to_mo, tc_to_tp, tp_to_tc,
    # Transit
    transit_flux, transit_flux_uniform,
    kipping_q_to_u, kipping_u_to_q,
    # Priors
    AbstractPrior, PriorSpec, Fixed, ModJeffreys,
    UniformPrior, NormalPrior, LogUniformPrior,
    BetaPrior, ModJeffreysPrior, FixedPrior, SinePrior,
    bounds, is_fixed, fixed_value, in_support,
    prior_transform, prior_transform!,
    logpdf_sum, prior_to_dict, prior_from_dict,
    # Physical validation
    PHYSICAL_BOUNDS, extract_base_name, physical_bounds, validate_physical,
    # Model configuration
    ExternalPrior,
    PlanetDataSources, has_rv, has_pm, has_as, has_rm, has_ttv,
    RV_ONLY, PM_ONLY, RVPM, RVAS, RVPMAS, RVPM_RM, RVPMAS_RM,
    RVPM_TTV, RVPMAS_TTV, RVPM_RM_TTV, PM_TTV,
    RVPM_TTV_NB, RVPMAS_TTV_NB, PM_TTV_NB,
    RVPM_RM_R, RVPMAS_RM_R,
    RVPM_RM_A, RVPMAS_RM_A,
    SB_SOURCE, BINARY, BINARY_RV, has_sb,
    has_ttv_nb, has_rm_r, has_rm_a, has_any_rm,
    rm_reloaded_signal,
    ParametrizationConfig, InstrumentConfig, SystemicIndices,
    PlanetBlock, RVOnlyBlock, PMOnlyBlock, RVPMBlock,
    RVASBlock, RVPMASBlock, SB2Block, is_sb2,
    planet_K_A, planet_K_B,
    has_K, has_geometry, has_AS,
    ParamsConfig, ParamsLayout, Params,
    n_total, n_unfrozen, n_frozen, param_index,
    n_rv_instruments, n_pm_instruments,
    # Runtime theta state
    Theta,
    set_unfrozen!, unfrozen_values, unfrozen_values!,
    get_param, set_param!,
    planet_P, planet_K, planet_e_w, planet_time_anchor, planet_b_rr,
    planet_a, planet_M_sec,
    planet_inc, planet_Omega, astrom_plx, astrom_M_pri,
    system_vsini, system_f_light,
    n_p, set_n_p!, planet_indices, is_noise_model_active,
    rv_gamma, rv_sigma, rv_dvdt, rv_d2vdt2,
    pm_offset, pm_jitter, pm_dilution,
    rho_s,
    log_prior,
    # Data container + likelihood + target
    Data, n_rv, n_phot, has_astrometry,
    RelAstromData, HGCAData, IADData, GOSTData, G23HData, GaiaDR3Data,
    n_relast, n_iad, n_gost, n_g23h,
    jyear_to_mjd, mjd_epochs,
    relastrom_log_likelihood, hgca_log_likelihood,
    iad_log_likelihood, gost_log_likelihood,
    g23h_log_likelihood,
    along_scan_projection, astrom_log_likelihood,
    obs_based_log_prior,
    mass_function, msec_from_K, a_from_P,
    relastrom_offset, star_reflex_offset, star_reflex_pm,
    gost_window_avg_pm, gost_5param_fit,
    load_hgca_row, load_orvara_rv, load_orvara_relast,
    load_hip_iad, load_gost, load_gaia_dr3,
    fetch_hip_iad, fetch_gost,
    load_tess_lc, load_vizier_rv, export_lightcurve,
    rv_log_likelihood, rv_predictions,
    transit_log_likelihood, phot_predictions,
    sky_separation, rho_s_to_a_Rs,
    build_transform,
    NereusTarget, build_target,
    # Derived parameters
    msini, planet_mass, planet_radius, planet_radius_earth,
    planet_density, planet_density_earth,
    inclination_deg, semimajor_axis_au, semimajor_axis_from_rho,
    stellar_mass_from_density,
    equilibrium_temperature, incident_flux, incident_flux_earth,
    tsm, esm, is_amd_stable, is_gladman_stable, check_stability,
    DerivedParams, compute_derived, conditional_gamma,
    # Reporting
    ParamStats,
    summarize_fitted, summarize_derived,
    print_results, save_results,
    compute_model_stats,
    print_transdim_summary, save_transdim_summary,
    print_ess_rhat,
    polynomial_stein_discrepancy, psd_bootstrap_test, print_psd,
    # Noise models
    NoiseModel, MeanModifier, SequentialNoise, CovarianceNoise, AdditiveCovariance,
    ActivityDecorrelation, ActivityJitter, MAModel, ARModel,
    CeleriteSHO, CeleriteRotation, CeleriteRotationFM17, MaternGP,
    ErrorScale, NightlyOffset, HarmonicBlock, StudentT,
    default_noise_menu,
    ActivityGP, ActivityGPCoeffs, IndicatorFloor,
    activity_kernel_blocks, activity_kernel_blocks_fm17,
    activity_kernel_blocks_fm17_celerite,
    activity_gp_covariance, activity_gp_log_likelihood,
    activity_gp_predict, activity_gp_decompose_rv,
    plot_activity_gp_latent, plot_activity_gp_decomposition,
    noise_channel, noise_instruments,
    validate_noise_models,
    # Trans-dimensional
    TransDimState, TransDimConfig, LikelihoodCounter,
    BirthStrategy, PriorBirth, InformedBirth, JointInformedBirth, DonorBirth, MoMSBirth,
    spike_slab_log_prior,
    activate_planet!, deactivate_planet!, active_planets,
    n_active_planets, first_inactive_planet,
    activate_noise!, deactivate_noise!, active_noise, is_noise_active,
    activate_as!, deactivate_as!, is_as_active, is_planet_as_active,
    propose_as_birth, propose_as_death,
    propose_planet_birth, propose_planet_death,
    propose_planet_alias_jump,
    propose_noise_birth, propose_noise_death,
    reset_transdim_caches!,
    rjmcmc_accept,
    # Progress display
    ProgressBar,
    # Samplers
    sample_nuts, nuts_diagnostics,
    sample_pt, sample_pt_warm, sample_pt_hmc,
    EvidenceAccumulator, EvidenceReport,
    update_evidence!, evidence_report,
    ti_trapezoidal, ti_plus, ss_plus, hybrid_evidence,
    sample_rjmcmc, sample_moms, sample_moms_ns,
    model_probabilities, bayes_factors,
    sample_nested,
    sample_ensemble,
    sample_map, MAPResult,
    sample_ess,
    ofti_sample,
    pathfinder_init,
    pathfinder_warmstart_moms_ns,
    sample_ptemcee, PTemceeResult,
    sample_transdim_ptemcee, TransDimPTemceeResult,
    run_job,
    rm_signal, rm_signal_arome, rm_signal_at_time, planet_sky_position,
    sample_pa, PAResult,
    sample_nested_ins, INSResult,
    sample_nested_dynamic, DynamicNSResult,
    sample_smc,
    sample_pt_whitening, WhiteningPTResult,
    # I/O
    save_chains, load_chains,
    save_lightcurve, load_lightcurve,
    save_detrend_result,
    save_periodogram_set, save_acf_result,
    save_detection_limits,
    # Photometry detrending (preprocessing — optional)
    mask_transits, find_segments,
    detrend_savgol, detrend_gp, detrend_notch, detrend_locor, detrend_rotation,
    LightCurve, clean_lightcurve, trim_lightcurve, sectors,
    sigma_clip_mask, sigma_clip_lightcurve,
    find_transits,
    transit_mask,
    window_to_transits,
    radius_from_depth, chen_kipping_mass, K_estimate,
    find_rotation_period, find_rotation_period_sectors,
    plot_acf, plot_acf_sectors, plot_pacf_sectors,
    # Periodograms
    Periodogram, GLSPgram, BGLSPgram, SBGLSPgram, L1Pgram, PgramPeak, PgramSet,
    GLSSectorSet,
    gls_periodogram, bgls_periodogram, sbgls_periodogram, l1_periodogram_compute,
    find_rv_planets, gls_rotation,
    plot_periodogram, plot_periodograms_stacked, plot_gls_sectors_stacked,
    plot_sbgls_heatmap,
    # Plotting
    nereus_theme,
    plot_rv_timeseries, plot_rv_phasefold,
    plot_pm_timeseries, plot_pm_phasefold,
    plot_trace, plot_posteriors, plot_histograms, plot_corner,
    plot_detrending, plot_transit_phasefold, plot_posteriors_lp,
    plot_orbit_skyplane, plot_pm_residuals, plot_rv_astrom_phasefold,
    plot_iad_residuals, plot_epoch_astrometry_orbit,
    plot_relastrom_timeseries, plot_relastrom_residuals,
    plot_g23h_residuals, plot_pm_anomaly,
    plot_ttv_diagram, plot_ttv_diagram_multipanel, plot_transit_overlay,
    plot_ttv_oc,
    plot_transit_overlay_fit,
    plot_rm,
    measure_per_transit_tcs, ttvc_envelope,
    # Posterior predictive checks
    PPCResult, posterior_predictive_check, plot_ppc,
    # Detection limits
    DetectionLimitResult, detection_limits, plot_detection_limits,
    # PSIS-LOO / WAIC
    LooResult, compute_loo,
    # Post-fit health guard
    FitHealthCheck, FitHealthReport, assess_fit,
    # Label-switching (slot-permutation) diagnostic
    SlotSwitchStat, LabelSwitchingReport, label_switching, print_label_switching,
    CoherenceResult, coherence_discriminant,
    CandidateVerdict, vet_candidate

# =====================================================================
# Module init
# =====================================================================
#
# Force OpenBLAS to a single thread at module load. Two reasons:
#
#   1. Under juliacall (every exoautomata worker), PythonCall holds
#      the Python GIL while Julia executes. OpenBLAS spawns its own
#      worker thread pool (default = nproc, often 16+) at first BLAS
#      call. Those workers park on futex between calls and interact
#      with the GIL handshake in pathological ways — NOI-106823 on
#      `pt_warm` hung for 20+ minutes with 17/19 threads parked on
#      futex inside Pathfinder, even after the IS-resample NaN and
#      Pathfinder `ThreadedEx → SequentialEx` fixes. The remaining
#      worker pool was OpenBLAS.
#
#   2. Outside juliacall, Nereus's Julia-level threading
#      (`Threads.@threads :static` in PT / ptemcee / OFTI / PA, plus
#      `Threads.@spawn` chain dispatch in RJMCMC / ensemble / NUTS)
#      already saturates the available cores at the per-chain
#      granularity. Nested BLAS threading is pure overhead — Julia
#      and OpenBLAS would compete for the same cores. The Nereus
#      benchmarks (memory project_enzyme_issue) showed BLAS
#      multithreading didn't improve any sampler on our matrix sizes
#      (≤ 100×100 typical).
#
# The set is per-process — every Nereus workload gets the right
# default without the user having to remember.
using LinearAlgebra: BLAS
function __init__()
    BLAS.set_num_threads(1)
end


# Public API surface (api.jl / features.jl)
export fit_rv, fit_transit, fit_astrometry, fit_rm, fit_tomography,
    fit_ttv, fit_binary, fit_joint, run_engine, Stopping, ENGINES,
    FEATURE_ACTIONS, default_rv_planet, default_astrom_planet

end # module Nereus
