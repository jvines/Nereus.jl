# The sampler registry. ONE list, deliberately in its own file.
#
# There used to be three, and they had drifted: `ENGINES` here (17 names),
# `_KNOWN_SAMPLERS` in runner.jl (15) and `_SAMPLER_FNS` in runner.jl (14).
# The consequences were all silent:
#   - `ofti` was in the runner lists but not ENGINES, so it was unreachable
#     from fit_* and from the Python client.
#   - `ensemble`, `ess` and `map` were in ENGINES only, so a run_job config
#     naming them was rejected as an unknown sampler.
#   - `pt_hmc` was validated by _KNOWN_SAMPLERS but absent from _SAMPLER_FNS,
#     so it skipped _assert_supported_kwargs entirely and bad keywords went
#     deep into Julia before failing.
#
# This file is included before both api.jl and runner.jl, which derive from it
# rather than restating it. Adding a sampler means adding it here, once.
const ENGINES = Dict{String, Function}(
    "pt"                => sample_pt,
    "pt_hmc"            => sample_pt_hmc,
    "pt_whitening"      => sample_pt_whitening,
    "ptemcee"           => sample_ptemcee,
    "transdim_ptemcee"  => sample_transdim_ptemcee,
    "nested"            => sample_nested,
    "nested_ins"        => sample_nested_ins,
    "nested_dynamic"    => sample_nested_dynamic,
    "moms"              => sample_moms,
    "daedalus"           => sample_daedalus,
    "rjmcmc"            => sample_rjmcmc,
    "nuts"              => sample_nuts,
    "map"               => sample_map,
    "smc"               => sample_smc,
    "ensemble"          => sample_ensemble,
    "ess"               => sample_ess,
    "pa"                => sample_pa,
    "ofti"              => ofti_sample,
)
