# Trans-dimensional (model-selection) recovery validation for Nereus.
#
# The fixed-dim matrix in `sampler_validation.jl` certifies that, GIVEN the
# number of planets, every sampler recovers the orbits or fails loud. It
# says NOTHING about whether the trans-dim samplers find the right NUMBER
# of planets — even its 2-planet generator FREEZES n_p=2. This harness
# closes that: it injects a KNOWN number of planets k and checks whether
# each model-selection sampler recovers k (and fails loud, not confidently
# wrong, when it can't).
#
# THE BAR (recover-or-fail-loud, one dimension up):
#   The marginal model posterior P(N_p = k | data) — tabulated from the
#   `:n_planets` column every trans-dim sampler stores, via the exported
#   `model_probabilities` — must put its MODE on the true k. The dangerous
#   failure (the trans-dim analog of silent-wrong) is a CONFIDENT WRONG
#   model: P(modal) high AND modal ≠ k_true — a hallucinated planet (over-
#   fit, esp. on the k=0 pure-noise control) or a missed planet (under-fit).
#   A BROAD model posterior (no confident mode) is honest "can't tell" —
#   fail-loud, acceptable. So: pass = recovered OR not-confident; the only
#   unacceptable outcome is confident-and-wrong.
#
# Samplers under test (all take `td::TransDimConfig`, all store :n_planets):
#   rjmcmc, pt (with td), transdim_ptemcee, moms, moms_ns.
#
# Generators inject k ∈ {0, 1, 2} STRONG, well-separated signals so a
# correct sampler MUST resolve the model — we test model identification on
# resolvable data, not marginal detection. k=0 (pure noise) is the critical
# over-fitting control: under MoMS's inclusion_prior=0.5 the MODEL prior
# actually favours n_p=1 (Binomial(max_k,0.5)), so recovering n_p=0 on
# noise is a real test that the Occam/evidence penalty beats the prior.
#
# Headline metric is MODEL recovery (right number of planets). Conditional
# orbital recovery within the correct model reduces to the already-certified
# fixed-dim problem; a coarse period-presence check is included as a
# secondary diagnostic.
#
# GATE: `transdim_self_test()` validates the scorer on SYNTHETIC n_planets
# posteriors with known verdicts (all-k → recovered/confident; wrong-k →
# silent-wrong; flat → ambiguous/fail-loud) before any sampler is trusted.

include(joinpath(@__DIR__, "sampler_validation.jl"))

using Random, Statistics, Printf
import MCMCChains

const TD_SAMPLERS = ["rjmcmc", "pt", "transdim_ptemcee", "moms", "moms_ns"]

# =====================================================================
# 1. Scorer — model recovery + trans-dim silent-wrong
# =====================================================================

"""
    score_transdim_model(chains, k_true, max_k; confident=0.6) -> NamedTuple

Tabulate P(N_p=k) from the chain's `:n_planets` column and classify:
`recovered` (modal k == k_true), `confident` (P(modal) ≥ `confident`),
`silent_wrong` (confident AND wrong — the unacceptable outcome), and the
recover-or-fail-loud `pass` (recovered OR not confident).
"""
function score_transdim_model(chains, k_true::Int, max_k::Int; confident::Float64 = 0.6)
    probs = model_probabilities(chains; max_kplanet = max_k)   # Dict{Int,Float64}
    # findmax over a Dict returns (maxprob, modal_k).
    pmax, modal_k = findmax(probs)
    p_true = get(probs, k_true, 0.0)
    recovered = modal_k == k_true
    is_conf   = pmax >= confident
    silent_wrong = is_conf && !recovered
    pass = recovered || !is_conf           # only confident-and-wrong fails
    return (probs = probs, modal_k = modal_k, p_modal = pmax, p_true = p_true,
            recovered = recovered, confident = is_conf,
            silent_wrong = silent_wrong, pass = pass, k_true = k_true)
end

# Secondary diagnostic: among samples at the correct model size, is each
# injected period PRESENT in some slot (within `rtol`)? Coarse — the precise
# orbital CI within a fixed model is the already-certified fixed-dim test.
function score_transdim_periods(chains, k_true::Int, planet_truth, max_k::Int;
                                rtol::Float64 = 0.15)
    k_true == 0 && return (checked = false, found = Int[], n_at_ktrue = 0)
    np = Int.(vec(Array(chains[:n_planets])))
    pnames = String.(MCMCChains.names(chains, :parameters))
    pcols = [n for n in pnames if startswith(n, "P_k")]
    isempty(pcols) && return (checked = false, found = Int[], n_at_ktrue = 0)
    at_k = findall(==(k_true), np)
    n_at = length(at_k)
    n_at == 0 && return (checked = true, found = zeros(Int, length(planet_truth)),
                         n_at_ktrue = 0)
    # All slot-periods over the correct-model subset.
    periods = Float64[]
    for c in pcols
        v = vec(Array(chains[Symbol(c)]))
        append!(periods, v[at_k])
    end
    found = Int[]
    for pt in planet_truth
        # A planet is "present" if a non-trivial fraction of subset
        # slot-periods sit within rtol of its true period.
        # Threshold relative to one slot's share (1/max_k): only k_true of
        # the max_k slots are active per sample, so the parked inactive slots
        # dilute the fraction. "Present" = at least half of one active slot's
        # worth of periods cluster within rtol of the true period.
        frac = count(p -> abs(p - pt.P) <= rtol * pt.P, periods) / length(periods)
        push!(found, frac > (0.5 / max_k) ? 1 : 0)
    end
    return (checked = true, found = found, n_at_ktrue = n_at)
end

# =====================================================================
# 2. Self-test — synthetic n_planets posteriors (THE GATE)
# =====================================================================

# Build a minimal Chains carrying only an :n_planets column from a vector.
function _fake_np_chain(np::Vector{Int})
    arr = reshape(Float64.(np), :, 1, 1)
    return MCMCChains.Chains(arr, [:n_planets])
end

"""
    transdim_self_test() -> Bool

Validate the model-recovery scorer on synthetic n_planets posteriors with
known verdicts. Returns `true` iff every control classifies as expected.
"""
function transdim_self_test()
    println("="^70)
    println("TRANS-DIM SCORER SELF-TEST (synthetic n_planets posteriors)")
    println("="^70)
    mk(np) = _fake_np_chain(np)
    # (label, np-vector, k_true, max_k, expect_recovered, expect_silent_wrong)
    cases = [
        ("all n_p=1, truth 1 → recovered/confident", fill(1, 500), 1, 2, true,  false),
        ("all n_p=1, truth 2 → SILENT-WRONG",         fill(1, 500), 2, 3, false, true),
        ("all n_p=0, truth 0 (noise) → recovered",    fill(0, 500), 0, 2, true,  false),
        ("confident n_p=1 on truth-0 noise → SILENT-WRONG (hallucinated)",
                                                       vcat(fill(1,450),fill(0,50)), 0, 2, false, true),
        ("flat 0/1/2, truth 1 → ambiguous (fail-loud, NOT silent-wrong)",
                                                       repeat([0,1,2], 167), 1, 2, false, false),
        ("70% n_p=2 truth 2 → recovered/confident",   vcat(fill(2,350),fill(1,150)), 2, 3, true, false),
    ]
    ok_all = true
    for (lbl, np, k, mx, exp_rec, exp_sw) in cases
        s = score_transdim_model(mk(np), k, mx)
        ok = (s.recovered == exp_rec) && (s.silent_wrong == exp_sw)
        ok_all &= ok
        @printf("%-62s modal=%d p=%.2f rec=%-5s sw=%-5s  %s\n",
                lbl, s.modal_k, s.p_modal, s.recovered, s.silent_wrong,
                ok ? "✓" : "✗ EXPECTED rec=$exp_rec sw=$exp_sw")
    end
    println("-"^70)
    println(ok_all ? "✅ trans-dim scorer VALIDATED" :
                     "❌ trans-dim scorer BROKEN — do not trust verdicts")
    println("="^70)
    return ok_all
end

# =====================================================================
# 3. Generators — inject k known planets, n_p FREE up to max_k
# =====================================================================

# All slots share a common broad-but-bracketing period/K prior so a planet
# can occupy any slot at any period in range (the trans-dim search space);
# γ/σ keep data-driven defaults (this is recovery, not SBC).
function _td_overrides(max_k::Int)
    ov = Dict{String, PriorSpec}()
    for k in 1:max_k
        ov["P_k$k"] = LogUniformPrior(2.0, 50.0)
        ov["K_k$k"] = UniformPrior(0.0, 60.0)
    end
    return ov
end

function _td_config(max_k::Int)
    return TransDimConfig(
        max_kplanet = max_k,
        birth_strategies = [PriorBirth(), InformedBirth()],
        birth_weights = [0.3, 0.7],
        transdim_fraction = 0.3,
    )
end

# Returns (params, data, target, k_true, planet_truth, td, max_k, label).
function gen_td_k0()
    max_k = 2
    rng = MersenneTwister(33_000_001)
    n = 50; t = sort(rand(rng, n) .* 150.0)
    rv = 2.0 .* randn(MersenneTwister(33_000_002), n)     # pure noise
    params, data, target = _build_rv_target(t, rv, fill(2.0, n), max_k, _td_overrides(max_k))
    return params, data, target, 0, NamedTuple[], _td_config(max_k), max_k, "k0_noise"
end

# NOTE on the injection template: the default n_p prior is
# FixedPrior(max_kplanet), so rv_predictions computes ALL max_k planet
# slots. If we inject k < max_k planets, the unset slots have P=0 → Kepler
# ÷0 → NaN predictions → NaN rv. So the injection template is sized to
# EXACTLY k_true (no unset slots → clean preds); the FIT target then uses
# the larger search max_k so the trans-dim sampler can over- and under-fit.
function gen_td_k1()
    fit_max_k = 2
    rng = MersenneTwister(33_000_011)
    n = 50; t = sort(rand(rng, n) .* 150.0)
    tmpl_p, tmpl_d, _ = _build_rv_target(t, zeros(n), fill(2.0, n), 1, _td_overrides(1))
    Tp = tmpl_d.t_ref - 2.0
    truth = [(P = 12.0, K = 30.0, e = 0.10, ω = 0.5, Tp = Tp)]
    preds = _inject_rv(tmpl_p, tmpl_d, truth)
    rv = preds .+ 2.0 .* randn(MersenneTwister(33_000_012), n)
    params, data, target = _build_rv_target(t, rv, fill(2.0, n), fit_max_k, _td_overrides(fit_max_k))
    return params, data, target, 1, truth, _td_config(fit_max_k), fit_max_k, "k1_strong"
end

function gen_td_k2()
    fit_max_k = 3                                        # allow under- AND over-fit
    rng = MersenneTwister(33_000_021)
    n = 70; t = sort(rand(rng, n) .* 180.0)
    tmpl_p, tmpl_d, _ = _build_rv_target(t, zeros(n), fill(2.0, n), 2, _td_overrides(2))
    truth = [(P = 8.0,  K = 28.0, e = 0.05, ω = 0.8, Tp = tmpl_d.t_ref - 1.5),
             (P = 30.0, K = 22.0, e = 0.15, ω = 2.1, Tp = tmpl_d.t_ref - 5.0)]
    preds = _inject_rv(tmpl_p, tmpl_d, truth)
    rv = preds .+ 2.0 .* randn(MersenneTwister(33_000_022), n)
    params, data, target = _build_rv_target(t, rv, fill(2.0, n), fit_max_k, _td_overrides(fit_max_k))
    return params, data, target, 2, truth, _td_config(fit_max_k), fit_max_k, "k2_strong"
end

const TD_GENERATORS = Dict("k0" => gen_td_k0, "k1" => gen_td_k1, "k2" => gen_td_k2)

# =====================================================================
# 4. Sampler dispatch — call each trans-dim sampler, return chains
# =====================================================================

function run_transdim_sampler(name::String, target::NereusTarget, data::Data,
                              td::TransDimConfig; seed::Int = 42)
    t0 = time(); chains = nothing; ok = false; err = ""
    try
        if name == "rjmcmc"
            chains, _ = sample_rjmcmc(target, data; td = td, n_samples = 40_000,
                                       n_warmup = 15_000, seed = seed, show_progress = false)
        elseif name == "pt"
            chains, _ = sample_pt(target; td = td, n_rounds = 13, n_chains = 10,
                                   seed = seed, show_report = false)
        elseif name == "transdim_ptemcee"
            res = sample_transdim_ptemcee(target, data; td = td, n_temps = 8,
                                           n_walkers = 90, n_steps = 4000,
                                           n_burnin = 2000, seed = seed,
                                           show_progress = false)
            chains = res.chains
        elseif name == "moms"
            # informed_birth_fraction=0.7: moms's birth does NOT consult
            # td.birth_strategies — it births via InformedBirth w.p.
            # informed_birth_fraction, else a Gaussian RW around the parked
            # off-values. At the default 0.0 a SINGLE chain births only by RW
            # and cannot reach a well-separated 2nd planet's period (the
            # ensemble samplers find it by walker count; moms can't). 0.7
            # matches the InformedBirth weight the td config gives the others.
            chains, _, _ = sample_moms(target, data; td = td, n_samples = 30_000,
                                        n_warmup = 15_000, seed = seed,
                                        informed_birth_fraction = 0.7,
                                        show_progress = false)
        elseif name == "moms_ns"
            chains, _, _ = sample_moms_ns(target, data; td = td, n_live = 400,
                                           dlogz = 0.3, n_mcmc = 30, seed = seed,
                                           show_progress = false)
        else
            error("unknown trans-dim sampler `$name`")
        end
        ok = true
    catch e
        ok = false; err = sprint(showerror, e)
    end
    return (chains = chains, ok = ok, err = err, runtime_s = time() - t0)
end

"""
    run_td_one(sampler, gen_key; seed) -> scored cell

Build the generator's trans-dim target, run the sampler, score model
recovery + the period-presence diagnostic.
"""
function run_td_one(sampler::String, gen_key::String; seed::Int = 42)
    params, data, target, k_true, truth, td, max_k, label = TD_GENERATORS[gen_key]()
    r = run_transdim_sampler(sampler, target, data, td; seed = seed)
    if !r.ok || r.chains === nothing
        return (sampler = sampler, gen = gen_key, k_true = k_true, ok = false,
                err = r.err, runtime_s = r.runtime_s)
    end
    m = score_transdim_model(r.chains, k_true, max_k)
    p = score_transdim_periods(r.chains, k_true, truth, max_k)
    return (sampler = sampler, gen = gen_key, k_true = k_true, ok = true, err = "",
            runtime_s = r.runtime_s, model = m, periods = p)
end

function print_td_cell(c)
    if !c.ok
        @printf("%-18s %-10s k=%d  ERROR: %s\n", c.sampler, c.gen, c.k_true,
                first(c.err, 70))
        return
    end
    m = c.model
    probstr = join([@sprintf("%d:%.2f", k, m.probs[k]) for k in sort(collect(keys(m.probs)))], " ")
    verdict = m.silent_wrong ? "❌ SILENT-WRONG (confident k=$(m.modal_k)≠$(c.k_true))" :
              m.recovered    ? (m.confident ? "✅ RECOVERED" : "~ recovered (unsure)") :
                               "⚠ AMBIGUOUS (fail-loud)"
    pstr = c.periods.checked ? "  periods_found=$(c.periods.found)" : ""
    @printf("%-18s %-10s k=%d  P(Np)={%s}  modal=%d(%.2f)  %s%s  [%.0fs]\n",
            c.sampler, c.gen, c.k_true, probstr, m.modal_k, m.p_modal, verdict, pstr,
            c.runtime_s)
end

# =====================================================================
# Entry point: gate, then matrix or a single (sampler, gen) via ARGS.
# =====================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    transdim_self_test() || error("trans-dim scorer self-test failed — aborting")
    if length(ARGS) >= 1
        sampler = ARGS[1]
        gens = length(ARGS) >= 2 ? [ARGS[2]] : ["k0", "k1", "k2"]
        sw = 0
        for g in gens
            c = run_td_one(sampler, g)
            print_td_cell(c)
            c.ok && c.model.silent_wrong && (sw += 1)
        end
        println(sw == 0 ? "\n✅ no silent-wrong for $sampler" :
                          "\n❌ $sw silent-wrong cell(s) for $sampler")
    end
end
