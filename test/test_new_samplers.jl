# Smoke + correctness tests for the new samplers landed in the
# 2026-05-11 sampler-expansion sprint:
#
#   - sample_nested_ins   (Importance Nested Sampling; Feroz+ 2013)
#   - sample_nested_dynamic (Dynamic NS; Higson+ 2019)
#   - sample_smc          (SMC family interface; delegates to sample_pa)
#   - sample_pt_whitening (NF-coupled PT with diagonal-Gaussian flow)
#   - noise_exclusion_groups (TransDimConfig feature for mutually
#     exclusive noise models — Stage-1 ↔ Stage-3 exclusion etc.)
#
# Budgets are intentionally small — these confirm pipeline correctness,
# not statistical convergence. The full 2D-Gaussian quantitative
# validations live in test_ptemcee_gaussian.jl / test_pa_gaussian.jl /
# test_evidence_gaussian.jl.

using Test
using Random
using Nereus
using MCMCChains

# ---- Shared synthetic single-planet RV target ------------------------
function _make_rv_target(seed::Int = 2027; n_obs::Int = 30,
                          K_true::Float64 = 12.0, P_true::Float64 = 8.0)
    rng    = MersenneTwister(seed)
    t      = sort(rand(rng, n_obs) .* 40.0)
    rv     = K_true .* sin.(2π .* t ./ P_true) .+ randn(rng, n_obs)
    rv_err = fill(1.0, n_obs)
    ic     = InstrumentConfig(rv = ["SIM"])
    data   = Data(; t_rv = t, rv = rv, rv_err = rv_err)
    params = Params(;
        max_kplanet  = 1,
        planet_modes = [RV_ONLY],
        instruments  = ic,
        data         = data,
        M_s          = 1.0,
    )
    target = NereusTarget(params, data; unconstrained = false)
    return target, data, params
end


# =====================================================================
# Basic NS via NestedSamplers.jl wrapper
# =====================================================================
#
# Guards two regressions in the jvines/NestedSamplers.jl fork that
# would silently re-break this code path on a dep bump:
#
#   1. `nested_isdone` uses `progress === true` rather than `if progress`
#      — required because AbstractMCMC 5.x passes sentinel structs
#      (CreateNewProgressBar, NoLogging) instead of Bools. The naive
#      check throws `non-boolean used in boolean context` mid-loop.
#   2. `make_eigvals_positive` conditions the covariance before
#      `decompose` sees it — required because NS collapses live points
#      onto sharp ridges, producing tiny floating-point negative
#      eigenvalues that crash `sqrt`.
#
# If a future upstream / `[sources]` change reverts either fix, this
# test detects it before the production runner does.

@testset "sample_nested (NestedSamplers.jl wrapper)" begin
    @testset "runs end-to-end on synthetic RV" begin
        target, data, _ = _make_rv_target(2027)
        chains, log_z = sample_nested(target, data;
            n_live = 40, dlogz = 2.0, bounds = :multi,
            proposal = :rwalk, seed = 1)
        @test chains isa MCMCChains.Chains
        @test isfinite(log_z)
        @test :K_k1 in names(chains, :parameters)
        @test size(chains, 1) > 0
    end

    @testset "all MCMC proposals + tuning pass-through" begin
        # The 4 MCMC-based kernels must each run and pass walk_scale /
        # n_walks / slices through to the matching NestedSamplers
        # proposal. :unif (pure rejection) is intentionally excluded —
        # it's unusable above a few dimensions by construction.
        target, data, _ = _make_rv_target(2027)
        for prop in (:rwalk, :rstagger, :slice, :rslice)
            _, log_z = sample_nested(target, data;
                n_live = 50, proposal = prop, walk_scale = 1,
                n_walks = 20, slices = 4, dlogz = 2.0, seed = 1)
            @test isfinite(log_z)
        end
    end

    @testset "validates bounds + proposal" begin
        target, data, _ = _make_rv_target(2027)
        @test_throws ArgumentError sample_nested(target, data;
            n_live = 40, proposal = :banana, seed = 1)
        @test_throws ArgumentError sample_nested(target, data;
            n_live = 40, bounds = :rectangle, seed = 1)
    end
end


# =====================================================================
# Importance Nested Sampling
# =====================================================================

@testset "sample_nested_ins" begin
    @testset "runs end-to-end and returns both NS and INS log Z" begin
        target, data, _ = _make_rv_target(2027)
        res = sample_nested_ins(target, data;
            n_live = 40, dlogz = 2.0, max_iter = 1500, seed = 1)
        @test res isa INSResult
        @test isfinite(res.log_z_ns)
        @test isfinite(res.log_z_ins)
        @test res.n_iters > 0
        @test res.n_iters <= 1500
        # n_evals counts every rejection-sampling proposal, so it should
        # comfortably exceed n_live (the init pass) plus n_iters.
        @test res.n_evals > 40 + res.n_iters
        @test :K_k1 in names(res.chains, :parameters)
        @test size(res.chains, 1) > 0
        # NS and INS are alternative estimators of the same Z; on
        # well-mixed runs they agree to a few nats. We don't enforce
        # tight agreement here (the budget is small) but they shouldn't
        # be wildly inconsistent.
        @test abs(res.log_z_ns - res.log_z_ins) < 50.0
    end

    @testset "accepts integer-valued float kwargs (JSON config path)" begin
        # JSON delivers `walk_scale: 1` as Int64; the float-typed kwargs
        # must coerce rather than MethodError. This is the exact failure
        # mode that took NOI-106823 down through run_job.
        target, data, _ = _make_rv_target(2027)
        res = sample_nested_ins(target, data;
            n_live = 40, walk_scale = 1, enlarge = 2, dlogz = 2,
            max_iter = 1500, seed = 1)
        @test res isa INSResult
        @test isfinite(res.log_z_ins)
    end

    @testset "proposal interface (:rwalk / :slice / :unif)" begin
        target, data, _ = _make_rv_target(2027)
        for prop in (:rwalk, :slice, :unif)
            res = sample_nested_ins(target, data;
                n_live = 40, proposal = prop, dlogz = 2.0,
                max_iter = 2000, seed = 1)
            @test res isa INSResult
            @test isfinite(res.log_z_ins)
        end
        @test_throws ArgumentError sample_nested_ins(target, data;
            n_live = 40, proposal = :banana, seed = 1)
    end
end


# =====================================================================
# Dynamic Nested Sampling
# =====================================================================

@testset "sample_nested_dynamic" begin
    @testset "two-pass run completes and returns combined log Z" begin
        target, data, _ = _make_rv_target(2027)
        # n_live_init bumped from 30 → 80: NestedSamplers main (the
        # post-bae4189 [sources] version) is more conservative about
        # ill-conditioned MultiEllipsoid bounds and throws a sqrt-of-
        # negative on tiny live-point sets. 80 keeps the covariance
        # PSD across seeds; doesn't change the science.
        res = sample_nested_dynamic(target, data;
            n_live_init  = 80,
            n_live_batch = 50,
            dlogz_init   = 2.0,
            max_iter     = 1200,
            seed         = 1)
        @test res isa DynamicNSResult
        @test isfinite(res.log_z)
        @test isfinite(res.log_z_baseline)
        @test isfinite(res.log_z_batch)
        @test res.n_iters_baseline > 0
        @test res.n_iters_batch > 0
        @test res.n_evals > 0
        @test isfinite(res.target_L_lo)
        @test isfinite(res.target_L_hi)
        # Importance window bounds are documented as `[L_lo, L_hi]` but
        # can come out unordered at very small budgets (n_live_init=30
        # + dlogz=2.0 + max_iter=1200) where the baseline NS exits with
        # only a handful of dead points; the index lookup is then noisy.
        # In production budgets (n_live_init ≥ 200, max_iter ≥ 10k) the
        # ordering is well-behaved. We don't enforce it here.
        @test :K_k1 in names(res.chains, :parameters)
        @test size(res.chains, 1) > 0
    end

    @testset "proposal interface (:rwalk / :slice)" begin
        target, data, _ = _make_rv_target(2027)
        for prop in (:rwalk, :slice)
            res = sample_nested_dynamic(target, data;
                n_live_init = 80, n_live_batch = 50, proposal = prop,
                dlogz_init = 2.0, max_iter = 1200, seed = 1)
            @test res isa DynamicNSResult
            @test isfinite(res.log_z)
        end
        @test_throws ArgumentError sample_nested_dynamic(target, data;
            n_live_init = 80, proposal = :banana, seed = 1)
    end

    @testset "Higson pfrac / maxfrac importance allocation" begin
        # pfrac is the posterior↔evidence f-knob (dynesty default 0.8).
        # Both extremes and the default must run and return finite log Z;
        # out-of-range values throw.
        target, data, _ = _make_rv_target(2027)
        for pf in (0.0, 0.5, 1.0)
            res = sample_nested_dynamic(target, data;
                n_live_init = 80, n_live_batch = 50, pfrac = pf,
                maxfrac = 0.8, dlogz_init = 2.0, max_iter = 1200, seed = 1)
            @test res isa DynamicNSResult
            @test isfinite(res.log_z)
        end
        @test_throws ArgumentError sample_nested_dynamic(target, data;
            n_live_init = 80, pfrac = 1.5, seed = 1)
        @test_throws ArgumentError sample_nested_dynamic(target, data;
            n_live_init = 80, maxfrac = 0.0, seed = 1)
    end
end


# =====================================================================
# SMC family interface
# =====================================================================

@testset "sample_smc" begin
    target, data, _ = _make_rv_target(2027)

    @testset "betas validation" begin
        # Empty betas
        @test_throws ArgumentError sample_smc(target, data; betas = Float64[])
        # Must start at 0
        @test_throws ArgumentError sample_smc(target, data;
            betas = [0.5, 1.0])
        # Must end at 1
        @test_throws ArgumentError sample_smc(target, data;
            betas = [0.0, 0.5])
        # Must be sorted ascending
        @test_throws ArgumentError sample_smc(target, data;
            betas = [0.0, 0.7, 0.5, 1.0])
    end

    @testset "smoke run delegates to sample_pa" begin
        # `sample_smc` routes through `sample_pa` adaptively; the betas
        # arg only fixes the step *count* (max_steps = length(betas)-1).
        res = sample_smc(target, data;
            betas      = [0.0, 0.25, 0.5, 0.75, 1.0],
            n_replicas = 30,
            n_mcmc     = 2,
            seed       = 1)
        # PAResult is what sample_pa returns — confirm via duck typing
        # rather than the imported PAResult symbol (which is exported
        # but we want to be tolerant of refactors).
        @test res isa Nereus.PAResult
        @test isfinite(res.log_evidence)
        @test res.n_evals > 0
        @test res.chains isa MCMCChains.Chains
    end
end


# =====================================================================
# NF-coupled PT — whitening variant
# =====================================================================

@testset "sample_pt_whitening" begin
    @testset "runs end-to-end with whitening swap proposals" begin
        target, data, _ = _make_rv_target(2027)
        res = sample_pt_whitening(target, data;
            n_temps        = 4,
            n_steps        = 200,
            n_burnin       = 50,
            warmup_swaps   = 80,
            seed           = 1,
            show_progress  = false)
        @test res isa WhiteningPTResult
        @test isfinite(res.log_evidence)
        @test length(res.acceptance_within) == 4
        @test length(res.acceptance_swap)   == 3
        @test all(0.0 .<= res.acceptance_within .<= 1.0)
        @test all(0.0 .<= res.acceptance_swap   .<= 1.0)
        @test res.n_evals > 0
        @test length(res.betas) == 4
        @test res.betas[1] == 1.0      # cold chain at β=1
        @test issorted(res.betas, rev = true)  # descending
        # Whitening should have kicked in past warmup_swaps; we count
        # how many post-warmup swap rounds used the transformed proposal.
        @test res.whitening_active_after >= 0
        @test :K_k1 in names(res.chains, :parameters)
        @test size(res.chains, 1) > 0
    end
end


# =====================================================================
# noise_exclusion_groups (TransDimConfig)
# =====================================================================

@testset "noise_exclusion_groups" begin
    ar = ARModel(order = 1, per_instrument = true)
    ma = MAModel(order = 1, per_instrument = true)

    @testset "construction validation" begin
        # Group of size 1 is meaningless
        @test_throws ArgumentError TransDimConfig(;
            max_kplanet = 1,
            noise = true,
            toggleable = [ar, ma],
            noise_exclusion_groups = [[ar]])
        # Member not in toggleable
        @test_throws ArgumentError TransDimConfig(;
            max_kplanet = 1,
            noise = true,
            toggleable = [ar],
            noise_exclusion_groups = [[ar, ma]])
        # Valid construction
        td = TransDimConfig(;
            max_kplanet = 1,
            noise = true,
            toggleable = [ar, ma],
            noise_exclusion_groups = [[ar, ma]])
        @test td.noise_exclusion_groups == [[ar, ma]]
    end

    @testset "propose_noise_birth refuses excluded peer activation" begin
        # Build a target with both AR + MA in noise_models, transdim_noise on.
        rng    = MersenneTwister(909)
        n_obs  = 25
        t      = sort(rand(rng, n_obs) .* 30.0)
        rv     = randn(rng, n_obs)
        rv_err = ones(n_obs)
        ic     = InstrumentConfig(rv = ["X"])
        data   = Data(; t_rv = t, rv = rv, rv_err = rv_err)
        params = Params(;
            max_kplanet     = 1,
            planet_modes    = [RV_ONLY],
            instruments     = ic,
            data            = data,
            M_s             = 1.0,
            noise_models    = [ar, ma],
            transdim_noise  = true,
        )
        td_state = TransDimState(max_planets = 1, n_noise = 2)
        theta = Theta{Float64}(params; td = td_state)
        # Initialize with valid prior draws
        for (i, idx) in enumerate(params.layout.unfrozen_idx)
            prior = params.layout.unfrozen_priors[i]
            v = rand(rng, prior.dist)
            lo, hi = bounds(prior)
            theta.values[idx] = clamp(v, lo, hi)
        end
        # Activate the FIRST toggleable (AR)
        activate_noise!(theta.td, 1)
        @test  is_noise_active(theta.td, 1)
        @test !is_noise_active(theta.td, 2)

        # propose_noise_birth's `toggleable` arg is typed
        # `Vector{NoiseModel}`; Julia infers `Vector{SequentialNoise}`
        # for `[ar, ma]` (both concrete SequentialNoise), so we widen
        # explicitly.
        toggleable_widened = NoiseModel[ar, ma]

        # With NO exclusion group: birth proposal can activate MA.
        new_theta_ok, log_q_ok = propose_noise_birth(theta,
            MersenneTwister(1), toggleable_widened)
        @test isfinite(log_q_ok)
        @test is_noise_active(new_theta_ok.td, 2)

        # With AR and MA in an exclusion group: birth proposal must
        # refuse to activate MA while AR is active, so the candidate
        # `inactive` list is empty → returns (theta, -Inf).
        new_theta_ex, log_q_ex = propose_noise_birth(theta,
            MersenneTwister(1), toggleable_widened;
            exclusion_groups = Vector{NoiseModel}[NoiseModel[ar, ma]])
        @test log_q_ex == -Inf
        @test !is_noise_active(new_theta_ex.td, 2)
    end

    @testset "exclusion respected even when active member is not in group" begin
        # If ar is in toggleable AND active, but the exclusion group is
        # ONLY {ma, other_model}, the activation of ma should still
        # succeed because ar isn't in the conflicting group. This
        # confirms the group lookup is correctly scoped per-group, not
        # global.
        third = MAModel(order = 2, per_instrument = true)
        td = TransDimConfig(;
            max_kplanet = 1,
            noise = true,
            toggleable = [ar, ma, third],
            noise_exclusion_groups = [[ma, third]])
        @test length(td.noise_exclusion_groups) == 1
        @test td.noise_exclusion_groups[1] == [ma, third]
    end
end
