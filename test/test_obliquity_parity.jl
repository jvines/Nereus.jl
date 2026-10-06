# The obliquity Nereus target reproduces the bespoke log-posteriors it replaces
# -- the velocities-only fit, the Doppler-shadow fit (per-night and shared
# amplitude) and the joint fit with each set of terms -- TERM BY TERM, at
# random points, to 1e-9 relative.
#
# The bespoke references are the library log-posteriors (`tomogram_logpost`,
# `joint_obliquity_logpost`) and a verbatim copy of the velocities-only one in
# the fixture. They drop prior normalisations and sample some coordinates in
# log10, so the framework's prior + log|Jacobian| must differ from theirs by a
# CONSTANT, the analytic normalisation C computed below; the likelihood terms
# must agree outright.
#
# One deliberate difference, flagged: the bespoke velocities-only fit had no
# prior on the per-night offsets (improper), where the paper states U(μ ± s).
# The target carries the paper's prior, so C there includes -log(2s) per night
# and the target rejects offsets outside it.
using Test
using Nereus
using Nereus: Theta, log_prior, rv_log_likelihood, tomogram_log_likelihood,
              NereusTarget, RMNight, TomoNight, set_param!
using Random, Statistics

@isdefined(OS_P) || include(joinpath(@__DIR__, "fixtures", "obliquity_synthetic.jl"))

const OP_DIR = os_write_data(mktempdir())
const OP_RM, OP_TOMO = os_load(OP_DIR)
const OP_TAGS = [nt.tag for nt in OP_TOMO]
const OP_TOL = 1e-9

op_params(rm, tomo; vel = false, shared = false) = begin
    d, names = obliquity_data(rm; tomo_nights = tomo)
    b, a = vel ? (OS_ARS[1] * cos(OS_INC), OS_ARS[1]) : (OS_B, OS_ARS)
    p = obliquity_params(d, names; P = OS_P, Tc = OS_TC0, b = b, a_Rs = a,
        rr = OS_RR, vsini = OS_VSINI_MS, K = isempty(rm) ? nothing : OS_K,
        sigma0 = isempty(rm) ? nothing : rm, beta_p_floor = 2000.0,
        occultation = :point, ld = (OS_U1, OS_U2), shared_alpha = shared)
    return d, names, p
end

hlog2π = 0.5 * log(2π)
rv_box_C() = -log(14.0) - log(log10(100 / 0.2)) - log(log10(60 / 0.2)) - log(4.5)
tomo_box_C() = -log(23.0) - log(6.0) - log(log10(60.0)) - log(log10(12 / 0.05)) - log(6.0)

# random points: around the truth, every nuisance inside its box
function op_point(rng, kind, rm)
    θ = Float64[OS_LAM + 0.6randn(rng)]
    if kind === :vel
        append!(θ, [25_900 + 1500randn(rng), 368 + 27randn(rng)])
    else
        append!(θ, [25.9 + 1.5randn(rng), 0.75 + 0.01randn(rng), 6.82 + 0.09randn(rng)])
        kind === :shadow_shared && push!(θ, 0.5 + 2rand(rng))
        for _ in OP_TOMO
            kind === :shadow_shared || push!(θ, 0.5 + 2rand(rng))
            append!(θ, [3 + 10rand(rng), -3.5 + rand(rng), 0.3 + 1.2rand(rng),
                        -1 + 1.5rand(rng), -3.2 + 0.8rand(rng)])
        end
        isempty(rm) || push!(θ, 368 + 27randn(rng))
    end
    for (q, n) in enumerate(rm)
        lo, hi = Nereus._rm_gamma_bounds(rm)[q]
        append!(θ, [mean(n.rv) + 0.2(hi - lo) * (rand(rng) - 0.5), 1 + 6rand(rng),
                    -0.5 + 2rand(rng), 0.0 + 1.5rand(rng), 0.5 + 2rand(rng)])
    end
    return θ
end

function op_check(kind, d, names, p, bespoke; n = 40, use_tomo = !isempty(d.tomo), C)
    rng = MersenneTwister(hash(kind))
    worst = (rv = 0.0, tomo = 0.0, prior = 0.0)
    for k in 1:n
        θ = op_point(rng, kind, isempty(names) ? RMNight[] : OP_RM[1:length(names)])
        k > n ÷ 2 && (θ[1] = -π + 2π * (k - n ÷ 2 - 0.5) / (n ÷ 2))   # lambda round the circle
        bt = bespoke(θ)
        th, lJ = os_seat(kind === :joint_rv ? :joint : kind, θ, p;
                         rm_tags = names, tomo_tags = use_tomo ? OP_TAGS : String[],
                         ntomo_theta = kind === :vel ? 0 : length(OP_TAGS))
        fp, fr, ft = log_prior(th), rv_log_likelihood(th, d), tomogram_log_likelihood(th, d)
        @test isfinite(bt.prior) && isfinite(fp) && isfinite(fr) && isfinite(ft)
        brv = hasproperty(bt, :rv) ? bt.rv : 0.0
        btm = hasproperty(bt, :tomo) ? bt.tomo : 0.0
        isempty(names) || (worst = merge(worst, (rv = max(worst.rv, abs(fr - brv) / abs(brv)),)))
        use_tomo && (worst = merge(worst, (tomo = max(worst.tomo, abs(ft - btm) / abs(btm)),)))
        worst = merge(worst, (prior = max(worst.prior,
                                          abs(fp + lJ - bt.prior - C) / abs(bt.prior + C)),))
        # and the target is that sum
        tg = NereusTarget(p, d; unconstrained = false)
        @test tg([th.values[s] for s in p.layout.unfrozen_idx]) ≈ fp + fr + ft rtol = 1e-14
    end
    @test worst.rv <= OP_TOL
    @test worst.tomo <= OP_TOL
    @test worst.prior <= OP_TOL
    return worst
end

@testset "obliquity target = bespoke log-posteriors, term by term" begin

    @testset "velocities only (fit_NGTS33_rm_bayes model)" begin
        d, names, p = op_params(OP_RM, TomoNight[]; vel = true)
        γb = Nereus._rm_gamma_bounds(OP_RM)
        C = -log(2π) + (-log(OS_VSINI_MS[2]) - hlog2π) + (-log(OS_K[2]) - hlog2π) +
            3 * rv_box_C() + sum(-log(hi - lo) for (lo, hi) in γb)   # γ: flagged
        w = op_check(:vel, d, names, p, θ -> os_bespoke_vel(θ, OP_RM); C = C)
        @info "parity, velocities" w
        # The flagged difference: an offset outside U(μ ± s) is finite for the
        # bespoke model (it had no offset prior) and rejected by the target.
        θ = op_point(MersenneTwister(2), :vel, OP_RM)
        θ[4] = γb[1][2] + 50.0
        @test isfinite(os_bespoke_vel(θ, OP_RM).prior)
        th, _ = os_seat(:vel, θ, p; rm_tags = names, tomo_tags = String[])
        @test log_prior(th) == -Inf
    end

    @testset "Doppler shadow, per-night amplitude (tomogram_logpost)" begin
        d, names, p = op_params(RMNight[], OP_TOMO)
        C = -log(2π) + (-log(1.5) - hlog2π) + (-log(OS_B[2]) - hlog2π) +
            (-log(OS_ARS[2]) - hlog2π) + 3 * (-log(20.0) + tomo_box_C())
        w = op_check(:shadow, d, names, p, θ -> os_bespoke_tomo(θ, OP_TOMO); C = C)
        @info "parity, shadow" w
    end

    @testset "Doppler shadow, shared amplitude (tomogram_logpost, shared_α)" begin
        d, names, p = op_params(RMNight[], OP_TOMO; shared = true)
        C = -log(2π) + (-log(1.5) - hlog2π) + (-log(OS_B[2]) - hlog2π) +
            (-log(OS_ARS[2]) - hlog2π) - log(20.0) + 3 * tomo_box_C()
        w = op_check(:shadow_shared, d, names, p,
                     θ -> os_bespoke_tomo(θ, OP_TOMO; shared = true); C = C)
        @info "parity, shadow shared alpha" w
    end

    @testset "joint, both terms (joint_obliquity_logpost)" begin
        d, names, p = op_params(OP_RM, OP_TOMO)
        C = -log(2π) + (-log(1.5) - hlog2π) + (-log(OS_B[2]) - hlog2π) +
            (-log(OS_ARS[2]) - hlog2π) + (-log(OS_K[2]) - hlog2π) +
            3 * rv_box_C() + 3 * (-log(20.0) + tomo_box_C())
        w = op_check(:joint, d, names, p, θ -> os_bespoke_joint(θ, OP_TOMO, OP_RM); C = C)
        @info "parity, joint" w
    end

    @testset "joint, velocities only (use_tomogram = false)" begin
        # The bespoke vector still carries the 18 line-profile coordinates, which
        # enter nothing (improper and unbounded there); the target has none.
        d, names, p = op_params(OP_RM, TomoNight[])
        C = -log(2π) + (-log(1.5) - hlog2π) + (-log(OS_B[2]) - hlog2π) +
            (-log(OS_ARS[2]) - hlog2π) + (-log(OS_K[2]) - hlog2π) + 3 * rv_box_C()
        w = op_check(:joint_rv, d, names, p,
                     θ -> os_bespoke_joint(θ, OP_TOMO, OP_RM; use_tomogram = false);
                     use_tomo = false, C = C)
        @info "parity, joint velocities only" w
    end

    @testset "joint, one night of velocities" begin
        d, names, p = op_params(OP_RM[3:3], OP_TOMO)
        C = -log(2π) + (-log(1.5) - hlog2π) + (-log(OS_B[2]) - hlog2π) +
            (-log(OS_ARS[2]) - hlog2π) + (-log(OS_K[2]) - hlog2π) +
            rv_box_C() + 3 * (-log(20.0) + tomo_box_C())
        w = op_check(:joint, d, names, p,
                     θ -> os_bespoke_joint(θ, OP_TOMO, OP_RM[3:3]); C = C)
        @info "parity, joint one night" w
    end
end
