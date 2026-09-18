# Both prior-free evidence estimators, against KNOWN truth on a CURVED target.
#
# The existing tests use a Gaussian in a box: correct, but an ellipsoid, and a
# reference fitted to it is trivially good. The failure modes that matter on
# real RV posteriors are curvature (the e-omega ridge, which is what makes
# nested sampling under-mix here) and reference draws that are correlated MCMC
# output rather than independent samples.
#
# Construction: compose the Gaussian-in-a-box with a PAIRWISE shear
#     T(y)_i = y_i + b*(y_{i-1}^2 - 1)      for even i
# which is lower triangular with unit diagonal, so |det J_T| = 1 EXACTLY and
# the integral is unchanged: log Z = -d*log(2L) still, to machine precision.
# Curvature is dialled by b at fixed, known truth.
#
# The shear must NOT be chained (T_i depending on T_{i-1}) -- iterating a square
# fifteen times blows the coordinates up super-exponentially and the sample
# covariance goes singular.
using Test
using Nereus
using Nereus: _bridge_iterate, _reference_path_core
using LinearAlgebra, Statistics, Random

@testset "evidence estimators on a curved target" begin
    d, L, s2 = 15, 20.0, 1.0
    rng = MersenneTwister(3)
    A = randn(rng, d, d); Sig = A * A' / d + 0.5I
    Lc = cholesky(Symmetric(Sig)).L
    iS = inv(Symmetric(Sig)); ld = logdet(Symmetric(Sig))
    truth = -d * log(2L)

    T(y, b)    = (x = copy(y); for i in 2:2:d; x[i] = y[i] + b*(y[i-1]^2 - s2); end; x)
    Tinv(x, b) = (y = copy(x); for i in 2:2:d; y[i] = x[i] - b*(x[i-1]^2 - s2); end; y)
    mklp(b) = y -> begin
        x = T(y, b)
        any(abs.(x) .> L) ? -Inf :
            -0.5*dot(x, iS*x) - 0.5d*log(2π) - 0.5ld - d*log(2L)
    end

    N = 20_000
    for b in (0.0, 0.3, 1.0)
        lp = mklp(b)
        X = Lc * randn(rng, d, N)
        Y = similar(X); for i in 1:N; Y[:, i] = Tinv(@view(X[:, i]), b); end

        mu = vec(mean(Y; dims = 2)); S = cov(Y; dims = 2) + 1e-8I
        Lh = cholesky(Symmetric(S)).L
        logq(y) = (z = Lh \ (y .- mu); -0.5*sum(abs2, z) - 0.5d*log(2π) - logdet(Lh))

        l1 = filter(isfinite, [lp(@view Y[:, i]) - logq(@view Y[:, i]) for i in 1:N])
        l2 = Float64[]
        for _ in 1:N
            y = mu + Lh * randn(rng, d); v = lp(y)
            isfinite(v) && push!(l2, v - logq(y))
        end
        bz, _, ok = _bridge_iterate(l1, l2, 1000, 1e-12)
        rp = _reference_path_core(lp, Y; n_particles = 1024, n_beta = 3000,
                                  n_steps = 16, seed = 7)

        @test ok
        # Observed |error| <= 0.09 for both across b = 0 .. 1. The bar is set an
        # order of magnitude above that, and three below the ~176 nats by which
        # the beta-path estimators miss a real signal-locked posterior.
        @test abs(bz - truth) < 0.5
        @test abs(rp.log_z_ais - truth) < 0.5
        @test rp.ess > 0.25 * rp.n_particles
        @test rp.support_frac > 0.9
    end

    # Reference fitted from CORRELATED draws, as a real PT chain gives -- every
    # other test hands these estimators independent samples, which is the one
    # condition real chains do not satisfy.
    b = 0.3; lp = mklp(b)
    Yc = Matrix{Float64}(undef, d, N)
    y = Tinv(Lc * randn(rng, d), b); lpy = lp(y); keep = 1
    for it in 1:(N * 4)
        prop = y .+ 0.4 .* (Lc * randn(rng, d)); lpp = lp(prop)
        if isfinite(lpp) && log(rand(rng)) < lpp - lpy; y = prop; lpy = lpp; end
        if it % 4 == 0 && keep <= N; Yc[:, keep] = y; keep += 1; end
    end
    rpc = _reference_path_core(lp, Yc; n_particles = 1024, n_beta = 3000,
                               n_steps = 16, seed = 7)
    @test abs(rpc.log_z_ais - truth) < 0.5      # observed 0.02

    # Bridge on the SAME well-mixed correlated draws. It is fine here
    # (observed -0.012); the leg exists so the sticky case below is a
    # contrast against a measured pass, not against nothing.
    bridge_err(Y) = begin
        mu = vec(mean(Y; dims = 2)); S = cov(Y; dims = 2) + 1e-8I
        Lh = cholesky(Symmetric(S)).L
        lq(y) = (z = Lh \ (y .- mu); -0.5*sum(abs2, z) - 0.5d*log(2π) - logdet(Lh))
        l1 = filter(isfinite, [lp(@view Y[:, i]) - lq(@view Y[:, i]) for i in 1:size(Y, 2)])
        l2 = Float64[]
        for _ in 1:size(Y, 2)
            y = mu + Lh * randn(rng, d); v = lp(y)
            isfinite(v) && push!(l2, v - lq(y))
        end
        first(_bridge_iterate(l1, l2, 1000, 1e-12)) - truth
    end
    @test abs(bridge_err(Yc)) < 0.5

    # STICKY chain: a tiny step scale, no thinning. n_eff collapses to ~20,
    # far below the d + d(d+1)/2 = 135 parameters of the Gaussian reference,
    # so `q` is estimated from too few independent points to be usable.
    # Bridge consumes `q` directly and goes several nats LOW (observed
    # -3.2); reference-path anneals away from `q` and survives (observed
    # +0.17). Splitting the sample does NOT rescue bridge (-2.8), so this is
    # a badly ESTIMATED reference, not training-set re-use. Documented in
    # docs/src/evidence.md -- the guidance to check ESS rests on this.
    Ys = Matrix{Float64}(undef, d, N)
    y = Tinv(Lc * randn(rng, d), b); lpy = lp(y)
    for it in 1:N
        prop = y .+ 0.05 .* (Lc * randn(rng, d)); lpp = lp(prop)
        if isfinite(lpp) && log(rand(rng)) < lpp - lpy; y = prop; lpy = lpp; end
        Ys[:, it] = y
    end
    rps = _reference_path_core(lp, Ys; n_particles = 1024, n_beta = 3000,
                               n_steps = 16, seed = 7)
    be = bridge_err(Ys)
    @info "sticky-chain evidence" bridge_err = be refpath_err = rps.log_z_ais - truth
    @test abs(rps.log_z_ais - truth) < 1.0        # reference-path holds
    @test abs(be) > 2 * abs(rps.log_z_ais - truth)  # bridge does not
end
