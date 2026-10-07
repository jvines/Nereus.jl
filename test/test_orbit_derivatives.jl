# Derivatives in e at and near e = 0.
#
# Kepler's solver returned its initial guess M + 0.85 sign(sin M) e whenever
# that already met the tolerance, which it does at e = 0 (and below e ~ 1e-10):
# an AD backend tracing the iteration then got ∂E/∂e = 0.85 sign(sin M), not
# sin M. And the true anomaly's clamp `min(max(e, 0), ...)` returned the
# constant 0 at e = 0 (Base's `max` picks its second argument on a tie), so
# ∂f/∂e was lost there too. A ForwardDiff gradient of a likelihood in e at a
# circular orbit was wrong by a factor of order one.
#
# The references here are the implicit-function derivatives of Kepler's
# equation and one-sided (right) finite differences of second order, since e
# cannot go below 0.

using Test
using Nereus
using ForwardDiff
using ReverseDiff
using Random: MersenneTwister

# d/dx f at x0 from the right: (-3 f(x0) + 4 f(x0 + h) - f(x0 + 2h)) / 2h, O(h²).
_od_fd(f, x0, h) = (-3 * f(x0) + 4 * f(x0 + h) - f(x0 + 2h)) / (2h)

@testset "Kepler's equation: derivatives of the solution" begin
    for e in (0.0, 1e-14, 1e-11, 1e-6, 0.3, 0.9, 0.99)
        for M in range(-3.1, 3.1; length = 63)
            E = kepler_solve(M, e)
            sE, cE = sincos(E)
            dEde, dEdM = sE / (1 - e * cE), 1 / (1 - e * cE)
            # ForwardDiff (the Dual method)
            @test ForwardDiff.derivative(x -> kepler_solve(M, x), e) ≈ dEde rtol = 1e-9 atol = 1e-14
            @test ForwardDiff.derivative(x -> kepler_solve(x, e), M) ≈ dEdM rtol = 1e-9
            # A backend that traces the iteration (the Newton step on convergence)
            g = ReverseDiff.gradient(v -> kepler_solve(v[1], v[2]), [M, e])
            @test g[1] ≈ dEdM rtol = 1e-9
            @test g[2] ≈ dEde rtol = 1e-9 atol = 1e-14
        end
    end
    # The Dual's value is the Float64 solution, to the bit; and an M far from
    # [-π, π) keeps the same derivatives.
    for (M, e) in ((1.3, 0.0), (-2.2, 0.4), (40.0, 0.7), (-1e4, 0.2))
        d = kepler_solve(ForwardDiff.Dual(M, 1.0), ForwardDiff.Dual(e, 0.0))
        @test ForwardDiff.value(d) === kepler_solve(M, e)
        E = kepler_solve(M, e)
        @test ForwardDiff.partials(d)[1] ≈ 1 / (1 - e * cos(E)) rtol = 1e-12
    end
    # Second derivative at e = 0: E = M + e sin M + e² sin M cos M + O(e³).
    for M in (-2.5, -0.4, 0.9, 2.0)
        d2 = ForwardDiff.derivative(e -> ForwardDiff.derivative(x -> kepler_solve(M, x), e), 0.0)
        @test d2 ≈ sin(2M) rtol = 1e-9
    end
    # Mixed tags (a Dual in M only, a Dual in e only)
    @test ForwardDiff.derivative(x -> kepler_solve(x, 0.0), 0.7) ≈ 1.0
    @test ForwardDiff.gradient(v -> kepler_solve(v[1], v[2]), [0.7, 0.0]) ≈ [1.0, sin(0.7)]
end

@testset "true anomaly: ∂f/∂e at e = 0" begin
    for E in range(-3.0, 3.0; length = 25)
        # f = 2 atan(√(1+e) sin(E/2), √(1-e) cos(E/2)): ∂f/∂e = sin E at e = 0.
        @test ForwardDiff.derivative(x -> Nereus.true_anomaly(E, x), 0.0) ≈ sin(E) atol = 1e-14
        for e in (1e-12, 0.2, 0.8)
            fe(x) = Nereus.true_anomaly(E, x)
            h = 1e-6 * max(e, 1e-3)
            fd = e > 2h ? (fe(e + h) - fe(e - h)) / 2h : _od_fd(fe, e, 1e-5)
            @test ForwardDiff.derivative(fe, e) ≈ fd rtol = 1e-6 atol = 1e-9
        end
    end
end

@testset "orbit kernels: d/de at e = 0 against finite differences" begin
    ts = collect(range(0.0, 7.0; length = 400))
    P, K, ω, M0, tref = 3.1, 25.0, 0.7, 0.4, 1.0
    rv(e) = sum(t -> rv_keplerian(t, P, K, e, ω, M0, tref) * (1 + t), ts)
    @test ForwardDiff.derivative(rv, 0.0) ≈ _od_fd(rv, 0.0, 1e-5) rtol = 1e-6
    @test ForwardDiff.derivative(rv, 1e-12) ≈ _od_fd(rv, 1e-12, 1e-5) rtol = 1e-6

    # The photometric kernels, at a transit: sky separation and sin(ω + f).
    Tp, b, aR = 0.3, 0.4, 8.0
    tt = collect(range(-0.6, 0.6; length = 301)) .+ tp_to_tc(Tp, P, 0.0, ω)
    sep(e) = sum(t -> sky_separation(t, P, e, ω, Tp, b, aR), tt)
    sep_ws(e) = (o = Nereus._sky_orbit(P, e, ω, Tp, b, aR);
                 sum(t -> sum(Nereus._sky_separation_signed(o, t)), tt))
    pos_ws(e) = (o = Nereus._sky_orbit(P, e, ω, Tp, b, aR);
                 sum(t -> sum(Nereus._sky_position(o, t) .* (1, 2, 3)), tt))
    for g in (sep, sep_ws, pos_ws)
        @test ForwardDiff.derivative(g, 0.0) ≈ _od_fd(g, 0.0, 1e-5) rtol = 1e-6
    end
end

# A planet with an RV and a transit, in the (e, ω) parametrisation so that e is
# a parameter of its own and can sit at exactly 0.
function _od_fit(; ew = :ew, time = :Tc)
    rng = MersenneTwister(5)
    t_rv = sort(30 .* rand(rng, 25))
    rv = 20 .* sin.(2π .* (t_rv .- 1.1) ./ 3.3) .+ randn(rng, 25)
    t_ph = collect(range(0.0, 10.0; length = 3000))
    ph = @. abs(mod(t_ph - 1.1 + 3.3 / 2, 3.3) - 3.3 / 2)
    flux = 1.0 .+ 4e-4 .* randn(rng, length(t_ph)); flux[ph .< 0.06] .-= 0.008
    data = Data(; t_rv, rv, rv_err = fill(1.0, 25), rv_inst = ones(Int, 25),
                  t_phot = t_ph, flux, flux_err = fill(4e-4, length(t_ph)),
                  phot_inst = ones(Int, length(t_ph)))
    params = Params(; max_kplanet = 1, planet_modes = [RVPM],
                      instruments = InstrumentConfig(rv = ["HARPS"], pm = ["TESS"]),
                      data, parametrization = ParametrizationConfig(; ew, time),
                      M_s = 1.0, R_s = 1.0, stability = :none)
    th = Theta{Float64}(params)
    ix = params.layout.name_to_idx
    v = Dict("P_k1" => 3.3, "K_k1" => 20.0, "b_k1" => 0.3, "rr_k1" => 0.09,
             "gamma_HARPS" => 0.0, "sigma_HARPS" => 1.0, "q1_TESS" => 0.3, "q2_TESS" => 0.2,
             String(time) * "_k1" => (time === :Tc ? 1.1 : time === :Tp ? 0.6 : 0.8))
    for (k, x) in v
        th.values[ix[k]] = x
    end
    return params, data, th
end

# log L as a function of the named parameters `names`, the rest at `th`.
function _od_ll(params, data, th, names)
    ix = params.layout.name_to_idx
    return function (x)
        T = eltype(x)
        t = Theta{T}(params)
        t.values .= th.values
        for (i, nm) in enumerate(names)
            t.values[ix[nm]] = x[i]
        end
        return rv_log_likelihood(t, data) + transit_log_likelihood(t, data)
    end
end

@testset "likelihood: d/de at a circular orbit" begin
    for time in (:Tc, :Tp, :Mo)
        params, data, th = _od_fit(; ew = :ew, time)
        th.values[params.layout.name_to_idx["w_k1"]] = 0.7
        ll = _od_ll(params, data, th, ["ecc_k1"])
        f(e) = ll([e])
        g0 = ForwardDiff.gradient(ll, [0.0])[1]
        @test isfinite(f(0.0))
        @test g0 ≈ _od_fd(f, 0.0, 1e-6) rtol = 1e-4
        # Just above 0 the derivative is continuous with the one at 0.
        @test ForwardDiff.gradient(ll, [1e-9])[1] ≈ g0 rtol = 1e-4
        # The workspace likelihoods (Float64) see the same function.
        ws = Nereus.PTWorkspace(params, 1, 0; n_obs = length(data.t_rv),
                                n_phot = length(data.t_phot))
        fws(e) = (t = Theta{Float64}(params); t.values .= th.values;
                  t.values[params.layout.name_to_idx["ecc_k1"]] = e;
                  rv_log_likelihood(t, data, ws) + transit_log_likelihood(t, data, ws))
        @test _od_fd(fws, 0.0, 1e-6) ≈ g0 rtol = 1e-4
    end
end

@testset "(sesinw, secosw) at the origin" begin
    # With the transit time as the anchor, e = s² + c² has zero gradient at the
    # origin and ω (undefined there) only enters terms that vanish with e: the
    # gradient is 0, finite, and what finite differences give. (With Tp or Mo
    # as the anchor the model has no derivative at the origin: the transit time
    # Tp + (π/2 - ω) P/2π, or the phase Mo + ω, depends on the direction from
    # which the origin is approached.)
    params, data, th = _od_fit(; ew = :sesinw, time = :Tc)
    ll = _od_ll(params, data, th, ["sesinw_k1", "secosw_k1"])
    g = ForwardDiff.gradient(ll, [0.0, 0.0])
    @test g == [0.0, 0.0]
    # Central differences there are O(h) (e = h²): ~2e4 h on this fit.
    h = 1e-6
    @test abs((ll([h, 0.0]) - ll([-h, 0.0])) / 2h) < 0.1
    @test abs((ll([0.0, h]) - ll([0.0, -h])) / 2h) < 0.1
    # Close to the origin, every anchor: ForwardDiff against fourth-order
    # central differences.
    for time in (:Tc, :Tp, :Mo)
        params, data, th = _od_fit(; ew = :sesinw, time)
        ll = _od_ll(params, data, th, ["sesinw_k1", "secosw_k1"])
        for x in ([0.01, 0.0], [0.0, -0.02], [0.03, 0.04])
            gx = ForwardDiff.gradient(ll, x)
            @test all(isfinite, gx)
            # Under Tp and Mo the transit moves by P/2π per radian of ω, so
            # log L varies on a scale ~1e-2 |x| here: a much smaller step.
            hx = (time === :Tc ? 1e-4 : 1e-6) * hypot(x...)
            d(i, k) = ll(x .+ k * hx .* (1:2 .== i))
            fd = [(8 * (d(i, 1) - d(i, -1)) - (d(i, 2) - d(i, -2))) / (12hx) for i in 1:2]
            @test gx ≈ fd rtol = 1e-5
        end
    end
end
