# The obliquity likelihood in a sampler's scratch must be the allocating one,
# bit for bit.
#
# pt_emcee (and every sampler with a per-slot PTWorkspace) evaluates the
# residual maps through `tomogram_log_likelihood(theta, data, ws)`, which
# writes into preallocated buffers instead of building ~1 MB of matrices per
# call. A fit that is already running must not see a single bit change when it
# is resumed on the new code, so every comparison here is `===`, never `≈`.
using Test
using Nereus
using Nereus: Theta, PTWorkspace, TomoWorkspace, SymEigenWork, tomogram_log_likelihood,
              set_param!, _sym_eigen!, _kern, obliquity_noise_menu, KernelDistances,
              _matern_upper!, _celerite_upper!, celerite_kernel_dense, sho_coefficients,
              _eval_channel_likelihood, ChannelWork, rv_log_likelihood, CeleriteSHO,
              MaternGP, NoiseModel, _rm_disc_flux, transit_flux, _decode_rm_state,
              planet_indices
using LinearAlgebra: Symmetric, eigen
using Random
using Random: randperm
import ForwardDiff

include(joinpath(@__DIR__, "fixtures", "obliquity_synthetic.jl"))

@testset "obliquity likelihood in a workspace" begin
    dir = os_write_data(mktempdir())
    rm_nights, tomo = os_load(dir)
    d, names = obliquity_data(rm_nights; tomo_nights = tomo)
    base = (P = OS_P, Tc = OS_TC0, b = OS_B, a_Rs = OS_ARS, rr = OS_RR,
            vsini = OS_VSINI_MS, K = OS_K, sigma0 = rm_nights, ld = (OS_U1, OS_U2))
    build(; kw...) = obliquity_params(d, names; base..., kw...)
    new_ws(p) = PTWorkspace(p, p.config.max_kplanet, length(p.config.noise_models);
                            n_obs = length(d.t_rv), n_phot = length(d.t_phot))

    # Prior draws and stretch-like combinations of them (some leave the support).
    function points(p, n; seed = 1)
        tg = NereusTarget(p, d; unconstrained = false)
        rng = MersenneTwister(seed)
        pr = [Nereus._draw_from_prior(tg, rng) for _ in 1:n]
        mix = [(z = (rand(rng) + 1)^2 / 2; a = pr[rand(rng, 1:n)]; b = pr[rand(rng, 1:n)];
                b .+ z .* (a .- b)) for _ in 1:n]
        return vcat(pr, mix)
    end
    function seat!(th, p, x)
        for (j, idx) in enumerate(p.layout.unfrozen_idx)
            th.values[idx] = x[j]
        end
        return th
    end

    # One workspace reused across all points, as a sampler slot reuses it.
    function same_everywhere(p; n = 60, td = nothing)
        th = td === nothing ? Theta{Float64}(p) : Theta{Float64}(p; td = td)
        ws = new_ws(p); tw = TomoWorkspace(p, d.tomo)
        nfin = 0; nsame = 0
        for x in points(p, n)
            seat!(th, p, x)
            a = tomogram_log_likelihood(th, d)
            b = tomogram_log_likelihood(th, d, ws)
            c = tomogram_log_likelihood(th, d, tw)
            nsame += (a === b && a === c)
            nfin += isfinite(a)
        end
        return nsame, 2n, nfin
    end

    @testset "$label" for (label, kw) in (
            ("Matern time kernel (standard)", (;)),
            ("oscillator time kernel", (tomo_noise = :sho,)),
            ("white maps", (tomo_noise = :white,)),
            ("eccentric orbit", (ecc = :free,)),
            ("one shadow amplitude", (shared_alpha = true,)),
            ("point occultation", (occultation = :point,)))
        p = build(; kw...)
        nsame, n, nfin = same_everywhere(p)
        @test nsame == n
        @test nfin >= n ÷ 4          # the comparison covered real values
    end

    @testset "trans-dim noise selection" begin
        menu = obliquity_noise_menu(names, [nt.tag for nt in tomo])
        p = build(; noise_models = menu.noise_models, transdim_noise = true)
        nm = length(p.config.noise_models)
        for pattern in 1:6
            td = Nereus.TransDimState(max_planets = 1, n_noise = nm)
            Nereus.activate_planet!(td, 1)
            rng = MersenneTwister(pattern)
            for i in 1:nm
                td.noise_active[i] = rand(rng, Bool)
            end
            nsame, n, _ = same_everywhere(p; n = 15, td = td)
            @test nsame == n
        end
    end

    @testset "allocation" begin
        p = build()
        th = seat!(Theta{Float64}(p), p, points(p, 5)[1])
        ws = new_ws(p)
        @test isfinite(tomogram_log_likelihood(th, d, ws))
        a_ws = @allocated tomogram_log_likelihood(th, d, ws)
        a_al = @allocated tomogram_log_likelihood(th, d)
        @test a_ws < a_al ÷ 20
    end

    @testset "ForwardDiff passes through" begin
        p = build()
        x = points(p, 5)[1]
        ws = new_ws(p)
        f(ws) = y -> begin
            th = Theta{eltype(y)}(p)
            seat!(th, p, y)
            ws === nothing ? tomogram_log_likelihood(th, d) :
                             tomogram_log_likelihood(th, d, ws)
        end
        @test ForwardDiff.gradient(f(nothing), x) == ForwardDiff.gradient(f(ws), x)
    end

    @testset "a workspace follows the data it is given" begin
        p = build()
        rm2, tomo2 = os_load(os_write_data(mktempdir(); seed = 99))
        d2, _ = obliquity_data(rm2; tomo_nights = tomo2)
        ws = new_ws(p)
        th = Theta{Float64}(p)
        for x in points(p, 10)
            seat!(th, p, x)
            @test tomogram_log_likelihood(th, d, ws) === tomogram_log_likelihood(th, d)
            @test tomogram_log_likelihood(th, d2, ws) === tomogram_log_likelihood(th, d2)
        end
    end

    @testset "RV channel ($label)" for (label, kw) in (
            ("one oscillator per night", (;)),
            ("one Matern per night", (rv_noise = :matern,)),
            ("white", (rv_noise = :white,)),
            ("one global oscillator", (noise_models = NoiseModel[CeleriteSHO(channel = :rv)],)),
            ("oscillators on two of three nights",
             (noise_models = NoiseModel[CeleriteSHO(channel = :rv, instruments = ["A"]),
                                        CeleriteSHO(channel = :rv, instruments = ["C"])],)))
        p = build(; kw...)
        ws = new_ws(p)
        cw = ChannelWork(p, d.t_rv, d.rv_inst, :rv)
        th = Theta{Float64}(p)
        rng = MersenneTwister(21)
        n = length(d.t_rv)
        nsame = 0; ntot = 0
        for x in points(p, 40)
            seat!(th, p, x)
            # Both oscillator branches: Q below and above 1/2.
            for (k, idx) in p.layout.name_to_idx
                startswith(k, "gp_log_Q") && (th.values[idx] = log(rand(rng, (0.3, 0.45, 0.7, 5.0))))
            end
            r = 50 .* randn(rng, n); v = 10 .+ 100 .* rand(rng, n)
            a = _eval_channel_likelihood(th, r, v, d.t_rv, d.rv_inst, :rv, 2π)
            b = _eval_channel_likelihood(th, r, v, d.t_rv, d.rv_inst, :rv, 2π, ws)
            c = _eval_channel_likelihood(th, r, v, d.t_rv, d.rv_inst, :rv, 2π, cw)
            nsame += (a === b && a === c); ntot += 1
            @test rv_log_likelihood(th, d, ws) === rv_log_likelihood(th, d)
        end
        @test nsame == ntot
    end

    @testset "RV channel: trans-dim masks, allocation" begin
        menu = obliquity_noise_menu(names, [nt.tag for nt in tomo])
        p = build(; noise_models = menu.noise_models, transdim_noise = true)
        nm = length(p.config.noise_models)
        n = length(d.t_rv)
        for pattern in 1:8
            td = Nereus.TransDimState(max_planets = 1, n_noise = nm)
            Nereus.activate_planet!(td, 1)
            rng = MersenneTwister(100 + pattern)
            for i in 1:nm
                td.noise_active[i] = rand(rng, Bool)
            end
            th = Theta{Float64}(p; td = td)
            ws = new_ws(p)
            for x in points(p, 10)
                seat!(th, p, x)
                r = 50 .* randn(rng, n); v = 10 .+ 100 .* rand(rng, n)
                @test _eval_channel_likelihood(th, r, v, d.t_rv, d.rv_inst, :rv, 2π) ===
                      _eval_channel_likelihood(th, r, v, d.t_rv, d.rv_inst, :rv, 2π, ws)
            end
        end
        p = build()
        ws = new_ws(p)
        th = seat!(Theta{Float64}(p), p, points(p, 5)[1])
        r = 50 .* randn(n); v = 10 .+ 100 .* rand(n)
        f(ws) = _eval_channel_likelihood(th, r, v, d.t_rv, d.rv_inst, :rv, 2π, ws)
        f(ws)
        @test (@allocated f(ws)) <= 512
        @test (@allocated _eval_channel_likelihood(th, r, v, d.t_rv, d.rv_inst, :rv, 2π)) > 4096
    end

    @testset "RM: the limb-darkening law built once per call" begin
        for (u1, u2) in ((OS_U1, OS_U2), (0.0, 0.0), (-0.0, 0.0), (0.6, 0.0), (0.0, 0.4),
                         (1.0, 0.0))
            ld = Nereus.QuadLimbDark([u1, u2])
            uniform = iszero(u1) && iszero(u2)
            for p in (0.05, OS_RR, 0.3), z in range(0.0, 1.0 + p + 0.05; length = 97)
                @test _rm_disc_flux(uniform, ld, z, p) === transit_flux(z, p, u1, u2)
            end
        end
        p = build()
        th = seat!(Theta{Float64}(p), p, points(p, 5)[1])
        _, st = _decode_rm_state(th, planet_indices(th), [OS_P]; t_ref = d.t_ref)
        @test !st.ld_uniform
        @test st.ld.u_n[2] === st.u1 && st.ld.u_n[3] === st.u2
        ws = new_ws(p)
        rv_log_likelihood(th, d, ws)
        @test (@allocated rv_log_likelihood(th, d, ws)) < 3000
    end

    @testset "sky position in two parts is the one-piece formula" begin
        # planet_sky_position as it was written before it was split into
        # _sky_phase and _sky_from_phase.
        function sky_ref(t, P, e, ω, Tp, b, a_Rs)
            M = 2π * (t - Tp) / P
            E = Nereus.kepler_solve(M, e)
            f = Nereus.true_anomaly(E, e)
            r = a_Rs * (1 - e * e) / (1 + e * cos(f))
            one_minus_e2 = 1 - e * e
            cosi = b * (1 + e * sin(ω)) / max(a_Rs * one_minus_e2, eps())
            cosi = clamp(cosi, -1.0, 1.0)
            sini = sqrt(max(1 - cosi * cosi, zero(cosi)))
            fw = f + ω
            return (r * (-cos(fw)), r * (sin(fw) * cosi), r * (sin(fw) * sini))
        end
        rng = MersenneTwister(31)
        for _ in 1:2000
            P = 0.5 + 10rand(rng); e = rand(rng, (0.0, 0.3 * rand(rng), 0.9 * rand(rng)))
            ω = 2π * rand(rng) - π; Tp = 2459000 + 5randn(rng); t = Tp + 20randn(rng)
            b = 1.2rand(rng); a = 1.5 + 20rand(rng)
            @test all(Nereus.planet_sky_position(t, P, e, ω, Tp, b, a) .=== sky_ref(t, P, e, ω, Tp, b, a))
        end
    end

    # With noise models the workspace RV path and the allocating one are the
    # same arithmetic, so the allocating path is the reference. Without them
    # the workspace path has always computed the Keplerian differently (the
    # half-angle form of its velocity cache), so there the reference is the
    # workspace path with no history: a fresh workspace per call.
    @testset "RV orbital phases kept across calls ($label)" for (label, kw, ref) in (
            ("fixed ephemeris, oscillators", (P = OS_P, Tc = OS_TC0), :alloc),
            ("free P, Tc, e and ω", (P = (OS_P, 1e-4), Tc = (OS_TC0, 1e-3), ecc = :free), :alloc),
            ("point occultation", (occultation = :point, ecc = :free), :alloc),
            ("no noise models", (tomo_noise = :white, rv_noise = :white), :fresh),
            ("no noise models, free ephemeris",
             (tomo_noise = :white, rv_noise = :white, P = (OS_P, 1e-4), ecc = :free), :fresh))
        p = build(; kw...)
        ws = new_ws(p)
        th = Theta{Float64}(p)
        pts = points(p, 40)
        rng = MersenneTwister(5)
        nsame = 0; n = 0
        # Each point twice in a row, then in random order: the cache is both
        # hit and refreshed.
        for x in vcat(repeat(pts; inner = 2), pts[randperm(rng, length(pts))])
            seat!(th, p, x)
            a = rv_log_likelihood(th, d, ws)
            b = ref === :alloc ? rv_log_likelihood(th, d) : rv_log_likelihood(th, d, new_ws(p))
            nsame += a === b; n += 1
        end
        @test nsame == n
    end

    @testset "kernel factors from distance groups" begin
        rng = MersenneTwister(8)
        upper_same(A, B) = all(A[i, j] === B[i, j] for j in axes(A, 2) for i in 1:j)
        grids = (collect(range(-42.0, 42.0; length = 57)),          # NGTS-33 velocities
                 collect(range(-42, 42; length = 25)),
                 2459986.4 .+ sort(rand(rng, 37)) ./ 5,             # exposure times
                 [1.0, 1.0, 2.5, 2.5, 7.0],                          # repeated values
                 [3.0])
        for x in grids
            kd = KernelDistances(x)
            n = length(x)
            for ℓ in (0.01, 1.7, 8.0, 59.9)
                K = fill(NaN, n, n)
                @test upper_same(_matern_upper!(K, kd, ℓ, 1.0, false), _kern(x, ℓ))
                σ2 = 0.37^2
                @test upper_same(_matern_upper!(K, kd, ℓ, σ2, true), σ2 .* _kern(x, ℓ))
            end
            for (S0, Q, w0) in ((2.0, 0.3, 5.0), (1e3, 4.0, 20.0))
                c = sho_coefficients(S0, Q, w0)
                K = fill(NaN, n, n)
                @test upper_same(_celerite_upper!(K, kd, c...), celerite_kernel_dense(x, c...))
            end
        end
        # A uniform grid has one distance per lag: 57 kernel evaluations, not 1653.
        @test length(KernelDistances(grids[1]).d) == 57
    end

    @testset "dsyevr through preallocated buffers is eigen(Symmetric(K))" begin
        rng = MersenneTwister(4)
        for n in (1, 2, 5, 14, 25, 37, 57)
            ew = SymEigenWork(n)
            for trial in 1:5
                x = sort(randn(rng, n)) .* 10
                K = trial == 1 ? _kern(x, 3.0) : (A = randn(rng, n, n); A' * A)
                E = eigen(Symmetric(K))
                @test _sym_eigen!(ew, K)
                @test all(ew.W .=== E.values)
                @test all(ew.Z .=== E.vectors)
            end
        end
        # Where eigen throws, the workspace path throws the same.
        p = build()
        th = seat!(Theta{Float64}(p), p, points(p, 5)[1])
        set_param!(th, "matern_sigma_tomo_A", Inf)
        e1 = try tomogram_log_likelihood(th, d); nothing catch err; err end
        e2 = try tomogram_log_likelihood(th, d, new_ws(p)); nothing catch err; err end
        @test e1 !== nothing
        @test typeof(e1) == typeof(e2)
        @test sprint(showerror, e1) == sprint(showerror, e2)
    end

    # rjmcmc checkpoints a workspace with `_ws_snapshot`. The obliquity scratch
    # is rebuilt on first use, so it is left out, and a workspace restored from
    # the snapshot of a warm one gives the same bits.
    @testset "checkpoint snapshot leaves the obliquity scratch out" begin
        p = build()
        ws = new_ws(p)
        th = Theta{Float64}(p)
        pts = points(p, 10)
        ref = Float64[]
        for x in pts
            seat!(th, p, x)
            push!(ref, rv_log_likelihood(th, d, ws) + tomogram_log_likelihood(th, d, ws))
        end
        @test ws.tomo !== nothing && ws.rv_orbit !== nothing
        snap = Nereus._ws_snapshot(ws)
        @test isempty(intersect(keys(snap), (:tomo, :rv_channel, :rv_orbit)))
        ws2 = Nereus._ws_restore!(new_ws(p), snap)
        @test ws2.tomo === nothing && ws2.rv_channel === nothing && ws2.rv_orbit === nothing
        got = Float64[]
        for x in pts
            seat!(th, p, x)
            push!(got, rv_log_likelihood(th, d, ws2) + tomogram_log_likelihood(th, d, ws2))
        end
        @test all(got .=== ref)
    end
end
