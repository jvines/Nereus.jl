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
              set_param!, _sym_eigen!, _kern, obliquity_noise_menu
using LinearAlgebra: Symmetric, eigen
using Random
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
end
