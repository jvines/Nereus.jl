# The epoch-astrometry orbit: abscissae on the sky, along their scan axes.
#
# `plot_epoch_astrometry_orbit` existed but no run ever drew it -- it had no
# plot-kind name, so `run_job` and `fit_*(plots = ...)` could not ask for it,
# and an IAD-only fit came back with a bare model ellipse (`orbit_skyplane`
# has nothing to overlay for 1-D data). The version that did exist plotted the
# per-parameter MEDIAN orbit (χ²/N 1.65 on Gaia-4 against 1.38 at the max-lp
# draw), kept its own copy of the catalogue design, ignored every companion but
# the one drawn, labelled the apocentre distance "a₀", and scattered 824
# unbinned CCD abscissae along their scan lines.
#
# Pinned here: the O−C are the likelihood's (Σ(O−C/σ)² is χ²_min), they vanish
# on noiseless data with one companion or two, normal points are one per FoV
# transit and never span a change of scan angle or instrument, and the runner
# knows the figure by name and draws it.

using Nereus, Test, Random, MCMCChains, LinearAlgebra
using Nereus: IADData

# Gaia-like epoch astrometry: FoV transits of 9 CCD abscissae read 4.9 s apart
# at one scan angle, drawn from a known orbit (or two) plus a catalogue
# solution. `inst2` puts the second half of the transits in a second
# instrument with its own along-scan zero point. `active` makes the chain
# trans-dim shaped (`n_planets`, `planet_active_k`) with only those slots
# present -- and only those slots in the data; the others stay parked at
# real, massive orbits, as samplers leave them.
function _epoch_orbit_target(; n_fov = 40, σ = 0.08, seed = 11, two = false,
                               noise = true, inst2 = false, active = nothing)
    rng = MersenneTwister(seed)
    t0 = 57000.0
    tf = sort(t0 .+ 1800 .* rand(rng, n_fov))
    t, psi, inst = Float64[], Float64[], Int[]
    for (i, tk) in enumerate(tf)
        ψ = 2π * rand(rng)
        for c in 0:8
            push!(t, tk + c * 5.6e-5); push!(psi, ψ)
            push!(inst, inst2 && i > n_fov ÷ 2 ? 2 : 1)
        end
    end
    n = length(t)
    plxf = sin.(2π .* (t .- t0) ./ 365.25 .- psi)
    pmf  = (t .- 57388.5) ./ 365.25
    iad = IADData(t = t, abscissa = zeros(n), abscissa_err = fill(σ, n),
                  psi = psi, parallax_factor = plxf, pm_factor = pmf,
                  inst = inst,
                  ref_params = inst2 ? [ntuple(_ -> 0.0, 5), ntuple(_ -> 0.0, 5)] : nothing,
                  abscissa_kind = inst2 ? [:absolute, :absolute] : nothing)
    pl = (a = LogUniformPrior(0.3, 4.0), M_sec = LogUniformPrior(0.001, 0.05),
          sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
          inc = SinePrior(), Omega = UniformPrior(0.0, 2pi),
          Mo = UniformPrior(0.0, 2pi))
    target = build_target(M_pri = 0.644, planets = two ? (b = pl, c = pl) : (b = pl,),
                          iad = iad, plx = NormalPrior(13.6, 0.02), M_s = 0.644)
    truth = Dict("a_k1" => 1.4, "M_sec_k1" => 0.011, "sesinw_k1" => 0.45,
                 "secosw_k1" => 0.3, "inc_k1" => 2.1, "Omega_k1" => 3.0,
                 "Mo_k1" => 1.0, "plx" => 13.6)
    two && merge!(truth, Dict("a_k2" => 0.5, "M_sec_k2" => 0.008,
                              "sesinw_k2" => 0.1, "secosw_k2" => -0.2,
                              "inc_k2" => 1.0, "Omega_k2" => 0.7, "Mo_k2" => 4.0))
    theta = Theta{Float64}(target.params)
    for (k, v) in truth
        set_param!(theta, k, v)
    end
    _, orbs, M_secs = Nereus._iad_active_orbits(theta, 0.644, 13.6, target.data.t_ref)
    inj = active === nothing ? eachindex(orbs) : active
    for j in 1:n
        reflex = sum(Nereus.along_scan_projection(
                         Nereus.star_reflex_offset(orbs[k], t[j], M_secs[k])..., psi[j])
                     for k in inj; init = 0.0)
        zp = inst[j] == 1 ? (1.3, -0.7) : (-4.0, 2.5)
        target.data.iad.abscissa[j] = reflex + 13.6 * plxf[j] +
            zp[1] * sin(psi[j]) + zp[2] * cos(psi[j]) +
            40.0 * sin(psi[j]) * pmf[j] - 25.0 * cos(psi[j]) * pmf[j] +
            (noise ? σ * randn(rng) : 0.0)
    end
    # Chains: the truth at the maximum lp, perturbed draws below it.
    nm = target.params.layout.unfrozen_names
    rows = [[truth[x] for x in nm]]
    for _ in 1:5
        push!(rows, [truth[x] * (1 + 0.05 * randn(rng)) for x in nm])
    end
    td_cols = active === nothing ? Symbol[] :
              vcat(:n_planets, [Symbol("planet_active_$k") for k in 1:length(orbs)])
    td_vals = active === nothing ? Float64[] :
              vcat(length(active), [k in active ? 1.0 : 0.0 for k in 1:length(orbs)])
    arr = zeros(length(rows), length(nm) + length(td_cols) + 1, 1)
    for (i, r) in enumerate(rows)
        arr[i, 1:length(nm), 1] = r
        arr[i, length(nm)+1:end-1, 1] = td_vals
        arr[i, end, 1] = i == 1 ? 0.0 : -100.0 * i
    end
    chains = Chains(arr, vcat(Symbol.(nm), td_cols, :lp))
    return target, theta, chains
end

_chi2_min(theta, data) = let iad = data.iad, n_inst = Nereus.n_iad_inst(iad),
                             n_q = Nereus._iad_n_q(n_inst)
    _, orbs, Ms = Nereus._iad_active_orbits(theta, astrom_M_pri(theta),
                                            astrom_plx(theta), data.t_ref)
    r = zeros(n_iad(iad))
    Nereus._iad_residuals!(r, iad, orbs, Ms, astrom_plx(theta))
    A = zeros(n_q, n_q); v = zeros(n_q)
    rWr, _ = Nereus._iad_normal_equations!(A, v, iad, r, iad.pm_factor,
                                           Nereus._iad_pos_cols(n_inst))
    first(Nereus._iad_solve(A, v, rWr, n_q))
end

@testset "epoch astrometry orbit" begin

    @testset "normal points: one per FoV transit, never across ψ or instrument" begin
        t   = [0.0, 5e-5, 1e-4, 0.074, 0.07405, 0.2, 0.20005, 0.20010, 0.0741]
        psi = [1.0, 1.0,  1.0,  1.0,   1.0,     2.0, 2.0,     0.5,     1.0]
        err = [0.1, 0.2,  0.1,  0.1,   0.1,     0.1, 0.1,     0.1,     0.1]
        inst = [1, 1, 1, 1, 2, 1, 1, 1, 2]
        iad = IADData(t = t, abscissa = zeros(9), abscissa_err = err, psi = psi,
                      inst = inst)
        e = [1.0, 4.0, 1.0, 2.0, 3.0, -1.0, 1.0, 7.0, 5.0]
        np = Nereus._iad_normal_points(iad, e, 0.01)
        # {1,2,3}; {4} (next FoV transit); {6,7}; {8} (ψ changes);
        # {5,9} (instrument 2 -- 4 and 5 are 3 s apart but never merge)
        @test np.size == [3, 1, 2, 1, 2]
        w = 1 ./ err[1:3] .^ 2
        @test np.e[1] ≈ sum(w .* e[1:3]) / sum(w)
        @test np.σ[1] ≈ 1 / sqrt(sum(w))
        @test np.t[1] ≈ sum(w .* t[1:3]) / sum(w)
        @test np.ux[1] ≈ sin(1.0)
        @test np.uy[1] ≈ cos(1.0)
        @test np.e[3] ≈ 0.0
        @test np.inst == [1, 1, 1, 1, 2]
        # Hipparcos-like singletons pass through untouched.
        np1 = Nereus._iad_normal_points(iad, e, 0.0)
        @test all(==(1), np1.size)
        @test sort(np1.e) == sort(e)
    end

    @testset "O−C are the likelihood's: Σ(O−C/σ)² == χ²_min" begin
        target, theta, _ = _epoch_orbit_target()
        fit = Nereus._epoch_astrometry_oc(theta, target.data, 1)
        @test fit !== nothing
        # χ²_min is rᵀWr − vᵀA⁻¹v, which cancels the whole proper-motion
        # signal, so it agrees to round-off, not to the last bit. A wrong
        # design column is off at O(1).
        @test sum(abs2, fit.oc ./ target.data.iad.abscissa_err) ≈
              _chi2_min(theta, target.data) rtol = 1e-6
        # and they look like the injected noise
        @test 0.7 < sum(abs2, fit.oc ./ target.data.iad.abscissa_err) / length(fit.oc) < 1.3
    end

    @testset "noiseless: every abscissa lands on the orbit" begin
        for (two, inst2) in ((false, false), (true, false), (false, true))
            target, theta, _ = _epoch_orbit_target(; noise = false, two, inst2)
            for k in 1:(two ? 2 : 1)
                # With two companions the OTHER one's reflex must already be
                # gone from the O−C, or planet k's points would sit off its
                # orbit by the other's along-scan signal (~0.1-0.3 mas here).
                fit = Nereus._epoch_astrometry_oc(theta, target.data, k)
                @test maximum(abs, fit.oc) < 1e-9
            end
        end
    end

    @testset "nothing to draw" begin
        target, theta, _ = _epoch_orbit_target()
        @test Nereus._epoch_astrometry_oc(theta, target.data, 2) === nothing
        @test Nereus._epoch_astrometry_oc(theta, Data(t_rv = [0.0, 1.0],
                                          rv = [0.0, 0.0], rv_err = [1.0, 1.0]), 1) === nothing
    end

    @testset "the figure, per companion" begin
        target, _, chains = _epoch_orbit_target(; two = true)
        out = mktempdir()
        for k in 1:2
            fig = plot_epoch_astrometry_orbit(chains, target.params, target.data;
                                               planet_idx = k, output = out)
            @test !isempty(fig.content)
            @test isfile(joinpath(out, "models", "epoch_astrometry_orbit_K$k.png"))
        end
        # no third companion: an empty figure and no file
        fig3 = plot_epoch_astrometry_orbit(chains, target.params, target.data;
                                            planet_idx = 3, output = out)
        @test isempty(fig3.content)
        @test !isfile(joinpath(out, "models", "epoch_astrometry_orbit_K3.png"))
    end

    @testset "trans-dim: parked slots are neither subtracted nor drawn" begin
        # Slot 2 exists in the layout and carries a massive parked orbit, but
        # no draw has it active and the data contain only slot 1. Without the
        # draw's active set on theta, slot 2's reflex was subtracted from
        # slot 1's O−C (0.16 mas against σ = 0.08) and drawn as a companion.
        target, _, chains = _epoch_orbit_target(; two = true, noise = false,
                                                  active = [1])
        θ1 = first(Nereus._planet_draw(chains, target.params, 1; bf_cutoff = 5.0))
        @test collect(Nereus.planet_indices(θ1)) == [1]
        @test maximum(abs, Nereus._epoch_astrometry_oc(θ1, target.data, 1).oc) < 1e-9
        @test Nereus._planet_draw(chains, target.params, 2; bf_cutoff = 5.0) === nothing
        out = mktempdir()
        @test isempty(plot_epoch_astrometry_orbit(chains, target.params, target.data;
                                                  planet_idx = 2, output = out).content)
        @test !isfile(joinpath(out, "models", "epoch_astrometry_orbit_K2.png"))

        # A death mid-list: slot 1 gone, slot 2 alive, n_planets == 1. Going
        # by `n_planets ≥ k` would call slot 2 absent and slot 1 present.
        target, _, chains = _epoch_orbit_target(; two = true, noise = false,
                                                  active = [2])
        θ2 = first(Nereus._planet_draw(chains, target.params, 2; bf_cutoff = 5.0))
        @test collect(Nereus.planet_indices(θ2)) == [2]
        @test maximum(abs, Nereus._epoch_astrometry_oc(θ2, target.data, 2).oc) < 1e-9
        @test Nereus._planet_draw(chains, target.params, 1; bf_cutoff = 5.0) === nothing
        # and the runner publishes only what it wrote
        out = mktempdir()
        @test Nereus._dispatch_plot("epoch_astrometry_orbit", chains, target.params,
                                    target.data, out, Dict{Symbol, Any}()) ==
              "models/epoch_astrometry_orbit_K*.png"
        @test readdir(joinpath(out, "models")) == ["epoch_astrometry_orbit_K2.png"]
    end

    @testset "the runner knows it by name" begin
        target, _, chains = _epoch_orbit_target()
        @test "epoch_astrometry_orbit" in Nereus._KNOWN_PLOTS
        @test "epoch_astrometry_orbit" in
              Nereus._auto_plot_kinds(chains, target.params, target.data)
        out = mktempdir()
        # an unrelated shared plot kwarg must not cost this figure
        kw = Dict{Symbol, Any}(:save_pdf => false, :n_draws => 50)
        pat = Nereus._dispatch_plot("epoch_astrometry_orbit", chains,
                                    target.params, target.data, out, kw)
        @test pat == "models/epoch_astrometry_orbit_K*.png"
        @test isfile(joinpath(out, "models", "epoch_astrometry_orbit_K1.png"))
        # IAD but no companion ever present: nothing written, and no pattern
        # published for `_warn_unmatched_plot_patterns` to call stale.
        target, _, chains = _epoch_orbit_target(; two = true, active = Int[])
        out = mktempdir()
        @test Nereus._dispatch_plot("epoch_astrometry_orbit", chains, target.params,
                                    target.data, out, Dict{Symbol, Any}()) === nothing
        @test !isdir(joinpath(out, "models")) ||
              isempty(readdir(joinpath(out, "models")))
    end
end
