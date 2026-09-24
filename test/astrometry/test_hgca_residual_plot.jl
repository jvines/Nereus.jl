# The HGCA residual figure pictures the likelihood's model, not a sketch of it.
#
# `plot_pm_residuals` took planet `planet_idx`'s INSTANTANEOUS reflex PM at all
# three epochs. `hgca_log_likelihood` sums every active companion, uses the
# mean reflex velocity over the baseline at the Hipparcos–Gaia epoch (Brandt
# 2021 Eq. 1), and GOST Mode B at the Gaia epoch. So the residuals and the
# "marginalized χ²" in the figure were not the fit's whenever a second
# companion existed or the period was comparable to the 25-yr baseline. Both
# now come from `_hgca_model_pm`.

using Nereus, Test, MCMCChains

@testset "HGCA residual figure uses the likelihood's model" begin
    epochs = mjd_epochs((1991.25, 2004.0, 2016.0))
    mk(pmra, pmdec) = HGCAData(; epochs = epochs, pmra = pmra, pmdec = pmdec,
                               sigma_pmra = (0.5, 0.02, 0.05),
                               sigma_pmdec = (0.5, 0.02, 0.05),
                               plx = 20.0, plx_err = 0.05, hip_id = 1)
    pl = (a = LogUniformPrior(0.3, 40.0), M_sec = LogUniformPrior(0.001, 0.5),
          sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
          inc = SinePrior(), Omega = UniformPrior(0.0, 2pi),
          Mo = UniformPrior(0.0, 2pi))
    # GOST scans over the Gaia window: the Gaia epoch goes through Mode B in
    # the likelihood, which the figure's instantaneous reflex did not.
    tg = collect(range(mjd_epochs((2014.7,))[1], mjd_epochs((2017.3,))[1]; length = 40))
    gost = GOSTData(; t = tg, psi = 2π .* ((1:40) ./ 41), parallax_factor = sin.(tg ./ 58.0))
    target_for(h) = build_target(M_pri = 1.0, planets = (b = pl, c = pl), hgca = h,
                                 gost = gost, plx = NormalPrior(20.0, 0.05), M_s = 1.0)
    # Two companions, the outer with P ≈ 23 yr — comparable to the baseline.
    truth = Dict("a_k1" => 8.0, "M_sec_k1" => 0.06, "sesinw_k1" => 0.3,
                 "secosw_k1" => 0.2, "inc_k1" => 1.1, "Omega_k1" => 2.0, "Mo_k1" => 1.0,
                 "a_k2" => 3.0, "M_sec_k2" => 0.03, "sesinw_k2" => -0.1,
                 "secosw_k2" => 0.3, "inc_k2" => 0.7, "Omega_k2" => 4.0, "Mo_k2" => 3.0,
                 "plx" => 20.0)
    target0 = target_for(mk((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    θ = Theta{Float64}(target0.params)
    for (k, v) in truth
        set_param!(θ, k, v)
    end
    μra, μdec = Nereus._hgca_model_pm(θ, target0.data.hgca, target0.data,
                                      astrom_M_pri(θ), astrom_plx(θ), target0.data.t_ref)
    # noiseless catalogue PMs: the model plus a barycentric motion
    target = target_for(mk(Tuple(μra .+ 120.0), Tuple(μdec .- 45.0)))

    nm = target.params.layout.unfrozen_names
    arr = zeros(3, length(nm) + 1, 1)
    for i in 1:3
        arr[i, 1:end-1, 1] = [truth[x] for x in nm]
        arr[i, end, 1] = i == 1 ? 0.0 : -10.0 * i
    end
    chains = Chains(arr, vcat(Symbol.(nm), :lp))

    fig = plot_pm_residuals(chains, target.params, target.data; planet_idx = 1)
    txt = only(filter(t -> occursin("χ²", t),
                      [string(p[:text][]) for c in fig.content if c isa Nereus.Axis
                       for p in c.scene.plots if p isa Nereus.CairoMakie.Makie.Text]))
    # the model reproduces the data exactly: χ² is zero. With planet 1's
    # instantaneous reflex alone, companion 2 and the baseline-averaged HG
    # epoch were residuals at 0.02-0.05 mas/yr errors.
    @test parse(Float64, match(r"χ² = ([0-9.eE+-]+)", txt)[1]) < 1e-6

    # PM anomaly: an observed catalogue point, less the marginalized
    # barycentric PM, lands on the likelihood's model at its epoch. Referenced
    # to the catalogue's Hipparcos–Gaia PM instead, it was off by the mean
    # reflex over the baseline.
    out = mktempdir()
    fig = plot_pm_anomaly(chains, target.params, target.data; planet_idx = 1,
                          output = out, n_draws = 5)
    @test isfile(joinpath(out, "models", "pm_anomaly_K1.png"))
    ax = first(c for c in fig.content if c isa Nereus.Axis)
    pts(lbl) = [Tuple(p[1][]) for p in ax.scene.plots
                if p isa Nereus.CairoMakie.Makie.Scatter && get(p.attributes, :label, nothing) !== nothing &&
                   p[:label][] == lbl]
    obs = only(pts("HGCA")); mod = only(pts("HGCA model"))
    @test maximum(abs(o[2] - m[2]) for (o, m) in zip(obs, mod)) < 1e-3   # Float32 points
end
