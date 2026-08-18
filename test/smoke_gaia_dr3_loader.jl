using Nereus, Printf, LinearAlgebra

# Synthesized Gaia DR3-style row.
gaia_row = (
    ra = 21.04792, dec = 41.27516, parallax = 17.92,
    pmra = -89.30, pmdec = -57.42,
    ra_error = 0.022, dec_error = 0.018, parallax_error = 0.026,
    pmra_error = 0.030, pmdec_error = 0.024,
    ra_dec_corr        = -0.12,
    ra_parallax_corr   =  0.04, ra_pmra_corr        = -0.02,
    ra_pmdec_corr      =  0.08,
    dec_parallax_corr  = -0.05, dec_pmra_corr       =  0.06,
    dec_pmdec_corr     =  0.01,
    parallax_pmra_corr = -0.10, parallax_pmdec_corr =  0.03,
    pmra_pmdec_corr    = -0.18,
    ref_epoch          = 2016.0,
)

# 1) Default fiducial = Gaia row → α₀ = δ₀ = 0
g1 = load_gaia_dr3(gaia_row)
@printf("Default fiducial:\n")
@printf("  params: α₀=%.4f δ₀=%.4f ϖ=%.4f μα=%.4f μδ=%.4f\n", g1.params...)
@printf("  t_ref (MJD) = %.2f (J2016.0 expected ≈ 57388.5)\n", g1.t_ref)

# 2) Fiducial 1 mas SW of Gaia → expect α₀ ≈ +1, δ₀ ≈ +1 mas.
ra_hip  = gaia_row.ra  - 1.0 / (3.6e6 * cos(deg2rad(gaia_row.dec)))
dec_hip = gaia_row.dec - 1.0 / 3.6e6
g2 = load_gaia_dr3(gaia_row; ra_fiducial=ra_hip, dec_fiducial=dec_hip)
@printf("\nFiducial 1 mas SW of Gaia:\n")
@printf("  params: α₀=%.4f δ₀=%.4f ϖ=%.4f μα=%.4f μδ=%.4f\n", g2.params...)

# 3) Covariance check.
@printf("\nCov diag: %.6f %.6f %.6f %.6f %.6f\n", diag(g1.cov)...)
@printf("Expected: %.6f %.6f %.6f %.6f %.6f\n",
        gaia_row.ra_error^2, gaia_row.dec_error^2, gaia_row.parallax_error^2,
        gaia_row.pmra_error^2, gaia_row.pmdec_error^2)
expected_45 = gaia_row.pmra_pmdec_corr * gaia_row.pmra_error * gaia_row.pmdec_error
@printf("cov[4,5] = %.6e  expected %.6e\n", g1.cov[4,5], expected_45)
@printf("Σ posdef: %s\n", isposdef(Symmetric(g1.cov)))

# 4) Dict-keyed input.
gaia_dict = Dict(string(k) => v for (k, v) in pairs(gaia_row))
g3 = load_gaia_dr3(gaia_dict)
@printf("\nDict input matches NamedTuple: %s\n",
        g3.params == g1.params && g3.cov == g1.cov)

# 5) CSV roundtrip.
mktempdir() do dir
    p = joinpath(dir, "gaia.csv")
    open(p, "w") do io
        ks = collect(string.(keys(gaia_row)))
        vs = [string(getfield(gaia_row, Symbol(k))) for k in ks]
        println(io, join(ks, ","))
        println(io, join(vs, ","))
    end
    g4 = load_gaia_dr3(p)
    @printf("CSV input matches: %s\n",
            g4.params == g1.params && isapprox(g4.cov, g1.cov))
end

# 6) Missing-key error.
try
    bad_row = NamedTuple{Tuple(k for k in keys(gaia_row) if k != :pmra_pmdec_corr)}(
        Tuple(getfield(gaia_row, k) for k in keys(gaia_row) if k != :pmra_pmdec_corr))
    load_gaia_dr3(bad_row)
    println("FAIL: expected ArgumentError on missing key")
catch e
    if e isa ArgumentError && occursin("pmra_pmdec_corr", e.msg)
        println("Missing-key error: caught ArgumentError correctly")
    else
        println("FAIL: wrong error type: $(typeof(e)) — $(e)")
    end
end

# 7) End-to-end with iad_log_likelihood joint dispatch (degenerate orbit).
n_h = 30
t_h = 48349.0 .+ collect(range(0, 3*365.25; length=n_h))
ψ_h = collect(range(0, 2π; length=n_h))
σ_h = 1.5 .* ones(n_h)
abscissa = randn(n_h) .* σ_h
plxf = 0.3 .* sin.(2π .* (t_h .- 48349.0) ./ 365.25)
pmf = (t_h .- 48349.0) ./ 365.25
iad = IADData(t=t_h, abscissa=abscissa, abscissa_err=σ_h, psi=ψ_h,
               parallax_factor=plxf, pm_factor=pmf)
n_g = 30
t_g = 56863.0 .+ collect(range(0, 1080; length=n_g))
ψ_g = collect(range(0, 2π; length=n_g))
plxf_g = 0.3 .* sin.(2π .* (t_g .- 56863.0) ./ 365.25)
gost = GOSTData(t=t_g, psi=ψ_g, parallax_factor=plxf_g)
relast = RelAstromData(t=[t_h[1]], ra_off=[0.0], dec_off=[0.0],
                       ra_err=[10.0], dec_err=[10.0])

planets_spec = (b = (P=1500.0, M_sec=0.001,
                     sesinw=Nereus.ew_to_sesinw(0.1, 0.3)[1],
                     secosw=Nereus.ew_to_sesinw(0.1, 0.3)[2],
                     Mo=0.0, inc=1.2, Omega=1.5),)
target = build_target(M_pri=0.85, plx=g1.params[3],
                      planets=planets_spec, relAST=relast, iad=iad,
                      gost=gost, gaia_dr3=g1)
td = Nereus.TransDimState(max_planets=1)
Nereus.activate_planet!(td, 1)
th = Theta(target.params; td=td)
Nereus._init_systemics_from_prior!(th, Random.MersenneTwister(1))
ll = Nereus.iad_log_likelihood(th, target.data)
@printf("\nJoint IAD+Gaia loaded-from-row LL: %.4f (finite: %s)\n",
        ll, isfinite(ll))
