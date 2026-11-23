# fit_binary(; secondary) is the double-lined (SB2) route. It sent planet_mode
# "SB2", which run_job's validator rejects (the modes are BINARY / BINARY_RV),
# and put the secondary RVs under a `rv_secondary` key nothing reads. SB2 is the
# BINARY_RV mode plus a `component` tag on each RV row.

using Nereus, Test, Random

function _sb2_rvs(rng; n = 30, P = 12.0, KA = 15.0, KB = 20.0, e = 0.1, ω = 0.7)
    t = sort(rand(rng, n)) .* 60.0 .+ 2.46e6
    M = 2π .* (t .- 2.46e6) ./ P
    E = copy(M)
    for _ in 1:30; E .= M .+ e .* sin.(E); end
    f = 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
    shape = cos.(f .+ ω) .+ e * cos(ω)
    err = fill(0.1, n)
    a = (t = t, rv = KA .* shape .+ 0.05 .* randn(rng, n), rv_err = err)
    b = (t = t, rv = -KB .* shape .+ 0.05 .* randn(rng, n), rv_err = err)
    return a, b
end

@testset "fit_binary with a secondary runs as a double-lined binary" begin
    a, b = _sb2_rvs(MersenneTwister(1))
    dir = mktempdir()
    r = Nereus.fit_binary(; rv = (S = a,), secondary = (S = b,), output_dir = dir,
        engine = Dict("engine" => "pt_emcee",
                      "options" => Dict("n_temps" => 2, "n_walkers" => 40,
                                        "n_steps" => 40, "n_burnin" => 10)))
    @test r.summary["op"] == "fit_binary"

    txt = join(readlines(joinpath(dir, "summary.json")), "\n")
    for key in ("mass_ratio_q", "M_A_sini3", "M_B_sini3", "a_sini_au")
        @test occursin(key, txt)
    end
end
