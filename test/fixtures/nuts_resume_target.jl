# The target of the sample_nuts resume tests (test_nuts_resume.jl), in a file of
# its own so that the child process of the queued-chain test builds the same one.
#
# An eccentric orbit with Mo = 0 at the reference epoch (the median time): its
# posterior straddles Mo's 0/2π seam, so the warm start moves Mo's window, which
# a resume must then put back. The pre-search is long enough (1200 steps) to
# find the mode: it moved the window for every seed tried, where 200 steps moved
# it for some and not others.
using Statistics: median

const _NR_N = 60
const _NR_T, _NR_RV = let r = MersenneTwister(7)
    t = sort!(100 .* rand(r, _NR_N))
    t, [Nereus.rv_keplerian(ti, 4.23, 40.0, 0.4, 1.0, 0.0, median(t)) for ti in t] .+
       1.5 .* randn(r, _NR_N)
end

# A fresh target per run: the warm start moves circular windows on the target.
_nr_target(rv = _NR_RV) = build_target(
    planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _NR_T, rv = rv, rv_err = fill(1.5, _NR_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))

const _NR_KW = (n_warmup = 200, n_chains = 4, warm_temps = 4, warm_walkers = 20,
                warm_steps = 1200, warm_burnin = 600, progress = false)
