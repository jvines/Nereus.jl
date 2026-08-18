# Birth/death proposals for trans-dimensional MCMC.
#
# Implements the RJMCMC acceptance ratio (Green 1995). All proposal
# functions return (new_theta, log_q_ratio) where log_q_ratio is the
# log proposal ratio term in Green's formula. The caller computes the
# full acceptance:
#
#   log_α = (log_L_new - log_L_old) + (log_π_new - log_π_old) + log_q_ratio
#
# The prior ratio (log_π_new - log_π_old) is computed by the caller via
# log_prior(new_theta) - log_prior(old_theta). For prior birth, the
# prior terms from the new planet cancel with the proposal density,
# simplifying the ratio.

using Random
using Distributions


# Split by concern. The single file reached 2035 lines carrying planet births,
# noise births, swaps, annealed births, informed proposals and the astrometric
# coupling move — past the point where it can be held in one head. Order matters
# only for `const` definitions consumed at load time, which is why the shared
# accept rule and the informed-proposal constants come before their users.

include("proposals/planets.jl")
include("proposals/alias.jl")
include("proposals/noise.jl")
include("proposals/annealed.jl")
include("proposals/accept.jl")
include("proposals/moms.jl")
include("proposals/informed_gp.jl")
include("proposals/informed_ad.jl")
include("proposals/swaps.jl")
include("proposals/coupling.jl")
