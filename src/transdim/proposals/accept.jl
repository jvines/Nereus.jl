# The RJMCMC accept/reject rule (Green 1995).
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# RJMCMC accept/reject
# =====================================================================

"""
    rjmcmc_accept(log_L_old, log_L_new, log_π_old, log_π_new, log_q_ratio, rng) -> Bool

Standard Green (1995) reversible-jump acceptance.
"""
function rjmcmc_accept(log_L_old::Real, log_L_new::Real,
                        log_π_old::Real, log_π_new::Real,
                        log_q_ratio::Real, rng::AbstractRNG)
    log_α = (log_L_new - log_L_old) + (log_π_new - log_π_old) + log_q_ratio
    return log(rand(rng)) < log_α
end

