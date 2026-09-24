# Physical constants for Nereus.
#
# Values taken from IAU 2015 resolution B3 (nominal solar conversion
# constants) and CODATA 2018 where applicable. SI units unless noted.
#
# These are `const` bindings so the compiler can constant-fold them.

# --- Time ---
const DAY_SECONDS = 86400.0              # s
const YEAR_SECONDS = 365.25 * DAY_SECONDS # s (Julian year)
# The year in which Kepler's third law reads P²[yr] = a³[AU] / M_total[M_sun]
# EXACTLY, i.e. the one for which G ≡ 4π² in (M_sun, AU, yr): 2π/k days, with
# k = 0.01720209895 the Gaussian gravitational constant. Every a ↔ P conversion
# in those units uses it; the Julian year is 1.89e-5 short of it, and the
# difference is not cosmetic. PlanetOrbits recomputes an orbit's period from
# (a, M) with this value, so a Julian-year `a_from_P` handed it an orbit whose
# period was NOT the sampled P: astrometry then ran at P·(1 + 1.89e-5) while the
# RV likelihood ran at P (the two drift apart over the baseline of a joint fit),
# and Mo → Mo + 2π, which src/circular.jl relabels by, moved the astrometric
# mean anomaly by 2π(1 − 1.89e-5) instead of exactly one turn.
# Julian years remain right for what they measure: epoch differences, proper
# motions, and the P_yr column of a science table.
const KEPLER_YEAR_DAYS = 365.2568983840419

# --- Gravitation ---
const G_SI  = 6.67430e-11                 # m^3 kg^-1 s^-2  (CODATA 2018)
const G_CGS = 6.67430e-8                  # cm^3 g^-1 s^-2

# GM_sun, IAU 2015 B3 nominal (S^N_☉). An orbit measures GM, never G and M
# separately: GM_sun is known to ten digits, G to five. So every GM_sun in
# Nereus comes from HERE, and the solar mass below is derived from it rather
# than the other way round.
#
# It used to be the other way round, in four conventions at once: G_CGS·M_SUN_G
# (2.57e-4 high, in the derived semi-major axis and the RV mass function),
# `_G_SI·_M_SUN_KG` with its own 1.989e30 (astrometric mass function),
# `6.674e-11 · 1.989e30` (gravity darkening), and a hardcoded 1.3271244e26 in
# six more places, which was the only right one. Two routes to the same `a`
# then disagreed by 8.6e-5, and the RV and astrometric mass functions did not
# use the same GM.
const GM_SUN_SI  = 1.32712440018e20       # m^3 s^-2
const GM_SUN_CGS = 1.32712440018e26       # cm^3 s^-2

# --- Sun (IAU 2015 nominal) ---
# Derived from GM (see above), so G·M_SUN is GM_SUN by construction. The old
# hardcoded 1.98892e30 kg is the pre-2010 solar mass and is 2.6e-4 heavy
# against this GM; M_sun/M_jup came out 1047.83 instead of 1047.57.
const M_SUN_KG  = GM_SUN_SI / G_SI        # kg ≈ 1.98841e30
const M_SUN_G   = GM_SUN_CGS / G_CGS      # g
const R_SUN_M   = 6.9570e8                # m
const R_SUN_CM  = 6.9570e10               # cm
const L_SUN_W   = 3.828e26                # W

# --- Jupiter ---
const M_JUP_KG  = 1.89813e27              # kg
const M_JUP_G   = 1.89813e30              # g
const R_JUP_M   = 7.1492e7                # m
const R_JUP_CM  = 7.1492e9                # cm

# --- Earth ---
const M_EARTH_KG = 5.9722e24              # kg
const R_EARTH_M  = 6.3781e6               # m

# --- Unit conversions ---
const AU_M  = 1.49597870700e11            # m
const AU_CM = 1.49597870700e13            # cm
const PARSEC_M = 3.0856775814913673e16    # m

# --- Stellar ratios (handy derived) ---
const RSUN_PER_RJUP = R_SUN_M / R_JUP_M
const MSUN_PER_MJUP = M_SUN_KG / M_JUP_KG

# --- Radiation ---
const SIGMA_SB_SI = 5.670374419e-8        # W m^-2 K^-4
const SIGMA_SB_CGS = 5.670374419e-5       # erg cm^-2 s^-1 K^-4

# --- Convenience math ---
const TWO_PI = 2π
const HALF_PI = π / 2
