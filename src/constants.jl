# Physical constants for Nereus.
#
# Values taken from IAU 2015 resolution B3 (nominal solar conversion
# constants) and CODATA 2018 where applicable. SI units unless noted.
#
# These are `const` bindings so the compiler can constant-fold them.

# --- Time ---
const DAY_SECONDS = 86400.0              # s
const YEAR_SECONDS = 365.25 * DAY_SECONDS # s (Julian year)

# --- Gravitation ---
const G_SI  = 6.67430e-11                 # m^3 kg^-1 s^-2  (CODATA 2018)
const G_CGS = 6.67430e-8                  # cm^3 g^-1 s^-2

# --- Sun (IAU 2015 nominal) ---
const M_SUN_KG  = 1.98892e30              # kg
const M_SUN_G   = 1.98892e33              # g
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
