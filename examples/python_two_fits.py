"""The two workshop fits, from Python.

    ipython
    %run python_two_fits.py

One warm daemon does both, so the ~20 s `using Nereus` is paid once. First
ever run also fetches the runtime bundle (~465 MB) and precompiles it.

Julia equivalents: examples/01_gaia4_astrometry_only.ipynb and
examples/02_hd114762_joint_rv_astrometry.ipynb.
"""
# Prior dicts use "type", NOT "dist" -- api.jl:557 reads get(d, "type").
# Values: normal | uniform | loguniform | fixed | sine.
# (run_job JSON configs use a DIFFERENT vocabulary: {"type":"NormalPrior",
#  "args":[...]}. These two are not interchangeable.)
import math
import os
import astronereus
from astronereus import engines, RV, Astrometry

# ---------------------------------------------------------------- session
# One session, reused. Threads default to the machine; an 8-core box, so
# these take roughly twice the times quoted below (measured on 16).
s = astronereus.session().start()
print("daemon:", s.ping())

GAIA4_SID    = 1457486023639239296
HD114762_SID = 3937211745905473024

# ================================================================ FIT 1
# Gaia-4: astrometry alone. ~3 min on 16 cores.
#
# The parallax prior is not optional. The abscissae constrain the photocentre
# wobble a0 ~ M_sec * plx, so mass and distance trade off exactly without an
# independent distance. M_pri and plx are BOTH priors and live in `priors`;
# there are no `parallax=` / `m_pri=` keywords.
gaia4 = s.fit_astrometry(
    iad={"catalogue": "gaia_dr4", "source_id": GAIA4_SID},
    planets=1,
    priors={
        "plx":       {"type": "normal", "mu": 13.628, "sigma": 0.021},
        "M_pri":     {"type": "fixed",  "value": 0.644},
        "a_k1":      {"type": "loguniform", "lo": 0.3,   "hi": 4.0},
        "M_sec_k1":  {"type": "loguniform", "lo": 0.001, "hi": 0.05},
        "sesinw_k1": {"type": "uniform", "lo": -1.0, "hi": 1.0},
        "secosw_k1": {"type": "uniform", "lo": -1.0, "hi": 1.0},
        "inc_k1":    {"type": "sine"},
        "Omega_k1":  {"type": "uniform", "lo": 0.0, "hi": math.tau},
        "Mo_k1":     {"type": "uniform", "lo": 0.0, "hi": math.tau},
    },
    engine=engines.PTEmcee(n_temps=16, n_walkers=100, n_steps=3000,
                           n_burnin=1500, init_strategy="prior", seed=42),
    output_dir=os.path.expanduser("~/nereus_out/gaia4"),
)
print(gaia4)                       # .params .log_z .derived .figures .raw
for k in ("a_k1", "M_sec_k1", "inc_k1", "plx"):
    print(f"  {k:10s} {gaia4.params.get(k)}")
# published: P 571.3 d, M 11.8 MJup, e 0.338, i 116.9 deg

# ================================================================ FIT 2
# HD 114762: RV + Gaia DR4 astrometry jointly. ~3 min on 16 cores.
#
# n_temps=24, NOT the default 16. At 16 with this seed the eccentricity
# collapses to 0.002 against a published 0.335 -- silently, with tight error
# bars, and only `min swap acceptance` to show for it. 24 is also faster.
#
# The RV file ships inside the bundle.
rvfile = os.path.join(astronereus.runtime_dir(), "depot", "dev", "Nereus",
                      "test", "data", "hd114762_rv.dat")
rv = {}
for line in open(rvfile):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    t, v, e, ins = line.split()
    d = rv.setdefault(ins, {"t": [], "rv": [], "rv_err": []})
    d["t"].append(float(t) - 2_400_000.5)      # BJD -> MJD, as the notebook does
    d["rv"].append(float(v))
    d["rv_err"].append(float(e))
print({k: len(v["t"]) for k, v in rv.items()})   # j = Keck/HIRES, lick = Lick

hd = s.fit_joint(
    RV(data={"HIRES": rv["j"], "Lick": rv["lick"]}, trend_order=1),
    Astrometry(iad={"catalogue": "gaia_dr4", "source_id": HD114762_SID}),
    planets=1,
    priors={
        "plx":       {"type": "normal", "mu": 25.36, "sigma": 0.30},
        "M_pri":     {"type": "fixed",  "value": 0.83},
        # TIGHT prior on a, bracketing the 35-yr-established P = 83.92 d.
        # A targeted mass/inclination fit, not a blind period search.
        "a_k1":      {"type": "loguniform", "lo": 0.30,  "hi": 0.45},
        # reaches STELLAR -- if this stopped in the planetary regime the fit
        # could not find the answer. This is where the result is decided.
        "M_sec_k1":  {"type": "loguniform", "lo": 0.003, "hi": 0.5},
        "sesinw_k1": {"type": "uniform", "lo": -1.0, "hi": 1.0},
        "secosw_k1": {"type": "uniform", "lo": -1.0, "hi": 1.0},
        "inc_k1":    {"type": "sine"},
        "Omega_k1":  {"type": "uniform", "lo": 0.0, "hi": math.tau},
        "Mo_k1":     {"type": "uniform", "lo": 0.0, "hi": math.tau},
        "sigma_HIRES": {"type": "loguniform", "lo": 0.5, "hi": 50.0},
        "sigma_Lick":  {"type": "loguniform", "lo": 0.5, "hi": 50.0},
    },
    engine=engines.PTEmcee(n_temps=24, n_walkers=100, n_steps=3000,
                           n_burnin=1500, init_strategy="prior", seed=42),
    output_dir=os.path.expanduser("~/nereus_out/hd114762"),
)
print(hd)
for k in ("a_k1", "M_sec_k1", "inc_k1", "plx"):
    print(f"  {k:10s} {hd.params.get(k)}")
print("  derived:", hd.derived)
# expected, matching the Julia notebook:
#   e 0.333 +/- 0.008, i 3.48 deg, M_sec 0.198 +/- 0.003 Msun, plx 25.37
#   Kiefer+ 2019 gives 0.103 +0.030 -0.025 -- ours is +3.2 sigma from that.

# s.close()   # or s.stop(); the shared session also closes at exit
