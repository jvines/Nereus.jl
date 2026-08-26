# Gaia DR4 epoch astrometry: the data and the models

What the epoch-astrometry product actually contains, how it relates to Hipparcos IAD, the
models fitted to it, and how to fit them.

**This document is tool-agnostic on purpose.** Everything below is a property of the data, the
measurement geometry or the statistics — no package API, no code. It should be reproducible in
any language. A separate hands-on document covers one specific implementation.

**Provenance of every number.** Measured directly from the pre-release VOTable
`GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml` (1,183,282 bytes), published 2026-06-26 and
self-identifying as `release = "Gaia DR4_RC3"`.

**Scope warning, state it on slide 1.** That file is a *release candidate* containing **12
illustrative sources**. Production DR4 lands **2 December 2026**. Nothing here establishes
that the production release keeps this schema, populates the across-scan columns (empty here),
or ever sets `used_by_agis_ac` (false here). Every measured number is a fact about RC3.

Statements resting on inference rather than measurement are marked **[INFERENCE]**. Those are
the ones to phrase carefully in front of an audience.

---

# Part 1 — What the data is

## 1.1 What you get

A single VOTable, 1.18 MB for 12 sources, distributed as a zip from the ESA public data area
(`.../Gaia_DR4/dr4-prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip`).
Publisher `ESA/Gaia/DPAC`.

For each source: the individual **along-scan centroid measurements**, one per CCD crossing,
that AGIS itself consumed to build the catalogue solution. This is the input to the astrometric
solution, not a product derived from it.

## 1.2 File format: VOTable 1.4 with a base64 BINARY2 payload

Namespace `http://www.ivoa.net/xml/VOTable/v1.3`. Structurally trivial:

```
<VOTABLE version="1.4">
  <RESOURCE type="results" utype="spec:Spectrum">     <- 1 RESOURCE (the utype is spurious)
    <TIMESYS ID="time_frame" refposition="BARYCENTER"
             timeorigin="2455197.5" timescale="TCB"/>
    <TABLE>                                            <- 1 TABLE, no name
      9  x <PARAM>   (citation, release, help, issues, data, terms, home, publisher, creator)
      37 x <FIELD>   (the schema - 1.4)
      <DATA><BINARY2><STREAM encoding='base64'> ... </STREAM></BINARY2></DATA>
```

No `<TABLEDATA>`, no VOTable-1.0 `<BINARY>`. One table, one stream, 37 columns.

Practical consequence: **most VOTable readers in most languages will not open this.** BINARY2
support is far less common than TABLEDATA. Astropy handles it; many others do not. Expect to
decode the stream yourself, which is easy because the layout is rigid.

**Stream layout, verified byte-exact.** Base64 decodes to **867,888 bytes = 1008 records x
861 bytes**, remainder 0. One record:

| part | bytes |
|---|---|
| BINARY2 null mask, `ceil(37/8)` | 5 |
| 5 x 8-byte long/double scalars | 40 |
| 9 x 4-byte float scalars | 36 |
| 3 x 2-byte short scalars | 6 |
| 2 x 1-byte booleans | 2 |
| 18 x 4-byte array-length prefixes | 72 |
| array payloads (16 arrays x 10 elems; 2 arrays x 0 elems) | 700 |
| **total** | **861** |

**Big-endian throughout.** Booleans are the ASCII bytes `'T'`/`'F'`/`'?'`, not 0/1.
Variable-length array columns carry a 4-byte element count immediately before the payload.

**The null mask is real and informative.** BINARY2 writes every value regardless of nullity
(NaN for floats, zero-length arrays for absent array columns), so a reader can ignore the mask
and still work. Decoded anyway, there are exactly two distinct masks across the 1008 records:

- `00 00 48 00 00` (911 rows) -> NULL: `centroid_pos_ac`, `centroid_pos_error_ac`
- `01 70 48 00 00` (97 rows) -> additionally NULL: `obs_time_bary_corr`, `zeta`,
  `parallax_factor_al`, `parallax_factor_ac`

That second mask is the mechanism behind the NaN sentinels you will hit (1.6). Ignoring the
mask is safe *here* only because those 97 rows also happen to carry no usable abscissae — see
the trap in 5.1.

## 1.3 What one row physically is

**One row = one field-of-view transit of one source**, holding an array of **10 CCD
crossings**. Not one CCD observation, and not one 6-hour scan. Getting this wrong is the
fastest way to mis-weight the data by a factor of ~9.

Measured intra-transit timing over all 1008 rows (median of successive time differences):

```
slot  1 ->  2 :  7.261 s     <- gap between strips
slot  2 ->  3 :  4.856 s     |
slot  3 ->  4 :  4.860 s     |
   ...                       +- one CCD every 4.86 s
slot  9 -> 10 :  4.857 s     |
total slot 1 -> slot 10      : 46.14 s   (median transit duration; max 51.00 s)
```

Slot 1 is never used: its `gates` value is 12 in all 1008 rows and its `used_by_agis_al` is
**T in 0.0% of rows**. Slots 2-9 are used ~83.5% of the time, slot 10 71.7%. A transit
therefore contributes at most 9 usable along-scan abscissae:

```
n_used_AL :  0 -> 158 rows,  5 -> 1,  6 -> 4,  7 -> 10,  8 -> 147,  9 -> 688
```

> **[INFERENCE]** Reading the 10 slots as Sky Mapper + AF1...AF9 rests on three measured facts
> (slot 1 never used in AL, its gate constant at 12, and the 7.26 s slot-1->2 gap against
> 4.86 s thereafter). Neither the VOTable nor its metadata names the CCD strips. On a slide,
> say "the first slot is never used by AGIS" (fact), not "AF1 is excluded" (inference).

Within one transit the scan angle rotates by a median of **0.001584 deg (5.7 arcsec)** (max
0.002458 deg) and the parallax factor is a **single scalar for the whole transit**. The 10 CCD
abscissae are 10 quasi-simultaneous, nominally independent centroid measurements of the same
crossing; their internal scatter is **median 0.153 mas** over 850 transits, consistent with
the quoted per-CCD error.

**Note the asymmetry that trips people up: the scan angle is per-CCD, the parallax factor is
per-transit.** Broadcasting the wrong one is silent and wrong.

## 1.4 Every column

Ranges measured over the whole file. Scalars are per-transit; `[10]`/`[0]` marks the
variable-length per-CCD arrays.

### Scalars (19)

| # | name | dtype | unit | meaning | measured |
|---|---|---|---|---|---|
| 1 | `solution_id` | long | - | AGIS solution identifier | constant `2888461026632663040` |
| 2 | `source_id` | long | - | Gaia source id | 12 distinct |
| 3 | `transit_id` | long | - | transit identifier | 1008 rows, **1008 unique**; correlation with time = 1.0000000000; tick ~ **1.5625 ns/unit** |
| 4 | `ra0` | double | deg | RA of the reference point | 42.0867 ... 294.8278 |
| 5 | `dec0` | double | deg | Dec of the reference point | -16.7025 ... +61.4102 |
| 6 | `agis_source_excess_noise` | float | mas | unscaled AGIS excess noise | 0 for 8 sources; 0.1190 (Gaia-4), 0.6189, 1.2272 (HD 114762), **6.5829 (Gaia BH3)** |
| 7 | `obs_time_bary_corr` | float | ns | barycentric correction to `obs_time_tcb` | -3.5923e11 ... +3.6256e11 ns = **-359.2 ... +362.6 s**; NaN in 97 rows |
| 8 | `zeta` | float | deg | across-scan field angle of the reference point | -0.4035 ... +0.4057; NaN in the same 97 |
| 9 | `parallax_factor_al` | float | - | along-scan parallax factor | -0.7230 ... +0.7264 (median -0.0111); NaN in the same 97 |
| 10 | `parallax_factor_ac` | float | - | across-scan parallax factor | 0.6968 ... 0.7323 (median 0.7127) - pinned near 1/sqrt(2) |
| 11 | `nu_eff_used_in_astrometry` | float | um^-1 | effective wavenumber used in AGIS/IPD | 1.3848 ... 1.5798 |
| 12 | `nu_eff_error` | float | um^-1 | its standard error | 1.60e-4 ... 5.24e-2 (median 2.5e-4) |
| 13 | `transit_acq_flags` | short | - | on-board acquisition bitmask | 32 distinct (8240, 8241, ..., 8756) |
| 14 | `transit_proc_flags` | short | - | transit-level processing bitmask | 5 distinct: -32208, -24016, 560, 564, 8756 (signed 16-bit, bit 15 set) |
| 15 | `multipeak` | boolean | - | AstroElementary is multipeak | **F in all 1008** |
| 16 | `blended` | boolean | - | peak matched to another source | **F in all 1008** |
| 17 | `g_mag` | float | mag | G (CU5 or on-board) | 7.1519 ... 20.4561 |
| 18 | `g_class` | short | - | window class in SM/AF/XP | 0, 1, 2, 3 |
| 19 | `ac_rate` | float | pixel/s | across-scan drift rate at the transit | -1.0365 ... +0.9994 |

### Per-CCD arrays (18)

| # | name | dtype | unit | meaning | measured (flattened, 10080 slots) |
|---|---|---|---|---|---|
| 20 | `obs_time_tcb` | long[10] | ns | effective observing time, TCB, since the time origin | ~2.0e17 (1.6) |
| 21 | `scan_pos_angle` | double[10] | deg | scan position angle | -178.84 ... +179.46; 1139 NaN |
| 22 | `colour_factor_al` | float[10] | - | along-scan colour factor | -296.28 ... 176.64 (median 0.77); 0 NaN (filler 0.0) |
| 23 | `colour_factor_ac` | float[10] | - | across-scan colour factor | -30.32 ... 368.39; 5652 NaN |
| 24 | **`centroid_pos_al`** | double[10] | **mas** | **the measurement: along-scan coordinate of the image centroid** | -933.12 ... +46009.93 (the 46" outlier is an unused slot); 1139 NaN |
| 25 | `centroid_pos_ac` | double[**0**] | mas | measured across-scan coordinate | **empty in every row - NULL** |
| 26 | `calculated_pos_ac` | double[10] | mas | *predicted* across-scan coordinate | -6190 ... +387582; 1139 NaN |
| 27 | **`centroid_pos_error_al`** | float[10] | **mas** | 1-sigma on `centroid_pos_al` | 0.0399 ... 23.52 usable, plus 6 sentinels at **2.0628e8 mas**; 1139 NaN |
| 28 | `centroid_pos_error_ac` | float[**0**] | mas | 1-sigma on the across-scan centroid | **empty in every row - NULL** |
| 29 | **`used_by_agis_al`** | boolean[10] | - | CCD used along-scan by AGIS-DR4 | **T in 7467 / 10080 = 74.1%** |
| 30 | `used_by_agis_ac` | boolean[10] | - | CCD used across-scan by AGIS-DR4 | **T in 0 / 10080** |
| 31 | `ccd_proc_flags` | short[10] | - | CCD-level processing bitmask | 2833/10080 nonzero |
| 32 | `ipd_error_al` | float[10] | mas | 1-sigma of the IDU/IPD along-scan centroid | 0.0307 ... sentinel; 187 NaN |
| 33 | `ipd_error_ac` | float[10] | mas | 1-sigma of the IPD across-scan centroid (NaN for 1-D windows) | 0.0789 ... sentinel; 5704 NaN |
| 34 | `gates` | short[10] | - | gate per CCD | 0->6389, 4->824, 7->366, 8->436, 9->28, 10->334, 11->212, 12->1491; slot 1 always 12 |
| 35 | `source_dist_to_last_ci` | float[10] | pixel | distance to last charge injection | 0.44 ... 1999.4 (median 993.5); 1195 NaN |
| 36 | `sub_pixel_coord` | float[10] | pixel | along-scan sub-pixel phase | 0.0000 ... 0.9999 |
| 37 | `mu` | float[10] | pixel | across-scan coordinate of the window on the CCD | 13 ... 1991 (median 992.5) - the 1966-pixel extent |

Three format facts that matter more than the rest:

1. **This release is along-scan only.** `centroid_pos_ac` and `centroid_pos_error_ac` are
   zero-length in all 1008 rows and `used_by_agis_ac` is `F` in all 10080 slots. Only
   `calculated_pos_ac` (the *predicted* across-scan position) is present. **Every usable datum
   is a 1-D projection.** Any 1-D-only treatment you see is not a modelling simplification —
   it is what the product contains.
2. `gates` tracks brightness: gate 4 only ever appears at G = 7.15, gates 7/8 only at
   G = 8.97, gate 0 carries the bulk (median G = 14.26). **[INFERENCE]** which gate number
   means "ungated" is not established here, and gates 11/12 appearing at G ~ 19.8 breaks any
   simple monotonic reading. Show the cross-tab, not a semantics.
3. The huge error values (2.0628e8 mas ~ 57 deg) are "no measurement" sentinels. There are 6;
   **all 6 have `used_by_agis_al = F`**. Separately, one 46009 mas centroid (source
   2309425390592896, slot 1) has error 0.3024 — finite and positive — and the *only* thing
   excluding it is the AGIS flag. **A finite-and-positive test alone will keep a 46-arcsecond
   outlier.** Filter on `used_by_agis_al`.

> **[INFERENCE]** `parallax_factor_ac` pinned at ~0.71 is consistent with Gaia's fixed 45-degree
> solar aspect angle, but that explanation appears nowhere in the file.

## 1.5 The 12 sources

Per-source yield, brightness, and the 5-parameter weighted least-squares solution from the
parsed abscissae. `pre_rms` and `post_rms` are sqrt(mean((residual/sigma)^2)) before and after
the single-star fit — **dimensionless, not mas**.

| source_id | note | G | transits | N_abs | sigma_med | baseline | plx (mas) | pmra* (mas/yr) | pmdec (mas/yr) | pre_rms | post_rms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1457486023639239296 | Gaia-4 | 11.909 | 115 | 824 | 0.082 | 4.94 yr | 13.6215 | -75.5511 | 17.9404 | 1031.10 | **1.865** |
| 1663617687609809280 | | 14.255 | 95 | 781 | 0.157 | 5.21 yr | 10.1051 | -124.5727 | -135.8241 | 1118.40 | 1.037 |
| 2237987199365376 | QSO | 19.755 | 88 | 702 | 3.398 | 5.05 yr | -0.0894 | -0.0093 | 0.1053 | 1.04 | 1.035 |
| 435469040545191680 | | 13.973 | 81 | 649 | 0.140 | 4.76 yr | 0.9977 | 3.4218 | -5.8124 | 71.25 | 1.035 |
| 4181040337841125632 | | 8.975 | 97 | 637 | 0.105 | 4.75 yr | 1.0013 | 2.9041 | -0.2945 | 24.38 | 0.971 |
| 60730287810150016 | QSO | 20.456 | 73 | 608 | 4.473 | 5.02 yr | 0.0153 | 0.2211 | -0.0567 | 1.00 | 0.999 |
| 2309425390592896 | | 14.014 | 75 | 597 | 0.142 | 5.05 yr | 0.9269 | 7.1999 | -3.3227 | 59.95 | 0.993 |
| 20694084440761600 | | 14.046 | 74 | 576 | 0.145 | 5.05 yr | 4.9420 | 25.4424 | -16.5051 | 221.89 | 1.012 |
| 3937211745905473024 | HD 114762 | 7.152 | 90 | 558 | 0.125 | 4.92 yr | 25.6457 | -582.0555 | -0.7278 | 4193.08 | **10.265** |
| 4318465066420528000 | Gaia BH3 | 11.231 | 77 | 558 | 0.085 | 5.10 yr | 0.9833 | -30.9460 | -148.5497 | 2282.12 | **81.198** |
| 10973744521070720 | QSO | 19.885 | 62 | 515 | 3.537 | 5.04 yr | -0.0175 | -0.3405 | 0.0802 | 1.04 | 1.032 |
| 3926186255616949504 | | 18.967 | 81 | 462 | 1.914 | 5.13 yr | 0.9619 | -9.3398 | -0.5124 | 5.21 | 1.014 |

Read this table three ways:

- **`pre_rms` is discovery space** — the raw abscissa residual in sigma units with *no model at
  all*. The three faint sources sit at ~1.0 because they have neither parallax nor proper
  motion. Every star sits at 20-4000.
- **`post_rms` is the orbit column.** Nine sources land on a 0.97-1.04 floor. Exactly the three
  published orbital systems do not. **That excess is the orbit** — with no orbit model anywhere
  in the computation.
- The parser is right: max deviation from ESA's own reference solution over (plx, pmra*, pmdec)
  across all 12 sources is **5.9e-04 mas** (4.4).

> **[INFERENCE]** The three |plx| < 0.02 mas sources at G ~ 19.8-20.5 are extragalactic QSOs
> by their measured parallaxes; no catalogue cross-match was run. Six of the twelve sources
> are unidentified here; several sit at plx ~ 1.0 mas, which looks deliberate but has no
> stated basis.

## 1.6 The time system — and the TIMESYS mislabel

```xml
<TIMESYS ID="time_frame" refposition="BARYCENTER" timeorigin="2455197.5" timescale="TCB"/>
```

- **timescale = TCB.** Correct.
- **timeorigin = JD 2455197.5 = MJD 55197.0 = 2010-01-01T00:00:00 TCB.** Correct.
- **`refposition="BARYCENTER"` is wrong for `obs_time_tcb` as stored.** The field is described
  as "effective observing time as TCB", and there is a *separate* column,
  `obs_time_bary_corr`, "barycentric correction to obsTimeTcb". If the times were already at
  the barycentre that column would be meaningless. They are at the **spacecraft** (Gaia at L2);
  the correction moves them to the barycentre.

This is established empirically, not read from documentation. Fit `obs_time_bary_corr` for
HD 114762 with a 365.25 d sinusoid:

```
offset    = -2.03 s
amplitude = 461.52 s          rms residual 3.375 s   (n = 89 transits)
geometric prediction  1.01 AU * cos(beta) / c = 463.25 s   [ecliptic beta = 23.196 deg]
```

A clean annual light-time term of exactly the amplitude the source's ecliptic latitude
demands, with a few-second residual attributable to Gaia's Lissajous excursion about L2.
Across all 12 sources the observed half-range tracks cos(beta): e.g. beta = 60.99 deg ->
predicted 244.4 s, observed 244.8 s.

> **[INFERENCE]** If ESA's intent is that `refposition` describes the column *after* the
> correction is applied, the label is defensible. As written it describes `obs_time_tcb`, and
> `obs_time_tcb` is at the spacecraft. Either way: **add the correction.**

**What a reader must do:**

```
t_ns   = obs_time_tcb[j] + obs_time_bary_corr        # per-CCD time + per-transit scalar
t_MJD  = 55197.0 + t_ns * 1e-9/86400                 # barycentric MJD (TCB)
dt_yr  = 2010.0  + t_ns * 1e-9/(365.25*86400) - 2017.5   # Julian yr from the J2017.5 ref epoch
```

Cross-check: J2017.5 = MJD 57936.375, and `dt_yr` computed the second way agrees with
`(t_MJD - 57936.375)/365.25` to 1.2e-13 yr over HD 114762's 558 abscissae.

Two subtleties:

- `obs_time_bary_corr` is **one 32-bit float scalar per transit**, applied identically to all
  10 CCD times. Over a 46 s transit its own variation is far below the ~10 microsecond float32
  quantum at 3.6e11 ns, so this is exact for practical purposes.
- **97 rows carry NaN `obs_time_bary_corr`** (the NULL-mask rows of 1.2, which also lack
  `zeta` and both parallax factors). In RC3 all 97 also have **zero** usable CCDs, so a
  `used_by_agis_al` filter removes them before the NaN can propagate. **That is a coincidence
  of this file, not a guarantee** — see the trap in 5.1.

**Coverage.** Raw rows span J2014.5763 - J2020.0372 (MJD 56868.48 - 58863.10). Retained
abscissae span MJD 56941.67 - 58863.10 = 2014-10-11 -> 2020-01-15 (5.26 yr file-wide), with
`dt_yr` in [-2.723, +2.537]. Per source **4.75-5.21 yr**, ~1.5x the DR3 baseline.
J2017.5 = MJD 57936.375 = 2017-07-02 sits near the middle.

## 1.7 Repeated transits

**RC3 contains no duplicate `transit_id`: 1008 rows, 1008 unique ids.** If you have heard a
claim that the pre-release ships duplicates, it is wrong — do not repeat it.

**It does contain a different repetition.** 24 pairs of rows are separated by < 60 s within
the same source — 19 for HD 114762, 3 for Gaia BH3, 2 for 4181040337841125632 — with time
differences of 0.0000-0.0002 s. These are re-detections of the *same* physical crossing, and
their `transit_id`s **differ** (by ~2e6 ticks, about 3 ms), so an id-based deduplication keeps
both.

What saves you is that AGIS already rejected one member of every such pair. The census of
(usable CCDs in A, usable CCDs in B) over all 24 pairs is `(0,0)->5, (0,9)->11, (9,0)->8` —
**no pair has usable data on both sides.** For HD 114762 this collapses 90 rows to **71
distinct physical transits**, 63 of which contribute abscissae.

**The anti-double-count protection that actually fires in DR4 is `used_by_agis_al`.** An
id-based guard matters only if you concatenate epoch files or splice sources by hand.

> **[INFERENCE]** That these near-simultaneous pairs are bright-star window duplication is a
> reading. They could equally be a segmented window split or a duplicate AstroElementary.
> Likewise the 1.5625 ns/tick (= 1/0.64 GHz) `transit_id` encoding: monotonicity with time is
> measured at r = 1.0000000000 and the tick looks exact, but `transit_id` is clearly a
> compound key (two rows at the same instant differ by ~2e6) and its bit layout was not
> decoded.

## 1.8 What a reader must produce

Whatever your language, the astrometric model needs exactly **six vectors, one element per
surviving CCD crossing** — not per transit:

| quantity | unit | from |
|---|---|---|
| epoch `t` | MJD (TCB, barycentric) | `55197.0 + (obs_time_tcb + obs_time_bary_corr) * ns->day` |
| abscissa `w` | mas | `centroid_pos_al` |
| error `sigma` | mas | `centroid_pos_error_al` |
| scan angle `psi` | rad | `deg2rad(scan_pos_angle)` — **per CCD** |
| parallax factor `p` | - | `parallax_factor_al` — **per transit, broadcast to its CCDs** |
| PM time baseline `dt` | Julian yr from J2017.5 | `2010.0 + (obs_time_tcb + bary_corr) * ns->yr - 2017.5` |

with rows kept only where `used_by_agis_al` is true and every one of the six is finite
(`sigma > 0`).

Retention on RC3: 10,080 CCD slots -> 7,467 with `used_by_agis_al == T`. Per source, 462-824
abscissae.

Two behaviours to expect:

- **The epochs are not sorted.** Rows come in VOTable order — for HD 114762 the first row is
  MJD 57529.29 while the minimum is 57042.39. Sort if your code assumes monotonicity.
- **`centroid_pos_al` is an absolute position, not a residual** (contrast Hipparcos, 2.1).
  HD 114762's abscissae have RMS 534.9 mas; Gaia-4's 81.3 mas.

Global ranges across all 12 sources: abscissa -933.12 ... +1162.42 mas, error 0.0399 ... 23.52
mas (median 0.147), psi -3.121 ... +3.132 rad, parallax factor -0.723 ... +0.726, dt -2.723 ...
+2.537 yr.

---

# Part 2 — Relation to Hipparcos IAD

The headline: **DR4 epoch astrometry is Hipparcos Intermediate Astrometric Data with better
numbers.** Same measurement geometry, same design matrix, same likelihood. If you have code
that fits Hipparcos IAD, it fits DR4 epoch data after a format adapter and nothing else.

## 2.1 What Hipparcos IAD is

The van Leeuwen (2007) "new reduction" distributes, per HIP source, a plain 7-column ASCII
file (from the 348 MB ESA bundle `ResRec_JavaTool_2014.zip`):

```
# IORB   EPOCH    PARF    CPSI    SPSI     RES   SRES
   158 -1.2141  0.5681 -0.9121  0.4099   -0.57   0.74
   158 -1.2141  0.5685 -0.9123  0.4095    0.90   0.70
```

- `EPOCH` — time relative to **1991.25** in Julian years; this *is* the proper-motion factor.
- `PARF` — the along-scan parallax factor.
- `CPSI`, `SPSI` — the two scan-direction coefficients.
- `RES` — the along-scan abscissa **residual**, in mas, *relative to the published catalogue
  5-parameter solution*.
- `SRES` — 1-sigma on that residual. **`SRES < 0` flags a scan rejected during the catalogue
  reduction** — the direct analogue of `used_by_agis_al == false`.

One row = one field-of-view transit. The measurement is 1-D: the projection of the source onto
the great-circle scan direction. A single transit cannot fix a 2-D position; the ensemble of
scan angles over the mission does. **The per-epoch scan angle is not metadata — it is half the
data.**

**The convention trap.** The scan angle must be built as `atan2(CPSI, SPSI)`, *not*
`atan2(SPSI, CPSI)`. The column names lie relative to the North-through-East convention used
in 3.1: the file's `CPSI` plays the role of sin(psi) and `SPSI` of cos(psi). Getting this
backwards gives `pi/2 - psi`, which silently swaps RA and Dec in the reflex projection. 3.1
shows why that is invisible to a 5-parameter fit and lethal to an orbit.

Measured, HIP 64426 = HD 114762 and HIP 16537 = eps Eri:

| | HIP 64426 | HIP 16537 |
|---|---|---|
| retained transits (SRES > 0) | 53 | 78 |
| median sigma | 2.290 mas | 0.805 mas |
| baseline | 3.071 yr | 2.53 yr |
| psi coverage | -155.8 ... +72.4 deg | - |
| `PARF` | -0.669 ... +0.685 | \|PARF\| <= 0.690 |
| pm baseline | -1.228 ... +1.844 yr | -1.214 ... +1.313 yr |
| raw abscissa RMS | 2.296 mas | 0.700 mas |
| 5-param LSQ result | dRA*=-0.0002, dDec=-0.0018, **dplx=+0.0023**, dpmra*=-0.0001, dpmdec=+0.0018 | dDec=-0.002, dplx=+0.001, dpm ~ +0.002 |
| residual RMS in sigma units | 0.860 | 0.796 |

That second-to-last row is the point: fitting the 5-parameter model to Hipparcos abscissae
returns **corrections of a few microarcseconds**, because the catalogue solution has already
been subtracted. The sub-unity scatter says van Leeuwen's errors are mildly conservative, as
is well known.

## 2.2 The exact correspondence

| role in the abscissa model | Hipparcos IAD | Gaia DR4 epoch |
|---|---|---|
| epoch | `EPOCH` (yr from 1991.25) -> MJD | `obs_time_tcb` (ns since 2010.0 TCB, **per CCD**) + `obs_time_bary_corr` (ns, per transit) -> MJD |
| scan direction | `atan2(CPSI, SPSI)` | `deg2rad(scan_pos_angle)` (**per CCD**) |
| parallax factor | `PARF` | `parallax_factor_al` (**per transit**, broadcast) |
| PM time baseline | `EPOCH` itself (yr from 1991.25) | `t_jyear - 2017.5` (yr from J2017.5) |
| the measurement | `RES` — abscissa **residual** vs catalogue, mas | `centroid_pos_al` — **absolute** position vs `(ra0, dec0)`, mas |
| its 1-sigma | `SRES` (mas; < 0 = rejected scan) | `centroid_pos_error_al` (mas) |
| rejection flag | `SRES < 0` | `used_by_agis_al == 'F'` |

## 2.3 Where they genuinely differ

1. **Per-CCD vs per-transit.** Hipparcos gives one abscissa per transit; DR4 gives up to nine.
   HD 114762: 53 Hipparcos points vs 558 DR4 points from 90 transits.
2. **Precision.** 2.290 mas (Hipparcos, V ~ 7) vs 0.125 mas (DR4, G = 7.15) — a factor **18**
   per point, and sqrt(N) on top. Against eps Eri the factor is 10 (0.805 vs 0.082).
3. **Baseline and epoch.** 3.07 yr around J1991.25 vs 4.92 yr around J2017.5. Two epochs 26 yr
   apart — which is what makes the Hipparcos-Gaia long-baseline acceleration channel exist.
4. **Residual vs absolute** — the largest structural difference, and the one that turns out
   not to matter (2.4).
5. **Scan-angle convention.** Hipparcos needs `atan2(CPSI, SPSI)`; DR4's `scan_pos_angle` goes
   in with only deg->rad. Both land in the same basis.
6. **Parallax-factor granularity.** DR4's is per transit while scan angle and time are per
   CCD. Over 46 s the broadcast is exact to well below any relevant precision.
7. **Time system.** DR4 epochs are TCB nanoseconds since 2010-01-01.0 with a per-transit
   barycentric correction of +/-360 s that must be added; Hipparcos epochs come pre-barycentred
   as Julian years from 1991.25.
8. **Rejection semantics.** Negative error vs explicit boolean.
9. **Independence.** Per-CCD abscissae within a transit are normally treated as independent
   Gaussians. Tested on four quiet sources, the variance of the transit-mean whitened residual
   against the 1/n_ccd expectation gives ratios **0.85, 1.27, 0.83, 1.21** (sampling error
   ~0.17 at 70-90 transits). Consistent with independence; **a ~20% correlated per-transit
   component is not excluded.**
10. **Sky sampling.** DR4 is 10x more numerous per source (824 vs 78), but 824 CCDs are only
    115 independent pointings against Hipparcos's 78 genuinely independent transits. **The
    gain in sky sampling is a factor 1.5, not 10.** The gain in precision is real; the gain in
    geometry is modest.

## 2.4 Why one likelihood serves both: linear invariance

The 5-vector q = (dRA0, dDec0, dplx, dpmra*, dpmdec) is marginalised under a **flat improper
prior** (3.7). Therefore adding any exact 5-parameter linear signal to the abscissae cannot
change the likelihood.

Verified directly: adding q = (123.4, -56.7, 8.9, -1.23, 4.56) through the design matrix to
Gaia-4's abscissae changed the log-likelihood by **3.6e-7 nats out of 574.95** — float noise.

So whether the abscissa is a raw absolute position (Gaia: the fit returns plx = 25.65 mas) or
a pre-subtracted residual (Hipparcos: the fit returns dplx = 0.0023 mas) is **invisible to the
likelihood**. The design matrix, the marginalisation and the returned log-likelihood are
identical; only the interpretation of q changes.

That is the deep reason one function serves both, and it is worth stating rather than leaving
as "the model has the same structure".

**Your intuition is not invariant, though**: "a 1 mas residual" means something completely
different in the two files.

## 2.5 HD 114762 in both datasets

The same-star comparison, because it removes every confound:

| | Hipparcos IAD | Gaia DR4 epoch |
|---|---|---|
| N | 53 | 558 |
| sigma_med | 2.290 mas | 0.1254 mas |
| baseline | 3.071 yr | 4.922 yr |
| raw abscissa RMS | 2.296 mas | 534.942 mas |
| post-5p RMS | 2.296 mas | 1.263 mas |
| residual RMS in sigma units after 5p | 0.860 | **10.265** |
| 5-param result | corrections <= 0.0023 mas | plx = 25.6457, pmra* = -582.0555, pmdec = -0.7278 |

Projecting the companion's along-scan reflex onto each dataset's own scan angles, at
M_sec = 0.139 Msun, i = 4.84 deg, P = 83.92 d, plx = 25.36 mas (-> a_star = 1.350 mas):

| | RMS of the projected reflex | sigma_med | S/N per point | sqrt(N) x S/N |
|---|---|---|---|---|
| Hipparcos | 1.259 mas | 2.290 | 0.55 | **4.0** |
| Gaia DR4 | 1.166 mas | 0.125 | 9.30 | **220** |

Same star, same orbit, same likelihood — a factor **55** in cumulative signal-to-noise. That
is the entire reason the sin(i) break is clean now and was marginal in 1997.

> Caveat: the elements used here (P = 83.9151 d, e = 0.335, omega = 201 deg) are literature
> values, not a fit. The contrast is dominated by the sigma ratio and sqrt(N) and is robust to
> modest changes in e/omega.

---

# Part 3 — The models

## 3.1 The one equation, and the sign convention

Every absolute-astrometry model here is a **1-D along-scan abscissa** model. The projection of
a 2-D sky displacement onto the scan direction is

```
d_eta(t_i) = d_alpha*(t_i) * sin(psi_i) + d_delta(t_i) * cos(psi_i)
```

**The convention is pinned:** psi = 0 -> +Dec, psi = pi/2 -> +RA, so the along-scan unit vector
in (alpha*, delta) is (sin psi, cos psi) — psi measured from North through East.

The **full single-star model** is:

```
w_i = dalpha0 * sin(psi_i) + ddelta0 * cos(psi_i) + plx * p_i
      + pmra* * sin(psi_i) * dt_i + pmdec * cos(psi_i) * dt_i     [+ photocentre orbit]
```

with `dt_i` in Julian years relative to J2017.5 (DR4) or J1991.25 (Hipparcos). Equivalently
w_i = (dalpha0 + pmra* dt_i) sin psi_i + (ddelta0 + pmdec dt_i) cos psi_i + plx p_i.

**Empirical verification of the convention** — this is the part worth showing, because
self-consistency cannot catch a swap:

Replacing psi -> pi/2 - psi **exactly swaps the fitted pmra and pmdec and leaves the fit
quality unchanged**. On source 1663617687609809280: as-coded (plx, pmra*, pmdec) =
(10.1051, -124.5727, -135.8241); swapped = (10.1051, -135.8241, -124.5727); residual scatter
1.0368 in both cases. Published DR3 for that source is (10.1265, -124.5857, -135.8230). Same
result on two other sources.

**The fit quality is blind to the swap; only the catalogue identification distinguishes them.**

Now the orbit side, on Gaia-4 (824 abscissae):

- no reflex: log L = **-163.982** (correct psi) vs **-163.980** (swapped) — identical to 2e-3,
  because both bases span the same 2-D plane.
- best over a 181-point mean-anomaly grid at the published elements: log L = **+574.951**
  (correct) vs **-458.009** (swapped) — **1033 nats lost.**

That is the whole argument for why this convention matters, in two numbers: **a 5-parameter
fit cannot detect the error, and an orbit fit loses a thousand nats to it.**

> **[INFERENCE]** That DR4's `scan_pos_angle` is *documented* as North-through-East was not
> checked against ESA material. The convention was established empirically here, which is a
> stronger claim for a talk anyway.

## 3.2 The 5-parameter single-star solution

The whole thing is four lines of weighted least squares:

```
s = sin(psi);  c = cos(psi)
X = [ s,  c,  p,  s*dt,  c*dt ]           # n x 5 design matrix
W = 1/sigma
q = (X .* W) \ (w .* W)                   # weighted LSQ
```

This reproduces ESA's own reference solution to **5.9e-04 mas** across all 12 sources.

Because the model is linear-Gaussian in all five parameters, the posterior is exact from one
Cholesky factorisation — mean = MAP = global maximum, full covariance, no optimiser and no
local-minimum question:

```
Gaia-4      dra_cosdec  =    0.00031 +/- 0.00365 mas
            ddec        =    0.00082 +/- 0.00469 mas
            parallax    =   13.62145 +/- 0.00621 mas
            pmra_cosdec =  -75.55112 +/- 0.00265 mas/yr
            pmdec       =   17.94037 +/- 0.00300 mas/yr     chi2/dof = 3.50

HD 114762   parallax    =   25.64553 +/- 0.01074            chi2/dof = 106.31
Gaia BH3    parallax    =    0.98330 +/- 0.00623            chi2/dof = 6652.72
QSO 6073... parallax    =    0.01527 +/- 0.47861            chi2/dof = 1.01
```

Two points worth a slide each.

**First, the position offsets come out at ~1e-4 mas.** `(ra0, dec0)` *is* the DR4 position at
J2017.5 to within tens of microarcseconds. The file hands you a reference point already at the
solution, so the abscissae are already quasi-residuals — which is why a plain LSQ converges
instantly and why the Hipparcos/DR4 structural difference of 2.3 is so easy to miss.

**Second, chi2/dof is the orbit detector**: 1.01 for a QSO, 3.5 / 106 / 6653 for the three
known orbital systems. No orbit model required.

**Caveat to state honestly:** those uncertainties are formal, from `centroid_pos_error_al`
alone. If you omit the chromatic term (1.4, column 22) and do not fold in
`agis_source_excess_noise`, they are optimistic relative to published Gaia uncertainties, and
Gaia-4's chi2/dof = 3.50 is partly orbit and partly un-modelled systematics. That split was not
quantified here.

## 3.3 The solution ladder: 5p / 7p / 9p

The ladder is **one** abscissa model with a growing number of free linear terms:

```
w_i = dra* sin(psi) + ddec cos(psi) + plx p_i                     (5p, with proper motion)
      + pmra* sin(psi) dt + pmdec cos(psi) dt
      + acc_ra* sin(psi) dt^2/2 + acc_dec cos(psi) dt^2/2         (7p)
      + jrk_ra* sin(psi) dt^3/6 + jrk_dec cos(psi) dt^3/6         (9p)
```

The rungs are strictly **nested prefixes** — the first five columns of the 9p design matrix are
the 5p design matrix.

**Where the 1/2 and 1/6 come from.** They are the Taylor coefficients of the reflex track, not
fudge factors. Expanding a bounded reflex displacement about the reference epoch,

```
x(dt) = x0 + xdot dt + xddot dt^2/2 + xdddot dt^3/6 + ...
```

Putting 1/2! and 1/3! *into the basis function* means the fitted coefficient **is** the
derivative: `acc_ra` is d2(alpha*)/dt2 in mas/yr^2, `jrk_ra` is d3(alpha*)/dt3 in mas/yr^3.
Without it the coefficients are in arbitrary units and no prior on them means anything.

**Conditioning is a side benefit, not the reason.** On real DR4 data (source
4181040337841125632, N=637, T=4.75 yr): cond(X'WX) at order 9 is 2.5e2 normalised versus 8.8e2
with raw dt^k — a factor 3.5, and every column RMS within a factor ~3 of unity. Real, modest,
not the point.

### The closed form, and why the priors must be proper

Every rung is linear-Gaussian in **all** its parameters, so its marginal likelihood is exact —
no sampler, no estimator error bar. For `w = Xq + eps`, `eps ~ N(0, W^-1)`, `W = diag(1/sigma^2)`,
prior `q ~ N(0, S)`:

```
log Z = -n/2 log(2 pi) + 1/2 log|W|
        - 1/2 (w'Ww - v'M^-1 v)
        - 1/2 log|S| - 1/2 log|M|         with  M = X'WX + S^-1,  v = X'Ww
```

`-1/2 log|S| - 1/2 log|M|` is the Occam factor. Cost: **22.5 microseconds per rung.** A
per-source posterior over the ladder is three matrix factorisations, which is a very different
proposition at 1.8 billion sources than an MCMC each.

**The prior must be proper or the Bayes factors are undefined.** The orbit-fitting likelihood
(3.7) marginalises the catalogue correction under a *flat improper* prior. That is fine for
orbit fitting, where the resulting constant cancels between orbit samples. It is **fatal for
model comparison**, because the constant does not cancel between models with different numbers
of columns — Lindley's paradox, arriving on schedule. Model comparison needs a normalised
prior.

**A prior that means something.** State the prior in **mas of sky displacement** and convert:
demand that a column's contribution equal `a0_max` at `dt = accel_yr`, and the 2 and 6 fall out
as the inverses of the 1/2 and 1/6 in the basis:

```
sigma_acc * accel_yr^2/2 = a0_max   =>   sigma_acc = 2 a0_max / accel_yr^2
sigma_jrk * accel_yr^3/6 = a0_max   =>   sigma_jrk = 6 a0_max / accel_yr^3
```

With a0_max = 5 mas over accel_yr = 10 yr: sigma_acc = **0.1 mas/yr^2**, sigma_jrk =
**0.03 mas/yr^3**. The framing is "how curved could a real long-period companion make this",
which is the hypothesis being tested.

**This is the single most consequential choice in the comparison.** Widening the acceleration
prior by a decade costs ~log(10) per added parameter, **~4.6 nats for the two acceleration
terms — a factor of 100 in odds per decade of prior width.** Measured across 5 decades of
prior width on the 12 RC3 sources, 1 of 12 changes its preferred rung.

It does **not** bias the 5-parameter block: default vs an effectively-flat 1e6 mas prior shifts
parallax by at most 1e-3 mas and pmra by 3e-3 mas/yr. The prior matters only through the Occam
factor on the acceleration and jerk columns.

### Measured over all 12 RC3 sources

```
source_id                N dt_span    logZ5      logZ7      logZ9    P5   P7   P9   c2_5   c2_7   c2_9 best
2237987199365376       702    5.05   -1920.1    -1920.1    -1920.1  .34  .33  .33   1.08   1.08   1.08 5
2309425390592896       597    5.05     269.1      264.4      263.5  .99  .01  .00   0.99   1.00   1.00 5
10973744521070720      515    5.04   -1444.0    -1444.1    -1444.1  .37  .32  .32   1.08   1.08   1.08 5
20694084440761600      576    5.04     214.2      209.8      210.1  .97  .01  .02   1.03   1.04   1.03 5
60730287810150016      608    5.02   -1813.5    -1813.4    -1813.4  .31  .34  .34   1.01   1.01   1.01 9  (!)
435469040545191680     649    4.76     279.8      276.3      275.4  .96  .03  .01   1.08   1.08   1.08 5
1457486023639239296    824    4.94    -300.9     -259.9     -255.6  .00  .01  .99   3.50   3.39   3.38 9  INADEQUATE
1663617687609809280    781    5.21    -423.1     -427.2     -428.1  .98  .02  .01   1.08   1.08   1.08 5
3926186255616949504    462    5.13    -992.0     -992.3     -992.3  .39  .30  .30   1.04   1.04   1.05 5
3937211745905473024    558    4.92  -35577.9   -33891.4   -32622.1  .00  .00 1.00 106.31 100.50  94.62 9  INADEQUATE
4181040337841125632    637    4.75     502.7      500.4      499.0  .89  .08  .02   0.95   0.94   0.95 5
4318465066420528000    558    5.10 -1839105.7  -495854.5  -334302.6 .00  .00 1.00 6652.7 1788.3 1140.1 9  INADEQUATE
```

Three teaching points, all visible in that table.

- **All three known orbital systems land in the INADEQUATE column.** A model-probability
  computation normalises over the rungs you gave it and **cannot tell you whether any of them
  fits.** Gaia BH3 gets P(9p) = 1.000 with a **1.5-million-nat** evidence improvement and
  chi2/dof = **1140**: the ladder faithfully picks the least hopeless of three hopeless models,
  on precisely the source that matters most. **Always read a goodness-of-fit alongside the
  probability.**
- **Source 60730287810150016 reports best = 9p with P = [0.31, 0.34, 0.34].** Three
  near-identical evidences; the argmax is noise. **Never read `best` without `prob`.**
- **Source 4181040337841125632 has acc_ra = -0.0211 +/- 0.0086 — a 2.1-sigma per-component
  "detection" — and the ladder still says 5p at P = 0.89.** The Occam factor doing its job.
  Contrast with a hard frequentist significance threshold, which would have called it.

Per-source acceleration precision, from real data — **the number to quote for "what DR4 will
actually deliver"**: sigma_acc ~ **0.005-0.01 mas/yr^2** for the well-observed sources.

```
2237987199365376      -0.0320 +/- 0.0941    0.0352 +/- 0.0919   s7=0.4
2309425390592896      -0.0021 +/- 0.0114   -0.0028 +/- 0.0083   s7=0.2
10973744521070720      0.0102 +/- 0.0946   -0.0112 +/- 0.0896   s7=0.1
20694084440761600      0.0006 +/- 0.0119    0.0035 +/- 0.0100   s7=0.2
60730287810150016     -0.0458 +/- 0.0968    0.0414 +/- 0.0914   s7=0.5
435469040545191680     0.0073 +/- 0.0110    0.0143 +/- 0.0100   s7=1.1
1457486023639239296   -0.0191 +/- 0.0043   -0.0415 +/- 0.0048   s7=7.0
1663617687609809280    0.0075 +/- 0.0083   -0.0066 +/- 0.0081   s7=0.9
3926186255616949504    0.0223 +/- 0.0816   -0.0525 +/- 0.0697   s7=0.5
3937211745905473024   -0.2758 +/- 0.0119   -0.4557 +/- 0.0085   s7=36.4
4181040337841125632   -0.0211 +/- 0.0086   -0.0010 +/- 0.0054   s7=2.1
4318465066420528000   -7.5114 +/- 0.0053    3.7915 +/- 0.0048   s7=1168.8
```

> **Convention trap when comparing against published columns.** Folding the 1/6 into the basis
> makes the jerk coefficient the literal third derivative. If Gaia's published `deriv_accel_*`
> columns use a different normalisation, a naive comparison is **off by a factor of 6**. Check
> the NSS data model before quoting published jerk values against your own.

## 3.4 When you get 7p or 9p instead of 5p or an orbit

**Gaia's own rule is a cascade of frequentist cuts** (as described for DR3: trigger on
RUWE > 1.4, try 9p first, accept on s9 > 12 and F2 < 25, else fall back to 7p, else orbital),
and the winner is published as fact. Three consequences: boundary sources are reported as one
type with no uncertainty propagated; the thresholds control catalogue-level false positives,
not per-source probability; and **"the 7p sample" is a population defined by a cut**, so any
inference on it inherits the cut.

**The physical regime, measured.** A circular photocentre reflex of a0 = 0.5 mas injected into
the real DR4 scan pattern of source 4181040337841125632 (N = 637, baseline **T = 4.75 yr**,
median sigma = 0.105 mas):

```
  P [yr]     P/T     P5     P7     P9  best      c2_5      c2_7      c2_9  verdict
     1.0    0.21   0.00   0.00   1.00     9      9.35      9.20      7.97  NO RUNG FITS -> orbital
     3.0    0.63   0.00   0.00   1.00     9      8.20      7.65      6.45  NO RUNG FITS -> orbital
     5.0    1.05   0.00   0.00   1.00     9      8.13      3.10      1.50  9p
    10.0    2.11   0.00   0.00   1.00     9      1.97      1.11      1.05  9p
    15.0    3.16   0.00   0.57   0.43     7      1.23      1.05      1.05  7p
    25.0    5.27   0.44   0.44   0.11     5      1.06      1.05      1.05  5p (reflex absorbed)
   100.0   21.07   0.99   0.01   0.00     5      1.04      1.05      1.05  5p
```

Scanning amplitude as well (entry = best rung; `x` = no rung adequate, chi2/dof > 2):

```
a0 [mas]     P=3   P=5  P=10  P=15  P=25  P=50 P=100 P=300 P=1000
0.2           9p    9p    9p    7p    5p    5p    5p    5p     5p
0.5         x(9)    9p    9p    7p    5p    5p    5p    5p     5p
2.0         x(9)  x(9)    9p    9p    7p    7p    5p    5p     5p
5.0         x(9)  x(9)    9p    9p    9p    7p    5p    5p     5p
20.0        x(9)  x(9)  x(9)    9p    9p    9p    7p    5p     5p
```

**Rule of thumb**, with T the mission baseline (RC3 spans 4.75-5.21 yr per source; nominal DR4
is ~5.5 yr):

- **P <~ T** — the reflex is not a polynomial over the window. Every rung has chi2/dof >> 1 ->
  an **orbital** solution is required. Note the ladder still reports best = 9p at P ~ 1.0; that
  verdict is worthless without the goodness of fit.
- **T <~ P <~ 3T** — cubic term detectable -> **9p**.
- **P ~ 3T** — quadratic only -> **7p**.
- **P >~ 5T** — curvature drops below the noise, reflex absorbed into position + proper motion
  -> **5p**, and the companion is invisible in the astrometry alone.

The 7p/9p boundary follows from the basis functions: at |dt| = T/2 the ratio of the jerk term
to the acceleration term is (omega T/2)/3 = **pi T / (3P)**. Jerk becomes comparable to
acceleration when P ~ T — i.e. **9p is chosen precisely as you approach the orbital regime**,
which is why 9p sources are the ones most likely to deserve a Keplerian.

The upper boundary is **signal-to-noise dependent, not purely P/T**: a 20 mas reflex is still
7p at P = 100 yr (21 T). "7p/9p regime" means *curvature detectable but full orbit not*, and
both halves depend on amplitude and per-point error.

## 3.5 How to read a published 7p or 9p solution

### What an acceleration is

A 7p solution gives (acc_alpha*, acc_delta) in mas/yr^2 and nothing else about the companion.
The measured quantity is the projected **instantaneous gravitational acceleration** of the
photocentre:

```
|accel| [mas/yr^2] = 4 pi^2 * plx[mas] * M_c / (M_tot^(2/3) * P^(4/3))     (M in Msun, P in yr)
```

equivalently, in the Newtonian form that makes the degeneracy obvious,

```
|accel| = plx[mas] * (G M_c / r^2) [AU/yr^2],    G = 4 pi^2 AU^3 Msun^-1 yr^-2
```

Sanity check: Sun + Jupiter at 10 pc -> **0.139 mas/yr^2**, matching a0 * omega^2 = 0.496 mas x
(2 pi / 11.86)^2 exactly.

### The mass-period degeneracy

An acceleration measures **GM_c/r^2** — one number, two unknowns. The locus is

**M_c proportional to r^2 proportional to (M_tot P^2)^(2/3), i.e. M_c ~ P^(4/3)**

For a 3-sigma detection at sigma_acc = 0.01 mas/yr^2 (accel = 0.03), M_pri = 1 Msun,
plx = 50 mas (20 pc):

| P [yr] | a_rel [AU] | M_c [Msun] | M_c [M_Jup] |
|---|---|---|---|
| 10 | 4.64 | 0.0003 | 0.3 |
| 20 | 7.37 | 0.0008 | 0.9 |
| 50 | 13.57 | 0.0028 | 2.9 |
| 100 | 21.54 | 0.0071 | 7.4 |
| 300 | 44.81 | 0.0305 | 32.0 |
| 1000 | 100.00 | 0.1520 | 159.2 |

**The same measured acceleration is a Saturn at 10 yr and a 0.15 Msun M dwarf at 1000 yr.**
Nothing in the astrometry breaks it. A DR4 7p solution supports statements of the form "there
is a companion with M_c/r^2 = X", and **not** "there is a planet". Papers quoting a single mass
from an acceleration are importing a period from somewhere else — a direct-imaging
non-detection, an RV trend, or a population prior on separation. **Say which.**

What *does* break it:

- **RV** — an RV trend gives the line-of-sight acceleration; with the two astrometric
  components you get the full 3-D acceleration vector. Still M_c/r^2, but an RV *second*
  derivative adds a second equation.
- **A jerk (9p)** — this is why 9p matters scientifically. jerk/accel ~ omega = 2 pi/P gives an
  *independent* handle on P, hence on M_c. **A 9p solution is qualitatively stronger than a 7p,
  not just bigger.**
- **Imaging non-detection** — rules out the high-mass/large-separation end of the track.
- **A longer baseline** — Hipparcos-to-Gaia proper-motion differences encode the acceleration
  over 26 yr rather than within the mission.

**Operational rule: a published solution type is a statement about which model won a
comparison, not a statement that the model fits.** Gaia's cascade does apply goodness-of-fit
gates, but the published product carries no per-source P(model | data) and no residual for you
to check.

## 3.6 The companion: photocentre wobble and a0

The observable is the **photocentre** displacement. For a dark companion that is the star's
reflex about the barycentre:

```
a0 [mas] = a_rel [AU] * M_sec / (M_pri + M_sec) * plx [mas]
```

Verified numerically on the published Gaia-4 solution (P = 571.3 d, e = 0.338,
M_pri = 0.644 Msun, M_c = 11.8 M_J, plx = 13.628 mas): a_rel = 1.17037 AU from Kepler III, so
a0 = **0.27418 mas** against the published 0.279 mas — 2%. Max projected reflex over one period
came out 0.36685 mas vs a0(1+e) = 0.36680. The scaling is exact in plx and near-exact in M_sec
(the M_tot denominator gives the small departure).

Equivalently, eliminating a_rel through Kepler III:

```
a0 [mas] = plx * M_sec * (M_pri + M_sec)^(-2/3) * P_yr^(2/3)
```

**Campbell versus Thiele-Innes.** The Thiele-Innes constants (A, B, F, G) make the model linear
in four of its parameters, which is why catalogue pipelines use them. Their drawback is that
priors natural in (A, B, F, G) are unphysical, and the inversion back to (a, i, omega, Omega)
is exactly where sign ambiguities hide. The two are algebraically identical:

```
x = r (cos(nu+w) sin(Om) + sin(nu+w) cos(i) cos(Om))     [AU, +RA]
y = r (cos(nu+w) cos(Om) - sin(nu+w) cos(i) sin(Om))     [AU, +Dec]
```

substituting r cos(nu) = a(cos E - e) and r sin(nu) = a sqrt(1-e^2) sin E gives
x = a(BX + GY), y = a(AX + FY) with the textbook A, B, F, G. **Sample in Campbell elements,
evaluate through Thiele-Innes if you like — but do not sample in Thiele-Innes.**

## 3.7 The likelihood: marginalise the five, sample the orbit

The single most important structural fact about fitting epoch astrometry:

**The 5-parameter astrometric solution is re-fitted analytically at every likelihood call,
conditional on the current orbit. It is never sampled.**

This is the van Leeuwen (2007) section 17.3 / Brandt (2018) algorithm. In full:

```
# stage 1 - subtract the orbital reflex, per epoch, summed over companions
d_eta_orb,i = SUM_k [ d_alpha*_k(t_i) sin(psi_i) + d_delta_k(t_i) cos(psi_i) ]
r_i         = w_i - d_eta_orb,i

# stage 2 - analytically marginalise the 5 catalogue nuisances
x_i      = ( sin psi_i,  cos psi_i,  p_i,  sin psi_i dt_i,  cos psi_i dt_i )
W        = diag(1/sigma_i^2)
A        = X' W X            # 5x5
v        = X' W r
chi2_min = r'Wr - v' A^-1 v
log L    = -chi2_min/2 - SUM log sigma_i - log det(A)/2 - (n-5)/2 log(2 pi)
```

`log det A` is the marginalisation Jacobian; `(n-5)` counts the absorbed degrees of freedom.

**Why it must be done per call rather than once up front** — the dynamic range, on Gaia-4:

```
  contribution of each model term to w (peak-to-peak, mas):
    position  dalpha* sin(psi) + ddelta cos(psi) :   0.0017
    parallax  plx * p_AL                         :  19.6749
    prop.mot. pm * (sin,cos) * dt                : 325.3067
    data      w                                  : 325.5101
  orbit                                          :   0.5860
  per-point error                                :   0.0819
```

**You are fitting a 0.5 mas signal underneath a 325 mas one.** Fit the five once, subtract, and
the orbit partially absorbs into them — see 4.4 for the 8-sigma bias that produces.

Cost: ~**230 microseconds per call at N = 824** (~280 ns per abscissa), dominated by one Kepler
solve plus the 5x5 accumulation.

**Two traps in this algorithm.**

1. **If the design matrix is rank-deficient the Cholesky fails.** In practice that means
   somebody built the data container without the parallax factor or the PM baseline and got
   zeros. If your code falls back to an *unmarginalised* Gaussian instead of erroring, you get
   a finite number with no warning and a fit that will go anywhere: measured, **+622 becomes
   -437,831,126**. Make this loud.
2. **A, sum log sigma and log det A are orbit-independent**, so they are additive constants
   across orbit samples. Fine for orbit fitting and for evidence within a fixed dataset; **not**
   fine for comparing models with different design widths (3.3).

## 3.8 Degeneracies, measured on real likelihoods

**(Omega -> Omega+pi together with omega -> omega+pi) is an EXACT symmetry.**
delta log L = **0.000e+00**, bit-identical; max deviation over an epoch grid = 3.5e-16 mas.
Algebraically all four Thiele-Innes constants are invariant. This is the classic "you cannot
tell which node is ascending without radial velocities."

**Omega -> Omega+pi alone** is not a symmetry: delta log L = **-3421.6**.

**i -> 180-i is NOT degenerate in epoch astrometry.** At fixed (Omega, omega, M0):
delta log L = -582. Re-maximising over (Omega, omega, M0) on a 41^3 grid at each inclination:
log L = 689.75 at i = 116.9 deg vs 112.51 at i = 63.1 deg — a penalty of **hundreds of nats**.
Time-resolved astrometry sees the *sense* of the reflex motion on the sky, and DR4's scan-angle
coverage (-167 to +173 deg) resolves it.

> Quote this as "hundreds of nats", not a precise figure: the measurement held (a, M_sec, e) at
> published values and re-maximised only three angles on a grid. A full re-fit at the mirror
> inclination would reduce the penalty by an unmeasured amount.

**RV is exactly blind to both of those**, measured on HD 114762:

```
i -> pi - i                            dlogL_RV =    -0.000000   dlogL_AST =    -31218.6
Omega -> Omega + pi                    dlogL_RV =    +0.000000   dlogL_AST =    -52618.7
omega -> omega + pi                    dlogL_RV = +176515.11      dlogL_AST =    -52618.7
Omega+pi AND omega+pi                  dlogL_RV = +176515.11      dlogL_AST =        +0.0
```

RV sees only sin(i), and Omega never enters the RV model at all. **So the joint fit breaks the
node ambiguity that astrometry alone cannot touch** — because RV *is* sensitive to omega. That
is a second, rarely-mentioned, payoff of joint fitting.

### The parallax-mass degeneracy

The naive statement — a0 proportional to M_c * plx, so mass and distance trade off — is
correct. It is sharper than that.

**There are two different parallaxes in the problem.** The parallax that scales the reflex from
AU to mas, and the parallactic signal in the data (column 3 of the design matrix) whose
coefficient is one of the five nuisances marginalised away under a flat prior. **Nothing ties
them together.** With the orbit switched off, the likelihood is numerically independent of the
assumed parallax:

```
plx =   0.500 mas  ->  logL = -163.980656
plx =  13.628 mas  ->  logL = -163.980656
plx = 500.000 mas  ->  logL = -163.980660
```

A 1000x change moves the likelihood by 4e-6 nats. **On an epoch-astrometry-only fit, the
parallax posterior is the prior, full stop.**

The degeneracy itself, on the real abscissae, walking the ridge M_sec x plx = const with `a`
re-derived each step so the *period* stays at 571.3 d:

```
   f    plx[mas]   M_sec[Msun]     a[AU]     a0[mas]     logL
0.25      3.4070    0.045057     1.19015    0.265141   572.85
0.50      6.8140    0.022528     1.17703    0.271082   568.34
1.00     13.6280    0.011264     1.17037    0.274180   565.62
2.00     27.2560    0.005632     1.16700    0.275763   564.13
4.00     54.5120    0.002816     1.16531    0.276562   563.36
```

A factor of **16 in companion mass costs 9.5 nats.** Essentially flat.

Control — move plx alone, so a0 is not compensated:

```
plx =  6.814 mas  a0 = 0.1371 mas   logL =   446.13
plx = 13.628 mas  a0 = 0.2742 mas   logL =   565.62
plx = 27.256 mas  a0 = 0.5484 mas   logL =  -667.23
```

A factor of 2 in plx costs **1233 nats**. **The abscissae pin a0 ferociously and are blind to
its factorisation into M_c and plx.** The astrometry measures an angle; only the parallax prior
turns it into a mass.

Worth having in your pocket: for Gaia-4 the adopted literature parallax is 13.628 +/- 0.021,
the DR4-epoch 5-param solution gives 13.6215, and the DR3 catalogue gives 13.5448. Three
parallaxes within 0.08 mas of each other, and the mass moves with whichever you pick.

## 3.9 What RV adds: the sin(i) break

RV constrains `M_sec sin(i) (M_pri+M_sec)^(-2/3) P^(-1/3)`; astrometry constrains
`a0 = plx M_sec (M_pri+M_sec)^(-2/3) P^(2/3)`. **The shared mass combination cancels in the
ratio:**

```
a0 / K = plx * P_yr * sqrt(1-e^2) * (365.25 F)^(1/3) / sin(i)

=>  sin(i) = 3.357076e-5 * plx[mas] * P[yr] * sqrt(1-e^2) * K[m/s] / a0[mas]
```

One equation, no sampler needed to see why it works. Check with HD 114762 (plx = 25.36,
P = 83.9151 d, e = 0.335, K = 615 m/s, a0 = 1.3433 mas): **sin i = 0.084375 -> i = 4.840 deg**,
which is the joint-fit answer.

**Why RV alone cannot do it**, holding M sin(i) = 11 M_J fixed and scanning i:

```
   i[deg]   M_sec[Msun]   M_sec[M_J]     K[m/s]    a0[mas]
    90.00     0.01050        11.00     608.574     0.1122
    30.00     0.02100        22.00     603.557     0.2225
    10.00     0.06047        63.35     585.588     0.6215
     6.20     0.09723       101.85     570.008     0.9727
     4.84     0.12445       130.37     559.117     1.2213
```

**K moves 8% across an order of magnitude in true mass — RV is effectively blind.** a0 moves
11x, from 0.11 to 1.22 mas, against a DR4 per-CCD error of 0.125 mas over 558 abscissae.

**The picture to show** is the ridge walk: pin K at 615 m/s and P at 83.9151 d, re-derive M_sec
at each inclination, and evaluate each channel separately:

```
i[deg]  M_sec[MJ] Msini[MJ]    K[m/s]    a0[mas]     logL_RV    logL_AST  logL_joint
  1.50   596.5413    15.616     615.0     4.3298   -210271.8    -65550.5   -275822.2
  2.00   408.2661    14.248     615.0     3.2476   -210271.8    -35189.8   -245461.6
  3.00   249.1474    13.039     615.0     2.1656   -210271.8    -18940.9   -229212.7
  4.00   179.0320    12.489     615.0     1.6248   -210271.8    -16110.1   -226381.9
  4.84   144.7743    12.215     615.0     1.3433   -210271.8    -16034.2   -226306.0
  6.00   114.5240    11.971     615.0     1.0843   -210271.8    -16811.8   -227083.6
  8.00    84.2414    11.724     615.0     0.8144   -210271.8    -18490.7   -228762.5
 12.00    55.2410    11.485     615.0     0.5451   -210271.8    -21062.1   -231333.9
 20.00    33.0410    11.301     615.0     0.3314   -210271.8    -23779.9   -234051.7
 40.00    17.3767    11.170     615.0     0.1763   -210271.8    -26267.7   -236539.4
 90.00    11.1169    11.117     615.0     0.1133   -210271.8    -28066.9   -238338.7
```

**The RV column is identical to every printed digit across a 54x range in true mass; the
astrometry column swings 5e4 nats and peaks at i = 4.84 deg.** HD 114762 b at
M sin i = 12 M_J is a 0.10-0.14 Msun **star**.

> Absolute values are meaningless here — omega, M0 and Omega are held at plausible-but-unfitted
> values and only the RV offset is re-optimised. The claims are the *flatness* of the RV column
> and the *curvature* of the astrometry column.

**What the mass estimate then rests on.** Since a0 fixes only the product, and the parallax
prior sets the scale, a quoted companion mass propagates the uncertainty in **plx** and in
**M_pri**. If the host mass is held fixed — which is the usual default — the quoted interval
does not include host-mass uncertainty. Say so.

---

# Part 4 — Worked example: Gaia-4, bytes to orbit

One source, followed end to end. Every number below is measured.

## 4.1 Raw rows

```
rows in stream (field-of-view transits, all sources) = 1008
distinct source_id = 12
Gaia-4 (1457486023639239296): 115 FoV transit rows

transit_id      = 17414137829279201
  ra0,dec0      = 209.5063269, 31.6954997 deg
  g_mag         = 11.909   agis_source_excess_noise = 0.1190 mas
  obs_time_tcb  = [147896865626617553, 147896875541906333, 147896880397700773] ... n_ccd=10
  obs_time_bary_corr = -3.426657e+11 ns (transit scalar)
  scan_pos_angle= [-40.6496, -40.65003, -40.65024] deg
  parallax_factor_al = +0.215413   parallax_factor_ac = +0.722171
  centroid_pos_al    = [-173.2879, -173.1236, -173.3955] mas
  centroid_pos_error_al = [0.1264, 1.8056, 0.6701] mas
  centroid_pos_ac    = EMPTY (length 0 - 1-D windows carry no across-scan)
  used_by_agis_al    = ['F', 'F', 'F']
  n_ccd used AL      = 0 / 10

transit_id      = 43662094342451958
  obs_time_bary_corr = -5.597093e+10 ns
  scan_pos_angle= [-102.99537, -102.99552, -102.99558] deg
  parallax_factor_al = -0.693010
  centroid_pos_al    = [-114.5451, -114.7407, -114.5629] mas
  centroid_pos_error_al = [0.1128, 0.1132, 0.0748] mas
  used_by_agis_al    = ['F', 'T', 'T']
  n_ccd used AL      = 9 / 10
```

Three things to say from this:

- **`centroid_pos_al` is not a residual.** -173 mas is real astrometric signal, mostly proper
  motion.
- **Per-CCD error within one transit varies by 20x** (0.126, 1.806, 0.670 mas). AGIS's own flag
  is doing the outlier rejection; trust it and apply no sigma cut beyond sigma > 0.
- **Whole transits get rejected**: 22 of Gaia-4's 115 have zero usable CCDs; 158 of 1008
  file-wide. And `obs_time_bary_corr = -3.43e11 ns = -342.7 s` is the Roemer delay — the right
  order for L2, and a good check that the column is being read as nanoseconds.

## 4.2 What comes out

```
N = 824 along-scan abscissae (used_by_agis_al = true, finite, sigma > 0)
  time span : MJD 57038.634 -> 58843.174  (1804.5 d = 4.94 yr)
  dt        : -2.4579 -> 2.4827 Julian yr rel. J2017.5
  abscissa  : min -149.141  max 176.369  mean -20.537  sd 78.742  mas   (rms 81.33)
  sigma     : min 0.0494  median 0.0819  max 0.2123 mas
  parallax factor: -0.7180 -> 0.7264 (mean 0.0005)

FoV transits (unique transit_id) = 115 ; AL CCD abscissae kept = 824 => 7.17 CCDs/transit
Gaia-4: 1150 CCD windows total, 824 used_by_agis_al = true (71.7%)
```

The rows the likelihood actually sees:

```
   i        t [MJD]    dt [yr]    psi [deg]     p_AL  w=absc[mas]    sigma[mas]
   1    57383.44969   -1.51383  -102.9955   -0.69301    -114.7407    0.1132
   2    57383.44973   -1.51383  -102.9956   -0.69301    -114.5629    0.0748
   3    57383.44979   -1.51383  -102.9956   -0.69301    -114.8388    0.0726
   4    57383.44985   -1.51383  -102.9957   -0.69301    -114.7313    0.0722
   5    57383.44991   -1.51383  -102.9958   -0.69301    -114.6995    0.0720
   6    57383.44997   -1.51383  -102.9959   -0.69301    -114.7976    0.0805
   7    57383.45002   -1.51383  -102.9959   -0.69301    -114.8598    0.0824
   8    57383.45008   -1.51383  -102.9960   -0.69301    -114.7179    0.0694
```

Eight rows spanning 3.4 s of wall clock. **The 824 "data points" are 115 independent sky
samplings, each measured about seven times.** The likelihood does not know that.

Scan-angle coverage is what turns 1-D measurements into a 2-D solution:

```
scan angle, 12 x 30 deg bins:
    0- 30 :   35  #########            180-210 :   45  ############
   30- 60 :   81  #####################          210-240 :   71  ##################
   60- 90 :   97  #########################      240-270 :   53  ##############
   90-120 :  107  ###########################    270-300 :   90  #######################
  120-150 :   27  #######                        300-330 :  107  ###########################
  150-180 :   61  ################               330-360 :   50  #############
```

Full 360 deg coverage, factor-4 non-uniform, **101 distinct scan angles**. The angle in the
file runs -167.16 to +173.45 deg; nothing needs range-checking or wrapping, because only its
sine and cosine are ever used.

**HD 114762 is the cautionary counter-example**: 558 abscissae but only **70 distinct scan
angles**, and in 30-degree bins from -180: `[62, 88, 72, 0, 0, 124, 17, 44, 52, 0, 0, 99]` —
modulo 180 degrees, a genuine 60-degree hole with zero coverage. N is not the figure of merit;
angular coverage is.

## 4.3 The 5-parameter solution vs ESA

```
max |ours - ESA| over (plx, pmra*, pmdec), 12 sources = 5.912e-04 mas (or mas/yr)
  1457486023639239296 : delta = -1.34e-05, +1.98e-04, -8.37e-05
  1663617687609809280 : delta_max = 5.91e-04
  2237987199365376    : delta_max = 1.55e-06
```

**Units, signs, the ns->yr conversion, the J2017.5 epoch and the barycentric correction are
all confirmed simultaneously by this single comparison** — get any one of them wrong and the
parallax moves by percent, not by 1e-4. If you write your own reader, this is the test to run
first.

Gaia-4 in detail:

```
  dRA*  = +0.000312 mas      (offset from ra0  = 209.5063269 deg)
  dDec  = +0.000819 mas      (offset from dec0 = 31.6954997 deg)
  plx   = +13.62146 mas      [DR3 catalogue 13.5448; literature adopted 13.628 +/- 0.021]
  pmra* = -75.55114 mas/yr
  pmdec = +17.94038 mas/yr

  residuals of the single-star fit (orbit NOT modelled):
  RMS  = 0.1543 mas   (in sigma units 1.8654;  median sigma = 0.0819 mas)
  min/max = -0.5466 / +0.7732 mas ; chi2 = 2867.2 for 824 - 5 = 819 dof => chi2/dof = 3.50
```

Per-transit weighted-mean residuals make the orbit visible with **no orbital model at all**:

```
  per-FoV-transit weighted mean residual (first 12 transits, time-ordered):
  (93 groups recovered by 0.01-d clustering of the abscissa epochs)
         t [MJD]  n_ccd    <r> [mas]      sigma
     57038.63423      8      +0.1527     0.0241
     57038.70829      9      +0.2247     0.0442
     57059.63340      9      +0.2201     0.0306
     57155.75682      9      -0.1258     0.0279
     57155.83083      9      -0.0909     0.0318
     57170.83402      9      +0.0409     0.0270
     57209.22451      9      -0.0579     0.0284
     57209.29847      8      -0.0885     0.0299
     57245.02977      9      -0.0054     0.0296
     57245.27991      9      +0.0450     0.0304
     57266.21044      9      -0.0270     0.0218
     57266.28445      9      +0.0372     0.0296

  scatter of the per-transit mean residuals = 0.1221 mas (peak -0.2452 / +0.3165)
```

0.15-0.22 mas deviations against 0.024-0.044 mas transit errors — **5-7 sigma per transit,
coherent in time.** The signal is naked-eye before any Keplerian. This is the slide that makes
epoch astrometry feel real to an audience.

## 4.4 The orbit, from astrometry alone

An 8-parameter fit (a, M_sec, sqrt(e) sin w, sqrt(e) cos w, M0, i, Omega, plx) with an
informative parallax prior, sampled with parallel tempering, **6.4 minutes on four cores**:

```
Posterior medians [+1 sigma, -1 sigma]      (published):
  P_days =    578.6 [+3.0, -3.0]     (571.3)
  M_Jup  =    10.83 [+0.38, -0.34]   (11.8)
  e      =    0.406 [+0.062, -0.061] (0.338)
  i_deg  =    121.2 [+2.3, -2.3]     (116.9)
```

A published Gaia+RV orbit, recovered **from astrometry alone**.

Now the money numbers — take the posterior median, project it onto each scan direction,
subtract, and refit the five nuisances:

```
posterior-median orbit: P=578.56 d, e=0.309, w=29.9 deg, Omega=176.1 deg, i=121.2 deg,
                        M_sec=0.01033 Msun (10.83 M_J)
a0 = a_rel (M_sec/M_tot) plx = 1.1797 AU x 0.01579 x 13.628 mas = 0.2539 mas
orbit AL signal: min -0.2784  max +0.3076  peak-to-peak 0.5860 mas  (median sigma 0.0819)

=== along-scan residuals, N = 824 abscissae ===
model                                        RMS[mas]   RMS/sigma      chi2 chi2/dof
5-param single star (orbit NOT modelled)       0.1543      1.8654    2867.2     3.50
5-param + Keplerian photocentre orbit          0.1017      1.1821    1151.4     1.42

delta chi2 = 1715.8 for 7 extra orbital parameters
RMS drops 0.1543 -> 0.1017 mas (34%); scatter now 1.24x the median per-CCD sigma

5-param solution shifts once the orbit is in the model:
                 no orbit     with orbit            delta
  dRA*[mas]       0.00031       -0.06542     -0.06573
  dDec[mas]       0.00082        0.12033      0.11951
  plx[mas]       13.62146       13.62040     -0.00106
  pmra*[mas/yr] -75.55114      -75.52927      0.02186
  pmdec[mas/yr]  17.94038       17.96491      0.02452

log-likelihood: single star = -163.98 ; with orbit = 693.94 ; delta = +857.92 nats
```

delta chi2 / 2 = 858.0 and the log-likelihood difference is 857.9 — **agreement to 0.1 nat**, an
internal check that the marginalisation is doing nothing unexpected.

**The bottom table is the underrated one.** Including the orbit moves pmdec by 0.025 mas/yr and
the position by 0.12 mas — about 8 sigma on the catalogue-quoted PM errors. **An unmodelled
0.25 mas photocentre orbit does not merely inflate residuals; it biases the astrometric
parameters.** That is exactly why HD 114762's DR3 pmDec (+1.062) and its orbit-marginalised
value (-0.549) differ by 1.6 mas/yr.

**Do not claim the residual floor is white.** 0.102 mas is 1.24x the median per-point error, so
something is still unmodelled at the ~0.06 mas level. Candidates: the chromatic term, correlated
per-transit attitude error, or the 0.119 mas excess noise not added in quadrature.

## 4.5 The ladder, before and after the Keplerian

```
ladder on raw abscissae : P = [0.000, 0.014, 0.986], best 9p, chi2/dof = [3.50, 3.39, 3.38], adequate = false
ladder after the Kepler : P = [0.993, 0.006, 0.001], best 5p, chi2/dof = [1.41, 1.41, 1.41], adequate = true
delta logZ(5p) from removing the Keplerian = +858.0 nats
```

Before: "9p, and none of them fit." After: "5p, P = 0.993, adequate." +858 nats — matching the
likelihood difference of 4.4 to 0.1 nat **by a completely independent route** (closed-form
Gaussian evidence at 22.5 microseconds versus an MCMC-marginalised likelihood).

That is a self-contained argument that the curvature in Gaia-4's abscissae is **Keplerian, not
generic polynomial drift**: a cubic in time cannot absorb it, and a Kepler orbit restores the
star to "5-parameter, adequate".

## 4.6 Cross-check against the published DR3 catalogue

```
DR4 epoch fit (ours) vs Gaia DR3 catalogue

source_id              plx_us  plx_cat  n_sig    pmra_us  pmra_cat  n_sig   pmdec_us  pmdec_cat  n_sig   RUWE  solv
2237987199365376      -0.0891  -0.2860    0.3    -0.0093   -0.0222    0.0     0.1054    -0.0259    0.3   1.15    31
2309425390592896       0.9269   0.9317   -0.2     7.1999    7.2472   -2.0    -3.3227    -3.3119   -0.5   0.97    31
20694084440761600      4.9421   4.9858   -1.9    25.4424   25.4539   -0.5   -16.5051   -16.5554    2.7   1.04    31
435469040545191680     0.9977   1.0215   -1.0     3.4218    3.4502   -1.2    -5.8124    -5.8213    0.5   0.94    31
1457486023639239296   13.6214  13.5448    3.5   -75.5511  -75.4116   -6.9    17.9404    18.0633   -7.8   1.50    31
1663617687609809280   10.1051  10.1265   -1.3  -124.5725 -124.5857    0.7  -135.8239  -135.8230   -0.1   1.08    31
3937211745905473024   25.6455  26.1979   -5.1  -582.0545 -580.9987   -8.4    -0.7279     1.0621  -12.6   3.16    95
4181040337841125632    1.0013   1.0281   -1.6     2.9041    2.8742    1.7    -0.2945    -0.2868   -0.6   0.88    31
4318465066420528000    0.9833   1.6443   -9.5   -30.9460  -22.2346 -139.7  -148.5497  -155.2765  114.6   3.41    31

adequate sources compared: 9   (27 parameter comparisons)
|n_sigma|: median 0.76  max 2.70
solution-type agreement (ours 5p vs catalogue 5p/6p): 8/9
```

**This is what empirically certifies the scan-angle convention.** Using `deg2rad(scan_pos_angle)`
directly with the along-scan unit vector (sin psi, cos psi) reproduces the catalogue proper
motion in *both* components, in sign and magnitude, on nine sources. A swapped sine/cosine
shows up as an RA/Dec transposition instantly.

Expect 1-2 sigma disagreement and do **not** "fix" it: the catalogue is DR3 (~2.8 yr baseline)
and the epoch data is DR4-format (~5.5 yr). What must not happen is a sign error or a factor of
2 or 1000.

**Look at the three excluded sources anyway.** Gaia-4 at 6.9/7.8 sigma in PM and BH3 at
140/115 sigma are not parser errors — they are the orbit contaminating both the DR3 catalogue
solution and the single-star DR4 fit, *differently*, because the baselines differ. **RUWE
tracks it exactly**: 0.88-1.15 for the well-behaved sources, 1.50 for Gaia-4, 3.16 and 3.41 for
the two loud ones — and RUWE falls straight out of chi2/dof on the epoch abscissae while never
being used in the computation.

## 4.7 Second worked example: HD 114762, RV + DR4 jointly

13 free parameters (the 7 orbital, plx, two RV offsets, two RV jitters, one RV trend),
**8.0 minutes**:

```
RV: 24 HIRES + 35 Lick, baseline 29.3 yr
AST: 558 DR4 along-scan abscissae

posterior-median per-channel logL:  RV = -200.2 (N=59)   ASTROM = -2146.1 (N=558)
  -> RV chi2/N = 0.97

Posterior medians [+1 sigma, -1 sigma]      (Kiefer+ 2019):
  P_days   =    83.92 [+0.00, -0.00]      (83.92)
  e        =    0.343 [+0.001, -0.002]    (0.335)
  M sin i  =    12.27 [+0.03, -0.04] M_J  (~11, the RV-only value)
  i_deg    =     4.84 [+0.08, -0.08]      (6.2 +1.9 -1.3)
  M_true   =    0.139 [+0.002, -0.003] Msun  (0.103 +0.030 -0.025, a STAR)
           =    145.4 [+2.5, -2.8] M_J        (108 +31 -26)     -> +1.21 sigma from Kiefer+ 2019

  -> RV alone: M sin i = 12.3 M_J (looks like a giant planet)
  -> RV + DR4 astrometry: i = 4.8 deg => M = 0.139 Msun (a low-mass star)
```

Same before/after treatment:

```
HD 114762: N=558 abscissae, sigma_med = 0.1254 mas
  a0 = 1.7526 mas ; AL orbit peak-to-peak = 4.654 mas
  AL residual RMS  5p only = 1.2630 mas -> 5p+Kepler = 0.3810 mas
  chi2 58791.1 -> 5530.7  (chi2/dof 106.3 -> 10.13)
  5-param plx 25.6457 -> 25.3147 mas ; pmRA* -582.056 -> -582.443 ; pmDec -0.728 -> -0.549
  (Gaia DR3 catalogue: plx 26.198, pmRA* -580.999, pmDec +1.062, RUWE 3.16)
```

**Be honest about the residual: chi2/dof goes 106 -> 10.1, not -> 1.** Real structure remains —
the known wide outer M-dwarf companion, or a photocentre effect the point-mass reflex model does
not capture. The mass measurement survives (RV chi2/N = 0.97, and the inclination is pinned by a
1.75 mas photocentre orbit against 0.125 mas errors), but this is not a "the model fits" slide.

And note again: the parallax moves 0.33 mas and pmDec moves 1.6 mas/yr once the orbit is in the
model, on a source whose DR3 RUWE is 3.16. **The catalogue's own 5-parameter solution is
measurably corrupted by the companion.**

## 4.8 What the excess noise actually is

```
source_id                   G excess[mas]  chi_RMS   RMS[mas]    med sigma  note
1457486023639239296     11.91     0.1190     1.865     0.1543     0.0819  Gaia-4
1663617687609809280     14.26     0.0000     1.037     0.1703     0.1568
4181040337841125632      8.97     0.0000     0.971     0.1114     0.1045
2309425390592896        14.01     0.0000     0.993     0.1477     0.1419
3937211745905473024      7.15     1.2272    10.265     1.2630     0.1254  HD 114762
4318465066420528000     11.23     6.5829    81.198     6.6318     0.0853  Gaia BH3

  3937211745905473024   sqrt(sigma^2+eps^2)=1.2335  actual RMS=1.2630  ratio=1.02  HD 114762
  4318465066420528000   sqrt(sigma^2+eps^2)=6.5835  actual RMS=6.6318  ratio=1.01  Gaia BH3
  1457486023639239296   sqrt(sigma^2+eps^2)=0.1445  actual RMS=0.1543  ratio=1.07  Gaia-4
  4181040337841125632   sqrt(sigma^2+eps^2)=0.1045  actual RMS=0.1114  ratio=1.07
```

`agis_source_excess_noise` is **0.0000 for every well-behaved source** and 0.119 / 1.227 /
6.583 mas for **exactly the three orbital ones**, and sqrt(sigma^2 + eps^2) predicts the
single-star residual RMS to 1-7%.

**The excess noise IS the unmodelled orbit**, quantified by AGIS and shipped in the file — a
one-line detection statistic computable at catalogue scale before fitting anything.

It is also exactly why you should **not** fold it into the per-point errors before fitting an
orbit: doing so absorbs the signal you are trying to measure. The cost of leaving it out is that
your formal parameter errors are optimistic. Pick deliberately and say which you did.

## 4.9 Cost

```
parse the whole 1.18 MB VOTable (12 sources)      = 0.380 s
one orbit log-likelihood on 824 abscissae         = 230.1 microseconds
  -> a 12-round x 8-chain tempered run does ~65,520 evals; measured wall time 6.4 min
closed-form ladder evidence, one rung             = 22.5 microseconds
```

That last number is what makes a per-source model posterior affordable at catalogue scale
rather than a demonstration on a handful of stars. DR4 ships ~1.8 billion sources; three matrix
factorisations each is a very different proposition from an MCMC each.

---

# Part 5 — Traps, in the order they will bite

1. **Filter on `used_by_agis_al`, not on finiteness.** A finite-and-positive test alone keeps a
   46-arcsecond centroid whose quoted error is a perfectly ordinary 0.30 mas (1.4). AGIS's flag
   is the outlier rejection.
2. **Guard every quantity for finiteness, not just the abscissa.** `obs_time_bary_corr`,
   `parallax_factor_al` and `zeta` are NULL (NaN) in 97 of 1008 RC3 rows, and
   `scan_pos_angle` is NaN in 1139 of 10080 CCD slots. In RC3 all 97 NaN-time rows happen to
   carry zero usable CCDs, so a flag filter alone is enough — **that is a coincidence of this
   file, not a property of the format.** A usable abscissa beside a NULL time would put NaN into
   your epoch and your PM baseline, and the likelihood would silently go non-finite, which reads
   as a bad model rather than bad data.
3. **`centroid_pos_al` is not a residual.** RMS 81 mas for Gaia-4 against 2.3 mas for the
   Hipparcos analogue. Treating it like Hipparcos `RES` is a factor-1000 error.
4. **Times are nanoseconds since 2010-01-01 TCB**, and `obs_time_bary_corr` is a per-transit
   scalar in ns that must be added to every per-CCD time. Magnitude +/-360 s. The
   `refposition="BARYCENTER"` label in the TIMESYS block does not describe the stored column.
5. **`parallax_factor_al` is per-transit; `scan_pos_angle` is per-CCD.** Broadcasting the wrong
   one is silent.
6. **824 abscissae are 115 independent pointings.** Standard treatments assume independence
   within a transit; the data are consistent with that but do not exclude a ~20% correlated
   component. N is not the figure of merit — angular coverage is (HD 114762: 558 abscissae, 70
   distinct angles, a genuine 60-degree hole).
7. **The scan-angle convention is invisible to a 5-parameter fit and worth 1033 nats to an
   orbit.** Certify it against published proper motions in *both* components before fitting
   anything (4.6). For Hipparcos IAD it is `atan2(CPSI, SPSI)`, against the column names.
8. **Model probabilities normalise over the rungs you supply and cannot say whether any of them
   fits.** Always read a chi2/dof beside them. Gaia BH3: P(9p) = 1.000, chi2/dof = 1140.
9. **Never read `best` without `prob`.** Three near-identical evidences make the argmax noise
   (source 60730287810150016: best = 9p at P = 0.34).
10. **Bayes factors between rungs need proper priors.** A flat improper prior on the linear
    block gives undefined model comparison, and the prior width moves the answer by ~4.6 nats
    per decade.
11. **The parallax prior sets the mass scale**, not the abscissae: a0 = a_rel M_sec/M_tot plx,
    and a factor 16 in mass along the ridge costs 9.5 nats.
12. **An unmodelled orbit biases the catalogue parameters**, it does not just inflate the
    residuals — 8 sigma on Gaia-4's proper motion, 1.6 mas/yr on HD 114762's pmDec.
13. **RC3 has no duplicate `transit_id`.** Do not repeat a "the pre-release ships duplicates"
    claim. It does have 24 near-simultaneous re-detections, and AGIS has already flagged one
    member of every pair.
14. **"post_rms" style diagnostics are usually in sigma units, not mas.** HD 114762's 10.26 and
    BH3's 81.20 are dimensionless. Check before quoting.

---

# For the talk

**The four things an audience most needs to understand.**

**1. A scanning astrometer measures a 1-D projection, and DR4 gives you nothing else.** One row
is one field-of-view transit carrying up to 9 usable along-scan centroids over 46 s; the
across-scan columns are empty in the pre-release. The only thing that turns 1-D measurements
into 2-D astrometry is the scan angle rotating between transits — and it does not rotate
uniformly (Gaia-4: 101 distinct angles, full coverage; HD 114762: 70 angles with a genuine
60-degree hole). **The scan angle is half the data, not metadata.**

**2. DR4 epoch astrometry is Hipparcos IAD with better numbers.** Same design vector
x = (sin psi, cos psi, p_AL, sin psi dt, cos psi dt), same marginalised 5-vector, same
likelihood. The one structural difference — Hipparcos ships residuals, DR4 ships absolute
positions — is *provably* invisible: adding an arbitrary 5-parameter linear signal changes
log L by 3.6e-7 nats out of 575, because the 5-vector carries a flat improper prior. What
changed is the numbers: for HD 114762, 53 -> 558 points, 2.29 -> 0.125 mas, and a cumulative
reflex S/N of 4 -> 220. **A factor 55, on the same star with the same model.**

**3. The astrometry measures an angle; only the parallax prior turns it into a mass.**
a0 = a_rel M_sec/(M_pri+M_sec) plx. Walking the ridge M_sec x plx = const at fixed period costs
**9.5 nats over a factor of 16 in companion mass**; moving plx alone by a factor of 2 costs
**1233 nats**. The parallax that scales the reflex never even enters the data term — the
parallactic signal in the abscissae is one of the five marginalised nuisances. Adding RV closes
it: across an order of magnitude in true mass K moves 8% while a0 moves 11x, which is how
HD 114762's M sin i = 12.27 M_J becomes M = 0.139 Msun at i = 4.84 deg.

**4. An unmodelled orbit does not just inflate residuals — it biases the catalogue solution,
and the file tells you so.** `agis_source_excess_noise` is 0.0000 for every quiet source and
0.119 / 1.227 / 6.583 mas for exactly the three orbital ones, and sqrt(sigma^2 + eps^2) predicts
the single-star residual RMS to 1-7%. Meanwhile fitting the orbit shifts Gaia-4's pmdec by
0.025 mas/yr (~8 sigma) and HD 114762's pmDec by 1.6 mas/yr against DR3. RUWE (1.50, 3.16, 3.41
versus 0.88-1.15) falls straight out of the epoch chi2/dof without ever being used in the
computation.

**The single most effective demonstration** — the solution ladder on Gaia-4, run twice:

```
ladder on raw abscissae : P = [0.000, 0.014, 0.986], best 9p, chi2/dof = [3.50, 3.39, 3.38], adequate = FALSE
ladder after the Kepler : P = [0.993, 0.006, 0.001], best 5p, chi2/dof = [1.41, 1.41, 1.41], adequate = TRUE
delta logZ(5p) = +858.0 nats     [the orbit likelihood gives +857.9 - agreement to 0.1 nat]
```

It lands three separate points in one slide.

(i) The curvature in the abscissae is **Keplerian, not generic polynomial drift** — a cubic in
time cannot absorb it, and a Kepler orbit restores the star to "5-parameter, adequate".

(ii) Two completely independent routes — a closed-form Gaussian evidence at 22.5 microseconds
and an MCMC-marginalised likelihood — agree to 0.1 nat. That is a real correctness check, not a
plot.

(iii) The `adequate = FALSE` in the "before" row is the honest half: **model probabilities
normalise over the rungs you gave them and cannot tell you whether any of them fits.** Gaia BH3
gets P(9p) = 1.000 with a 1.5-million-nat evidence improvement and chi2/dof = 1140 — the ladder
confidently picks the least hopeless of three hopeless models on precisely the source that
matters most.

**If you have room for one more slide**, it is the `pre_rms` / `post_rms` column over all 12
sources (1.5): nine sources on a 1.0 floor, and exactly the three published orbital systems
above it at 1.87, 10.26 and 81.20. **That is the entire scientific case for epoch astrometry,
in one column, with no orbit model anywhere in it.**

**And the one-liner for the 7p/9p audience**: most DR4 non-single-star solutions will be
accelerations, not orbits. An acceleration measures GM_c/r^2 and nothing else, so
**M_c ~ P^(4/3)** — the same number is a Saturn at 10 yr or a 0.15 Msun star at 1000 yr. Any
single mass quoted from a 7p solution came from outside the astrometry; a 9p jerk is
qualitatively better because jerk/accel ~ 2 pi/P gives the period independently. DR4's
per-source acceleration precision, measured on RC3, is **0.005-0.01 mas/yr^2**.
