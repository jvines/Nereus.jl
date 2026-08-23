#!/usr/bin/env python3
"""Export RVs for a given star_id from the exoautomata postgres DB into
the Nereus CSV schema used by the test/ fit scripts.

Columns match `test/data/hd18599.csv`:
    bjd, rv, rv_error,
    bisector_span, bisector_span_error,
    fwhm, fwhm_error,
    s_index, s_index_error,
    halpha, halpha_error,
    log_rhk, log_rhk_error,
    crx, crx_error,
    instrument, provenance

Connection: uses libpq environment variables (PGHOST, PGUSER, PGPASSWORD,
PGDATABASE — set these from the exoautomata creds before running).

Usage:
    PGHOST=192.168.0.50 PGUSER=... PGDATABASE=... \\
        python3 _export_rvs.py --star-id 9303 --out hd33636.csv
    PGHOST=192.168.0.50 PGUSER=... PGDATABASE=... \\
        python3 _export_rvs.py --star-id 4793 --out eps_eri.csv

Filters:
- is_disabled = FALSE, is_outlier = FALSE
- order by BJD ascending

Outlier rejection at fit time is handled by Nereus's per-instrument
5σ MAD filter, NOT here — this script just dumps the raw clean data.
"""
import argparse
import csv
import os
import sys

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print(
        "psycopg2 not found. Install with: pip install psycopg2-binary",
        file=sys.stderr,
    )
    sys.exit(1)


COLS = [
    "bjd",
    "rv",
    "rv_error",
    "bisector_span",
    "bisector_span_error",
    "fwhm",
    "fwhm_error",
    "s_index",
    "s_index_error",
    "halpha",
    "halpha_error",
    "log_rhk",
    "log_rhk_error",
    "crx",
    "crx_error",
    "instrument",
    "provenance",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star-id", type=int, required=True)
    ap.add_argument("--out", type=str, required=True,
                    help="Output CSV path")
    ap.add_argument("--include-disabled", action="store_true",
                    help="Include rows marked is_disabled=true")
    ap.add_argument("--include-outliers", action="store_true",
                    help="Include rows marked is_outlier=true")
    args = ap.parse_args()

    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "192.168.0.50"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD", ""),
        dbname=os.environ.get("PGDATABASE", "exoautomata"),
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    where = ["star_id = %s"]
    params = [args.star_id]
    if not args.include_disabled:
        where.append("is_disabled = FALSE")
    if not args.include_outliers:
        where.append("is_outlier = FALSE")
    sql = (
        f"SELECT {', '.join(COLS[:-1])}, provenance::text AS provenance "
        f"FROM radial_velocities WHERE {' AND '.join(where)} "
        f"ORDER BY bjd ASC"
    )
    cur.execute(sql, params)
    rows = cur.fetchall()
    if not rows:
        print(f"No rows for star_id={args.star_id}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[c] if r[c] is not None else "" for c in COLS])

    print(f"Wrote {len(rows)} rows to {args.out}")

    # Quick summary
    insts = {}
    for r in rows:
        key = (r["instrument"], r["provenance"])
        insts.setdefault(key, 0)
        insts[key] += 1
    print("Per-instrument breakdown:")
    for (inst, prov), n in sorted(insts.items(), key=lambda x: -x[1]):
        print(f"  {inst:<15s} {prov:<20s} {n}")


if __name__ == "__main__":
    main()
