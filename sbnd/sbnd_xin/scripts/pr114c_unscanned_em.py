#!/usr/bin/env python3
"""doc pr/114 round 7 -- the beam EM-shower events the pr/114 scan never covered.

The pr/114 display sample is the corrected NCpi0 list (46) + the curated nueCC48
arm (48) + 4 owner adds = 98.  That is a *topology-curated* sample, not a
coverage-complete one: it was built from two of the three pr/113 verdict lists,
and `prep_em_scan.scan_sample()` further keeps only the `nuecc48` arm rows of the
nueCC list.  So of the 171 mcp1k+mcp2k events whose leading EM shower clears
100 MeV, only 30 are on screen.

This script names the other 141, tagged by which pr/113 bucket they fall in:

  numucc_em   is_numucc_em   muon primary >= 30 cm and em_max >= 100      (79)
  nuecc       is_nuecc_reco  no muon, no gamma pair, vertex-rooted e      (40)
  other_em    none of the three -- em_max >= 100 but no muon primary, no
              gamma pair and no vertex-rooted electron.  These fall through
              pr/113's own ladder and appear on no delivered list.         (22)

"Unscanned" is defined against the manifest the live scan reads, NOT against
em_labels/: an event in the manifest is on screen whether or not the owner has
got to it yet.  Both inputs are read-only.

Usage:
  python scripts/pr114c_unscanned_em.py            # writes docs/pr/pr114c-*.index.txt
  python scripts/pr114c_unscanned_em.py --bucket numucc_em --out /dev/stdout
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)

BEAM = ("mcp1k", "mcp2k")
EM_FLOOR = 100.0


def bucket(r):
    """The pr/113 priority ladder, plus the bucket pr/113 has no name for."""
    if int(r["is_numucc_em"]):
        return "numucc_em"
    if int(r["is_ncpi0_reco"]):
        return "ncpi0"
    if int(r["is_nuecc_reco"]):
        return "nuecc"
    return "other_em"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=os.path.join(
        SX, "docs", "pr", "pr113-emshower-sample.tsv"))
    ap.add_argument("--manifest", default=os.path.join(
        SX, "em_display", "em114-manifest.tsv"))
    ap.add_argument("--out", default=os.path.join(
        SX, "docs", "pr", "pr114c-beam-em100-unscanned.index.txt"))
    ap.add_argument("--bucket", default=None,
                    help="restrict to one bucket (numucc_em|nuecc|other_em)")
    args = ap.parse_args()

    with open(args.census) as fh:
        census = list(csv.DictReader(fh, delimiter="\t"))
    with open(args.manifest) as fh:
        scanned = {(r["sample"], r["event"])
                   for r in csv.DictReader(fh, delimiter="\t")}

    beam = [r for r in census
            if r["sample"] in BEAM and float(r["em_max"]) >= EM_FLOOR]
    rows = [r for r in beam if (r["sample"], r["evt"]) not in scanned]
    if args.bucket:
        rows = [r for r in rows if bucket(r) == args.bucket]
    # sample then numeric event: the same order run_em114_probe.sh feeds the
    # chain in, so an arm's pr_evt* dirs and the manifest read alike.
    rows.sort(key=lambda r: (r["sample"], int(r["evt"])))

    with open(args.out, "w") as fh:
        fh.write("# doc pr/114 r7 -- mcp1k+mcp2k events with em_max >= %g MeV\n"
                 "# that are NOT in %s\n"
                 "# sample\torigin\trun\tsubrun\tevent\tem_max_MeV\tem_tier"
                 "\tEnu_MeV\tmu_len_cm\tn_pio_showers\tpio_mass\n"
                 % (EM_FLOOR, os.path.relpath(args.manifest, SX)))
        for r in rows:
            fh.write("\t".join([
                r["sample"], bucket(r), r["run"], r["subrun"], r["evt"],
                r["em_max"], r["em_tier"], r["Enu"], r["mu_len"],
                r["n_pio_showers"], r["pio_mass"]]) + "\n")

    from collections import Counter
    c = Counter(bucket(r) for r in rows)
    print("beam events with em_max >= %g MeV : %d" % (EM_FLOOR, len(beam)))
    print("  already in %s : %d"
          % (os.path.basename(args.manifest), len(beam) - len(
              [r for r in beam if (r["sample"], r["evt"]) not in scanned])))
    print("wrote %d rows to %s" % (len(rows), args.out))
    for k in ("numucc_em", "nuecc", "other_em", "ncpi0"):
        if c[k]:
            print("  %-10s %3d   (mcp1k %d, mcp2k %d)"
                  % (k, c[k],
                     sum(1 for r in rows
                         if bucket(r) == k and r["sample"] == "mcp1k"),
                     sum(1 for r in rows
                         if bucket(r) == k and r["sample"] == "mcp2k")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
