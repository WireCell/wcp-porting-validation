#!/usr/bin/env python3
"""Sidecars + manifest for an arm reconstructed over the em114c (141-event) set.

A fork of prep_pr117.py, not a generalisation of it (CLAUDE.md sec 2 "fork by
duplication"; the run_em114c_probe.sh / em117_score.py precedent).  The only
difference is WHICH scan sample it resolves run/subrun against:

    prep_pr117.py   RSE from em114-manifest.tsv    (the 97-event pr/115 scan)
    prep_pr121.py   RSE from em114c-manifest.tsv   (the 141-event pr/116 scan)

prep_pr117.py stays byte-untouched: it is the input to every pr/117-120 score
table already on disk, and a shared-RSE-source refactor would put those tables'
provenance in question for no gain.

    ./prep_pr121.py --tag 114cnow work-em114c-prodnow-mcp1k work-em114c-prodnow-mcp2k
      -> emprep-114cnow/emprep-evt<N>.json
      -> em114c-114cnow-manifest.tsv   (columns: sample run subrun event dump)
"""
import argparse, csv, glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from prep_em_scan import parse_probes  # noqa: E402  (probe parser reused)

RE_SAMPLE = re.compile(r"-([a-z0-9]+)$")

# M13: the scan-time prepdir/manifest are the record the 141 labels were made
# against.  Refuse to write over either, or over the pr/115 pair.
PROTECTED = ("emprep", "emprep-c", "em114-manifest.tsv", "em114c-manifest.tsv")


def rse_of():
    """run/subrun from the em114c manifest (RSE is stable per event)."""
    man = os.path.join(HERE, "em114c-manifest.tsv")
    out = {}
    with open(man) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[int(r["event"])] = (r["run"], r["subrun"], r["sample"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="arm tag, e.g. 114cnow")
    ap.add_argument("roots", nargs="+", help="arm work roots")
    args = ap.parse_args()

    prepdir = os.path.join(HERE, "emprep-%s" % args.tag)
    manifest = os.path.join(HERE, "em114c-%s-manifest.tsv" % args.tag)
    for p in (prepdir, manifest):
        if os.path.basename(p) in PROTECTED:
            sys.exit("refuse to touch the scan-time %s (M13)" % p)

    roots = [r if os.path.isabs(r) else os.path.join(SX, r) for r in args.roots]
    for r in roots:
        if not os.path.isdir(r):
            sys.exit("no such arm root: %s" % r)
    parse_probes(roots, prepdir)

    rse = rse_of()
    rows = []
    for root in roots:
        m = RE_SAMPLE.search(os.path.basename(root.rstrip("/")))
        sample = m.group(1) if m else "?"
        for dump in sorted(glob.glob(os.path.join(root, "pr_evt*",
                                                  "calib-pr-evt*.json"))):
            evt = int(re.search(r"pr_evt(\d+)", dump).group(1))
            run, subrun, _ = rse.get(evt, ("0", "0", sample))
            rows.append((sample, run, subrun, evt, os.path.relpath(dump, SX)))
    rows.sort(key=lambda r: r[3])
    with open(manifest, "w") as fh:
        fh.write("sample\trun\tsubrun\tevent\tdump\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print("wrote %s (%d rows) and %s" % (manifest, len(rows), prepdir))


if __name__ == "__main__":
    main()
