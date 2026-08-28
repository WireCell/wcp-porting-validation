#!/usr/bin/env python3
"""doc pr/117 -- sidecars + manifest for a KNOB arm's own reconstruction.

prep_em_scan.py's manifest hardcodes the prod0825 dumps (correct for the scan
display, whose stage-2 arms are physics-equal to prod0825).  A pr/117 knob arm
is NOT physics-equal -- that is the whole point -- so its scoring inputs must
both come from the arm itself:

    sidecar  = parsed from the arm's own stdout.log probes (membership)
    dump     = the arm's own pr_evt<N>/calib-pr-evt<N>.json

This is a round-owned helper, not an edit to prep_em_scan.py (it imports the
probe parser from there; CLAUDE.md sec 2: the production file stays
byte-untouched).  Output prepdir/manifest are NEW paths per arm tag (M13:
never emprep/, never em114-manifest.tsv).

    ./prep_pr117.py --tag 117onK1 work-pr117r1-onK1-mcp1k [more arm roots...]
      -> emprep-117onK1/emprep-evt<N>.json
      -> em117-<tag>-manifest.tsv   (columns: sample run subrun event dump)
"""
import argparse, csv, glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from prep_em_scan import parse_probes  # noqa: E402  (probe parser reused)

# arm root name -> manifest sample name: work-pr117r1-onK1-mcp1k => mcp1k
RE_SAMPLE = re.compile(r"-([a-z0-9]+)$")


def rse_of(dump_path):
    """run/subrun from the em114 manifest (RSE is stable per event)."""
    man = os.path.join(HERE, "em114-manifest.tsv")
    out = {}
    with open(man) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[int(r["event"])] = (r["run"], r["subrun"], r["sample"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="arm tag, e.g. 117onK1")
    ap.add_argument("roots", nargs="+", help="arm work roots")
    args = ap.parse_args()

    prepdir = os.path.join(HERE, "emprep-%s" % args.tag)
    manifest = os.path.join(HERE, "em117-%s-manifest.tsv" % args.tag)
    for p in (prepdir, manifest):
        if os.path.basename(p) in ("emprep", "em114-manifest.tsv"):
            sys.exit("refuse to touch the scan-time %s (M13)" % p)

    roots = [r if os.path.isabs(r) else os.path.join(SX, r) for r in args.roots]
    for r in roots:
        if not os.path.isdir(r):
            sys.exit("no such arm root: %s" % r)
    parse_probes(roots, prepdir)

    rse = rse_of(None)
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
