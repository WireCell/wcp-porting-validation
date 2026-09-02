#!/usr/bin/env python3
"""doc 95 -- per-event tagger table for the 25-event MC debug sample.

One row per IN-BEAM bundle (in_beam==1), joined to the Bee index and the RSE.
Four of the 25 events have two in-beam bundles, so rows > events on purpose.

nusel-evt<ID>.tsv is SPACE-PADDED, not tab-separated -- split on whitespace.

Written atomically (tmp + rename): a killed builder must not leave a short
file that the next reader mistakes for a complete table.

Repro:
  python3 scripts/dbg25_table.py -m bee/dbg25/dbg25.manifest.tsv \
      -o bee/dbg25/dbg25-tagger-summary.tsv
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLS = ("bee_idx", "entry", "rse", "verdict", "main_id", "t0_us", "flash_pe",
        "len_main_cm", "npts_main", "tgm", "stm", "fc", "lm", "stmfit",
        "n_bundle_evt", "n_inbeam")


def read_manifest(path):
    rows = []
    with open(path) as fh:
        head = None
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            f = line.split("\t")
            if head is None:
                head = f
                continue
            rows.append(dict(bee_idx=int(f[0]), entry=int(f[1]), run=int(f[2]),
                             subrun=int(f[3]), event=f[4], ql_root=f[5],
                             pr_root=f[6]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    out = []
    for r in read_manifest(args.manifest):
        ev = r["event"]
        tsv = os.path.join(HERE, r["pr_root"], f"pr_evt{ev}", f"nusel-evt{ev}.tsv")
        if not os.path.isfile(tsv):
            sys.exit(f"ERROR: missing {tsv}")
        with open(tsv) as fh:
            lines = [l.split() for l in fh if l.strip()]
        head, data = lines[0], lines[1:]
        ix = {c: i for i, c in enumerate(head)}
        # sanity: the file must be the event the manifest claims
        for d in data:
            got = (int(d[ix["run"]]), int(d[ix["subrun"]]), int(d[ix["event"]]))
            want = (r["run"], r["subrun"], int(ev))
            if got != want:
                sys.exit(f"ERROR: {tsv} holds RSE {got}, manifest says {want} "
                         f"-- wrong work root for this event")
        inbeam = [d for d in data if d[ix["in_beam"]] == "1"]
        if not inbeam:
            inbeam = [None]
        for d in inbeam:
            if d is None:
                out.append([r["bee_idx"], r["entry"],
                            f'{r["run"]}-{r["subrun"]}-{ev}', "NO-INBEAM-BUNDLE"]
                           + ["-"] * 10 + [len(data), 0])
                continue
            g = lambda c: d[ix[c]]
            out.append([r["bee_idx"], r["entry"],
                        f'{r["run"]}-{r["subrun"]}-{ev}', g("label"),
                        g("main_id"), g("flash_time_us"), g("flash_pe"),
                        g("len_main_cm"), g("npts_main"), g("tgm"), g("stm"),
                        g("fc"), g("lm"), g("stmfit"), len(data), len(inbeam)])

    tmp = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(tmp, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for row in out:
            fh.write("\t".join(str(x) for x in row) + "\n")
    os.rename(tmp, args.output)
    print(f"wrote {args.output}: {len(out)} in-beam bundle rows")

    from collections import Counter
    c = Counter(r[3] for r in out)
    print("verdicts:", dict(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
