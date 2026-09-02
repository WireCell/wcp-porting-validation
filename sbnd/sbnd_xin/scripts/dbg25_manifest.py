#!/usr/bin/env python3
"""doc 95 -- build the Bee manifest for the 25-event MC debug sample.

Order is ART-FILE ORDER (entry 0..24).  Nothing in the request named a
reference set to line up with, and the Bee index is quoted forever once the
owner scans, so the file's own order is the one defensible choice.

The group (a/b) is DERIVED here the same way scripts/dbg25_groups.sh derives
it -- first occurrence of a bare event id goes to a, later ones to b -- so the
manifest cannot disagree with which work root actually holds the event.

Repro:
  python3 scripts/dbg25_manifest.py -o bee/dbg25/dbg25.manifest.tsv
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(HERE, "input_files_reco1", "staged-dbg25", "entry_event_map.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--map", default=MAP)
    args = ap.parse_args()

    rows, seen = [], {}
    with open(args.map) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        assert head[:5] == ["entry", "run", "subrun", "event", "caf_ns"], head
        for line in fh:
            if not line.strip():
                continue
            e, r, s, ev, _ = line.rstrip("\n").split("\t")[:5]
            grp = "b" if ev in seen else "a"
            seen[ev] = seen.get(ev, 0) + 1
            rows.append((int(e), int(r), int(s), ev, grp))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    missing = []
    with open(args.output, "w") as fh:
        fh.write("# doc 95 -- Bee manifest, ART-FILE ORDER (bee_idx == art entry)\n")
        fh.write("# group a = first occurrence of a bare event id (20 events)\n")
        fh.write("# group b = the 5 second occurrences (ids 12,14,22,31,34)\n")
        fh.write("bee_idx\tentry\trun\tsubrun\tevent\tql_root\tpr_root\n")
        for i, (e, r, s, ev, grp) in enumerate(rows):
            ql = f"work-dbg25{grp}-ql"
            pr = f"work-dbg25{grp}-pr"
            for p in (os.path.join(HERE, ql, f"ql_evt{ev}", "mabc-all-apa.zip"),
                      os.path.join(HERE, pr, f"pr_evt{ev}", "mabc-pr.zip")):
                if not os.path.isfile(p):
                    missing.append(f"entry {e} ({r}-{s}-{ev}): {p}")
            fh.write(f"{i}\t{e}\t{r}\t{s}\t{ev}\t{ql}\t{pr}\n")
    print(f"wrote {args.output}: {len(rows)} rows")
    if missing:
        print(f"!! {len(missing)} inputs not present yet:", file=sys.stderr)
        for m in missing[:10]:
            print("   " + m, file=sys.stderr)
        return 1
    print("all 25 inputs present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
