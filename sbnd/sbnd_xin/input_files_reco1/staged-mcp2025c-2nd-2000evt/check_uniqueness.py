#!/usr/bin/env python3
"""doc pr/82 sec 2.1 -- rebuild the entry->(run,subrun,event) map and gate on it.

WHY THIS IS A SEPARATE STEP.  Imaging writes to work/evt<ID>, keyed on the BARE
event number, which is unique only within a (run, subrun).  If two staged
entries share an event ID -- within this sample, or against the first 1000 --
they silently clobber each other and every downstream product is a mixture of
two events with no error anywhere.  The first-1k dir was collision-free by luck,
not by construction (its PROVENANCE.txt says so), and this sample is riskier:
it spans the SAME two runs, so a collision can be cross-sample too.

The gate deliberately runs AFTER staging.  The authoritative run/subrun/event
comes from the staged per-event metadata -- the same JSON member
run_pr_chain_batch.sh:1155 reads -- so the ~4 min of staging is spent before the
assertions can fire.  That is the first-1k precedent and it is cheap; imaging,
the expensive stage, is what this actually protects.  A FileIndex-derived
forecast is NOT an acceptable substitute: one was tried during planning and got
the run number wrong (said 18255 for part1 entry 0; the staged metadata says
18259) while getting the event ID right, which is the worst possible failure
mode for a uniqueness argument.

THE MAP FORMAT IS LOAD-BEARING.  run_full1k_nusel.sh:66 is
    awk -F'\t' -v e="$1" '$1==e {print $4; exit}'
-- tab-separated, entry in column 1, event in column 4, with a header line.  A
map in any other shape makes evt_of() return empty, which is NOT an error: the
worker writes `rc=90 no-event-map-row` and exits 0, so a 2000/2000 silent skip
reports as a completed batch.  (The snippet in doc pr/82 sec 2.1 got this wrong
twice -- it emitted space-separated 4-column rows with no entry index, and its
'*_metadata.json' wildcard matched TWO archive members, so it crashed on
`json.decoder.JSONDecodeError: Extra data` before it could even do that.  Both
are fixed here.)

Repro:
  python3 input_files_reco1/staged-mcp2025c-2nd-2000evt/check_uniqueness.py --build
  python3 input_files_reco1/staged-mcp2025c-2nd-2000evt/check_uniqueness.py
"""
import argparse
import io
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST1K = os.path.join(os.path.dirname(HERE), "staged-mcp2025c-1000evt",
                       "entry_event_map.tsv")
MAP = os.path.join(HERE, "entry_event_map.tsv")


def read_meta(entry):
    """(run, subrun, event, caf_ns) from a staged entry's opflash metadata.

    The wildcard is `opflash_tensorset_*_metadata.json` and not `*_metadata.json`
    on purpose: the archive also holds `opflash_tensor_<EVT>_0_metadata.json`
    (content `{"name":"opflash"}`), and matching both concatenates two JSON
    documents onto one stream.
    """
    p = os.path.join(HERE, "e%d" % entry, "opflash_apa0.tar.gz")
    if not os.path.isfile(p):
        return None
    with tarfile.open(p, "r:gz") as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if base.startswith("opflash_tensorset_") and base.endswith("_metadata.json"):
                d = json.load(io.TextIOWrapper(tf.extractfile(m)))
                return (int(d["run"]), int(d["subrun"]), int(d["event"]),
                        int(d["frame_apply_at_caf"]))
    return None


def build(n):
    rows, missing = [], []
    for i in range(n):
        m = read_meta(i)
        if m is None:
            missing.append(i)
            continue
        rows.append((i,) + m)
    with open(MAP, "w") as fh:
        fh.write("entry\trun\tsubrun\tevent\tcaf_ns\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print("wrote %s: %d rows" % (MAP, len(rows)))
    if missing:
        print("!! %d entries had no staged metadata: %s"
              % (len(missing), missing[:20]))
    return len(missing) == 0


def load(path):
    out = []
    with open(path) as fh:
        head = fh.readline()
        assert head.rstrip("\n").split("\t")[:5] == \
            ["entry", "run", "subrun", "event", "caf_ns"], \
            "unexpected header in %s: %r" % (path, head)
        for line in fh:
            if not line.strip():
                continue
            e, r, s, ev, caf = line.rstrip("\n").split("\t")[:5]
            out.append((int(e), int(r), int(s), int(ev), int(caf)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)build the map first")
    ap.add_argument("-n", type=int, default=2000)
    args = ap.parse_args()

    ok = True
    if args.build:
        ok = build(args.n) and ok

    rows = load(MAP)
    print("map rows: %d (expected %d)" % (len(rows), args.n))
    if len(rows) != args.n:
        print("FAIL: incomplete map"); ok = False

    # 1 -- distinct (run, subrun, event)
    rse = set((r, s, e) for _, r, s, e, _ in rows)
    print("A1 distinct (run,subrun,event): %d / %d  %s"
          % (len(rse), len(rows), "OK" if len(rse) == len(rows) else "FAIL"))
    ok = ok and len(rse) == len(rows)

    # 2 -- distinct BARE event ids: the condition work/evt<ID> actually needs
    ev = [e for _, _, _, e, _ in rows]
    dup = sorted({x for x in ev if ev.count(x) > 1}) if len(set(ev)) != len(ev) else []
    print("A2 distinct bare event ids : %d / %d  %s%s"
          % (len(set(ev)), len(rows), "OK" if not dup else "FAIL",
             "" if not dup else "  dups=%s" % dup[:20]))
    ok = ok and not dup

    # 3 -- zero intersection with the first 1000
    first = load(FIRST1K)
    fev = set(e for _, _, _, e, _ in first)
    clash = sorted(set(ev) & fev)
    print("A3 overlap with first 1k   : %d  %s%s"
          % (len(clash), "OK" if not clash else "FAIL",
             "" if not clash else "  %s" % clash[:20]))
    ok = ok and not clash

    # 4 -- caf values are not the `-caf auto` fallback signature.  All values
    #      congruent to 0 mod 256 is how yuhw's bad ncpi0 extraction was caught
    #      (doc 71 sec 3); the first 1k spans 245..3322 ns.
    caf = [c for *_, c in rows]
    allmod = all(c % 256 == 0 for c in caf)
    print("A4 caf_ns range %d..%d, all==0 mod 256: %s  %s"
          % (min(caf), max(caf), allmod, "FAIL" if allmod else "OK"))
    ok = ok and not allmod

    runs = {}
    for _, r, s, _, _ in rows:
        runs[(r, s)] = runs.get((r, s), 0) + 1
    print("run/subrun census: %s" % sorted(runs.items()))

    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
