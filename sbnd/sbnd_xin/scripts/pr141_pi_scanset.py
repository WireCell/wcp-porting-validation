#!/usr/bin/env python3
"""doc pr/141 rec 2 -- the pi-typed (|pdg| = 211) split scan set, on the arm the
split viewer actually reads.

READ-ONLY.  doc pr/139 sec 22.5 left the EM-only splitter restriction at n = 2:
the splitter fires on non-EM-typed candidates with purity 0.000 (1 fire, 0
confirmed cuts), so restricting it to |pdg| = 11 looks free -- but two negatives
cannot set a rule.  doc pr/141 sec 6.2 rebuilt that population on the current
production arm and found 8 objects; this script rebuilds it where
`split_display` can show it.

WHY IT IS RE-SELECTED RATHER THAN REUSED: `split_model.load_object` reads the
doc pr/136 `onV1c90` arm (pr137_lib.dump globs work-pr136-<arm>-*), and shower
node ids DRIFT between arms -- 388 is node 23028 on work-pr140r2-off and 23032
on work-pr136-onV1c90.  Handing the viewer pr140r2 ids would silently show the
wrong objects, so the selection is redone inside `build_population`, which is
the same population every split-scan of this campaign was drawn from.

Output columns are the six `split_viewer.worklist()` reads.

    python3 scripts/pr141_pi_scanset.py --tsv docs/pr/pr141-pi-scanset.tsv
"""
import argparse, os, sys, glob

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "scripts"))
os.chdir(SX)
import pr137_lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="docs/pr/pr141-pi-scanset.tsv")
    ap.add_argument("--pdg", type=int, default=211)
    ap.add_argument("--min-mev", type=float, default=100.0)
    args = ap.parse_args()

    events = []
    for s in ("mcp1k", "mcp2k", "ncpi0", "nuecc48"):
        p = "/home/xqian/tmp/pr139-manifest-%s.lst" % s
        if os.path.exists(p):
            events += [int(x) for x in open(p).read().split()]
    events = sorted(set(events))
    print("events to scan: %d" % len(events))

    rows = []
    for i, ev in enumerate(events):
        try:
            pop = L.build_population(ev)
        except Exception:
            continue
        for r in pop:
            sr = r.get("rec") or {}
            pdg = abs(int(sr.get("particle_id") or 0))
            E = sr.get("kine_charge") or 0.0
            if pdg != args.pdg:
                continue
            if E < args.min_mev:
                continue
            rows.append(dict(event=ev, node=r["node"], stratum="S-PI",
                             owner_scan="1", proxy_cls=r.get("cls", ""),
                             Q="%.4g" % r.get("Q", 0.0),
                             E="%.1f" % E,
                             nseg=sr.get("num_segments"),
                             length="%.1f" % (sr.get("total_length") or 0.0),
                             conn=sr.get("start_connection_type")))
        if (i + 1) % 60 == 0:
            print("  ... %d/%d events, %d objects so far" % (i + 1, len(events), len(rows)))

    rows.sort(key=lambda r: -float(r["E"]))
    hdr = ["event", "node", "stratum", "owner_scan", "proxy_cls", "Q",
           "E", "nseg", "length", "conn"]
    with open(args.tsv, "w") as fh:
        fh.write("# doc pr/141 rec 2 -- pi-typed (|pdg|=%d) split candidates,\n"
                 "# >= %.0f MeV, selected inside build_population on the pr/136\n"
                 "# onV1c90 arm so split_display's node ids are correct.\n"
                 % (args.pdg, args.min_mev))
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[h]) for h in hdr) + "\n")
    print("\n%d objects -> %s" % (len(rows), args.tsv))
    for r in rows:
        print("  evt %-7d node %-7d E=%8s MeV  len=%7s cm  nseg=%-3s conn=%s  proxy=%s"
              % (r["event"], r["node"], r["E"], r["length"], r["nseg"],
                 r["conn"], r["proxy_cls"]))


if __name__ == "__main__":
    main()
