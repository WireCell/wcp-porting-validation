#!/usr/bin/env python3
"""pr/130 item 4 -- collapse the q_miss fragment pool onto WHOLE reconstructed objects.

pr130_qmiss_census.py classifies each missing SEGMENT.  That is the wrong unit
for a hand scan: the SPLIT root and the STOLEN segments held by it are usually
the same physical object, so a scanner asked to judge 174 fragments is really
being asked the same question a few dozen times.

Grouping the fragment pool by (event, labelled shower, holding reconstructed
object) collapses it, and splits the pool in two:

  WHOLE    the labelled shower's target contains 100% of the object's segments
           -> one binary call: should this object merge into that shower?
  PARTIAL  it wants only part of the object -> a genuine partition question,
           and the harder one.

`q_main` is the part of the wanted charge that is ALREADY in the main cluster
(dump `is_main_cluster`); under the pr/128 105074 precedent that charge is
already the candidate's energy, so it is re-attribution rather than loss.
Ranking is by `q_out = q_wanted - q_main`.  Measured per charge, not
all-or-nothing per object: a reconstructed shower can straddle the main cluster
boundary (314838's object 13010 has its root in main cluster 13 and a member in
cluster 89), and an all-or-nothing flag mislabels those.

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  scripts/pr130_qmiss_objects.py > docs/pr/pr130-qmiss-objects.txt

READ-ONLY (M13).  Reads docs/pr/pr130-qmiss-census.tsv, so run the census first.
"""
import collections
import csv
import os
import sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))
import em117_score as S                                    # noqa: E402

SETS = {"98": ("emscan-0827", "em117-pr130q98-manifest.tsv", "emprep-pr130q98"),
        "141": ("emscan-0828-agent5", "em114c-pr130q141-manifest.tsv", "emprep-pr130q141")}


def main():
    tsv = os.path.join(SX, "docs", "pr", "pr130-qmiss-census.tsv")
    rows = [r for r in csv.DictReader(open(tsv), delimiter="\t")
            if r["adj"] == "False" and r["cls"] in ("SPLIT", "STOLEN", "UNTOUCHED")]
    grp = collections.defaultdict(list)
    for r in rows:
        for h in (r["held_by"].split(",") if r["held_by"] else ["-"]):
            grp[(r["set"], int(r["event"]), int(r["shower"]), h)].append(r)

    os.chdir(os.path.join(SX, "em_display"))
    cache = {}

    def digest(setlab, ev):
        if (setlab, ev) not in cache:
            _, manf, prepdir = SETS[setlab]
            man = S.load_manifest(manf)
            cache[(setlab, ev)] = S.digest_dump(S.load_dump(man[ev]["dump"]),
                                                S.load_prep(ev, prepdir))[:2]
        return cache[(setlab, ev)]

    out = []
    for (sl, ev, shw, h), rr in grp.items():
        q = sum(float(x["q"]) for x in rr)
        q_main = sum(float(x["q"]) for x in rr if x["is_main_cluster"] == "1")
        if h == "-":
            out.append(dict(set=sl, event=ev, shower=shw, obj=-1, nseg=0, want=len(rr),
                            q=q, q_obj=q, frac=0.0, kind="NOHOLDER",
                            q_main=q_main, q_out=q - q_main))
            continue
        actual, seginfo = digest(sl, ev)
        obj = actual.get(int(h), set())
        q_obj = sum(seginfo.get(s, {}).get("charge", 0.0) for s in obj)
        want = {int(x["seg"]) for x in rr} & obj
        frac = (len(want) / len(obj)) if obj else 0.0
        out.append(dict(set=sl, event=ev, shower=shw, obj=int(h), nseg=len(obj),
                        want=len(want), q=q, q_obj=q_obj, frac=frac,
                        kind="WHOLE" if frac >= 0.999 else "PARTIAL",
                        q_main=q_main, q_out=q - q_main))

    tot = sum(r["q"] for r in out)
    print("=" * 86)
    print("pr/130 item 4 -- the q_miss fragment pool as WHOLE OBJECTS")
    print("=" * 86)
    print("%d segments collapse to %d (event, shower, object) triples over %d events\n"
          % (len(rows), len(out), len(set(r["event"] for r in out))))
    for kind in ("WHOLE", "PARTIAL", "NOHOLDER"):
        kk = [r for r in out if r["kind"] == kind]
        q = sum(r["q"] for r in kk)
        qm = sum(r["q_main"] for r in kk)
        print("%-9s %3d object(s)  %.4e  %5.1f%%   (already main-cluster: %.4e = %.1f%%)"
              % (kind, len(kk), q, 100 * q / tot if tot else 0,
                 qm, 100 * qm / q if q else 0))

    print("\n" + "=" * 86)
    print("SCAN ORDER -- outside the main cluster, WHOLE objects first (one binary call each)")
    print("=" * 86)
    print("%-4s %-8s %-8s %-8s %5s %5s %11s %11s %11s %5s %s"
          % ("set", "event", "shower", "object", "nseg", "want", "q_out",
             "q_wanted", "q_object", "frac", "kind"))
    for r in sorted([x for x in out if x["q_out"] > 0],
                    key=lambda x: (x["kind"] != "WHOLE", -x["q_out"])):
        print("%-4s %-8d %-8d %-8d %5d %5d %11.3e %11.3e %11.3e %5.2f %s"
              % (r["set"], r["event"], r["shower"], r["obj"], r["nseg"], r["want"],
                 r["q_out"], r["q"], r["q_obj"], r["frac"], r["kind"]))

    print("\n" + "=" * 86)
    print("BY EVENT -- outside-main-cluster charge, the Bee scan ranking")
    print("=" * 86)
    by = collections.defaultdict(lambda: [0.0, 0, 0])
    for r in out:
        if r["q_out"] <= 0:
            continue
        b = by[(r["set"], r["event"])]
        b[0] += r["q_out"]
        b[1] += 1
        b[2] += 1 if r["kind"] == "WHOLE" else 0
    for (sl, ev), (q, n, nw) in sorted(by.items(), key=lambda x: -x[1][0]):
        print("  set%-4s %-8d %.4e   %d object(s), %d whole" % (sl, ev, q, n, nw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
