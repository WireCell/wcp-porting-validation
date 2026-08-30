#!/usr/bin/env python3
"""pr/130 item 4 -- the F12 walk-add track guard's decline population.

pr130_qmiss_census.py's DECLINED class says a shipped guard refused charge the
scanner wants.  That is only actionable if the wanted subset can be SEPARATED
from everything else the same guard declines, so this script measures the
denominator: every `SHOWER_ABSORB EXCLUDE` on both manifests, with the four
scanner-wanted ones (plus the one the owner has already ruled a correct
decline) flagged.

The guard is `guard_excludes` in PRShower.cxx:827-836 -- F12, doc pr/40 round
6, `absorb_track_guard`.  It is the ANCESTOR of the pr/123->pr/130 seat family,
not one of its additions: it fires inside the shower flood-fill, on
`segment_is_straight_long_track`, for any non-electron PID (electrons only when
`em_straight_min_len > 0`, which only examine_shower_1 passes).

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  scripts/pr130_qmiss_f12.py > docs/pr/pr130-qmiss-f12.txt

READ-ONLY over em_labels/, the calib dumps and emprep-pr130q{98,141}/ (M13).
"""
import collections
import os
import sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))
import em117_score as S                                    # noqa: E402

SETS = [
    ("98", "emscan-0827", "em117-pr130q98-manifest.tsv", "emprep-pr130q98"),
    ("141", "emscan-0828-agent5", "em114c-pr130q141-manifest.tsv", "emprep-pr130q141"),
]

# (event, segment) pairs the hand scan marks IN for a shower that did not get
# them -- i.e. the DECLINED rows of pr130-qmiss-census.tsv.  318769 is the one
# the OWNER ruled a correct decline (pr/129 reject), so it is the control.
WANTED = {(169626, 53070), (54341, 96033), (54341, 34015), (395597, 20006)}
OWNER_REJECTED = {(318769, 19005)}


def collect():
    rows = []
    cwd = os.getcwd()
    os.chdir(os.path.join(SX, "em_display"))
    try:
        for label, tag, manifest, prepdir in SETS:
            man = S.load_manifest(manifest)
            labs = S.load_labels(tag)
            for ev in sorted(labs):
                m = man.get(ev)
                if not m or not m.get("dump"):
                    continue
                dump = S.load_dump(m["dump"])
                prep = S.load_prep(ev, prepdir)
                if dump is None or prep is None:
                    continue
                _, seginfo, _ = S.digest_dump(dump, prep)
                for sid, recs in (prep.get("absorb") or {}).items():
                    for r in recs:
                        if r.get("how") != "walk_exclude":
                            continue
                        i = int(sid)
                        info = seginfo.get(i, {})
                        mark = ("WANT" if (ev, i) in WANTED else
                                "OWNER-NO" if (ev, i) in OWNER_REJECTED else "")
                        rows.append(dict(set=label, event=ev, seg=i, mark=mark,
                                         q=info.get("charge", 0.0),
                                         length=info.get("length", 0.0),
                                         pdg=r.get("pdg"), site=r.get("site")))
    finally:
        os.chdir(cwd)
    return rows


def main():
    rows = collect()
    print("=" * 78)
    print("pr/130 item 4 -- F12 walk-add guard (PRShower.cxx:827) decline population")
    print("=" * 78)
    for label, _, _, _ in SETS:
        rr = [r for r in rows if r["set"] == label]
        # a segment can be excluded at more than one site (409888/13001 is
        # taped twice); the denominator must be DISTINCT segments or the
        # declined charge is double-counted.
        dist = {}
        for r in rr:
            dist[(r["event"], r["seg"])] = r
        q = sum(x["q"] for x in dist.values())
        w = [x for x in dist.values() if x["mark"]]
        qw = sum(x["q"] for x in w)
        print("\n--- set %s ---" % label)
        print("  exclude records     %3d" % len(rr))
        print("  distinct segments   %3d              charge %.4e" % (len(dist), q))
        print("  scanner wants       %3d of them      charge %.4e  = %.1f%%"
              % (len(w), qw, 100.0 * qw / q if q else 0.0))
        print("  pdg  :", dict(collections.Counter(r["pdg"] for r in rr)))
        print("  site :", dict(collections.Counter(r["site"] for r in rr)))

    print("\n" + "=" * 78)
    print("SEPARABILITY -- every decline under 30 cm, ordered by length")
    print("A threshold exists only if the WANT rows sit at one end.")
    print("=" * 78)
    print("%-8s %-4s %-8s %-8s %10s %6s %5s %s"
          % ("mark", "set", "event", "seg", "q", "len", "pdg", "site"))
    for r in sorted([x for x in rows if x["length"] < 30], key=lambda x: x["length"]):
        print("%-8s %-4s %-8d %-8d %10.3e %6.1f %5s %s"
              % (r["mark"], r["set"], r["event"], r["seg"], r["q"],
                 r["length"], r["pdg"], r["site"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
