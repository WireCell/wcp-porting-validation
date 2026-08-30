#!/usr/bin/env python3
"""doc pr/130 item C -- can anything at the pass4_angle seat separate the
scanner-condemned admissions from the legitimate ones?

Item C is the remainder of the affirmative q_extra pool after items 1b/B:
the ~14 condemned segments `pass4_angle` placed, none of which any guard had
declined (so the Part 4/5 "overruled guard" fix cannot reach them).  Its two
signatures from pr130-qextra-refresh.md Part 1 are 286655 (four segments
admitted at 137-150 deg) and 278420 (seven at 98-125 cm).

This censuses EVERY pass4_angle admission over both manifests at the current
production point and asks whether the condemned ones separate on any taped
feature.  Same method and the same bar as Part 4's ten-feature hunt: a
candidate separator must not interleave with the legitimate population.

Repro:
  ./scripts/pr130_arms.sh 98 vtxcen 1 ; ./scripts/pr130_arms.sh 141 vtxcen141 1
  ./scripts/pr130_pass4angle_census.py
"""
import collections
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
ARMS = ["work-pr130r1-vtxcen-*", "work-pr130r1-vtxcen141-*"]

# Scanner-condemned segments whose absorber the label store records as
# pass4_angle (pr130-qextra-attrib.txt).  event -> {seg ids}
CONDEMNED = {
    278420: {67033, 68034, 69035, 70036, 73039, 75041, 78044},
    286655: {67016, 69018, 81062, 82063},
    72786:  {9009, 31033},
    400504: {21003},
}

NUM = re.compile(r"(\w+)=(-?[\d.]+)")
FEATS = ("len_cm", "med_dqdx_mip", "pair_dis_cm", "front_dis_cm",
         "body_dis_cm", "snap_dis_cm", "angle_v1", "angle_v2", "tier",
         "cur_len_cm", "cur_nseg", "divert")


def rows():
    for pat in ARMS:
        for arm in sorted(glob.glob(os.path.join(SX, pat))):
            for lg in sorted(glob.glob(os.path.join(arm, "pr_evt*", "stdout.log"))):
                ev = int(re.search(r"pr_evt(\d+)", lg).group(1))
                admitted = set()
                geom = {}
                for ln in open(lg, errors="replace"):
                    if "SHOWER_ABSORB PASS4_GEOM" in ln:
                        d = dict(NUM.findall(ln))
                        geom[int(d["seg"])] = {k: float(d[k]) for k in FEATS if k in d}
                    elif "SHOWER_ABSORB DIRECT" in ln and "site=pass4_angle" in ln:
                        m = re.search(r"\bseg=(\d+)", ln)
                        if m:
                            admitted.add(int(m.group(1)))
                for seg in admitted:
                    if seg in geom:
                        r = dict(geom[seg]); r["event"] = ev; r["seg"] = seg
                        r["bad"] = seg in CONDEMNED.get(ev, ())
                        yield r


def sep(name, bad, good):
    """Report whether `name` separates the condemned from the legitimate."""
    if not bad or not good:
        return "  %-14s (insufficient data)" % name
    blo, bhi, glo, ghi = min(bad), max(bad), min(good), max(good)
    if blo > ghi:
        return "  %-14s SEPARATES: condemned [%.2f, %.2f] ALL ABOVE legit [%.2f, %.2f]  margin %.2f" % (
            name, blo, bhi, glo, ghi, blo - ghi)
    if bhi < glo:
        return "  %-14s SEPARATES: condemned [%.2f, %.2f] ALL BELOW legit [%.2f, %.2f]  margin %.2f" % (
            name, blo, bhi, glo, ghi, glo - bhi)
    ov = sum(1 for g in good if blo <= g <= bhi)
    return "  %-14s interleaved: condemned [%.2f, %.2f] vs legit [%.2f, %.2f]  (%d/%d legit inside)" % (
        name, blo, bhi, glo, ghi, ov, len(good))


def main():
    allr = list(rows())
    if not allr:
        sys.exit("no PASS4_GEOM rows -- run the census arms first (PROBES=1)")
    bad = [r for r in allr if r["bad"]]
    good = [r for r in allr if not r["bad"]]
    print("pass4_angle admissions at the current production point:")
    print("  %d total over %d events;  %d scanner-condemned, %d not"
          % (len(allr), len({r['event'] for r in allr}), len(bad), len(good)))
    missing = {(e, s) for e, ss in CONDEMNED.items() for s in ss} - {(r["event"], r["seg"]) for r in bad}
    if missing:
        print("  NOTE %d condemned segment(s) are no longer admitted here "
              "(earlier round took them): %s"
              % (len(missing), ", ".join("%d/%d" % m for m in sorted(missing))))
    print("\n=== the condemned admissions ===")
    print("  %8s %8s %8s %8s %8s %8s %8s %6s" % ("event", "seg", "len_cm", "angle_v1", "angle_v2", "pair_dis", "body_dis", "tier"))
    for r in sorted(bad, key=lambda r: (r["event"], r["seg"])):
        print("  %8d %8d %8.1f %8.1f %8.1f %8.1f %8.1f %6.0f"
              % (r["event"], r["seg"], r.get("len_cm", -1), r.get("angle_v1", -1),
                 r.get("angle_v2", -1), r.get("pair_dis_cm", -1),
                 r.get("body_dis_cm", -1), r.get("tier", -1)))
    print("\n=== does any taped feature separate them? (same bar as Part 4) ===")
    for f in FEATS:
        b = [r[f] for r in bad if f in r]
        g = [r[f] for r in good if f in r]
        print(sep(f, b, g))
    print("\n=== tier occupancy ===")
    for lbl, grp in (("condemned", bad), ("legit", good)):
        c = collections.Counter(int(r.get("tier", -1)) for r in grp)
        print("  %-10s %s" % (lbl, dict(sorted(c.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
