#!/usr/bin/env python3
"""doc pr/132 round 10 -- the pair opening-angle census (Front A).

Borrows the pr126 selector like pr132_gamma_ledger.py.  Joins the
WCT_PI0_ANGLE_DEBUG tape (PI0_ANGLE lines in <arm>/pr_evt<ID>/stdout.log)
with the hand pi0 labels: for every hand pair whose BOTH gamma shower ids
appear in one recorded path-1 pairing, report the legacy start-ray mass and
angle next to the centroid-ray variant, and the label opening angle (angle
between the two per-gamma label axes).  Classifies:

  BELOW->IN     m_start below the (100,160)+offset window, m_cent inside
  BELOW->BELOW  below on both rays
  IN->OUT       m_start inside, centroid ray pushes it OUT (the damage class)
  IN->IN        inside on both
  other         everything else

The tape is read from the arms named --arm-prefix (e.g. work-pr132-r10ang).
"""
import argparse, csv, math, os, re, sys, importlib.util
from collections import Counter

_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(os.path.dirname(__file__), "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFSET = 10.0
def inwin(m):
    d = m - 135.0 + OFFSET
    return -25.0 < d < 35.0

def ang_between(a, b):
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    if na <= 0 or nb <= 0: return None
    c = max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b)) / (na * nb)))
    return math.degrees(math.acos(c))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest141"); ap.add_argument("--manifest98")
    ap.add_argument("--overlay-tag"); ap.add_argument("--tsv")
    ap.add_argument("--arm-prefix", default="work-pr132-r10ang")
    a = ap.parse_args()
    if a.manifest98 or a.manifest141:
        newsets = []
        for t in SEL.SETS:
            t = list(t)
            if t[0] == "98" and a.manifest98: t[4] = a.manifest98
            if t[0] == "141" and a.manifest141: t[4] = a.manifest141
            newsets.append(tuple(t))
        SEL.SETS = newsets
    overlay = SEL.load_labels(a.overlay_tag) if a.overlay_tag else {}

    pat = re.compile(r"PI0_ANGLE vtx=(-?\d+) sh1=(\d+) sh2=(\d+) ct1=(\d+) ct2=(\d+) "
                     r"E1=([\d.\-]+) E2=([\d.\-]+) m_start=([\d.\-]+) a_start=([\d.\-]+) "
                     r"m_cent=([\d.\-]+) a_cent=([\d.\-]+)")
    rows, cls = [], Counter()
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            sample = mrow.get("sample") or mrow.get("det") or ""
            log = os.path.join(SX, f"{a.arm_prefix}-{sample}", f"pr_evt{ev}", "stdout.log")
            tape = {}
            if os.path.exists(log):
                for line in open(log, errors="replace"):
                    m = pat.search(line)
                    if not m: continue
                    s1, s2 = int(m.group(2)), int(m.group(3))
                    rec = dict(vtx=int(m.group(1)), ct1=int(m.group(4)), ct2=int(m.group(5)),
                               E1=float(m.group(6)), E2=float(m.group(7)),
                               m_start=float(m.group(8)), a_start=float(m.group(9)),
                               m_cent=float(m.group(10)), a_cent=float(m.group(11)))
                    # keep the entry whose m_start is closest to the window center
                    key = (min(s1, s2), max(s1, s2))
                    old = tape.get(key)
                    if old is None or abs(rec["m_start"] - 125) < abs(old["m_start"] - 125):
                        tape[key] = rec
            for labsrc, rec in (("base", labels.get(ev)), ("overlay", overlay.get(ev))):
                g = ((rec or {}).get("pio") or {}).get("gammas")
                if not g or not all(x in g and (g[x].get("energy") or 0) > 0 for x in ("1", "2")):
                    continue
                try:
                    i1 = int(g["1"].get("shower") or -1); i2 = int(g["2"].get("shower") or -1)
                except (TypeError, ValueError):
                    continue
                t = tape.get((min(i1, i2), max(i1, i2)))
                ax1, ax2 = g["1"].get("axis"), g["2"].get("axis")
                a_lab = ang_between(ax1, ax2) if (ax1 and ax2) else None
                if t is None:
                    rows.append([setname, sample, ev, labsrc, "no-pair-on-tape",
                                 "", "", "", "", f"{a_lab:.1f}" if a_lab else ""])
                    cls["no-pair-on-tape"] += 1
                    continue
                mi, mc = t["m_start"], t["m_cent"]
                if not inwin(mi) and mi < 110 and mc >= 0 and inwin(mc): k = "BELOW->IN"
                elif not inwin(mi) and mi < 110 and (mc < 0 or not inwin(mc)): k = "BELOW->BELOW/OUT"
                elif inwin(mi) and (mc < 0 or not inwin(mc)): k = "IN->OUT"
                elif inwin(mi) and inwin(mc): k = "IN->IN"
                else: k = "other"
                cls[k] += 1
                rows.append([setname, sample, ev, labsrc, k,
                             f"{mi:.1f}", f"{t['a_start']:.1f}", f"{mc:.1f}", f"{t['a_cent']:.1f}",
                             f"{a_lab:.1f}" if a_lab else ""])
    print("=== hand-pair opening-angle census (start-ray vs centroid-ray) ===")
    for k, v in cls.most_common(): print(f"  {k:18s} {v}")
    print("=== rows ===")
    for r in rows:
        print("  " + "\t".join(str(x) for x in r))
    if a.tsv:
        with open(a.tsv, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["setname", "sample", "event", "labelsrc", "klass",
                        "m_start", "a_start", "m_cent", "a_cent", "a_label"])
            w.writerows(rows)
        print(f"wrote {a.tsv} ({len(rows)} rows)")

if __name__ == "__main__":
    sys.exit(main())
