#!/usr/bin/env python3
"""doc pr/136 round 2 -- attribute every firing of shower_pass4_prefilter_v1_escape.

The knob lets a pass-4 cross-cluster candidate past the `angle_v2 > 30`
pre-filter when it still satisfies an angle_v1 clause of the acceptance
disjunction (NeutrinoShowerClustering.cxx:2433).  Firing is NOT admission: the
escape only buys the candidate a full evaluation, and the disjunction can still
refuse it (that is the point -- the escape restores the algorithm's own test
rather than widening it).  This script separates the two, because "the knob
fired 400 times" and "the knob moved 400 segments" are very different facts.

  FIRED      a `SHOWER_XCLUS ESCAPE` line exists for the (shower, segment) pair
  ADMITTED   ... and that segment is a member of that shower in the ON arm
  REFUSED    ... and it is not (the full disjunction, or the associated-vertex
             guard at :2455, declined it after all)

It then cross-checks the ON arm against the OFF arm's sidecar so the REAL
membership delta -- including the chain effect, where admitting a segment to
shower A changes what shower B sees later in the same pass -- is reported next
to the direct one.  doc pr/136 sec 10.6 predicted the delta would exceed the 10
seed segments; this is where that is checked rather than assumed.

READ-ONLY.

    scripts/pr136_escape_census.py --on-arm 'work-pr136-onV1-*' \\
        --on-prep emprep-136onV1 --off-prep emprep-136f086
"""
import argparse, collections, csv, glob, json, os, re

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)
ED = os.path.join(SX, "em_display")

RE_ESC = re.compile(r"SHOWER_XCLUS ESCAPE site=pass4_prefilter_v1 shower=(-?\d+) seg=(-?\d+) "
                    r"angle_v1=([-\d.]+) angle_v2=([-\d.]+) pair_dis_cm=([-\d.]+)")
# the 10 segments doc pr/136 sec 10.3 predicted from the OFF tape
SEED = {(142421, 7010), (314838, 13010), (52044, 24009), (84229, 9058),
        (105946, 53029), (409634, 69032), (105946, 53030), (409634, 69033),
        (105946, 54032), (54341, 77019)}


def prep(d, ev):
    p = os.path.join(d, "emprep-evt%d.json" % ev)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def members(pr):
    """node -> {seg: dQ}"""
    out = {}
    for node, e in ((pr or {}).get("showers") or {}).items():
        out[int(node)] = {int(m["seg"]): float(m.get("dQ") or 0.0)
                          for m in (e.get("members") or ())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-arm", default="work-pr136-onV1-*")
    ap.add_argument("--on-prep", default="emprep-136onV1")
    ap.add_argument("--off-prep", default="emprep-136f086")
    ap.add_argument("--tsv", default="docs/pr/pr136-escape-census.tsv")
    a = ap.parse_args()

    onp = a.on_prep if os.path.isabs(a.on_prep) else os.path.join(ED, a.on_prep)
    offp = a.off_prep if os.path.isabs(a.off_prep) else os.path.join(ED, a.off_prep)

    rows = []
    ev_fired = set()
    for log in sorted(glob.glob(os.path.join(SX, a.on_arm, "pr_evt*", "stdout.log"))):
        ev = int(re.search(r"pr_evt(\d+)", log).group(1))
        hits = []
        with open(log, errors="replace") as fh:
            for ln in fh:
                if "ESCAPE site=pass4_prefilter_v1" not in ln:
                    continue
                m = RE_ESC.search(ln)
                if m:
                    hits.append((int(m.group(1)), int(m.group(2)),
                                 float(m.group(3)), float(m.group(4)), float(m.group(5))))
        if not hits:
            continue
        ev_fired.add(ev)
        mon = members(prep(onp, ev))
        # a segment can end up in a shower other than the escaping one
        owner_on = {s: n for n, ms in mon.items() for s in ms}
        for (shw, seg, a1, a2, pd) in hits:
            held = seg in mon.get(shw, {})
            rows.append(dict(event=ev, shower=shw, seg=seg,
                             angle_v1=a1, angle_v2=a2, pair_dis_cm=pd,
                             q=round(mon.get(shw, {}).get(seg, 0.0), 1),
                             verdict="ADMITTED" if held else "REFUSED",
                             final_owner=owner_on.get(seg, -1),
                             seed=int((ev, seg) in SEED)))

    print("shower_pass4_prefilter_v1_escape -- FIRINGS AND ADMISSIONS  (doc pr/136 r2)")
    print("  arm %s;  ON sidecar %s" % (a.on_arm, a.on_prep))
    print("  escape fired %d times over %d events" % (len(rows), len(ev_fired)))
    c = collections.Counter(r["verdict"] for r in rows)
    qa = sum(r["q"] for r in rows if r["verdict"] == "ADMITTED")
    print("  ADMITTED %d (q=%.3e)   REFUSED %d" % (c["ADMITTED"], qa, c["REFUSED"]))
    seen_seed = {(r["event"], r["seg"]) for r in rows if r["seed"]}
    adm_seed = {(r["event"], r["seg"]) for r in rows
                if r["seed"] and r["verdict"] == "ADMITTED"}
    print("  of the 10 sec 10.3 seed segments: %d fired, %d admitted"
          % (len(seen_seed), len(adm_seed)))
    miss = SEED - seen_seed
    if miss:
        print("  seed segments that did NOT fire: %s"
              % ", ".join("%d/%d" % t for t in sorted(miss)))

    # ---- the real membership delta, OFF vs ON, over every event ------------
    print("\nMEMBERSHIP DELTA OFF -> ON over every event with both sidecars")
    d_ev = d_seg = 0
    dq = 0.0
    moved = []
    for f in sorted(glob.glob(os.path.join(offp, "emprep-evt*.json"))):
        ev = int(re.search(r"emprep-evt(\d+)", f).group(1))
        po, pn = prep(offp, ev), prep(onp, ev)
        if not po or not pn:
            continue
        mo, mn = members(po), members(pn)
        oo = {s: n for n, ms in mo.items() for s in ms}
        on = {s: n for n, ms in mn.items() for s in ms}
        qs = {s: q for ms in mo.values() for s, q in ms.items()}
        qs.update({s: q for ms in mn.values() for s, q in ms.items()})
        diff = {s for s in set(oo) | set(on) if oo.get(s, -1) != on.get(s, -1)}
        if diff:
            d_ev += 1
            d_seg += len(diff)
            dq += sum(qs.get(s, 0.0) for s in diff)
            moved.append((ev, len(diff), sum(qs.get(s, 0.0) for s in diff)))
    print("  %d events changed, %d segments re-owned, q=%.3e" % (d_ev, d_seg, dq))
    print("  (the escape fired in %d events -- any excess is the chain effect)"
          % len(ev_fired))
    if moved:
        print("  top movers by charge:")
        for ev, n, q in sorted(moved, key=lambda x: -x[2])[:12]:
            print("    evt %-8d %3d segment(s)  q=%.3e  %s"
                  % (ev, n, q, "escape fired" if ev in ev_fired else "CHAIN ONLY"))

    if rows:
        o = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(o, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (o, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
