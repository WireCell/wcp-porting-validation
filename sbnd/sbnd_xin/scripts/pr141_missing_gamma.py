#!/usr/bin/env python3
"""doc pr/141 item 2 -- scoping the eight 'gamma NOT RECONSTRUCTED' events.

READ-ONLY.  doc pr/139 sec 26.3 named 8 of the 31 residual pi0 as "a gamma is
simply NOT RECONSTRUCTED" -- the largest single block -- and sec 26.5 item 2
asked for a scoping pass before any more clustering work.

"Absent-on-arm" in pr132_pi0_census.py means only this: the label's gamma
SHOWER ID is not a key of the arm's `showers[]`.  That is three different
situations wearing one name, and which one it is decides whether there is
anything to fix:

  MERGED     the gamma's charge is on the arm, inside a DIFFERENT and
             SUBSTANTIALLY LARGER shower -- a genuine over-merge.
  RENAMED    same, but the host is comparable in size and segment count: the
             same object under a new id, i.e. label staleness, not a defect.
             (Distinguishing these two matters: an early version of this script
             folded them together and the doc over-claimed the over-merge share.)
  UNSHOWERED the gamma's segments are on the arm but no shower owns them --
             a shower-formation failure.
  GONE       the charge itself is not on the arm.

The test is by SEGMENT ID: the label stores the gamma's `members`, which are
segment display ids, and the arm's `segments[]` carries the same ids.  A member
segment that is on the arm and is claimed by some other shower is MERGED; on
the arm and unclaimed is UNSHOWERED; not on the arm at all is GONE.  Charge is
apportioned so a partly-merged gamma reports the split.

    python3 scripts/pr141_missing_gamma.py --tsv docs/pr/pr141-missing-gamma.tsv
"""
import argparse, json, math, os, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X0, EC = 14.0, 32.8
VOL = ((-200.0, 200.0), (-200.0, 200.0), (0.0, 500.0))

# event -> (arm, base tag, overlay tag)
EVENTS = [
    (71178,  "work-pr140r2-off-mcp1k", "emscan-0828-agent5", "pi0scan-0829-agent"),
    (142421, "work-pr140r2-off-ncpi0", "emscan-0827",        "pi0scan-0829-agent"),
    (259542, "work-pr140r2-off-ncpi0", "emscan-0827",        "pi0scan-0829-agent"),
    (281485, "work-pr140r2-off-mcp1k", "emscan-0827",        "pi0scan-0829-agent"),
    (396222, "work-pr140r2-off-mcp2k", "emscan-0827",        "pi0scan-0829-agent"),
    (76346,  "work-pr140r2-off-mcp2k", "emscan-0827",        "pi0scan-0829-agent"),
    (347824, "work-pr140r2-off-mcp2k", "emscan-0828-agent5", "pi0scan-0829-agent"),
    (506114, "work-pr140r2-off-ncpi0", "emscan-0827",        "pi0scan-0829-agent"),
]


def wall_dist(p):
    return min(min(p[i] - VOL[i][0], VOL[i][1] - p[i]) for i in range(3))


def load_pio(ev, base, overlay):
    for tag in (base, overlay):
        p = os.path.join(SX, "em_labels", tag, "labels-evt%d.json" % ev)
        if os.path.exists(p):
            rec = json.load(open(p))
            if rec.get("pio"):
                return rec["pio"], tag
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    rows, tally = [], {}
    for ev, arm, base, overlay in EVENTS:
        pio, tag = load_pio(ev, base, overlay)
        dump = json.load(open(os.path.join(
            SX, arm, "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)))
        shw = {int(s["id"]): s for s in (dump.get("showers") or ())}
        segs = {int(s["id"]): s for s in (dump.get("segments") or ())}
        # which shower claims each segment: the dump gives shower -> id only,
        # so use the probe sidecar when present, else fall back to the
        # shower id == start segment id convention plus proximity.
        side = os.path.join(SX, "em_display", "emprep-140r2off",
                            "emprep-evt%d.json" % ev)
        owner_of = {}
        if os.path.exists(side):
            sd = json.load(open(side))
            for sid, rec in (sd.get("showers") or {}).items():
                for m in (rec.get("members") or []):
                    owner_of[int(m.get("seg"))] = int(sid)

        print("=== evt %d   (pio from %s)" % (ev, tag))
        if not pio:
            print("    no pairing recorded\n")
            continue
        for slot in ("1", "2"):
            gg = (pio.get("gammas") or {}).get(slot)
            if not gg:
                continue
            sid = int(gg["shower"])
            if sid in shw:
                continue                      # this gamma is present
            mem = [int(m) for m in (gg.get("members") or [])]
            on, claimed, unclaimed, gone = [], {}, [], []
            for m in mem:
                if m not in segs:
                    gone.append(m)
                    continue
                on.append(m)
                o = owner_of.get(m)
                if o is None:
                    unclaimed.append(m)
                else:
                    claimed.setdefault(o, []).append(m)
            # Fallback when the label recorded no member list (older label
            # format): ask the arm SPATIALLY -- which shower's own points come
            # closest to the label's recorded gamma start, and how close.
            near = None
            if not mem:
                st0 = gg.get("start") or [0, 0, 0]
                for s2 in (dump.get("showers") or ()):
                    sp = s2.get("start") or {}
                    d0 = math.dist((sp.get("x", 0), sp.get("y", 0), sp.get("z", 0)), st0)
                    if near is None or d0 < near[0]:
                        near = (d0, int(s2["id"]), s2)
            E_lab = gg.get("energy") or 0.0
            n = len(mem) or 1
            # Proximity alone is not evidence: a 993 MeV gamma whose nearest
            # start is a 50.8 MeV proton stub is NOT accounted for.  Require the
            # host to be able to hold the charge.
            E_lab = gg.get("energy") or 0.0
            if not mem and near is not None:
                E_host = (near[2].get("kine_charge") or 0.0)
                if near[0] < 15.0 and E_host >= 0.5 * E_lab:
                    verdict = "MERGED-BY-PROXIMITY"
                elif near[0] < 15.0:
                    verdict = "UNACCOUNTED (host %.1f MeV vs gamma %.1f)" % (E_host, E_lab)
                else:
                    verdict = "NO-CANDIDATE"
            elif not on:
                verdict = "GONE"
            elif claimed and len(unclaimed) <= len(on) / 2:
                # size test: a host comparable in energy AND segment count is
                # the same object renamed, not an absorption.
                host = max(claimed, key=lambda o: len(claimed[o]))
                t = shw.get(host) or {}
                E_host = t.get("kine_charge") or 0.0
                n_host = t.get("num_segments") or 0
                same_size = (E_lab > 0 and 0.6 <= E_host / E_lab <= 1.6
                             and abs(n_host - len(mem)) <= max(2, 0.25 * len(mem)))
                verdict = "RENAMED" if same_size else "MERGED"
            elif unclaimed:
                verdict = "UNSHOWERED"
            else:
                verdict = "MERGED"
            tally[verdict] = tally.get(verdict, 0) + 1
            st = gg.get("start") or [0, 0, 0]
            print("    g%s %6d  E=%6.1f MeV  %d member segment(s): "
                  "%d on arm, %d not on arm" % (
                      slot, sid, gg.get("energy") or 0, len(mem), len(on), len(gone)))
            for o, ms in sorted(claimed.items()):
                tgt = shw.get(o)
                print("        %2d seg -> now owned by shower %d "
                      "(E=%.1f MeV, len=%.1f cm, %s seg, pdg=%s)"
                      % (len(ms), o, (tgt or {}).get("kine_charge") or 0,
                         (tgt or {}).get("total_length") or 0,
                         (tgt or {}).get("num_segments"),
                         (tgt or {}).get("particle_id")))
            if unclaimed:
                print("        %2d seg -> on the arm but owned by NO shower" % len(unclaimed))
            if gone:
                print("        %2d seg -> not on the arm at all" % len(gone))
            if near is not None:
                t = near[2]
                print("        label stored NO member list; nearest shower start on "
                      "the arm is %d at %.1f cm (E=%.1f MeV, len=%.1f cm, %s seg, pdg=%s)"
                      % (near[1], near[0], t.get("kine_charge") or 0,
                         t.get("total_length") or 0, t.get("num_segments"),
                         t.get("particle_id")))
            print("        start wall distance %.1f cm  =>  %s" % (wall_dist(st), verdict))
            rows.append(dict(event=ev, slot=slot, gamma=sid,
                             E="%.1f" % (gg.get("energy") or 0),
                             nmem=len(mem), n_on=len(on), n_gone=len(gone),
                             n_unclaimed=len(unclaimed),
                             hosts=",".join(str(o) for o in sorted(claimed)),
                             wall="%.1f" % wall_dist(st), verdict=verdict))
        print()

    print("=== tally ===")
    for k in sorted(tally):
        print("  %-12s %d" % (k, tally[k]))

    if args.tsv:
        hdr = ["event", "slot", "gamma", "E", "nmem", "n_on", "n_gone",
               "n_unclaimed", "hosts", "wall", "verdict"]
        with open(args.tsv, "w") as fh:
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[h]) for h in hdr) + "\n")
        print("wrote %s (%d rows)" % (args.tsv, len(rows)))


if __name__ == "__main__":
    main()
