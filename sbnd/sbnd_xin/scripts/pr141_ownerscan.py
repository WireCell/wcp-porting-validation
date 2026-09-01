#!/usr/bin/env python3
"""doc pr/141 item 3 -- the owner's 2026-08-31 scan, scored, plus the two
mechanisms the owner named.

READ-ONLY.  Reads em_labels/pi0mass-0904-owner/ (the owner's own tag) and the
production dumps work-pr140r2-off-*, and writes only its --tsv.

The owner scanned the nine mass failures and reported two mechanisms:

  * "the gamma of pi0 is close to the detector boundary, so we miss some
    energies" -- a CONTAINMENT loss, which drives the reconstructed mass DOWN;
  * "likely overclustering leading to overestimation of energy thus
    overestimating the pi0 mass" -- which drives it UP.

Both are testable here.  For containment the script measures, per gamma, how
much room the shower had: the distance from its start point, ALONG ITS OWN
AXIS, to the SBND active-volume wall (x,y in +-200 cm, z in 0..500 cm, taken
from the point clouds themselves), and the wall distance of its furthest
member point.  A LAr EM shower needs roughly 5-10 X0 ~ 70-140 cm to contain
(X0 = 14.0 cm), so `room` well below that is energy over the wall.

For over-clustering the script reports the size of each gamma (charge, length,
segment count) next to the factor the pair's mass would have to move by.

It also re-scores the owner's UPDATED pairings against the finder's window,
which is the point of the scan: three events were re-paired.

    python3 scripts/pr141_ownerscan.py --tsv docs/pr/pr141-ownerscan.tsv
"""
import argparse, json, math, os, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = "pi0mass-0904-owner"
OFFSET, WIN = 10.0, (100.0, 160.0)
X0 = 14.0          # LAr radiation length, cm -- quoted for scale only
VOL = ((-200.0, 200.0), (-200.0, 200.0), (0.0, 500.0))   # SBND active volume
EC = 32.8          # LAr critical energy, MeV (electrons)


def contained_frac(E_mev, room_cm):
    """Fraction of an EM shower's energy deposited within `room` of its start.

    PDG longitudinal profile, dE/dt = E b (bt)^(a-1) e^(-bt) / Gamma(a) with
    t in radiation lengths, b ~ 0.5 and t_max = (a-1)/b = ln(E/Ec) + C, C =
    -0.5 for a photon-initiated shower.  The contained fraction to depth T is
    then the regularised lower incomplete gamma P(a, bT).  This is a textbook
    average profile, not a fit to our detector: it is used here only to ask
    whether the ORDER of the missing energy matches the room available, which
    is what the owner's boundary claim needs.
    """
    if E_mev is None or E_mev <= 0 or room_cm is None:
        return None
    b = 0.5
    tmax = max(0.1, math.log(max(E_mev, 1.1 * EC) / EC) - 0.5)
    a = b * tmax + 1.0
    T = max(0.0, room_cm) / X0
    # regularised lower incomplete gamma P(a, bT) by series (bT is small here)
    x = b * T
    if x <= 0:
        return 0.0
    term = 1.0 / a
    ssum = term
    for n in range(1, 400):
        term *= x / (a + n)
        ssum += term
        if term < 1e-12 * ssum:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * ssum)

ARM = {21073: "work-pr140r2-off-ncpi0",
       168432: "work-pr140r2-off-mcp1k", 280159: "work-pr140r2-off-mcp2k",
       286655: "work-pr140r2-off-mcp1k", 348691: "work-pr140r2-off-mcp1k",
       397630: "work-pr140r2-off-mcp2k", 409634: "work-pr140r2-off-mcp1k",
       71872: "work-pr140r2-off-mcp2k", 283713: "work-pr140r2-off-mcp1k"}
# the pairing as it stood BEFORE this scan (pr141_massfail.py)
WAS = {21073: (60081, 63100), 168432: (22006, 49028), 280159: (24015, 95114),
       286655: (19006, 80055), 348691: (49046, 50073), 397630: (19010, 33038),
       409634: (21002, 69032), 71872: (79074, 64044), 283713: (67051, 47021)}


def wall_dist(p):
    return min(min(p[i] - VOL[i][0], VOL[i][1] - p[i]) for i in range(3))


def room_along(p, u):
    """distance from p along unit vector u to the first wall."""
    best = 1e9
    for i in range(3):
        for lim, sgn in ((VOL[i][0], -1.0), (VOL[i][1], 1.0)):
            if u[i] * sgn <= 1e-9:
                continue
            t = (lim - p[i]) / u[i]
            if t > 0:
                best = min(best, t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    rows = []
    for ev in sorted(ARM):
        lp = os.path.join(SX, "em_labels", TAG, "labels-evt%d.json" % ev)
        rec = json.load(open(lp))
        pio = rec.get("pio")
        dump = json.load(open(os.path.join(
            SX, ARM[ev], "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)))
        by = {int(s["id"]): s for s in (dump.get("showers") or ())}
        segs = {int(s["id"]): s for s in (dump.get("segments") or ())}

        if not pio:
            print("=== evt %d: NO PAIRING in the owner's tag "
                  "(seeded from a label whose pio was empty)" % ev)
            print()
            rows.append(dict(event=ev, g1="", g2="", changed="no-pairing",
                             m_vtx="", in_window="", note="seed lost the pio"))
            continue

        g = pio["gammas"]
        ids = [int(g[s]["shower"]) for s in ("1", "2")]
        was = WAS[ev]
        changed = "RE-PAIRED %s+%s -> %s+%s" % (was[0], was[1], ids[0], ids[1]) \
            if set(ids) != set(was) else "same pair"
        m_vtx = pio.get("mass_vertex_convention")
        m_ax = pio.get("mass_axis_convention")
        inwin = m_vtx is not None and WIN[0] <= m_vtx <= WIN[1]

        print("=== evt %d   %s" % (ev, changed))
        print("    mass: vertex %.1f  axis %.1f   %s"
              % (m_vtx or 0, m_ax or 0,
                 "IN WINDOW (%.0f,%.0f)" % WIN if inwin else "outside"))
        cont = []
        for slot in ("1", "2"):
            gg = g[slot]
            sid = int(gg["shower"])
            d = by.get(sid)
            ax = gg.get("axis") or [0, 0, 0]
            n = math.sqrt(sum(x * x for x in ax)) or 1.0
            u = [x / n for x in ax]
            st = gg.get("start") or [0, 0, 0]
            room = room_along(st, u)
            # furthest member point and its wall distance
            far, fard = None, None
            for m in (gg.get("members") or []):
                sg = segs.get(int(m))
                if not sg:
                    continue
                for p in (sg.get("points") or ()):
                    q = (p["x"], p["y"], p["z"])
                    dd = math.dist(q, st)
                    if far is None or dd > far:
                        far, fard = dd, wall_dist(q)
            print("    g%s %6d  E=%7.1f MeV  len=%6.1f cm  nseg=%-3s pdg=%-4s "
                  "start_wall=%5.1f  ROOM along axis=%6.1f cm (%.1f X0)  "
                  "far_point_wall=%s"
                  % (slot, sid, (d or {}).get("kine_charge") or 0,
                     (d or {}).get("total_length") or 0,
                     (d or {}).get("num_segments"), (d or {}).get("particle_id"),
                     wall_dist(st), room, room / X0,
                     "%.1f" % fard if fard is not None else "-"))
            cont.append((sid, room, fard))
        # the containment verdict: a gamma with < 5 X0 of room in front of it
        # the owner's containment claim, made quantitative: correct each
        # gamma's energy by its expected contained fraction and re-mass.
        fr = [contained_frac((by.get(c[0]) or {}).get("kine_charge"), c[1])
              for c in cont]
        m_corr = None
        if m_vtx and all(f and f > 0.02 for f in fr):
            m_corr = m_vtx / math.sqrt(fr[0] * fr[1])
            print("    containment correction: f(g%d)=%.2f  f(g%d)=%.2f  "
                  "=> mass %.1f -> %.1f  %s"
                  % (cont[0][0], fr[0], cont[1][0], fr[1], m_vtx, m_corr,
                     "IN WINDOW" if WIN[0] <= m_corr <= WIN[1] else "still outside"))
        tight = [c for c in cont if c[1] < 5 * X0]
        if tight:
            print("    CONTAINMENT: %s ha%s under 5 X0 (70 cm) of room -- "
                  "energy over the wall is expected here"
                  % (", ".join("g%d (%.0f cm)" % (c[0], c[1]) for c in tight),
                     "s" if len(tight) == 1 else "ve"))
        print()
        rows.append(dict(event=ev, g1=ids[0], g2=ids[1], changed=changed,
                         m_vtx="%.1f" % (m_vtx or 0),
                         m_axis="%.1f" % (m_ax or 0),
                         in_window=inwin,
                         room1="%.1f" % cont[0][1], room2="%.1f" % cont[1][1],
                         wallfar1=("%.1f" % cont[0][2]) if cont[0][2] is not None else "",
                         wallfar2=("%.1f" % cont[1][2]) if cont[1][2] is not None else "",
                         frac1=("%.3f" % fr[0]) if fr[0] else "",
                         frac2=("%.3f" % fr[1]) if fr[1] else "",
                         m_corr=("%.1f" % m_corr) if m_corr else "",
                         note=""))

    if args.tsv:
        hdr = ["event", "g1", "g2", "changed", "m_vtx", "m_axis", "in_window",
               "room1", "room2", "wallfar1", "wallfar2",
               "frac1", "frac2", "m_corr", "note"]
        with open(args.tsv, "w") as fh:
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(h, "")) for h in hdr) + "\n")
        print("wrote %s (%d rows)" % (args.tsv, len(rows)))


if __name__ == "__main__":
    main()
