#!/usr/bin/env python3
"""doc pr/141 item 3 -- why the nine hand pi0 pairs reconstruct at the wrong mass.

READ-ONLY (CLAUDE.md M13).  Writes only its own --tsv.

doc pr/139 sec 26.3 closed with nine events in the class "pair found, mass
outside the (100,160) with-vertex window": scaled masses 54, 57, 70, 78, 88, 94,
203, 226, 277 MeV.  Seven of the nine are wrong by factors, not by a scale, and
sec 26.5 item 3 asked which of two mechanisms does it: a wrong PAIRING, or a
grossly mis-clustered gamma.

This script says what can be said without a scan, so the scan only has to
adjudicate what is left.  Per event it prints:

  * the hand pair as the label recorded it: both gamma energies at scan-time
    scale, both angle conventions, both masses;
  * the SAME pair evaluated the way the finder evaluates it.  The finder is
    neither convention: `local_dirs` is `get_init_dir()` for a shower attached
    to the candidate vertex (conn_type 1) and the vertex->shower ray for an
    associated one (NeutrinoShowerClustering.cxx:7547-7620, mass at :7707).
    So the label's two masses BRACKET the finder's, and which bracket end
    applies is decided per gamma by `start_connection_type` in the dump;
  * what would have to move for the pair to land at 135: the energy product
    (a clustering defect) or the opening angle (a start-point / vertex defect);
  * what the reconstruction paired INSTEAD, from the dump's own pio_id /
    pio_mass -- the direct test of "wrong pairing";
  * the best alternative partner for each hand gamma among the event's EM
    showers, under the vertex convention.

Energies: labels are scan-time (kine_shower_fudge_factor 0.80); the arm runs at
0.86, so every label energy is multiplied by 0.80/0.86 and every mass with it
(mass goes as sqrt(E1 E2), so mass scales by the same factor).

    python3 scripts/pr141_massfail.py --tsv docs/pr/pr141-massfail.tsv
"""
import argparse, json, math, os, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# doc pr/139 sec 26.3.  The seven wrong-by-factors, then the two near-misses
# (scaled 88 and 94) that a pure energy-scale correction could still reach.
EVENTS = [
    # event, sample, arm-dir, base label tag, pio tag (base wins; overlay when base has none)
    (21073,  "ncpi0",   "work-pr140r2-off-ncpi0",  "emscan-0827",        "emscan-0827",        "primary"),
    (168432, "mcp1k",   "work-pr140r2-off-mcp1k",  "emscan-0828-agent5", "emscan-0828-agent5", "primary"),
    (280159, "mcp2k",   "work-pr140r2-off-mcp2k",  "emscan-0828-agent5", "emscan-0828-agent5", "primary"),
    (286655, "mcp1k",   "work-pr140r2-off-mcp1k",  "emscan-0828-agent5", "pi0scan-0829-agent", "primary"),
    (348691, "mcp1k",   "work-pr140r2-off-mcp1k",  "emscan-0828-agent5", "pi0scan-0829-agent", "primary"),
    (397630, "mcp2k",   "work-pr140r2-off-mcp2k",  "emscan-0828-agent5", "pi0scan-0829-agent", "primary"),
    (409634, "mcp1k",   "work-pr140r2-off-mcp1k",  "emscan-0827",        "emscan-0827",        "primary"),
    (71872,  "mcp2k",   "work-pr140r2-off-mcp2k",  "emscan-0828-agent5", "emscan-0828-agent5", "near-miss"),
    (283713, "mcp1k",   "work-pr140r2-off-mcp1k",  "emscan-0828-agent5", "emscan-0828-agent5", "near-miss"),
]

SCALE = 0.80 / 0.86          # scan-time fudge -> arm fudge
OFFSET = 10.0                # m_pi0_mass_offset, C++ default and production
WIN = (135.0 - OFFSET - 25.0, 135.0 - OFFSET + 35.0)   # (100, 160) MeV
                             # NeutrinoShowerClustering.cxx: reject when
                             # mass-135+offset >= 35 or <= -25


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def ang_deg(u, v):
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu <= 0 or nv <= 0:
        return None
    c = sum(u[i] * v[i] for i in range(3)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pi0_mass(e1, e2, th_deg):
    if e1 is None or e2 is None or th_deg is None:
        return None
    return math.sqrt(4.0 * e1 * e2 * math.sin(math.radians(th_deg) / 2.0) ** 2)


def load_pio(ev, base_tag, pio_tag):
    """The pairing the census scored, with base-wins-on-conflict."""
    for tag in (base_tag, pio_tag):
        p = os.path.join(SX, "em_labels", tag, "labels-evt%d.json" % ev)
        if not os.path.exists(p):
            continue
        rec = json.load(open(p))
        if rec.get("pio"):
            return rec, rec["pio"], tag
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    rows = []
    for ev, sample, arm, base_tag, pio_tag, cls in EVENTS:
        dump = json.load(open(os.path.join(
            SX, arm, "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)))
        by = {int(s["id"]): s for s in (dump.get("showers") or ())}
        mv = dump.get("main_vertex") or {}
        vtx = [mv.get("x"), mv.get("y"), mv.get("z")]
        mvid = None
        for v in (dump.get("vertices") or ()):
            if v.get("is_main"):
                mvid = int(v.get("id"))
                break

        rec, pio, used_tag = load_pio(ev, base_tag, pio_tag)
        g = pio["gammas"]
        gs = []
        for slot in ("1", "2"):
            gg = g[slot]
            sid = int(gg["shower"])
            d = by.get(sid)
            gs.append(dict(
                slot=slot, sid=sid,
                e_lab=gg["energy"], e_arm=gg["energy"] * SCALE,
                axis=gg["axis"], start=gg["start"],
                nmem=len(gg.get("members") or []),
                pid_lab=gg.get("particle_id"),
                e_dump=(d or {}).get("kine_best"),
                conn=(d or {}).get("start_connection_type"),
                pid=(d or {}).get("particle_id"),
                length=(d or {}).get("total_length"),
                nseg=(d or {}).get("num_segments"),
                pio_id=(d or {}).get("pio_id"),
                svtx=(d or {}).get("start_vertex_id"),
                present=d is not None))

        # the label's own two conventions, rescaled to the arm
        th_ax = pio.get("theta_axis_convention")
        th_vx = pio.get("theta_vertex_convention")
        m_ax = (pio.get("mass_axis_convention") or 0) * SCALE or None
        m_vx = (pio.get("mass_vertex_convention") or 0) * SCALE or None

        # the finder's own hybrid: init_dir when attached to the vertex
        # (conn_type 1), vertex->start ray when associated.
        dirs = []
        for gg in gs:
            if gg["svtx"] is not None and mvid is not None and gg["svtx"] == mvid:
                dirs.append(("axis", gg["axis"]))
            else:
                dirs.append(("ray", vsub(gg["start"], vtx)))
        branch = "%s/%s" % (dirs[0][0], dirs[1][0])
        th_hy = ang_deg(dirs[0][1], dirs[1][1])
        m_hy = pi0_mass(gs[0]["e_arm"], gs[1]["e_arm"], th_hy)

        # what would have to move.  mass ~ sqrt(E1 E2) sin(th/2).
        need_eprod = (135.0 / m_hy) ** 2 if m_hy else None
        need_sin = (135.0 / m_hy) if m_hy else None
        need_th = (2.0 * math.degrees(math.asin(min(1.0,
                   math.sin(math.radians(th_hy) / 2.0) * need_sin)))
                   if (m_hy and th_hy) else None)
        th_reach = need_sin is not None and math.sin(
            math.radians(th_hy) / 2.0) * need_sin <= 1.0

        # what the reconstruction paired instead
        reco_pairs = {}
        for s in (dump.get("showers") or ()):
            pid = s.get("pio_id")
            if pid is not None and int(pid) >= 0:
                reco_pairs.setdefault(int(pid), []).append(s)
        reco_txt = []
        for pid in sorted(reco_pairs):
            mem = sorted(reco_pairs[pid], key=lambda s: -(s.get("kine_best") or 0))
            reco_txt.append("pio%d[%s] m=%.1f" % (
                pid, ",".join("%d:%.0f" % (int(s["id"]), s.get("kine_best") or 0)
                              for s in mem),
                (mem[0].get("pio_mass") or 0)))
        # did the reco pair share a gamma with the hand pair?
        hand_ids = {gs[0]["sid"], gs[1]["sid"]}
        reco_ids = {int(s["id"]) for v in reco_pairs.values() for s in v}
        shared = sorted(hand_ids & reco_ids)

        # best alternative partner for each hand gamma, vertex convention
        alts = []
        for gg in gs:
            if not gg["present"]:
                alts.append("-")
                continue
            best = None
            s_self = by[gg["sid"]]
            p_self = [s_self["start"]["x"], s_self["start"]["y"], s_self["start"]["z"]]
            for s in (dump.get("showers") or ()):
                sid = int(s["id"])
                if sid == gg["sid"]:
                    continue
                if abs(int(s.get("particle_id") or 0)) != 11:
                    continue
                e2 = s.get("kine_best") or 0
                if e2 < 10.0:
                    continue
                p2 = [s["start"]["x"], s["start"]["y"], s["start"]["z"]]
                th = ang_deg(vsub(p_self, vtx), vsub(p2, vtx))
                m = pi0_mass(gg["e_arm"], e2, th)
                if m is None:
                    continue
                d = abs(m - 135.0)
                if best is None or d < best[0]:
                    best = (d, sid, e2, th, m)
            alts.append("-" if best is None else
                        "%d:E=%.0f th=%.0f m=%.0f" % (best[1], best[2], best[3], best[4]))

        # The finder is GREEDY: among admissible pairs it takes the one whose
        # |mass - 135 + offset| is smallest.  So an in-window hand pair can
        # still lose to a wrong pair that sits closer to the peak.
        hand_delta = abs(m_hy - 135.0 + OFFSET) if m_hy else None
        reco_best = None
        for pid in sorted(reco_pairs):
            mm = (reco_pairs[pid][0].get("pio_mass") or 0)
            d = abs(mm - 135.0 + OFFSET)
            if reco_best is None or d < reco_best[0]:
                reco_best = (d, pid, mm)
        if m_hy is None:
            compete = "-"
        elif not (WIN[0] <= m_hy <= WIN[1]):
            compete = "hand pair NOT admissible (mass outside window)"
        elif reco_best is None:
            compete = "hand pair admissible, and the reco found NO pi0 at all"
        elif reco_best[0] < hand_delta:
            compete = ("hand pair ADMISSIBLE but LOSES the greedy pick: "
                       "hand |d|=%.1f vs reco pio%d m=%.1f |d|=%.1f"
                       % (hand_delta, reco_best[1], reco_best[2], reco_best[0]))
        else:
            compete = ("hand pair admissible and would WIN (|d|=%.1f vs %.1f) "
                       "-- blocked elsewhere" % (hand_delta, reco_best[0]))

        rows.append(dict(
            hand_delta=hand_delta, compete=compete,
            event=ev, sample=sample, cls=cls, pio_tag=used_tag,
            g1=gs[0]["sid"], e1=gs[0]["e_arm"], conn1=gs[0]["conn"],
            len1=gs[0]["length"], nseg1=gs[0]["nseg"], pid1=gs[0]["pid"],
            g2=gs[1]["sid"], e2=gs[1]["e_arm"], conn2=gs[1]["conn"],
            len2=gs[1]["length"], nseg2=gs[1]["nseg"], pid2=gs[1]["pid"],
            th_axis=th_ax, m_axis=m_ax, th_vtx=th_vx, m_vtx=m_vx,
            th_hybrid=th_hy, m_hybrid=m_hy, branch=branch,
            vertex_how=pio.get("vertex_how"),
            svtx1=gs[0]["svtx"], svtx2=gs[1]["svtx"], main_vtx_id=mvid,
            need_eprod=need_eprod, need_theta=need_th, theta_reachable=th_reach,
            in_window=(m_hy is not None and WIN[0] <= m_hy <= WIN[1]),
            reco_pi0="; ".join(reco_txt) or "none",
            shares_gamma=",".join(str(x) for x in shared) or "-",
            alt1=alts[0], alt2=alts[1],
            e1_dump=gs[0]["e_dump"], e2_dump=gs[1]["e_dump"]))

    hdr = ["event", "sample", "cls", "pio_tag", "g1", "e1", "conn1", "len1",
           "nseg1", "pid1", "g2", "e2", "conn2", "len2", "nseg2", "pid2",
           "th_axis", "m_axis", "th_vtx", "m_vtx", "th_hybrid", "m_hybrid",
           "need_eprod", "need_theta", "theta_reachable", "in_window",
           "branch", "vertex_how", "svtx1", "svtx2", "main_vtx_id",
           "hand_delta", "compete",
           "reco_pi0", "shares_gamma", "alt1", "alt2", "e1_dump", "e2_dump"]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return "%.3f" % v
        return str(v)

    print("# doc pr/141 item 3 -- the nine mass failures, arm scale (fudge 0.86)")
    print("# window on the arm scale: (%.0f, %.0f) MeV\n" % WIN)
    for r in rows:
        print("=== evt %d (%s, %s)  pio from %s" % (
            r["event"], r["sample"], r["cls"], r["pio_tag"]))
        print("    g1 %6d  E=%6.1f MeV  conn=%s  len=%5.1f cm  nseg=%s  pdg=%s"
              % (r["g1"], r["e1"], r["conn1"], r["len1"] or 0, r["nseg1"], r["pid1"]))
        print("    g2 %6d  E=%6.1f MeV  conn=%s  len=%5.1f cm  nseg=%s  pdg=%s"
              % (r["g2"], r["e2"], r["conn2"], r["len2"] or 0, r["nseg2"], r["pid2"]))
        print("    theta  axis %6.1f deg -> m %6.1f | vertex %6.1f deg -> m %6.1f "
              "| FINDER(hybrid) %6.1f deg -> m %6.1f  %s"
              % (r["th_axis"] or 0, r["m_axis"] or 0, r["th_vtx"] or 0,
                 r["m_vtx"] or 0, r["th_hybrid"] or 0, r["m_hybrid"] or 0,
                 "IN WINDOW" if r["in_window"] else "out"))
        print("    finder branch %s (start vtx %s / %s vs main %s); label vertex_how=%s"
              % (r["branch"], r["svtx1"], r["svtx2"], r["main_vtx_id"],
                 r["vertex_how"]))
        print("    to reach 135: E1*E2 x %.2f   or   theta %.1f -> %.1f deg%s"
              % (r["need_eprod"] or 0, r["th_hybrid"] or 0, r["need_theta"] or 0,
                 "" if r["theta_reachable"] else "  (UNREACHABLE at any angle)"))
        print("    competition: %s" % r["compete"])
        print("    reco paired instead: %s   shares gamma: %s"
              % (r["reco_pi0"], r["shares_gamma"]))
        print("    best alt partner  g1 -> %s" % r["alt1"])
        print("                      g2 -> %s" % r["alt2"])
        print()

    if args.tsv:
        with open(args.tsv, "w") as fh:
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write("\t".join(fmt(r[h]) for h in hdr) + "\n")
        print("wrote %s (%d rows)" % (args.tsv, len(rows)))


if __name__ == "__main__":
    main()
