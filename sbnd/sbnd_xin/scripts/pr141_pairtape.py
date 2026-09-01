#!/usr/bin/env python3
"""doc pr/141 item 3 -- read WCT_PI0_PAIR_DEBUG and say what the FINDER did.

READ-ONLY.  Replaces the offline geometry model in pr141_massfail.py, which the
tape refuted on its first event: for 168432 the model predicted the axis branch
and m = 101.6, and the finder's own row says

    PI0_PAIR P1 pair sh1=22006 sh2=49028 ct1=2 ct2=2 vtx=19001 ... m=53.5

i.e. the ray branch, exactly the census's `mass_vertex_convention`.  So the
dump's `start_vertex_id` is NOT the quantity `get_svc()` compares against
`cand_vtx`, and no offline model of `local_dirs` is trustworthy.  This script
reads the answer instead of deriving it.

Per event it reports, for the OWNER'S pair:
  * did both gammas reach the pool at all (`assoc ... acc=`), and
  * did the pair get a mass row, and what mass, and
  * what was accepted, vetoed or win-rejected instead.

    python3 scripts/pr141_pairtape.py --tsv docs/pr/pr141-pairtape.tsv
"""
import argparse, json, os, re, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFSET, WIN = 10.0, (100.0, 160.0)

# event -> (arm with the tape, the hand pair from the label)
ARMS = {
    21073:  "work-pr141dbg-pair2-ncpi0",
    168432: "work-pr141dbg-pair-mcp1k",
    280159: "work-pr141dbg-pair-mcp2k",
    286655: "work-pr141dbg-pair-mcp1k",
    348691: "work-pr141dbg-pair2-mcp1k",
    397630: "work-pr141dbg-pair2-mcp2k",
    409634: "work-pr141dbg-pair2-mcp1k",
    71872:  "work-pr141dbg-pair-mcp2k",
    283713: "work-pr141dbg-pair2-mcp1k",
}
HAND = {   # from pr141_massfail.py (label pio, base wins over overlay)
    21073: (60081, 63100), 168432: (22006, 49028), 280159: (24015, 95114),
    286655: (19006, 80055), 348691: (49046, 50073), 397630: (19010, 33038),
    409634: (21002, 69032), 71872: (79074, 64044), 283713: (67051, 47021),
}

RE_PAIR = re.compile(r"pair sh1=(-?\d+) sh2=(-?\d+) ct1=(-?\d+) ct2=(-?\d+) "
                     r"vtx=(-?\d+) E1=([\d.]+) E2=([\d.]+) m=([\d.]+)")
RE_ASSOC = re.compile(r"assoc vtx=(-?\d+) sh=(-?\d+) E=([\d.]+) angle=([\d.]+) "
                      r"crumb=(\d+) acc=(\d+)")
RE_ACC = re.compile(r"accept sh1=(-?\d+) sh2=(-?\d+) vtx=(-?\d+) m=([\d.]+)")
RE_VETO = re.compile(r"veto sh1=(-?\d+) sh2=(-?\d+)")
RE_WR = re.compile(r"winreject sh1=(-?\d+) sh2=(-?\d+) best_m=([\d.]+)")
RE_BEGIN = re.compile(r"begin main_vtx=(-?\d+) n_cand_vtx=(\d+) n_disc=(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    rows = []
    for ev in sorted(ARMS, key=lambda e: -1 if e in (168432, 280159, 71872) else 0):
        log = os.path.join(SX, ARMS[ev], "pr_evt%d" % ev, "stdout.log")
        if not os.path.exists(log):
            print("evt %d: NO TAPE (%s)" % (ev, log))
            continue
        lines = [l for l in open(log, errors="replace") if "PI0_PAIR" in l]
        g1, g2 = HAND[ev]
        key = tuple(sorted((g1, g2)))

        begin = RE_BEGIN.search("".join(lines))
        pairs, assoc, acc, veto, wr = [], [], [], [], []
        for l in lines:
            m = RE_PAIR.search(l)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                pairs.append((tuple(sorted((a, b))), int(m.group(5)),
                              float(m.group(6)), float(m.group(7)), float(m.group(8))))
            m = RE_ASSOC.search(l)
            if m:
                assoc.append((int(m.group(2)), int(m.group(1)), float(m.group(3)),
                              float(m.group(4)), int(m.group(5)), int(m.group(6))))
            m = RE_ACC.search(l)
            if m:
                acc.append((tuple(sorted((int(m.group(1)), int(m.group(2))))),
                            int(m.group(3)), float(m.group(4))))
            m = RE_VETO.search(l)
            if m:
                veto.append(tuple(sorted((int(m.group(1)), int(m.group(2))))))
            m = RE_WR.search(l)
            if m:
                wr.append((tuple(sorted((int(m.group(1)), int(m.group(2))))),
                           float(m.group(3))))

        hand_rows = [p for p in pairs if p[0] == key]
        seen = {s for p in pairs for s in p[0]}
        assoc_by = {}
        for sh, vtx, E, ang, crumb, a in assoc:
            assoc_by.setdefault(sh, []).append((vtx, E, ang, crumb, a))

        # verdict
        if not hand_rows:
            missing = [s for s in key if s not in seen]
            why = []
            for s in missing:
                if s in assoc_by:
                    r = assoc_by[s]
                    why.append("g%d rejected at assoc (angle %s, acc=0)"
                               % (s, "/".join("%.1f" % x[2] for x in r)))
                else:
                    why.append("g%d never appears in the tape" % s)
            verdict = ("PAIR NEVER FORMED: " + "; ".join(why)) if missing else \
                      "PAIR NEVER FORMED: both gammas in the pool but no pair row"
            best = None
        else:
            best = min(hand_rows, key=lambda p: abs(p[4] - 135.0 + OFFSET))
            inwin = WIN[0] <= best[4] <= WIN[1]
            if key in veto:
                verdict = "pair formed at m=%.1f then VETOED" % best[4]
            elif not inwin:
                verdict = "pair formed at m=%.1f -- OUTSIDE (%.0f,%.0f)" % (
                    best[4], WIN[0], WIN[1])
            elif key in [a[0] for a in acc]:
                verdict = "pair ACCEPTED at m=%.1f" % best[4]
            else:
                wrm = [x for x in wr if x[0] == key]
                verdict = ("pair formed at m=%.1f, IN WINDOW, but lost: %s"
                           % (best[4], "winreject" if wrm else "not accepted"))

        acc_txt = "; ".join("%d+%d m=%.1f" % (a[0][0], a[0][1], a[2]) for a in acc) or "none"
        print("=== evt %d   hand pair %d + %d" % (ev, g1, g2))
        if begin:
            print("    tape: main_vtx=%s n_cand_vtx=%s n_disconnected=%s"
                  % begin.groups())
        print("    hand-pair rows: %s"
              % ("; ".join("vtx=%d E1=%.1f E2=%.1f m=%.1f" % (p[1], p[2], p[3], p[4])
                           for p in hand_rows) or "NONE"))
        for s in key:
            if s in assoc_by:
                print("    assoc g%d: %s" % (s, "; ".join(
                    "vtx=%d E=%.1f angle=%.1f crumb=%d acc=%d" % r for r in assoc_by[s])))
        print("    accepted: %s" % acc_txt)
        print("    => %s" % verdict)
        print()
        rows.append(dict(event=ev, g1=g1, g2=g2,
                         n_pair_rows=len(hand_rows),
                         m_finder=("%.1f" % best[4]) if best else "",
                         accepted=acc_txt, verdict=verdict))

    if args.tsv:
        hdr = ["event", "g1", "g2", "n_pair_rows", "m_finder", "accepted", "verdict"]
        with open(args.tsv, "w") as fh:
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[h]) for h in hdr) + "\n")
        print("wrote %s (%d rows)" % (args.tsv, len(rows)))


if __name__ == "__main__":
    main()
