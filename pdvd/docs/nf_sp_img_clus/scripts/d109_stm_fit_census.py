#!/usr/bin/env python3
"""doc pdvd/109 -- why the PDHD Bee `stm_fit` layer draws ~5x more than PDVD's.  Read-only.

    python3 d109_stm_fit_census.py

Everything comes from products already on disk: the per-event job log and mabc-pr.zip of
each arm.  Nothing is re-run.

Four measurements, in the order the argument needs them:

  1  LAYER OCCUPANCY.  Clusters and points in `data/0/0-stm_fit-global.json` per event,
     both detectors.  This is the number the owner saw in Bee.

  2  WHICH SET EACH DETECTOR DRAWS.  Compare the layer's cluster count against two
     candidate populations read from the same log: the clusters the tagger FITTED
     (`persist_stm_fit` records) and the clusters it TAGGED (`STM=1` verdicts).  A layer
     with no gate draws the fitted set; `require_flag:'STM'` draws the tagged set.

  3  THE GATE, APPLIED OFFLINE.  Intersect the PDHD layer's cluster_id array with the
     STM=1 set from its own log.  `require_flag` is a per-cluster filter on exactly this
     dump, so this is what the gated layer will contain -- an exact prediction, not a
     model.  Also counts the clusters that are tagged but carry NO fit, which is the
     case that distinguishes require_flag from require_pc.

  4  THE RESIDUAL.  Mains, flash groups and mains-per-flash-group per event, from
     CreateSteinerGraph's beam_window line, to size the part of the difference that is
     NOT the missing gate.

Arms: PDHD d108hflip (the flipped production config, doc pdvd/108), PDVD q29flip.  Both
carry save_stm_fit=true and the same tagger knobs; see sec 3 of the doc.
"""
import re, os, glob, json, zipfile
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
ARMS = [("pdhd", "d108hflip"), ("pdvd", "q29flip")]

R_VERDICT = re.compile(r"visit: TaggerCheckSTM: cluster (\d+) (?:→|->) STM=(\d) TGM=(\d)")
R_PERSIST = re.compile(r"persist_stm_fit: cluster (\d+) stmfit pass=(\d+) status=(-?\d+)")
R_WINDOW  = re.compile(r"TaggerCheckSTM: beam_window_only \[([-\d.]+), ([-\d.]+)\) us: "
                       r"(\d+) main\(s\) evaluated, (\d+) out of window")
R_GRAPH   = re.compile(r"CreateSteinerGraph: beam_window_only .*: kept (\d+) of (\d+) cluster\(s\) "
                       r"\((\d+) in-window main\(s\), (\d+) flash group\(s\)\)")
BEE = "data/0/0-stm_fit-global.json"


def read_event(d):
    """One event dir -> the log-derived sets and the Bee layer arrays."""
    lg = glob.glob(os.path.join(d, "wct_pr_*.log"))
    if not lg:
        return None
    txt = open(lg[0], errors="replace").read()
    verdict = {int(c): (int(s), int(t)) for c, s, t in R_VERDICT.findall(txt)}
    fitted = {int(c) for c, _p, _s in R_PERSIST.findall(txt)}
    tagged = {c for c, (s, _t) in verdict.items() if s == 1}
    w = R_WINDOW.search(txt)
    g = R_GRAPH.search(txt)
    z = os.path.join(d, "mabc-pr.zip")
    cid = np.array([], int)
    if os.path.exists(z):
        with zipfile.ZipFile(z) as zz:
            if BEE in zz.namelist():
                j = json.loads(zz.read(BEE))
                cid = np.array(j.get("cluster_id", []), int)
    return dict(
        ev=os.path.basename(d), evaluated=len(verdict), fitted=len(fitted), tagged=len(tagged),
        drawn_c=len(set(cid.tolist())), drawn_p=len(cid),
        gated_c=len(set(cid.tolist()) & tagged),
        gated_p=int(np.isin(cid, list(tagged)).sum()) if len(cid) else 0,
        tagged_nofit=len(tagged - fitted),
        win=(float(w.group(1)), float(w.group(2))) if w else None,
        outwin=int(w.group(4)) if w else None,
        live=int(g.group(2)) if g else 0, mains=int(g.group(3)) if g else 0,
        groups=int(g.group(4)) if g else 0,
    )


def load(det, arm):
    rows = [r for r in (read_event(d) for d in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}"))) if r]
    return rows


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def stat(rows, k, label, per_event=True):
    v = col(rows, k)
    if per_event:
        print("    %-34s tot %8d   mean %8.2f   p50 %7.1f   p90 %8.1f   max %7d"
              % (label, v.sum(), v.mean(), np.percentile(v, 50), np.percentile(v, 90), v.max()))
    return v


def main():
    print(__doc__.split("\n\n")[0])
    print()
    D = {}
    for det, arm in ARMS:
        D[det] = (arm, load(det, arm))

    print("=" * 96)
    print("1. THE BEE `stm_fit` LAYER AS DRAWN  (what the owner saw)")
    print("=" * 96)
    for det, (arm, rows) in D.items():
        print("  %s  arm %s  (%d events)" % (det.upper(), arm, len(rows)))
        stat(rows, "drawn_c", "clusters drawn in the layer")
        stat(rows, "drawn_p", "points drawn in the layer")
    hc, vc = col(D["pdhd"][1], "drawn_c"), col(D["pdvd"][1], "drawn_c")
    hp, vp = col(D["pdhd"][1], "drawn_p"), col(D["pdvd"][1], "drawn_p")
    print("  PDHD / PDVD  clusters per event %.2fx    points per event %.2fx"
          % (hc.mean() / vc.mean(), hp.mean() / vp.mean()))

    print()
    print("=" * 96)
    print("2. WHICH POPULATION EACH LAYER DRAWS")
    print("=" * 96)
    print("    %-6s %10s %10s %10s %10s   %s" % ("det", "evaluated", "fitted", "tagged", "drawn", "drawn set =="))
    for det, (arm, rows) in D.items():
        e, f, t, d = (col(rows, k).sum() for k in ("evaluated", "fitted", "tagged", "drawn_c"))
        same = "fitted" if d == f else ("tagged" if d == t else "NEITHER")
        print("    %-6s %10d %10d %10d %10d   %s" % (det, e, f, t, d, same))
        # per-event identity, not just the total
        pf = int((col(rows, "drawn_c") == col(rows, "fitted")).sum())
        pt = int((col(rows, "drawn_c") == col(rows, "tagged")).sum())
        print("           per-event: drawn==fitted on %d/%d events, drawn==tagged on %d/%d"
              % (pf, len(rows), pt, len(rows)))
    print()
    print("    step efficiencies (pooled):")
    for det, (arm, rows) in D.items():
        e, f, t = (col(rows, k).sum() for k in ("evaluated", "fitted", "tagged"))
        print("      %-6s fitted/evaluated %5.1f%%   tagged/fitted %5.1f%%   tagged/evaluated %5.1f%%"
              % (det, 100 * f / e, 100 * t / f, 100 * t / e))

    print()
    print("=" * 96)
    print("3. THE MISSING GATE, APPLIED OFFLINE TO PDHD")
    print("=" * 96)
    rows = D["pdhd"][1]
    nc, gc = stat(rows, "drawn_c", "clusters NOW (no gate)"), stat(rows, "gated_c", "clusters GATED require_flag:'STM'")
    np_, gp = stat(rows, "drawn_p", "points NOW (no gate)"), stat(rows, "gated_p", "points GATED")
    print("    reduction: clusters %.2fx, points %.2fx" % (nc.sum() / gc.sum(), np_.sum() / gp.sum()))
    tn = col(rows, "tagged_nofit")
    print("    tagged-but-never-fitted clusters (require_flag keeps, require_pc would hide): "
          "tot %d, events with >=1: %d" % (tn.sum(), (tn > 0).sum()))
    print()
    print("    PDHD gated vs PDVD as-is, per event:")
    print("      clusters  %6.2f  vs %6.2f   -> %.2fx" % (gc.mean(), vc.mean(), gc.mean() / vc.mean()))
    print("      points    %6.1f  vs %6.1f   -> %.2fx" % (gp.mean(), vp.mean(), gp.mean() / vp.mean()))

    print()
    print("=" * 96)
    print("4. THE RESIDUAL -- how much of the difference is NOT the gate")
    print("=" * 96)
    for det, (arm, rows) in D.items():
        m, g, l = col(rows, "mains"), col(rows, "groups"), col(rows, "live")
        print("  %s (%d events)" % (det.upper(), len(rows)))
        print("    live clusters/evt %7.2f   in-window mains/evt %7.2f   flash groups/evt %7.2f"
              % (l.mean(), m.mean(), g.mean()))
        print("    mains per flash group  pooled %.3f   p50 %.3f" % (m.sum() / g.sum(), np.percentile(m / g, 50)))
        print("    mains per live cluster pooled %.4f" % (m.sum() / l.sum()))
        wins = {r["win"] for r in rows}
        outs = {r["outwin"] for r in rows}
        print("    beam window %s us, out-of-window counts across events: %s  <- the gate is INERT"
              % (wins.pop() if len(wins) == 1 else wins, outs))


if __name__ == "__main__":
    main()
