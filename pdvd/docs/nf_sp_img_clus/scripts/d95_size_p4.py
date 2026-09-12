#!/usr/bin/env python3
"""doc pdvd/95 -- the owner's point 4, SIZED OFFLINE ONLY.

The owner (2026-09-11): "there will be cases where no Michel is identified, but if the above
energy is clearly visible with some points out of the STM track trajectory, I think we should
naturally add a Michel electron to the particle flow?  Note, this has to be the case the STM's
dQ/dx is highly consistent with expectation."

That is a rule with three clauses, and all three already exist as measurable quantities:

  1. no Michel identified        -> is_stm 1 AND michel_found 0
  2. energy clearly visible      -> the region energy of this round, at a stated radius
  3. STM dQ/dx consistent        -> Bragg-confirmed, contrast >= bragg_contrast_min x expected.
                                    Doc pdvd/82 measured this gate on the record: it holds on
                                    94 % of STM_ONLY and 81 % of STM_MICHEL but only 15 % of
                                    THRU -- a 6.5x shrink of the control population, and
                                    exactly the owner's "highly consistent with expectation".

THE BAR IS PRE-REGISTERED (/home/xqian/tmp/p95/pred.txt, written before any arm ran):
build it in a doc 96 ONLY at added-set purity >= 0.8 AND 0 THRU fires.
  targets  = record STM_MICHEL with michel_found 0   (a Michel the chain is missing)
  controls = record STM_ONLY at is_stm 1             (owner: stopper, and NO Michel)
           + record THRU at is_stm 1                 (already an is_stm false positive)

THE HONEST PRIOR, also pre-registered: I expect this NOT to clear.  Doc pdvd/86 sec 8.2 sized
the neighbouring rule (the post-verdict anchor-tail Michel) at added-set purity 0.5 and it was
dropped; doc pdvd/94 then measured this very target population and found no charge to find --
`past` 1.5 points against a null of 2.0, off == 0 on 5 of 10.  What is genuinely NEW here is
that neither of those tested measured-minus-muon-prediction, which is a different quantity from
the raw off-fit charge doc 94 measured.  So it is worth measuring, and the expectation is
recorded in advance precisely so a result that DOES clear cannot be waved through afterwards,
and one that does not cannot be quietly reframed.

This script only SIZES.  It writes no C++ and flips nothing: admitting a Michel would write
michel_found and the particle flow, which is a physics change needing its own round and
probably a blind scan of its own fires.
"""
import argparse, collections, glob, os, sys
import numpy as np
import uproot

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"


def combine(q, n, weights=(0.25, 0.25, 1.0), asym=0.04):
    w = [0.0 if n[p] == 0 else weights[p] for p in range(3)]
    if sum(w) <= 0:
        return 0.0
    o = sorted(range(3), key=lambda p: q[p])
    med, mx = q[o[1]], q[o[2]]
    if med + mx > 0 and abs(med - mx) / (med + mx) > asym:
        keep = [o[0], o[1]]
        wk = [w[p] for p in keep]
        return sum(q[p] * w[p] for p in keep) / sum(wk) if sum(wk) > 0 else 0.0
    return sum(q[p] * w[p] for p in range(3)) / sum(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="p95vq2db")
    ap.add_argument("--radius", type=float, default=10.0, help="region radius, cm (POST-HOC, see doc sec on R)")
    ap.add_argument("--own", action="store_true", help="restrict to own_blob cells (doc 78 item 9's scope)")
    a = ap.parse_args()

    if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx7_smx8_smx9_verdicts.json"):
        sys.exit("STM_SCAN_RECORD must name the merged smx1a..smx9 record")
    rec = C.load_record()

    rows = []
    K = None
    for d in sorted(glob.glob("%s/pdvd/work/*_%s" % (IMG, a.arm))):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn) or os.path.getsize(fn) < 100000:
            continue
        try:
            f = uproot.open(fn)
        except Exception:
            continue
        if "T_stm_michel" not in f or "T_stm_michel_2d" not in f:
            continue
        ev = os.path.basename(d)[: -len(a.arm) - 1]
        t = f["T_stm_michel"].arrays(library="np")
        c = f["T_stm_michel_2d"].arrays(library="np")
        cid = c["cluster_id"]
        for i in range(len(t["cluster_id"])):
            key = "%s/%d" % (ev, int(t["cluster_id"][i]))
            if key not in rec or not C.judged(rec[key]):
                continue
            if int(t["is_stm"][i]) != 1 or int(t["michel_found"][i]) != 0:
                continue          # clause 1: the chain called a stop and found no Michel
            sel = cid == t["cluster_id"][i]
            if not sel.any():
                continue
            ch = np.asarray(c["charge"])[sel]; pm = np.asarray(c["pred_mu"])[sel]
            pa = np.asarray(c["pred_all"])[sel]; xs = np.asarray(c["xshared"])[sel]
            pl = np.asarray(c["plane"])[sel]; ds = np.asarray(c["d_stop_cm"])[sel]
            ro = np.asarray(c["role"])[sel]
            own = np.asarray(c["own_blob"])[sel] if "own_blob" in c else np.ones(sel.sum(), int)
            con = np.where(xs > 0, np.maximum(pa - pm, 0.0), ch - pm)
            m = (ds >= 0) & (ds <= a.radius)
            if a.own:
                m = m & (own > 0)
            q = [float(con[m & (pl == p)].sum()) for p in range(3)]
            n = [int((m & (pl == p)).sum()) for p in range(3)]
            e = combine(q, n)
            if K is None:
                q40 = [float(con[(ds >= 0) & (ds <= 40) & (pl == p)].sum()) for p in range(3)]
                n40 = [int(((ds >= 0) & (ds <= 40) & (pl == p)).sum()) for p in range(3)]
                base = combine(q40, n40)
                if abs(base) > 1:
                    K = float(t["michel_ke_q2d_region"][i]) / base
            # clause 3: the STM's own dQ/dx consistent with a stopping muon (doc 82's gate)
            contrast = float(t["contrast"][i]); expected = float(t["contrast_expected"][i])
            bragg = contrast >= 0.6 * expected if expected > 0 else False
            rows.append(dict(key=key, e=e, bragg=bragg,
                             role0=int((m & (ro == 0)).sum()),
                             verdict=rec[key]["verdict"],
                             tgt=C.is_michel(rec[key]),
                             ctl_only=C.is_stopper(rec[key]) and not C.is_michel(rec[key]),
                             thru=not C.is_stopper(rec[key]),
                             conf=rec[key].get("confidence", "?")))
    K = K or 1.0
    for r in rows:
        r["mev"] = r["e"] * K

    nt = sum(1 for r in rows if r["tgt"])
    print("candidates with is_stm 1 and michel_found 0: %d  (targets %d, STM_ONLY %d, THRU %d)"
          % (len(rows), nt, sum(1 for r in rows if r["ctl_only"]), sum(1 for r in rows if r["thru"])))
    print("radius %.1f cm%s; MeV per combined electron %.4g\n" % (a.radius, " (own_blob only)" if a.own else "", K))

    print("=== the rule swept over the energy threshold; Bragg gate ON (the owner's clause 3)")
    print("  %-9s %8s %8s %8s %10s %10s" % ("E_min MeV", "targets", "STM_ONLY", "THRU", "purity", "clears bar?"))
    for emin in (2, 4, 6, 8, 10, 12, 15, 20, 25):
        fires = [r for r in rows if r["bragg"] and r["mev"] >= emin]
        g = sum(1 for r in fires if r["tgt"]); s = sum(1 for r in fires if r["ctl_only"]); h = sum(1 for r in fires if r["thru"])
        pur = g / len(fires) if fires else float("nan")
        ok = (len(fires) > 0 and pur >= 0.8 and h == 0)
        print("  %-9d %8d %8d %8d %10s %10s" % (emin, g, s, h, "%.2f" % pur if fires else "   n/a", "YES" if ok else "no"))

    print("\n=== the same WITHOUT the Bragg gate, to show what that clause is worth")
    print("  %-9s %8s %8s %8s %10s" % ("E_min MeV", "targets", "STM_ONLY", "THRU", "purity"))
    for emin in (4, 8, 12, 20):
        fires = [r for r in rows if r["mev"] >= emin]
        g = sum(1 for r in fires if r["tgt"]); s = sum(1 for r in fires if r["ctl_only"]); h = sum(1 for r in fires if r["thru"])
        print("  %-9d %8d %8d %8d %10s" % (emin, g, s, h, "%.2f" % (g / len(fires)) if fires else "   n/a"))

    print("\n=== the targets, item by item (what the rule would have to catch)")
    for r in sorted((r for r in rows if r["tgt"]), key=lambda r: -r["mev"]):
        print("  %-16s conf %-7s region %6.1f MeV  bragg %-5s  role0 cells %4d" % (
            r["key"], r["conf"], r["mev"], r["bragg"], r["role0"]))

    print("\n=== the loudest controls (what would be admitted as a phantom Michel)")
    for r in sorted((r for r in rows if not r["tgt"]), key=lambda r: -r["mev"])[:10]:
        print("  %-16s %-10s conf %-7s region %6.1f MeV  bragg %-5s" % (
            r["key"], r["verdict"], r["conf"], r["mev"], r["bragg"]))


if __name__ == "__main__":
    main()
