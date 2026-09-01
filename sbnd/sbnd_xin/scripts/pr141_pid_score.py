#!/usr/bin/env python3
"""doc pr/141 sec 22 -- score the owner's mu-typed PID verdicts.

READ-ONLY.  Reads docs/pr/pr141-pidset.tsv (the served set, built and committed
BEFORE the scan) and the owner's verdicts, and writes only its --tsv.

The predictor was pre-registered in scripts/pr141_pidset.py before the set was
served: mean segment length L/nseg < 40 cm => EM.  This script grades it on the
14 informative objects ONLY -- the 4 predicted-TRACK controls are 300-500 cm
cosmics that nobody would call showers, so they can only inflate accuracy.

Owner verdicts, 2026-08-31, verbatim; "?" = unclear, slightly favouring the
answer given (owner's own gloss), carried here as weak=True.

    python3 scripts/pr141_pid_score.py --tsv docs/pr/pr141-pid-score.tsv
"""
import argparse, csv, os

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYP = 1.657
# (event, obj): (verdict, weak)
OWNER = {
    (282979, 41018): ("EM", False), (283713, 17007): ("TRACK", False),
    (286681, 72040): ("EM", False), (348691, 51080): ("EM", False),
    (77328, 16017): ("TRACK", False), (99838, 14004): ("TRACK", False),
    (170098, 70030): ("EM", False), (176502, 20005): ("TRACK", True),
    (281165, 16006): ("TRACK", False), (292524, 9018): ("TRACK", False),
    (294174, 16028): ("EM", True), (318769, 65037): ("EM", False),
    (392901, 127030): ("TRACK", False), (396323, 10009): ("TRACK", False),
    (397401, 16012): ("TRACK", False), (499577, 13031): ("TRACK", False),
    (259542, 81053): ("TRACK", True), (235435, 2024): ("EM", False),
}
# doc pr/141 sec 9.5, the n=6 basis the predictor was fitted on -- reported
# separately, never pooled with the test set.
BASIS = {(283713, 23011): "EM", (350354, 18009): "EM", (122660, 54071): "EM",
         (392901, 23017): "TRACK", (280159, 90098): "TRACK", (294174, 25030): "TRACK"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="docs/pr/pr141-pidset.tsv")
    ap.add_argument("--tsv", default=None)
    a = ap.parse_args()

    rows = []
    with open(os.path.join(SX, a.set)) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            rows = list(csv.DictReader([line] + fh.readlines(), delimiter="\t"))
            break
    by = {(int(r["event"]), int(r["obj"])): r for r in rows}

    out, nEM, nTR = [], 0, 0
    for k, (v, weak) in sorted(OWNER.items()):
        r = by.get(k)
        if r is None:
            print("!! served object not in the set TSV: %s" % (k,))
            continue
        pred = r["predicted"]
        out.append(dict(event=k[0], obj=k[1], predicted=pred, owner=v,
                        weak=int(weak), agree=int(pred == v),
                        kine_charge=float(r["kine_charge"]),
                        dE_if_EM=round((HYP - 1.0) * float(r["kine_charge"]), 1),
                        nseg=int(r["nseg"]), length=float(r["length"]),
                        cm_per_seg=float(r["cm_per_seg"]),
                        kine_range=float(r["kine_range"]), conn=int(r["conn"])))
    info = [r for r in out if r["predicted"] == "EM"]
    ctrl = [r for r in out if r["predicted"] == "TRACK"]

    print("=== the predictor, graded on the 14 informative objects ===")
    hit = sum(r["agree"] for r in info)
    print("  predicted EM, owner says EM   : %d" % sum(1 for r in info if r["owner"] == "EM"))
    print("  predicted EM, owner says TRACK: %d" % sum(1 for r in info if r["owner"] == "TRACK"))
    print("  -> precision on the predicted-EM class: %d/%d = %.3f" % (hit, len(info), hit / len(info)))
    print("=== the 4 controls (no information, stated before the scan) ===")
    print("  predicted TRACK, owner says TRACK: %d of %d" % (
        sum(1 for r in ctrl if r["owner"] == "TRACK"), len(ctrl)))
    print("  (pooling them would report %d/%d = %.3f -- do not)" % (
        hit + sum(r["agree"] for r in ctrl), len(out),
        (hit + sum(r["agree"] for r in ctrl)) / len(out)))

    em = [r for r in out if r["owner"] == "EM"]
    emS = [r for r in em if not r["weak"]]
    print("\n=== the physics: mu-typed objects that are really EM ===")
    print("  hand-typed this round : %d of 18 (%d at full confidence)" % (len(em), len(emS)))
    print("  with sec 9.5's six    : %d EM of 24 mu-typed objects hand-typed in total"
          % (len(em) + sum(1 for v in BASIS.values() if v == "EM")))
    print("  Enu missing, all EM   : %.0f MeV over %d objects (239 events)"
          % (sum(r["dE_if_EM"] for r in em), len(em)))
    print("  Enu missing, firm only: %.0f MeV over %d objects"
          % (sum(r["dE_if_EM"] for r in emS), len(emS)))

    print("\n=== is ANY offline feature a separator?  (POST-HOC, n=18) ===")
    for name in ("cm_per_seg", "nseg", "length", "kine_charge", "conn"):
        e = sorted(r[name] for r in out if r["owner"] == "EM")
        t = sorted(r[name] for r in out if r["owner"] == "TRACK")
        ov = not (e[-1] < t[0] or t[-1] < e[0])
        print("  %-12s EM %-32s TRACK %-32s %s"
              % (name, "[%g .. %g]" % (e[0], e[-1]), "[%g .. %g]" % (t[0], t[-1]),
                 "overlaps" if ov else "SEPARATES"))
    kr = [(r["kine_charge"] / r["kine_range"] if r["kine_range"] > 0 else float("inf"), r)
          for r in out]
    e = sorted(x for x, r in kr if r["owner"] == "EM")
    t = sorted(x for x, r in kr if r["owner"] == "TRACK")
    print("  %-12s EM [%.2f .. %.2f]  TRACK [%.2f .. %.2f]  %s"
          % ("kc/krange", e[0], e[-1], t[0], t[-1],
             "overlaps" if not (e[-1] < t[0] or t[-1] < e[0]) else "SEPARATES"))

    print("\n%-8s %-7s %-9s %-7s %-5s %-7s %-8s %-9s" %
          ("event", "obj", "predicted", "owner", "ok", "nseg", "len", "cm/seg"))
    for r in sorted(out, key=lambda r: -r["kine_charge"]):
        print("%-8d %-7d %-9s %-7s %-5s %-7d %-8.1f %-9.1f%s"
              % (r["event"], r["obj"], r["predicted"], r["owner"],
                 "yes" if r["agree"] else "NO", r["nseg"], r["length"],
                 r["cm_per_seg"], "   (unclear)" if r["weak"] else ""))

    if a.tsv:
        cols = ["event", "obj", "predicted", "owner", "weak", "agree", "kine_charge",
                "dE_if_EM", "nseg", "length", "cm_per_seg", "kine_range", "conn"]
        with open(os.path.join(SX, a.tsv), "w") as fh:
            fh.write("# doc pr/141 sec 22 -- owner mu-typed PID verdicts vs the pre-registered predictor\n")
            fh.write("\t".join(cols) + "\n")
            for r in sorted(out, key=lambda r: -r["kine_charge"]):
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
        print("\nwrote %s (%d rows)" % (a.tsv, len(out)))


if __name__ == "__main__":
    main()
