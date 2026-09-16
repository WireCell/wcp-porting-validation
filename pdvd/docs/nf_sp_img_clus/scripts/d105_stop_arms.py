#!/usr/bin/env python3
"""doc pdvd/105 sec 5 -- the four gate inputs that exist nowhere in the output (read-only).

stm_michel_michel_gate (StmMichelFunctions.cxx:668-688) decides on six quantities.  Four are persisted
(michel_len, michel_mip, michel_kink_deg, michel_far_len) and doc 104 sec 2 refuted cuts on all of them.  The other
two -- a.shower_like and a.terminal -- are not persisted anywhere, and neither are two richer measurements the
component computes for its own DEBUG line:

    CheckSTM_Michel stop-arm: cluster C seg S kind K len L cm far_len F cm mip M kink D deg
                              shower B terminal T kink5 K5 far_full FF kink_w KW

  kink5     the same kink measured over a 5 cm window instead of th.dir_window's 15 cm.  A genuine Michel leaves
            the stop at a sharp angle; a fit wobble at the end of a MIS-TERMINATED muon should look straighter the
            closer in you measure.  So kink5 - kink is a shape the 15 cm number cannot express.
  far_full  the track length reachable beyond the arm's far vertex, walked to 100 cm with the STOP FENCED OFF --
            where michel_far_len stops at michel_max_len (25 cm) and reports 0 for a terminal arm.  This is the
            direct test of doc 104 sec 3's mechanism: if the trajectory stopped short, the muon CONTINUES past the
            arm and far_full is large; a real Michel ends.

kind is the classifier's verdict for that arm: 1 = kMichel (admitted), 0 = kOther, and a kContinuation arm is
logged with the continuation clause's own kind.  Every arm examined at the stop is logged, admitted or not, so
this is also the only view of the REJECTED arms.

The lines are per event (one log per event dir), so the cluster key is <run>_<evt>/<cluster>.  DEBUG logging must
be on in the arm: the stock production job logs nothing here.

    python3 d105_stop_arms.py --arm d105hdiag > figs/105_stop_arms.txt
"""
import argparse, glob, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d104_michel_features as F

RX = re.compile(r"stop-arm: cluster (\d+) seg (\d+) kind (\d+) len ([-\d.]+) cm far_len ([-\d.]+) cm "
                r"mip ([-\d.]+) kink ([-\d.]+) deg shower (\d+) terminal (\d+) kink5 ([-\d.]+) "
                r"far_full ([-\d.]+) kink_w ([-\d.]+)")
FIELDS = ["seg", "kind", "len", "far_len", "mip", "kink", "shower", "terminal", "kink5", "far_full", "kink_w"]


def arms(det, arm):
    """{key: [per-stop-arm dicts]} parsed from each event's PR log"""
    out = {}
    for d in sorted(glob.glob(f"{F.IMG}/{det}/work/*_{arm}")):
        ev = os.path.basename(d)[:-len(arm) - 1]
        for lg in glob.glob(f"{d}/wct_pr_*.log"):
            for line in open(lg, errors="replace"):
                m = RX.search(line)
                if not m:
                    continue
                g = m.groups()
                out.setdefault(f"{ev}/{int(g[0])}", []).append(
                    dict(zip(FIELDS, [int(g[1]), int(g[2])] + [float(x) for x in g[3:7]]
                             + [int(g[7]), int(g[8])] + [float(x) for x in g[9:]])))
    return out


def auc(pos, neg):
    if not pos or not neg:
        return float("nan"), (float("nan"), float("nan"))
    def one(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float((np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b)))
    rng = np.random.default_rng(105)
    bs = [one(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(400)]
    return one(pos, neg), (float(np.percentile(bs, 16)), float(np.percentile(bs, 84)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="pdhd")
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    D = F.load(a.det)
    tp, fp = F.split(D)
    A = arms(a.det, a.arm)
    print(f"# doc pdvd/105 sec 5: stop-arm DEBUG records from {a.arm} ({a.det})")
    print(f"# clusters with at least one logged stop arm: {len(A)}; total arms {sum(len(v) for v in A.values())}")
    print("# truth/population are the graded cells'; see sec 3 for whether this arm reconstructs the same.")

    def admitted(k):
        """the arm the gate admitted (kind 1), else None"""
        return next((r for r in sorted(A.get(k, []), key=lambda r: -r["len"]) if r["kind"] == 1), None)

    tpk = [k for k, _ in tp if admitted(k)]
    fpk = [k for k, _ in fp if admitted(k)]
    print(f"# matched to an ADMITTED arm: TP {len(tpk)} of {len(tp)}, FP {len(fpk)} of {len(fp)}")
    if not fpk:
        print("\n   no false positive has a logged admitted stop arm -- nothing to separate on this population")
    else:
        print("\n== separation on the admitted arm (AUC > 0.5 => larger on TRUE Michels)")
        feats = {"kink": lambda r: r["kink"], "kink5": lambda r: r["kink5"],
                 "kink5 - kink": lambda r: r["kink5"] - r["kink"],
                 "far_full": lambda r: r["far_full"], "far_full - far_len": lambda r: r["far_full"] - r["far_len"],
                 "shower": lambda r: r["shower"], "terminal": lambda r: r["terminal"],
                 "mip": lambda r: r["mip"], "len": lambda r: r["len"]}
        for nm, fn in feats.items():
            P = [fn(admitted(k)) for k in tpk]
            N = [fn(admitted(k)) for k in fpk]
            s, (lo, hi) = auc(P, N)
            qp = np.percentile(P, [25, 50, 75])
            print(f"   {nm:18s} TP q1/med/q3 {qp[0]:7.2f} {qp[1]:7.2f} {qp[2]:7.2f} | FP "
                  + " ".join(f"{x:6.1f}" for x in sorted(N)) + f" | AUC {s:.3f} [{lo:.3f}, {hi:.3f}]")
        print("\n== the false positives' admitted arms in full")
        for k in sorted(fpk):
            r = admitted(k)
            print(f"   {k:16s} " + " ".join(f"{f} {r[f]}" for f in FIELDS))

    print("\n== every logged arm at the stop, by classifier verdict (kind), TP vs FP clusters")
    for lab, ks in (("TRUE Michels", [k for k, _ in tp if k in A]), ("FALSE positives", [k for k, _ in fp if k in A])):
        kinds = {}
        for k in ks:
            for r in A[k]:
                kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"   {lab:16s} clusters {len(ks):4d}  arms by kind {dict(sorted(kinds.items()))}")


if __name__ == "__main__":
    main()
