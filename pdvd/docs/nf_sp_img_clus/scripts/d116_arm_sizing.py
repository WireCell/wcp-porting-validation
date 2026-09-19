#!/usr/bin/env python3
"""doc pdvd/116 sec 3 -- size the Michel-arm levers on their ELIGIBLE population before they are pre-registered.
Read-only.  Parses the per-event log of one arm for the CheckSTM_Michel "stop-arm:" DEBUG line (one per arm leaving
the stop: kind, len, far_len, mip, kink, shower, terminal), joins T_stm_michel (is_stm, michel_found) and the truth
(d113_grade precedence), and reports, on accepted stoppers WITHOUT a Michel:
  (a) kOther arms that would become a Michel if michel_min_kink_deg were lowered (mip in (lo, hi), len + far_len <=
      max_len, not shower-admitted, kink in [K, 30)), split by hand Michel truth (positive / negative / unlabelled);
  (b) kOther arms that would qualify if michel_max_len_cm were raised (25 < len + far_len <= L, kink >= 30 or shower);
  (c) the range-energy vetoes (n_michel_range_veto > 0) by michel_ke_best (the KE of the vetoed piece), split by truth.
And on accepted stoppers WITH a Michel (the current true / false positives): the same quantities, so the levers'
cost side is visible (a lower kink threshold also keeps arms that are Michels today).

Usage: d116_arm_sizing.py --det pdvd --arm d115vp3bwp05 --out figs/116_arm_sizing_pdvd.txt
"""
import argparse, collections, glob, gzip, os, re, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import uproot
import d113_grade as G
import d103_union_grade as U

RX = re.compile(r"CheckSTM_Michel stop-arm: cluster (\d+) seg (\d+) kind (\d) len ([\d.]+) cm far_len ([\d.]+) cm mip ([\d.]+) kink ([-\d.]+) deg shower (\d) terminal (\d)")
KIND = {0: "other", 1: "michel", 2: "continuation", 3: "delta", 4: "hadron"}
MIP_LO, MIP_HI, MAX_LEN, MIN_KINK, SHOWER_KINK = 0.3, 2.0, 25.0, 30.0, 15.0


def arms_of(det, arm):
    out = collections.defaultdict(list)
    for d in sorted(glob.glob(f"{U.IMG}/{det}/work/*_{arm}")):
        ev = os.path.basename(d)[:-len(arm) - 1]
        for lg in glob.glob(f"{d}/wct_pr_*.log") + glob.glob(f"{d}/wct_pr_*.log.gz"):
            op = gzip.open if lg.endswith(".gz") else open
            with op(lg, "rt", errors="replace") as f:
                for line in f:
                    m = RX.search(line)
                    if m:
                        c, seg, kind, ln, fl, mip, kink, sh, term = m.groups()
                        out[f"{ev}/{int(c)}"].append(dict(seg=int(seg), kind=int(kind), len=float(ln), far=float(fl), mip=float(mip),
                                                          kink=float(kink), shower=int(sh), terminal=int(term)))
    return out


def rows_of(det, arm):
    out = {}
    for d in glob.glob(f"{U.IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(
                ["cluster_id", "is_stm", "michel_found", "n_michel_range_veto", "michel_ke_best", "michel_len", "michel_dis_cm", "michel_conn_type"], library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            out[f"{ev}/{int(t['cluster_id'][i])}"] = {k: float(t[k][i]) for k in t if k != "cluster_id"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T, counts = G.truth(a.det)
    R = rows_of(a.det, a.arm)
    ARMS = arms_of(a.det, a.arm)
    P = U.population(a.det, T, {"A0": R}, False, stm_only_unset_negative=(a.det == "pdhd"))
    mt = P["m_truth"]
    def truth_of(k):
        if k in mt:
            return "pos" if mt[k] else "neg"
        return "unlabelled" if k not in T else "excluded"
    L = [f"# doc pdvd/116 arm sizing {a.det} {a.arm}: candidates {len(R)}, clusters with stop-arm lines {len(ARMS)}, truth {counts}",
         f"# production thresholds: michel_mip ({MIP_LO}, {MIP_HI}), michel_max_len_cm {MAX_LEN}, michel_min_kink_deg {MIN_KINK}, michel_shower_min_kink_deg {SHOWER_KINK}"]
    stoppers_no = [k for k, r in R.items() if r["is_stm"] > 0 and r["michel_found"] == 0]
    stoppers_yes = [k for k, r in R.items() if r["is_stm"] > 0 and r["michel_found"] > 0]
    L.append(f"accepted stoppers: with Michel {len(stoppers_yes)} (truth {collections.Counter(truth_of(k) for k in stoppers_yes)}), "
             f"without {len(stoppers_no)} (truth {collections.Counter(truth_of(k) for k in stoppers_no)}); "
             f"stoppers without a Michel and with no stop-arm line at all: {sum(1 for k in stoppers_no if k not in ARMS)}")
    # (a) kink lever
    L.append("\n== (a) michel_min_kink_deg lowered: kOther arms on stoppers WITHOUT a Michel that pass mip / len and would be admitted at kink >= K")
    for K in (25.0, 20.0, 15.0):
        cnt = collections.Counter(); keys = []
        for k in stoppers_no:
            for ar in ARMS.get(k, []):
                if ar["kind"] == 0 and MIP_LO < ar["mip"] < MIP_HI and ar["len"] + ar["far"] <= MAX_LEN and K <= ar["kink"] < MIN_KINK \
                        and not (ar["shower"] and ar["kink"] >= SHOWER_KINK):
                    cnt[truth_of(k)] += 1; keys.append((k, ar["kink"], ar["len"], ar["mip"])); break
        L.append(f"  K = {K:4.0f}: admits {sum(cnt.values())} stoppers: {dict(cnt)}" + (f"   e.g. {keys[:6]}" if keys else ""))
    # (b) max length lever
    L.append("\n== (b) michel_max_len_cm raised: kOther arms on stoppers WITHOUT a Michel with kink >= 30 (or shower-admitted), mip ok, 25 < len + far_len <= Lmax")
    for Lm in (30.0, 35.0, 40.0):
        cnt = collections.Counter(); keys = []
        for k in stoppers_no:
            for ar in ARMS.get(k, []):
                tot = ar["len"] + ar["far"]
                if ar["kind"] == 0 and MIP_LO < ar["mip"] < MIP_HI and MAX_LEN < tot <= Lm and \
                        ((ar["kink"] >= MIN_KINK) or (ar["shower"] and ar["kink"] >= SHOWER_KINK)):
                    cnt[truth_of(k)] += 1; keys.append((k, round(tot, 1), ar["kink"])); break
        L.append(f"  Lmax = {Lm:4.0f}: admits {sum(cnt.values())} stoppers: {dict(cnt)}" + (f"   e.g. {keys[:6]}" if keys else ""))
    # (c) range-energy vetoes
    L.append("\n== (c) michel_range_energy vetoes on stoppers WITHOUT a Michel (n_michel_range_veto > 0), by the vetoed piece's KE (michel_ke_best) and distance")
    for kmin in (10.0, 7.0, 5.0, 3.0):
        cnt = collections.Counter()
        for k in stoppers_no:
            r = R[k]
            if r["n_michel_range_veto"] > 0 and r["michel_ke_best"] >= kmin:
                cnt[truth_of(k)] += 1
        L.append(f"  ke_min = {kmin:4.0f} MeV would admit {sum(cnt.values())} stoppers: {dict(cnt)}")
    vet = [(k, R[k]["michel_ke_best"], R[k]["michel_dis_cm"], truth_of(k)) for k in stoppers_no if R[k]["n_michel_range_veto"] > 0]
    L.append(f"  all vetoed: {len(vet)}; {sorted(vet, key=lambda x: -x[1])[:12]}")
    # the cost side: current Michels by kink / length / KE
    L.append("\n== current Michels (stoppers WITH a Michel): the Michel arm's kink and total length by truth")
    for lab in ("pos", "neg", "unlabelled"):
        ks = [k for k in stoppers_yes if truth_of(k) == lab]
        kinks = [ar["kink"] for k in ks for ar in ARMS.get(k, []) if ar["kind"] == 1]
        lens = [ar["len"] + ar["far"] for k in ks for ar in ARMS.get(k, []) if ar["kind"] == 1]
        kes = [R[k]["michel_ke_best"] for k in ks]
        if ks:
            L.append(f"  {lab:10s} n {len(ks)}: kink p10/p50/p90 {np.percentile(kinks, [10, 50, 90]) if kinks else '-'}; kink < 30 (shower-admitted) {sum(1 for x in kinks if x < 30)}; "
                     f"len+far p50/p90 {np.percentile(lens, [50, 90]) if lens else '-'}; KE p10/p50 {np.percentile(kes, [10, 50])}; KE < 5 MeV {sum(1 for x in kes if x < 5)}")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
