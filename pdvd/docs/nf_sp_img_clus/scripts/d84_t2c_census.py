#!/usr/bin/env python3
"""doc pdvd/84 -- doc 78 action item 3: re-grade T2c (moved_stop_michel_guard) on
the current record, and census every T2c fire across the PDVD arms on disk.

Read-only.  Reads T_stm_michel from pdvd/work/<run>_<evt>_<arm>/tracking-pr.root
and the owner's record (smx1a + smx3 + smx4 verdicts).

  sec 1  T2c on the production arm (default p83vprod): every veto / exemption,
         by item, with the record's verdict.
  sec 2  every distinct T2c instance (item, kink, KE, len, far, is_stm) across
         the arm list, with the arms it appears on.
  sec 3  for each candidate separator (kink, KE, reach = len + far_len): the
         THRU range against the STM_MICHEL range over all instances, and the
         open window a threshold would have to sit in.

Usage: d84_t2c_census.py [--prod p83vprod] [--arms "a b c"]
"""
import argparse, collections, glob, json, os
import uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
REC = IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json"
ARMS = ("d67v d68a3 d68d4 p79vcap p79vprod p80bcen p80boff p80vcen p80voff p81vleg p81voff "
        "p81vq2d p82bcs p82bk10 p82boff p82btp p82btp0 p82vcs p82vk10 p82voff p82vprod p82vtp "
        "p82vtp0 p83v10 p83v5 p83v7 p83voff p83vprod")
BR = ["cluster_id", "n_michel_veto", "michel_kink_deg", "michel_ke_best", "michel_len",
      "michel_far_len", "is_stm", "michel_found", "n_retreat", "n_split", "reject_bits"]


def read(arm):
    out = {}
    for d in sorted(glob.glob(IMG + "/pdvd/work/*_" + arm)):
        fn = d + "/tracking-pr.root"
        if not os.path.exists(fn):
            continue
        ev = os.path.basename(d)[: -len(arm) - 1]
        try:
            f = uproot.open(fn)
            if "T_stm_michel" not in f:
                continue
            tr = f["T_stm_michel"]
            ex = "n_michel_veto_exempt" in tr.keys()
            t = tr.arrays(BR + (["n_michel_veto_exempt"] if ex else []), library="np")
        except Exception:
            continue
        for i in range(len(t["cluster_id"])):
            r = {b: t[b][i] for b in BR}
            r["n_michel_veto_exempt"] = int(t["n_michel_veto_exempt"][i]) if ex else 0
            out["%s/%d" % (ev, t["cluster_id"][i])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default="p83vprod")
    ap.add_argument("--arms", default=ARMS)
    a = ap.parse_args()
    R = {v["key"]: v for v in json.load(open(REC))}
    verdict = lambda k: R.get(k, {}).get("verdict", "unjudged")

    print("=== 1. T2c on the production arm %s" % a.prod)
    P = read(a.prod)
    n = 0
    for k, r in sorted(P.items()):
        if int(r["n_michel_veto"]) or int(r["n_michel_veto_exempt"]):
            n += 1
            print("  %-14s %-11s %s  kink %6.2f deg  KE %5.2f MeV  len %4.1f + far %4.1f cm  retreat %d split %d  is_stm %d  michel_found %d  bits %d" % (
                k, verdict(k), "VETO  " if int(r["n_michel_veto"]) else "EXEMPT",
                r["michel_kink_deg"], r["michel_ke_best"], r["michel_len"], r["michel_far_len"],
                int(r["n_retreat"]), int(r["n_split"]), int(r["is_stm"]), int(r["michel_found"]), int(r["reject_bits"])))
    print("  %d of %d candidates" % (n, len(P)))

    print("\n=== 2. every distinct T2c instance across %d arms" % len(a.arms.split()))
    inst = collections.defaultdict(list)
    for arm in a.arms.split():
        for k, r in read(arm).items():
            v, x = int(r["n_michel_veto"]), int(r["n_michel_veto_exempt"])
            if not (v or x):
                continue
            key = (k, round(float(r["michel_kink_deg"]), 2), round(float(r["michel_ke_best"]), 2),
                   round(float(r["michel_len"]), 1), round(float(r["michel_far_len"]), 1), int(r["is_stm"]))
            inst[key].append((arm, "V" if v else "X"))
    for key in sorted(inst):
        k, kink, ke, ln, far, s = key
        arms = inst[key]
        print("  %-14s %-11s kink %6.2f  KE %5.2f  len %4.1f far %4.1f reach %5.1f  is_stm %d  %s x%-2d %s" % (
            k, verdict(k), kink, ke, ln, far, ln + far, s, "/".join(sorted({f for _, f in arms})),
            len(arms), " ".join(x for x, _ in arms[:5]) + (" ..." if len(arms) > 5 else "")))
    print("  %d instances on %d items" % (len(inst), len({key[0] for key in inst})))

    print("\n=== 3. separators over every instance: THRU range vs STM_MICHEL range")
    for name, f in (("kink (deg)", lambda key: key[1]), ("KE (MeV)", lambda key: key[2]),
                    ("reach = len + far (cm)", lambda key: key[3] + key[4])):
        thru = [f(key) for key in inst if verdict(key[0]) == "THRU"]
        sig = [f(key) for key in inst if verdict(key[0]) == "STM_MICHEL"]
        ok = max(thru) < min(sig)
        print("  %-24s THRU %6.2f..%6.2f  STM_MICHEL %6.2f..%6.2f  -> %s" % (
            name, min(thru), max(thru), min(sig), max(sig),
            "separates, window (%.2f, %.2f]" % (max(thru), min(sig)) if ok else "DOES NOT separate"))
    other = sorted({verdict(key[0]) for key in inst} - {"THRU", "STM_MICHEL"})
    print("  verdicts other than THRU / STM_MICHEL among the instances: %s" % (other or "none"))


if __name__ == "__main__":
    main()
