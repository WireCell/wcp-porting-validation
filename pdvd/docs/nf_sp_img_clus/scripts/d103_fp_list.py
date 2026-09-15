#!/usr/bin/env python3
"""doc pdvd/103 sec 6 -- the already-labelled items whose tag changes from production to both-on.  Read-only.

For is_stm and michel_found, on the graders' fixed populations (see d103_bounds.py / d101_stm_grade_*.py):
  NEW_FP   hand negative, tagged only in the arm      -> the owner adjudicates these (purity cost)
  LOST_TP  hand positive, tagged only in the base    -> with the reason: absent (TaggerCheckSTM did not pass it;
           its lowest-pass status in the arm from the job log) or the CheckSTM_Michel reject bits it gained
  GONE_FP  hand negative, tagged only in the base
  NEW_TP   hand positive, tagged only in the arm
Each row carries the label source (owner / agent) and confidence, so a cost carried by agent-only labels is visible.

Michel truth: PDHD hand stoppers, michel_kind in (attached, both) (d21_michel_census precedence); PDVD verdict
STM_MICHEL on every judged item (census_lib).

Usage: d103_fp_list.py --det pdhd --base d101hnew --arm d102hcs > figs/103_moves_pdhd.tsv
"""
import argparse, csv, glob, json, os, re, sys
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
BITS = ["no_chain", "stop_unmatched", "no_bragg", "shape_flat", "not_muon_pid", "continuation", "stop_near_boundary",
        "vertex_hadron", "short", "profile_sparse", "plateau_off_mip", "stop_into_dead", "cluster_not_track",
        "profile_geometry"]
R_PASS = re.compile(r"persist_stm_fit: cluster (\d+) stmfit pass=(\d+) status=(-?\d+)")


def strip(v):
    return v[5:] if v and v.startswith("FRAG_") else v


def rows(det, arm):
    out = {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            t = uproot.open(f"{d}/tracking-pr.root")["T_stm_michel"].arrays(
                ["cluster_id", "is_stm", "michel_found", "reject_bits"], library="np")
        except Exception:
            continue
        for c, s, m, rb in zip(t["cluster_id"], t["is_stm"], t["michel_found"], t["reject_bits"]):
            out[f"{ev}/{int(c)}"] = (int(s), int(m), {n for i, n in enumerate(BITS) if int(rb) >> i & 1})
    return out


def status(det, arm):
    out = {}
    for d in glob.glob(f"{IMG}/{det}/work/*_{arm}"):
        ev = os.path.basename(d)[:-len(arm) - 1]
        for lg in glob.glob(f"{d}/wct_pr_*.log")[:1]:
            for line in open(lg, errors="replace"):
                m = R_PASS.search(line)
                if m:
                    k = f"{ev}/{m.group(1)}"
                    out[k] = min(out.get(k, (99, 0)), (int(m.group(2)), int(m.group(3))))
    return {k: v[1] for k, v in out.items()}


def truth(det, base_rows):
    """key -> (is_stopper, is_michel or None, source, confidence)"""
    T = {}
    if det == "pdhd":
        X = f"{IMG}/pdhd/docs/scan"
        pop = {"%s/%s" % (r["event"], r["cluster"]) for r in csv.DictReader(
            [l for l in open(f"{X}/smx18/pdhd_stm_michel_scan_key_p82bhoff.tsv") if not l.startswith("#")], delimiter="\t")}
        for r in json.load(open(f"{X}/pdhd_stm_michel_smx22_verdicts.json")):
            if r["key"] not in pop:
                continue
            if r.get("owner_review"):
                v, mk, src = strip(r["owner_review"]["verdict"]), r["owner_review"].get("michel_kind"), "owner_review"
            elif r.get("owner_smx1"):
                o = r["owner_smx1"]
                v, mk, src = strip(o.get("choice") or o.get("label")), o.get("michel_kind"), "owner"
            else:
                v, mk, src = strip(r["verdict"]), r.get("michel_kind"), "agent"
            if v in ("MESSY", "UNCLEAR"):
                continue
            st = v in ("STM_MICHEL", "STM_ONLY")
            mi = (mk in ("attached", "both")) if (st and mk is not None) else None
            T[r["key"]] = (st, mi, src, r.get("confidence", ""))
    else:
        import census_lib as C
        for k, r in C.load_record().items():
            if C.judged(r) and k in base_rows:
                T[k] = (C.is_stopper(r), C.is_michel(r), "record", r.get("confidence", ""))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, choices=["pdhd", "pdvd"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    B, A = rows(a.det, a.base), rows(a.det, a.arm)
    SA = status(a.det, a.arm)
    T = truth(a.det, B)
    print("# doc pdvd/103: tag moves on already-labelled items, %s -> %s (%s)" % (a.base, a.arm, a.det))
    print("\t".join(["chain", "move", "key", "label_source", "confidence", "arm_candidate", "arm_tagger_status",
                     "arm_reject_bits_new"]))
    summary = {}
    for chain, idx in (("is_stm", 0), ("michel_found", 1)):
        for k in sorted(T):
            st, mi, src, conf = T[k]
            t = st if chain == "is_stm" else mi
            if t is None:
                continue
            gb = bool(B.get(k, (0, 0, set()))[idx])
            ga = bool(A.get(k, (0, 0, set()))[idx])
            if gb == ga:
                continue
            move = ("LOST_TP" if gb else "NEW_TP") if t else ("NEW_FP" if ga else "GONE_FP")
            summary[(chain, move)] = summary.get((chain, move), 0) + 1
            cand = "yes" if k in A else "no"
            bits = ",".join(sorted(A[k][2] - B.get(k, (0, 0, set()))[2])) if k in A else ""
            print("\t".join([chain, move, k, src, str(conf), cand, str(SA.get(k, "none")), bits]))
    for (chain, move), n in sorted(summary.items()):
        print(f"# {chain} {move}: {n}")


if __name__ == "__main__":
    main()
