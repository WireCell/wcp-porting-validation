#!/usr/bin/env python3
"""doc pdvd/99 -- the re-scan list for a carried record: every item a hand-scan verdict cannot be carried onto blindly.

    python3 d99_rescan.py --match <d99_match.py --out json> --arm p98von --tsv ../d99/rescan_p98von.tsv

An item is flagged (one row, all reasons joined) when
  match      status != ok                       (lost / split / merged / ambiguous / collision / no_arm_event)
  candidate  no T_stm_michel row for the matched cluster on the new arm
  stop       the new arm's stop moved > 5 cm from p96vprod's (x offset applied)
  t0         the matched cluster moved > 1 cm in x (a changed flash match)
  class      the verdict is MESSY / UNCLEAR / FRAG_*   (the call was about how reco assembled the object)
  michel_tag a michel / gamma tag whose cluster or segment did not remap
  pin        a placed pin or pin_rr now lies > 3 cm from the new arm's fit
Rows carry the original key, the new key, the volume (p96vprod stop_x sign), the verdict and the match numbers.
"""
import argparse, collections, json

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
REC = IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--record", default=REC)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--baseline", default=None, help="the same list on the identity match (p96vprod); reason types"
                                                      " already flagged there are not new")
    a = ap.parse_args()
    R = {r["key"]: r for r in json.load(open(a.record))}
    BASE = collections.defaultdict(set)
    if a.baseline:
        for l in open(a.baseline):
            if l.startswith("#") or l.startswith("volume\t"):
                continue
            f = l.rstrip("\n").split("\t")
            BASE[f[1]] = {w.split(":")[0] for w in f[-1].split(";")}
    M = json.load(open(a.match))
    rows = []
    for o in M:
        r = R[o["key"]]
        why = []
        if o.get("status") != "ok":
            why.append("match:" + str(o.get("status")))
        if o.get("status") == "ok" and not o.get("candidate"):
            why.append("candidate:none")
        if (o.get("stop_move_cm") or 0) > 5.0:
            why.append("stop:%.1fcm" % o["stop_move_cm"])
        if o.get("t0_moved"):
            why.append("t0:dx%.1f" % o["dx"])
        v = r["verdict"]
        if v in ("MESSY", "UNCLEAR") or v.startswith("FRAG_"):
            why.append("class:" + v)
        for sid, t in (o.get("tags") or {}).items():
            # cluster level only: the scan arms carried the survey TLA (companion segments fitted), production does
            # not, so a Michel/gamma SEGMENT fails to remap even onto p96vprod itself (G4a: 65 items).  The cluster
            # is the unit the owner's tags and the chain's roles join on (memory: join scan tags at cluster level).
            if t.get("tag") in ("michel", "gamma") and t.get("new_cluster") is None:
                why.append(f"michel_tag:{sid}:{t.get('cluster_status')}")
        for nm in ("pin_dist_to_new_fit_cm", "pin_rr_dist_to_new_fit_cm"):
            if (o.get(nm) or 0) > 3.0:
                why.append("pin:%s=%.1f" % (nm.split("_dist")[0], o[nm]))
        if why:
            new = sorted({w.split(":")[0] for w in why} - BASE.get(o["key"], set())) if a.baseline else []
            rows.append((o.get("volume", "?"), o["key"], o.get("new_key") or "-", v, r.get("michel_kind") or "-",
                         o.get("status"), o.get("f_fwd"), o.get("f_rev"), o.get("dx"), o.get("stop_move_cm"),
                         ",".join(new) or "-", ";".join(why)))
    rows.sort()
    with open(a.tsv, "w") as f:
        f.write(f"# doc pdvd/99 re-scan list for arm {a.arm} (match {a.match.split('/')[-1]}); one row per flagged record item;"
                f" new_vs_identity = reason types not already flagged on the identity match{'' if a.baseline else ' (no baseline given)'}\n")
        f.write("volume\told_key\tnew_key\tverdict\tmichel_kind\tstatus\tf_fwd\tf_rev\tdx_cm\tstop_move_cm\tnew_vs_identity\treasons\n")
        for row in rows:
            f.write("\t".join("" if x is None else str(x) for x in row) + "\n")
    n = collections.Counter(r[0] for r in rows)
    reasons = collections.Counter(w.split(":")[0] for r in rows for w in r[-1].split(";"))
    print(f"{a.arm}: flagged {len(rows)} of {len(M)}  by volume {dict(n)}  by reason {dict(reasons)}")
    if a.baseline:
        nn = [r for r in rows if r[-2] != "-"]
        print(f"  NEW vs identity: {len(nn)} items  by volume {dict(collections.Counter(r[0] for r in nn))}"
              f"  by new reason {dict(collections.Counter(w for r in nn for w in r[-2].split(',')))}")
    hard = [r for r in rows if any(w.split(":")[0] in ("match", "candidate") for w in r[-1].split(";"))]
    print(f"  cannot be carried without a look (match/candidate): {len(hard)}  by volume {dict(collections.Counter(r[0] for r in hard))}"
          f"  by verdict {dict(collections.Counter(r[3] for r in hard))}")


if __name__ == "__main__":
    main()
