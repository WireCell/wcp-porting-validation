#!/usr/bin/env python3
"""doc pdvd/105 sec 4 -- the companion-piece geometry the stock output does not carry (read-only).

Doc 104 sec 2 refuted four admission levers and sec 5 of doc 105 refutes the arm-profile one, but all of them act
on the ATTACHED path (conn 1).  Six of PDHD's nine Michel false positives are conn 2 (bridged) or conn 3 (charge
only), or have too few fitted points to profile: for those the tagger reached out to a companion PIECE, and the
only quantities that describe that reach -- d_stop, d_body and the gate code that dropped each rejected piece --
exist solely when survey_enable / publish_other_arms are on.  This script reads them from the diagnostic arm.

THE HYPOTHESIS.  CheckSTM_Michel.cxx:3752-3757 rejects a piece when d_body < d_stop: closer to the muon's body
than to its stop means a delta ray, not a Michel.  That is a BINARY test on a continuous quantity.  A real Michel
should sit far outside it (d_body >> d_stop, the piece hangs off the stop and nothing else); a fragment of a
mis-terminated muon should sit near the boundary.  So the candidate discriminator is the MARGIN, d_body / d_stop
or d_body - d_stop, not the sign.

ROLES (CheckSTM_Michel::add_points): 1 chain, 2 delta, 3 Michel object member, 4 residual/gamma-collect,
5 capture gamma, 6 survey (fitted but unclaimed, survey_enable), 7 other stop arm (publish_other_arms), 8 census.

TWO JOIN TRAPS, both found the hard way:
  * rej / d_stop / d_body are populated ONLY on survey rows (role 6).  Role-3 (the admitted Michel) and role-7
    (other stop arms) carry -1 in all three -- an "admitted piece d_stop" read off a role-3 row is not a distance,
    it is the unset sentinel.
  * a role-6 row's seg_id encodes the cluster the PIECE belongs to (cluster*1000 + graph_index), which for a
    companion is a DIFFERENT cluster from the stopping muon's.  The STM cluster is in the row's own cluster_id
    branch, so that is what the join must use.  Joining on seg_id // 1000 silently finds nothing.

rej codes are doc pdvd/53's table (CheckSTM_Michel.cxx:3625-3650): the first gate to drop the companion owns the
row; 9 = the survey reached it and neither the Michel nor the gamma stage was ever offered it; 10 = no stop vertex.
Note the survey radius here is 60 cm against the Michel stage's michel_dot_radius_cm of 15, so most role-6 rows are
pieces that were never candidates -- the ledger is a census of the neighbourhood, not of near misses.

READ sec 3 FIRST.  If d105_diag_neutrality.py reports the diagnostic arm is a different reconstruction, everything
here describes THAT arm: it characterises the population and can motivate a cut, but no number is quotable against
the doc 103 grade until the cut is re-run in the graded arm.

    python3 d105_companion_gates.py --arm d105hdiag > figs/105_companion_gates.txt
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uproot
import d104_michel_features as F

HEAD = ["cluster_id", "michel_found", "michel_conn_type", "michel_dis_cm", "michel_len", "michel_ke_best",
        "n_dots", "n_dot_clusters_unfit", "n_survey_segs", "n_survey_unfit", "n_other_published"]
PTS = ["q", "role", "seg_id", "cluster_id", "rej", "d_stop", "d_body"]
SENT = 1e7          # d_body sentinel when the chain is shorter than dot_body_exclusion_cm


def pieces(det, arm):
    """{key: dict(admitted=[...], rejected=[...], other=[...])} of per-piece (d_stop, d_body, rej) rows."""
    out = {}
    for d in sorted(glob.glob(f"{F.IMG}/{det}/work/*_{arm}")):
        ev = os.path.basename(d)[:-len(arm) - 1]
        try:
            f = uproot.open(f"{d}/tracking-pr.root")
            H = f["T_stm_michel"].arrays(HEAD, library="np")
            T = f["T_stm_michel_pts"].arrays(PTS, library="np")
        except Exception:
            continue
        cl = T["seg_id"] // 1000
        for i, cid in enumerate(H["cluster_id"]):
            cid = int(cid)
            k = f"{ev}/{cid}"
            rec = dict(head={c: float(H[c][i]) for c in HEAD[1:]}, admitted=[], survey=[], other=[])
            for role, bucket in ((3, "admitted"), (6, "survey"), (7, "other")):
                # role 3/7 are keyed by the STM cluster through seg_id; role 6 by its own cluster_id branch
                s = ((T["cluster_id"] == cid) if role == 6 else (cl == cid)) & (T["role"] == role)
                if not s.any():
                    continue
                for sid in np.unique(T["seg_id"][s]):
                    m = s & (T["seg_id"] == sid)
                    rec[bucket].append(dict(seg=int(sid), n=int(m.sum()),
                                            d_stop=float(np.median(T["d_stop"][m])),
                                            d_body=float(np.median(T["d_body"][m])),
                                            rej=int(np.median(T["rej"][m]))))
            out[k] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="pdhd")
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    D = F.load(a.det)
    tp, fp = F.split(D)
    P = pieces(a.det, a.arm)
    tpk = [k for k, _ in tp if k in P]
    fpk = [k for k, _ in fp if k in P]
    print(f"# doc pdvd/105 sec 4: companion-piece geometry from the diagnostic arm {a.arm} ({a.det})")
    print(f"# truth and population are the graded cells' ({D['cells'][0]} / {D['cells'][1]}); see sec 3 for whether")
    print(f"# this arm reconstructs the same.  Michel TP {len(tpk)} of {len(tp)}, FP {len(fpk)} of {len(fp)} matched.")
    print(f"# d_body = {SENT:.0e} is the sentinel for a chain shorter than dot_body_exclusion_cm (5 cm).")

    for lab, ks in (("TRUE Michels", tpk), ("FALSE positives", fpk)):
        conn = {}
        for k in ks:
            c = int(P[k]["head"]["michel_conn_type"])
            conn[c] = conn.get(c, 0) + 1
        print(f"\n== {lab}: {len(ks)}  by conn_type {dict(sorted(conn.items()))}")
        print(f"   {'key':16s} {'conn':>4s} {'dis_cm':>7s} {'nAdm':>4s} {'nSurv':>5s} {'nOther':>6s} | "
              f"nearest SURVEYED piece: d_stop / d_body / rej")
        for k in sorted(ks):
            h = P[k]["head"]
            sv = [r for r in P[k]["survey"] if r["d_stop"] >= 0]
            best = min(sv, key=lambda r: r["d_stop"]) if sv else None
            if best is None:
                near = "   (no surveyed piece with a distance)"
            elif best["d_body"] >= SENT:
                near = f"{best['d_stop']:8.2f} {'sentinel':>9} {best['rej']:4d}"
            else:
                near = f"{best['d_stop']:8.2f} {best['d_body']:9.2f} {best['rej']:4d}"
            print(f"   {k:16s} {int(h['michel_conn_type']):4d} {h['michel_dis_cm']:7.2f} {len(P[k]['admitted']):4d} "
                  f"{len(P[k]['survey']):5d} {len(P[k]['other']):6d} | {near}")

    # the survey gate codes on the pieces this arm did NOT admit, TP vs FP
    print("\n== surveyed pieces by gate code (role 6), summed over the class (doc pdvd/53 table)")
    for lab, ks in (("TRUE Michels", tpk), ("FALSE positives", fpk)):
        codes = {}
        for k in ks:
            for r in P[k]["survey"]:
                codes[r["rej"]] = codes.get(r["rej"], 0) + 1
        n = sum(codes.values())
        print(f"   {lab:16s} {n:4d} rejected pieces  {dict(sorted(codes.items()))}")


if __name__ == "__main__":
    main()
