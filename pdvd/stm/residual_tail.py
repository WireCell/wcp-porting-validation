#!/usr/bin/env python3
"""doc pdvd/25 sec 13.7 follow-up: the fit's residual tail is where the tagger already
puts the Michel candidate.

eval_stm_core_impl anchors the stop at the Bragg peak it finds (end_L) and calls
everything past it the residual; its own veto chain is written "If residual does not
look like a michel electron" (TaggerCheckSTM.cxx:2795), i.e. a SHORT, SOFT residual is
allowed through as the Michel of an accepted stopping muon.  Sec 13.7's four routes all
anchored on the fit PATH END -- a median 13.8 cm PAST that stop (stm_eval_anchor.py) --
so they searched beyond the candidate, which is consistent with the null result and with
the entry-end control matching.

This script measures the tail itself: length, charge, collinearity with the muon, and the
sec-13.7 energy conversion E = Q x 23.6 eV / 0.5 (shower recomb) / 0.8 (fudge).
Controls: the same observable on TGM-tagged passes (through-going: no Michel exists) and
the tail-length/charge split for straight (muon continuation) vs kinked (Michel-like) tails.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 stm/residual_tail.py --tag stm2 --out stm/residual_tail_stm2.tsv
"""
import argparse, os, sys, glob, re
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_stm_sample as C

E_PER_E = 23.6e-6 / 0.5 / 0.8      # MeV per electron, sec 13.7 convention

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="stm2")
ap.add_argument("--out", default=None)
ap.add_argument("--max-tail", type=float, default=60.0, help="cm; longer tails are muon continuation, not a Michel")
a = ap.parse_args()

rows = []
for wd in sorted(glob.glob(os.path.join(C.PDVD, "work", f"*_{a.tag}"))):
    fp = os.path.join(wd, "tracking-stm.root")
    if not os.path.exists(fp):
        continue
    m = re.match(r"^(\d{6})_(\d+)", os.path.basename(wd))
    if not m:
        continue
    run, idx = m.group(1), m.group(2)
    try:
        f = uproot.open(fp)
    except Exception:
        continue
    if "T_rec_charge" not in f or "T_stm_eval" not in f:
        continue
    t = f["T_rec_charge"].arrays(["x", "y", "z", "q", "nq", "rr", "ndf", "status"], library="np")
    if len(t["ndf"]) == 0:
        continue
    tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    ee = f["T_stm_eval"].arrays(library="np")
    verdicts = C.read_verdicts(os.path.join(wd, f"wct_pr_{run}_{idx}.log"))
    res_of = {}
    for i in range(len(ee["verdict"])):
        if ee["verdict"][i] == 1:
            res_of[(int(ee["cluster_id"][i]), int(ee["pass"][i]))] = (float(ee["res_length"][i]),
                                                                     float(ee["ave_res_dqdx"][i]))
    for b in sorted(set(t["ndf"].tolist())):
        cid, pss = int(b) // 10, int(b) % 10
        v = verdicts.get(cid, {})
        kind = None
        if v.get("stm") == 1 and v.get("tgm") != 1 and int(t["status"][t["ndf"] == b][0]) == 0:
            kind = "STM"
        elif v.get("tgm") == 1:
            kind = "TGM"
        if kind is None or (cid, pss) not in res_of:
            continue
        res, aveq = res_of[(cid, pss)]
        mk = t["ndf"] == b
        rr = t["rr"][mk]
        dQ = (t["q"][mk] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
        x, y, z = t["x"][mk], t["y"][mk], t["z"][mk]
        tail = rr < res                         # points past the tagger's stop
        n = int(tail.sum())
        if n < 2:
            rows.append(dict(event=f"{run}_{idx}", cluster=cid, kind=kind, res=res, n=n,
                             E=0.0, straight=float("nan"), aveq=aveq))
            continue
        p = np.stack([x[tail], y[tail], z[tail]], 1)
        seglen = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        chord = float(np.linalg.norm(p[-1] - p[0]))
        rows.append(dict(event=f"{run}_{idx}", cluster=cid, kind=kind, res=res, n=n,
                         E=float(np.sum(dQ[tail])) * E_PER_E,
                         straight=chord / (seglen + 1e-9), aveq=aveq))

for kind in ("STM", "TGM"):
    r = [x for x in rows if x["kind"] == kind]
    if not r:
        continue
    res = np.array([x["res"] for x in r]); E = np.array([x["E"] for x in r])
    st = np.array([x["straight"] for x in r]); n = np.array([x["n"] for x in r])
    sel = (res > 0.5) & (res < a.max_tail) & (n >= 3)
    print(f"\n=== {kind}: {len(r)} passes with an accepting eval record; "
          f"tail 0.5-{a.max_tail:.0f} cm with >=3 points: {int(sel.sum())} ===")
    print(f"  res_length median {np.median(res):.1f} cm; tail energy median {np.median(E[sel]):.1f} MeV" if sel.sum() else "")
    if not sel.sum():
        continue
    print("  tail energy [MeV]: " + "  ".join(
        f"{lo}-{hi}: {int(np.sum((E[sel]>=lo)&(E[sel]<hi)))}"
        for lo, hi in [(0, 10), (10, 20), (20, 35), (35, 55), (55, 80), (80, 1e9)]))
    kinked = sel & (st < 0.9)
    print(f"  collinear (chord/arclen >= 0.9, muon-continuation-like): {int(np.sum(sel & (st>=0.9)))}; "
          f"kinked/diffuse (< 0.9, Michel-like): {int(kinked.sum())}")
    if kinked.sum():
        print("  kinked tail energy [MeV]: " + "  ".join(
            f"{lo}-{hi}: {int(np.sum((E[kinked]>=lo)&(E[kinked]<hi)))}"
            for lo, hi in [(0, 10), (10, 20), (20, 35), (35, 55), (55, 80), (80, 1e9)]))
        print(f"  kinked median {np.median(E[kinked]):.1f} MeV, "
              f"fraction above the 52.8 MeV endpoint: {100*np.mean(E[kinked] > 52.8):.0f} %")

if a.out:
    with open(a.out, "w") as fo:
        fo.write("event\tcluster\tkind\tres_len_cm\tn_tail\tE_tail_MeV\tstraightness\tave_res_dqdx\n")
        for r in rows:
            fo.write(f"{r['event']}\t{r['cluster']}\t{r['kind']}\t{r['res']:.2f}\t{r['n']}\t{r['E']:.2f}\t{r['straight']:.3f}\t{r['aveq']:.0f}\n")
    print("\nwrote", a.out)
