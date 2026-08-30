#!/usr/bin/env python3
"""doc pr/132 round 5 -- the per-gamma charge ledger (EM-clustering sizing).

Borrows the pr126 selector like pr132_pi0_census.py.  For every hand pi0
gamma (base + overlay labels), compare the label energy against the matched
reco shower's kine_charge and against the charge of collinear sibling
showers (within CONE deg of the label axis from the label start), then
classify the EM-clustering defect:

  OK           0.80 <= ratio <= 1.25
  UNDER+SIBS   ratio < 0.80 and collinear siblings close >= 50% of the deficit
               -> build-time merge target (the in-scope EM clustering fix)
  UNDER-nosibs ratio < 0.80, deficit NOT in collinear siblings
               -> missing/unclustered charge (imaging / charge recovery)
  OVER         ratio > 1.25   -> over-merge (split target)
  ABSENT       no matched shower on the arm
"""
import argparse, csv, math, os, sys, importlib.util
from collections import Counter

_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(os.path.dirname(__file__), "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

CONE = 20.0
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest141"); ap.add_argument("--manifest98")
    ap.add_argument("--overlay-tag"); ap.add_argument("--tsv")
    a = ap.parse_args()
    if a.manifest98 or a.manifest141:
        newsets = []
        for t in SEL.SETS:
            t = list(t)
            if t[0] == "98" and a.manifest98: t[4] = a.manifest98
            if t[0] == "141" and a.manifest141: t[4] = a.manifest141
            newsets.append(tuple(t))
        SEL.SETS = newsets
    overlay = SEL.load_labels(a.overlay_tag) if a.overlay_tag else {}

    rows = []
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            dump = SEL.load_json(mrow["dump"])
            if not dump: continue
            for labsrc, rec in (("base", labels.get(ev)), ("overlay", overlay.get(ev))):
                g = ((rec or {}).get("pio") or {}).get("gammas")
                if not g or not all(x in g and (g[x].get("energy") or 0) > 0 for x in ("1","2")):
                    continue
                showers = dump.get("showers") or []
                by = {int(s["id"]): s for s in showers}
                for gi in ("1","2"):
                    lab = g[gi]
                    e_lab = lab["energy"]
                    axis = lab.get("axis"); start = lab.get("start") or lab.get("reco_start")
                    sh = by.get(int(lab.get("shower") or -1))
                    if sh is None:
                        rows.append(dict(setname=setname, sample=mrow.get("sample",tag), event=ev,
                                         labelsrc=labsrc, gamma=gi, e_lab=round(e_lab,1),
                                         e_sh=-1, ratio=-1, e_sibs=0, nsibs=0, klass="ABSENT"))
                        continue
                    e_sh = sh.get("kine_charge") or 0.0
                    ratio = e_sh / e_lab if e_lab > 0 else -1
                    e_sibs = 0.0; nsibs = 0
                    if axis and start:
                        am = math.sqrt(sum(c*c for c in axis)) or 1.0
                        for s2 in showers:
                            if s2 is sh or int(s2["id"]) == int(lab.get("shower") or -1): continue
                            st2 = s2.get("start") or {}
                            v = (st2.get("x",0)-start[0], st2.get("y",0)-start[1], st2.get("z",0)-start[2])
                            vm = math.sqrt(sum(c*c for c in v))
                            if vm <= 0.5 or vm > 120: continue
                            cosang = sum(v[i]*axis[i] for i in range(3)) / (vm*am)
                            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                            if ang < CONE:
                                e_sibs += s2.get("kine_charge") or 0.0; nsibs += 1
                    if ratio > 1.25: klass = "OVER"
                    elif ratio >= 0.80: klass = "OK"
                    else:
                        deficit = e_lab - e_sh
                        klass = "UNDER+SIBS" if (deficit > 0 and e_sibs >= 0.5*deficit) else "UNDER-nosibs"
                    rows.append(dict(setname=setname, sample=mrow.get("sample",tag), event=ev,
                                     labelsrc=labsrc, gamma=gi, e_lab=round(e_lab,1),
                                     e_sh=round(e_sh,1), ratio=round(ratio,2),
                                     e_sibs=round(e_sibs,1), nsibs=nsibs, klass=klass))
    cnt = Counter(r["klass"] for r in rows)
    n = len(rows)
    print("=== per-gamma charge ledger (%d gammas) ===" % n)
    for k in ("OK","UNDER+SIBS","UNDER-nosibs","OVER","ABSENT"):
        print("  %-13s %3d  %.1f%%" % (k, cnt[k], 100.0*cnt[k]/n if n else 0))
    print("\n=== the defect rows ===")
    for r in rows:
        if r["klass"] != "OK":
            print("  %(setname)s %(event)-7s %(labelsrc)-7s g%(gamma)s %(klass)-13s e_lab=%(e_lab)7.1f e_sh=%(e_sh)7.1f ratio=%(ratio)5.2f sibs=%(e_sibs)7.1f (n=%(nsibs)d)" % r)
    if a.tsv:
        with open(a.tsv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (a.tsv, len(rows)))

if __name__ == "__main__":
    main()
