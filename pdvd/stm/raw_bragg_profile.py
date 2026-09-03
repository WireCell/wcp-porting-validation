#!/usr/bin/env python3
"""doc pdvd/25 sec 13.6 follow-up: is the missing Bragg rise in the CHARGE or in the FIT?

end_reach.py splits the accepted STM passes into those whose imaging charge really ends at
the fit end and those where the muon demonstrably continues.  BOTH have a flat fitted
dQ/dx at the stop (contrast median 0.9 / 1.1).  A muon whose charge ends must show a Bragg
rise, so either these are not stopping muons or the FIT's dQ/dx is under-read at the
terminus (dQ under-collected, or dx over-assigned, on the last points).

This measures the profile from the imaging charge alone: every Bee `clustering-global`
point within --veto cm of the fitted path is assigned to its nearest path point, and the
charge is summed per residual-range bin and divided by the path length in that bin.  No
fit dQ or dx enters.  Comparing this contrast with the fit's answers the question.

Repro:
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
  python3 stm/raw_bragg_profile.py --tag stm2 --out stm/raw_bragg_stm2.tsv
"""
import argparse, glob, json, os, re, sys, zipfile
import numpy as np, uproot
HERE = os.path.dirname(os.path.abspath(__file__)); PDVD = os.path.dirname(HERE); sys.path.insert(0, HERE)
import collect_stm_sample as C

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="stm2")
ap.add_argument("--out")
ap.add_argument("--veto", type=float, default=2.5, help="cm; charge within this of the path belongs to the track")
ap.add_argument("--max-absx", type=float, default=305.0)
a = ap.parse_args()

def bee_points(workdir):
    z = zipfile.ZipFile(os.path.join(workdir, "mabc-pr.zip"))
    for n in z.namelist():
        if n.endswith("-clustering-global.json"):
            d = json.loads(z.read(n))
            return (np.column_stack([d["x"], d["y"], d["z"]]).astype(float), np.asarray(d["q"], float))
    raise RuntimeError("no clustering-global layer in " + workdir)

BINS = [(0, 2), (20, 40)]
# Negative control: the same estimator run from the ENTRY end (rr = max).  Assigning every
# nearby charge point to its NEAREST path point piles charge that lies beyond a terminus
# into that terminus' bin, at BOTH ends -- so an entry-end contrast near 1 is what licenses
# reading the stop-end contrast as a Bragg rise (see feedback: a guard needs a causal
# negative control).
rows = []
for wd in sorted(glob.glob(os.path.join(PDVD, "work", f"*_{a.tag}"))):
    fp = os.path.join(wd, "tracking-stm.root")
    m = re.match(r"^(\d{6})_(\d+)", os.path.basename(wd))
    if not os.path.exists(fp) or not m or not os.path.exists(os.path.join(wd, "mabc-pr.zip")):
        continue
    run, idx = m.groups()
    try:
        f = uproot.open(fp); P, Q = bee_points(wd)
    except Exception:
        continue
    t = f["T_rec_charge"].arrays(["x", "y", "z", "q", "nq", "rr", "ndf", "status"], library="np")
    if len(t["ndf"]) == 0:
        continue
    tr = f["Trun"].arrays(["dQdx_scale", "dQdx_offset"], library="np")
    vd = C.read_verdicts(os.path.join(wd, f"wct_pr_{run}_{idx}.log"))
    for b in sorted(set(t["ndf"].tolist())):
        cid = int(b) // 10
        v = vd.get(cid, {}); mk = t["ndf"] == b
        if v.get("stm") != 1 or v.get("tgm") == 1 or int(t["status"][mk][0]) != 0 or mk.sum() < 40:
            continue
        rr = t["rr"][mk]
        pts = np.stack([t["x"][mk], t["y"][mk], t["z"][mk]], 1)
        o = np.argsort(rr); rr = rr[o]; pts = pts[o]
        if abs(pts[0][0]) > a.max_absx or rr.max() < 42:
            continue
        # nearest fitted path point for every Bee point within the veto tube
        near = np.linalg.norm(P[:, None, :] - pts[None, :, :], axis=2) if len(P) * len(pts) < 4e7 else None
        if near is None:
            continue
        j = np.argmin(near, axis=1); dmin = near[np.arange(len(P)), j]
        sel = dmin <= a.veto
        if sel.sum() < 20:
            continue
        rrp = rr[j[sel]]; qp = Q[sel]
        # fit dQ/dx for the same track (the sec 13.6 census observable)
        dQ = (t["q"][mk][o] - tr["dQdx_offset"][0]) / tr["dQdx_scale"][0]
        dx = t["nq"][mk][o]
        fit_dqdx = np.where(dx > 0, dQ / np.maximum(dx, 1e-9), 0.0)
        rr_ent = rr.max() - rr          # residual range measured from the ENTRY end
        rrp_ent = rr.max() - rrp
        vals = {}
        ok = True
        for lo, hi in BINS:
            sm = (rrp >= lo) & (rrp < hi)
            path = float(np.sum(dx[(rr >= lo) & (rr < hi)]))
            se = (rrp_ent >= lo) & (rrp_ent < hi)
            path_e = float(np.sum(dx[(rr_ent >= lo) & (rr_ent < hi)]))
            if sm.sum() < 5 or path <= 0 or se.sum() < 5 or path_e <= 0:
                ok = False; break
            vals[(lo, hi)] = float(qp[sm].sum()) / path
            vals[("ent", lo, hi)] = float(qp[se].sum()) / path_e
            fs = (rr >= lo) & (rr < hi) & (fit_dqdx > 0)
            vals[("fit", lo, hi)] = float(np.median(fit_dqdx[fs])) if fs.sum() >= 3 else np.nan
        if not ok or not np.isfinite(vals.get(("fit", 0, 2), np.nan)) or not np.isfinite(vals.get(("fit", 20, 40), np.nan)):
            continue
        rows.append((f"{run}_{idx}", cid,
                     vals[(0, 2)], vals[(20, 40)], vals[(0, 2)] / vals[(20, 40)],
                     vals[("fit", 0, 2)], vals[("fit", 20, 40)], vals[("fit", 0, 2)] / vals[("fit", 20, 40)],
                     vals[("ent", 0, 2)] / vals[("ent", 20, 40)]))

r = np.array([x[2:] for x in rows], float)
print(f"accepted STM passes profiled from the imaging charge (tag {a.tag}): {len(rows)}")
if len(rows):
    craw, cfit = r[:, 2], r[:, 5]
    print(f"\n  RAW imaging charge/cm: rr<2 median {np.median(r[:,0]):.3g} e/cm; "
          f"20-40 cm median {np.median(r[:,1]):.3g} e/cm")
    print(f"  FIT dQ/dx:             rr<2 median {np.median(r[:,3]):.3g} e/cm; "
          f"20-40 cm median {np.median(r[:,4]):.3g} e/cm")
    print(f"\n  contrast from RAW charge: median {np.median(craw):.2f};  >=2x: {100*np.mean(craw>=2):.0f} %;  >=1.5x: {100*np.mean(craw>=1.5):.0f} %")
    print(f"  contrast from the FIT:    median {np.median(cfit):.2f};  >=2x: {100*np.mean(cfit>=2):.0f} %;  >=1.5x: {100*np.mean(cfit>=1.5):.0f} %")
    cent = r[:, 6]
    print(f"  contrast from RAW charge, ENTRY end (negative control): median {np.median(cent):.2f};  >=2x: {100*np.mean(cent>=2):.0f} %")
    print(f"  raw/fit contrast ratio median {np.median(craw/cfit):.2f}; stop/entry raw contrast median {np.median(craw/cent):.2f}")
if a.out:
    with open(a.out, "w") as fo:
        fo.write("event\tcluster\traw_near\traw_mid\traw_contrast\tfit_near\tfit_mid\tfit_contrast\traw_contrast_entry\n")
        for x in rows:
            fo.write("\t".join([x[0], str(x[1])] + [f"{v:.4g}" for v in x[2:]]) + "\n")
    print("\nwrote", a.out)
