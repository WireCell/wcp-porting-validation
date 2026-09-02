#!/usr/bin/env python3
"""Tagger-independent Michel test on raw imaging charge (doc pdvd/25 sec 13.7).

For every Bee `clustering-global` cluster of the arm with >= --min-pts points
and a principal-axis extent >= --min-len cm: project the points on the axis,
take the charge density (e/cm) in the last --end cm at each end vs the middle
30-70 % of the track; an end with density ratio >= --rise is a raw Bragg
end.  Clusters with exactly ONE rising end (the other < --flat) are raw
stopper candidates; the flat end is the built-in control.  At each end the
Michel activity is the same-cluster charge within --radius cm of the end
point that lies farther than --veto cm from the axis line (the muon's own
charge), converted with the prototype shower convention (23.6 eV / 0.5 / 0.8).
"""
import argparse, glob, json, os, re, sys, zipfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PDVD = os.path.dirname(HERE)
E_PER_E = 23.6e-6 / 0.5 / 0.8; ENDPOINT = 52.8

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="stm2"); ap.add_argument("-o", "--out", required=True); ap.add_argument("--tsv")
    ap.add_argument("--min-pts", type=int, default=300); ap.add_argument("--min-len", type=float, default=100.0); ap.add_argument("--end", type=float, default=5.0)
    ap.add_argument("--rise", type=float, default=2.0); ap.add_argument("--flat", type=float, default=1.3); ap.add_argument("--radius", type=float, default=12.0); ap.add_argument("--veto", type=float, default=2.5)
    ap.add_argument("--max-absx", type=float, default=305.0)
    a = ap.parse_args(); rows = []; ncl = 0; nlong = 0
    for wd in sorted(glob.glob(os.path.join(PDVD, "work", f"*_{a.tag}"))):
        zp = os.path.join(wd, "mabc-pr.zip")
        if not os.path.exists(zp): continue
        try:
            z = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            print("skip (bad/partial zip):", zp, file=sys.stderr); continue
        names = [n for n in z.namelist() if n.endswith("-clustering-global.json")]
        if not names: continue
        d = json.loads(z.read(names[0])); P = np.column_stack([d["x"], d["y"], d["z"]]).astype(float); Q = np.asarray(d["q"], float); cid = np.asarray(d["cluster_id"], int)
        ev = re.sub(r"_%s$" % a.tag, "", os.path.basename(wd))
        for c in np.unique(cid):
            m = cid == c
            if m.sum() < a.min_pts: continue
            ncl += 1; X = P[m]; q = Q[m]; mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); ax = Vt[0]
            t = (X - mu) @ ax; t0, t1 = t.min(), t.max(); L = t1 - t0
            if L < a.min_len: continue
            nlong += 1
            perp = np.linalg.norm((X - mu) - np.outer(t, ax), axis=1)
            ontrack = perp <= a.veto
            mid = ontrack & (t > t0 + 0.3 * L) & (t < t0 + 0.7 * L); dmid = q[mid].sum() / (0.4 * L) if mid.sum() else 0.0
            if dmid <= 0: continue
            res = {}
            for side, sel_end, tend in (("lo", ontrack & (t < t0 + a.end), t0), ("hi", ontrack & (t > t1 - a.end), t1)):
                dens = q[sel_end].sum() / a.end; endpt = mu + ax * tend
                near = np.linalg.norm(X - endpt, axis=1) <= a.radius; off = near & (perp > a.veto)
                res[side] = dict(ratio=dens / dmid, E_off=float(q[off].sum() * E_PER_E), n_off=int(off.sum()), x=float(endpt[0]), endpt=endpt)
            r_lo, r_hi = res["lo"]["ratio"], res["hi"]["ratio"]
            if r_lo >= a.rise and r_hi < a.flat: rise, flat = "lo", "hi"
            elif r_hi >= a.rise and r_lo < a.flat: rise, flat = "hi", "lo"
            else: continue
            if abs(res[rise]["x"]) > a.max_absx or abs(res[flat]["x"]) > a.max_absx: continue   # both ends away from the CRP band
            rows.append(dict(event=ev, cluster=int(c), L=float(L), npts=int(m.sum()), rise_ratio=res[rise]["ratio"], flat_ratio=res[flat]["ratio"],
                             E_stop=res[rise]["E_off"], E_ctrl=res[flat]["E_off"], n_stop=res[rise]["n_off"], n_ctrl=res[flat]["n_off"], stop_x=res[rise]["x"], stop_y=float(res[rise]["endpt"][1]), stop_z=float(res[rise]["endpt"][2])))
    Es = np.array([r["E_stop"] for r in rows]); Ec = np.array([r["E_ctrl"] for r in rows])
    print(f"clusters >= {a.min_pts} pts: {ncl}; >= {a.min_len} cm: {nlong}; ONE raw Bragg end (>= {a.rise}x, other < {a.flat}x, both ends |x| <= {a.max_absx}): {len(rows)}")
    if len(rows):
        print(f"  E_off at the Bragg end: median {np.median(Es):.1f} MeV, > 5 MeV {np.mean(Es > 5):.2f}, in (5,60] {np.mean((Es > 5) & (Es <= 60)):.2f}, > 60 {np.mean(Es > 60):.2f}")
        print(f"  E_off at the flat end (control): median {np.median(Ec):.1f} MeV, > 5 MeV {np.mean(Ec > 5):.2f}, in (5,60] {np.mean((Ec > 5) & (Ec <= 60)):.2f}, > 60 {np.mean(Ec > 60):.2f}")
        print(f"  stop > control by > 5 MeV: {np.mean(Es - Ec > 5):.2f}; control > stop by > 5 MeV: {np.mean(Ec - Es > 5):.2f}")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write(f"# raw stopper candidates, tag {a.tag}: one end density >= {a.rise}x the middle (last {a.end} cm), other end < {a.flat}x; E_off = same-cluster charge within {a.radius} cm of the end, > {a.veto} cm off the axis, x 23.6 eV/0.5/0.8\n")
            f.write("event\tcluster\tL_cm\tnpts\trise_ratio\tflat_ratio\tE_stop_MeV\tE_ctrl_MeV\tn_stop\tn_ctrl\tstop_x\tstop_y\tstop_z\n")
            for r in rows: f.write("\t".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in (r["event"], r["cluster"], r["L"], r["npts"], r["rise_ratio"], r["flat_ratio"], r["E_stop"], r["E_ctrl"], r["n_stop"], r["n_ctrl"], r["stop_x"], r["stop_y"], r["stop_z"])) + "\n")
        print("wrote", a.tsv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); bins = np.arange(0, 121, 4)
    ax[0].hist(Es, bins=bins, color="tab:red", alpha=0.6, label=f"raw Bragg end (n={len(Es)})"); ax[0].hist(Ec, bins=bins, histtype="step", color="k", lw=1.5, label="flat end (control)")
    ax[0].axvline(ENDPOINT, color="k", ls="--", label="52.8 MeV"); ax[0].set_xlabel(f"off-axis charge within {a.radius:.0f} cm of the end [MeV, shower convention]"); ax[0].set_ylabel("clusters"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
    ax[1].scatter(Ec, Es, s=10, alpha=0.6); ax[1].plot([0, 120], [0, 120], "k:", lw=1); ax[1].set_xlim(0, 120); ax[1].set_ylim(0, 120); ax[1].set_xlabel("flat end [MeV]"); ax[1].set_ylabel("Bragg end [MeV]"); ax[1].grid(alpha=0.2)
    fig.suptitle(f"PDVD raw stopper candidates ({a.tag}): Michel activity at the Bragg end vs the flat end", fontsize=10); fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)
main()
