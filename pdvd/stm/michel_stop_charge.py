#!/usr/bin/env python3
"""Raw-charge Michel energy around each STM stop (doc pdvd/25 sec 13.7, ask 6).

The chain's shower finder rarely builds a shower at a PDVD stop
(michel_stop_end.py: 18 stubs on 255 stops), so this extractor measures the
Michel activity directly from the imaging charge: for every STM-tagged,
non-TGM, accepted STM pass (collect_stm_sample.iter_blocks -> fitted path
and its rr = 0 stop) it sums the Bee `clustering-global` point charge inside
a sphere of --radius cm around the stop, EXCLUDING points within --veto cm of
any fitted point of the muon's own path.  Energy follows the prototype's
cal_kine_charge convention for EM showers (NeutrinoID_energy_reco.h:83-88,248):
E = Q / R / f x 23.6 eV, R = 0.5, f = 0.8 -- uBooNE-field numbers, so the
absolute scale is a placeholder until PDVD's own recombination (M7) is fit.
The same sum at the ENTRY end (rr = max) of the same track is the negative
control: Michel activity must show at the stop, not at the entry.
"""
import argparse, json, os, sys, zipfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PDVD = os.path.dirname(HERE); sys.path.insert(0, HERE)
import collect_stm_sample as C
ENDPOINT = 52.8; E_PER_E = 23.6e-6 / 0.5 / 0.8   # MeV per electron

def bee_points(workdir):
    z = zipfile.ZipFile(os.path.join(workdir, "mabc-pr.zip"))
    for n in z.namelist():
        if n.endswith("-clustering-global.json"):
            d = json.loads(z.read(n)); return np.column_stack([d["x"], d["y"], d["z"]]).astype(float), np.asarray(d["q"], float), np.asarray(d["cluster_id"], int)
    raise RuntimeError("no clustering-global layer in " + workdir)

def sphere_sum(P, Q, path, centre, radius, veto):
    d = np.linalg.norm(P - centre, axis=1); s = d <= radius
    if not s.any(): return 0.0, 0
    idx = np.where(s)[0]; keep = np.ones(len(idx), bool)
    for pp in path:   # path points near the centre only
        if np.linalg.norm(pp - centre) > radius + veto: continue
        keep &= np.linalg.norm(P[idx] - pp, axis=1) > veto
    return float(Q[idx][keep].sum()), int(keep.sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="stm2"); ap.add_argument("-o", "--out", required=True); ap.add_argument("--tsv")
    ap.add_argument("--radius", type=float, default=20.0); ap.add_argument("--veto", type=float, default=2.5)
    ap.add_argument("--max-absx", type=float, default=305.0, help="stops closer to a CRP than this are skipped (near-CRP charge rise)")
    ap.add_argument("--same-cluster", action="store_true", help="count only Bee points whose cluster_id equals the muon's cluster ident")
    a = ap.parse_args()
    rows = []; cache = {}
    for bk in C.iter_blocks(a.tag):
        v = bk["verdict"]
        if v.get("stm") != 1 or v.get("tgm") == 1 or bk["status"] != 0: continue
        key = (bk["event"], bk["cluster"])
        if any(r["event"] == key[0] and r["cluster"] == key[1] for r in rows): continue
        rr, dq = bk["rr"], bk["dqdx"]; path = np.column_stack([bk["x"], bk["y"], bk["z"]])
        i0, i1 = int(np.argmin(rr)), int(np.argmax(rr)); stop, entry = path[i0], path[i1]
        if abs(stop[0]) > a.max_absx or abs(entry[0]) > 339.0: pass
        near = (rr < 2) & (dq > 0); mid = (rr >= 20) & (rr < 40) & (dq > 0)
        con = float(np.median(dq[near]) / np.median(dq[mid])) if near.sum() >= 3 and mid.sum() >= 5 else float("nan")
        if bk["workdir"] not in cache: cache[bk["workdir"]] = bee_points(bk["workdir"])
        P, Q, cid = cache[bk["workdir"]]
        if a.same_cluster:
            m = cid == bk["cluster"]
            if m.sum() == 0: continue
            P, Q = P[m], Q[m]
        qs, ns = sphere_sum(P, Q, path, stop, a.radius, a.veto); qe, ne = sphere_sum(P, Q, path, entry, a.radius, a.veto)
        rows.append(dict(event=bk["event"], evtno=bk["evtno"], cluster=bk["cluster"], contrast=con, L=float(rr.max()), stop=stop, entry=entry,
                         q_stop=qs, n_stop=ns, q_entry=qe, n_entry=ne, E_stop=qs * E_PER_E, E_entry=qe * E_PER_E, stop_near_crp=abs(stop[0]) > a.max_absx))
    # ident check: fraction of fitted path points with a same-cluster Bee point within 1 cm (first 30 stops)
    if a.same_cluster and rows: print("(same-cluster mode; cluster ident = Bee cluster_id assumed)")
    print(f"STM stops: {len(rows)} ({sum(r['stop_near_crp'] for r in rows)} within {339.9 - a.max_absx:.0f} cm of a CRP, excluded from the spectrum)")
    sel = [r for r in rows if not r["stop_near_crp"]]
    for lab, cut in (("all stops", lambda r: True), ("Bragg contrast >= 2", lambda r: r["contrast"] >= 2), ("contrast >= 1.5", lambda r: r["contrast"] >= 1.5), ("no rise (contrast < 1.2)", lambda r: r["contrast"] < 1.2)):
        g = [r for r in sel if cut(r)]
        if not g: continue
        Es = np.array([r["E_stop"] for r in g]); Ee = np.array([r["E_entry"] for r in g])
        print(f"  {lab}: n={len(g)} E_stop median={np.median(Es):.1f} MeV, >5 MeV {np.mean(Es > 5):.2f}, in (5,60] {np.mean((Es > 5) & (Es <= 60)):.2f} | ENTRY control median={np.median(Ee):.1f}, >5 MeV {np.mean(Ee > 5):.2f}")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write(f"# raw-charge Michel energy around STM stops, tag {a.tag}: radius {a.radius} cm, path veto {a.veto} cm, E = Q/0.5/0.8 x 23.6 eV (prototype shower convention)\n")
            f.write("event\tevtno\tcluster\tcontrast\tmu_L_cm\tstop_x\tstop_y\tstop_z\tq_stop_e\tn_stop\tE_stop_MeV\tq_entry_e\tn_entry\tE_entry_MeV\tstop_near_crp\n")
            for r in rows: f.write("\t".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in (r["event"], r["evtno"], r["cluster"], r["contrast"], r["L"], *r["stop"], r["q_stop"], r["n_stop"], r["E_stop"], r["q_entry"], r["n_entry"], r["E_entry"], int(r["stop_near_crp"]))) + "\n")
        print("wrote", a.tsv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); bins = np.arange(0, 121, 4)
    g2 = [r for r in sel if r["contrast"] >= 2]; g0 = [r for r in sel if r["contrast"] < 1.2]
    ax[0].hist([r["E_stop"] for r in sel], bins=bins, color="tab:blue", alpha=0.5, label=f"all STM stops (n={len(sel)})")
    ax[0].hist([r["E_stop"] for r in g2], bins=bins, color="tab:red", alpha=0.8, label=f"Bragg contrast >= 2 (n={len(g2)})")
    ax[0].axvline(ENDPOINT, color="k", ls="--", label="52.8 MeV endpoint"); ax[0].set_xlabel(f"charge within {a.radius:.0f} cm of the stop, off the muon path [MeV, shower convention]"); ax[0].set_ylabel("stops"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
    ax[1].hist([r["E_stop"] for r in g2], bins=bins, color="tab:red", alpha=0.6, label="stop end"); ax[1].hist([r["E_entry"] for r in g2], bins=bins, histtype="step", color="k", lw=1.5, label="ENTRY end (control)")
    ax[1].axvline(ENDPOINT, color="k", ls="--"); ax[1].set_xlabel("MeV"); ax[1].set_title("Bragg-confirmed stops: stop vs entry end", fontsize=9); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2)
    ax[2].scatter([r["contrast"] for r in sel], [r["E_stop"] for r in sel], s=8, alpha=0.6); ax[2].axhline(ENDPOINT, color="k", ls="--"); ax[2].set_xlim(0, 6); ax[2].set_ylim(0, 150)
    ax[2].set_xlabel("Bragg contrast of the STM pass"); ax[2].set_ylabel("E at the stop [MeV]"); ax[2].grid(alpha=0.2)
    fig.suptitle(f"PDVD raw-charge Michel activity at STM stops, tag {a.tag}", fontsize=10); fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)
main()
