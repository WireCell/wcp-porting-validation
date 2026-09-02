#!/usr/bin/env python3
"""Michel candidates anchored on the STM STOP END (doc pdvd/25 sec 13.7, ask 6).

The flags-6-8 route (michel_energy.py) takes the highest-energy shower at the
neutrino-PR main vertex, which for a cosmic bundle is often the entry end or a
kink -- it returned 285 cm / 657 MeV "Michels".  This scorer instead reads the
accepted STM pass of each STM-tagged, non-TGM main from tracking-stm.root
(collect_stm_sample.iter_blocks: rr = 0 at the fitted stop) and takes, from
the same bundle's calib-pr dump, the showers whose START lies within
--max-dist cm of the stop and whose total_length <= --max-len cm (a Michel
electron of <= 53 MeV travels ~< 25 cm).  Energy = kine_best (MeV; the dump's
best of range/charge/dQdx).  The muon segment itself is excluded by
particle_id != 13 || length cap.

Usage: python3 stm/michel_stop_end.py --tag stm1 -o docs/pics/pdvd_michel_stop_end.png --tsv stm/michel_stop_end.tsv
"""
import argparse, glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PDVD = os.path.dirname(HERE); sys.path.insert(0, HERE)
import collect_stm_sample as C
ENDPOINT = 52.8

def xyz(p):
    if isinstance(p, dict): return np.array([p.get("x", np.nan), p.get("y", np.nan), p.get("z", np.nan)], float)
    return np.array(p[:3], float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stm1"); ap.add_argument("-o", "--out", required=True); ap.add_argument("--tsv")
    ap.add_argument("--max-dist", type=float, default=5.0); ap.add_argument("--max-len", type=float, default=40.0)
    ap.add_argument("--min-contrast", type=float, default=0.0, help="require this Bragg contrast on the STM pass (0 = any)")
    a = ap.parse_args()
    stops = {}   # (event, cluster) -> dict(stop=xyz, contrast, L, npts, block)
    for bk in C.iter_blocks(a.tag):
        v = bk["verdict"]
        if v.get("stm") != 1 or v.get("tgm") == 1 or bk["status"] != 0: continue
        rr, dq = bk["rr"], bk["dqdx"]; i0 = int(np.argmin(rr))
        near = (rr < 2) & (dq > 0); mid = (rr >= 20) & (rr < 40) & (dq > 0)
        con = float(np.median(dq[near]) / np.median(dq[mid])) if near.sum() >= 3 and mid.sum() >= 5 else float("nan")
        key = (bk["event"], bk["cluster"])
        if key not in stops:   # first accepted pass wins (forward before backward)
            stops[key] = dict(stop=np.array([bk["x"][i0], bk["y"][i0], bk["z"][i0]]), contrast=con, L=float(rr.max()), npts=bk["npts"], block=bk["block"], evtno=bk["evtno"])
    rows = []; n_bundle = 0; n_with_dump = 0
    for (ev, cid), st in sorted(stops.items()):
        d = os.path.join(PDVD, "work", f"{ev}_{a.tag}"); dumps = glob.glob(os.path.join(d, "calib-pr-evt*.json"))
        if not dumps: continue
        n_with_dump += 1
        dump = json.load(open(dumps[0]))
        for c in dump.get("candidates") or [dump]:
            mv = c.get("main_vertex") or {}
            if mv.get("cluster_id") != cid: continue
            n_bundle += 1
            if a.min_contrast and not (st["contrast"] >= a.min_contrast): continue
            cands = []
            segq = {}
            for sg in c.get("segments") or []:
                segq.setdefault(sg.get("shower_id"), 0.0)
                segq[sg.get("shower_id")] += sum((pt.get("dQ") or 0.0) for pt in (sg.get("points") or []))
            for s in c.get("showers") or []:
                # charge-sum energy of the shower's own segments (prototype cal_kine_charge convention:
                # E = Q x 23.6 eV / 0.5 recombination for EM showers), filled here because kine_charge is 0 for small showers
                s["E_segQ"] = segq.get(s.get("id"), 0.0) * 23.6e-6 / 0.5
                sp = xyz(s.get("start")); dist = float(np.linalg.norm(sp - st["stop"]))
                if dist <= a.max_dist and (s.get("total_length") or 0.0) <= a.max_len and s.get("particle_id") != 13:
                    cands.append((dist, s))
            if not cands: continue
            dist, s = min(cands, key=lambda t: t[0])
            rows.append(dict(event=ev, evtno=st["evtno"], cluster=cid, nu_index=c.get("nu_index", -1), contrast=st["contrast"], mu_L=st["L"],
                             stop=st["stop"], dist=dist, E=s.get("kine_best"), E_charge=s.get("kine_charge"), E_range=s.get("kine_range"), E_segQ=s.get("E_segQ"),
                             length=s.get("total_length"), pdg=s.get("particle_id"), sid=s.get("id"), conn=s.get("start_connection_type"),
                             f7=((c.get("tagger") or {}).get("cosmict_flag_7")), f7f=((c.get("tagger") or {}).get("cosmict_7_filled"))))
    E = np.array([r["E"] for r in rows if r["E"] is not None], float)
    print(f"STM stops with a dump: {n_with_dump} dirs, {n_bundle} bundles matched; Michel candidates (shower start <= {a.max_dist} cm from the stop, length <= {a.max_len} cm): {len(rows)}")
    if len(E): print(f"  kine_best: n={len(E)} median={np.median(E):.1f} MeV, <= endpoint {np.sum(E <= ENDPOINT)}, > endpoint {np.sum(E > ENDPOINT)}; with Bragg contrast >= 2: {sum(1 for r in rows if r['contrast'] >= 2)}")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write(f"# Michel candidates anchored on the STM stop end, tag {a.tag}: max_dist {a.max_dist} cm, max_len {a.max_len} cm, min_contrast {a.min_contrast}\n")
            f.write("event\tevtno\tcluster\tnu_index\tcontrast\tmu_L_cm\tstop_x\tstop_y\tstop_z\tdist_cm\tkine_best_MeV\tkine_charge_MeV\tkine_range_MeV\tsegQ_E_MeV\tlength_cm\tpdg\tshower_id\tconn\tf7\tf7_filled\n")
            for r in rows:
                f.write("\t".join(str(v) if not isinstance(v, float) else f"{v:.2f}" for v in (r["event"], r["evtno"], r["cluster"], r["nu_index"], r["contrast"], r["mu_L"], *r["stop"], r["dist"], r["E"], r["E_charge"], r["E_range"], r["E_segQ"], r["length"], r["pdg"], r["sid"], r["conn"], r["f7"], r["f7f"])) + "\n")
        print("wrote", a.tsv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(E, bins=np.arange(0, 121, 5), color="tab:blue", alpha=0.8, label=f"all (n={len(E)})")
    Eb = np.array([r["E"] for r in rows if r["E"] is not None and r["contrast"] >= 2], float)
    if len(Eb): ax[0].hist(Eb, bins=np.arange(0, 121, 5), color="tab:red", alpha=0.8, label=f"STM pass with Bragg contrast >= 2 (n={len(Eb)})")
    ax[0].axvline(ENDPOINT, color="k", ls="--", label="52.8 MeV endpoint"); ax[0].set_xlabel("shower kine_best at the muon stop [MeV]"); ax[0].set_ylabel("candidates"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
    ax[0].set_title(f"PDVD Michel candidates, tag {a.tag}: start <= {a.max_dist} cm from the STM stop, length <= {a.max_len} cm", fontsize=8)
    ax[1].scatter([r["length"] for r in rows], [r["E"] or 0 for r in rows], s=12, c=["tab:red" if r["contrast"] >= 2 else "tab:blue" for r in rows])
    ax[1].axhline(ENDPOINT, color="k", ls="--"); ax[1].set_xlabel("shower total length [cm]"); ax[1].set_ylabel("kine_best [MeV]"); ax[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)
main()
