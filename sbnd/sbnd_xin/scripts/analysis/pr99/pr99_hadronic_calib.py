#!/usr/bin/env python3
"""doc pr/99 round 3 (A5) -- roster-wide calibration of the hadronic-shower
scalars, mirroring the C++ predicate in NeutrinoShowerClustering.cxx
(shower_hadronic_tag) bin-for-bin:

  axis     = shower member fit points (calib JSON `points`), arc proxy
             s = |p - shower.start|
  n_in(b)  = imaged charge points (clustering-global) within r_cyl=8cm of
             ANY of bin b's axis points (3cm bins over the first
             min(smax,30)cm; per-bin union, no cross-bin dedup), counted
             only when OWNED: the point's nearest fit point over ALL
             segments belongs to a member (without this, vertex-region
             activity from other prongs inflates a real electron's early
             bins and fakes a shrinking profile -- 46363: raw 2365->335 but
             owned growth 3.1)
  growth   = mean(last two populated bins) / mean(first two populated bins)
             (the round-1 "ends" definition: electrons 2.3-8.8, misID'd
             hadrons <= 0.7 on the 109-shower roster)
  bragg    = max(last-2-bin median q) / median(other bins' median q) over
             the FULL trajectory (member T_rec_charge q; the C++ uses fit
             dQ/dx -- ratio-equivalent at uniform dx)
  stem     = median of the first 6 stem_dqdx samples (MIP units; the
             proton-stem branch reaches 395148 at 3.22 while
             pair-conversion gamma stems read ~2)
  verdict  = smax>=10 && (growth<0.7 || (bragg>=3 && growth<1.2)
             || (stem>=3.0 && growth<1.2))

One row per conn-1 |pdg|==11 shower with smax >= 6 cm.  Repro:
  python3 scripts/analysis/pr99/pr99_hadronic_calib.py \
    work-pr99r2-on3-nuecc48:nuecc48 work-pr99r2-on3-ncpi0:ncpi0 \
    work-pr99r2-on3-mcp1k:mcp1k work-pr99r2-on3-mcp2k:mcp2k \
    > /home/xqian/tmp/pr99r3/hadronic_calib2.tsv
"""
import glob, json, os, sys, zipfile
import numpy as np
from scipy.spatial import cKDTree

R_CYL, BIN, SCAN, MINLEN = 8.0, 3.0, 30.0, 10.0
GROWTH_MAX, GROWTH_BRAGG, BRAGG_RATIO, STEM_RATIO = 0.7, 1.2, 3.0, 3.0

print("sample\tevt\tshower\tprim\tnue\tE_MeV\tnseg\tsmax\tstem\tg_owned\tq_trunk\tq_term\tbragg\tverdict")
for arg in sys.argv[1:]:
    arm, smp = arg.split(":") if ":" in arg else (arg, arg)
    for d in sorted(glob.glob(f"{arm}/pr_evt*/")):
        evt = os.path.basename(d.rstrip("/"))[6:]
        try:
            cal = json.load(open(f"{d}calib-pr-evt{evt}.json"))
            z = zipfile.ZipFile(f"{d}mabc-pr.zip")
            g = json.loads(z.read([n for n in z.namelist() if n.endswith("clustering-global.json")][0]))
            chg = np.c_[g["x"], g["y"], g["z"]]
            import uproot
            a = uproot.open(f"{d}tracking-pr.root")["T_rec_charge"].arrays(
                ["x", "y", "z", "q", "real_cluster_id"], library="np")
        except Exception as e:
            print(f"# {smp} {evt} SKIP {e}", file=sys.stderr); continue
        if not len(chg): continue
        segpts, seglab = [], []
        for s in cal["segments"]:
            P = np.array([[p["x"], p["y"], p["z"]] if isinstance(p, dict) else p for p in s["points"]])
            segpts.append(P); seglab.append(np.full(len(P), s["id"]))
        allfit = np.vstack(segpts); alllab = np.concatenate(seglab)
        nearlab = alllab[cKDTree(allfit).query(chg)[1]]
        nue = cal.get("tagger", {}).get("nue_score", -99)
        mv = np.array([cal["main_vertex"][k] for k in "xyz"])
        c1 = [s for s in cal["showers"] if s.get("start_connection_type") == 1
              and abs(s.get("particle_id", 0)) == 11]
        prim, pmax = None, -1
        for s in c1:
            if np.linalg.norm(np.array([s["start"][k] for k in "xyz"]) - mv) < 0.5 and s["kine_best"] > pmax:
                pmax, prim = s["kine_best"], s["id"]
        for sh in c1:
            mem = [s for s in cal["segments"] if s.get("shower_id") == sh["id"]]
            if not mem: continue
            mids = {s["id"] for s in mem}
            sp = np.array([sh["start"][k] for k in "xyz"])
            pts = np.vstack([np.array([[p["x"], p["y"], p["z"]] if isinstance(p, dict) else p
                                       for p in s["points"]]) for s in mem])
            s_arc = np.linalg.norm(pts - sp, axis=1)
            smax = s_arc.max()
            if smax < 6.0: continue
            owned = np.isin(nearlab, list(mids))
            cto = cKDTree(chg[owned]) if owned.any() else None
            scan = min(smax, SCAN); nb = max(1, int(np.ceil(scan / BIN)))
            bo = np.zeros(nb)
            for b in range(nb):
                axm = pts[(s_arc >= b * BIN) & (s_arc < min((b + 1) * BIN, scan))]
                if len(axm) == 0 or cto is None: continue
                idx = set()
                for hits in cto.query_ball_point(axm, R_CYL): idx.update(hits)
                bo[b] = len(idx)
            f = bo[:2][bo[:2] > 0]; l = bo[-2:][bo[-2:] > 0]
            gro = l.mean() / f.mean() if len(f) and len(l) else -1.0
            msk = np.isin(a["real_cluster_id"], list(mids))
            qq, qp = a["q"][msk], np.c_[a["x"][msk], a["y"][msk], a["z"][msk]]
            nbf = max(1, int(np.ceil(smax / BIN)))
            q_trunk = q_term = -1.0
            if len(qq):
                sq = np.linalg.norm(qp - sp, axis=1)
                bidx = np.minimum((sq / BIN).astype(int), nbf - 1)
                meds = [np.median(qq[bidx == b]) if (bidx == b).any() else None for b in range(nbf)]
                trunk = [m for b, m in enumerate(meds) if m is not None and b < nbf - 2]
                term  = [m for b, m in enumerate(meds) if m is not None and b >= nbf - 2]
                if trunk: q_trunk = float(np.median(trunk))
                if term:  q_term = float(max(term))
            bragg = q_term / q_trunk if q_trunk > 0 and q_term > 0 else -1.0
            stem = float(np.median(sh["stem_dqdx"][:6])) if sh.get("stem_dqdx") else -1.0
            verdict = int(smax >= MINLEN and gro >= 0 and
                          (gro < GROWTH_MAX or (bragg >= BRAGG_RATIO and gro < GROWTH_BRAGG)
                           or (stem >= STEM_RATIO and gro < GROWTH_BRAGG)))
            print(f"{smp}\t{evt}\t{sh['id']}\t{int(sh['id'] == prim)}\t{nue:.2f}\t{sh['kine_best']:.1f}\t"
                  f"{sh['num_segments']}\t{smax:.1f}\t{stem:.2f}\t{gro:.3f}\t{q_trunk:.0f}\t{q_term:.0f}\t"
                  f"{bragg:.2f}\t{verdict}")
