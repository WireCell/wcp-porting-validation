#!/usr/bin/env python3
"""Chain-consistency trace: imaging points vs gauss decon vs raw0 (post-NF
ADC), W plane, along validated tracks' anode-end corridors.  See
02_pdvd-anode-time-consistency.md section 8.8 for the findings.

A) validate (y,z,x_raw)->(W chan,tick) on actual cluster points (gauss>0?)
   and measure the tick offset of the gauss ridge vs prediction;
B) probe the missing corridor (end_u -> -3) with the calibrated offset;
C) heatmaps (gauss + raw0) around the end region."""
import json, glob, os, sys, tarfile, io
import numpy as np
sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/img_plot")
import geom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
DEC = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/ql_display"
STORE = geom.load_store("/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2")
# (evt, uid, gid or None -> looked up in decisions files)
CASES = [
    ("298609", 38, None),        # boundary bot; end = satellite fragment (see doc 8.8)
    ("298791", 259, None),       # boundary bot, drift-aligned; imaging drops real SP charge
    ("298693", 4000003, None),   # boundary top; V-topology scatter apex mid-volume
    ("298777", 181, 130),        # GOLD full-drift crosser; reconstructs to +1.2
    ("298749", 3, 105),          # GOLD full-drift crosser; reconstructs through u=0
]

def find_dirs(ev):
    for d in sorted(glob.glob(os.path.join(WORK, "039252_*_anodefix"))):
        if os.path.isfile(os.path.join(d, "calib-evt%s.json" % ev)):
            return d, d[:-len("_anodefix")]

def dec_gid(ev, uid):
    for sub in ("decisions-boundary", "decisions-crossers"):
        p = os.path.join(DEC, sub, "decisions-evt%s.jsonl" % ev)
        if os.path.isfile(p):
            for ln in open(p):
                ln = ln.strip()
                if ln:
                    r = json.loads(ln)
                    if r["verdict"] in ("keep","add") and r["main_cluster_uid"] == uid:
                        return r["flash_gid"]

PG = {}
def pg(a, f):
    if (a,f) not in PG: PG[(a,f)] = geom.PlaneGeom(STORE, a, f, 2)
    return PG[(a,f)]

def which_af(y, z, side):
    for a in ((0,1,2,3) if side=="bot" else (4,5,6,7)):
        for f in (0,1):
            g = pg(a,f)
            ys = np.concatenate([g.tails[:,0], g.heads[:,0]]); zs = np.concatenate([g.tails[:,1], g.heads[:,1]])
            if ys.min()-0.3 <= y <= ys.max()+0.3 and zs.min()-0.3 <= z <= zs.max()+0.3:
                return a, f
    return None, None

def chan_tick(p, side, V):
    a, f = which_af(p[1], p[2], side)
    if a is None: return None
    g = pg(a, f)
    centers = 0.5*(g.tails+g.heads)
    k = int(round((p[2]-centers[0,1])/g.pitch_cm))
    if not (0 <= k < len(g.chans)): return None
    dirx = 1 if side=="bot" else -1
    ti = (p[0] - geom.wplane_x_cm(STORE, a, f))*dirx/V/0.5
    return a, int(g.chans[k]), int(round(ti))

def load_frames(pdir, anode, ev):
    out = {}
    with tarfile.open(os.path.join(pdir, "protodune-sp-dnnroi-frames-anode%d.tar.bz2" % anode)) as tf:
        for m in tf.getmembers():
            for tag0, tag in (("gauss","gauss%d"%anode), ("raw","raw%d"%anode)):
                for kind in ("frame","channels"):
                    if m.name == "%s_%s_%s.npy"%(kind,tag,ev):
                        out[(kind,tag0)] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return out

for ev, uid, gid in CASES:
    if gid is None: gid = dec_gid(ev, uid)
    adir, pdir = find_dirs(ev)
    d = json.load(open(os.path.join(adir, "calib-evt%s.json" % ev)))
    V = d["drift_speed"]
    fb = {x["gid"]: x for x in d["flashes"]}
    c = {cc["uid"]: cc for cc in d["clusters"]}[uid]
    g = {int(k): v for k, v in d["geometry"].items()}[c["apa"]]
    offs = d["trigger_offsets_us"]
    t = fb[gid]["time"] if c["apa"]==0 else fb[gid].get("time1", fb[gid]["time"]+offs[1]-offs[0])
    side = "bot" if c["apa"]==0 else "top"
    P = np.column_stack([np.asarray(c["x"],float), np.asarray(c["y"],float), np.asarray(c["z"],float)])
    xo = g["sign_offset"]*t*V
    U = g["s"]*(P[:,0]+xo-g["anode_x"])
    end_u = U.min()
    frames = {}
    def fr_of(a):
        if a not in frames: frames[a] = load_frames(pdir, a, ev)
        return frames[a]
    # --- A) mapping validation on actual points (u < end+25) ---
    pts = P[U < end_u+25]
    hits, dts = 0, []
    for p in pts[::max(1, len(pts)//150)]:
        r = chan_tick(p, side, V)
        if r is None: continue
        a, ch, ti = r
        fr = fr_of(a)
        rg = np.where(fr[("channels","gauss")] == ch)[0]
        if not len(rg) or not (30 <= ti < fr[("frame","gauss")].shape[1]-30): continue
        row = fr[("frame","gauss")][rg[0]]
        s = np.array([row[ti+dt-6:ti+dt+7].sum() for dt in range(-20, 21)])
        if s.max() > 500:
            hits += 1
            dts.append(int(s.argmax())-20)
    dt0 = int(np.median(dts)) if dts else 0
    print("\n#### evt %s uid %d gid %d side %s end_u %+0.2f" % (ev, uid, gid, side, end_u))
    print("  A) mapping check on %d actual points near end: gauss-hit %d/%d, ridge tick offset median %+d (MAD %.1f)"
          % (len(pts[::max(1,len(pts)//150)]), hits, len(pts[::max(1,len(pts)//150)]), dt0,
             np.median(np.abs(np.array(dts)-dt0)) if dts else -1))
    # --- B) corridor probe with calibrated dt0 ---
    sel = U < end_u+60
    Pn = P[sel]; ctr = Pn.mean(0)
    _,_,vt = np.linalg.svd(Pn-ctr, full_matrices=False)
    dirr = vt[0]
    if dirr[0]*g["s"] > 0: dirr = -dirr
    p_end = Pn[np.argmin(U[sel])]
    print("  B) corridor beyond imaging end (search ch+-6, tick dt0%+d +-10):" % dt0)
    print("     %6s | %9s | %8s %7s | %5s %5s %5s" % ("u","gauss13t","rawsum","rawpk","anode","chan","tick"))
    corr = []
    for s in np.arange(0, (end_u+3)/0.5+1) * (0.5/abs(dirr[0])):
        p = p_end + s*dirr
        u = g["s"]*(p[0]+xo-g["anode_x"])
        r = chan_tick(p, side, V)
        if r is None: continue
        a, ch, ti = r
        ti += dt0
        fr = fr_of(a)
        chg = fr[("channels","gauss")]; Fg = fr[("frame","gauss")]; Frw = fr[("frame","raw")]
        rg = np.where(chg == ch)[0]
        if not len(rg) or not (30 <= ti < Fg.shape[1]-30): continue
        best = (0.0, 0, 0)
        for dc in range(-6, 7):
            rr = rg[0]+dc
            if not (0 <= rr < Fg.shape[0]): continue
            row = Fg[rr]
            for dt in range(-10, 11):
                s13 = row[ti+dt-6:ti+dt+7].sum()
                if s13 > best[0]: best = (float(s13), dc, dt)
        gsum, dc, dt = best
        rr = np.where(fr[("channels","raw")] == ch)[0][0]+dc
        rwin = Frw[rr, ti+dt-10:ti+dt+11]
        corr.append((u, gsum, float(rwin.sum()), float(rwin.max()), a, ch+dc, ti+dt))
        if abs(u-round(u)) < 0.26:
            mark = " <-- ANODE" if abs(u) < 0.3 else ""
            print("     %+6.1f | %9.0f | %8.1f %7.1f | %5d %5d %5d%s" % (u, gsum, rwin.sum(), rwin.max(), a, ch+dc, ti+dt, mark))
    # --- C) heatmaps around the end ---
    r0info = chan_tick(p_end, side, V)
    if r0info:
        a_end, ch_end, ti_end = r0info
        ti_end += dt0
        fr = fr_of(a_end)
        chg = fr[("channels","gauss")]; Fg = fr[("frame","gauss")]; Frw = fr[("frame","raw")]
        # actual track points mapped, for overlay
        ov = []
        for p in P[U < end_u+30]:
            r = chan_tick(p, side, V)
            if r and r[0] == a_end: ov.append((r[1], r[2]+dt0))
        ov = np.array(ov) if ov else np.zeros((0,2))
        cw = 40; tw = 250
        rmid = np.where(chg == ch_end)[0][0]
        sl = slice(max(0, rmid-cw), min(Fg.shape[0], rmid+cw))
        t0_ = max(0, ti_end-tw); t1_ = min(Fg.shape[1], ti_end+tw)
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        for ax, F, ttl, vmax in ((axs[0], Fg, "gauss decon (post-DNNROI)", None),
                                 (axs[1], Frw, "raw0 = post-NF ADC", 50)):
            im = F[sl, t0_:t1_]
            ax.imshow(im, aspect="auto", origin="lower", cmap="viridis",
                      extent=[t0_, t1_, chg[sl.start], chg[sl.stop-1]],
                      vmax=(max(np.percentile(im,99.9),1) if vmax is None else vmax), vmin=0)
            if len(ov): ax.plot(ov[:,1], ov[:,0], "r.", ms=1.5, alpha=.6, label="imaging track points")
            cc = [x for x in corr if x[4]==a_end]
            ax.plot([x[6] for x in cc], [x[5] for x in cc], "w.", ms=2, label="missing-segment corridor")
            au = [x for x in cc if abs(x[0]) < 0.4]
            if au: ax.plot(au[0][6], au[0][5], "w*", ms=15, label="anode crossing (u=0)")
            ax.set_title("%s   evt%s uid%d (%s, anode%d)" % (ttl, ev, uid, side, a_end), fontsize=10)
            ax.set_ylabel("W channel")
            ax.legend(fontsize=7, loc="upper right")
        axs[1].set_xlabel("tick (0.5 us)")
        fig.tight_layout()
        fig.savefig("chain_%s_%d.png" % (ev, uid), dpi=110)
        plt.close(fig)
        print("  C) wrote chain_%s_%d.png" % (ev, uid))
