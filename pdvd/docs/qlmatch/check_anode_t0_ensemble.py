#!/usr/bin/env python3
"""Multi-track anode/T0 demonstration: for validated tracks with imaging
ends near u=0, trace the W-plane corridor from 5 cm on-track through u=-4
and measure where the gauss/raw signal actually stops (u_stop).  If anode
position + T0 are right, u_stop ~ 0 across tracks in BOTH volumes."""
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
CASES = [  # (evt, uid, gid|None) validated tracks with ends near new u=0
    ("298581", 4000161, None), ("298763", 4000244, None),
    ("298567", 4000058, None), ("298637", 4000172, None),   # top
    ("298595", 167, None), ("298581", 9, None),
    ("298567", 132, None), ("298749", 88, None),            # bottom
    ("298777", 181, 130), ("298749", 3, 105),               # gold (reference)
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

FR = {}
def load_frames(pdir, anode, ev):
    key = (pdir, anode)
    if key in FR: return FR[key]
    out = {}
    with tarfile.open(os.path.join(pdir, "protodune-sp-dnnroi-frames-anode%d.tar.bz2" % anode)) as tf:
        for m in tf.getmembers():
            for tag0, tag in (("gauss","gauss%d"%anode), ("raw","raw%d"%anode)):
                for kind in ("frame","channels"):
                    if m.name == "%s_%s_%s.npy"%(kind,tag,ev):
                        out[(kind,tag0)] = np.load(io.BytesIO(tf.extractfile(m).read()))
    FR[key] = out
    return out

summary = []
for ev, uid, gid in CASES:
    if gid is None: gid = dec_gid(ev, uid)
    adir, pdir = find_dirs(ev)
    d = json.load(open(os.path.join(adir, "calib-evt%s.json" % ev)))
    V = d["drift_speed"]
    fb = {x["gid"]: x for x in d["flashes"]}
    cb = {cc["uid"]: cc for cc in d["clusters"]}
    if uid not in cb or gid is None or gid not in fb:
        print("skip", ev, uid, gid); continue
    c = cb[uid]
    g = {int(k): v for k, v in d["geometry"].items()}[c["apa"]]
    offs = d["trigger_offsets_us"]
    t = fb[gid]["time"] if c["apa"]==0 else fb[gid].get("time1", fb[gid]["time"]+offs[1]-offs[0])
    side = "bot" if c["apa"]==0 else "top"
    P = np.column_stack([np.asarray(c["x"],float), np.asarray(c["y"],float), np.asarray(c["z"],float)])
    xo = g["sign_offset"]*t*V
    U = g["s"]*(P[:,0]+xo-g["anode_x"])
    end_u = U.min()
    sel = U < end_u+60
    Pn = P[sel]; ctr = Pn.mean(0)
    _,_,vt = np.linalg.svd(Pn-ctr, full_matrices=False)
    dirr = vt[0]
    if dirr[0]*g["s"] > 0: dirr = -dirr
    p_end = Pn[np.argmin(U[sel])]
    # tick-offset calibration on actual points
    dts = []
    for p in Pn[::max(1,len(Pn)//80)]:
        r = chan_tick(p, side, V)
        if r is None: continue
        a, ch, ti = r
        fr = load_frames(pdir, a, ev)
        if ("channels","gauss") not in fr: continue
        rg = np.where(fr[("channels","gauss")] == ch)[0]
        if not len(rg) or not (30 <= ti < fr[("frame","gauss")].shape[1]-30): continue
        row = fr[("frame","gauss")][rg[0]]
        s = np.array([row[ti+dt-6:ti+dt+7].sum() for dt in range(-20,21)])
        if s.max() > 500: dts.append(int(s.argmax())-20)
    dt0 = int(np.median(dts)) if dts else 0
    # corridor: u from end+5 down to -4
    prof = []
    for s in np.arange(-10, (end_u+4)/0.5+1) * (0.5/abs(dirr[0])):
        p = p_end + s*dirr
        u = g["s"]*(p[0]+xo-g["anode_x"])
        r = chan_tick(p, side, V)
        if r is None: continue
        a, ch, ti = r
        ti += dt0
        fr = load_frames(pdir, a, ev)
        if ("channels","gauss") not in fr: continue
        chg = fr[("channels","gauss")]; Fg = fr[("frame","gauss")]; Frw = fr[("frame","raw")]
        rg = np.where(chg == ch)[0]
        if not len(rg) or not (30 <= ti < Fg.shape[1]-30): continue
        best = (0.0, 0, 0)
        for dc in range(-4, 5):
            rr = rg[0]+dc
            if not (0 <= rr < Fg.shape[0]): continue
            row = Fg[rr]
            for dt in range(-8, 9):
                s13 = row[ti+dt-6:ti+dt+7].sum()
                if s13 > best[0]: best = (float(s13), dc, dt)
        gsum, dc, dt = best
        rr = np.where(fr[("channels","raw")] == ch)[0][0]+dc
        rwin = Frw[rr, ti+dt-10:ti+dt+11]
        prof.append((u, gsum, float(rwin.max()), a, ch+dc, ti+dt))
    # u_stop: walking down in u, last sample with gauss>1500 before 2 consecutive <800
    prof.sort(key=lambda r: -r[0])
    u_stop = None
    for i, (u, gsum, rpk, a, ch, ti) in enumerate(prof):
        if gsum > 1500: u_stop = u
    # better: minimum u with signal
    # contiguous walk down in u from the imaging end (gaps <= 1.1 cm allowed)
    def walk(idx_val, thr):
        stop = None
        for u, val in sorted(idx_val, key=lambda r: -r[0]):
            if u > end_u + 0.6: continue
            if val > thr:
                if stop is None or stop - u <= 1.1:
                    stop = u
            elif stop is not None and stop - u > 1.1:
                break
        return stop if stop is not None else float('nan')
    u_stop = walk([(u, gsum) for (u, gsum, rpk, a, ch, ti) in prof], 1500)
    u_stop_raw = walk([(u, rpk) for (u, gsum, rpk, a, ch, ti) in prof], 40)
    summary.append((ev, uid, gid, side, end_u, u_stop, u_stop_raw, dt0))
    print("evt %s uid %-8d gid %3d %s | img end %+5.2f | gauss stops %+5.2f | raw stops %+5.2f | dt0 %+d" %
          (ev, uid, gid, side, end_u, u_stop, u_stop_raw, dt0))
    # plot profile
    fig, ax = plt.subplots(figsize=(7, 3.4))
    uu = [r[0] for r in prof]; gs = [r[1] for r in prof]; rp = [r[2] for r in prof]
    ax.plot(uu, np.array(gs)/max(max(gs),1), "o-", ms=3, label="gauss (norm)")
    ax.plot(uu, np.array(rp)/max(max(rp),1), "s-", ms=3, label="raw0 peak (norm)")
    ax.axvline(0, color="r", ls="--", label="anode (u=0, flash time)")
    ax.axvline(end_u, color="gray", ls=":", label="imaging end")
    ax.set_xlabel("u [cm]"); ax.set_ylabel("signal (norm)")
    ax.set_title("evt%s uid%d %s" % (ev, uid, side), fontsize=9)
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig("prof_%s_%d.png" % (ev, uid), dpi=110); plt.close(fig)

print("\n== SUMMARY: signal-stop u vs anode (0) ==")
for side in ("bot", "top"):
    ss = [r for r in summary if r[3] == side]
    gs = [r[5] for r in ss if not np.isnan(r[5])]
    print("  %s: n=%d  gauss-stop median %+5.2f  [%s]" %
          (side, len(gs), np.median(gs), ", ".join("%+.1f" % v for v in sorted(gs))))
