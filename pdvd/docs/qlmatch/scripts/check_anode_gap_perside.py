#!/usr/bin/env python3
"""Bottom-TPC anode-gap check with stopping-muon discrimination.
Same line-probe machinery as check_anode_gap_imglevel.py, reported per side,
plus per-track end-vs-mid charge ratio (Bragg check): a stopping muon rises
~x2-3 in dQ/dx near its end; a through-going MIP stays ~1."""
import glob, json, os, re, zipfile
import numpy as np

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
DEC = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/ql_display"
TAG = "anodefix"

def read_dec(path):
    out = []
    if os.path.isfile(path):
        for ln in open(path):
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                if r["verdict"] in ("keep", "add"):
                    out.append(r)
    return out

results = []
for cal in sorted(glob.glob(os.path.join(WORK, "039252_*_%s" % TAG, "calib-evt*.json"))):
    idx = int(re.match(r"039252_(\d+)_", os.path.basename(os.path.dirname(cal))).group(1))
    ev = os.path.basename(cal)[len("calib-evt"):-len(".json")]
    d = json.load(open(cal))
    fb = {x["gid"]: x for x in d["flashes"]}
    cb = {c["uid"]: c for c in d["clusters"]}
    geom = {int(k): g for k, g in d["geometry"].items()}
    drift = d["drift_speed"]
    offs = d.get("trigger_offsets_us") or [0.0, 0.0]
    targets = []
    for sub in ("decisions-boundary", "decisions-crossers"):
        for r in read_dec(os.path.join(DEC, sub, "decisions-evt%s.jsonl" % ev)):
            targets.append((r["main_cluster_uid"], r["flash_gid"]))
    imgP = imgU = None
    for uid, gid in set(targets):
        c = cb.get(uid)
        if c is None or gid not in fb: continue
        g = geom[c["apa"]]
        t = fb[gid]["time"] if c["apa"] == 0 else fb[gid].get("time1", fb[gid]["time"] + offs[1]-offs[0])
        xo = g["sign_offset"] * t * drift
        P = np.column_stack([np.asarray(c["x"], float), np.asarray(c["y"], float),
                             np.asarray(c["z"], float)])
        Q = np.asarray(c["q"], float)
        U = g["s"] * (P[:, 0] + xo - g["anode_x"])
        end_u = U.min()
        if not (0.0 <= end_u <= 12.0): continue
        W = np.column_stack([U, P[:, 1], P[:, 2]])
        sel = U < end_u + 150
        if sel.sum() < 10: continue
        Wn = W[sel]; ctr = Wn.mean(0)
        _, _, vt = np.linalg.svd(Wn - ctr, full_matrices=False)
        dirv = vt[0]
        if dirv[0] > 0: dirv = -dirv
        p0 = Wn[np.argmin(Wn[:, 0])]
        steps = (0.0 - p0[0]) / dirv[0] if dirv[0] != 0 else 1e9
        entry = p0 + steps * dirv
        if not (g["y_lo"]+10 < entry[1] < g["y_hi"]-10 and g["z_lo"]+10 < entry[2] < g["z_hi"]-10):
            continue
        # Bragg check: charge per point in last 15cm along track vs 30-120cm from end
        s_all = (W - p0) @ dirv    # 0 at anode-most point, negative back along track
        dist = -s_all              # distance from the anode-side end, along track
        q_end = Q[(dist >= 0) & (dist < 15)]
        q_mid = Q[(dist >= 30) & (dist < 120)]
        ratio = (np.median(q_end) / np.median(q_mid)) if len(q_end) >= 5 and len(q_mid) >= 20 else np.nan
        results.append(dict(side="bot" if c["apa"] == 0 else "top",
                            end_u=end_u, ratio=ratio, ev=ev, uid=uid))

for side in ("bot", "top"):
    rs = [r for r in results if r["side"] == side]
    u = np.array([r["end_u"] for r in rs])
    rat = np.array([r["ratio"] for r in rs])
    print("== %s volume: interior-%s tracks, end u in [0,12] ==" %
          (side, "exit" if side == "bot" else "entry"))
    print("  n=%d  gap median %+5.2f  MAD %4.2f" % (len(u), np.median(u),
          np.median(np.abs(u - np.median(u)))))
    h, e = np.histogram(u, bins=12, range=(0, 12))
    print("  end-u hist 0..12cm, 1cm bins: %s" % " ".join("%2d" % c for c in h))
    ok = rat[~np.isnan(rat)]
    print("  Bragg ratio (end 0-15cm / mid 30-120cm): n=%d median %.2f;  >1.5: %d  (stopping-like)"
          % (len(ok), np.median(ok), (ok > 1.5).sum()))
    mip = [r for r in rs if not np.isnan(r["ratio"]) and r["ratio"] <= 1.5]
    um = np.array([r["end_u"] for r in mip])
    if len(um):
        print("  MIP-like only (ratio<=1.5): n=%d  gap median %+5.2f" % (len(um), np.median(um)))
