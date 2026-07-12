#!/usr/bin/env python3
"""Line-probe: for validated tracks whose anode-side end sits at small u,
extend the local track line toward u=0 and look for ANY imaging points
(img-global, all clusters, q>0) along it.  Distinguishes clustering
fragmentation from a genuine imaging-level near-anode gap."""
import glob, json, os, re, sys, zipfile
import numpy as np

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
DEC = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/ql_display"
TAG = "anodefix"
TUBE = 4.0      # cm radius around the line
UMAX_END = 40.0 # only probe tracks whose cluster anode-end u < this
NEAR = 150.0    # cm of track used for the local direction fit

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

def probe(ev_idx, ev, uid, gid, kind):
    cal = os.path.join(WORK, "039252_%d_%s" % (ev_idx, TAG), "calib-evt%s.json" % ev)
    d = json.load(open(cal))
    fb = {x["gid"]: x for x in d["flashes"]}
    cb = {c["uid"]: c for c in d["clusters"]}
    geom = {int(k): g for k, g in d["geometry"].items()}
    drift = d["drift_speed"]
    offs = d.get("trigger_offsets_us") or [0.0, 0.0]
    c = cb.get(uid)
    if c is None or gid not in fb:
        return None
    g = geom[c["apa"]]
    t = fb[gid]["time"] if c["apa"] == 0 else fb[gid].get("time1", fb[gid]["time"] + offs[1]-offs[0])
    xo = g["sign_offset"] * t * drift
    P = np.column_stack([np.asarray(c["x"], float), np.asarray(c["y"], float),
                         np.asarray(c["z"], float)])
    U = g["s"] * (P[:, 0] + xo - g["anode_x"])
    end_u = U.min()
    if end_u > UMAX_END:
        return None
    # local direction near the anode end, in (u,y,z)
    W = np.column_stack([U, P[:, 1], P[:, 2]])
    sel = U < end_u + NEAR
    if sel.sum() < 10:
        return None
    Wn = W[sel]
    ctr = Wn.mean(0)
    _, _, vt = np.linalg.svd(Wn - ctr, full_matrices=False)
    dirv = vt[0]
    if dirv[0] > 0: dirv = -dirv          # point toward decreasing u
    p0 = Wn[np.argmin(Wn[:, 0])]          # anode-most cluster point
    # img-global points of the whole event
    zp = zipfile.ZipFile(os.path.join(WORK, "039252_%d_%s" % (ev_idx, TAG), "mabc-all-apa.zip"))
    di = json.loads(zp.read([n for n in zp.namelist() if n.endswith("0-img-global.json")][0]))
    q = np.asarray(di["q"], float)
    Pi = np.column_stack([np.asarray(di["x"], float), np.asarray(di["y"], float),
                          np.asarray(di["z"], float)])[q > 0]
    Ui = g["s"] * (Pi[:, 0] + xo - g["anode_x"])
    Wi = np.column_stack([Ui, Pi[:, 1], Pi[:, 2]])
    # points beyond the cluster end, along the line, within the tube
    ahead = Wi[Wi[:, 0] < end_u + 2.0]
    if len(ahead):
        rel = ahead - p0
        s = rel @ dirv
        perp = np.linalg.norm(rel - np.outer(s, dirv), axis=1)
        ontube = ahead[(perp < TUBE) & (s > -5)]
    else:
        ontube = np.empty((0, 3))
    beyond = ontube[ontube[:, 0] < end_u - 1.0]
    img_min = float(beyond[:, 0].min()) if len(beyond) >= 3 else None
    # entry point (u=0) interior?
    steps = (0.0 - p0[0]) / dirv[0] if dirv[0] != 0 else 1e9
    entry = p0 + steps * dirv
    interior = (g["y_lo"]+10 < entry[1] < g["y_hi"]-10 and g["z_lo"]+10 < entry[2] < g["z_hi"]-10)
    return dict(ev=ev, uid=uid, gid=gid, kind=kind, side="bot" if c["apa"]==0 else "top",
                end_u=end_u, img_min=img_min, n_beyond=len(beyond),
                entry_y=entry[1], entry_z=entry[2], interior=interior,
                dip=np.degrees(np.arcsin(abs(dirv[0]))))

# build the target list: validated boundary a->c tracks + crosser halves
targets = []
for cal in sorted(glob.glob(os.path.join(WORK, "039252_*_%s" % TAG, "calib-evt*.json"))):
    idx = int(re.match(r"039252_(\d+)_", os.path.basename(os.path.dirname(cal))).group(1))
    ev = os.path.basename(cal)[len("calib-evt"):-len(".json")]
    for r in read_dec(os.path.join(DEC, "decisions-boundary", "decisions-evt%s.jsonl" % ev)):
        targets.append((idx, ev, r["main_cluster_uid"], r["flash_gid"], "boundary"))
    for r in read_dec(os.path.join(DEC, "decisions-crossers", "decisions-evt%s.jsonl" % ev)):
        targets.append((idx, ev, r["main_cluster_uid"], r["flash_gid"], "crosser"))

res = []
for idx, ev, uid, gid, kind in targets:
    try:
        r = probe(idx, ev, uid, gid, kind)
    except Exception as e:
        print("ERR", ev, uid, e); continue
    if r: res.append(r)

print("%-8s %-9s %-4s %-8s | cluster end_u | img-line min_u (n) | entry y/z interior | dip deg" % ("kind","ev","side","uid"))
for r in sorted(res, key=lambda r: r["end_u"]):
    print("%-8s %-9s %-4s %-8d |   %+7.2f    |  %s (n=%d) | %s (y%+.0f z%.0f) | %4.1f" % (
        r["kind"], r["ev"], r["side"], r["uid"], r["end_u"],
        ("%+7.2f" % r["img_min"]) if r["img_min"] is not None else "  none ",
        r["n_beyond"], "Y" if r["interior"] else "n", r["entry_y"], r["entry_z"], r["dip"]))
gaps = [ (r["img_min"] if r["img_min"] is not None else r["end_u"]) for r in res if r["interior"]]
print("\ninterior-entry tracks: n=%d  final gap (img-aware) median %+5.2f  [%+.2f..%+.2f]" %
      (len(gaps), np.median(gaps), min(gaps), max(gaps)))

# --- angle / side summary over interior-entry tracks with end_u in [0,12] ---
sel = [r for r in res if r["interior"] and 0.0 <= r["end_u"] <= 12.0]
print("\ninterior-entry tracks, end_u in [0,12]: n=%d" % len(sel))
for lo, hi in ((0, 25), (25, 45), (45, 60), (60, 90)):
    b = [r["end_u"] for r in sel if lo <= r["dip"] < hi]
    if b:
        print("  dip %2d-%2d deg: n=%2d  gap median %+5.2f" % (lo, hi, len(b), np.median(b)))
for side in ("bot", "top"):
    b = [r["end_u"] for r in sel if r["side"] == side]
    if b:
        print("  side %s: n=%2d  gap median %+5.2f" % (side, len(b), np.median(b)))
if len(sel) > 2:
    dd = np.array([r["dip"] for r in sel]); uu = np.array([r["end_u"] for r in sel])
    print("  corr(dip, gap) = %+.2f" % np.corrcoef(dd, uu)[0, 1])
