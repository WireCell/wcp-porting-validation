#!/usr/bin/env python3
"""High-statistics extension of §8.9: W-plane-only signal-stop u at the anode
for the FULL validated boundary sample, per side (BDE/bottom vs TDE/top).

Where check_anode_t0_ensemble.py hand-picked 10 tracks, this scans every
hand-validated boundary track (ql_display/decisions-boundary, verdict
keep/add) whose imaging anode end sits near the grid plane, and for each
traces the W (collection) deconvolved corridor exactly as §8.9 did:
constant-tick dt0 calibration on the track points, windowed ridge march
(±4 ch / ±8 ticks) from 5 cm on-track down through u = −4, then a
contiguity walk (gaps ≤ 1.1 cm) to the last u with gauss > 1500 (raw > 40).

W-PLANE ONLY: the corridor reads frame_gauss<anode>/frame_raw<anode> (the
collection-plane deconvolved output); nothing here uses the 3-D imaging
endpoint except to seed the corridor start.  Contrast check A/A' of
check_anode_time_consistency.py, which measure the all-plane 3-D imaging
PCA end.

Usage (from this directory):
  python3 scripts/check_anode_stop_ensemble.py            # all 18 events, boundary
  python3 scripts/check_anode_stop_ensemble.py --margin 8 # tighten anode-end gate
"""
import argparse
import glob
import io
import json
import os
import sys
import tarfile

import numpy as np

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/img_plot")
import geom  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
DEC = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/ql_display"
STORE = geom.load_store(
    "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2")

RUN = "039252"  # run prefix for work-dir/decisions globs; override with --run

PG = {}


def pg(a, f):
    if (a, f) not in PG:
        PG[(a, f)] = geom.PlaneGeom(STORE, a, f, 2)
    return PG[(a, f)]


def which_af(y, z, side):
    for a in ((0, 1, 2, 3) if side == "bot" else (4, 5, 6, 7)):
        for f in (0, 1):
            g = pg(a, f)
            ys = np.concatenate([g.tails[:, 0], g.heads[:, 0]])
            zs = np.concatenate([g.tails[:, 1], g.heads[:, 1]])
            if ys.min() - 0.3 <= y <= ys.max() + 0.3 and \
               zs.min() - 0.3 <= z <= zs.max() + 0.3:
                return a, f
    return None, None


def chan_tick(p, side, V):
    a, f = which_af(p[1], p[2], side)
    if a is None:
        return None
    g = pg(a, f)
    centers = 0.5 * (g.tails + g.heads)
    k = int(round((p[2] - centers[0, 1]) / g.pitch_cm))
    if not (0 <= k < len(g.chans)):
        return None
    dirx = 1 if side == "bot" else -1
    ti = (p[0] - geom.wplane_x_cm(STORE, a, f)) * dirx / V / 0.5
    return a, int(g.chans[k]), int(round(ti))


FR = {}


def load_frames(pdir, anode, ev):
    key = (pdir, anode)
    if key in FR:
        return FR[key]
    out = {}
    tp = os.path.join(pdir, "protodune-sp-dnnroi-frames-anode%d.tar.bz2" % anode)
    if not os.path.isfile(tp):
        FR[key] = out
        return out
    with tarfile.open(tp) as tf:
        for m in tf.getmembers():
            for tag0, tag in (("gauss", "gauss%d" % anode),
                              ("raw", "raw%d" % anode)):
                for kind in ("frame", "channels"):
                    if m.name == "%s_%s_%s.npy" % (kind, tag, ev):
                        out[(kind, tag0)] = np.load(
                            io.BytesIO(tf.extractfile(m).read()))
    FR[key] = out
    return out


def find_dirs(ev, tag):
    for d in sorted(glob.glob(os.path.join(WORK, "%s_*_%s" % (RUN, tag)))):
        if os.path.isfile(os.path.join(d, "calib-evt%s.json" % ev)):
            # Frames live in the tagged dir when it regenerated them (e.g.
            # the ctoffset reprocess); the QL-only _anodefix pass reused the
            # production dir's frames, so fall back to the stripped dir.
            if glob.glob(os.path.join(
                    d, "protodune-sp-dnnroi-frames-anode*.tar.bz2")):
                return d, d
            return d, d[:-len("_%s" % tag)]
    return None, None


def read_decisions(sub, ev):
    p = os.path.join(DEC, "decisions-%s" % sub, "decisions-evt%s.jsonl" % ev)
    out = []
    if os.path.isfile(p):
        for ln in open(p):
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                if r["verdict"] in ("keep", "add"):
                    out.append(r)
    return out


def trace(ev, uid, gid, tag):
    adir, pdir = find_dirs(ev, tag)
    if adir is None:
        return {"skip": "no-dir"}
    d = json.load(open(os.path.join(adir, "calib-evt%s.json" % ev)))
    V = d["drift_speed"]
    fb = {x["gid"]: x for x in d["flashes"]}
    cb = {cc["uid"]: cc for cc in d["clusters"]}
    if uid not in cb or gid not in fb:
        return {"skip": "uid/gid-absent"}
    c = cb[uid]
    g = {int(k): v for k, v in d["geometry"].items()}[c["apa"]]
    offs = d["trigger_offsets_us"]
    t = fb[gid]["time"] if c["apa"] == 0 else \
        fb[gid].get("time1", fb[gid]["time"] + offs[1] - offs[0])
    side = "bot" if c["apa"] == 0 else "top"
    P = np.column_stack([np.asarray(c["x"], float),
                         np.asarray(c["y"], float),
                         np.asarray(c["z"], float)])
    if len(P) < 5:
        return {"skip": "too-few-points"}
    xo = g["sign_offset"] * t * V
    U = g["s"] * (P[:, 0] + xo - g["anode_x"])
    end_u = U.min()                       # anode (min-u) end
    sel = U < end_u + 60
    Pn = P[sel]
    ctr = Pn.mean(0)
    _, _, vt = np.linalg.svd(Pn - ctr, full_matrices=False)
    dirr = vt[0]
    if dirr[0] * g["s"] > 0:
        dirr = -dirr
    if abs(dirr[0]) < 0.05:
        return {"skip": "track-parallel-to-wireplane"}
    p_end = Pn[np.argmin(U[sel])]
    # constant tick offset calibration on the track points
    dts = []
    for p in Pn[::max(1, len(Pn) // 80)]:
        r = chan_tick(p, side, V)
        if r is None:
            continue
        a, ch, ti = r
        fr = load_frames(pdir, a, ev)
        if ("channels", "gauss") not in fr:
            continue
        rg = np.where(fr[("channels", "gauss")] == ch)[0]
        if not len(rg) or not (30 <= ti < fr[("frame", "gauss")].shape[1] - 30):
            continue
        row = fr[("frame", "gauss")][rg[0]]
        s = np.array([row[ti + dt - 6:ti + dt + 7].sum() for dt in range(-20, 21)])
        if s.max() > 500:
            dts.append(int(s.argmax()) - 20)
    dt0 = int(np.median(dts)) if dts else 0
    # corridor march: u from end+5 down to -4
    prof = []
    for s in np.arange(-10, (end_u + 4) / 0.5 + 1) * (0.5 / abs(dirr[0])):
        p = p_end + s * dirr
        u = g["s"] * (p[0] + xo - g["anode_x"])
        r = chan_tick(p, side, V)
        if r is None:
            continue
        a, ch, ti = r
        ti += dt0
        fr = load_frames(pdir, a, ev)
        if ("channels", "gauss") not in fr:
            continue
        chg = fr[("channels", "gauss")]
        Fg = fr[("frame", "gauss")]
        Frw = fr[("frame", "raw")]
        rg = np.where(chg == ch)[0]
        if not len(rg) or not (30 <= ti < Fg.shape[1] - 30):
            continue
        best = (0.0, 0, 0)
        for dc in range(-4, 5):
            rr = rg[0] + dc
            if not (0 <= rr < Fg.shape[0]):
                continue
            row = Fg[rr]
            for dt in range(-8, 9):
                s13 = row[ti + dt - 6:ti + dt + 7].sum()
                if s13 > best[0]:
                    best = (float(s13), dc, dt)
        gsum, dc, dt = best
        rr = np.where(fr[("channels", "raw")] == ch)[0]
        if not len(rr):
            continue
        rr = rr[0] + dc
        rwin = Frw[rr, ti + dt - 10:ti + dt + 11]
        prof.append((u, gsum, float(rwin.max())))
    if not prof:
        return {"skip": "no-corridor"}

    def walk(idx_val, thr):
        stop = None
        for u, val in sorted(idx_val, key=lambda r: -r[0]):
            if u > end_u + 0.6:
                continue
            if val > thr:
                if stop is None or stop - u <= 1.1:
                    stop = u
            elif stop is not None and stop - u > 1.1:
                break
        return stop if stop is not None else float("nan")

    u_g = walk([(u, gs) for (u, gs, rp) in prof], 1500)
    u_r = walk([(u, rp) for (u, gs, rp) in prof], 40)
    return dict(side=side, end_u=end_u, u_gauss=u_g, u_raw=u_r, dt0=dt0, V=V)


def stats(a):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    if len(a) == 0:
        return "n=0"
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return ("n=%3d  median %+6.2f  MAD %5.2f  mean %+6.2f  rms %5.2f "
            "[%+.2f,%+.2f]" % (len(a), med, mad, a.mean(), a.std(),
                               a.min(), a.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="039252")
    ap.add_argument("--tag", default="anodefix")
    ap.add_argument("--margin", type=float, default=12.0,
                    help="keep tracks whose imaging anode end u <= margin (cm)")
    ap.add_argument("--subs", default="boundary,crossers")
    ap.add_argument("--sample", default="decisions",
                    choices=("decisions", "bundles"),
                    help="track source.  'decisions' = hand-validated "
                         "boundary/crossers (uids must match the dump's "
                         "clustering -- use for the ORIGINAL/_anodefix pass). "
                         "'bundles' = all auto-selected span>=30 bundles in the "
                         "dump (uid-independent -- use for RE-CLUSTERED passes "
                         "like _ctoff whose uids differ from the decisions).")
    args = ap.parse_args()
    global RUN
    RUN = args.run

    events = sorted({os.path.basename(f)[len("calib-evt"):-len(".json")]
                     for f in glob.glob(os.path.join(
                         WORK, "%s_*_%s" % (RUN, args.tag), "calib-evt*.json"))})

    def pairs_for(ev):
        """Yield (uid, gid) track seeds for one event per --sample mode."""
        if args.sample == "decisions":
            for sub in args.subs.split(","):
                for r in read_decisions(sub, ev):
                    yield r["main_cluster_uid"], r["flash_gid"]
            return
        adir, _ = find_dirs(ev, args.tag)
        if adir is None:
            return
        d = json.load(open(os.path.join(adir, "calib-evt%s.json" % ev)))
        cb = {c["uid"]: c for c in d["clusters"]}
        for b in d.get("bundles", []):
            uid, gid = b.get("main_cluster"), b.get("flash_gid")
            if not b.get("auto_selected") or uid == 3999999 or uid not in cb:
                continue
            cc = cb[uid]
            span = float(np.linalg.norm([max(cc["x"]) - min(cc["x"]),
                                         max(cc["y"]) - min(cc["y"]),
                                         max(cc["z"]) - min(cc["z"])]))
            if span >= 30.0:
                yield uid, gid

    acc = {"gauss": {"bot": [], "top": []}, "raw": {"bot": [], "top": []},
           "img": {"bot": [], "top": []}}
    excl = {}
    rows = []
    seen = set()
    for ev in events:
        if True:
            for uid, gid in pairs_for(ev):
                if (ev, uid) in seen:
                    continue
                seen.add((ev, uid))
                out = trace(ev, uid, gid, args.tag)
                if "skip" in out:
                    excl[out["skip"]] = excl.get(out["skip"], 0) + 1
                    continue
                # physicality cleaning (mirror check_cathode_velocity)
                if out["end_u"] > args.margin:
                    excl["img-end-too-deep"] = excl.get("img-end-too-deep", 0) + 1
                    continue
                if abs(out["dt0"]) >= 18:
                    excl["dt0-saturated"] = excl.get("dt0-saturated", 0) + 1
                    continue
                if not np.isfinite(out["u_gauss"]):
                    excl["no-gauss-stop"] = excl.get("no-gauss-stop", 0) + 1
                    continue
                acc["gauss"][out["side"]].append(out["u_gauss"])
                acc["raw"][out["side"]].append(out["u_raw"])
                acc["img"][out["side"]].append(out["end_u"])
                rows.append((ev, uid, out["side"], out["end_u"],
                             out["u_gauss"], out["u_raw"], out["dt0"]))

    print("== per-track anode ends (u cm; 0 = shield FV anode boundary "
          "|x|=339.91cm, + = into volume toward cathode) ==")
    for ev, uid, side, eu, ug, ur, dt0 in rows:
        print("  evt%s uid %-8d %s | img end %+5.2f | gauss %+5.2f | raw %+5.2f "
              "| dt0 %+d" % (ev, uid, side, eu, ug, ur, dt0))
    print("\n== exclusions ==")
    for k, v in sorted(excl.items()):
        print("  %-28s %d" % (k, v))
    print("\n== SUMMARY: anode ends per side (u=0 = shield FV boundary) ==")
    print("   img = 3-D imaging endpoint; gauss/raw = W-plane signal stop")
    for kind in ("img", "gauss", "raw"):
        for side in ("bot", "top"):
            print("  %-5s %s : %s" % (kind, side, stats(acc[kind][side])))
    print("  --")
    for kind in ("img", "gauss", "raw"):
        d = acc[kind]["bot"] and acc[kind]["top"]
        if d:
            mb = np.median([x for x in acc[kind]["bot"] if np.isfinite(x)])
            mt = np.median([x for x in acc[kind]["top"] if np.isfinite(x)])
            print("  %-5s bot-top median split : %+.2f cm" % (kind, mb - mt))

    json.dump({kind: {s: acc[kind][s] for s in ("bot", "top")}
               for kind in ("img", "gauss", "raw")},
              open(os.path.join(HERE, "anode_stop_ensemble.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
