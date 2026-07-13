#!/usr/bin/env python3
"""Drift-velocity calibration from validated cathode-crossing tracks, using
the W-plane deconvolved (gauss) signal directly -- NOT the 3-D imaging.

Follows the anode/T0 calibration of pdvd-anode-time-consistency.md sec 8.9:
with the anode position and per-side time base now demonstrated (signal
stops at the U plane to ~1-2 cm at the validated flash times), the ONLY
free scale left in the drift direction is the velocity, and the cathode
end of a cathode-crossing track is its meter (D*(dv/v) ~ 3.4 cm per 1%).

Method, per validated crosser half (decisions-crossers, verdict keep/add,
45 xTPC pairs = 45 bottom + 45 top half-tracks in 10 events of run 039252):

  1. fold the pair's hand-validated flash time per side (time / time1) and
     compute u = s*(x_raw + sign_offset*t*v - anode_x) for the cluster
     points (u = 0 at the U-plane anode, u_cathode = 338.51 at the cathode
     SURFACE, 60 mm block);
  2. take the cathode-side end (max u), local direction from the last
     60 cm, calibrate the constant tick offset dt0 on actual points
     (windowed ridge match, as sec 8.8/8.9);
  3. march the W-plane corridor through and past the imaging end (windowed
     ridge search +-4 ch, +-8 ticks; 13-tick gauss sums, 21-tick raw
     windows) and contiguity-walk (gaps <= 1.1 cm) UP in u to the last
     sample above threshold: u_stop (gauss > 1500; raw peak > 40);
  4. velocity: the per-side ANODE stop median b_side (per tag, see
     B_ANODE_BY_TAG) anchors the time base.  Any common time-base error --
     including the sec 2.3 SP intrinsic-shift excess -- shifts
     u_stop_cathode and b_side identically and CANCELS in the difference, so

         v_true = v_assumed * D_c / (u_stop_cathode - b_side),
         D_c = u_cathode = 338.51 cm (U plane -> cathode surface).

Sec 8.12 update (--tag ctoff, now the default): the ctoffset = -5.5 us
reprocess pins the anode edge at u ~ 0 on both sides (sec 8.11), so the
anchored and DIRECT metrics coincide and the owner-chosen calibration
target is the DIRECT distance:

    choose v so that median(D_c - u_stop * v/V) = 0  per side
    =>  v_pin(side) = V * D_c / median(u_stop, side).

This deliberately absorbs the cathode-end detection shortfall delta_c into
v as a convention -- the exact cathode-side analogue of sec 8.11 absorbing
the anode-end shortfall into ctoffset.  The gauss walk is primary (same
walk that set the anode pin); the raw walk is the cross-check.

uid bridge (re-clustered tags): the decisions-file uids belong to the OLD
production clustering; on a re-clustered tag (_ctoff) they no longer map.
Each decisions row is bridged by its validated (flash_gid, apa): candidate
clusters are the calib bundles' main_cluster on the same flash and side,
tried in (auto_selected, npoints) order until one passes the trace's
physicality cuts.  Flash gids are verified stable across the reprocess
(sec 8.11: 197/197 identical times, light chain untouched).

Circularity caveat (owner-stated): the crosser sample and its flash
assignments were selected at v = 1.568; a ~1% velocity error moves the
meeting point x_mid by ~1.7 cm per half -- well inside the validation
tolerance -- so the selection does not pin v and the measurement is not
vacuous, but a large (>3%) error could have demoted genuine pairs.

Usage (from this directory):
  python3 check_cathode_velocity.py                       # _ctoff (sec 8.12)
  python3 check_cathode_velocity.py --tag anodefix        # historical sec 8.10
                                                          # (inputs deleted in the
                                                          # 2026-07-12 work/ cleanup;
                                                          # numbers frozen in the doc)
Outputs are tag-suffixed (cathode_velocity_stops_<tag>.png / _tracks_<tag>.json);
the frozen sec 8.10 outputs (no suffix) are never overwritten.
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORK = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/work"
DEC = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/ql_display/decisions-crossers"
STORE = geom.load_store(
    "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunevd-wires-larsoft-v5.json.bz2")

RUN = "039252"   # set from --run
TAG = "ctoff"    # set from --tag

# Per-side anode signal-stop medians (time-base anchors), per tag:
#  - anodefix: the sec 8.9 ensemble (10 hand-picked tracks + boundary scan).
#  - ctoff: the sec 8.11 check-2 same-track/validated-flash medians (gauss
#    bot n=4 / top n=11) -- the ctoffset calibration pins these at ~0 by
#    construction.  The post-shift raw-walk anchors were not separately
#    re-measured (the _anodefix inputs the bridge needed are gone); raw
#    anchors are set 0, so anchored-raw is indicative only.  NOTE the
#    auto-bundle ensemble on ctoff (--sample bundles --margin 3) sits at
#    gauss -1.18/-1.42 instead -- that population uses AUTO-matched flashes
#    and inherits the sec 8.11 check-3 re-matching churn; the validated
#    numbers are the calibration reference.
B_ANODE_BY_TAG = {
    "anodefix": {"gauss": {"bot": +1.60, "top": -0.20},
                 "raw":   {"bot": +1.19, "top": -0.48}},
    "ctoff":    {"gauss": {"bot": +0.09, "top": +0.08},
                 "raw":   {"bot": 0.0, "top": 0.0}},
    # vcal = ctoff + drift speed 1.568->1.586 (sec 8.12).  The anode pin is
    # scale-invariant (u ~ 0), so the ctoff anchors carry over unchanged.
    "vcal":     {"gauss": {"bot": +0.09, "top": +0.08},
                 "raw":   {"bot": 0.0, "top": 0.0}},
}
GAUSS_THR, RAW_THR = 1500.0, 40.0


def find_dirs(ev):
    for d in sorted(glob.glob(os.path.join(WORK, "%s_*_%s" % (RUN, TAG)))):
        if os.path.isfile(os.path.join(d, "calib-evt%s.json" % ev)):
            # Frames live in the tagged dir when the tag regenerated them
            # (e.g. _ctoff); the QL-only _anodefix pass reused the production
            # dir's frames, so fall back to the stripped dir.
            if glob.glob(os.path.join(
                    d, "protodune-sp-dnnroi-frames-anode*.tar.bz2")):
                return d, d
            return d, d[:-len("_%s" % TAG)]
    return None, None


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
            if (ys.min() - 0.3 <= y <= ys.max() + 0.3
                    and zs.min() - 0.3 <= z <= zs.max() + 0.3):
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
    path = os.path.join(pdir, "protodune-sp-dnnroi-frames-anode%d.tar.bz2" % anode)
    if os.path.isfile(path):
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                for tag0, tag in (("gauss", "gauss%d" % anode),
                                  ("raw", "raw%d" % anode)):
                    for kind in ("frame", "channels"):
                        if m.name == "%s_%s_%s.npy" % (kind, tag, ev):
                            out[(kind, tag0)] = np.load(
                                io.BytesIO(tf.extractfile(m).read()))
    FR[key] = out
    return out


CAL = {}


def load_calib(ev):
    if ev in CAL:
        return CAL[ev]
    adir, pdir = find_dirs(ev)
    if adir is None:
        CAL[ev] = (None, None, None)
    else:
        d = json.load(open(os.path.join(adir, "calib-evt%s.json" % ev)))
        CAL[ev] = (d, adir, pdir)
    return CAL[ev]


def load_rows():
    rows, seen = [], set()
    for p in sorted(glob.glob(os.path.join(DEC, "decisions-evt*.jsonl"))):
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r["verdict"] not in ("keep", "add"):
                continue
            key = (r["event"], r["main_cluster_uid"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def bridge_candidates(d, gid, apa):
    """Candidate cluster uids on a re-clustered tag for a validated
    (flash_gid, apa): the calib bundles' main_cluster on the same flash and
    side, auto-selected first, then by descending size."""
    cb = {c["uid"]: c for c in d["clusters"]}
    cands = {}
    for b in d.get("bundles", []):
        if b.get("flash_gid") != gid:
            continue
        uid = b.get("main_cluster")
        if uid in (None, 3999999) or uid not in cb:
            continue
        if cb[uid]["apa"] != apa:
            continue
        auto = bool(b.get("auto_selected"))
        cands[uid] = max(cands.get(uid, False), auto)
    return sorted(cands,
                  key=lambda u: (not cands[u], -cb[u]["npoints"]))


def trace(ev, uid, gid):
    """Return dict with u_end (imaging cathode end), u_stop gauss/raw, etc."""
    d, adir, pdir = load_calib(ev)
    if d is None:
        return dict(skip="no-workdir")
    V = d["drift_speed"]
    fb = {x["gid"]: x for x in d["flashes"]}
    cb = {c["uid"]: c for c in d["clusters"]}
    if uid not in cb or gid not in fb:
        return dict(skip="missing-uid-or-gid")
    c = cb[uid]
    g = {int(k): v for k, v in d["geometry"].items()}[c["apa"]]
    offs = d["trigger_offsets_us"]
    t = fb[gid]["time"] if c["apa"] == 0 else \
        fb[gid].get("time1", fb[gid]["time"] + offs[1] - offs[0])
    side = "bot" if c["apa"] == 0 else "top"
    P = np.column_stack([np.asarray(c["x"], float),
                         np.asarray(c["y"], float),
                         np.asarray(c["z"], float)])
    if len(P) < 20:
        return dict(skip="tiny-cluster")
    xo = g["sign_offset"] * t * V
    U = g["s"] * (P[:, 0] + xo - g["anode_x"])
    end_u = float(U.max())
    Dc = g["u_cathode"]
    if end_u < Dc - 25:
        return dict(skip="imaging-end-far-from-cathode", end_u=end_u,
                    side=side)
    sel = U > end_u - 60
    Pn = P[sel]
    _, _, vt = np.linalg.svd(Pn - Pn.mean(0), full_matrices=False)
    dirr = vt[0]
    if dirr[0] * g["s"] < 0:          # orient so u increases along dirr
        dirr = -dirr
    if abs(dirr[0]) < 0.05:
        return dict(skip="perp-to-drift", side=side)
    p_end = Pn[np.argmax(U[sel])]
    # tick-offset calibration on actual points of the end segment
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
        s = np.array([row[ti + dt - 6:ti + dt + 7].sum()
                      for dt in range(-20, 21)])
        if s.max() > 500:
            dts.append(int(s.argmax()) - 20)
    if len(dts) < 5:
        return dict(skip="dt0-uncalibratable", side=side)
    dt0 = int(np.median(dts))
    # corridor: u from end_u-12 up through max(end_u, Dc)+8
    prof, n_oob = [], 0
    s_lo = -12.0 / abs(dirr[0])
    s_hi = (max(0.0, Dc - end_u) + 8.0) / abs(dirr[0])
    for s in np.arange(s_lo, s_hi + 1e-9, 0.5 / abs(dirr[0])):
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
        if not len(rg):
            continue
        if not (30 <= ti < Fg.shape[1] - 30):
            n_oob += 1
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
        rr = np.where(fr[("channels", "raw")] == ch)[0][0] + dc
        rwin = Frw[rr, ti + dt - 10:ti + dt + 11]
        prof.append((u, gsum, float(rwin.max())))
    if n_oob > 0 and (not prof or max(p[0] for p in prof) < Dc + 2):
        return dict(skip="tick-window-truncated", side=side, end_u=end_u)
    if len(prof) < 10:
        return dict(skip="corridor-empty", side=side)

    def walk_up(idx_val, thr):
        stop = None
        for u, val in sorted(idx_val, key=lambda r: r[0]):
            if u < end_u - 0.6:
                continue
            if val > thr:
                if stop is None or u - stop <= 1.1:
                    stop = u
            elif stop is not None and u - stop > 1.1:
                break
        return stop if stop is not None else float("nan")

    u_g = walk_up([(u, gv) for u, gv, rv in prof], GAUSS_THR)
    u_r = walk_up([(u, rv) for u, gv, rv in prof], RAW_THR)
    return dict(side=side, t=t, end_u=end_u, u_gauss=u_g, u_raw=u_r,
                dt0=dt0, Dc=Dc, dirx=abs(dirr[0]), V=V)


def clean_flag(out):
    """Physicality/quality cuts.  Signal contiguous BEYOND the cathode
    surface by more than smearing (+5 cm) cannot be this track's drift
    charge -- it is a cross-T0 over-merge or corridor contamination (the
    same tail that corrupted the old span pile-up, sec 8.6).  dt0 at the
    +-20 scan edge means the tick calibration failed."""
    Dc = out["Dc"]
    if out["end_u"] > Dc + 5:
        return "img-end-beyond-full-drift"
    if abs(out["dt0"]) >= 18:
        return "dt0-scan-saturated"
    if not np.isfinite(out["u_gauss"]):
        return "no-gauss-stop"
    if out["u_gauss"] > Dc + 5 or (np.isfinite(out["u_raw"])
                                   and out["u_raw"] > Dc + 5):
        return "stop-beyond-full-drift"
    return None


def main():
    global RUN, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="039252")
    ap.add_argument("--tag", default="ctoff",
                    help="work-dir tag (calib + frames); decisions uids are "
                         "bridged via (flash_gid, apa) when the tag was "
                         "re-clustered")
    args = ap.parse_args()
    RUN, TAG = args.run, args.tag
    if TAG not in B_ANODE_BY_TAG:
        sys.exit("no B_ANODE anchors defined for tag %r -- add them from "
                 "check_anode_stop_ensemble.py --tag %s" % (TAG, TAG))
    B_ANODE = B_ANODE_BY_TAG[TAG]

    rows = load_rows()
    print("validated crosser halves (keep/add, deduped): %d" % len(rows))
    res, skips = [], {}
    used_ct = set()
    for r in rows:
        ev = r["event"][len("evt"):]
        d, adir, pdir = load_calib(ev)
        if d is None:
            skips["no-workdir"] = skips.get("no-workdir", 0) + 1
            continue
        gid = r["flash_gid"]
        out, uid_used = None, None
        for uid in bridge_candidates(d, gid, r["apa"]):
            if (ev, uid) in used_ct:
                continue
            o = trace(ev, uid, gid)
            if "skip" not in o:
                out, uid_used = o, uid
                break
            out = o    # remember last skip reason
        if out is None:
            out = dict(skip="bridge-no-candidate")
        if "skip" in out:
            skips[out["skip"]] = skips.get(out["skip"], 0) + 1
            print("  skip %-38s evt%s dec-uid %d gid %d" %
                  (out["skip"], ev, r["main_cluster_uid"], gid))
            continue
        used_ct.add((ev, uid_used))
        out.update(ev=ev, uid=uid_used, dec_uid=r["main_cluster_uid"],
                   gid=gid)
        out["flag"] = clean_flag(out)
        res.append(out)
        print("evt%s uid %-8d %s gid %3d t %+9.2f | img end %7.2f | "
              "gauss stop %7.2f | raw stop %7.2f | dt0 %+3d | |dirx| %.2f%s" %
              (ev, out["uid"], out["side"], out["gid"], out["t"],
               out["end_u"], out["u_gauss"], out["u_raw"], out["dt0"],
               out["dirx"], "  EXCL:" + out["flag"] if out["flag"] else ""))
    print("\nskips:", skips)
    from collections import Counter
    print("exclusions:", dict(Counter(x["flag"] for x in res if x["flag"])))
    res = [x for x in res if x["flag"] is None]
    print("clean halves: %d (%d bot, %d top)" %
          (len(res), sum(x["side"] == "bot" for x in res),
           sum(x["side"] == "top" for x in res)))

    Dc = res[0]["Dc"] if res else 338.51
    V = res[0]["V"] if res else 0.1568
    print("\nD_c (U plane -> cathode surface) = %.2f cm ; v_assumed = %.4f "
          "cm/us = %.3f mm/us" % (Dc, V, 10 * V))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    summary = {}
    for kind in ("gauss", "raw"):
        print("\n== %s-walk cathode stops and velocity ==" % kind)
        for side in ("bot", "top"):
            us = np.array([x["u_%s" % kind] for x in res
                           if x["side"] == side
                           and np.isfinite(x["u_%s" % kind])])
            if not len(us):
                continue
            b = B_ANODE[kind][side]
            med = np.median(us)
            mad = np.median(np.abs(us - med))
            sem = 1.4826 * mad / np.sqrt(len(us))
            span = med - b
            v = V * Dc / span
            dv_stat = v * sem / span
            summary[(kind, side)] = (len(us), med, mad, b, span, v, dv_stat)
            print("  %s: n=%2d  u_stop median %7.2f (MAD %.2f, sem %.2f)  "
                  "b_anode %+5.2f  span %7.2f  ->  v = %.4f +- %.4f mm/us" %
                  (side, len(us), med, mad, sem, b, span,
                   10 * v, 10 * dv_stat))
            print("      sorted stops: %s" %
                  " ".join("%.1f" % u for u in sorted(us)))
        # pooled two-side value (the single-velocity candidate)
        us_all = np.array([x["u_%s" % kind] for x in res
                           if np.isfinite(x["u_%s" % kind])])
        if len(us_all):
            med = np.median(us_all)
            mad = np.median(np.abs(us_all - med))
            sem = 1.4826 * mad / np.sqrt(len(us_all))
            v = V * Dc / med          # direct pin: b ~ 0 by construction
            dv = v * sem / med
            summary[(kind, "all")] = (len(us_all), med, mad, 0.0, med, v, dv)
            print("  ALL: n=%2d  u_stop median %7.2f (MAD %.2f, sem %.2f)  "
                  "->  v_pin(direct) = %.4f +- %.4f mm/us" %
                  (len(us_all), med, mad, sem, 10 * v, 10 * dv))
    # persist per-track results for downstream metrics
    with open("cathode_velocity_tracks_%s.json" % TAG, "w") as f:
        json.dump([{k: (float(v) if isinstance(v, (float, np.floating))
                        else v) for k, v in x.items()} for x in res],
                  f, indent=1)

    # endpoint-distance metric: distance of the cathode-end signal stop to
    # the cathode surface, under velocity hypotheses.  A stop measured at
    # u_stop in the v_assumed frame has drift time u_stop/V, so under a
    # hypothesis v' it sits at u_stop*(v'/V); the anode anchor b_side sits
    # at ~zero drift time and is scale-invariant to < 0.01 cm.
    #   d_direct   = D_c - u_stop'            (+ = short of the cathode)
    #   d_anchored = D_c - (u_stop' - b_side) (time-base removed; equals
    #                d_direct when the anode is pinned at u ~ 0)
    # v_pin per side: the velocity that zeroes the DIRECT median (owner
    # calibration target, sec 8.12) -- computed from the gauss walk.
    VBEST = {}
    for side in ("bot", "top"):
        if ("gauss", side) in summary:
            med = summary[("gauss", side)][1]
            VBEST[side] = V * Dc / med
    print("\n== per-side v_pin(direct, gauss): %s ==" %
          "  ".join("%s %.4f mm/us" % (s, 10 * vv)
                    for s, vv in VBEST.items()))
    print("\n== endpoint-distance-to-cathode metric (median +- MAD, cm; "
          "+ = short of the cathode surface) ==")
    for kind in ("gauss", "raw"):
        for side in ("bot", "top"):
            us = np.array([x["u_%s" % kind] for x in res
                           if x["side"] == side
                           and np.isfinite(x["u_%s" % kind])])
            if not len(us) or side not in VBEST:
                continue
            b = B_ANODE[kind][side]
            row = ["  %-5s %s:" % (kind, side)]
            for label, vh in (("v=%.3f" % (10 * V), V),
                              ("v_pin", VBEST[side])):
                sc = us * vh / V
                dd = Dc - sc
                da = Dc - (sc - b)
                row.append("%s[%.4f]: direct %+5.2f+-%4.2f anchored "
                           "%+5.2f+-%4.2f" %
                           (label, 10 * vh, np.median(dd),
                            np.median(np.abs(dd - np.median(dd))),
                            np.median(da),
                            np.median(np.abs(da - np.median(da)))))
            print("  ".join(row))

    # angle cross-check on the gauss stops (threshold loss is angle-
    # dependent; a velocity error is not)
    print("\n== angle cross-check (gauss stop vs |dir_x| at the cathode "
          "end) ==")
    for side in ("bot", "top"):
        pts = [(x["dirx"], x["u_gauss"]) for x in res if x["side"] == side
               and np.isfinite(x["u_gauss"])]
        if len(pts) < 5:
            continue
        a = np.array(pts)
        r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
        lo = np.median(a[a[:, 0] < np.median(a[:, 0]), 1])
        hi = np.median(a[a[:, 0] >= np.median(a[:, 0]), 1])
        b = B_ANODE["gauss"][side]
        print("  %s: corr(|dirx|, u_stop) = %+.2f ; median u_stop "
              "low-|dirx| half %7.2f (v %.4f) vs high-|dirx| half %7.2f "
              "(v %.4f)" %
              (side, r, lo, 10 * V * Dc / (lo - b),
               hi, 10 * V * Dc / (hi - b)))

    for ax, kind in zip(axes, ("gauss", "raw")):
        for side, col in (("bot", "tab:blue"), ("top", "tab:red")):
            us = [x["u_%s" % kind] for x in res if x["side"] == side
                  and np.isfinite(x["u_%s" % kind])]
            ax.hist(us, bins=np.arange(325, 346, 0.5), alpha=0.55,
                    color=col, label="%s (n=%d)" % (side, len(us)))
        ax.axvline(Dc, color="k", ls="--", lw=1,
                   label="cathode surface %.2f" % Dc)
        ax.set_xlabel("cathode-end %s-stop u [cm]" % kind)
        ax.set_ylabel("halves")
        ax.legend(fontsize=8)
        ax.set_title("%s walk" % kind, fontsize=10)
    fig.suptitle("validated xTPC crossers, run %s (%s), "
                 "v_assumed = %.3f mm/us" % (RUN, TAG, 10 * V), fontsize=10)
    fig.tight_layout()
    fig.savefig("cathode_velocity_stops_%s.png" % TAG, dpi=110)
    print("\nwrote cathode_velocity_stops_%s.png" % TAG)
    return summary


if __name__ == "__main__":
    main()
