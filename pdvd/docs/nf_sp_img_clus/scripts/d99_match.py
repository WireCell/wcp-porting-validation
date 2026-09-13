#!/usr/bin/env python3
"""doc pdvd/99 -- carry the PDVD STM/Michel hand-scan record onto a re-clustered arm by geometry.

    python3 d99_match.py --arm p98von [--base-arm p96vprod] [--record R] [--out J] [--carried C]
    python3 d99_match.py --arm p96vprod                     # G4a identity control
    python3 d99_match.py --arm p96vprod --shuffle           # G4b negative control

Why: the record is keyed "run_evt/cluster_id".  Every arm it was scanned or graded on reads the
d51vclus clustering (p96vprod's pctree is a symlink into it), so the ids never moved.  A rerun of
SP + imaging + clustering renumbers clusters, and d25_bragg_michel.items()'s A.get(key) would then
attach a verdict to whatever object now carries that id.

Baseline geometry of a key = the image points of its cluster in the BASE arm's Bee dump
(mabc-pr.zip data/0/0-clustering-global.json, the clustering the scanner looked at).  For each
new-arm cluster in the same event:
    f_fwd = fraction of baseline points with a new-cluster point within R_CM
    f_rev = fraction of that new cluster's points with a baseline point within R_CM
Points are drawn at their bundle's t0-corrected x (prep_stm_michel_scan.py:549-554), so a changed
flash match moves a cluster wholesale in x.  If the best f_fwd < F_OK, an x offset is estimated from
(y,z)-nearest pairs (median new_x - base_x over the majority new cluster) and the match is rescored.

Pre-registered thresholds (doc pdvd/99 sec 4; not tuned on any grade):
    R_CM 1.0, F_OK 0.7, second-best >= F_SPLIT 0.2 -> split, f_rev < F_REV 0.5 -> merged,
    best < 2x second -> ambiguous, two keys -> one new cluster -> collision, |dx| > 1 cm -> t0 moved,
    stop displacement > 5 cm -> stop_moved (flag only).

Tags (segment ids = cluster*1000 + graph index of the scan's SOURCE arm) are remapped at cluster
level (sid // 1000, the companion cluster matched the same way; memory rule: join tags per
cluster) and, where the source prep carries the segment's points, at segment level with the
census_score.match_segment rule (mean nearest distance of a //12 subsample < 1.5 cm) against the new
arm's calib-pr segments of the mapped cluster.
"""
import argparse, collections, glob, json, math, os, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import uproot
from scipy.spatial import cKDTree

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
WORK = IMG + "/pdvd/work"
REC = IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_smx5_smx6_smx7_smx8_smx9_verdicts.json"
PREP = IMG + "/pdhd/stm_michel_scan/prep-pdvd"
# record `source` prefix -> its prep set (the arm each tag id belongs to); smx1a = prep-pdvd (d53v)
SRC_PREP = {"smx1a": PREP, "smx3": PREP + "-smx3", "smx4": PREP + "-smx4", "smx5": PREP + "-smx5",
            "smx6": PREP + "-smx6", "smx7": PREP + "-smx7", "smx8": PREP + "-smx8", "smx9": PREP + "-smx9"}
FALLBACK = ["smx9", "smx8", "smx7", "smx6", "smx5", "smx4", "smx3", "smx1a"]
R_CM, F_OK, F_SPLIT, F_REV, DX_T0, STOP_MOVED, SEG_CM = 1.0, 0.7, 0.2, 0.5, 1.0, 5.0, 1.5
SENTINEL = 1.0e6          # a cluster with no t0 is drawn at x ~ 1.48e8 in the Bee dump


def evt_of(key):
    return key.split("/")[0]


def load_bee(evt, arm):
    f = f"{WORK}/{evt}_{arm}/mabc-pr.zip"
    if not os.path.exists(f):
        return None
    j = json.loads(zipfile.ZipFile(f).read("data/0/0-clustering-global.json"))
    xyz = np.c_[j["x"], j["y"], j["z"]].astype(float)
    cid = np.asarray(j["cluster_id"], int)
    return xyz, cid


def load_stm(evt, arm):
    f = f"{WORK}/{evt}_{arm}/tracking-pr.root"
    out = {}
    if not os.path.exists(f):
        return out, {}
    u = uproot.open(f)
    names = [k.split(";")[0] for k in u.keys()]
    if "T_stm_michel" not in names:
        return out, {}
    t = u["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found", "stop_x", "stop_y", "stop_z"],
                                 library="np")
    for i, c in enumerate(t["cluster_id"]):
        out[int(c)] = {k: float(t[k][i]) for k in t}
    p = u["T_stm_michel_pts"].arrays(["cluster_id", "role", "x", "y", "z", "rr"], library="np")
    prof = {}
    for c in np.unique(p["cluster_id"]):
        m = (p["cluster_id"] == c) & (p["role"] == 1)
        if m.any():
            prof[int(c)] = (np.c_[p["x"][m], p["y"][m], p["z"][m]].astype(float), p["rr"][m].astype(float))
    return out, prof


def load_segments(evt, arm):
    """{cluster: [(segment id, points)]} from T_rec_charge of tracking-pr.root and tracking-stm.root --
    the same source prep_stm_michel_scan.particle_flow reads (sub_cluster_id = cluster*1000 + index).
    calib-pr-evt*.json 'segments' holds only the fitted chain (8 segments on 039252_0) and misses
    the companion segments most tags name."""
    pts = collections.defaultdict(list)
    for fn in ("tracking-pr.root", "tracking-stm.root"):
        f = f"{WORK}/{evt}_{arm}/{fn}"
        if not os.path.exists(f):
            continue
        u = uproot.open(f)
        if "T_rec_charge" not in [k.split(";")[0] for k in u.keys()]:
            continue
        t = u["T_rec_charge"].arrays(["x", "y", "z", "sub_cluster_id"], library="np")
        for s in np.unique(t["sub_cluster_id"]):
            if s < 1000:
                continue
            m = t["sub_cluster_id"] == s
            pts[int(s)].append(np.c_[t["x"][m], t["y"][m], t["z"][m]].astype(float))
    segs = collections.defaultdict(list)
    for s, arrs in pts.items():
        segs[s // 1000].append((s, np.unique(np.vstack(arrs), axis=0)))
    return segs


def src_key(r):
    return (r.get("source") or "smx1a").split()[0]


def payload(key, src):
    for s in [src] + [x for x in FALLBACK if x != src]:
        p = "%s/smprep-%s.json" % (SRC_PREP[s], key.replace("/", "-c"))
        if os.path.exists(p):
            return json.load(open(p)), s
    return None, None


def seg_points(key, sid, src):
    """points of tagged segment sid from the key's own source prep, then the fallback chain."""
    for s in [src] + [x for x in FALLBACK if x != src]:
        p = "%s/smprep-%s.json" % (SRC_PREP[s], key.replace("/", "-c"))
        if not os.path.exists(p):
            continue
        for g in json.load(open(p))["pf"]["seg"]:
            if str(g["id"]) == str(sid):
                return np.c_[g["x"], g["y"], g["z"]].astype(float), s
    return None, None


def cover(tree, pts, r):
    """fraction of pts with a tree point within r"""
    if len(pts) == 0:
        return 0.0
    d, _ = tree.query(pts, k=1, distance_upper_bound=r)
    return float(np.mean(np.isfinite(d)))


def match_cluster(B, new_xyz, new_cid, new_tree):
    """B: baseline points.  -> dict(best, second, f_fwd, f2, f_rev, dx, retried)"""
    def score(Bs):
        hits = new_tree.query_ball_point(Bs, R_CM)
        cnt = collections.Counter()
        for h in hits:
            for c in set(new_cid[h].tolist()):
                cnt[c] += 1
        if not cnt:
            return None, 0.0, None, 0.0
        (c1, n1), *rest = cnt.most_common(2)
        c2, n2 = (rest[0] if rest else (None, 0))
        return c1, n1 / len(Bs), c2, n2 / len(Bs)
    dx, retried = 0.0, False
    c1, f1, c2, f2 = score(B)
    if f1 < F_OK:
        yz = cKDTree(new_xyz[:, 1:])
        d, i = yz.query(B[:, 1:], k=1, distance_upper_bound=R_CM)
        ok = np.isfinite(d)
        if ok.sum() >= max(5, 0.3 * len(B)):
            cc = new_cid[i[ok]]
            maj = collections.Counter(cc.tolist()).most_common(1)[0][0]
            sel = cc == maj
            dxs = new_xyz[i[ok][sel], 0] - B[ok][sel, 0]
            # the (y,z) pairs are not the true partners along x; take the offset that maximises coverage
            cands = np.unique(np.round(np.r_[np.median(dxs), dxs[:: max(1, len(dxs) // 200)]], 1))
            best = (f1, 0.0, (c1, f1, c2, f2))
            for cdx in cands:
                s = score(B + np.array([cdx, 0.0, 0.0]))
                if s[1] > best[0]:
                    best = (s[1], float(cdx), s)
            if best[1] != 0.0:
                dx, retried = best[1], True
                c1, f1, c2, f2 = best[2]
    frev = 0.0
    if c1 is not None:
        Bt = cKDTree(B + np.array([dx, 0.0, 0.0]))
        frev = cover(Bt, new_xyz[new_cid == c1], R_CM)
    return dict(best=None if c1 is None else int(c1), second=None if c2 is None else int(c2),
                f_fwd=round(f1, 4), f2=round(f2, 4), f_rev=round(frev, 4), dx=round(dx, 2), retried=retried)


def status_of(m):
    if m["best"] is None or m["f_fwd"] < F_OK:
        return "lost"
    if m["f2"] >= F_SPLIT:
        return "split"
    if m["f_rev"] < F_REV:
        return "merged"
    if m["f2"] > 0 and m["f_fwd"] < 2 * m["f2"]:
        return "ambiguous"
    return "ok"


def do_event(job):
    evt, recs, base_arm, arm, arm_evt = job
    base = load_bee(evt, base_arm)
    new = load_bee(arm_evt, arm)
    if base is None or new is None:
        return [dict(key=r["key"], status="no_arm_event") for r in recs]
    bxyz, bcid = base
    nxyz, ncid = new
    ntree = cKDTree(nxyz)
    bstm, _ = load_stm(evt, base_arm)
    nstm, nprof = load_stm(arm_evt, arm)
    nsegs = load_segments(arm_evt, arm)
    cmatch = {}

    def cluster_map(c):
        if c not in cmatch:
            B = bxyz[bcid == c]
            cmatch[c] = match_cluster(B, nxyz, ncid, ntree) if len(B) else None
        return cmatch[c]

    out = []
    for r in recs:
        key = r["key"]
        c = int(key.split("/")[1])
        m = cluster_map(c)
        if m is None:
            out.append(dict(key=key, status="no_base_cluster"))
            continue
        st = status_of(m)
        row = dict(key=key, status=st, **m)
        row["t0_moved"] = abs(m["dx"]) > DX_T0
        row["new_key"] = None if m["best"] is None else f"{arm_evt}/{m['best']}"
        row["candidate"] = m["best"] in nstm
        b = bstm.get(c)
        if b is not None and row["candidate"]:
            n = nstm[m["best"]]
            row["stop_move_cm"] = round(math.dist((b["stop_x"] + m["dx"], b["stop_y"], b["stop_z"]),
                                                  (n["stop_x"], n["stop_y"], n["stop_z"])), 2)
            row["new_is_stm"], row["new_michel_found"] = int(n["is_stm"]), int(n["michel_found"])
        if b is not None:
            row["base_is_stm"], row["base_michel_found"] = int(b["is_stm"]), int(b["michel_found"])
            row["volume"] = "top" if b["stop_x"] > 0 else "bottom"
        # tags: cluster level, then segment level
        src = src_key(r)
        tags = {}
        for sid, lab in (r.get("tags") or {}).items():
            whole = str(sid).startswith("C")           # "C<cid>" = a whole-cluster tag, no segment
            tc = int(str(sid)[1:]) if whole else int(sid) // 1000
            tm = m if tc == c else cluster_map(tc)
            ent = dict(tag=lab, old_cluster=tc)
            if tm is None or status_of(tm) not in ("ok", "merged"):
                ent.update(new_cluster=None, cluster_status=("no_base_cluster" if tm is None else status_of(tm)))
            elif whole:
                ent.update(new_cluster=tm["best"], cluster_status=status_of(tm), new_seg=None, seg_status="whole_cluster")
            else:
                ent.update(new_cluster=tm["best"], cluster_status=status_of(tm))
                P, psrc = seg_points(key, sid, src)
                if P is None:
                    ent["new_seg"], ent["seg_status"] = None, "no_source_points"
                else:
                    P = P[:: max(1, len(P) // 12)] + np.array([tm["dx"], 0.0, 0.0])
                    best = (None, 1e9)
                    for nid, S in nsegs.get(tm["best"], []):
                        dd = float(np.mean(cKDTree(S).query(P, k=1)[0]))
                        if dd < best[1]:
                            best = (nid, dd)
                    ok = best[0] is not None and best[1] < SEG_CM
                    ent.update(new_seg=best[0] if ok else None, seg_mean_nn_cm=round(best[1], 2) if best[0] else None,
                               seg_status="ok" if ok else "no_segment_within_1.5cm", seg_src=psrc)
            tags[sid] = ent
        row["tags"] = tags
        # pins: placed point (shifted by the cluster's dx) and pin_rr -> xyz on the base profile -> new rr
        pin = r.get("pin") or {}
        if all(k in pin for k in ("x", "y", "z")):
            row["pin_xyz_new"] = [pin["x"] + m["dx"], pin["y"], pin["z"]]
        if r.get("pin_rr") is not None:
            pay, _ = payload(key, src)
            if pay is not None:
                mu = pay["muon"]
                rr = np.asarray(mu["rr"], float)
                j = int(np.argmin(np.abs(rr - float(r["pin_rr"]))))
                row["pin_rr_xyz_new"] = [mu["x"][j] + m["dx"], mu["y"][j], mu["z"][j]]
        for nm in ("pin_xyz_new", "pin_rr_xyz_new"):
            if nm in row and row["candidate"] and m["best"] in nprof:
                X, RR = nprof[m["best"]]
                k = int(np.argmin(np.linalg.norm(X - np.array(row[nm]), axis=1)))
                row[nm.replace("xyz_new", "rr_new")] = round(float(RR[k]), 2)
                row[nm.replace("xyz_new", "dist_to_new_fit_cm")] = round(float(np.linalg.norm(X[k] - row[nm])), 2)
        out.append(row)
    # collisions: two keys whose match (any status that would be carried) is the same new cluster -- the carried
    # record would hold one new key twice with two labels (p98voff first pass: 2 such keys from merged/split items)
    carried = ("ok", "merged", "split", "ambiguous")
    hit = collections.Counter(o["best"] for o in out if o.get("status") in carried)
    for o in out:
        if o.get("status") in carried and hit[o["best"]] > 1:
            o["status_before_collision"] = o["status"]
            o["status"] = "collision"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base-arm", default="p96vprod")
    ap.add_argument("--record", default=REC)
    ap.add_argument("--out", default=None, help="per-key match json")
    ap.add_argument("--carried", default=None, help="write the carried record (new keys) here")
    ap.add_argument("--shuffle", action="store_true", help="G4b: pair each event with another event's arm")
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    rec = json.load(open(a.record))
    by_evt = collections.defaultdict(list)
    for r in rec:
        by_evt[evt_of(r["key"])].append(r)
    evts = sorted(by_evt)
    partner = {e: (evts[(i + 7) % len(evts)] if a.shuffle else e) for i, e in enumerate(evts)}
    jobs = [(e, by_evt[e], a.base_arm, a.arm, partner[e]) for e in evts]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for res in ex.map(do_event, jobs):
            rows.extend(res)
    rows.sort(key=lambda o: o["key"])
    RB = {r["key"]: r for r in rec}
    st = collections.Counter(o["status"] for o in rows)
    print(f"record {os.path.basename(a.record)}  n={len(rec)}  base={a.base_arm}  arm={a.arm}"
          f"{'  SHUFFLED' if a.shuffle else ''}")
    print("status:", dict(sorted(st.items())))
    same = sum(1 for o in rows if o.get("new_key") == o["key"])
    print(f"new_key == old key: {same}/{len(rows)}")
    for vol in ("top", "bottom"):
        s = collections.Counter(o["status"] for o in rows if o.get("volume") == vol)
        print(f"  {vol:6s}", dict(sorted(s.items())))
    for v in ("STM_MICHEL", "STM_ONLY", "THRU", "MESSY", "FRAG_THRU", "UNCLEAR"):
        s = collections.Counter(o["status"] for o in rows if RB[o["key"]]["verdict"] == v)
        print(f"  {v:10s}", dict(sorted(s.items())))
    ok = [o for o in rows if o["status"] == "ok"]
    if ok:
        ff = np.array([o["f_fwd"] for o in ok]); fr = np.array([o["f_rev"] for o in ok])
        dx = np.array([o["dx"] for o in rows if "dx" in o])
        print("ok f_fwd p5/p50 %.3f/%.3f  f_rev p5/p50 %.3f/%.3f" % (np.percentile(ff, 5), np.median(ff),
                                                                   np.percentile(fr, 5), np.median(fr)))
        print("x offsets: retried %d, |dx|>1cm %d, dx p5/p50/p95 %.2f/%.2f/%.2f" % (
            sum(o.get("retried", False) for o in rows), sum(o.get("t0_moved", False) for o in rows),
            np.percentile(dx, 5), np.median(dx), np.percentile(dx, 95)))
        sm = np.array([o["stop_move_cm"] for o in rows if "stop_move_cm" in o])
        if len(sm):
            print("stop move (candidate, matched): n %d  p50 %.2f  p90 %.2f  >5cm %d" % (
                len(sm), np.median(sm), np.percentile(sm, 90), int((sm > STOP_MOVED).sum())))
        print("candidate on new arm: %d of %d matched-any" % (
            sum(o.get("candidate", False) for o in rows), sum(o.get("best") is not None for o in rows)))
    tg = collections.Counter(); tgs = collections.Counter()
    for o in rows:
        for t in (o.get("tags") or {}).values():
            tg[t.get("cluster_status")] += 1
            tgs[t.get("seg_status")] += 1
    print("tags cluster-level:", dict(tg), " segment-level:", dict(tgs))
    pr = [o for o in rows if "pin_rr_rr_new" in o or "pin_rr_new" in o]
    print("pins carried with a new rr: %d" % len(pr))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    if a.carried:
        keep = {"verdict", "michel_kind", "confidence", "tranche", "scan_id", "source", "owner_review", "mech",
                "pin", "pin_rr", "notes", "evidence"}
        car = []
        for o in rows:
            # collision: two labelled objects fused into one new cluster -- their labels conflict and the grader would
            # count the cluster twice; it goes to the re-scan list instead
            if o.get("new_key") is None or o["status"] in ("lost", "collision", "no_arm_event", "no_base_cluster"):
                continue
            r = RB[o["key"]]
            e = {k: r[k] for k in keep if k in r}
            e["key"] = o["new_key"]
            e["carried_from"] = o["key"]
            e["source_record"] = os.path.basename(a.record)
            e["base_arm"], e["target_arm"] = a.base_arm, a.arm
            e["match"] = {k: o.get(k) for k in ("status", "f_fwd", "f2", "f_rev", "dx", "t0_moved",
                                                "stop_move_cm", "candidate")}
            e["tags"] = {str(t["new_seg"]) if t.get("new_seg") is not None else f"C{t['new_cluster']}"
                         if t.get("new_cluster") is not None else f"unmapped:{sid}": t["tag"]
                         for sid, t in (o.get("tags") or {}).items()}
            e["tag_map"] = o.get("tags")
            for nm in ("pin_xyz_new", "pin_rr_new", "pin_rr_xyz_new", "pin_rr_rr_new"):
                if nm in o:
                    e[nm] = o[nm]
            car.append(e)
        dup = [k for k, n in collections.Counter(e["key"] for e in car).items() if n > 1]
        json.dump(car, open(a.carried, "w"), indent=1)
        print(f"carried {len(car)} items -> {a.carried}; duplicate new keys {len(dup)}")


if __name__ == "__main__":
    main()
