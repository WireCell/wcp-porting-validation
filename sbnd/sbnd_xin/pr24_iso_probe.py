#!/usr/bin/env python3
"""doc pr/24 -- isochronous-shower / main-vertex-host probe.

Reads a PR arm's per-event tracking-pr.root (T_rec_charge / T_tagger / T_kine /
T_proj_data) and reports, per event:

  * which real_cluster_id hosts the reconstructed neutrino vertex, its total
    fitted track length, and the longest cluster in the event -- the pair that
    exposes a "vertex escaped onto a stub" main-cluster swap (doc pr/24 sec 2);
  * how isochronous the host and the longest cluster are: drift-x extent of the
    fit points vs their y-z diagonal (the same spirit as iso_band_like(),
    clus/src/clustering_neutrino.cxx:78, minus its 80 cm length floor -- the
    271851 shower is 70.8 cm and would fail that floor);
  * per-plane charge coverage sum(q_pred)/sum(q) of the host cluster.

With --detail EVT it additionally prints the host cluster's segment table
(length, chord/path straightness, tick span, dQ/dx) and the pairwise 2-D
projection-overlap matrix between segments -- the ghost-vs-real-branch test:
two segments that share most of their (tick, wire) pixels in >= 2 planes are
tomographic images of one another, not two tracks.

Usage:
  ./pr24_iso_probe.py <arm_dir> [--out out.tsv] [--jobs 8] [--events 271851 ...]
  ./pr24_iso_probe.py <arm_dir> --detail 271851
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import uproot

# A segment's 3-D points are grouped by sub_cluster_id = real_cluster_id*1000 + seg_index.
SID_DIV = 1000


def _seg_len(x, y, z):
    if len(x) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.stack([x, y, z], 1), axis=0), axis=1).sum())


def _load(path):
    f = uproot.open(path)
    t = f["T_rec_charge"].arrays(library="np")
    tg = f["T_tagger"].arrays(library="np")
    try:
        kin = f["T_kine"].arrays(library="np")
    except Exception:
        kin = None
    return f, t, tg, kin


def _cluster_table(t):
    """cluster id -> dict(len, npts, xext, yzdiag).

    T_rec_charge's real_cluster_id is written per SEGMENT (it equals
    sub_cluster_id), so the cluster is sub_cluster_id // 1000.  Points with
    sub_cluster_id < 0 are the un-segmented remainder and are ignored.
    """
    out = {}
    sids = t["sub_cluster_id"]
    cids = np.where(sids >= 0, sids // SID_DIV, -1)
    for cid in sorted(set(cids.tolist())):
        if cid < 0:
            continue
        m = cids == cid
        x, y, z = t["x"][m], t["y"][m], t["z"][m]
        tot = 0.0
        for sid in sorted(set(sids[m].tolist())):
            ms = sids == sid
            tot += _seg_len(t["x"][ms], t["y"][ms], t["z"][ms])
        out[int(cid)] = dict(
            len=tot,
            npts=int(m.sum()),
            xext=float(x.max() - x.min()),
            yzdiag=float(np.hypot(y.max() - y.min(), z.max() - z.min())),
        )
    return out


def _host_of(t, vtx):
    """cluster of the fit point closest to the neutrino vertex (and that distance)."""
    sids = t["sub_cluster_id"]
    m = sids >= 0
    if not m.any():
        return -1, 1e9
    d = np.sqrt((t["x"][m] - vtx[0]) ** 2 + (t["y"][m] - vtx[1]) ** 2 + (t["z"][m] - vtx[2]) ** 2)
    i = int(np.argmin(d))
    return int(sids[m][i] // SID_DIV), float(d[i])


def _coverage(f, rcid):
    """sum(q_pred)/sum(q) per plane for one cluster, from T_proj_data."""
    try:
        d = f["T_proj_data"].arrays(library="np")
        cids = list(np.array(d["cluster_id"][0]))
        if rcid not in cids:
            return {}
        i = cids.index(rcid)
        ch = np.array(d["channel"][0][i], dtype=float)
        q = np.array(d["charge"][0][i], dtype=float)
        qp = np.array(d["charge_pred"][0][i], dtype=float)
    except Exception:
        return {}
    res = {}
    # SBND channel layout: U < 2000 <= V < 6000 <= W (per-APA offsets keep the
    # ordering; only used to bin, never to index geometry).
    for name, lo, hi in (("u", 0, 2000), ("v", 2000, 6000), ("w", 6000, 1 << 30)):
        m = (ch >= lo) & (ch < hi)
        res[name] = float(qp[m].sum() / q[m].sum()) if m.sum() and q[m].sum() > 0 else float("nan")
    res["all"] = float(qp.sum() / q.sum()) if q.sum() > 0 else float("nan")
    return res


def _selected_main(arm, evt):
    """The cluster TaggerCheckNeutrino picked as the main one, from the job log.

    A later main-cluster swap (determine_overall_main_vertex_DL ->
    swap_main_cluster) leaves this line untouched, so comparing it with the
    cluster that actually hosts the final vertex detects the swap exactly --
    no length heuristic needed.
    """
    for name in ("stdout.log", "wct_pr_evt%s.log" % evt):
        p = os.path.join(arm, "pr_evt%s" % evt, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", errors="replace") as fh:
                for line in fh:
                    i = line.find("selected main cluster ")
                    if i >= 0:
                        return int(line[i + 22:].split()[0])
        except Exception:
            pass
    return -1


def probe_event(args):
    arm, evt = args
    path = os.path.join(arm, "pr_evt%s" % evt, "tracking-pr.root")
    try:
        f, t, tg, kin = _load(path)
    except Exception as e:
        return dict(event=evt, error=str(e)[:60])
    vtx = (float(tg["nu_x"][0]), float(tg["nu_y"][0]), float(tg["nu_z"][0]))
    tab = _cluster_table(t)
    host, vdis = _host_of(t, vtx)
    if not tab:
        return dict(event=evt, error="no clusters")
    longest = max(tab, key=lambda k: tab[k]["len"])
    h = tab.get(host, dict(len=0.0, npts=0, xext=0.0, yzdiag=0.0))
    L = tab[longest]
    cov = _coverage(f, longest)
    sel = _selected_main(arm, evt)
    sel_len = tab.get(sel, dict(len=0.0))["len"]
    return dict(
        event=evt,
        sel_main=sel, sel_len=sel_len,
        # swap = the final vertex host is NOT the cluster the tagger selected
        swap=int(sel >= 0 and host >= 0 and host != sel),
        cosmic=int(tg["cosmic_flag"][0]),
        enu=float(kin["kine_reco_Enu"][0]) if kin is not None else float("nan"),
        vx=vtx[0], vy=vtx[1], vz=vtx[2],
        host=host, host_len=h["len"], host_npts=h["npts"],
        host_xext=h["xext"], host_yz=h["yzdiag"],
        long_id=longest, long_len=L["len"], long_xext=L["xext"], long_yz=L["yzdiag"],
        vtx_dis=vdis,
        # escape = the vertex sits on a cluster far shorter than the event's longest
        escape=int(L["len"] > 20.0 and h["len"] < 0.25 * L["len"]),
        # iso = flat drift slab: x extent small vs the y-z footprint AND vs length
        iso_long=int(L["xext"] < max(25.0, 0.18 * L["len"]) and L["yzdiag"] > 30.0),
        cov_u=cov.get("u", float("nan")), cov_v=cov.get("v", float("nan")),
        cov_w=cov.get("w", float("nan")), cov_all=cov.get("all", float("nan")),
    )


def pixels(t, sid, plane, tw=4.0, ww=1.0):
    m = t["sub_cluster_id"] == sid
    return set(zip(np.round(t["pt"][m] / tw).astype(int),
                   np.round(t[plane][m] / ww).astype(int)))


def detail(arm, evt):
    path = os.path.join(arm, "pr_evt%s" % evt, "tracking-pr.root")
    f, t, tg, kin = _load(path)
    vtx = (float(tg["nu_x"][0]), float(tg["nu_y"][0]), float(tg["nu_z"][0]))
    tab = _cluster_table(t)
    host, vdis = _host_of(t, vtx)
    longest = max(tab, key=lambda k: tab[k]["len"])
    print("arm=%s evt=%s" % (arm, evt))
    print("  nu vertex (%.1f, %.1f, %.1f) cm, %.2f cm from the nearest fit point (cluster %d)"
          % (vtx + (vdis, host)))
    for rc in sorted(tab, key=lambda k: -tab[k]["len"])[:6]:
        c = tab[rc]
        tagl = []
        if rc == host:
            tagl.append("HOST")
        if rc == longest:
            tagl.append("LONGEST")
        print("  cluster %6d  L=%7.1f cm  n=%5d  x-ext=%6.1f  yz-diag=%6.1f  iso-ratio=%.3f %s"
              % (rc, c["len"], c["npts"], c["xext"], c["yzdiag"],
                 c["xext"] / max(c["len"], 1e-9), " ".join(tagl)))
    cov = _coverage(f, longest)
    if cov:
        print("  charge coverage of cluster %d (sum q_pred / sum q): u=%.3f v=%.3f w=%.3f all=%.3f"
              % (longest, cov["u"], cov["v"], cov["w"], cov["all"]))

    sids = [s for s in sorted(set(t["sub_cluster_id"].tolist()))
            if s >= 0 and s // SID_DIV == longest]
    keep = []
    print("\n  segments of cluster %d:" % longest)
    for sid in sids:
        m = t["sub_cluster_id"] == sid
        x, y, z = t["x"][m], t["y"][m], t["z"][m]
        ln = _seg_len(x, y, z)
        if ln < 5.0:
            continue
        keep.append(sid)
        chord = float(np.linalg.norm([x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]]))
        step = np.r_[0, np.linalg.norm(np.diff(np.stack([x, y, z], 1), axis=0), axis=1)]
        dqdx = np.where(step > 0, t["q"][m] / np.maximum(step, 1e-9), np.nan)
        print("    seg %6d n=%4d L=%6.1f chord/path=%.2f ticks=%5.1f shower=%d "
              "median dQ/dx=%8.0f  (%.1f,%.1f,%.1f)->(%.1f,%.1f,%.1f)"
              % (sid, int(m.sum()), ln, chord / max(ln, 1e-9),
                 float(t["pt"][m].max() - t["pt"][m].min()), int(t["flag_shower"][m][0]),
                 float(np.nanmedian(dqdx)), x[0], y[0], z[0], x[-1], y[-1], z[-1]))

    print("\n  pairwise 2-D projection overlap (fraction of the smaller pixel set):")
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            s1, s2 = keep[a], keep[b]
            fr = []
            for pl in ("pu", "pv", "pw"):
                A, B = pixels(t, s1, pl), pixels(t, s2, pl)
                fr.append(len(A & B) / max(1, min(len(A), len(B))))
            flag = "  <== shares >=2 planes" if sum(f > 0.25 for f in fr) >= 2 else ""
            print("    %6d vs %6d   u=%.2f v=%.2f w=%.2f%s" % (s1, s2, fr[0], fr[1], fr[2], flag))


def _seg_dir_at(x, y, z, at_end, reach=10.0):
    """Unit vector pointing from one end of a segment INTO the segment.

    `reach` cm of arc length is used rather than the raw first/last step, so a
    single jittery fit point cannot set the direction.
    """
    pts = np.stack([x, y, z], 1)
    if at_end:
        pts = pts[::-1]
    step = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    j = int(np.searchsorted(step, reach))
    j = min(max(j, 1), len(pts) - 1)
    v = pts[j] - pts[0]
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def junctions(arm, evt, min_len=5.0, join_tol=2.0, reach=10.0):
    """Segment junctions of the vertex-host cluster and their turn angles.

    A junction is a pair of >= min_len cm segments whose endpoints meet within
    join_tol.  `turn` is the deviation from straight-through: 0 deg means the
    two segments continue each other exactly, i.e. a vertex sitting inside a
    straight track -- the pathology doc pr/24 round 3 fixes (evt 284794's
    260.1 + 15.7 cm pair met at 0.9 deg).
    """
    path = os.path.join(arm, "pr_evt%s" % evt, "tracking-pr.root")
    f, t, tg, kin = _load(path)
    vtx = (float(tg["nu_x"][0]), float(tg["nu_y"][0]), float(tg["nu_z"][0]))
    host, _ = _host_of(t, vtx)
    sids = [s for s in sorted(set(t["sub_cluster_id"].tolist()))
            if s >= 0 and s // SID_DIV == host]
    segs = {}
    for sid in sids:
        m = t["sub_cluster_id"] == sid
        x, y, z = t["x"][m], t["y"][m], t["z"][m]
        if len(x) < 2 or _seg_len(x, y, z) < min_len:
            continue
        segs[sid] = (x, y, z, _seg_len(x, y, z))
    out = []
    ks = sorted(segs)
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            xa, ya, za, la = segs[ks[a]]
            xb, yb, zb, lb = segs[ks[b]]
            best = None
            for ea in (0, 1):
                for eb in (0, 1):
                    pa = np.array([xa[-1 if ea else 0], ya[-1 if ea else 0], za[-1 if ea else 0]])
                    pb = np.array([xb[-1 if eb else 0], yb[-1 if eb else 0], zb[-1 if eb else 0]])
                    d = float(np.linalg.norm(pa - pb))
                    if best is None or d < best[0]:
                        best = (d, ea, eb, pa)
            d, ea, eb, pj = best
            if d > join_tol:
                continue
            da = _seg_dir_at(xa, ya, za, ea, reach)
            db = _seg_dir_at(xb, yb, zb, eb, reach)
            if da is None or db is None:
                continue
            ang = float(np.degrees(np.arccos(np.clip(float(np.dot(da, db)), -1.0, 1.0))))
            out.append(dict(evt=evt, host=host, sa=ks[a], sb=ks[b], la=la, lb=lb,
                            gap=d, turn=180.0 - ang,
                            jx=float(pj[0]), jy=float(pj[1]), jz=float(pj[2])))
    return out


def junction_scan(arm, ref, evts, straight_deg=15.0, jobs=8):
    """Report straight-through junctions in `arm`, and what `ref` had for the
    same event -- the regression detector round 2 lacked (0 label / 0
    nu_evaluated moves said nothing about a vertex planted in a straight track).
    """
    print("evt\tarm_njunc\tarm_straight\tref_straight\tworst_turn\tworst_pair\tverdict")
    nbad = 0
    for e in evts:
        try:
            ja = junctions(arm, e)
        except Exception as ex:
            print("%s\tERR\t%s" % (e, ex))
            continue
        try:
            jr = junctions(ref, e) if ref else []
        except Exception:
            jr = []
        sa = [j for j in ja if j["turn"] < straight_deg]
        sr = [j for j in jr if j["turn"] < straight_deg]
        worst = min(ja, key=lambda j: j["turn"]) if ja else None
        verdict = "NEW-STRAIGHT-JUNCTION" if len(sa) > len(sr) else "ok"
        if verdict != "ok":
            nbad += 1
        print("%s\t%d\t%d\t%d\t%s\t%s\t%s"
              % (e, len(ja), len(sa), len(sr),
                 ("%.1f" % worst["turn"]) if worst else "-",
                 ("%d+%d L=%.1f/%.1f" % (worst["sa"], worst["sb"], worst["la"], worst["lb"]))
                 if worst else "-", verdict))
    print("# %d/%d events gained a straight-through junction (turn < %.0f deg) vs %s"
          % (nbad, len(evts), straight_deg, ref or "(no ref)"), file=sys.stderr)


COLS = ["event", "sel_main", "sel_len", "swap", "cosmic", "enu", "vx", "vy", "vz",
        "host", "host_len", "host_npts", "host_xext", "host_yz",
        "long_id", "long_len", "long_xext", "long_yz",
        "vtx_dis", "escape", "iso_long", "cov_u", "cov_v", "cov_w", "cov_all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--events", nargs="*")
    ap.add_argument("--detail")
    ap.add_argument("--junctions", action="store_true",
                    help="report straight-through segment junctions (doc pr/24 round 3)")
    ap.add_argument("--vs", help="reference arm to compare junction counts against")
    ap.add_argument("--straight-deg", type=float, default=15.0)
    a = ap.parse_args()

    if a.detail:
        detail(a.arm, a.detail)
        return

    if a.junctions:
        evts = a.events or sorted(os.path.basename(p)[6:]
                                  for p in glob.glob(os.path.join(a.arm, "pr_evt*"))
                                  if os.path.isdir(p))
        junction_scan(a.arm, a.vs, evts, a.straight_deg, a.jobs)
        return

    evts = a.events or sorted(os.path.basename(p)[6:]
                              for p in glob.glob(os.path.join(a.arm, "pr_evt*"))
                              if os.path.isdir(p))
    with Pool(a.jobs) as p:
        rows = p.map(probe_event, [(a.arm, e) for e in evts])

    good = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    fh = open(a.out, "w") if a.out else sys.stdout
    print("\t".join(COLS), file=fh)
    for r in good:
        print("\t".join(("%.3f" % r[c]) if isinstance(r[c], float) else str(r[c])
                        for c in COLS), file=fh)
    if a.out:
        fh.close()
    esc = [r for r in good if r["escape"]]
    iso = [r for r in good if r["iso_long"]]
    swp = [r for r in good if r["swap"]]
    # the pathology of doc pr/24: the vertex left the selected main cluster for
    # a much shorter host
    bad_swap = [r for r in swp if r["sel_len"] > 20.0 and r["host_len"] < 0.25 * r["sel_len"]]
    print("# %d events (%d unreadable); main-cluster swap %d (onto a <25%% host: %d); "
          "vertex-escape %d; isochronous-longest %d; swap+iso %d"
          % (len(good), len(bad), len(swp), len(bad_swap), len(esc), len(iso),
             len([r for r in bad_swap if r["iso_long"]])), file=sys.stderr)
    for r in bad_swap:
        print("#   evt %s: selected main %d (L=%.1f cm) -> vertex host %d (L=%.1f cm) iso=%d"
              % (r["event"], r["sel_main"], r["sel_len"], r["host"], r["host_len"],
                 r["iso_long"]), file=sys.stderr)


if __name__ == "__main__":
    main()
