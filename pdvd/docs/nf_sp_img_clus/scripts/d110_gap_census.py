#!/usr/bin/env python3
"""doc pdvd/110 -- whole-arm census behind the owner's stm_fit spots: holes Bee leaves in the layer, where
they come from, and which way each trajectory flip moved them.  Read-only on existing arms.

    python3 d110_gap_census.py > figs/110_gap_census.txt

sec 0  the owner's clicks: which layers of the arm the owner viewed hold a point within 3 cm, and the
       tagger verdict of the clicked cluster (includes the withdrawn cl115 click).
sec 1  Bee level, TAGGED clusters only (what the round-3 stm_fit layer draws; verdict from the job log):
       rows with q < 0 (Bee sst.js skips them), holes = runs of >= 3 consecutive q<0 rows and their arc
       length, off-charge rows, rows with wiggle > 1 cm; and the track_fit layer's clamped rows (q == 0,
       drawn) for the same clusters.  Each pre-flip / production pair is also compared on the clusters
       tagged in BOTH arms of the pair (the pairs share pctrees, so cluster ids agree).
sec 2  ROOT level, every fitted row of every fitted cluster and pass (T_rec_charge in tracking-stm.root):
       q<0 fraction by the number of planes whose own cell (rounded wire, rounded slice) holds no measured
       charge of the cluster (T_proj_data), by reg_flag x degenerate planes, and the wiggle p90 per stratum.
sec 3  continuity: steps between consecutive rows of one (cluster, pass).

Definitions (shared with d110_spot_figs.py):
  q            = dQ * 0.1 - 1000 (Trun); q < 0  <=>  dQ < 10 ke; Bee never loads such a row.
  off-charge   = no q > 0 cell of this cluster in T_proj_data on the row's own slice within +-1 wire of the
                 row's rounded wire, on that plane.  The exact own-cell test (d102's on_charge) is NOT used for
                 the strata: on PDVD it fails 20 % / 24 % of U / V rows against 6 % of W rows, and drops to
                 6-7 % with +-1 wire, i.e. a rounding offset on the PDVD induction planes; sec 2c prints both.
  degenerate   = central-difference pixel separation hypot(d wire, d slice) < 0.5 on that plane.
  wiggle       = distance of a row from the chord of the rows 3 before and 3 after (cm).
"""
import glob, json, os, re, sys, zipfile
from collections import Counter, defaultdict
import numpy as np
import uproot
from scipy.spatial import cKDTree

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
BASE = {"pdhd": (0, 3200, 6400), "pdvd": (0, 3808, 7616)}
ARMS = {"pdhd": [("d101hnew", "pre-flip"), ("d108hflip", "production")],
        "pdvd": [("q29flip", "shown in Bee set 51c1e410 (pre-flip)"), ("d103v0", "pre-flip"),
                 ("d103vflip", "production")]}
PAIRS = {"pdhd": ("d101hnew", "d108hflip"), "pdvd": ("d103v0", "d103vflip")}
CLICKS = [("pdvd", "039349_20", "q29flip", "v1", 80, (313.7, 95.9, 124.9)),
          ("pdvd", "039349_20", "q29flip", "v2", 80, (94.5, 30.7, 165.0)),
          ("pdvd", "039349_20", "q29flip", "v3", 25, (-270.2, -243.7, 36.0)),
          ("pdhd", "029107_16", "d109hstm", "h1", 108, (75.3, 438.3, 450.1)),
          ("pdhd", "029107_16", "d109hstm", "h2", 106, (182.0, 537.4, 401.8)),
          ("pdhd", "029107_16", "d109hstm", "h2-withdrawn", 115, (278.5, 474.6, 299.1))]
LAYERS = ("stm_fit", "track_fit", "stm", "steiner_graph", "clustering")
R_VERDICT = re.compile(r"visit: TaggerCheckSTM: cluster (\d+) (?:→|->) STM=(\d) TGM=(\d)")
R_SKIP = re.compile(r"visit: TaggerCheckSTM: cluster (\d+) already TGM; skipping")


def log_text(d):
    logs = sorted(glob.glob(f"{d}/wct_pr_*.log"))
    if not logs:
        raise SystemExit(f"no PR log in {d}")
    return open(logs[0], errors="replace").read()


def tagged_set(txt):
    return {int(c) for c, s, _t in R_VERDICT.findall(txt) if s == "1"}


def cells_by_cluster(f, det):
    """cluster -> plane -> set((channel, slice)) with measured charge > 0."""
    pd = f["T_proj_data"].arrays(library="np")
    out = defaultdict(lambda: [set(), set(), set()])
    base = BASE[det]
    for cid, ch, ts, q in zip(pd["cluster_id"][0], pd["channel"][0], pd["time_slice"][0], pd["charge"][0]):
        ch = np.asarray(ch).astype(int); ts = np.asarray(ts).astype(int); q = np.asarray(q)
        m = q > 0
        pl = np.searchsorted(np.array(base[1:]), ch, side="right")
        for p in range(3):
            mm = m & (pl == p)
            out[int(cid) // 10][p].update(zip(ch[mm].tolist(), ts[mm].tolist()))
    return out


def segments(t):
    key = t["cluster_id"].astype(np.int64) * 100 + t["pass"].astype(np.int64)
    idx = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    return [(int(t["cluster_id"][a]), a, b) for a, b in zip(idx[:-1], idx[1:])]


def wiggle(P, k=3):
    out = np.full(len(P), np.nan)
    if len(P) <= 2 * k:
        return out
    a, b = P[:-2 * k], P[2 * k:]
    u = b - a; n = np.linalg.norm(u, axis=1); ok = n > 0
    u[ok] /= n[ok, None]
    v = P[k:-k] - a
    w = np.linalg.norm(v - np.sum(v * u, axis=1)[:, None] * u, axis=1)
    w[~ok] = np.nan
    out[k:-k] = w
    return out


def neg_holes(q, P, minrows=3):
    """list of (rows, arc cm) for runs of >= minrows consecutive q < 0 rows."""
    out, i = [], 0
    while i < len(q):
        if q[i] < 0:
            j = i
            while j < len(q) and q[j] < 0:
                j += 1
            if j - i >= minrows:
                out.append((j - i, float(np.sum(np.linalg.norm(np.diff(P[i:j], axis=0), axis=1)))))
            i = j
        else:
            i += 1
    return out


def event_rows(det, d):
    """per fitted row: cluster, q, off-charge planes, reg any, degenerate planes, wiggle; plus per-segment
    steps and holes."""
    f = uproot.open(f"{d}/tracking-stm.root")
    t = f["T_rec_charge"].arrays(["x", "y", "z", "q", "pu", "pv", "pw", "pt", "cluster_id", "pass",
                                  "reg_flag_u", "reg_flag_v", "reg_flag_w"], library="np")
    cells = cells_by_cluster(f, det)
    n = len(t["q"])
    off = np.zeros(n, int); deg = np.zeros(n, int); wig = np.full(n, np.nan)
    own = np.zeros((n, 3), bool); tol = np.zeros((n, 3), bool)
    steps, holes = [], []
    P = np.c_[t["x"], t["y"], t["z"]]
    for cl, a, b in segments(t):
        sl = slice(a, b)
        c = cells.get(cl, [set(), set(), set()])
        for p, key in enumerate(("pu", "pv", "pw")):
            w = np.round(t[key][sl]).astype(int); s = np.round(t["pt"][sl]).astype(int)
            own[sl, p] = [(wi, si) not in c[p] for wi, si in zip(w.tolist(), s.tolist())]
            tol[sl, p] = [all((wi + dw, si) not in c[p] for dw in (-1, 0, 1)) for wi, si in zip(w.tolist(), s.tolist())]
            if b - a >= 3:
                sep = np.hypot(np.gradient(t[key][sl]), np.gradient(t["pt"][sl]))
                deg[sl] += (sep < 0.5).astype(int)
        wig[sl] = wiggle(P[sl])
        if b - a >= 2:
            steps.append((cl, np.linalg.norm(np.diff(P[sl], axis=0), axis=1)))
        holes.append((cl, neg_holes(t["q"][sl], P[sl])))
    off = tol.sum(1)
    reg = (t["reg_flag_u"] + t["reg_flag_v"] + t["reg_flag_w"]) > 0
    arc = {}
    for cl, st in steps:
        arc[cl] = arc.get(cl, 0.0) + float(st.sum())
    return dict(cl=t["cluster_id"].astype(int), q=t["q"], off=off, own=own, tol=tol, deg=deg, reg=reg, wig=wig,
                steps=steps, holes=holes, arc=arc)


def zip_trackfit_zero(d, clusters):
    with zipfile.ZipFile(f"{d}/mabc-pr.zip") as z:
        if "data/0/0-track_fit-global.json" not in z.namelist():
            return 0, 0, 1          # no PR candidate in this event: the layer is not written
        j = json.loads(z.read("data/0/0-track_fit-global.json"))
    cid = np.array(j["cluster_id"]); q = np.array(j["q"], float)
    m = np.isin(cid, list(clusters))
    return int(m.sum()), int((q[m] == 0).sum()), 0


def pct(a, b):
    return f"{100.0 * a / b:5.1f} %" if b else "   -  "


def sec0():
    print("## sec 0 -- the owner's clicks: layers of the arm viewed with a point within 3 cm\n")
    for det, ev, arm, name, cl, xyz in CLICKS:
        d = f"{IMG}/{det}/work/{ev}_{arm}"
        p = np.array(xyz)
        with zipfile.ZipFile(f"{d}/mabc-pr.zip") as z:
            parts = []
            for lay in LAYERS:
                j = json.loads(z.read(f"data/0/0-{lay}-global.json"))
                X = np.c_[j["x"], j["y"], j["z"]]; cid = np.array(j["cluster_id"])
                dd = np.linalg.norm(X - p, axis=1)
                k = int(np.argmin(dd))
                own = cid == cl
                down = float(dd[own].min()) if own.any() else float("nan")
                parts.append(f"{lay}: nearest {dd[k]:.2f} cm (cl{cid[k]}), cl{cl} {down:.2f} cm")
        txt = log_text(d)
        v = [f"STM={s} TGM={t_}" for c, s, t_ in R_VERDICT.findall(txt) if int(c) == cl]
        if not v and any(int(c) == cl for c in R_SKIP.findall(txt)):
            v = ["already TGM; skipped by TaggerCheckSTM"]
        print(f"* {name} {det} {ev} [{arm}] cl{cl} {xyz}: verdict {v[0] if v else 'not evaluated'}")
        for s in parts:
            print(f"    - {s}")
    print()


def main():
    print("# doc pdvd/110 gap census (d110_gap_census.py)\n")
    sec0()
    S1 = {}; S2 = {}; S3 = {}; TAG = {}
    for det in ("pdhd", "pdvd"):
        for arm, label in ARMS[det]:
            dirs = sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-stm.root"))
            acc = defaultdict(float); strat = defaultdict(lambda: [0, 0]); regdeg = defaultdict(lambda: [0, 0])
            wig_by_off = defaultdict(list); stepall = []; tags = {}
            planes = np.zeros((4, 3)); nrow = [0, 0]
            for f in dirs:
                d = os.path.dirname(f); ev = os.path.basename(d)[:-len(arm) - 1]
                tg = tagged_set(log_text(d)); tags[ev] = tg
                r = event_rows(det, d)
                # sec 2: every fitted row
                for k in range(4):
                    m = r["off"] == min(k, 3) if k < 3 else r["off"] >= 3
                    strat[k][0] += int(m.sum()); strat[k][1] += int((r["q"][m] < 0).sum())
                    wig_by_off[k].extend(r["wig"][m][np.isfinite(r["wig"][m])].tolist())
                for rg in (0, 1):
                    for dg in range(4):
                        m = (r["reg"] == bool(rg)) & (np.minimum(r["deg"], 3) == dg)
                        regdeg[(rg, dg)][0] += int(m.sum()); regdeg[(rg, dg)][1] += int((r["q"][m] < 0).sum())
                neg = r["q"] < 0
                planes[0] += r["own"].sum(0); planes[1] += r["tol"].sum(0)
                planes[2] += r["own"][neg].sum(0); planes[3] += r["tol"][neg].sum(0)
                nrow[0] += len(neg); nrow[1] += int(neg.sum())
                stepall.extend(np.concatenate([s for _c, s in r["steps"]]).tolist() if r["steps"] else [])
                # sec 1: tagged clusters only
                m = np.isin(r["cl"], list(tg))
                acc["events"] += 1; acc["tagged"] += len(tg)
                acc["rows"] += int(m.sum()); acc["neg"] += int((r["q"][m] < 0).sum())
                acc["dq_le0"] += int((r["q"][m] <= -1000).sum())      # dQ <= 0 itself
                acc["off1"] += int((r["off"][m] >= 1).sum())
                acc["wig1"] += int((r["wig"][m] > 1.0).sum())
                acc["arc"] += sum(v for c, v in r["arc"].items() if c in tg)
                for c, hs in r["holes"]:
                    if c in tg:
                        acc["holes"] += len(hs); acc["hole_cm"] += sum(h[1] for h in hs)
                        acc["hole_max"] = max(acc["hole_max"], max([h[1] for h in hs], default=0.0))
                ntf, nz, nolayer = zip_trackfit_zero(d, tg)
                acc["tf_rows"] += ntf; acc["tf_zero"] += nz; acc["tf_nolayer"] += nolayer
                acc.setdefault("per_event", {})[ev] = r
            S1[(det, arm)] = acc; S2[(det, arm)] = (strat, regdeg, wig_by_off, planes, nrow); S3[(det, arm)] = np.array(stepall)
            TAG[(det, arm)] = tags

    print("## sec 1 -- Bee level, tagged clusters (what the stm_fit layer draws)\n")
    print("| det | arm | role | events | tagged clusters | fit rows | fit length (m) | q<0 rows (not drawn) "
          "| of which dQ <= 0 | holes >= 3 rows | holes per 10 m | hole length total / max (cm) | off-charge rows (>=1 plane) "
          "| rows wiggle > 1 cm | track_fit rows / clamped to 0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for det in ("pdhd", "pdvd"):
        for arm, label in ARMS[det]:
            a = S1[(det, arm)]
            print(f"| {det} | `{arm}` | {label} | {int(a['events'])} | {int(a['tagged'])} | {int(a['rows'])} "
                  f"| {a['arc'] / 100:.1f} | {int(a['neg'])} ({pct(a['neg'], a['rows']).strip()}) | {int(a['dq_le0'])} ({pct(a['dq_le0'], a['neg']).strip()}) | {int(a['holes'])} "
                  f"| {10 * a['holes'] / (a['arc'] / 100):.2f} | {a['hole_cm']:.0f} / {a['hole_max']:.1f} "
                  f"| {pct(a['off1'], a['rows']).strip()} | {pct(a['wig1'], a['rows']).strip()} "
                  f"| {int(a['tf_rows'])} / {int(a['tf_zero'])} ({int(a['tf_nolayer'])} evt without the layer) |")
    print()
    SPLIT = {}
    print("### sec 1b -- pre-flip vs production on the clusters tagged in BOTH arms (same pctrees)\n")
    print("| det | pair | events | common tagged clusters | fit length pre / prod (m) | q<0 rows pre / prod "
          "| holes pre / prod | hole length pre / prod (cm) | clusters with a hole pre / prod | wiggle>1 cm rows pre / prod |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for det in ("pdhd", "pdvd"):
        A, B = PAIRS[det]
        pa, pb = S1[(det, A)]["per_event"], S1[(det, B)]["per_event"]
        evs = sorted(set(pa) & set(pb))
        tot = defaultdict(float); split = Counter()
        for ev in evs:
            common = TAG[(det, A)][ev] & TAG[(det, B)][ev]
            tot["cl"] += len(common)
            for side, r in (("a", pa[ev]), ("b", pb[ev])):
                m = np.isin(r["cl"], list(common))
                tot["neg" + side] += int((r["q"][m] < 0).sum())
                tot["wig" + side] += int((r["wig"][m] > 1.0).sum())
                tot["arc" + side] += sum(v for c, v in r["arc"].items() if c in common)
                withhole = set(); tot.setdefault("sets", {})
                for c, hs in r["holes"]:
                    if c in common and hs:
                        tot["h" + side] += len(hs); tot["hcm" + side] += sum(h[1] for h in hs); withhole.add(c)
                tot["hc" + side] += len(withhole)
                tot["sets"][side] = withhole
            ha, hb = tot["sets"]["a"], tot["sets"]["b"]
            split["gained"] += len(hb - ha); split["lost"] += len(ha - hb)
            split["both"] += len(ha & hb); split["neither"] += len(common - ha - hb)
        print(f"| {det} | `{A}` -> `{B}` | {len(evs)} | {int(tot['cl'])} | {tot['arca'] / 100:.1f} / {tot['arcb'] / 100:.1f} "
              f"| {int(tot['nega'])} / {int(tot['negb'])} | {int(tot['ha'])} / {int(tot['hb'])} "
              f"| {tot['hcma']:.0f} / {tot['hcmb']:.0f} | {int(tot['hca'])} / {int(tot['hcb'])} "
              f"| {int(tot['wiga'])} / {int(tot['wigb'])} |")
        SPLIT[det] = split
    print()
    print("### sec 1c -- per cluster (common tagged set): hole >= 3 rows only in production (gained), only pre-flip "
          "(lost), in both, in neither.  Cluster level, not location level.\n")
    print("| det | gained | lost | both | neither |")
    print("|---|---|---|---|---|")
    for det in ("pdhd", "pdvd"):
        sp = SPLIT[det]
        print(f"| {det} | {sp['gained']} | {sp['lost']} | {sp['both']} | {sp['neither']} |")
    print()

    print("## sec 2 -- ROOT level, every fitted row (all fitted clusters, all passes)\n")
    print("### 2a -- q<0 fraction by the number of planes with no charge of the cluster within +-1 wire on the row's slice\n")
    print("| det | arm | 0 planes off | 1 | 2 | 3 | share of all q<0 rows at 0 / 1 / 2 / 3 | wiggle p90 (cm) at 0 / 1 / 2 / 3 |")
    print("|---|---|---|---|---|---|---|---|")
    for det in ("pdhd", "pdvd"):
        for arm, label in ARMS[det]:
            strat, _rd, wb, _pl, _nr = S2[(det, arm)]
            cells = [f"{pct(strat[k][1], strat[k][0]).strip()} of {strat[k][0]}" for k in range(4)]
            wp = " / ".join(f"{np.percentile(wb[k], 90):.2f}" if wb[k] else "-" for k in range(4))
            nn = sum(strat[k][1] for k in range(4))
            sh = " / ".join(f"{100 * strat[k][1] / nn:.0f} %" for k in range(4))
            print(f"| {det} | `{arm}` | " + " | ".join(cells) + f" | {sh} | {wp} |")
    print()
    print("### 2b -- q<0 fraction by reg_flag (any plane) x degenerate planes\n")
    print("| det | arm | reg | 0 degenerate | 1 | 2 | 3 | share of all q<0 rows at 0 / 1 / 2 / 3 |")
    print("|---|---|---|---|---|---|---|---|")
    for det in ("pdhd", "pdvd"):
        for arm, label in ARMS[det]:
            _s, rd, _w, _pl, _nr = S2[(det, arm)]
            for rg in (0, 1):
                cells = [f"{pct(rd[(rg, dg)][1], rd[(rg, dg)][0]).strip()} of {rd[(rg, dg)][0]}" for dg in range(4)]
                nn = sum(v[1] for v in rd.values())
                sh = " / ".join(f"{100 * rd[(rg, dg)][1] / nn:.0f} %" for dg in range(4))
                print(f"| {det} | `{arm}` | {rg} | " + " | ".join(cells) + f" | {sh} |")
    print()
    print("### 2c -- off-charge rate per plane: exact own cell vs +-1 wire, all rows and q<0 rows\n")
    print("| det | arm | own cell U / V / W | +-1 wire U / V / W | q<0 rows: own cell U / V / W | q<0 rows: +-1 wire U / V / W |")
    print("|---|---|---|---|---|---|")
    for det in ("pdhd", "pdvd"):
        for arm, label in ARMS[det]:
            _s, _rd, _w, pl, nr = S2[(det, arm)]
            f = lambda v, d: " / ".join(f"{100 * x / d:.1f}" for x in v)
            print(f"| {det} | `{arm}` | {f(pl[0], nr[0])} | {f(pl[1], nr[0])} | {f(pl[2], nr[1])} | {f(pl[3], nr[1])} |")
    print()
    print("## sec 3 -- continuity of the persisted fit (steps between consecutive rows of one cluster and pass)\n")
    print("| det | arm | steps | p50 (cm) | p99 (cm) | max (cm) | steps > 3 cm |")
    print("|---|---|---|---|---|---|---|")
    for det in ("pdhd", "pdvd"):
        for arm, label in ARMS[det]:
            s = S3[(det, arm)]
            print(f"| {det} | `{arm}` | {len(s)} | {np.median(s):.2f} | {np.percentile(s, 99):.2f} | {s.max():.2f} "
                  f"| {int((s > 3).sum())} |")
    print()


if __name__ == "__main__":
    main()
