#!/usr/bin/env python3
"""doc pdvd/97 round 2 -- pick the PDVD STM + Michel showcase events on the post-flip production arm.

    python3 d97r2_video_picks.py [--skip key,...] > ../../scan/d97r2/picks.txt

Read-only on production arm d103vflip (the flipped trajectory config, doc pdvd/103 sec 14) and the committed hand
records; writes pdvd/docs/scan/d97r2/picks.tsv, figs/97r2_picks_dqdx_rr.png and figs/97r2_picks_traj_{a,b}.png.
The round-1 script d97_video_picks.py and scan/d97/ are untouched; this script only imports its PF helpers.

OWNER CHOICES (2026-09-16): PDVD only; "good track trajectory"; every Michel situation offered (turn angle, detached,
Michel + gamma / pieces, energy extremes); one no-Michel contrast each (dots, bare).

TRUTH: d113_grade.truth("pdvd") -- own103v2 > own103v > p99rwon_carried_corrected > smx11, the folded precedence the
PDVD flip was graded on (doc pdvd/103 sec 13).  The record row that supplied each key (same precedence) gives the
confidence and the evidence text; its verdict is asserted equal to the truth row.

GATES, every pick (doc 97's Q rules restated for PDVD; counts printed per class):
  H    hand verdict as the class requires (FRAG_ stripped)
  Q4   the chain's Bragg-path accept: is_stm == 1 and topology_cleared_bits == 0
  Q5   michel_found == 1 for a Michel class, == 0 for a contrast class
  SIT  the class's situation, read from the CHAIN (T_stm_michel) -- the hand record gives only verdict + michel_kind
  Q6   the event's production mabc-pr.zip holds mc, track_fit, stm_fit and clustering
  Q7   the candidate's PF subtree holds only mu- / e- / gamma
  Q9   michel_ke_best <= 52.8 MeV and no e- / gamma node of the subtree reads above 52.8 MeV
  Q8   the PF shows the class at the stop (d97_video_picks.pf_count: "near" = node start/end within 50 cm of the stop,
       "Michel-near" within 15 cm; a gamma -> e- pseudo-carrier counts once), rule per class below
  TRAJ the owner's "good track trajectory", on the candidate cluster's rows of the production zip, file order split into
       runs at steps > 3 cm (never relaxed):
         (a) stm_fit: 0 Bee holes (runs of >= 3 consecutive q < 0 rows; Bee skips q < 0, doc pdvd/110)
         (b) stm_fit and track_fit: <= 5 % of rows with ridge offset > 1 cm (d111_stage_attrib.Ridge on the cluster's
             clustering-global points; doc pdvd/111 P1)
         (c) track_fit: <= 5 % of rows with chord wiggle > 1 cm (d111_eval.wiggle, k = 3; doc pdvd/110)
  Q2   owner source (owner_review row, or record confidence "owner") or confidence "high" -> tier 0; else tier 1

CLASSES, pre-registered fill order (rarest first); a key already used (pick or runner-up) by an earlier class is
excluded from the later ones:
  class          hand                      chain situation                         Q8
  detached       STM_MICHEL                michel_conn_type == 2 (bridged)          Michel-near >= 1
  backward       STM_MICHEL, attached      conn 1, michel_kink_deg >= 120           attached (1 EM near, Michel-near, no gamma)
  multicluster   STM_MICHEL                michel_n_clusters >= 2                   Michel-near >= 1
  energetic      STM_MICHEL                35 <= michel_ke_best <= 52.8             Michel-near >= 1
  soft           STM_MICHEL                michel_ke_best < 15                      Michel-near >= 1
  forward        STM_MICHEL, attached      conn 1, 0 <= michel_kink_deg < 60        attached
  gamma          STM_MICHEL, both          n_michel_gammas >= 1                     both (Michel-near and >= 2 EM near or >= 2 clusters)
  perpendicular  STM_MICHEL, attached      conn 1, 60 <= michel_kink_deg < 120      attached
  dots           STM_ONLY, detached dots   michel_found == 0                        >= 1 gamma near
  bare           STM_ONLY, none            michel_found == 0                        no EM near

RANK within a class: tier, Bragg ratio (contrast / contrast_expected) descending, owner source first, fewer mu- nodes,
key.  Rank 1 goes to Bee; rank 2 is the recorded runner-up.

FALLBACK (doc 97's, unchanged): tier 1 (any scanner source) is the only relaxation.  A class still short is reported
(rc 3), never filled by dropping TRAJ, SIT or Q8.

VISUAL CHECK: every pick and runner-up is drawn (dQ/dx vs residual range; trajectory over the charge).  --skip key drops
one whose Bragg rise is not clear or whose trajectory visibly leaves the charge; the skipped keys go to the tsv header.

AMENDMENT A1 (2026-09-16, after viewing round 1's panels, before round 2; round-1 outputs kept as
scan/d97r2/picks_round1.* and figs/97r2_picks_*_round1.png): the visual check also skips a pick whose trajectory panel
does not show its class's situation at the stop.  Trigger: detached #1 039349_23/56 (michel_conn_type 2) draws its Michel
as a near-collinear continuation inside the muon's own cluster with no visible gap (close-up checked), so a "detached
Michel" caption would not be visible on screen.  A deep dQ/dx dip on the approach to the stop counts as "Bragg rise not
clear", as in round 1 (doc 97 sec 2.1, 028084_17/97).  No gate, threshold, class or ranking changed.
"""
import argparse, collections, json, math, os, sys, zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
import d97_video_picks as P97                      # noqa: E402  PF helpers (round 1, untouched)
import d113_grade as G                             # noqa: E402  folded PDVD truth
from d111_stage_attrib import Ridge                # noqa: E402  ridge offset (doc 111)
from d111_eval import holes, wiggle                # noqa: E402  Bee holes / chord wiggle (docs 110-111)

rr26 = P97.rr26
DET, ARM = "pdvd", "d103vflip"
NPICK = 2
ENDPOINT_MEV = 52.8
TRAJ_DEV1_MAX = 0.05
TRAJ_WIG1_MAX = 0.05
RUN_BREAK_CM = 3.0
OUT_TSV = IMG + "/pdvd/docs/scan/d97r2/picks.tsv"
FIG = IMG + "/pdvd/docs/nf_sp_img_clus/figs/97r2_picks"

MICHEL = ("STM_MICHEL",)
CLASSES = [  # name, hand verdict, hand kinds (None = any), michel_found, situation, Q8 rule
    ("detached", "STM_MICHEL", None, 1, lambda d: int(d["michel_conn_type"]) == 2, "michel_near"),
    ("backward", "STM_MICHEL", ("attached",), 1,
     lambda d: int(d["michel_conn_type"]) == 1 and d["michel_kink_deg"] >= 120, "attached"),
    ("multicluster", "STM_MICHEL", None, 1, lambda d: int(d["michel_n_clusters"]) >= 2, "michel_near"),
    ("energetic", "STM_MICHEL", None, 1, lambda d: 35.0 <= d["michel_ke_best"] <= ENDPOINT_MEV, "michel_near"),
    ("soft", "STM_MICHEL", None, 1, lambda d: d["michel_ke_best"] < 15.0, "michel_near"),
    ("forward", "STM_MICHEL", ("attached",), 1,
     lambda d: int(d["michel_conn_type"]) == 1 and 0 <= d["michel_kink_deg"] < 60, "attached"),
    ("gamma", "STM_MICHEL", ("both",), 1, lambda d: int(d["n_michel_gammas"]) >= 1, "both"),
    ("perpendicular", "STM_MICHEL", ("attached",), 1,
     lambda d: int(d["michel_conn_type"]) == 1 and 60 <= d["michel_kink_deg"] < 120, "attached"),
    ("dots", "STM_ONLY", ("detached dots",), 0, lambda d: True, "dots"),
    ("bare", "STM_ONLY", ("none",), 0, lambda d: True, "bare"),
]
STAGES = ["H", "Q4", "Q5", "SIT", "Q6", "Q7", "Q9", "Q8", "TRAJ", "Q2"]
BR = ["cluster_id", "is_stm", "michel_found", "topology_cleared_bits", "contrast", "contrast_expected", "muon_len",
      "michel_ke_best", "michel_ke_q2d_region", "michel_kink_deg", "michel_len", "michel_conn_type", "michel_dis_cm",
      "michel_n_clusters", "michel_n_pieces", "n_michel_gammas", "n_dots", "stop_x", "stop_y", "stop_z",
      "michel_start_x", "michel_start_y", "michel_start_z"]


def records_by_precedence():
    """key -> (record row, precedence tag), first holder in d113_grade.truth's order"""
    t = G.TRUTH["pdvd"]
    out = {}
    for path, tag in ((t["fold"], t["fold_tag"]), (t["owner"], "own103h"), (t["record"], "record"),
                      (t["new"], "new_agent")):
        for r in json.load(open(path)):
            out.setdefault(r["key"], (r, tag, os.path.basename(path)))
    return out


def q8(rule, k, ncl):
    if rule == "michel_near":
        return k["michel_near"] >= 1
    return P97.q8(rule, k, ncl)


def layer(z, name):
    d = json.loads(z.read(f"data/0/0-{name}-global.json"))
    return np.c_[d["x"], d["y"], d["z"]], np.asarray(d["cluster_id"]), np.asarray(d["q"], float)


def runs(P):
    if len(P) == 0:
        return []
    st = np.r_[np.inf, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    brk = list(np.where(st > RUN_BREAK_CM)[0]) + [len(P)]
    return [slice(a, b) for a, b in zip(brk[:-1], brk[1:])]


def traj_metrics(img, tf, sf, cl):
    mi = img[1] == cl
    R = Ridge(img[0][mi], img[2][mi])
    out = dict(ridge_ok=R.ok)
    for tag, L in (("sf", sf), ("tf", tf)):
        m = L[1] == cl
        P, q = L[0][m], L[2][m]
        out[f"{tag}_rows"] = int(m.sum())
        out[f"{tag}_qneg"] = int((q < 0).sum())
        out[f"{tag}_holes"] = int(sum(holes(q[s]) for s in runs(P)))
        wig = np.concatenate([wiggle(P[s]) for s in runs(P)]) if len(P) else np.zeros(0)
        out[f"{tag}_wig1"] = float(np.mean(wig > 1)) if len(P) else float("nan")
        d = R.offset(P) if (R.ok and len(P)) else np.full(len(P), np.nan)
        out[f"{tag}_dev1"] = float(np.mean(d > 1)) if len(P) and R.ok else float("nan")
        out[f"{tag}_dev_p90"] = float(np.percentile(d, 90)) if len(P) and R.ok else float("nan")
    return out


def traj_ok(t):
    return (t["ridge_ok"] and t["sf_rows"] > 0 and t["tf_rows"] > 0 and t["sf_holes"] == 0
            and t["sf_dev1"] <= TRAJ_DEV1_MAX and t["tf_dev1"] <= TRAJ_DEV1_MAX and t["tf_wig1"] <= TRAJ_WIG1_MAX)


def evidence(row):
    o = row.get("owner_review")
    if isinstance(o, dict):
        txt = o.get("notes") or o.get("evidence") or o.get("comment") or ""
        if txt:
            return "owner", txt
    return "record:" + str(row.get("confidence")), row.get("evidence") or row.get("notes") or ""


def candidates():
    import glob
    import uproot
    T, src = G.truth(DET)
    REC = records_by_precedence()
    count = collections.Counter()
    pool = collections.defaultdict(list)
    for evdir in sorted(glob.glob(f"{IMG}/{DET}/work/*_{ARM}")):
        ev = os.path.basename(evdir)[:-len(ARM) - 1]
        a = uproot.open(f"{evdir}/tracking-pr.root")["T_stm_michel"].arrays(BR, library="np")
        zc, mc, lay = None, None, None
        for i in range(len(a["cluster_id"])):
            d = {b: float(a[b][i]) for b in BR}
            cl = int(d["cluster_id"]); key = f"{ev}/{cl}"
            if key not in T:
                continue
            verdict, kind, tsrc = T[key]
            row, tag, recfile = REC[key]
            assert G.U.row_truth(row, tag)[0] == verdict, key
            for cname, hv, hk, mf, sit, rule in CLASSES:
                if verdict != hv or (hk is not None and kind not in hk):
                    continue
                count[(cname, "H")] += 1
                if not (int(d["is_stm"]) == 1 and int(d["topology_cleared_bits"]) == 0):
                    continue
                count[(cname, "Q4")] += 1
                if int(d["michel_found"]) != mf:
                    continue
                count[(cname, "Q5")] += 1
                if not sit(d):
                    continue
                count[(cname, "SIT")] += 1
                if zc is None:
                    z = zipfile.ZipFile(f"{evdir}/mabc-pr.zip")
                    names = {os.path.basename(n) for n in z.namelist()}
                    need = {"0-mc.json", "0-track_fit-global.json", "0-stm_fit-global.json", "0-clustering-global.json"}
                    zc = need <= names
                    if zc:
                        mc = json.loads(z.read("data/0/0-mc.json"))
                        lay = (layer(z, "clustering"), layer(z, "track_fit"), layer(z, "stm_fit"))
                if not zc:
                    continue
                count[(cname, "Q6")] += 1
                root = P97.pf_root(mc[0], cl)
                if root is None:
                    continue
                stop = (d["stop_x"], d["stop_y"], d["stop_z"])
                kinds = P97.pf_count(root, stop)
                if set(kinds) - {"mu-", "e-", "gamma", "em_near", "gamma_near", "michel_near", "em_ke_max"}:
                    continue
                count[(cname, "Q7")] += 1
                if d["michel_ke_best"] > ENDPOINT_MEV or kinds["em_ke_max"] > ENDPOINT_MEV:
                    continue
                count[(cname, "Q9")] += 1
                ncl = int(d["michel_n_clusters"])
                if not q8(rule, kinds, ncl):
                    continue
                count[(cname, "Q8")] += 1
                tm = traj_metrics(*lay, cl)
                if not traj_ok(tm):
                    continue
                count[(cname, "TRAJ")] += 1
                owner = tsrc == "owner_review" or row.get("confidence") == "owner"
                hi = owner or row.get("confidence") == "high"
                count[(cname, "Q2")] += int(hi)
                ce = d["contrast_expected"]
                who, ev_txt = evidence(row)
                pool[cname].append(dict(
                    cls=cname, key=key, tier=0 if hi else 1, owner=owner,
                    source=("owner" if tsrc == "owner_review" else f"{tsrc}:{row.get('confidence')}") + f" ({recfile})",
                    hand=f"{verdict}/{kind}", ratio=d["contrast"] / ce if ce > 0 else float("nan"),
                    contrast=d["contrast"], muon_len=d["muon_len"], ke_best=d["michel_ke_best"],
                    ke_region=d["michel_ke_q2d_region"], kink=d["michel_kink_deg"], mlen=d["michel_len"],
                    conn=int(d["michel_conn_type"]), mdis=d["michel_dis_cm"], ncl=ncl,
                    npieces=int(d["michel_n_pieces"]), n_mgam=int(d["n_michel_gammas"]), n_dots=int(d["n_dots"]),
                    n_mu=kinds["mu-"], n_e=kinds["e-"], n_gamma=kinds["gamma"], em_near=kinds["em_near"],
                    michel_near=kinds["michel_near"], gamma_near=kinds["gamma_near"], stop=stop,
                    mstart=(d["michel_start_x"], d["michel_start_y"], d["michel_start_z"]),
                    pf=P97.pf_string(root), ev_who=who, evidence=" ".join(str(ev_txt).split()), **tm))
    return pool, count, src


def pick(pool, skip):
    used, out = set(), {}
    for cname, *_ in CLASSES:
        rows = [r for r in pool[cname] if r["key"] not in skip and r["key"] not in used]
        rows.sort(key=lambda r: (r["tier"], -r["ratio"], not r["owner"], r["n_mu"], r["key"]))
        out[cname] = rows[:NPICK]
        used |= {r["key"] for r in out[cname]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="key,... rejected by the visual check")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    pool, count, src = candidates()
    print(f"# doc pdvd/97 round 2 picks; arm {ARM}; truth sources in precedence order (items taken): {src}")
    print("# skipped after the visual check:", ", ".join(sorted(skip)) or "(none)")
    print("\n== per-class stage counts (hand items surviving each gate, in order; Q2 = owner/high among TRAJ survivors)")
    for cname, *_ in CLASSES:
        print(f"  {cname:13s} " + " | ".join(f"{s} {count[(cname, s)]}" for s in STAGES))
    picks, short = [], []
    for cname, rows in pick(pool, skip).items():
        nskip = sum(1 for r in pool[cname] if r["key"] in skip)
        tiers = collections.Counter(r["tier"] for r in pool[cname])
        print(f"\n  {cname}: usable tier0 {tiers[0]} tier1 {tiers[1]}; visually skipped {nskip}")
        if not rows:
            short.append(f"{cname} (0 of {NPICK})")
        elif len(rows) < NPICK:
            short.append(f"{cname} runner-up missing")
        for i, r in enumerate(rows):
            r["rank"] = i + 1
            picks.append(r)
            flag = "" if r["tier"] == 0 else "  ** tier 1 (any scanner source) **"
            print(f"  {'PICK' if i == 0 else 'ALT '} {cname:13s} {r['key']:14s} ratio {r['ratio']:.2f} muon {r['muon_len']:.0f} cm "
                  f"| Michel ke_best {r['ke_best']:.1f} region {r['ke_region']:.1f} kink {r['kink']:.0f} len {r['mlen']:.1f} "
                  f"conn {r['conn']} dis {r['mdis']:.1f} ncl {r['ncl']} pieces {r['npieces']} gammas {r['n_mgam']} dots {r['n_dots']} "
                  f"| traj sf {r['sf_rows']} rows holes {r['sf_holes']} dev1 {100*r['sf_dev1']:.1f}% ; tf {r['tf_rows']} rows "
                  f"dev1 {100*r['tf_dev1']:.1f}% wig1 {100*r['tf_wig1']:.1f}% | src {r['source']}{flag}")
            print(f"       stop ({r['stop'][0]:.1f}, {r['stop'][1]:.1f}, {r['stop'][2]:.1f})  PF {r['pf']}")
            print(f"       evidence [{r['ev_who']}]: {r['evidence'][:400]}")
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    cols = ["cls", "rank", "key", "tier", "source", "hand", "ratio", "contrast", "muon_len", "ke_best", "ke_region",
            "kink", "mlen", "conn", "mdis", "ncl", "npieces", "n_mgam", "n_dots", "n_mu", "n_e", "n_gamma", "em_near",
            "michel_near", "gamma_near", "sf_rows", "sf_qneg", "sf_holes", "sf_dev1", "sf_dev_p90", "tf_rows",
            "tf_qneg", "tf_dev1", "tf_dev_p90", "tf_wig1", "stop", "mstart", "pf", "ev_who", "evidence"]
    with open(OUT_TSV, "w") as f:
        f.write(f"# doc pdvd/97 round 2 -- d97r2_video_picks.py on {ARM}; skipped after visual check: "
                f"{','.join(sorted(skip)) or 'none'}\n")
        f.write("\t".join(cols) + "\n")
        for r in picks:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    v = f"{v:.3f}"
                elif c in ("stop", "mstart"):
                    v = "%.1f,%.1f,%.1f" % v
                vals.append(str(v).replace("\t", " "))
            f.write("\t".join(vals) + "\n")
    print("\nwrote", OUT_TSV, len(picks), "rows")
    fig_dqdx(picks)
    fig_traj(picks)
    if short:
        print("SHORT CLASSES:", ", ".join(short))
        return 3
    return 0


def fig_dqdx(picks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rx, ry, _chk = rr26.ref_table(DET, ARM)
    P, _ = rr26.points(DET, ARM, {p["key"] for p in picks})
    ncls = len(CLASSES)
    fig, axs = plt.subplots(math.ceil(ncls / 2), 2 * NPICK, figsize=(4.0 * 2 * NPICK, 2.8 * math.ceil(ncls / 2)),
                            squeeze=False)
    for row in axs:
        for ax in row:
            ax.set_axis_off()
    for ci, (cname, *_) in enumerate(CLASSES):
        for p in [q for q in picks if q["cls"] == cname]:
            ax = axs[ci // 2][(ci % 2) * NPICK + p["rank"] - 1]
            ax.set_axis_on()
            r_, q_, _x = P[p["key"]]
            s = r_ <= 100
            ax.plot(r_[s], q_[s] / 1e3, ".", ms=3, color="C3" if p["rank"] == 1 else "C1")
            xx = np.linspace(0.3, 100, 300)
            ax.plot(xx, np.interp(xx, rx, ry) / 1e3, "k-", lw=0.8)
            ax.set_xlim(0, 100); ax.set_ylim(0, 250)
            ax.set_title(f"{cname} {'PICK' if p['rank'] == 1 else 'alt'}  {p['key']}\nBragg ratio {p['ratio']:.2f}"
                         + ("" if p["tier"] == 0 else "  (tier 1)"), fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("residual range [cm]", fontsize=7)
            ax.set_ylabel("dQ/dx [ke/cm]", fontsize=7)
    fig.suptitle(f"doc pdvd/97 r2 ({ARM}): dQ/dx vs residual range, role-1 chain points; black = chain's dqdx_ref",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG + "_dqdx_rr.png", dpi=100)
    plt.close(fig)
    print("wrote", FIG + "_dqdx_rr.png")


def fig_traj(picks):
    """per class row: pick (zoom view 1, zoom view 2, whole cluster) then runner-up (same); grey = clustering charge,
    colour = track_fit Bee q of the candidate cluster, orange = track_fit rows of other clusters in the box,
    x = the chain's stop, + = Michel start."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    AX = {"x": 0, "y": 1, "z": 2}
    VIEWS = [("z", "x"), ("y", "x"), ("z", "y")]
    half = len(CLASSES) // 2
    for part, sub in (("a", CLASSES[:half]), ("b", CLASSES[half:])):
        fig, axs = plt.subplots(len(sub), 6, figsize=(24, 4.0 * len(sub)), squeeze=False)
        for row in axs:
            for ax in row:
                ax.set_axis_off()
        for ci, (cname, *_) in enumerate(sub):
            for p in [q for q in picks if q["cls"] == cname]:
                ev, cl = p["key"].rsplit("/", 1); cl = int(cl)
                z = zipfile.ZipFile(f"{IMG}/{DET}/work/{ev}_{ARM}/mabc-pr.zip")
                img, tf = layer(z, "clustering"), layer(z, "track_fit")
                stop, ms = np.array(p["stop"]), np.array(p["mstart"])
                mc_ = img[1] == cl
                ext = img[0][mc_].max(0) - img[0][mc_].min(0)
                spread = [ext[AX[h]] * ext[AX[v]] for h, v in VIEWS]
                vfull = VIEWS[int(np.argmax(spread))]
                vz = [v for v in VIEWS if v != vfull][:2]
                for j, (view, zoom) in enumerate(((vz[0], True), (vz[1], True), (vfull, False))):
                    ax = axs[ci][(p["rank"] - 1) * 3 + j]
                    ax.set_axis_on()
                    h, v = AX[view[0]], AX[view[1]]
                    if zoom:
                        box = np.all(np.abs(img[0] - stop) < 40, axis=1)
                        tbox = np.all(np.abs(tf[0] - stop) < 40, axis=1)
                    else:
                        box = mc_
                        tbox = tf[1] == cl
                    ax.scatter(img[0][box, h], img[0][box, v], s=1, c="0.75", lw=0)
                    oth = tbox & (tf[1] != cl)
                    ax.scatter(tf[0][oth, h], tf[0][oth, v], s=3, c="orange", lw=0)
                    mine = tbox & (tf[1] == cl)
                    sc = ax.scatter(tf[0][mine, h], tf[0][mine, v], s=3, c=tf[2][mine], cmap="viridis", vmin=0,
                                    vmax=np.percentile(tf[2][tf[1] == cl], 95) if (tf[1] == cl).any() else 1, lw=0)
                    ax.plot(stop[h], stop[v], "rx", ms=8, mew=1.5)
                    if p["cls"] not in ("dots", "bare"):
                        ax.plot(ms[h], ms[v], "m+", ms=8, mew=1.5)
                    if zoom:
                        ax.set_xlim(stop[h] - 40, stop[h] + 40); ax.set_ylim(stop[v] - 40, stop[v] + 40)
                    ax.set_aspect("equal", adjustable="datalim" if not zoom else "box")
                    ax.set_xlabel(f"{view[0]} [cm]", fontsize=7); ax.set_ylabel(f"{view[1]} [cm]", fontsize=7)
                    ax.tick_params(labelsize=6)
                    t = f"{cname} {'PICK' if p['rank'] == 1 else 'alt'} {p['key']}" if j == 0 else (
                        f"stop +-40 cm" if zoom else "whole cluster")
                    if j == 0:
                        t += f"\nsf holes {p['sf_holes']} dev>1cm {100*p['sf_dev1']:.1f}%  tf dev>1cm {100*p['tf_dev1']:.1f}% wig>1cm {100*p['tf_wig1']:.1f}%"
                    ax.set_title(t, fontsize=8)
        fig.suptitle(f"doc pdvd/97 r2 ({ARM}): track_fit (colour = Bee q, candidate cluster; orange = other clusters) "
                     f"over clustering charge (grey); red x = stop, magenta + = Michel start", fontsize=10)
        fig.tight_layout()
        out = f"{FIG}_traj_{part}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
