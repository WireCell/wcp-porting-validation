#!/usr/bin/env python3
"""doc pdvd/97 -- pick the STM / STM+Michel showcase events for the owner's video, PDHD and PDVD.

    python3 d97_video_picks.py [--skip det:key,...] > ../../scan/d97/picks.txt

Read-only on the production arms (PDHD h28prod, PDVD p96vprod) and the committed hand records; writes
pdvd/docs/scan/d97/picks.tsv and pdvd/docs/nf_sp_img_clus/figs/97_picks_dqdx_rr.png.

CLASSES (hand record, rubric pdhd/docs/scan/pdhd_stm_michel_scan_rubric.md `michel_kind`):
  dots      STM_ONLY   + detached dots   (a gamma piece near the stop, no Michel)
  attached  STM_MICHEL + attached        (a Michel piece, no gamma)
  both      STM_MICHEL + both            (a Michel piece AND an isolated gamma piece)
  bare      STM_ONLY   + none            (nothing at the stop)

QUALIFICATION, tier 0 (every rule printed with its count):
  Q1 hand verdict STM_MICHEL / STM_ONLY in the class's michel_kind (FRAG_ stripped; truth rules of
     d25_bragg_michel: PDHD owner_review > owner_smx1 > agent on smx27, PDVD merged smx1a..smx9 flat verdict)
  Q2 owner source or `high` confidence
  Q3 PDHD APA0-strict (no role-1 point in APA0), as every PDHD headline in docs pdhd/23-28
  Q4 the chain's Bragg-path accept: is_stm == 1 AND topology_cleared_bits == 0
  Q5 the chain's michel_found agrees with the hand kind (1 for attached/both, 0 for dots/bare)
  Q6 the event's production mabc-pr.zip holds mc.json and track_fit (039252_11 does not)
  Q7 the candidate's particle-flow subtree (walked from its root node) holds only mu-/e-/gamma
  Q8 the PF shows the class AT THE STOP.  An EM object = an e- node not under a gamma pseudo-carrier, or a gamma
     node (a bridged Michel or a capture gamma renders as the renderer's gamma -> e- pseudo-carrier, doc pdvd/52
     sec 7.4, so pseudo-carrier + child count once).  "near" = the node's start or end within 50 cm of the chain's
     stop (michel_gamma_radius_cm = 50.0 in both production bags, pdhd/wct-pr-perevt.jsonnet:293,
     pdvd/wct-pr-perevt.jsonnet:406; C++ default 35); "Michel-near" = within 15 cm (michel_dot_radius_cm,
     C++ default on both).  Delta rays hang off the muon upstream and fall outside both.
       dots      >= 1 gamma node near
       attached  exactly 1 EM object near, it is Michel-near, and it is not a gamma node
       both      >= 1 EM object Michel-near AND (>= 2 EM objects near, or michel_n_clusters >= 2)
       bare      no EM object near
  Q9 michel_ke_best <= 52.8 MeV, and no e-/gamma node in the PF subtree reads above 52.8 MeV (a showcase Michel may
     not read above the free-decay endpoint; an EM node above it is a mis-typed track, e.g. PDHD 029107_16/106's
     e- chain to 163 MeV in round 3)

  ROUND-1 CORRECTION (2026-09-13): round 1 used "a gamma node anywhere in the subtree" for dots/both and nothing for
  attached/bare; a bridged Michel alone satisfied it for `both` (PDHD 028084_26/109 rendered only
  `gamma 3.68 MeV(e-)`).  ROUND-2 CORRECTION: round 2 counted EM objects anywhere in the subtree, so delta rays
  hundreds of cm upstream satisfied `both` (PDHD 028084_23/51, 028084_17/97) and spoiled `attached`
  (PDVD 039253_3/66 has a second e- 12.9 cm from the stop).  Q8 is now tied to the stop as above.  Every round's
  outputs are kept (scan/d97/picks_round{1,2,3}.*, figs/97_picks_dqdx_rr_round{1,2,3}.png); doc pdvd/97 sec 2.1
  records the sequence.

RANK: Bragg clarity contrast/contrast_expected descending, then owner source first, then fewer mu- nodes in the
PF (more readable), then the key.  The dQ/dx-vs-residual-range panel is the arbiter (VISUAL CHECK below).

PRE-REGISTERED FALLBACK (written before the first run, 2026-09-13): if a class has fewer than 2 picks, relax THAT class
only -- tier 1 drops Q2 (any scanner source), tier 2 also drops Q8.  Picks are filled from the lowest tier first.
AMENDED before round 2: tier 2 is WITHHELD for dots and both, because a pick without Q8 there contradicts its own
caption (a "Michel + isolated energy" PF showing no isolated piece).  A class still short is reported, not filled.
AMENDED before round 4: tier 2 is withheld for EVERY class -- round 3 filled PDHD bare #2 at tier 2 with 4 EM objects
near the stop, the opposite of "nothing at the stop"; for every class Q8 is the caption.  Tier 1 (any scanner
source) remains the only relaxation.

VISUAL CHECK: --skip det:key drops a pick whose dQ/dx-vs-RR panel does not show a clear Bragg peak; the skipped
keys are printed and written to the tsv header so the swap is on record.
"""
import argparse, collections, json, math, os, sys, zipfile

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
os.environ.setdefault("STM_SCAN_RECORD", IMG + "/pdhd/docs/scan/pdhd_stm_michel_smx27_verdicts.json")
sys.path.insert(0, IMG + "/pdhd/docs/scan/h25")
sys.path.insert(0, IMG + "/pdhd/docs/scan/d26")
import d25_bragg_michel as m                       # noqa: E402  truth join, doc pdhd/25-26
import d26_dqdx_rr as rr26                         # noqa: E402  role-1 points + expectation table, doc pdhd/26 sec 4

ARM = {"pdhd": "h28prod", "pdvd": "p96vprod"}
CLASSES = [("dots", "STM_ONLY", "detached dots"), ("attached", "STM_MICHEL", "attached"),
           ("both", "STM_MICHEL", "both"), ("bare", "STM_ONLY", "none")]
NO_TIER2 = {"dots", "attached", "both", "bare"}   # AMENDED before round 4: every class (see header)
NPICK = 2
ENDPOINT_MEV = 52.8
MICHEL_R_CM = 15.0      # michel_dot_radius_cm, C++ default, set by neither production bag (doc pdvd/52 sec 5.1)
NEAR_R_CM = 50.0        # michel_gamma_radius_cm, 50.0 in both production bags (pdhd :293, pdvd :406); C++ default 35
OUT_TSV = IMG + "/pdvd/docs/scan/d97/picks.tsv"
OUT_FIG = IMG + "/pdvd/docs/nf_sp_img_clus/figs/97_picks_dqdx_rr.png"
m.BR += ["n_dots", "michel_ke_q2d_region", "michel_n_clusters"]
STAGES = ["Q1 hand class", "Q4 Bragg-path accept", "Q5 michel_found agrees", "Q6 zip has mc+track_fit",
          "Q7 PF only mu-/e-/gamma", "Q9 ke_best <= 52.8", "Q8 PF shows the class", "Q2 owner/high (with Q4-Q9)"]


def pf_root(node, cl):
    if isinstance(node.get("id"), int) and node["id"] // 1000 == cl:
        return node
    for c in node.get("children") or []:
        r = pf_root(c, cl)
        if r:
            return r
    return None


def _dist(node, stop):
    d = node.get("data") or {}
    ends = [p for p in (d.get("start"), d.get("end")) if p]
    return min(math.dist(p, stop) for p in ends) if ends else float("inf")


def pf_count(node, stop, parent=None, out=None):
    """-> Counter of node particle names, plus the near-stop EM census (see Q8)"""
    out = collections.Counter() if out is None else out
    t = node["text"].split()[0]
    out[t] += 1
    if t in ("e-", "gamma"):
        try:
            out["em_ke_max"] = max(out["em_ke_max"], float(node["text"].split()[1]))
        except (IndexError, ValueError):
            pass
    if (t == "e-" and parent != "gamma") or t == "gamma":
        r = _dist(node, stop)
        if r <= NEAR_R_CM:
            out["em_near"] += 1
            out["gamma_near"] += t == "gamma"
        if r <= MICHEL_R_CM:
            out["michel_near"] += 1
    for c in node.get("children") or []:
        pf_count(c, stop, t, out)
    return out


def pf_string(node):
    t = " ".join(node["text"].split())
    ch = node.get("children") or []
    return t + ("(" + ", ".join(pf_string(c) for c in ch) + ")" if ch else "")


def q8(cls, k, ncl):
    if cls == "dots":
        return k["gamma_near"] >= 1
    if cls == "attached":
        return k["em_near"] == 1 and k["michel_near"] == 1 and k["gamma_near"] == 0
    if cls == "both":
        return k["michel_near"] >= 1 and (k["em_near"] >= 2 or ncl >= 2)
    return k["em_near"] == 0


def zip_ok(det, evt):
    z = f"{IMG}/{det}/work/{evt}_{ARM[det]}/mabc-pr.zip"
    if not os.path.exists(z):
        return None
    names = {os.path.basename(n) for n in zipfile.ZipFile(z).namelist()}
    return z if {"0-mc.json", "0-track_fit-global.json"} <= names else None


def source_label(det, rec, src):
    if det == "pdhd":
        return src if src != "agent" else "agent:" + str(rec.get("confidence"))
    return "pdvd-record:" + str(rec.get("confidence"))


def evidence(det, rec, src):
    """-> (who wrote the text, the text)"""
    if det == "pdhd" and src == "owner_review":
        o = rec["owner_review"]
        txt = o.get("notes") or o.get("evidence") or ""
        if txt:
            return "owner_review", txt
    return ("agent" if det == "pdhd" else "pdvd-record:" + str(rec.get("confidence"))), rec.get("evidence") or ""


def candidates(det):
    its, missing = m.items(det, ARM[det], "strict" if det == "pdhd" else None)
    recs = {r["key"]: r for r in json.load(open(m.HD_REC if det == "pdhd" else m.VD_REC))}
    count = collections.Counter()
    pool = collections.defaultdict(list)
    mc_cache = {}
    for k, v, kind, src, d in its:          # Q3 already applied by items() for PDHD (strict)
        cls = next((c for c, cv, ck in CLASSES if cv == v and ck == kind), None)
        if cls is None:
            continue
        count[(cls, "Q1 hand class")] += 1
        rec = recs[k]
        hi = src.startswith("owner") or rec.get("confidence") in ("high", "owner")
        if not (int(d["is_stm"]) == 1 and int(d["topology_cleared_bits"]) == 0):
            continue
        count[(cls, "Q4 Bragg-path accept")] += 1
        if (int(d["michel_found"]) == 1) != (cls in ("attached", "both")):
            continue
        count[(cls, "Q5 michel_found agrees")] += 1
        evt, cl = k.split("/")
        z = zip_ok(det, evt)
        if not z:
            continue
        count[(cls, "Q6 zip has mc+track_fit")] += 1
        if z not in mc_cache:
            mc_cache[z] = json.loads(zipfile.ZipFile(z).read("data/0/0-mc.json"))
        root = pf_root(mc_cache[z][0], int(cl))
        if root is None:
            continue
        stop = (float(d["stop_x"]), float(d["stop_y"]), float(d["stop_z"]))
        kinds = pf_count(root, stop)
        if set(kinds) - {"mu-", "e-", "gamma", "em_near", "gamma_near", "michel_near", "em_ke_max"}:
            continue
        count[(cls, "Q7 PF only mu-/e-/gamma")] += 1
        if float(d["michel_ke_best"]) > ENDPOINT_MEV or kinds["em_ke_max"] > ENDPOINT_MEV:
            continue
        count[(cls, "Q9 ke_best <= 52.8")] += 1
        ncl = int(d["michel_n_clusters"])
        ok8 = q8(cls, kinds, ncl)
        count[(cls, "Q8 PF shows the class")] += ok8
        count[(cls, "Q2 owner/high (with Q4-Q9)")] += int(ok8 and hi)
        tier = 0 if (ok8 and hi) else (1 if ok8 else 2)
        if tier == 2 and cls in NO_TIER2:
            continue
        ce = float(d["contrast_expected"])
        ratio = float(d["contrast"]) / ce if ce > 0 else float("nan")
        who, ev = evidence(det, rec, src)
        pool[cls].append(dict(det=det, cls=cls, key=k, tier=tier, source=source_label(det, rec, src),
                              ratio=ratio, contrast=float(d["contrast"]), muon_len=float(d["muon_len"]),
                              ke_best=float(d["michel_ke_best"]), ke_region=float(d["michel_ke_q2d_region"]),
                              n_dots=int(d["n_dots"]), ncl=ncl, n_mu=kinds["mu-"], n_e=kinds["e-"],
                              em_near=kinds["em_near"], michel_near=kinds["michel_near"],
                              n_gamma=kinds["gamma"], gamma_near=kinds["gamma_near"], pf=pf_string(root), stop=stop,
                              owner=src.startswith("owner"), ev_who=who, evidence=" ".join(ev.split())))
    return pool, count, missing


def pick(pool, skip):
    out = {}
    for cls, _, _ in CLASSES:
        rows = [r for r in pool[cls] if f"{r['det']}:{r['key']}" not in skip]
        rows.sort(key=lambda r: (r["tier"], -r["ratio"], not r["owner"], r["n_mu"], r["key"]))
        out[cls] = rows[:NPICK]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="det:key,... picks rejected by the visual Bragg check")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    print("# doc pdvd/97 video picks; PDHD record", os.environ["STM_SCAN_RECORD"])
    print("# skipped after the visual check:", ", ".join(sorted(skip)) or "(none)")
    allpicks, short = [], []
    for det in ("pdhd", "pdvd"):
        pool, count, missing = candidates(det)
        print(f"\n== {det} arm {ARM[det]}  (record items without a candidate in the arm: {missing})")
        for cls, _, _ in CLASSES:
            print(f"  {cls:9s} " + " | ".join(f"{s} {count[(cls, s)]}" for s in STAGES))
            tiers = collections.Counter(r["tier"] for r in pool[cls])
            nskip = sum(1 for r in pool[cls] if f"{det}:{r['key']}" in skip)
            print(f"            usable by tier: 0={tiers[0]} 1={tiers[1]} 2={tiers[2]}"
                  f"{'  (tier 2 withheld)' if cls in NO_TIER2 else ''}; visually skipped {nskip}")
        for cls, rows in pick(pool, skip).items():
            if len(rows) < NPICK:
                short.append(f"{det}:{cls} ({len(rows)} of {NPICK})")
            for i, r in enumerate(rows):
                r["rank"] = i + 1
                allpicks.append(r)
                flag = "" if r["tier"] == 0 else f"  ** RELAXED tier {r['tier']} **"
                print(f"  PICK {det} {cls:9s} #{i+1} {r['key']:15s} ratio {r['ratio']:.2f} len {r['muon_len']:.0f} cm "
                      f"ke_best {r['ke_best']:.1f} region {r['ke_region']:.1f} ncl {r['ncl']} "
                      f"EM near/michel-near/gamma-near {r['em_near']}/{r['michel_near']}/{r['gamma_near']} src {r['source']}{flag}")
                print(f"       PF {r['pf']}")
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    cols = ["det", "cls", "rank", "key", "tier", "source", "ratio", "contrast", "muon_len", "ke_best", "ke_region",
            "n_dots", "ncl", "n_mu", "n_e", "n_gamma", "em_near", "michel_near", "gamma_near", "stop", "pf",
            "ev_who", "evidence"]
    with open(OUT_TSV, "w") as f:
        f.write("# doc pdvd/97 -- d97_video_picks.py; skipped after visual check: %s\n" % (",".join(sorted(skip)) or "none"))
        f.write("\t".join(cols) + "\n")
        for r in allpicks:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    v = f"{v:.3f}"
                elif c == "stop":
                    v = "%.1f,%.1f,%.1f" % v
                vals.append(str(v).replace("\t", " "))
            f.write("\t".join(vals) + "\n")
    print("\nwrote", OUT_TSV, len(allpicks), "picks")
    figure(allpicks)
    if short:
        print("SHORT CLASSES:", ", ".join(short))
        return 3
    return 0


def figure(picks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(len(CLASSES), 2 * NPICK, figsize=(4.0 * 2 * NPICK, 2.9 * len(CLASSES)), squeeze=False)
    for row in axs:
        for ax in row:
            ax.set_axis_off()
    for det_i, det in enumerate(("pdhd", "pdvd")):
        rx, ry, _chk = rr26.ref_table(det, ARM[det])
        mine = [p for p in picks if p["det"] == det]
        if not mine:
            continue
        P, _ndup = rr26.points(det, ARM[det], {p["key"] for p in mine})
        for ci, (cls, _, _) in enumerate(CLASSES):
            for p in [q for q in mine if q["cls"] == cls]:
                ax = axs[ci][det_i * NPICK + p["rank"] - 1]
                ax.set_axis_on()
                r_, q_, _x = P[p["key"]]
                s = r_ <= 100
                ax.plot(r_[s], q_[s] / 1e3, ".", ms=3, color="C0" if det == "pdhd" else "C3")
                xx = np.linspace(0.3, 100, 300)
                ax.plot(xx, np.interp(xx, rx, ry) / 1e3, "k-", lw=0.8)
                ax.set_xlim(0, 100); ax.set_ylim(0, 250)
                ax.set_title(f"{det.upper()} {cls} #{p['rank']}  {p['key']}\nBragg ratio {p['ratio']:.2f}"
                             + ("" if p["tier"] == 0 else f"  (tier {p['tier']})"), fontsize=8)
                ax.tick_params(labelsize=7)
                ax.set_xlabel("residual range [cm]", fontsize=7)
                ax.set_ylabel("dQ/dx [ke/cm]", fontsize=7)
    fig.suptitle("doc pdvd/97 picks: dQ/dx vs residual range (role-1 chain points); black = expected muon (chain's dqdx_ref)",
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110)
    print("wrote", OUT_FIG)


if __name__ == "__main__":
    sys.exit(main())
