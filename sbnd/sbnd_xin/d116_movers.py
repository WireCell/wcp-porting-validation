#!/usr/bin/env python3
"""doc sbnd_xin/116: where the two arms disagree, adjudicated by TRUTH -- the mover taxonomy.

Doc pr/150 sec 4 had to classify vertex movers by a blind two-arm hand scan.  With the round-3 MC
the true vertex is known, so every disagreement between the baseline arm A and a cell arm B is a
table, not a scan.  Two questions, each answered for BOTH directions (A-only = lost, B-only = gained):

1. VERTEX movers -- true interactions in the FV whose nearest candidate vertex is within 5 cm in one
   arm only.  For a lost one, the cell arm's calib dump (calib-pr-evt<ID>.json: main_vertex, the
   `vertices` list with its main_candidate flags, vertex_scoreboard) says WHY, in doc pr/150's classes:
     no candidate     the cell arm has no T_tagger row for the event (candidate / bundle lost)
     not evaluated    a candidate exists but the neutrino PR did not run on it (no calib dump)
     CHOICE           a main-candidate vertex within 5 cm of the truth EXISTS in the cell arm;
                      the chooser (NeutrinoVertexFinder, DL re-rank) picked another one
     STRUCTURE-near   a vertex within 5 cm exists in the segmentation but is not a main candidate
     STRUCTURE        no vertex of the segmentation within 5 cm of the truth: the trajectory /
                      segment structure itself moved
   with the DL scoreboard route in both arms, the true class, Edep bin, the cathode band (|x| < 15 cm),
   and where the cell arm's nearest candidate ended up (5-20 / 20-50 / > 50 cm).

2. SELECTION movers -- signal interactions (numuCC for numu > 0.9, nueCC for nue > 7 / > 4) selected in
   one arm only, attributed to the FIRST failing stage in the other arm:
     no candidate -> not matched (< 5 cm) -> reco vertex outside FV -> score below the cut
   and, for the score stage, the score and reco_Enu deltas plus which tagger sub-scores / flags of the
   calib `tagger` block changed between the arms (counted over the movers, so the doc can name the
   BDT inputs that moved rather than guess).

Usage:
  python3 d116_movers.py --sample cv --cell tfull [--base-arm work-r3cv-d115pr] [--out docs/116_figs]
  -> <out>/116_movers_<sample>-<cell>.txt (report) and .tsv (one row per mover)
"""
import argparse
import collections
import csv
import glob
import json
import math
import os

MATCH_CM = 5.0
SCORE = {"numu": "numu_score", "nue": "nue_score"}
CUTS = [("numu", 0.9), ("nue", 7.0), ("nue", 4.0)]
f = float


def in_fv(x, y, z):
    return 5.0 < abs(x) < 190.0 and abs(y) < 190.0 and 10.0 < z < 450.0


def tclass(flav, ccnc):
    if ccnc == "NC":
        return "NC"
    return {"numu": "numuCC", "nue": "nueCC"}.get(flav, "CC-" + flav)


def band(d):
    if d is None or d < 0:
        return "no cand"
    return "<5" if d < 5 else "5-20" if d < 20 else "20-50" if d < 50 else ">50"


def edep_bin(e):
    return "<100" if e < 100 else "100-300" if e < 300 else ">=300"


ev = lambda r: (r["run"], r["subrun"], r["event"])


class Arm:
    def __init__(self, products, workdir):
        self.products, self.workdir = products, workdir
        self.C = list(csv.DictReader(open(os.path.join(products, "candidates.tsv")), delimiter="\t"))
        self.T = list(csv.DictReader(open(os.path.join(products, "truth.tsv")), delimiter="\t"))
        self.truth = {}
        for t in self.T:
            t["fv"] = in_fv(f(t["vx"]), f(t["vy"]), f(t["vz"]))
            t["cls"] = tclass(t["flav"], t["ccnc"])
            t["key"] = (ev(t), t["idx"])
            self.truth[t["key"]] = t
        self.by_ev = collections.defaultdict(list)
        for r in self.C:
            r["reco_fv"] = r["vertex_default"] == "0" and in_fv(f(r["nu_x"]), f(r["nu_y"]), f(r["nu_z"]))
            r["matched"] = r["vertex_default"] == "0" and 0 <= f(r["t_dist_cm"]) < MATCH_CM
            self.by_ev[ev(r)].append(r)
        # (run, subrun, event) -> pr_evt dir, from the per-event nusel tables (tiny, always written)
        self.dirs = {}
        for p in glob.glob(os.path.join(workdir, "f*", "pr_evt*", "nusel-evt*.tsv")) + \
                 glob.glob(os.path.join(workdir, "pr_evt*", "nusel-evt*.tsv")):
            with open(p) as fh:           # whitespace-aligned columns (run subrun event ...), not tabs
                fh.readline()
                row = fh.readline().split()
            if len(row) >= 3:
                self.dirs[(row[0], row[1], row[2])] = os.path.dirname(p)
        self._calib = {}

    def matched(self, key):
        t = self.truth[key]
        return 0 <= f(t["min_cand_dist_cm"]) < MATCH_CM

    def nearest_dist(self, key):
        t = self.truth[key]
        return None if t["event_has_candidate"] != "1" else f(t["min_cand_dist_cm"])

    def calib(self, e):
        if e in self._calib:
            return self._calib[e]
        d = self.dirs.get(e)
        j = None
        if d:
            fs = glob.glob(os.path.join(d, "calib-pr-evt*.json"))
            if fs:
                try:
                    j = json.load(open(fs[0]))
                except Exception:
                    j = None
        self._calib[e] = j
        return j

    def cand_for(self, key):
        """The candidate row matched to this true interaction (nearest, < 5 cm), or None."""
        e, idx = key
        rows = [r for r in self.by_ev.get(e, []) if r["matched"] and r["t_idx"] == idx]
        return min(rows, key=lambda r: f(r["t_dist_cm"])) if rows else None

    def signal_selected(self, flav, cut):
        col, sig = SCORE[flav], flav + "CC"
        out = set()
        for r in self.C:
            if r["reco_fv"] and f(r[col]) > cut and r["matched"]:
                t = self.truth.get((ev(r), r["t_idx"]))
                if t is not None and t["fv"] and t["cls"] == sig:
                    out.add(t["key"])
        return out


def vertex_class(arm, key):
    """Why arm B does not match true interaction `key` (which arm A matched)."""
    t = arm.truth[key]
    e = key[0]
    vt = (f(t["vx"]), f(t["vy"]), f(t["vz"]))
    if t["event_has_candidate"] != "1":
        return "no candidate", {}
    j = arm.calib(e)
    if j is None:
        return "not evaluated", {}
    info = {}
    mv = j.get("main_vertex") or {}
    if "x" in mv:
        info["d_main"] = math.dist(vt, (mv["x"], mv["y"], mv["z"]))
    vs = j.get("vertices") or []
    def dv(v):
        p = v.get("fit") or {}
        return math.dist(vt, (p["x"], p["y"], p["z"])) if "x" in p else math.inf
    d_all = min((dv(v) for v in vs), default=math.inf)
    d_mc = min((dv(v) for v in vs if v.get("main_candidate")), default=math.inf)
    info["d_any_vertex"], info["d_main_candidate"] = d_all, d_mc
    sb = j.get("vertex_scoreboard") or {}
    info["route"] = sb.get("route", "")
    info["dl_accepted"] = sb.get("dl_accepted", "")
    if d_mc < MATCH_CM:
        return "CHOICE", info
    if d_all < MATCH_CM:
        return "STRUCTURE-near", info
    return "STRUCTURE", info


def route_of(arm, e):
    j = arm.calib(e)
    if j is None:
        return "-"
    sb = j.get("vertex_scoreboard") or {}
    return f"{sb.get('route', '')}/{'dl' if sb.get('dl_accepted') else 'nodl'}"


def stage_of(arm, key, flav, cut):
    """First failing selection stage of `key` in `arm`."""
    t = arm.truth[key]
    if t["event_has_candidate"] != "1":
        return "no candidate", None
    r = arm.cand_for(key)
    if r is None:
        return "not matched (< 5 cm)", None
    if not r["reco_fv"]:
        return "reco vertex outside FV", r
    if f(r[SCORE[flav]]) <= cut:
        return "score below cut", r
    return "selected", r


def tagger_diff(ja, jb):
    """Names of tagger sub-scores / flags that differ between two calib dumps."""
    ta, tb = (ja or {}).get("tagger") or {}, (jb or {}).get("tagger") or {}
    out = []
    for k in sorted(set(ta) | set(tb)):
        va, vb = ta.get(k), tb.get(k)
        if va == vb:
            continue
        try:
            if abs(float(va) - float(vb)) < 1e-6:
                continue
        except (TypeError, ValueError):
            pass
        out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, choices=["cv", "nuecc"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--base-products", default=None)
    ap.add_argument("--base-arm", default=None)
    ap.add_argument("--cell-products", default=None)
    ap.add_argument("--cell-arm", default=None)
    ap.add_argument("--out", default="docs/116_figs")
    a = ap.parse_args()
    s = a.sample
    wtag = {"cv": "cv", "nuecc": "nue"}[s]
    A = Arm(a.base_products or f"products/d115/{s}", a.base_arm or f"work-r3{wtag}-d115pr")
    B = Arm(a.cell_products or f"products/d116/{s}-{a.cell}", a.cell_arm or f"work-r3{wtag}-d116{a.cell}")
    assert set(A.truth) == set(B.truth), "truth rows differ between arms"
    fv = [k for k, t in A.truth.items() if t["fv"]]

    out, rows = [], []
    p = out.append
    p(f"# doc 116 movers -- sample {s}, baseline vs {a.cell}, adjudicated by truth (d116_movers.py)")
    p(f"baseline {A.products} ({len(A.dirs)} event dirs)   cell {B.products} ({len(B.dirs)} event dirs)")
    p(f"true interactions in FV: {len(fv)}")

    # ---- 1. vertex movers
    ma = {k for k in fv if A.matched(k)}
    mb = {k for k in fv if B.matched(k)}
    p("")
    p(f"## 1. vertex within 5 cm: baseline {len(ma)}  cell {len(mb)}  lost {len(ma-mb)}  gained {len(mb-ma)}  both {len(ma&mb)}")
    for label, keys, other, own in (("LOST (matched in baseline only) -- classified in the CELL arm", sorted(ma - mb), B, A),
                                    ("GAINED (matched in cell only) -- classified in the BASELINE arm", sorted(mb - ma), A, B)):
        p(f"### {label}: {len(keys)}")
        cls = collections.Counter()
        by_class = collections.defaultdict(collections.Counter)
        by_edep = collections.defaultdict(collections.Counter)
        by_cath = collections.defaultdict(collections.Counter)
        by_band = collections.defaultdict(collections.Counter)
        by_route = collections.Counter()
        for k in keys:
            t = A.truth[k]
            c, info = vertex_class(other, k)
            cls[c] += 1
            by_class[t["cls"]][c] += 1
            by_edep[edep_bin(f(t["Edep"]))][c] += 1
            by_cath["cathode |x|<15" if abs(f(t["vx"])) < 15 else "bulk"][c] += 1
            by_band[band(other.nearest_dist(k))][c] += 1
            by_route[(route_of(own, k[0]), route_of(other, k[0]))] += 1
            rows.append(dict(direction=label.split()[0], run=k[0][0], subrun=k[0][1], event=k[0][2], idx=k[1],
                             cls=t["cls"], Edep="%.1f" % f(t["Edep"]), vx="%.1f" % f(t["vx"]), vy="%.1f" % f(t["vy"]),
                             vz="%.1f" % f(t["vz"]), kind="vertex", klass=c,
                             d_other="%.1f" % (other.nearest_dist(k) if other.nearest_dist(k) is not None else -1),
                             d_main="%.1f" % info.get("d_main", -1), d_main_candidate="%.1f" % min(info.get("d_main_candidate", -1), 999),
                             d_any_vertex="%.1f" % min(info.get("d_any_vertex", -1), 999),
                             route_own=route_of(own, k[0]), route_other=route_of(other, k[0]), flav="", cut="", stage="",
                             score_own="", score_other="", dEnu="", tagger_changed=""))
        order = ["no candidate", "not evaluated", "CHOICE", "STRUCTURE-near", "STRUCTURE"]
        p("  class:        " + "  ".join(f"{c} {cls[c]}" for c in order))
        for title, d in (("by true class", by_class), ("by Edep bin", by_edep), ("cathode band", by_cath),
                         ("other arm's nearest-candidate band", by_band)):
            p(f"  {title}:")
            for g in sorted(d):
                p(f"    {g:14s} " + "  ".join(f"{c} {d[g][c]}" for c in order if d[g][c]) + f"   (n={sum(d[g].values())})")
        p("  DL route (own arm -> other arm), top 6: " + "; ".join(f"{a_}->{b_}: {n}" for (a_, b_), n in by_route.most_common(6)))

    # ---- 2. selection movers
    for flav, cut in CUTS:
        sig = flav + "CC"
        sa, sb = A.signal_selected(flav, cut), B.signal_selected(flav, cut)
        n_sig = sum(1 for k in fv if A.truth[k]["cls"] == sig)
        p("")
        p(f"## 2. {sig} selected ({SCORE[flav]} > {cut:g}, reco vertex in FV): signal {n_sig}; baseline {len(sa)}  cell {len(sb)}  lost {len(sa-sb)}  gained {len(sb-sa)}")
        if not n_sig:
            continue
        for label, keys, other, own in (("LOST -- first failing stage in the CELL arm", sorted(sa - sb), B, A),
                                        ("GAINED -- first failing stage in the BASELINE arm", sorted(sb - sa), A, B)):
            st = collections.Counter()
            tag = collections.Counter()
            dsc, denu = [], []
            for k in keys:
                t = A.truth[k]
                stage, r_other = stage_of(other, k, flav, cut)
                st[stage] += 1
                r_own = own.cand_for(k)
                so = f(r_own[SCORE[flav]]) if r_own else float("nan")
                sot = f(r_other[SCORE[flav]]) if r_other else float("nan")
                changed = []
                if stage == "score below cut":
                    dsc.append(sot - so)
                    if r_own and r_other and f(r_own["reco_Enu"]) > 0 and f(r_other["reco_Enu"]) > 0:
                        denu.append(f(r_other["reco_Enu"]) - f(r_own["reco_Enu"]))
                    changed = tagger_diff(own.calib(k[0]), other.calib(k[0]))
                    for c in changed:
                        tag[c] += 1
                rows.append(dict(direction=label.split()[0], run=k[0][0], subrun=k[0][1], event=k[0][2], idx=k[1],
                                 cls=t["cls"], Edep="%.1f" % f(t["Edep"]), vx="%.1f" % f(t["vx"]), vy="%.1f" % f(t["vy"]),
                                 vz="%.1f" % f(t["vz"]), kind=f"sel_{flav}{cut:g}", klass="",
                                 d_other="%.1f" % (other.nearest_dist(k) if other.nearest_dist(k) is not None else -1),
                                 d_main="", d_main_candidate="", d_any_vertex="",
                                 route_own=route_of(own, k[0]), route_other=route_of(other, k[0]),
                                 flav=flav, cut=cut, stage=stage, score_own="%.3f" % so, score_other="%.3f" % sot,
                                 dEnu=("%.1f" % (f(r_other["reco_Enu"]) - f(r_own["reco_Enu"])) if r_own and r_other else ""),
                                 tagger_changed=",".join(changed)))
            p(f"### {label}: {len(keys)}")
            p("  stage: " + "  ".join(f"{k_} {v}" for k_, v in st.most_common()))
            if dsc:
                dsc.sort()
                p(f"  score delta (other - own) on the score-stage movers: median {dsc[len(dsc)//2]:+.3f}, "
                  f"min {dsc[0]:+.3f}, max {dsc[-1]:+.3f}")
            if denu:
                denu.sort()
                p(f"  reco_Enu delta (MeV) on those: median {denu[len(denu)//2]:+.1f}, |d|>25 MeV in {sum(abs(x) > 25 for x in denu)}/{len(denu)}")
            if tag:
                p("  tagger sub-scores / flags that changed (count over score-stage movers), top 12: "
                  + ", ".join(f"{k_} {v}" for k_, v in tag.most_common(12)))

    txt = "\n".join(out) + "\n"
    print(txt)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"116_movers_{s}-{a.cell}")
    open(stem + ".txt", "w").write(txt)
    cols = ["direction", "kind", "run", "subrun", "event", "idx", "cls", "Edep", "vx", "vy", "vz", "klass", "d_other",
            "d_main", "d_main_candidate", "d_any_vertex", "route_own", "route_other", "flav", "cut", "stage",
            "score_own", "score_other", "dEnu", "tagger_changed"]
    with open(stem + ".tsv", "w") as fo:
        fo.write("\t".join(cols) + "\n")
        for r in rows:
            fo.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"-> {stem}.txt / .tsv")


if __name__ == "__main__":
    main()
