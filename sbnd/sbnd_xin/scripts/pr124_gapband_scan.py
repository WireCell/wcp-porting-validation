#!/usr/bin/env python3
"""doc pr/124 front A -- gap-band qualifier scan (fork of pr123_prune_scan.py).

pr/123 shipped the final-body single-linkage prune at G=40 (SBND ON); G=25/30
were rejected on 1:1 collateral.  The worst residual rows (406125 qF1 0.097,
94392, 175896, 286655, 283515) sit exactly in the 25-40 cm band.  This scan
asks the owner's question: does a per-component QUALIFIER tame the collateral
that killed G=25?

Fork rationale (convention: fork, don't extend a shipped scan script): the
pr/123 scan aggregates member-level prune counts; this one materializes each
DETACHED COMPONENT as a row with qualifier features and a label class, then
runs single-qualifier cut searches.  Runs on post-flip production dumps
(work-pr123r1-r21flip141-*, work-d84r2-prod98-*), so surviving components are
exactly the ones a tighter second-tier prune would judge.

Component classes (component = single-linkage piece at G not containing the
shower start segment):
  BAD  -- contains >=1 OUT-marked member and no IN-marked member (want pruned)
  COL  -- contains >=1 IN-marked member and no OUT (must NOT be pruned)
  MIX  -- contains both (pruning is partly wrong either way)
  UNL  -- only unlabeled members (unknown; report, don't count as collateral)

Qualifiers per component:
  gap_cm   -- min point-pair distance to the keep (start-segment) component
  q_comp   -- summed point dQ;  q_frac = q_comp / whole-shower summed dQ
  len_sum  -- summed member lengths (cm)
  trk_pid  -- 1 if any member |pdg| in {13,211,2212}
  mdqdx    -- median point dQ/dx over the component / muon-plateau (MIP units,
              plateau taken from the dump's own dqdx_ref muon table tail)
  ang_body -- angle (deg) at the shower start vertex between the component
              charge centroid and the keep-component charge centroid

Repro:
  ./scripts/pr124_gapband_scan.py 'work-pr123r1-r21flip141-*' 'work-d84r2-prod98-*' \
      --tsv docs/pr/pr124-gapband-components.tsv
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
LABEL_DIRS = [os.path.join(SX, "em_labels", t)
              for t in ("emscan-0827", "emscan-0828-agent5")]
GS = [25.0, 30.0, 35.0, 40.0]
TRK_PIDS = {13, 211, 2212}


def load_labels(ev):
    for ld in LABEL_DIRS:
        p = os.path.join(ld, "labels-evt%d.json" % ev)
        if os.path.exists(p):
            em = json.load(open(p)).get("em") or {}
            marks = em.get("marks_by_shower") or {}
            out = {}
            for shw, mm in marks.items():
                ins = {int(s) for s, v in mm.items() if v == "in"}
                outs = {int(s) for s, v in mm.items() if v == "out"}
                out[int(shw)] = (ins, outs)
            return {"marks": out, "tag": os.path.basename(ld)}
    return None


def seg_pts(seg):
    return np.array([[p["x"], p["y"], p["z"]] for p in seg["points"]], dtype=float)


def min_dist(a, b):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return math.sqrt(d2.min())


def components(ids, dmat, gap):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in ids:
        for j in ids:
            if i < j and dmat[(i, j)] < gap:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    comp = {}
    for i in ids:
        comp.setdefault(find(i), set()).add(i)
    return list(comp.values())


def angle_deg(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return -1.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tsv", help="write component rows here")
    ap.add_argument("--seg-join", action="store_true",
                    help="join marks at SEGMENT level to each seg's CURRENT "
                         "owner shower (recovers events whose membership was "
                         "restructured after labeling, e.g. post pr/123 flip "
                         "286655/283515 -- an OUT mark recorded under an old "
                         "shower key follows the segment; approximation is "
                         "flagged in the doc)")
    args = ap.parse_args()
    roots = [r for g in args.roots for r in sorted(glob.glob(g))]

    rows = []          # component rows across all G
    seen = set()       # (ev, shw) dedup across overlapping arm roots
    for root in roots:
        for dj in sorted(glob.glob(os.path.join(root, "pr_evt*", "calib-pr-evt*.json"))):
            ev = int(os.path.basename(os.path.dirname(dj))[len("pr_evt"):])
            lab = load_labels(ev)
            if not lab:
                continue
            j = json.load(open(dj))
            segs = {s["id"]: s for s in j.get("segments", [])}
            verts = {v["id"]: v for v in j.get("vertices", [])}
            mip = (j.get("dqdx_ref", {}).get("muon") or [54657.7])[-1]
            marks = lab["marks"]
            if args.seg_join:
                # Re-key marks to each marked segment's CURRENT owner shower O:
                #   - a mark recorded for O itself keeps its verdict;
                #   - IN-for-X while owned by O != X means the seg belongs to X,
                #     so it counts OUT for O;
                #   - only OUT-of-elsewhere recorded => uninformative for O:
                #     dropped to unlabeled (the stale-label class, e.g. post
                #     pr/123-flip 286655/283515 -- listed for re-scan, never
                #     scored on a guess).
                orig = marks
                all_in = {s for ins_, _ in orig.values() for s in ins_}
                marks = {}
                for shw_x, (ins_x, outs_x) in orig.items():
                    for sid in ins_x | outs_x:
                        s = segs.get(sid)
                        owner = s.get("shower_id") if s else None
                        if owner is None or owner < 0 or owner not in segs:
                            continue
                        ins_o, outs_o = marks.setdefault(owner, (set(), set()))
                        if owner == shw_x:
                            (ins_o if sid in ins_x else outs_o).add(sid)
                        elif sid in all_in:
                            outs_o.add(sid)
                for owner, (ins_o, outs_o) in marks.items():
                    # a direct verdict for the owner wins over inferred OUT
                    outs_o -= ins_o
            for shw_key, (ins, outs) in marks.items():
                if (ev, shw_key) in seen:
                    continue
                members = [sid for sid, s in segs.items() if s.get("shower_id") == shw_key]
                if shw_key not in segs or len(members) < 2:
                    continue
                seen.add((ev, shw_key))
                pts = {sid: seg_pts(segs[sid]) for sid in members if segs[sid].get("points")}
                ids = [sid for sid in members if sid in pts and len(pts[sid])]
                if shw_key not in ids or len(ids) < 2:
                    continue
                dmat = {}
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        i, jd = ids[a], ids[b]
                        dmat[(min(i, jd), max(i, jd))] = min_dist(pts[i], pts[jd])
                sv = verts.get(segs[shw_key].get("start_vertex_id"))
                svp = (np.array([sv["fit"]["x"], sv["fit"]["y"], sv["fit"]["z"]])
                       if sv and isinstance(sv.get("fit"), dict) else pts[shw_key][0])

                def comp_q(cids):
                    q = 0.0
                    for sid in cids:
                        q += sum(abs(p.get("dQ", 0.0)) for p in segs[sid]["points"])
                    return q

                q_shower = comp_q(ids)
                for g in GS:
                    comps = components(sorted(ids), dmat, g)
                    keep = next(c for c in comps if shw_key in c)
                    kpts = np.vstack([pts[i] for i in sorted(keep)])
                    kq = np.concatenate(
                        [[abs(p.get("dQ", 0.0)) for p in segs[i]["points"]] for i in sorted(keep)])
                    kcen = (kpts * kq[:, None]).sum(0) / max(kq.sum(), 1e-9)
                    for c in comps:
                        if shw_key in c:
                            continue
                        cs = sorted(c)
                        n_out = sum(1 for s in cs if s in outs)
                        n_in = sum(1 for s in cs if s in ins)
                        n_unl = len(cs) - n_out - n_in
                        klass = ("MIX" if n_out and n_in else
                                 "BAD" if n_out else
                                 "COL" if n_in else "UNL")
                        cpts = np.vstack([pts[i] for i in cs])
                        cq = np.concatenate(
                            [[abs(p.get("dQ", 0.0)) for p in segs[i]["points"]] for i in cs])
                        ccen = (cpts * cq[:, None]).sum(0) / max(cq.sum(), 1e-9)
                        dqdx = []
                        for sid in cs:
                            for p in segs[sid]["points"]:
                                dx = p.get("dx", 0.0)
                                if dx > 1e-6:
                                    dqdx.append(abs(p.get("dQ", 0.0)) / dx / mip)
                        gap_cm = min(min_dist(pts[i], pts[k])
                                     for i in cs for k in sorted(keep))
                        rows.append(dict(
                            ev=ev, shw=shw_key, tag=lab["tag"], G=g,
                            nmem=len(cs), members=";".join(map(str, cs)),
                            n_out=n_out, n_in=n_in, n_unl=n_unl, klass=klass,
                            gap_cm=round(gap_cm, 2),
                            q_comp=round(comp_q(cs), 1),
                            q_frac=round(comp_q(cs) / max(q_shower, 1e-9), 4),
                            len_sum=round(sum(segs[i].get("length", 0.0) for i in cs), 2),
                            trk_pid=int(any(abs(segs[i].get("particle_id", 0)) in TRK_PIDS
                                            for i in cs)),
                            mdqdx=round(float(np.median(dqdx)), 3) if dqdx else -1.0,
                            ang_body=round(angle_deg(ccen - svp, kcen - svp), 1),
                        ))

    cols = ["ev", "shw", "tag", "G", "nmem", "n_out", "n_in", "n_unl", "klass",
            "gap_cm", "q_comp", "q_frac", "len_sum", "trk_pid", "mdqdx",
            "ang_body", "members"]
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print("wrote %d component rows -> %s" % (len(rows), args.tsv))

    # -- summary: components by class per G (band components only appear at
    #    the G that first detaches them; a row at G=25 absent at G=40 is the
    #    25-40 band).
    print("\nDetached components by class per G:")
    print("  G_cm   BAD   COL   MIX   UNL")
    for g in GS:
        cnt = {k: 0 for k in ("BAD", "COL", "MIX", "UNL")}
        for r in rows:
            if r["G"] == g:
                cnt[r["klass"]] += 1
        print("  %4.0f  %4d  %4d  %4d  %4d" % (g, cnt["BAD"], cnt["COL"],
                                               cnt["MIX"], cnt["UNL"]))

    # -- the band: components detached at G=25 that are NOT detached at G=40
    #    (i.e. what a second-tier prune would newly judge).
    at40 = {(r["ev"], r["shw"], r["members"]) for r in rows if r["G"] == 40.0}
    band = [r for r in rows if r["G"] == 25.0
            and (r["ev"], r["shw"], r["members"]) not in at40]
    print("\n25-40 band components (detached at G=25, still attached at G=40): %d"
          % len(band))
    for r in sorted(band, key=lambda r: (r["klass"], -r["q_comp"])):
        print("  %-4s evt%-7d shw=%-7d n=%-2d out/in/unl=%d/%d/%d gap=%-6.1f "
              "q=%-10.0f qf=%-6.3f len=%-6.1f trk=%d mdqdx=%-6.2f ang=%-6.1f mem=%s"
              % (r["klass"], r["ev"], r["shw"], r["nmem"], r["n_out"], r["n_in"],
                 r["n_unl"], r["gap_cm"], r["q_comp"], r["q_frac"], r["len_sum"],
                 r["trk_pid"], r["mdqdx"], r["ang_body"], r["members"]))

    # -- single-qualifier cut search over the band: a cut prunes the component
    #    when the predicate holds; score = BAD caught vs COL/MIX hit (IN-side
    #    collateral).  UNL reported but not scored.
    print("\nSingle-qualifier cut search over the band "
          "(caught BAD / hit COL+MIX / hit UNL):")
    nbad = sum(1 for r in band if r["klass"] == "BAD")
    ncol = sum(1 for r in band if r["klass"] in ("COL", "MIX"))
    print("  band totals: BAD=%d COL+MIX=%d UNL=%d"
          % (nbad, ncol, sum(1 for r in band if r["klass"] == "UNL")))
    cuts = []
    for thr in (0.05, 0.1, 0.15, 0.2, 0.3):
        cuts.append(("q_frac<%.2f" % thr, lambda r, t=thr: r["q_frac"] < t))
    for thr in (5, 10, 15, 20):
        cuts.append(("len_sum<%d" % thr, lambda r, t=thr: r["len_sum"] < t))
    for thr in (30, 45, 60, 90):
        cuts.append(("ang_body>%d" % thr, lambda r, t=thr: r["ang_body"] > t))
    for thr in (1.3, 1.6, 2.0):
        cuts.append(("mdqdx<%.1f" % thr, lambda r, t=thr: 0 <= r["mdqdx"] < t))
    cuts.append(("trk_pid==1", lambda r: r["trk_pid"] == 1))
    for thr in (30, 35):
        cuts.append(("gap>%d" % thr, lambda r, t=thr: r["gap_cm"] > t))
    for name, fn in cuts:
        cb = sum(1 for r in band if r["klass"] == "BAD" and fn(r))
        cc = sum(1 for r in band if r["klass"] in ("COL", "MIX") and fn(r))
        cu = sum(1 for r in band if r["klass"] == "UNL" and fn(r))
        flag = "  <-- zero-collateral" if cc == 0 and cb else ""
        print("  %-14s %3d/%d   %3d/%d   %3d%s" % (name, cb, nbad, cc, ncol, cu, flag))


if __name__ == "__main__":
    main()
