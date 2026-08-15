"""Base rates for proposed scan-kit discriminators, over every label (pr/80 R3).

Nothing goes into the kit until it has run through here.  Doc pr/80 sec 10.8 is
the cautionary case: a flag that looked decisive on three hand-picked events was
2.2x enriched once base-rated, and would have been useless as a veto.  A
discriminator earns its place only by separating true vertices from the OTHER
vertices the scanner is choosing between in the same events -- a rate on true
vertices alone means nothing.

Round 3 proposed four additions.  Exactly one survived this script:

  B1 boundary / containment  REJECTED  3.0% of true vertices sit on a
                                       through-going cluster vs 3.4% of
                                       candidates -- x1.12, no discrimination.
  B2 co-located vertex merge SHIPPED   r=0.8 cm collapses 3901 groups over 473
                                       labels and breaks zero of them.
  B3 collinear + cold middle  REJECTED too rare to act on: 7 true / 8 other
                                       firings in 473 events.
  B4 fragment / Michel census REJECTED fires at 83.3% of true vertices and
                                       97.8% of others -- anti-correlated and
                                       useless.

Run:  cd sbnd_xin && python3 vtx_rules/flag_baserates.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines                                 # noqa: E402
import scankit                                   # noqa: E402
import vtx_geom as G                             # noqa: E402
import vtx_io                                    # noqa: E402

WALL = 2.0          # cm, an end this close to a face is "at the wall"
FRAG_R = 20.0       # cm, radius of the fragment census
FRAG_MAX = 15.0     # cm, a segment this short counts as a fragment
RADII = (0.5, 0.8, 1.0, 1.5)


def truth_vertex(dump, click):
    """The candidate closest to the owner's click, if within 1 cm, else None."""
    best, bd = None, None
    for v in scankit.candidates(dump, merge_r=0):
        d = math.dist(click, scankit.vertex_xyz(v))
        if bd is None or d < bd:
            best, bd = v, d
    return best if (bd is not None and bd <= 1.0) else None


def free_ends(dump):
    """cluster id -> [(xyz, distance to nearest face, face name)] for free ends.

    Free = the vertex at that end carries no other segment.  A junction is not
    an end of the object.
    """
    seg_of, _ = scankit.attached(dump)
    out = {}
    for s in dump.get("segments", []):
        a, b = G.seg_end_xyz(s)
        for xyz, key in ((a, "start_vertex_id"), (b, "end_vertex_id")):
            if xyz is None:
                continue
            vid = s.get(key)
            if vid is not None and len(seg_of.get(vid, [])) > 1:
                continue
            d, f = G.face_distance(xyz)
            out.setdefault(s["cluster_id"], []).append((xyz, d, f))
    return out


def through_clusters(dump):
    return {cid for cid, ends in free_ends(dump).items()
            if sum(1 for _, d, _ in ends if d <= WALL) >= 2}


def collinear_cold(dump, v, seg_of):
    """The sec 10.8 trap as a computation: two back-to-back prongs, both rising
    away, dQ/dx coolest at the junction -- one scattered particle, not two."""
    segs = seg_of.get(v["id"], [])
    if len(segs) != 2:
        return False
    ang = G.prong_angle(segs[0], segs[1], v["id"])
    if ang is None or ang < 150.0:
        return False
    for s in segs:
        end = G.end_name_of_vertex(s, v["id"])
        d0, d1, _, _ = G.end_dqdx(s)
        near, far = (d0, d1) if end == "start" else (d1, d0)
        if near is None or far is None or far / near < 1.3:
            return False
    return True


def fragments_near(dump, p):
    n = 0
    for s in dump.get("segments", []):
        if s.get("length", 0.0) > FRAG_MAX:
            continue
        if any(math.dist(p, (q["x"], q["y"], q["z"])) <= FRAG_R
               for q in (s.get("points") or [])):
            n += 1
    return n


def merge_groups(pts, r):
    """Single-link grouping of [(vertex, xyz)] at radius r."""
    groups, seen = [], set()
    for v, p in pts:
        if v["id"] in seen:
            continue
        grp = [(v, p)]
        seen.add(v["id"])
        for w, q in pts:
            if w["id"] in seen:
                continue
            if any(math.dist(u[1], q) <= r for u in grp):
                grp.append((w, q))
                seen.add(w["id"])
        groups.append(grp)
    return groups


def main():
    labs = [L for L in vtx_io.load_labels()
            if baselines.deployed_dump_path(L) and L["truth"]]
    twin = {r: 0 for r in RADII}
    broke = {r: 0 for r in RADII}
    ct = co = nt = no = 0
    wall_true = thr_true = n_true = 0
    cand_thr = cand_n = ev_thr = ev_n = 0
    frag_true = frag_other = ft_n = fo_n = 0

    for i, L in enumerate(labs):
        with open(baselines.deployed_dump_path(L)) as fh:
            d = scankit.sanitize(json.load(fh))
        cands = scankit.candidates(d, merge_r=0)
        if not cands:
            continue
        click = L["truth"]
        tv = truth_vertex(d, click)
        seg_of, _ = scankit.attached(d)
        pts = [(v, scankit.vertex_xyz(v)) for v in cands]

        for r in RADII:
            groups = merge_groups(pts, r)
            twin[r] += sum(1 for g in groups if len(g) > 1)
            if tv is None:
                continue
            for g in groups:
                if any(w["id"] == tv["id"] for w, _ in g):
                    rep = max(g, key=lambda t: (t[0].get("degree", 0) or 0,
                                                -t[0]["id"]))
                    broke[r] += (math.dist(click, rep[1]) > 1.0)
                    break

        for v in cands:
            f = collinear_cold(d, v, seg_of)
            if tv is not None and v["id"] == tv["id"]:
                nt += 1
                ct += f
            else:
                no += 1
                co += f

        thr = through_clusters(d)
        ev_n += 1
        ev_thr += bool(thr)
        for v in cands:
            cand_n += 1
            cand_thr += (v["cluster_id"] in thr)
        if tv is not None:
            n_true += 1
            p = scankit.vertex_xyz(tv)
            wall_true += (G.face_distance(p)[0] <= WALL)
            thr_true += (tv["cluster_id"] in thr)
            ft_n += 1
            frag_true += bool(fragments_near(d, p))
            others = [v for v in cands if v["id"] != tv["id"]]
            if others:
                fo_n += 1
                frag_other += bool(fragments_near(
                    d, scankit.vertex_xyz(others[len(others) // 2])))
        if (i + 1) % 100 == 0:
            print("... %d/%d" % (i + 1, len(labs)), flush=True)

    pct = lambda a, b: 100.0 * a / max(1, b)            # noqa: E731
    print("\n%d labels with a usable dump and click\n" % len(labs))
    print("B2 co-located vertex merging  (SHIPPED at r=%.1f)" % scankit.MERGE_R)
    for r in RADII:
        print("  r=%.1f cm: %5d merged groups; labels BROKEN (representative "
              ">1 cm from its click): %d" % (r, twin[r], broke[r]))
    print("\nB3 collinear(>=150 deg) + cold middle  (REJECTED: too rare)")
    print("  at TRUE  vertices: %d/%d (%.2f%%)" % (ct, nt, pct(ct, nt)))
    print("  at OTHER vertices: %d/%d (%.2f%%)" % (co, no, pct(co, no)))
    print("\nB1 boundary / containment  (REJECTED: no discrimination)")
    print("  true vertex within %.0f cm of a face: %d/%d (%.1f%%)"
          % (WALL, wall_true, n_true, pct(wall_true, n_true)))
    print("  TRUE      vertices on a through-going cluster: %d/%d (%.1f%%)"
          % (thr_true, n_true, pct(thr_true, n_true)))
    print("  CANDIDATE vertices on a through-going cluster: %d/%d (%.1f%%)"
          % (cand_thr, cand_n, pct(cand_thr, cand_n)))
    if thr_true and cand_n:
        print("  ratio %.2fx -- a wall-touching cluster is essentially as "
              "likely to hold the vertex as any other"
              % ((cand_thr / cand_n) / (thr_true / n_true)))
    print("  events with >=1 through-going cluster: %d/%d (%.1f%%)"
          % (ev_thr, ev_n, pct(ev_thr, ev_n)))
    print("\nB4 fragments within %.0f cm  (REJECTED: anti-correlated)" % FRAG_R)
    print("  TRUE  vertices with >=1 fragment near: %d/%d (%.1f%%)"
          % (frag_true, ft_n, pct(frag_true, ft_n)))
    print("  OTHER vertices with >=1 fragment near: %d/%d (%.1f%%)"
          % (frag_other, fo_n, pct(frag_other, fo_n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
