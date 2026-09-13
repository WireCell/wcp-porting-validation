#!/usr/bin/env python3
"""doc pdhd/27 sec 2 -- the wrapped-wire face defect in michel_q2d's region distances, from the geometry and the cell table.

    python3 d27_wrap_face.py > wrap_face.txt

CLAIM UNDER TEST: CheckSTM_Michel.cxx:2001 starts every cell at fw.front() -- the FIRST (face, wire) its channel maps to
-- and only any_within (a Michel / gamma cloud match) moves it.  d_stop_cm / d_ctl_cm (2034-2044) are computed on that
(face, wire).  On PDHD every U/V channel is wrapped onto both faces of its APA, and AnodePlane (gen/src/AnodePlane.cxx
~215-223, faces -> planes -> wires) lists the face-0 wire first.  So on an APA whose active face is 1, a muon-footprint
U/V cell's distance is measured on the far face's wire, i.e. is not a distance.

1. GEOMETRY: channel -> wires from protodunehd-wires-larsoft-v1.json.bz2 in AnodePlane's order; face multiplicity and
   the face of the first entry, per (apa, plane).
2. INDEX CONVENTION: the chain's wire index against the file order, per face, on every recorded cell.  (Face 1 turns out
   to be mirrored, n - 1 - file index; needed by d27_ghost_twin.py.)
3. TABLE: every role-1 U/V cell's recorded (face, wire) is / is not the first geometry entry.
4. SCALE over all candidates of both production arms: the muon's face (the W role-1 cells' face -- collection channels
   are not wrapped), the fraction of U/V role-1 cells recorded on another face, and the control's U/V cell counts.
   PDVD is the comparison.
"""
import bz2, json, glob, collections
import numpy as np, uproot

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
GEOM = "/nfs/data/1/xqian/toolkit-dev/wire-cell-data/protodunehd-wires-larsoft-v1.json.bz2"
ARMS = (("pdhd", "h26q2dprod"), ("pdvd", "p96vprod"))
DUMP = {}

g = json.load(bz2.open(GEOM))["Store"]
c2w = collections.defaultdict(list); NW = {}
for a in g["anodes"]:
    a = a["Anode"]
    for fi in a["faces"]:
        f = g["faces"][fi]["Face"]
        for pi in f["planes"]:
            pl = g["planes"][pi]["Plane"]
            NW[(a["ident"], f["ident"], pl["ident"])] = len(pl["wires"])
            for idx, wi in enumerate(pl["wires"]):
                c2w[(a["ident"], g["wires"][wi]["Wire"]["channel"])].append((f["ident"], pl["ident"], idx))

print("=== 1. GEOMETRY (%s), AnodePlane order" % GEOM.rsplit("/", 1)[1])
print("  face idents per anode:", [[g["faces"][fi]["Face"]["ident"] for fi in a["Anode"]["faces"]] for a in g["anodes"]])
for apa in range(4):
    for pl in range(3):
        chs = [k for k, v in c2w.items() if k[0] == apa and v and v[0][1] == pl]
        mult = collections.Counter(len(set(x[0] for x in c2w[k])) for k in chs)
        first = collections.Counter(c2w[k][0][0] for k in chs)
        print(f"  apa {apa} plane {'UVW'[pl]}: channels {len(chs)}, distinct faces per channel {dict(mult)}, face of the FIRST listed wire {dict(first)}")

print("\n=== 2+3. RECORDED (face, wire) OF EVERY PDHD CELL vs THE GEOMETRY LIST")
conv = collections.Counter(); first = collections.Counter()
for fn in sorted(glob.glob(f"{IMG}/pdhd/work/*_h26q2dprod/tracking-pr.root")):
    u = uproot.open(fn)
    if "T_stm_michel_2d" not in u: continue
    C = u["T_stm_michel_2d"].arrays(["apa", "plane", "channel", "face", "wire", "role"], library="np")
    for apa, pl, ch, fa, wi, ro in zip(*(C[k].tolist() for k in ("apa", "plane", "channel", "face", "wire", "role"))):
        lst = [x for x in c2w[(apa, ch)] if x[1] == pl]
        idx = [x[2] for x in lst if x[0] == fa]; n = NW[(apa, fa, pl)]
        kind = "file index" if wi in idx else ("mirrored n-1-index" if wi in [n - 1 - i for i in idx] else "NEITHER")
        cls = "W" if pl == 2 else ("U/V role 1" if ro == 1 else "U/V other roles")
        conv[(fa, cls, kind)] += 1
        if pl < 2 and ro == 1:
            f0 = lst[0]; w0 = f0[2] if f0[0] == 0 else NW[(apa, f0[0], pl)] - 1 - f0[2]
            first[(apa, "FIRST entry" if (fa, wi) == (f0[0], w0) else "later entry")] += 1
for k in sorted(conv): print("  recorded face %d %-16s -> %-20s %d" % (k + (conv[k],)))
print("  U/V role-1 cells, recorded (face, wire) against the channel's first geometry entry, by apa:", dict(sorted(first.items())))

print("\n=== 4. SCALE, every candidate of the production arms")
for det, arm in ARMS:
    per = []; fa_ct = collections.Counter()
    for fn in sorted(glob.glob(f"{IMG}/{det}/work/*_{arm}/tracking-pr.root")):
        u = uproot.open(fn)
        if "T_stm_michel_2d" not in u: continue
        C = u["T_stm_michel_2d"].arrays(["cluster_id", "apa", "plane", "face", "role", "d_stop_cm"], library="np")
        T = u["T_stm_michel"].arrays(["cluster_id", "is_stm", "michel_found", "michel_q2d_ctl_n_u", "michel_q2d_ctl_n_v"], library="np")
        for i, c in enumerate(T["cluster_id"]):
            m = C["cluster_id"] == c; w1 = m & (C["plane"] == 2) & (C["role"] == 1)
            if not w1.any(): continue
            face = collections.Counter(C["face"][w1].tolist()).most_common(1)[0][0]
            apa = collections.Counter(C["apa"][w1].tolist()).most_common(1)[0][0]; fa_ct[(apa, face)] += 1
            uv = m & (C["plane"] < 2) & (C["role"] == 1)
            per.append(dict(sm=int(T["is_stm"][i]) == 1 and int(T["michel_found"][i]) == 1, apa=apa, face=face,
                            wrong=(uv & (C["face"] != face)).sum() / max(1, uv.sum()),
                            uv0=int(T["michel_q2d_ctl_n_u"][i]) == 0 and int(T["michel_q2d_ctl_n_v"][i]) == 0,
                            dW=np.median(C["d_stop_cm"][w1]),
                            dR=np.median(C["d_stop_cm"][uv & (C["face"] == face)]) if (uv & (C["face"] == face)).any() else np.nan,
                            dO=np.median(C["d_stop_cm"][uv & (C["face"] != face)]) if (uv & (C["face"] != face)).any() else np.nan))
    DUMP[det] = per
    wr = np.array([p["wrong"] for p in per])
    print(f"\n  [{det} {arm}] candidates with W role-1 cells {len(per)}; (apa, muon face): {dict(sorted(fa_ct.items()))}")
    print("    fraction of U/V role-1 cells recorded on another face, per candidate: =0 %d, (0,0.5) %d, [0.5,1) %d, =1 %d" % (
        (wr == 0).sum(), ((wr > 0) & (wr < 0.5)).sum(), ((wr >= 0.5) & (wr < 1)).sum(), (wr == 1).sum()))
    for lab, s in (("all", np.ones(len(per), bool)), ("is_stm & michel_found", np.array([p["sm"] for p in per]))):
        print(f"    [{lab}] n {s.sum()}: majority on another face {(wr[s] >= 0.5).sum()} ({(wr[s] >= 0.5).mean():.3f}); "
              f"control with no U and no V cell {sum(p['uv0'] for p, ss in zip(per, s) if ss)}")
    by = collections.defaultdict(list)
    for p in per: by[(p["apa"], p["face"])].append(p["wrong"])
    print("    median other-face fraction by (apa, muon face):", {k: round(float(np.median(v)), 2) for k, v in sorted(by.items())})
    print(f"    median d_stop of role-1 cells: W {np.nanmedian([p['dW'] for p in per]):.1f} cm | U/V on the muon's face "
          f"{np.nanmedian([p['dR'] for p in per]):.1f} | U/V on another face {np.nanmedian([p['dO'] for p in per]):.1f}")

print("\n=== 5. PER CELL (PDHD): the recorded wire against the channel's wires on its APA's ACTIVE face")
# the active face per APA is the face of its W cells (collection channels map to one face, sec 1)
wface = collections.defaultdict(collections.Counter)
for fn in sorted(glob.glob(f"{IMG}/pdhd/work/*_h26q2dprod/tracking-pr.root")):
    u = uproot.open(fn)
    if "T_stm_michel_2d" not in u: continue
    C = u["T_stm_michel_2d"].arrays(["apa", "plane", "face"], library="np")
    for apa, fa in zip(C["apa"][C["plane"] == 2].tolist(), C["face"][C["plane"] == 2].tolist()): wface[apa][fa] += 1
ACT = {apa: c.most_common(1)[0][0] for apa, c in wface.items()}
print("  W cells by (apa -> face counts):", {k: dict(v) for k, v in sorted(wface.items())}, "=> active face", ACT)
chain_idx = lambda apa, fa, pl, idx: idx if fa == 0 else NW[(apa, fa, pl)] - 1 - idx
mult = collections.Counter()
for (apa, ch), L in c2w.items():
    if L[0][1] == 2 or apa not in ACT: continue
    mult[(apa, "UVW"[L[0][1]], sum(1 for x in L if x[0] == ACT[apa]))] += 1
print("  U/V channels by the number of their wires on the active face (apa, plane, n):", dict(sorted(mult.items())))
BINS = np.logspace(-0.5, 2.7, 49); cls = collections.defaultdict(list)
for fn in sorted(glob.glob(f"{IMG}/pdhd/work/*_h26q2dprod/tracking-pr.root")):
    u = uproot.open(fn)
    if "T_stm_michel_2d" not in u: continue
    C = u["T_stm_michel_2d"].arrays(["apa", "plane", "channel", "face", "wire", "role", "d_stop_cm"], library="np")
    s = (C["plane"] < 2) & (C["role"] == 1)
    for apa, pl, ch, fa, wi, d in zip(*(C[k][s].tolist() for k in ("apa", "plane", "channel", "face", "wire", "d_stop_cm"))):
        act = [chain_idx(apa, x[0], pl, x[2]) for x in c2w[(apa, ch)] if x[0] == ACT[apa] and x[1] == pl]
        if fa != ACT[apa]: k = "other face (first-listed wire is on face 0)"
        elif len(act) == 1: k = "active face, channel has 1 active-face wire"
        else: k = "active face, 2 active-face wires, recorded the %s" % ("first" if wi == act[0] else "second")
        cls[k].append(d)
DUMP["percell_bins"] = BINS.tolist(); DUMP["percell"] = {}
for k in sorted(cls):
    v = np.array(cls[k]); DUMP["percell"][k] = np.histogram(np.clip(v, BINS[0], BINS[-1] * 0.999), BINS)[0].tolist()
    print(f"  {k:52s} n {len(v):6d}  d_stop p50 {np.median(v):6.1f}  p90 {np.percentile(v, 90):6.1f}  beyond 100 cm {(v > 100).mean():.3f}")
dv = collections.defaultdict(list)
for fn in sorted(glob.glob(f"{IMG}/pdvd/work/*_p96vprod/tracking-pr.root")):
    u = uproot.open(fn)
    if "T_stm_michel_2d" not in u: continue
    C = u["T_stm_michel_2d"].arrays(["plane", "role", "d_stop_cm"], library="np")
    for p in range(3): dv[p].extend(C["d_stop_cm"][(C["plane"] == p) & (C["role"] == 1)].tolist())
DUMP["percell_pdvd"] = {}
for p in range(3):
    v = np.array(dv[p]); DUMP["percell_pdvd"]["UVW"[p]] = np.histogram(np.clip(v, BINS[0], BINS[-1] * 0.999), BINS)[0].tolist()
    print(f"  PDVD plane {'UVW'[p]} role-1 cells n {len(v)}  d_stop p50 {np.median(v):.1f}  p90 {np.percentile(v, 90):.1f}  beyond 100 cm {(v > 100).mean():.4f}")
json.dump(DUMP, open(__file__.rsplit("/", 1)[0] + "/wrap_face.json", "w"), default=float)
