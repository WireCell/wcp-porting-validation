#!/usr/bin/env python3
"""doc pdvd/85 (doc 78 action item 5) -- the gamma collect's rejections, and the
stop-anchored collect, on one PDVD arm against the owner's merged scan record.
Read-only.

  A  P4 (doc 71, michel_gamma_collect): every companion cluster the gate saw,
     from the C++'s own DEBUG lines (michel-gamma-cl: ... gate N and
     michel-gamma-take: ... take T live L), by outcome x distance band x the
     record's verdict x the cluster's tag; the cone rejects listed by cos.
  B  role 4 (P4's blobs) and role 5 (the doc 51 capture gamma) CLUSTERS on
     judged items, split by the candidate's is_stm, with the role-5 clusters on
     is_stm-0 candidates named.
  C  the capture stage's offline twin (ring michel_dot_radius < d <= R, cluster
     length <= 10 cm, the cluster-level body test against the muon, delta and
     Michel rows beyond dot_body_exclusion_cm of the stop), validated against the
     arm's own role 5 inside 15-35 cm, then the ring widened to 35-50 cm on the
     candidates where P4 never ran (no michel-gamma: header), by is_stm.
  D  segment-id drift: the record's gamma tags whose segment id is not the arm's
     although the cluster carries a chain role; doc 78 sec 4.3's "76" re-derived
     at cluster level.
  E  (with --base) what moved between two arms: role-5 / role-4 clusters added
     or removed, by is_stm and tag, named; and the role-5 purity before / after.

EVERY tag join is at CLUSTER level (cluster id = segment id // 1000, or a
"C<id>" whole-cluster tag): P4 and the capture stage decide per cluster, and the
scan arms' PR graph indices differ from production's by one on some clusters, so
a segment-level join reads a collected cluster as "no role".  A cluster's tag is
gamma if any of its tags is gamma, else michel, else the first tag, else
"untagged"; "good" = gamma or michel.

Repro (doc 85 sec 0):
  python3 d85_gamma_census.py --arm p84vr65 --prep /home/xqian/tmp/p84/prep_p84vr65
  python3 d85_gamma_census.py --arm p85vwh  --prep /home/xqian/tmp/p85/prep_p85vwh \\
          --base p85voff --base-prep /home/xqian/tmp/p85/prep_p85voff
"""
import argparse, collections, glob, json, re
import numpy as np

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--prep", required=True)
ap.add_argument("--base")
ap.add_argument("--base-prep")
ap.add_argument("--record", default=IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
ap.add_argument("--sections", default="ABCDE")
args = ap.parse_args()

R = {v["key"]: v for v in json.load(open(args.record))}
JUDGED = {"STM_MICHEL", "STM_ONLY", "THRU", "MESSY", "FRAG_STM_MICHEL", "FRAG_STM_ONLY", "FRAG_THRU"}
STOP = {"STM_MICHEL", "STM_ONLY", "FRAG_STM_MICHEL", "FRAG_STM_ONLY"}
DOT_R, BODY_EXCL, GLEN = 15.0, 5.0, 10.0      # michel_dot_radius_cm, dot_body_exclusion_cm, stop_gamma_max_len_cm (C++ defaults; PDVD sets none)
RE_CL = re.compile(r"michel-gamma-cl: cluster (\d+) comp (\d+) d_stop ([\d.]+) len ([\d.]+) cos ([-\d.]+) "
                   r"d_mich ([\d.]+) d_body ([\d.e+]+) ke ([\d.]+) gate (\d+)")
RE_TK = re.compile(r"michel-gamma-take: cluster (\d+) comp (\d+) ke ([\d.]+) take (\d) live (\d)")
RE_HD = re.compile(r"michel-gamma: cluster (\d+) conn (\d)")


def load_prep(d):
    P = {}
    for fn in glob.glob(d + "/smprep-*.json"):
        p = json.load(open(fn)); P["%s/%d" % (p["event"], p["cluster_id"])] = p
    return P


def load_log(arm):
    G, T, H = {}, {}, set()
    for d in glob.glob(IMG + "/pdvd/work/*_" + arm):
        ev = d.split("/")[-1][: -len(arm) - 1]
        for fn in glob.glob(d + "/wct_pr_*.log"):
            for line in open(fn, errors="replace"):
                m = RE_CL.search(line)
                if m:
                    G[(ev, int(m[1]), int(m[2]))] = dict(d_stop=float(m[3]), len=float(m[4]), cos=float(m[5]), d_mich=float(m[6]),
                                                        d_body=float(m[7]), ke=float(m[8]), gate=int(m[9]))
                    continue
                m = RE_TK.search(line)
                if m:
                    T[(ev, int(m[1]), int(m[2]))] = (int(m[4]), int(m[5])); continue
                m = RE_HD.search(line)
                if m: H.add("%s/%s" % (ev, m[1]))
    return G, T, H


def cl_tag(v, cid):
    if v is None: return "unjudged"
    gs = [g for t, g in (v.get("tags") or {}).items()
          if str(t) == "C%d" % cid or (not str(t).startswith("C") and int(t) // 1000 == cid)]
    if not gs: return "untagged"
    for pref in ("gamma", "michel"):
        if pref in gs: return pref
    return gs[0]


def good(t): return "good" if t in ("gamma", "michel") else t


def role_clusters(p, key):
    return sorted({int(s) // 1000 for s in (p["verdict"][key]["seg"] or [])})


def band(d): return "<=35" if d <= 35 else "35-50" if d <= 50 else ">50"


P = load_prep(args.prep)
G, T, H = load_log(args.arm)
print("arm %s: %d payloads, %d on the record (%d judged); P4 log: %d gate lines, %d take lines, %d candidates with a header" % (
    args.arm, len(P), sum(1 for k in P if k in R), sum(1 for k in P if k in R and R[k]["verdict"] in JUDGED), len(G), len(T), len(H)))

# ============================ A ============================
if "A" in args.sections:
    print("\n=== A. P4's gate on every companion cluster it saw (C++ DEBUG lines), judged items, cluster-level tags")
    print("    gate: 0 pass, 1 radius, 2 length, 3 cone (cos < 0.5), 4 body (d_body <= d_mich), 5 blob energy, 6 non-finite")
    tab = collections.Counter(); per_gate = collections.defaultdict(collections.Counter)
    for (ev, c, cid), e in G.items():
        k = "%s/%d" % (ev, c); v = R.get(k)
        if v is None or v["verdict"] not in JUDGED: continue
        tk = T.get((ev, c, cid))
        out = "gate%d" % e["gate"]
        if e["gate"] == 0: out += " taken" if tk and tk[0] else " not-taken(live%s)" % (tk[1] if tk else "?")
        t = cl_tag(v, cid)
        tab[(out, band(e["d_stop"]), v["verdict"], t)] += 1
        if v["verdict"] in ("STM_MICHEL", "FRAG_STM_MICHEL"): per_gate[out][good(t)] += 1
    for kk in sorted(tab): print("    %-24s %-6s %-16s %-14s %3d" % (kk + (tab[kk],)))
    print("  on STM_MICHEL items, by outcome (good = gamma/michel):")
    for o in sorted(per_gate): print("    %-24s %s" % (o, dict(sorted(per_gate[o].items()))))
    print("  the cone's rejects on judged items, by cos band:")
    cb = collections.Counter()
    for (ev, c, cid), e in G.items():
        v = R.get("%s/%d" % (ev, c))
        if v is None or v["verdict"] not in JUDGED or e["gate"] != 3: continue
        cb[("cos>=0.3" if e["cos"] >= 0.3 else "0<=cos<0.3" if e["cos"] >= 0 else "cos<0", good(cl_tag(v, cid)))] += 1
    for kk in sorted(cb): print("    %-11s %-14s %3d" % (kk + (cb[kk],)))

# ============================ B ============================
def purity_table(P, label, name_is_stm0=True):
    c = collections.Counter(); named = []
    for k, p in sorted(P.items()):
        v = R.get(k)
        if v is not None and v["verdict"] not in JUDGED: continue
        for role, key in ((4, "dots"), (5, "gamma")):
            for cid in role_clusters(p, key):
                t = cl_tag(v, cid)
                c[(role, p["verdict"]["is_stm"], good(t))] += 1
                if role == 5 and not p["verdict"]["is_stm"]: named.append("%s c%d:%s(%s)" % (k, cid, t, v["verdict"] if v else "-"))
    print("  %s:" % label)
    for role in (4, 5):
        for s in (1, 0):
            row = {t: n for (r, ss, t), n in c.items() if r == role and ss == s}
            ng, nb = row.get("good", 0), row.get("delta / other", 0) + row.get("muon", 0)
            print("    role %d is_stm %d: %s  purity %s" % (role, s, dict(sorted(row.items())), "%.3f" % (ng / (ng + nb)) if ng + nb else "-"))
        tot = collections.Counter()
        for (r, ss, t), n in c.items():
            if r == role: tot[t] += n
        ng, nb = tot.get("good", 0), tot.get("delta / other", 0) + tot.get("muon", 0)
        print("    role %d all      : %s  purity %s" % (role, dict(sorted(tot.items())), "%.3f" % (ng / (ng + nb)) if ng + nb else "-"))
    if name_is_stm0 and named:
        print("    role-5 clusters on is_stm-0 candidates (%d): %s" % (len(named), " ".join(named)))
    return c


if "B" in args.sections:
    print("\n=== B. role-4 / role-5 clusters on judged items by the candidate's is_stm (purity = good / (good + delta/other + muon))")
    purity_table(P, args.arm)

# ============================ C ============================
def twin(p):
    """yield (cid, d_cl, len, body_ok) for every fitted same-bundle companion in the image."""
    vd = p["verdict"]; stop = np.array([vd["stop_x"], vd["stop_y"], vd["stop_z"]])
    B = [np.c_[p["muon"]["x"], p["muon"]["y"], p["muon"]["z"]]]
    for r in ("delta", "michel"):
        if vd[r]["x"]: B.append(np.c_[vd[r]["x"], vd[r]["y"], vd[r]["z"]])
    B = np.vstack(B); B = B[np.linalg.norm(B - stop, axis=1) >= BODY_EXCL]
    pts = collections.defaultdict(list)
    for im in ("image_near", "image_far"):
        I = p[im]
        for x, y, z, c in zip(I["x"], I["y"], I["z"], I["c"]): pts[c].append((x, y, z))
    for nc in p["near_clusters"]:
        cid = nc["id"]
        if cid == p["cluster_id"] or not nc["in_bundle"] or not nc["segs"] or cid not in pts: continue
        C = np.array(pts[cid]); dcl = float(np.min(np.linalg.norm(C - stop, axis=1)))
        dbody = float(np.min(np.linalg.norm(C[:, None, :] - B[None, :, :], axis=2))) if len(B) else 1e9
        yield cid, dcl, nc["length_cm"] or 0.0, dbody >= dcl


if "C" in args.sections:
    print("\n=== C. the capture stage (doc 51, role 5) -- offline twin; image points of each fitted same-bundle companion")
    val = collections.Counter(); ext = collections.Counter(); ext_named = collections.defaultdict(list)
    for k, p in sorted(P.items()):
        v = R.get(k); r5 = set(role_clusters(p, "gamma"))
        for cid, dcl, L, body_ok in twin(p):
            if DOT_R < dcl <= 35:
                val[("twin accepts" if (L <= GLEN and body_ok) else "twin rejects", "role 5" if cid in r5 else "no role 5")] += 1
            if 35 < dcl <= 50 and L <= GLEN and body_ok and k not in H and v is not None and v["verdict"] in JUDGED:
                t = good(cl_tag(v, cid)); ext[(p["verdict"]["is_stm"], t)] += 1
                ext_named[(p["verdict"]["is_stm"], t)].append("%s c%d(%s)" % (k, cid, v["verdict"]))
    print("  validation inside 15-35 cm (all candidates): %s" % dict(sorted(val.items())))
    print("  the ring widened to 35-50 cm, candidates where P4 never ran, judged items, twin-accepted clusters:")
    for kk in sorted(ext):
        print("    is_stm %d %-14s %3d  %s" % (kk + (ext[kk], " ".join(ext_named[kk]))))
    for s in (1, 0, None):
        ng = sum(n for (ss, t), n in ext.items() if t == "good" and (s is None or ss == s))
        nb = sum(n for (ss, t), n in ext.items() if t in ("delta / other", "muon") and (s is None or ss == s))
        print("    %s: good %d, delta/other %d, purity %s" % ("all" if s is None else "is_stm %d" % s, ng, nb, "%.3f" % (ng / (ng + nb)) if ng + nb else "-"))

# ============================ D ============================
if "D" in args.sections:
    print("\n=== D. segment-id drift, and doc 78 sec 4.3 at cluster level (judged stoppers, same-bundle near clusters)")
    dr = collections.Counter(); rows = []
    for k, v in R.items():
        if v["verdict"] not in STOP or k not in P: continue
        p = P[k]; cr = p["pf"]["chain_role"]
        for t, g in (v.get("tags") or {}).items():
            if g != "gamma" or str(t).startswith("C"): continue
            cid = int(t) // 1000
            if cid == p["cluster_id"]: continue
            nc = next((n for n in p["near_clusters"] if n["id"] == cid), None)
            seg_role = cr.get(str(t))
            cl_roles = sorted({r for s, r in cr.items() if not s.startswith("C") and int(s) // 1000 == cid})
            if seg_role is not None: dr["the tagged segment has a row"] += 1
            elif cl_roles: dr["segment has no row, its CLUSTER has role(s) %s" % cl_roles] += 1
            else: dr["no row in the cluster (segment %s the arm's)" % ("IS" if nc and int(t) in nc["segs"] else "is NOT")] += 1
            if nc is None or not nc["in_bundle"] or cl_roles: continue
            rows.append((k, cid, nc["d_stop"], nc["length_cm"] or 0.0, p["verdict"]["michel_found"], v["verdict"]))
    for kk, n in dr.most_common(): print("    %-62s %3d" % (kk, n))
    ntag = sum(1 for r in rows if r[2] <= 35 and r[3] <= GLEN)
    sub = {(r[0], r[1]): r for r in rows if r[2] <= 35 and r[3] <= GLEN}
    items = {k for k, _ in sub}
    wm = [kc for kc, r in sub.items() if r[4]]; nm = [kc for kc, r in sub.items() if not r[4]]
    print("  gamma tags in same-bundle clusters with NO chain role anywhere in the cluster, <= 35 cm and <= 10 cm long: %d tags, %d clusters on %d items" % (ntag, len(sub), len(items)))
    print("    on items with a Michel object (michel_found 1): %d on %d items; without: %d on %d items (record verdicts %s)" % (
        len(wm), len({k for k, _ in wm}), len(nm), len({k for k, _ in nm}), dict(collections.Counter(sub[kc][5] for kc in nm))))

    def why(k, cid):
        ev, c = k.split("/"); e = G.get((ev, int(c), cid))
        if e is not None:
            if e["gate"]: return "gate %d" % e["gate"]
            tk = T.get((ev, int(c), cid))
            return "gate 0, not taken (%s)" % ("Michel vetoed after phase 1" if tk and not tk[1] else "total-energy guard")
        if k not in H: return "P4 never ran (no attached/bridged Michel at phase 1)"
        return "no gate line (the Michel's own cluster, or claimed by another stage)"
    print("  what excluded them -- the C++'s own line for each cluster:")
    for lab, grp in (("with a Michel", wm), ("without", nm)):
        c = collections.Counter(why(k, cid) for k, cid in grp)
        print("    %-13s %s" % (lab, dict(c.most_common())))
    for k, cid in sorted(wm):
        r = sub[(k, cid)]
        print("      %-15s c%-4d %5.1f cm %4.1f cm long  %-11s %s" % (k, cid, r[2], r[3], r[5], why(k, cid)))

# ============================ E ============================
if "E" in args.sections and args.base:
    PB = load_prep(args.base_prep)
    print("\n=== E. %s vs %s: role-4 / role-5 clusters moved (all payloads; judged tags where the record has the item)" % (args.arm, args.base))
    for role, key in ((5, "gamma"), (4, "dots")):
        add = collections.Counter(); rem = collections.Counter(); na, nr = [], []
        for k in sorted(set(P) | set(PB)):
            a = set(role_clusters(P[k], key)) if k in P else set()
            b = set(role_clusters(PB[k], key)) if k in PB else set()
            v = R.get(k); s = (P.get(k) or PB.get(k))["verdict"]["is_stm"]; hdr = "P4" if k in H else "noP4"
            for cid in sorted(a - b):
                t = cl_tag(v, cid); add[(s, hdr, good(t))] += 1; na.append("%s c%d:%s" % (k, cid, t))
            for cid in sorted(b - a):
                t = cl_tag(v, cid); rem[(s, hdr, good(t))] += 1; nr.append("%s c%d:%s" % (k, cid, t))
        print("  role %d ADDED   %d: %s" % (role, sum(add.values()), dict(sorted(add.items()))))
        if na: print("      %s" % " ".join(na))
        print("  role %d REMOVED %d: %s" % (role, sum(rem.values()), dict(sorted(rem.items()))))
        if nr: print("      %s" % " ".join(nr))
    print("  purity, base then arm:")
    purity_table(PB, args.base, name_is_stm0=False)
    purity_table(P, args.arm, name_is_stm0=False)
print("DONE")
