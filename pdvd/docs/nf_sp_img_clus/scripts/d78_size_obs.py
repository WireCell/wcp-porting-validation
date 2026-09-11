#!/usr/bin/env python3
"""doc pdvd/78 -- the owner's three observations on the PDVD STM / Michel chain,
sized on one bare-production arm against the smx1a+smx3+smx4 record (read-only):

  A  "STM end point not found properly": the record's pins (pin.moved_cm) and
     pin_rr entries (the scanner's "the stop belongs at rr = X cm along the fit"),
     the record's michel tags that the chain carries as the muon itself (role 1),
     and -- for every pin_rr item on which neither the retreat (doc 57/74) nor the
     split (doc 58) fired -- the production movers' own tests read offline at the
     scanner's pin: tail median / plateau (collapse), peak / plateau, nearest PR
     vertex to the pin, the fit's bend at the pin
  B  "Michel not identified in the close-to-ISO case (missing track trajectory)":
     every judged STM_MICHEL item with michel_found 0, and where its michel-tagged
     segment lives: a chain row (role), a PR segment of the main cluster with no
     chain row at all (fitted by PR / never fitted), a near cluster (bundle,
     association), or nowhere within the 60 cm image; the piece-admission gates
     (doc 62 T3b: dot radius, max length, body exclusion) read offline on the
     orphan segments
  C  "Michel or photon clustering ... not associated segments": where every michel
     and gamma tag of the record lands relative to the candidate (chain role /
     main-cluster orphan / same-bundle near cluster / other bundle / unassociated),
     the near-cluster census around every judged stopper, and P4's reach (doc 71:
     michel_gamma_collect) over the gamma tags it did not collect

Repro (doc 78 sec 0):
  python3 d78_size_obs.py [--prep /home/xqian/tmp/p75/prep_p75vprod] > /home/xqian/tmp/p78/size_obs.txt

Segment ids: the record's tag keys are PR segment ids (cluster_id * 1000 + n) or
"C<cluster_id>" for a whole-cluster tag on an unfitted piece.  A tag "has a row"
when the chain wrote it into T_stm_michel_pts with a role (prep's pf.chain_role);
pf.seg is T_rec_charge, the PR's own segment table for the cluster.
"""
import argparse, collections, glob, json, math
import numpy as np

IMG = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img"
ap = argparse.ArgumentParser()
ap.add_argument("--prep", default="/home/xqian/tmp/p75/prep_p75vprod")
ap.add_argument("--record", default=IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
args = ap.parse_args()

R = {v["key"]: v for v in json.load(open(args.record))}
P = {}
for fn in glob.glob(args.prep + "/smprep-*.json"):
    d = json.load(open(fn))
    P["%s/%d" % (d["event"], d["cluster_id"])] = d
print("prep %s: record %d items, payloads %d, both %d" % (args.prep, len(R), len(P), len(set(R) & set(P))))
STOP = {"STM_MICHEL", "STM_ONLY", "FRAG_STM_MICHEL", "FRAG_STM_ONLY"}
MICH = {"STM_MICHEL", "FRAG_STM_MICHEL"}

# production values of the movers' tests (pdvd/wct-pr-perevt.jsonnet + CheckSTM_Michel.cxx defaults)
COLLAPSE, PEAK, PEAK_WIN, SPLIT_KINK, VTX_R = 0.5, 1.4, 15.0, 15.0, 1.5
DOT_R, DOT_LEN, BODY_EXCL = 15.0, 25.0, 5.0
GAMMA_R_PROD, GAMMA_R_CPP, GAMMA_LEN = 50.0, 35.0, 10.0


def muon_sorted(p):
    m = p["muon"]; o = np.argsort(m["rr"])
    return {k: np.asarray(m[k], float)[o] for k in ("x", "y", "z", "q", "rr")}


def role_of(p, sid):
    return p["pf"]["chain_role"].get(str(sid))


def where(p, sid):
    """(description, near_cluster row or None) for a record tag key."""
    r = role_of(p, sid)
    if r is not None:
        return ("row role %d" % r, None)
    whole = str(sid).startswith("C")
    cid = int(str(sid)[1:]) if whole else int(sid) // 1000
    if cid == p["cluster_id"]:
        infit = (not whole) and any(int(sg["id"]) == int(sid) for sg in p["pf"]["seg"])
        return ("main cluster, no row, %s" % ("PR-fitted" if infit else "never fitted"), None)
    pre = "WHOLE-cluster tag: " if whole else ""
    for nc in p["near_clusters"]:
        if nc["id"] == cid:
            return (pre + "near cluster %.0f cm, %s, %s" % (
                nc["d_stop"], "in bundle" if nc["in_bundle"] else "OTHER bundle",
                "assoc" if nc["is_associated"] else "UNASSOC"), nc)
    return (pre + "cluster not within 60 cm image", None)


def michel_tags(v):
    return [t for t, g in (v.get("tags") or {}).items() if g == "michel"]


# ============================ A. the stop ============================
print("\n=== A1. pins the scanner placed (pin.moved_cm = distance the stop was moved), items with a payload")
pins = [(k, R[k]) for k in R if R[k].get("pin") and R[k]["pin"].get("placed") and k in P]
bins = collections.Counter(); byv = collections.defaultdict(collections.Counter)
for k, v in pins:
    mv = v["pin"].get("moved_cm") or 0.0
    b = "<1" if mv < 1 else "1-3" if mv < 3 else "3-10" if mv < 10 else ">=10"
    bins[b] += 1; byv[v["verdict"]][b] += 1
print("  %d pins: moved_cm %s" % (len(pins), dict(bins)))
for vv in byv:
    print("    %-12s %s" % (vv, dict(byv[vv])))
for mv, k, v in sorted(((v["pin"].get("moved_cm") or 0.0, k, v) for k, v in pins), reverse=True):
    if mv < 3: continue
    p = P[k]; vd = p["verdict"]
    print("    %-14s %-11s moved %5.1f cm pin_rr %5s is_stm %d michel_found %d bits %4d kink %4d/%4d | michel tag roles %s" % (
        k, v["verdict"], mv, v["pin"].get("rr"), vd["is_stm"], vd["michel_found"], vd["reject_bits"], vd["kink_num"], p["npts"],
        ",".join(str(role_of(p, t)) for t in michel_tags(v)) or "-"))

print("\n=== A2. pin_rr items (the scanner: 'the real stop is rr = X cm back along the fit'), with the production movers' state")
prr = sorted((k for k in R if R[k].get("pin_rr") is not None and k in P), key=lambda k: -R[k]["pin_rr"])
moved = [k for k in prr if P[k]["verdict"]["n_retreat"] or P[k]["verdict"]["n_split"]]
print("  %d items; production moved the stop on %d of them (retreat/split fired): %s" % (
    len(prr), len(moved), " ".join("%s(retreat %.1f, pin_rr %.1f)" % (k, P[k]["verdict"]["retreat_len"] + P[k]["verdict"]["split_len"], R[k]["pin_rr"]) for k in moved)))
print("  the %d on which no mover fired, is_stm 0: %d, michel_found 0: %d" % (
    len(prr) - len(moved), sum(1 for k in prr if k not in moved and not P[k]["verdict"]["is_stm"]),
    sum(1 for k in prr if k not in moved and not P[k]["verdict"]["michel_found"])))
for k in prr:
    v = R[k]; vd = P[k]["verdict"]
    print("    %-14s %-15s pin_rr %5.1f is_stm %d michel_found %d bits %4d kink %4d/%4d retreat %d/%.1f split %d/%.1f | michel roles %s | %s" % (
        k, v["verdict"], v["pin_rr"], vd["is_stm"], vd["michel_found"], vd["reject_bits"], vd["kink_num"], P[k]["npts"],
        vd["n_retreat"], vd["retreat_len"], vd["n_split"], vd["split_len"],
        ",".join(str(role_of(P[k], t)) for t in michel_tags(v)) or "-", (v.get("notes") or "")[:70]))

print("\n=== A3. the production movers' tests read offline at the scanner's pin, on the pin_rr items where no mover fired")
print("   collapse: tail median (rows past the pin) < %.1f x plateau (median over rr in [pin+5, pin+40]); peak: 3-row mean >= %.1f x plateau in [pin-1, pin+%.0f];" % (COLLAPSE, PEAK, PEAK_WIN))
print("   retreat needs a PR vertex within %.1f cm of the pin point; split needs bend >= %.0f deg (5 cm windows either side of the pin)" % (VTX_R, SPLIT_KINK))
grp = collections.Counter(); hot_bend = []
for k in prr:
    if k in moved: continue
    p = P[k]; vd = p["verdict"]; m = muon_sorted(p); pr = R[k]["pin_rr"]
    tail = m["q"][m["rr"] < pr]; plat = m["q"][(m["rr"] > pr + 5) & (m["rr"] < pr + 40)]
    win = m["q"][(m["rr"] >= pr - 1) & (m["rr"] < pr + PEAK_WIN)]
    plateau = float(np.median(plat)) if plat.size else float("nan")
    tmed = float(np.median(tail)) / plateau if tail.size else float("nan")
    pk = max((float(np.mean(win[i:i + 3])) for i in range(max(1, win.size - 2))), default=float("nan")) / plateau if win.size else float("nan")
    i = int(np.argmin(np.abs(m["rr"] - pr))); P3 = np.c_[m["x"], m["y"], m["z"]]; pt = P3[i]
    vx = p["pf"]["vtx"]; V = np.c_[vx["x"], vx["y"], vx["z"]] if vx["x"] else np.zeros((0, 3))
    dv = float(np.min(np.linalg.norm(V - pt, axis=1))) if len(V) else -1.0
    a = P3[(m["rr"] > pr) & (m["rr"] < pr + 5)]; b = P3[(m["rr"] < pr) & (m["rr"] > pr - 5)]
    bend = float("nan")
    if len(a) > 1 and len(b) > 1:
        da = a[0] - a[-1]; db = b[0] - b[-1]
        c = float(np.dot(da, db) / np.linalg.norm(da) / np.linalg.norm(db)); bend = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    collapse, peak, vtx, kink = tmed < COLLAPSE, pk >= PEAK, 0 <= dv <= VTX_R, bend >= SPLIT_KINK
    g = ("retreat's tests pass offline" if (collapse and peak and vtx) else "split's tests pass offline" if (collapse and peak and kink)
         else "tail NOT collapsed" if not collapse else "no peak" if not peak else "collapsed, no vertex, bend < %.0f" % SPLIT_KINK)
    grp[g] += 1
    if g == "tail NOT collapsed": hot_bend.append(bend)
    print("    %-14s pin_rr %5.1f rows past pin %3d | tail/plateau %.2f peak/plateau %.2f | PR vtx %4.1f cm bend %5.1f deg | is_stm %d mf %d bits %3d -> %s" % (
        k, pr, tail.size, tmed, pk, dv, bend, vd["is_stm"], vd["michel_found"], vd["reject_bits"], g))
print("  groups: %s" % dict(grp))
if hot_bend:
    print("  'tail NOT collapsed' items with bend >= 14 deg: %d of %d (bends: %s)" % (sum(1 for b in hot_bend if b >= 14), len(hot_bend), " ".join("%.0f" % b for b in sorted(hot_bend))))

print("\n=== A4. record michel tags the chain carries as the muon itself (chain role 1), judged STM_MICHEL items")
fused = sorted({(k, t) for k in R if R[k]["verdict"] in MICH and k in P for t in michel_tags(R[k]) if role_of(P[k], t) == 1})
print("  %d tags on %d items" % (len(fused), len({k for k, _ in fused})))
for k, t in fused:
    vd = P[k]["verdict"]
    print("    %-14s seg %-7s is_stm %d michel_found %d bits %4d n_chain_segs %d pin_rr %s" % (k, t, vd["is_stm"], vd["michel_found"], vd["reject_bits"], vd["n_chain_segs"], R[k].get("pin_rr")))

# ============================ B. the Michel ============================
print("\n=== B1. judged STM_MICHEL items with michel_found 0: where the michel-tagged segment lives")
rows = []; cls = collections.Counter(); kinds = collections.Counter()
for k in sorted(R):
    v = R[k]
    if v["verdict"] not in MICH or k not in P: continue
    p = P[k]; vd = p["verdict"]
    if vd["michel_found"]: continue
    kinds[v["michel_kind"]] += 1
    mt = michel_tags(v)
    other = ["%s=%s@%s" % (t, g[:5], where(p, t)[0]) for t, g in (v.get("tags") or {}).items() if g == "gamma" or str(t).startswith("C")]
    if not mt:
        cls["no michel tag on the record"] += 1
        rows.append((k, v, "no michel tag; other tags: %s" % ("; ".join(other) or "-"))); continue
    ws = [where(p, t)[0] for t in mt]
    best = next((w for w in ws if w.startswith("row")), None) or next((w for w in ws if "no row" in w), None) or ws[0]
    cls[best.split(" %.0f" % 0)[0] if False else (best if best.startswith("row") else best.split(",")[0] + ("," + best.split(",")[-1] if "main" in best else ""))] += 1
    rows.append((k, v, " ; ".join(ws)))
print("  %d items; michel_kind %s" % (len(rows), dict(kinds)))
print("  classes: %s" % dict(cls))
for k, v, desc in rows:
    vd = P[k]["verdict"]
    print("    %-15s is_stm %d bits %4d conf %-6s kind %-14s pin_rr %-5s michel piece %d segs %.1f cm %.1f MeV dis %.1f | %s" % (
        k, vd["is_stm"], vd["reject_bits"], v["confidence"], v["michel_kind"], v.get("pin_rr") or "-",
        vd["n_michel_segs"], vd["michel_len"], vd["michel_ke_best"], vd["michel_dis_cm"], desc))

print("\n=== B2. orphan michel segments: record 'michel' tag on a PR segment of the MAIN cluster with no chain row (all judged stoppers)")
orph = []
for k in sorted(R):
    v = R[k]
    if v["verdict"] not in STOP or k not in P: continue
    p = P[k]; vd = p["verdict"]; m = muon_sorted(p); M = np.c_[m["x"], m["y"], m["z"]]; stop = M[0]
    for t in michel_tags(v):
        if str(t).startswith("C") or role_of(p, t) is not None or int(t) // 1000 != p["cluster_id"]: continue
        sg = next((s for s in p["pf"]["seg"] if int(s["id"]) == int(t)), None)
        if sg is None:
            orph.append((k, t, vd, None)); continue
        S = np.c_[sg["x"], sg["y"], sg["z"]]
        d_stop = float(np.min(np.linalg.norm(S - stop, axis=1)))
        body = M[m["rr"] > 5]
        d_body = float(np.min(np.linalg.norm(S[:, None, :] - body[None, :, :], axis=2))) if len(body) else -1.0
        end = M[m["rr"] < 5]; cosang = float("nan")
        if len(end) > 1 and len(S) > 1:
            de = end[0] - end[-1]
            ds = S[np.argmax(np.linalg.norm(S - stop, axis=1))] - S[np.argmin(np.linalg.norm(S - stop, axis=1))]
            cosang = float(np.dot(de, ds) / np.linalg.norm(de) / np.linalg.norm(ds))
        orph.append((k, t, vd, dict(len=sg["len_cm"], dq=sg["dqdx_med"], n=sg["npts"], d_stop=d_stop, d_body=d_body, cos=cosang)))
gapb = collections.Counter()
for k, t, vd, g in orph:
    gapb["never fitted" if g is None else "touches the stop" if g["d_stop"] <= 0.5 else "gap 0.5-3" if g["d_stop"] <= 3 else "gap 3-10" if g["d_stop"] <= 10 else "gap > 10"] += 1
print("  %d tags on %d items; on michel_found-0 items %d tags / %d items; nearest point to the stop: %s" % (
    len(orph), len({o[0] for o in orph}), sum(1 for o in orph if not o[2]["michel_found"]), len({o[0] for o in orph if not o[2]["michel_found"]}), dict(gapb)))
print("  T3b's gates (doc 62): michel_dot_radius_cm %.0f, dot_max_len_cm %.0f, dot_body_exclusion_cm %.0f; 'body-excluded?' = within %.0f cm of the muon body (rr > 5) and not touching the stop" % (DOT_R, DOT_LEN, BODY_EXCL, BODY_EXCL))
for k, t, vd, g in sorted(orph, key=lambda o: (o[2]["michel_found"], o[0])):
    if g is None:
        print("    %-14s seg %-7s mf %d is_stm %d | never fitted by PR" % (k, t, vd["michel_found"], vd["is_stm"])); continue
    tag = ("touches the stop" if g["d_stop"] <= 0.5 else "body-excluded?" if g["d_body"] < BODY_EXCL else "free piece")
    print("    %-14s seg %-7s mf %d is_stm %d bits %3d | len %5.1f cm dqdx_med %6.0f npts %3d | d_stop %4.1f d_body %4.1f cos(muon end, seg) %+.2f | %s | kind %s conf %s" % (
        k, t, vd["michel_found"], vd["is_stm"], vd["reject_bits"], g["len"], g["dq"], g["n"], g["d_stop"], g["d_body"], g["cos"], tag, R[k]["michel_kind"], R[k]["confidence"]))

print("\n=== B3. how the chain reads the record's 'detached dots' / 'both' Michels (judged STM_MICHEL)")
c2 = collections.Counter()
for k in R:
    v = R[k]
    if v["verdict"] in MICH and k in P and v["michel_kind"] in ("detached dots", "both"):
        vd = P[k]["verdict"]
        c2[(v["michel_kind"], ("found conn %d" % vd["michel_conn_type"]) if vd["michel_found"] else "NOT found", "is_stm %d" % vd["is_stm"])] += 1
for kk in sorted(c2):
    print("    %-14s %-12s %-8s %d" % (kk + (c2[kk],)))

# ============================ C. unassociated segments ============================
print("\n=== C1. where every michel / gamma tag of the record lands, relative to the candidate (all judged stoppers)")
land = collections.defaultdict(collections.Counter); unassoc_items = collections.defaultdict(set)
for k in R:
    v = R[k]
    if v["verdict"] not in STOP or k not in P: continue
    for t, g in (v.get("tags") or {}).items():
        if g not in ("michel", "gamma"): continue
        w, nc = where(P[k], t)
        key = w if (w.startswith("row") or "image" in w or "main" in w) else ("WHOLE " if "WHOLE" in w else "") + "near cluster: " + w.split(", ", 1)[1]
        land[g][key] += 1
        if nc and not nc["is_associated"]: unassoc_items[g].add(k)
for g in ("michel", "gamma"):
    print("  %s tags (%d): %s" % (g, sum(land[g].values()), dict(sorted(land[g].items(), key=lambda kv: -kv[1]))))
    print("    items with a %s tag in an UNASSOCIATED cluster: %d" % (g, len(unassoc_items[g])))
wc = collections.Counter()
for k in R:
    v = R[k]
    if v["verdict"] not in STOP or k not in P: continue
    for t, g in (v.get("tags") or {}).items():
        if str(t).startswith("C"):
            nc = next((n for n in P[k]["near_clusters"] if n["id"] == int(t[1:])), None)
            wc[(g, "in bundle" if nc and nc["in_bundle"] else "other/none", "assoc" if nc and nc["is_associated"] else "unassoc")] += 1
print("  whole-cluster tags (unfitted pieces) on judged stoppers, (tag, bundle, association): %s" % dict(wc))

print("\n=== C2. near-cluster census around every judged stopper (clusters with an image point within 20 / 60 cm of the stop)")
cnt = collections.Counter(); items_with = collections.Counter()
for k in R:
    v = R[k]
    if v["verdict"] not in STOP or k not in P: continue
    p = P[k]; seen = set()
    for nc in p["near_clusters"]:
        if nc["id"] == p["cluster_id"]: continue
        kk = ("<=20" if nc["d_stop"] <= 20 else "20-60", "in bundle" if nc["in_bundle"] else "other bundle",
              "assoc" if nc["is_associated"] else "unassoc", "PR-fitted" if nc["segs"] else "NO segs")
        cnt[kk] += 1; seen.add(kk)
    for kk in seen: items_with[kk] += 1
for kk in sorted(cnt):
    print("    %-6s %-13s %-8s %-10s clusters %4d on %3d items" % (kk + (cnt[kk], items_with[kk])))

print("\n=== C3. gamma tags in same-bundle near clusters with no chain role: P4's reach (michel_gamma_collect; radius %.0f in production, C++ %.0f; max len %.0f cm; cone cos 0.5)" % (GAMMA_R_PROD, GAMMA_R_CPP, GAMMA_LEN))
grows = []
for k in R:
    v = R[k]
    if v["verdict"] not in STOP or k not in P: continue
    p = P[k]
    for t, g in (v.get("tags") or {}).items():
        if g != "gamma" or str(t).startswith("C") or role_of(p, t) is not None: continue
        cid = int(t) // 1000
        nc = next((n for n in p["near_clusters"] if n["id"] == cid), None)
        if nc is None or cid == p["cluster_id"]: continue
        grows.append((k, t, nc["d_stop"], nc["length_cm"] or 0.0, p["verdict"]["n_stop_gammas"], p["verdict"]["michel_found"]))
d = collections.Counter()
for r in grows:
    d[("d<=%.0f" % GAMMA_R_CPP if r[2] <= GAMMA_R_CPP else "%.0f-%.0f" % (GAMMA_R_CPP, GAMMA_R_PROD) if r[2] <= GAMMA_R_PROD else ">%.0f" % GAMMA_R_PROD,
       "len<=%.0f" % GAMMA_LEN if r[3] <= GAMMA_LEN else "len>%.0f" % GAMMA_LEN)] += 1
print("  %d tags on %d items; items with any stop gamma already: %d; items with michel_found 0: %d" % (
    len(grows), len({r[0] for r in grows}), len({r[0] for r in grows if r[4] > 0}), len({r[0] for r in grows if not r[5]})))
for kk in sorted(d):
    print("    %-6s %-7s %d" % (kk + (d[kk],)))
if grows:
    print("  cluster length of the tagged gamma's cluster (cm): p50 %.1f p90 %.1f max %.1f" % tuple(np.percentile([r[3] for r in grows], [50, 90, 100])))
print("DONE")
