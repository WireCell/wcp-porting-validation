#!/usr/bin/env python3
"""doc pdvd/71 -- size P4 (michel_gamma_collect) BEFORE it is built.

Read-only.  The pool is a SURVEY arm (default prep_d71vsp: survey_enable, 60 cm
admission), because only there does every unclaimed companion fragment near the
stop carry a point row (role 6); the capture-gamma blobs (role 5) are included
for reference.  For every candidate with a fitted Michel (michel_conn_type 1 or
2) each unclaimed companion CLUSTER is measured the way the C++ phase 1 measures
it -- d_stop (closest point to the final stop), the stop -> blob-centroid vs
stop -> Michel-object-centroid cosine, the distance to the Michel object and to
the muon body (rr >= 5 cm) -- and joined to the owner's segment tags.

The energy here is an APPROXIMATION (sum of dQ/dx / MIP x 2.1 MeV/cm x step);
it is calibrated against the C++ stop_gamma_ke_tot on the items whose role-5
set is fully present, and only used to size the energy guards.  The built C++
logs its own per-blob energy (doc 71 sec 5 checks against that, not this).

Tags: a record's segment ids refer to the payload the scanner looked at, so each
tag is resolved in its record's baseline prep, by source: smx4 -> prep-pdvd-smx4
then smx3 then smx1a; smx3 -> prep-pdvd-smx3 then smx1a; smx1a -> prep-pdvd.
It is then matched into the arm by geometry (census_score.py's C2 rule).

Usage:
  STM_SCAN_RECORD=.../pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json \\
  python3 d71_p4_sizing.py --prep /home/xqian/tmp/d71/prep_d71vsp [--out DIR]
"""
import argparse, collections, json, os, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", default="/home/xqian/tmp/d71/prep_d71vsp")
ap.add_argument("--out", default=None)
ap.add_argument("--mip-dedx", type=float, default=2.1, help="MeV/cm per MIP for the approximate energy")
a = ap.parse_args()

R = C.load_record()
P, _ = C.load_payloads(a.prep, R)
SD = IMG + "/pdhd/stm_michel_scan/"
B = {n: C.load_payloads(SD + d, R)[0] for n, d in (("B1", "prep-pdvd"), ("B2", "prep-pdvd-smx3"), ("B3", "prep-pdvd-smx4"))}
ORDER = {"smx1a": ["B1"], "smx3": ["B2", "B1"], "smx4": ["B3", "B2", "B1"]}
J = sorted(k for k in R if C.judged(R[k]) and k in P)
print("record %s: %d records, %d judged with a payload in %s" % (os.path.basename(C.REC), len(R), len(J), a.prep))


def mean_nn(p0, p1):
    return float(np.mean([np.linalg.norm(p1 - p, axis=1).min() for p in p0[:: max(1, len(p0) // 12)]]))


def match(pts0, pay):
    best, bd = None, 1.5
    for s in pay["pf"]["seg"]:
        d = mean_nn(C.seg_points(s), pts0)
        if d < bd:
            best, bd = s, d
    return best


def resolve_tags(k, pay):
    """{arm segment id: tag} for record k, plus counters."""
    src = R[k].get("source", "smx1a").split()[0]
    out, amb, lost = {}, 0, collections.Counter()
    for t, tg in (R[k].get("tags") or {}).items():
        hits = [n for n in ORDER[src] if k in B[n] and t in C.seg_index(B[n][k])[0]]
        if not hits:
            lost[tg] += 1
            continue
        p0 = C.seg_points(C.seg_index(B[hits[0]][k])[0][t])
        if len(hits) > 1 and mean_nn(p0, C.seg_points(C.seg_index(B[hits[1]][k])[0][t])) > 1.5:
            amb += 1
        s = match(p0, pay)
        if s is None:
            lost[tg + " (no arm segment)"] += 1
            continue
        out[str(s["id"])] = tg
    return out, amb, lost


def approx_ke(sg):
    P3 = C.seg_points(sg)
    if len(P3) < 2:
        return 0.0
    q = np.clip(np.array(sg["dqdx"], float), 0, None)
    st = np.linalg.norm(np.diff(P3, axis=0), axis=1)
    step = (np.r_[st, 0] + np.r_[0, st]) / 2
    return float(np.sum(q / C.MIP_MEDIAN * a.mip_dedx * step))


rows, AMB, LOST = [], 0, collections.Counter()
for k in J:
    pay = P[k]; v = pay["verdict"]
    if not (v["michel_found"] and v["michel_conn_type"] in (1, 2)) or not len(v["michel"]["x"]):
        continue
    tm, amb, lost = resolve_tags(k, pay); AMB += amb; LOST.update(lost)
    stop = np.array([v["stop_x"], v["stop_y"], v["stop_z"]])
    M = np.c_[v["michel"]["x"], v["michel"]["y"], v["michel"]["z"]]
    u = M.mean(0) - stop
    if np.linalg.norm(u) < 1e-6:
        continue
    u = u / np.linalg.norm(u)
    mu = pay["muon"]
    body = np.c_[mu["x"], mu["y"], mu["z"]][np.array(mu["rr"]) >= 5.0]
    if len(v["delta"]["x"]):
        body = np.vstack([body, np.c_[v["delta"]["x"], v["delta"]["y"], v["delta"]["z"]]])
    role = pay["pf"]["chain_role"]; segs = {str(s["id"]): s for s in pay["pf"]["seg"]}
    bycl = collections.defaultdict(list)
    for sid, rl in role.items():
        if rl in (5, 6) and sid in segs:
            bycl[int(sid) // 1000].append((sid, rl))
    for cid, L in sorted(bycl.items()):
        pts = np.vstack([C.seg_points(segs[s]) for s, _ in L])
        ds = float(np.linalg.norm(pts - stop, axis=1).min())
        db = float(np.linalg.norm(pts[:, None, :] - body[None], axis=2).min()) if len(body) else 1e9
        dm = min(ds, float(np.linalg.norm(pts[:, None, :] - M[None], axis=2).min()))
        c = pts.mean(0) - stop
        cosv = float(c @ u / np.linalg.norm(c)) if np.linalg.norm(c) > 0 else 1.0
        tags = [tm.get(s, "untagged") for s, _ in L]
        tg = next((t for t in ("gamma", "michel", "muon", "delta / other") if t in tags), "untagged")
        rows.append(dict(key=k, scanm=bool(C.is_michel(R[k])), cid=cid, role=max(r for _, r in L), tag=tg,
                         ds=ds, db=db, dm=dm, cos=cosv, len=float(sum(segs[s]["len_cm"] for s, _ in L)),
                         ke=float(sum(approx_ke(segs[s]) for s, _ in L)), mke=v["michel_ke_best"], sg_ke=v["stop_gamma_ke_tot"]))
print("tags: ambiguous geometry across baselines %d; unresolved %s" % (AMB, dict(LOST)))
print("candidates with a fitted Michel (conn 1/2): %d; companion clusters (role 5/6): %d" % (len({r["key"] for r in rows}), len(rows)))

# the energy approximation against the C++ capture-gamma energy
cal = collections.defaultdict(float); sg = {}
for r in rows:
    if r["role"] == 5:
        cal[r["key"]] += r["ke"]; sg[r["key"]] = r["sg_ke"]
ratio = np.array([cal[k] / sg[k] for k in cal if sg[k] > 0])
K = 1.0 / float(np.median(ratio)) if len(ratio) else 1.0
if len(ratio):
    print("energy approx / C++ stop_gamma_ke_tot on %d items: median %.2f (q10 %.2f, q90 %.2f); rescale x%.2f"
          % (len(ratio), np.median(ratio), np.percentile(ratio, 10), np.percentile(ratio, 90), K))
for r in rows:
    r["keC"] = r["ke"] * K

T = ("gamma", "michel", "muon", "delta / other", "untagged")


def tab(name, sel):
    S = [r for r in rows if sel(r)]
    print("\n-- %s: %d clusters" % (name, len(S)))
    print("%-14s %5s | %6s %6s %6s | %6s %6s %6s | %6s %8s | %7s %7s" % ("tag", "n", "<15", "15-35", "35-60", "cos>.5", "0-.5", "<0", "dm<db", "len<=10", "keC q50", "keC q90"))
    for t in T:
        X = [r for r in S if r["tag"] == t]
        if not X:
            continue
        ke = np.array([r["keC"] for r in X])
        print("%-14s %5d | %6d %6d %6d | %6d %6d %6d | %6d %8d | %7.1f %7.1f" % (
            t, len(X), sum(r["ds"] < 15 for r in X), sum(15 <= r["ds"] < 35 for r in X), sum(r["ds"] >= 35 for r in X),
            sum(r["cos"] > .5 for r in X), sum(0 <= r["cos"] <= .5 for r in X), sum(r["cos"] < 0 for r in X),
            sum(r["dm"] < r["db"] for r in X), sum(r["len"] <= 10 for r in X), np.median(ke), np.percentile(ke, 90)))


tab("all fitted-Michel candidates", lambda r: True)
tab("scan-Michel items", lambda r: r["scanm"])
tab("NOT scan-Michel items (michel_found false positives)", lambda r: not r["scanm"])


def rule(r, R_, cone, body, L=10.0, kmax=20.0):
    return r["ds"] <= R_ and r["len"] <= L and (cone is None or r["cos"] >= cone) and (not body or r["dm"] < r["db"]) and r["keC"] <= kmax


print("\n=== the grid (len <= 10 cm, blob keC <= 20 MeV) ===")
print("%5s %6s %5s | %6s %6s %6s %7s | %s" % ("R", "cone", "body", "gamma", "delta", "unt", "purity", "items"))
for R_ in (35, 60):
    for cone in (None, 0.0, 0.5, 0.7):
        for body in (False, True):
            S = [r for r in rows if rule(r, R_, cone, body)]
            g = sum(r["tag"] == "gamma" for r in S); d = sum(r["tag"] == "delta / other" for r in S)
            print("%5d %6s %5s | %6d %6d %6d %7.3f | %d" % (R_, "none" if cone is None else "%.1f" % cone, body, g, d,
                                                            sum(r["tag"] == "untagged" for r in S), g / max(g + d, 1), len({r["key"] for r in S})))

for R_ in (35, 60):
    tot = collections.defaultdict(float); core = {}
    for r in rows:
        core[r["key"]] = r["mke"]
        if rule(r, R_, 0.5, True):
            tot[r["key"]] += r["keC"]
    c = np.array([core[k] for k in core]); t = np.array([core[k] + tot[k] for k in core])
    print("\nR %d, cone 0.5, body, no total cap: Michel KE on %d candidates  core q50 %.1f q90 %.1f max %.1f | core+blobs q50 %.1f q90 %.1f max %.1f"
          " | > 52.8 MeV %d -> %d, > 60 %d -> %d" % (R_, len(c), np.median(c), np.percentile(c, 90), c.max(), np.median(t),
                                                    np.percentile(t, 90), t.max(), (c > 52.8).sum(), (t > 52.8).sum(), (c > 60).sum(), (t > 60).sum()))
    bad = sorted({(r["key"], round(r["ds"], 1), round(r["cos"], 2)) for r in rows if r["tag"] in ("delta / other", "muon") and rule(r, R_, 0.5, True)})
    print("  tagged-bad clusters the rule takes:", bad)

if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "p4_rows.json"), "w"), indent=0)
    print("\nwrote", os.path.join(a.out, "p4_rows.json"))
