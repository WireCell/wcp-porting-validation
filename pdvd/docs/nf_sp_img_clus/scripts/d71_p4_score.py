#!/usr/bin/env python3
"""doc pdvd/71 -- grade P4 (michel_gamma_collect) on the owner's segment tags.

Read-only over the arms' preps, their per-event PR logs and the grading record.

  A  gamma-tag RECALL: the denominator is every gamma tag on a scan-Michel item
     whose Michel the arm found (michel_found 1); a tag is IN the Michel object
     when its matched arm segment has role 3 (the object) or 4 (P4's blobs).
     On a bare-production arm an unclaimed fragment has no row at all, so it
     reads "no row" -- which is exactly "not in the object".
  B  role-4 CONTAMINATION: every role-4 segment, joined to the tags --
     tagged-good (gamma, michel), tagged-bad (muon, delta / other), untagged
     (unknown: the owner's tags do not cover everything), plus the items the
     record says carry no Michel.
  C  ENERGY: michel_ke_best vs michel_ke_total, counts above 52.8 / 60 / 100 MeV,
     blobs refused by the total-energy guard, the largest additions by name.
  D  RULE CHECK: the C++ DEBUG lines re-judged by this file's twins of
     stm_michel_gamma_gate / stm_michel_gamma_take (0 mismatches expected), and
     for every TAKEN blob its d_stop / cos / d_mich / d_body re-derived from the
     payload's own rows (role-4 points, role-3 rows, the muon's rr >= 5 cm rows
     and the deltas) and compared with the logged numbers.
  E  what a wider radius adds (--on B vs --on A), split by why the narrower arm
     missed it: beyond the radius from the final stop, or inside it but beyond
     the ADMISSION radius from the tagger's stop.

Tags are resolved in each record's own baseline prep by source (smx4 -> smx3
-> smx1a), then matched into the arm by geometry (census_score.py's C2 rule).

Usage:
  STM_SCAN_RECORD=.../pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json \\
  python3 d71_p4_score.py --off /home/xqian/tmp/p4/prep_p4voff \\
      --on p4v35=/home/xqian/tmp/p4/prep_p4v35:35 --on p4v60=/home/xqian/tmp/p4/prep_p4v60:60
"""
import argparse, collections, glob, json, math, os, re, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--off", required=True, help="prep of the knob-off arm")
ap.add_argument("--off-arm", default="p4voff")
ap.add_argument("--on", action="append", required=True, help="ARM=PREP:RADIUS_CM, repeatable")
ap.add_argument("--work", default=IMG + "/pdvd/work")
ap.add_argument("--max-len", type=float, default=10.0)
ap.add_argument("--cos-min", type=float, default=0.5)
ap.add_argument("--max-ke", type=float, default=20.0)
ap.add_argument("--total-max", type=float, default=60.0)
ap.add_argument("--body-excl", type=float, default=5.0, help="dot_body_exclusion_cm")
ap.add_argument("--json", default=None)
a = ap.parse_args()

R = C.load_record()
SD = IMG + "/pdhd/stm_michel_scan/"
B = {n: C.load_payloads(SD + d, R)[0] for n, d in (("B1", "prep-pdvd"), ("B2", "prep-pdvd-smx3"), ("B3", "prep-pdvd-smx4"))}
ORDER = {"smx1a": ["B1"], "smx3": ["B2", "B1"], "smx4": ["B3", "B2", "B1"]}
ARMS = [(a.off_arm, a.off, None)]
for s in a.on:
    nm, rest = s.split("=", 1)
    pth, rad = rest.rsplit(":", 1)
    ARMS.append((nm, pth, float(rad)))
P = {nm: C.load_payloads(pth, R)[0] for nm, pth, _ in ARMS}
summary = {}


def mean_nn(p0, p1):
    return float(np.mean([np.linalg.norm(p1 - p, axis=1).min() for p in p0[:: max(1, len(p0) // 12)]]))


def match(pts0, pay):
    best, bd = None, 1.5
    for s in pay["pf"]["seg"]:
        d = mean_nn(C.seg_points(s), pts0)
        if d < bd:
            best, bd = s, d
    return best


def base_tags(k):
    """[(tag, baseline points)] for record k, resolved by source."""
    src = R[k].get("source", "smx1a").split()[0]
    out = []
    for t, tg in (R[k].get("tags") or {}).items():
        hit = next((n for n in ORDER[src] if k in B[n] and t in C.seg_index(B[n][k])[0]), None)
        if hit:
            out.append((tg, C.seg_points(C.seg_index(B[hit][k])[0][t])))
    return out


BT = {k: base_tags(k) for k in R if C.judged(R[k])}


def arm_tags(nm, k):
    """[(tag, matched arm segment or None)] and {arm seg id: tag}."""
    pay = P[nm][k]
    pairs, rev = [], {}
    for tg, p0 in BT.get(k, []):
        s = match(p0, pay)
        pairs.append((tg, s))
        if s is not None:
            rev.setdefault(str(s["id"]), tg)
    return pairs, rev


# ---------------------------------------------------------------- A. recall
print("=== A. gamma-tag recall: tags on scan-Michel items whose Michel the arm found ===")
print("%-8s %6s %6s | %7s %7s %7s %7s %7s | %7s" % ("arm", "items", "tags", "role 3", "role 4", "role 5", "other", "no row", "in obj"))
REV = {}
INOBJ = {}
for nm, pth, rad in ARMS:
    keys = sorted(k for k in P[nm] if k in BT and C.is_michel(R[k]) and P[nm][k]["verdict"]["michel_found"])
    cnt = collections.Counter(); per_item = {}
    for k in keys:
        pairs, rev = arm_tags(nm, k)
        REV[(nm, k)] = rev
        role = P[nm][k]["pf"]["chain_role"]
        inn = []
        for tg, s in pairs:
            if tg != "gamma":
                continue
            rl = role.get(str(s["id"])) if s is not None else None
            cnt[{3: "role 3", 4: "role 4", 5: "role 5", None: "no row"}.get(rl, "other")] += 1
            inn.append(rl in (3, 4))
        if inn:
            per_item[k] = (sum(inn), len(inn))
    n = sum(cnt.values())
    print("%-8s %6d %6d | %7d %7d %7d %7d %7d | %6.1f%%" % (nm, len(per_item), n, cnt["role 3"], cnt["role 4"], cnt["role 5"], cnt["other"], cnt["no row"],
                                                          100.0 * (cnt["role 3"] + cnt["role 4"]) / max(n, 1)))
    INOBJ[nm] = per_item
    summary.setdefault("recall", {})[nm] = dict(items=len(per_item), tags=n, **cnt)
for nm, _, rad in ARMS[1:]:
    gain = sorted((k, INOBJ[a.off_arm].get(k, (0, 0))[0], v[0], v[1]) for k, v in INOBJ[nm].items() if v[0] > INOBJ[a.off_arm].get(k, (0, 0))[0])
    print("\n%s: items whose in-object gamma count rose (%d): %s" % (nm, len(gain), " ".join("%s(%d->%d/%d)" % g for g in gain)))
    lost = sorted(k for k, v in INOBJ[nm].items() if v[0] < INOBJ[a.off_arm].get(k, (0, 0))[0])
    print("%s: items whose in-object gamma count FELL: %s" % (nm, " ".join(lost) or "none"))

# ---------------------------------------------------------------- B. contamination
print("\n=== B. role-4 segments against the tags ===")
for nm, pth, rad in ARMS[1:]:
    cnt = collections.Counter(); bad = []; nonmich = []; unt = collections.Counter()
    for k, pay in sorted(P[nm].items()):
        role = pay["pf"]["chain_role"]
        r4 = [sid for sid, rl in role.items() if rl == 4]
        if not r4:
            continue
        if k not in BT:
            cnt["item not judged"] += len(r4); continue
        rev = REV.get((nm, k)) or arm_tags(nm, k)[1]
        for sid in r4:
            tg = rev.get(sid, "untagged")
            cnt[tg] += 1
            if tg in ("muon", "delta / other"):
                bad.append("%s:%s(%s)" % (k, sid, tg))
            if tg == "untagged":
                unt["on a scan-Michel item" if C.is_michel(R[k]) else "on a no-Michel item"] += 1
        if not C.is_michel(R[k]):
            nonmich.append("%s[%s,%s](%d segs)" % (k, R[k]["verdict"], R[k].get("michel_kind"), len(r4)))
    good = cnt["gamma"] + cnt["michel"]; badn = cnt["muon"] + cnt["delta / other"]
    print("%s: role-4 segments %d -> %s" % (nm, sum(cnt.values()), dict(cnt)))
    print("   purity (gamma + michel) / tagged = %d / %d = %.3f; untagged %s" % (good, good + badn, good / max(good + badn, 1), dict(unt)))
    print("   tagged-bad: %s" % (" ".join(bad) or "none"))
    print("   items the record says carry NO Michel that P4 fired on: %s" % (" ".join(nonmich) or "none"))
    summary.setdefault("contamination", {})[nm] = dict(cnt=dict(cnt), purity=good / max(good + badn, 1), bad=bad, nonmich=nonmich)

# ---------------------------------------------------------------- C. energy
print("\n=== C. energy: michel_ke_best (the core, unchanged) vs michel_ke_total ===")
for nm, pth, rad in ARMS[1:]:
    V = [p["verdict"] for p in P[nm].values() if p["verdict"]["michel_found"]]
    if not V or "michel_ke_total" not in V[0]:
        print("%s: no michel_ke_total branch" % nm); continue
    kb = np.array([v["michel_ke_best"] for v in V]); kt = np.array([v["michel_ke_total"] for v in V])
    ng = np.array([v["n_michel_gammas"] for v in V]); cap = sum(v["n_michel_gamma_capped"] for v in V)
    print("%s: %d michel_found candidates, %d with >= 1 blob (%d blobs), %d blobs refused by the total-energy guard"
          % (nm, len(V), (ng > 0).sum(), ng.sum(), cap))
    print("   best  q50 %.1f q90 %.1f max %.1f | total q50 %.1f q90 %.1f max %.1f MeV"
          % (np.median(kb), np.percentile(kb, 90), kb.max(), np.median(kt), np.percentile(kt, 90), kt.max()))
    for thr in (52.8, 60.0, 100.0):
        print("   > %5.1f MeV: best %d, total %d" % (thr, (kb > thr).sum(), (kt > thr).sum()))
    add = sorted(((p["verdict"]["michel_ke_total"] - p["verdict"]["michel_ke_best"], k) for k, p in P[nm].items()
                  if p["verdict"]["michel_found"] and p["verdict"].get("n_michel_gammas", 0) > 0), reverse=True)
    print("   largest additions: %s" % " ".join("%s(+%.1f->%.1f)" % (k, d, P[nm][k]["verdict"]["michel_ke_total"]) for d, k in add[:8]))
    summary.setdefault("energy", {})[nm] = dict(n=len(V), with_blob=int((ng > 0).sum()), blobs=int(ng.sum()), capped=int(cap),
                                                over=[int((kt > t).sum()) for t in (52.8, 60.0, 100.0)], over_best=[int((kb > t).sum()) for t in (52.8, 60.0, 100.0)])


# ---------------------------------------------------------------- D. rule check
def gate(d_stop, ln, cosv, d_mich, d_body, ke, radius):
    if not all(math.isfinite(x) for x in (d_stop, ln, cosv, d_mich, d_body, ke)):
        return 6
    if d_stop > radius: return 1
    if ln > a.max_len: return 2
    if cosv < a.cos_min: return 3
    if d_body <= d_mich: return 4
    if ke > a.max_ke: return 5
    return 0


def take(core, kes):
    out, tot = [], core
    for e in kes:
        if not math.isfinite(core) or not math.isfinite(e) or e < 0 or tot + e > a.total_max:
            out.append(0)
        else:
            out.append(1); tot += e
    return out


RCL = re.compile(r"michel-gamma-cl: cluster (\d+) comp (\d+) d_stop (\S+) len (\S+) cos (\S+) d_mich (\S+) d_body (\S+) ke (\S+) gate (\d+)")
RTK = re.compile(r"michel-gamma-take: cluster (\d+) comp (\d+) ke (\S+) take (\d) live (\d) core (\S+)")
RCD = re.compile(r"michel-gamma: cluster (\d+) conn (\d) rows (\d+) dir (\S+) (\S+) (\S+)")
print("\n=== D. rule check: the C++ DEBUG lines against this file's twins ===")
for nm, pth, rad in ARMS[1:]:
    cl, tk, cd = collections.defaultdict(list), collections.defaultdict(list), {}
    for d in sorted(glob.glob(os.path.join(a.work, "*_" + nm))):
        ev = os.path.basename(d)[: -len(nm) - 1]
        for lg in glob.glob(os.path.join(d, "wct_pr_*.log")):
            for line in open(lg, errors="replace"):
                m = RCL.search(line)
                if m:
                    cl["%s/%s" % (ev, m.group(1))].append([int(m.group(2))] + [float(x) for x in m.group(3, 4, 5, 6, 7, 8)] + [int(m.group(9))]); continue
                m = RTK.search(line)
                if m:
                    tk["%s/%s" % (ev, m.group(1))].append((int(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6)))); continue
                m = RCD.search(line)
                if m:
                    cd["%s/%s" % (ev, m.group(1))] = np.array([float(x) for x in m.group(4, 5, 6)])
    ng = sum(len(v) for v in cl.values()); gm = []
    gcount = collections.Counter()
    for k, L in cl.items():
        for comp, ds, ln, cs, dm, db, ke, g in L:
            gcount[g] += 1
            g2 = gate(ds, ln, cs, dm, db, ke, rad)
            if g2 != g:
                gm.append("%s comp %d C++ %d twin %d" % (k, comp, g, g2))
    tm = []
    for k, L in tk.items():
        live = L[0][3]; core = L[0][4]
        want = take(core, [e for _, e, _, _, _ in L]) if live else [0] * len(L)
        if want != [t for _, _, t, _, _ in L]:
            tm.append(k)
        # the take lines must be the gate-0 blobs, nearest first
        g0 = sorted(((ds, comp) for comp, ds, *_r, g in cl.get(k, []) if g == 0))
        if [c for _, c in g0] != [c for c, *_ in L]:
            tm.append(k + " (order/membership)")
    print("%s: %d candidates offered blobs, %d blob lines; gate codes %s; gate mismatches %d; take mismatches %d"
          % (nm, len(cl), ng, dict(sorted(gcount.items())), len(gm), len(tm)))
    for s in (gm + tm)[:20]:
        print("   MISMATCH", s)
    # geometry of the taken blobs, from the payload's rows
    # The payload rows are rounded to 0.01 cm, so the direction re-derived from
    # them can differ from the C++ one by a few 1e-3; flag only > 5e-3.
    worst = dict(d_stop=0.0, cos=0.0, d_mich=0.0, d_body=0.0, dir=0.0); nchk = 0; gbad = []
    for k, L in tk.items():
        if k not in P[nm]:
            continue
        v = P[nm][k]["verdict"]; pay = P[nm][k]
        stop = np.array([v["stop_x"], v["stop_y"], v["stop_z"]])
        M = np.c_[v["michel"]["x"], v["michel"]["y"], v["michel"]["z"]]
        mu = pay["muon"]
        body = np.c_[mu["x"], mu["y"], mu["z"]][np.array(mu["rr"]) >= a.body_excl - 1e-9]
        if len(v["delta"]["x"]):
            body = np.vstack([body, np.c_[v["delta"]["x"], v["delta"]["y"], v["delta"]["z"]]])
        u = M.mean(0) - stop; u = u / np.linalg.norm(u)
        if k in cd:
            worst["dir"] = max(worst["dir"], float(np.abs(cd[k] - u).max()))
            if np.abs(cd[k] - u).max() > 5e-3:
                gbad.append("%s dir C++ %s payload %s" % (k, np.round(cd[k], 4), np.round(u, 4)))
        D4 = np.c_[v["dots"]["x"], v["dots"]["y"], v["dots"]["z"]]; S4 = np.array(v["dots"]["seg"])
        logged = {c[0]: c for c in cl.get(k, [])}
        for comp, ke, t, live, core in L:
            if not t:
                continue
            pts = D4[S4 // 1000 == comp]
            if not len(pts):
                gbad.append("%s comp %d taken but no role-4 row" % (k, comp)); continue
            ds = np.linalg.norm(pts - stop, axis=1).min()
            c = pts.mean(0) - stop; cs = float(c @ u / np.linalg.norm(c))
            dm = min(ds, np.linalg.norm(pts[:, None] - M[None], axis=2).min())
            db = np.linalg.norm(pts[:, None] - body[None], axis=2).min() if len(body) else 1e9
            lg = logged[comp]; nchk += 1
            for nmv, x, y in (("d_stop", ds, lg[1]), ("cos", cs, lg[3]), ("d_mich", dm, lg[4]), ("d_body", db, lg[5])):
                worst[nmv] = max(worst[nmv], abs(x - y))
    print("   taken blobs re-derived from the payload: %d; worst |payload - logged|: %s" % (nchk, {k: round(v, 4) for k, v in worst.items()}))
    for s in gbad[:10]:
        print("   GEOM", s)
    summary.setdefault("rule_check", {})[nm] = dict(candidates=len(cl), blob_lines=ng, gate_codes=dict(gcount), gate_mismatch=len(gm), take_mismatch=len(tm),
                                                    geom_checked=nchk, worst=worst, geom_flags=len(gbad))

# ---------------------------------------------------------------- E. the wider radius
if len(ARMS) >= 3:
    (na, pa, ra), (nb, pb, rb) = ARMS[1], ARMS[2]
    print("\n=== E. gamma tags in the object on %s but not on %s ===" % (nb, na))
    why = collections.Counter(); names = []
    for k in sorted(set(P[na]) & set(P[nb])):
        if k not in BT or not C.is_michel(R[k]):
            continue
        va, vb = P[na][k]["verdict"], P[nb][k]["verdict"]
        if not vb["michel_found"]:
            continue
        pa_, _ = arm_tags(na, k); pb_, _ = arm_tags(nb, k)
        rola, rolb = P[na][k]["pf"]["chain_role"], P[nb][k]["pf"]["chain_role"]
        for (tg, sa), (_, sb) in zip(pa_, pb_):
            if tg != "gamma" or sb is None or rolb.get(str(sb["id"])) not in (3, 4):
                continue
            if sa is not None and rola.get(str(sa["id"])) in (3, 4):
                continue
            pts = C.seg_points(sb)
            fin = np.array([vb["stop_x"], vb["stop_y"], vb["stop_z"]]); tag = np.array([vb["tagger_stop_x"], vb["tagger_stop_y"], vb["tagger_stop_z"]])
            dfin = np.linalg.norm(pts - fin, axis=1).min(); dtag = np.linalg.norm(pts - tag, axis=1).min()
            w = ("beyond %g cm of the final stop" % ra) if dfin > ra else (("inside %g of the final stop, beyond admission from the tagger's stop" % ra) if dtag > ra
                                                                            else "inside both (the fit or a gate moved)")
            why[w] += 1; names.append("%s(%.1f/%.1f)" % (k, dfin, dtag))
    print("   %d tags: %s" % (sum(why.values()), dict(why)))
    print("   ", " ".join(names[:60]))
    summary["wider"] = dict(why)

if a.json:
    json.dump(summary, open(a.json, "w"), indent=1, default=str)
    print("\nwrote", a.json)
