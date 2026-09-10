#!/usr/bin/env python3
"""docs pdvd/72 (P3b) + pdvd/73 (P2) -- grade the moved-stop turn exemption and
the attached-Michel-gate operating points on the owner's record and tags.

Read-only over the arms' preps, their per-event PR logs and the grading record.

  A  P3b: every moved-stop-veto fire and every exemption, by name, with the
     kink, the KE and the record's verdict.
  B  verdict movers against the OFF arm (is_stm, michel_found), by name, split
     into gained TP / new FP / lost TP / removed FP / not judged, and the
     census on the judged items.
  C  the Michel object on the owner's segment tags: every `michel` tag on a
     scan-Michel item, matched into the arm (census_score.py's C2 rule), and
     the role it got; recall = role 3.  Contamination = role-3 segments whose
     tag is muon / delta / other (gamma reported apart: the Michel's brems).
  D  michel_conn_type transitions and the object's len / KE on changed items.
  E  rule check: every stop-arm DEBUG line re-judged by this file's twin of
     stm_michel_classify_stop_arm with the arm's own P2 settings (0 mismatches
     expected, up to the line's 2-decimal rounding).
  F  (the OFF arm) the P2 diagnostic: what (b) at several caps and (c) at the
     5 cm window would admit, from the line's kink5 / far_full fields (the
     C++'s own numbers -- the payload points are not the fits), and the
     far-walk fence check (far_len vs far_full where the capped walk was exact).

Tags are resolved in each record's own baseline prep by source (smx4 -> smx3
-> smx1a), as in d71_p4_score.py.

Arm settings (after the prep path, comma-separated, default -1 = off):
  kmin  moved_stop_michel_kink_min (deg)       mlt   michel_mip_lo_turned
  mltk  michel_mip_lo_turned_kink_deg (60)     fsm   michel_far_len_shower_max_cm
  kw    michel_kink_window_cm

Usage:
  STM_SCAN_RECORD=.../pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json \\
  python3 d72_score.py --off p72voff=/home/xqian/tmp/p72/prep_p72voff \\
      --on p72vb60=/home/xqian/tmp/p72/prep_p72vb60:kmin=60 \\
      --on p72v2ab=/home/xqian/tmp/p72/prep_p72v2ab:kmin=60,mlt=0.15,fsm=60
"""
import argparse, collections, glob, json, os, re, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--off", required=True, help="NAME=PREP of the knob-off arm")
ap.add_argument("--on", action="append", default=[], help="NAME=PREP[:k=v,...], repeatable")
ap.add_argument("--work", default=IMG + "/pdvd/work")
ap.add_argument("--json", default=None)
a = ap.parse_args()

DEF = dict(kmin=-1.0, mlt=-1.0, mltk=60.0, fsm=-1.0, kw=-1.0)


def parse(spec):
    nm, rest = spec.split("=", 1)
    pth, _, kv = rest.partition(":")
    s = dict(DEF)
    for x in filter(None, kv.split(",")):
        k, v = x.split("=")
        s[k] = float(v)
    return nm, pth, s


OFF = parse(a.off)
ARMS = [OFF] + [parse(s) for s in a.on]
R = C.load_record()
P = {nm: C.load_payloads(pth, R)[0] for nm, pth, _ in ARMS}
S = {nm: s for nm, _, s in ARMS}
off = OFF[0]
summary = {}


def rv(k):
    return ("%s/%s" % (R[k]["verdict"], R[k].get("michel_kind"))) if k in R else "not-judged"


# ---------------------------------------------------------------- A. P3b
print("=== A. the moved-stop veto (T2c): fires and exemptions, by name ===")
for nm, _, s in ARMS:
    rows = []
    for k, p in sorted(P[nm].items()):
        v = p["verdict"]
        nv, ne = v.get("n_michel_veto", 0), v.get("n_michel_veto_exempt", 0)
        if nv or ne:
            rows.append((k, nv, ne, v["michel_kink_deg"], v["michel_ke_best"], v["michel_found"]))
    sp_m = [r[0] for r in rows if r[2] and r[0] in R and C.is_michel(R[r[0]])]
    sp_x = [r[0] for r in rows if r[2] and not (r[0] in R and C.is_michel(R[r[0]]))]
    print("%s (kmin %g): %d fires, %d vetoed, %d spared -- spared record Michels %s; spared non-Michels %s"
          % (nm, s["kmin"], len(rows), sum(1 for r in rows if r[1]), sum(1 for r in rows if r[2]), sp_m or "none", sp_x or "none"))
    for k, nv, ne, kk, ke, mf in rows:
        print("   %-14s veto %d exempt %d kink %6.1f ke %5.2f michel_found %d | %s" % (k, nv, ne, kk, ke, mf, rv(k)))
    summary.setdefault("p3b", {})[nm] = dict(fires=len(rows), spared_michel=sp_m, spared_other=sp_x)

# ---------------------------------------------------------------- B. movers
print("\n=== B. verdict movers against %s, by name, on the record ===" % off)


def census(nm, f, truth):
    tp = fp = fn = 0
    for k, p in P[nm].items():
        if k not in R or not C.judged(R[k]):
            continue
        t, y = truth(R[k]), bool(p["verdict"][f])
        tp += t and y; fp += (not t) and y; fn += t and not y
    return tp, fp, fn


for f, truth in (("is_stm", C.is_stopper), ("michel_found", C.is_michel)):
    print("  %s census (TP/FP/FN on judged): %s" % (f, "  ".join("%s %d/%d/%d" % ((nm,) + census(nm, f, truth)) for nm, _, _ in ARMS)))
for nm, _, _ in ARMS[1:]:
    keys = sorted(set(P[off]) & set(P[nm]))
    print("%s: common %d, only-%s %s, only-%s %s" % (nm, len(keys), off, sorted(set(P[off]) - set(P[nm])) or "-", nm, sorted(set(P[nm]) - set(P[off])) or "-"))
    for f, truth in (("is_stm", C.is_stopper), ("michel_found", C.is_michel)):
        cls = collections.defaultdict(list)
        for k in keys:
            x, y = P[off][k]["verdict"][f], P[nm][k]["verdict"][f]
            if x == y:
                continue
            if k not in R or not C.judged(R[k]):
                c = "not judged"
            else:
                t = truth(R[k])
                c = {(True, 1): "gained TP", (False, 1): "NEW FP", (True, 0): "LOST TP", (False, 0): "removed FP"}[(t, y)]
            cls[c].append("%s[%s %d->%d]" % (k, rv(k), x, y))
        print("   %s: %s" % (f, "; ".join("%s %d: %s" % (c, len(L), " ".join(L)) for c, L in sorted(cls.items())) or "no movers"))
        summary.setdefault("movers", {}).setdefault(nm, {})[f] = {c: L for c, L in cls.items()}

# ---------------------------------------------------------------- C. the object on the tags
SD = IMG + "/pdhd/stm_michel_scan/"
B = {n: C.load_payloads(SD + d, R)[0] for n, d in (("B1", "prep-pdvd"), ("B2", "prep-pdvd-smx3"), ("B3", "prep-pdvd-smx4"))}
ORDER = {"smx1a": ["B1"], "smx3": ["B2", "B1"], "smx4": ["B3", "B2", "B1"]}


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
    src = R[k].get("source", "smx1a").split()[0]
    out = []
    for t, tg in (R[k].get("tags") or {}).items():
        hit = next((n for n in ORDER[src] if k in B[n] and t in C.seg_index(B[n][k])[0]), None)
        if hit:
            out.append((tg, C.seg_points(C.seg_index(B[hit][k])[0][t])))
    return out


BT = {k: base_tags(k) for k in R if C.judged(R[k])}
TAGS = {}


def arm_tags(nm, k):
    if (nm, k) not in TAGS:
        pay = P[nm][k]
        pairs, rev = [], {}
        for tg, p0 in BT.get(k, []):
            s_ = match(p0, pay)
            pairs.append((tg, s_))
            if s_ is not None:
                rev.setdefault(str(s_["id"]), tg)
        TAGS[(nm, k)] = (pairs, rev)
    return TAGS[(nm, k)]


RN = {1: "role 1", 2: "role 2", 3: "role 3", 4: "role 4", 5: "role 5", 6: "role 6", 7: "role 7", None: "no row"}
print("\n=== C. `michel` tags on scan-Michel items, and the role the arm gave them ===")
INM = {}
for nm, _, _ in ARMS:
    cnt = collections.Counter(); per = {}
    for k in sorted(P[nm]):
        if k not in BT or not C.is_michel(R[k]):
            continue
        pairs, _ = arm_tags(nm, k)
        role = P[nm][k]["pf"]["chain_role"]
        rl = [role.get(str(s_["id"])) if s_ is not None else None for tg, s_ in pairs if tg == "michel"]
        if not rl:
            continue
        per[k] = (sum(r == 3 for r in rl), len(rl))
        for r in rl:
            cnt[RN.get(r, "other")] += 1
    n = sum(cnt.values())
    print("%-8s items %3d tags %3d | %s | role 3 = %.1f%%" % (nm, len(per), n, " ".join("%s %d" % kv for kv in sorted(cnt.items())), 100.0 * cnt["role 3"] / max(n, 1)))
    INM[nm] = per
    summary.setdefault("michel_tags", {})[nm] = dict(items=len(per), tags=n, **cnt)
for nm, _, _ in ARMS[1:]:
    up = sorted("%s(%d->%d/%d)" % (k, INM[off].get(k, (0, 0))[0], v[0], v[1]) for k, v in INM[nm].items() if v[0] > INM[off].get(k, (0, 0))[0])
    dn = sorted("%s(%d->%d/%d)" % (k, INM[off][k][0], INM[nm].get(k, (0, 0))[0], INM[off][k][1]) for k in INM[off] if INM[nm].get(k, (0, 0))[0] < INM[off][k][0])
    print("%s: michel tags into role 3 on %d items: %s" % (nm, len(up), " ".join(up) or "none"))
    print("%s: michel tags OUT of role 3 on %d items: %s" % (nm, len(dn), " ".join(dn) or "none"))
    summary.setdefault("michel_tags_delta", {})[nm] = dict(up=up, down=dn)
print("\n    role-3 segments against every tag (contamination: muon, delta / other; gamma = brems, reported apart)")
for nm, _, _ in ARMS:
    cnt = collections.Counter(); bad = []
    for k, pay in sorted(P[nm].items()):
        if k not in BT:
            continue
        role = pay["pf"]["chain_role"]
        r3 = [sid for sid, rl in role.items() if rl == 3]
        if not r3:
            continue
        _, rev = arm_tags(nm, k)
        for sid in r3:
            tg = rev.get(sid, "untagged")
            cnt[tg] += 1
            if tg in ("muon", "delta / other"):
                bad.append("%s:%s(%s)" % (k, sid, tg))
    print("%-8s role-3 segments %d: %s" % (nm, sum(cnt.values()), dict(sorted(cnt.items()))))
    summary.setdefault("role3_tags", {})[nm] = dict(cnt=dict(cnt), bad=bad)
for nm, _, _ in ARMS[1:]:
    b0 = set(summary["role3_tags"][off]["bad"]); b1 = set(summary["role3_tags"][nm]["bad"])
    print("%s: NEW tagged-bad role-3 segments vs %s: %s" % (nm, off, " ".join(sorted(b1 - b0)) or "none"))
    print("%s: tagged-bad role-3 segments gone: %s" % (nm, " ".join(sorted(b0 - b1)) or "none"))

# ---------------------------------------------------------------- D. conn transitions
print("\n=== D. michel_conn_type transitions against %s, and the object on changed items ===" % off)
for nm, _, _ in ARMS[1:]:
    keys = sorted(set(P[off]) & set(P[nm]))
    tr = collections.Counter(); ch = []
    for k in keys:
        x, y = P[off][k]["verdict"], P[nm][k]["verdict"]
        tr[(x["michel_conn_type"], y["michel_conn_type"])] += 1
        if (x["michel_conn_type"], x["michel_len"], x["michel_ke_best"]) != (y["michel_conn_type"], y["michel_len"], y["michel_ke_best"]):
            ch.append("%s[%s] conn %d->%d len %.1f->%.1f ke %.1f->%.1f kink %.0f->%.0f" % (k, rv(k), x["michel_conn_type"], y["michel_conn_type"], x["michel_len"], y["michel_len"],
                                                                                         x["michel_ke_best"], y["michel_ke_best"], x["michel_kink_deg"], y["michel_kink_deg"]))
    print("%s: transitions %s" % (nm, {"%d->%d" % t: n for t, n in sorted(tr.items()) if t[0] != t[1]} or "none"))
    for c in ch:
        print("   ", c)
    summary.setdefault("conn", {})[nm] = ch

# ---------------------------------------------------------------- E. rule check
RSA = re.compile(r"stop-arm: cluster (\d+) seg (\d+) kind (\d) len ([\d.]+) cm far_len ([\d.]+) cm mip ([\d.]+) kink ([-\d.]+) deg "
                 r"shower (\d) terminal (\d) kink5 ([-\d.]+) far_full ([\d.]+) kink_w ([-\d.]+)")


def twin(ln, far, mip, kink, sh, kw, s):
    """stm_michel_classify_stop_arm with PDVD production thresholds (C++ defaults
    plus michel_shower_min_kink_deg 15) and the arm's P2 settings."""
    kink_ok = kink >= 0
    if kink_ok and kink < 20 and ln > 3 and 0.7 <= mip <= 1.3:
        return 2
    reach = (ln <= 25 and far <= s["fsm"]) if (s["fsm"] >= 0 and sh) else (ln + far <= 25)
    lo = mip > 0.3 or (s["mlt"] >= 0 and kink_ok and kink >= s["mltk"] and mip > s["mlt"])
    charge = lo and mip < 2.0
    shadm = sh and (not kink_ok or kink >= 15)
    turn = (kink_ok and kink >= 30) or (s["kw"] > 0 and kw >= 0 and kw >= 30)
    return 1 if (reach and charge and (shadm or turn)) else 0


ARMLINES = {}
print("\n=== E. rule check: the stop-arm DEBUG lines against this file's twin ===")
for nm, _, s in ARMS:
    L = []
    for d in sorted(glob.glob(os.path.join(a.work, "*_" + nm))):
        ev = os.path.basename(d)[: -len(nm) - 1]
        for lg in glob.glob(os.path.join(d, "wct_pr_*.log")):
            for line in open(lg, errors="replace"):
                m = RSA.search(line)
                if m:
                    g = m.groups()
                    L.append(("%s/%s" % (ev, g[0]), int(g[1]), int(g[2])) + tuple(float(x) for x in g[3:7]) + (int(g[7]), int(g[8])) + tuple(float(x) for x in g[9:]))
    ARMLINES[nm] = L
    mm = [x for x in L if twin(x[3], x[4], x[5], x[6], x[7], x[11], s) != x[2]]
    kinds = collections.Counter(x[2] for x in L)
    print("%s: %d stop-arm lines, kinds %s; twin mismatches %d" % (nm, len(L), dict(sorted(kinds.items())), len(mm)))
    for x in mm[:12]:
        print("   MISMATCH %s seg %d C++ kind %d twin %d | len %.2f far %.2f mip %.2f kink %.1f sh %d kink_w %.1f"
              % (x[0], x[1], x[2], twin(x[3], x[4], x[5], x[6], x[7], x[11], s), x[3], x[4], x[5], x[6], x[7], x[11]))
    summary.setdefault("rule_check", {})[nm] = dict(lines=len(L), mismatches=len(mm))
    if not L:
        print("   (no lines: the arm predates the diagnostic fields)")

# ---------------------------------------------------------------- F. the diagnostic (OFF arm)
print("\n=== F. %s: what P2 (b) and (c) would admit, from the C++'s own kink5 / far_full ===" % off)
L = ARMLINES.get(off, [])
fence = [x for x in L if not x[8] and x[4] <= 25 and abs(x[4] - x[10]) > 0.02]
print("far-walk fence: %d non-terminal arms whose capped far_len (exact, <= 25 cm) differs from the fenced far_full: %s"
      % (len(fence), " ".join("%s/%d(%.2f vs %.2f)" % (x[0], x[1], x[4], x[10]) for x in fence[:10]) or "none"))
big = sorted(((x[10], x[0], x[1]) for x in L if x[10] >= 99.99), reverse=True)
print("arms with far_full at the 100 cm cap (a lower bound): %d" % len(big))
for tag, s in (("(b) 40 cm", dict(DEF, fsm=40)), ("(b) 60 cm", dict(DEF, fsm=60)), ("(b) 80 cm", dict(DEF, fsm=80)), ("(b) 100 cm", dict(DEF, fsm=99.99)),
               ("(a) 0.15", dict(DEF, mlt=0.15)), ("(a)+(b) 60", dict(DEF, mlt=0.15, fsm=60)), ("(c) 5 cm", dict(DEF, kw=5)),
               ("(a)+(b) 60+(c) 5", dict(DEF, mlt=0.15, fsm=60, kw=5))):
    adm = []
    for x in L:
        if x[2] != 0:
            continue
        # recompute with the diagnostic fields: far_full for (b), kink5 for (c)
        far = x[10] if (s["fsm"] >= 0 and x[7] and not x[8]) else x[4]
        if twin(x[3], far, x[5], x[6], x[7], x[9], s) == 1:
            adm.append(x)
    mf = collections.Counter()
    names = []
    for x in adm:
        k = x[0]
        v = P[off][k]["verdict"] if k in P[off] else None
        cls = rv(k)
        mf["michel_found 0" if v and not v["michel_found"] else "michel_found 1"] += 1
        names.append("%s/%d[%s mf %s far_full %.1f kink5 %.1f]" % (k, x[1], cls, v["michel_found"] if v else "?", x[10], x[9]))
    print("%-18s admits %2d kOther arms (%s)" % (tag, len(adm), dict(mf)))
    for n_ in names:
        print("      ", n_)
    summary.setdefault("diagnostic", {})[tag] = names

if a.json:
    json.dump(summary, open(a.json, "w"), indent=1, default=str)
    print("\nwrote", a.json)
