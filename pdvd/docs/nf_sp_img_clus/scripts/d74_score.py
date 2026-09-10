#!/usr/bin/env python3
"""doc pdvd/74 (P3) -- grade the P3 arms on the owner's record, tags and pins.

Read-only over the arms' preps and the grading record.  Fork of d72_score.py
(sections B-D are its code).

  A  every stop move against the OFF arm: the 18 production retreat/split items
     always, plus every candidate whose n_retreat / n_split / n_chain_segs /
     stop_dis / stop_move_p3_bits changed -- off -> on, with the record's verdict
  B  verdict movers (is_stm, michel_found), by name: gained TP / new FP / lost TP /
     removed FP / not judged, and the census on the judged items
  C  the Michel object on the owner's segment tags (role-3 recall, contamination)
  D  michel_conn_type transitions and the object's len / KE on changed items
  E  the stop against the scan's pins (the 25 pinned items), item by item: the pin is
     placed on the chain the scanner saw (its own baseline prep, by source), at rr = pin_rr
  F  the moved-stop veto (T2c): fires and exemptions that changed, with KE and kink
  G  prediction vs arm: d74_sizing.py's fire lists against the C++'s stop_move_p3_bits

Usage:
  STM_SCAN_RECORD=.../pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json \\
  python3 d74_score.py --off p74voff=/home/xqian/tmp/p74/prep_p74voff \\
      --on p74vts=/home/xqian/tmp/p74/prep_p74vts:ts ... [--pred pred.json]
"""
import argparse, collections, json, os, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
os.environ.setdefault("STM_SCAN_RECORD", IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--off", required=True, help="NAME=PREP of the knob-off arm")
ap.add_argument("--on", action="append", default=[], help="NAME=PREP[:variant], variant in cs / ts / tl; repeatable")
ap.add_argument("--pred", default=None, help="d74_sizing.py --json output")
ap.add_argument("--json", default=None)
a = ap.parse_args()


def parse(spec):
    nm, rest = spec.split("=", 1)
    pth, _, var = rest.partition(":")
    return nm, pth, var


OFF = parse(a.off)
ARMS = [OFF] + [parse(s) for s in a.on]
R = C.load_record()
P = {nm: C.load_payloads(pth, R)[0] for nm, pth, _ in ARMS}
# every candidate of every arm, judged or not (section A and G name them all)
PALL = {}
for nm, pth, _ in ARMS:
    PALL[nm] = {}
    for f in sorted(os.listdir(pth)):
        if f.startswith("smprep-") and f.endswith(".json"):
            PALL[nm][f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(pth, f)))
off = OFF[0]
summary = {}


def rv(k):
    return ("%s/%s" % (R[k]["verdict"], R[k].get("michel_kind"))) if k in R else "not-judged"


# ---------------------------------------------------------------- A. stop moves
print("=== A. stop moves against %s, off -> on ===" % off)
FIELDS = ("n_retreat", "n_split", "n_chain_segs", "stop_dis", "michel_conn_type", "michel_found", "is_stm", "stop_move_p3_bits")
MOVED0 = sorted(k for k, p in PALL[off].items() if p["verdict"]["n_retreat"] or p["verdict"]["n_split"])
print("%s: %d candidates where the retreat or split fired: %s" % (off, len(MOVED0), " ".join(MOVED0)))


def fmt(v, f):
    x = v.get(f, "-")
    return ("%.2f" % x) if isinstance(x, float) else str(x)


for nm, _, var in ARMS[1:]:
    keys = sorted(set(PALL[off]) & set(PALL[nm]))
    ch = []
    for k in keys:
        x, y = PALL[off][k]["verdict"], PALL[nm][k]["verdict"]
        diff = [f for f in FIELDS if f != "stop_move_p3_bits" and
                (round(float(x.get(f, 0)), 3) != round(float(y.get(f, 0)), 3))]
        if diff or y.get("stop_move_p3_bits", 0) or k in MOVED0:
            ch.append((k, diff, x, y))
    print("\n%s (%s): %d rows (the %d production movers + every change); only-%s %s, only-%s %s"
          % (nm, var, len(ch), len(MOVED0), off, sorted(set(PALL[off]) - set(PALL[nm])) or "-", nm, sorted(set(PALL[nm]) - set(PALL[off])) or "-"))
    print("   %-14s %-22s %s" % ("item", "record", " | ".join(f.replace("michel_", "m_") for f in FIELDS)))
    for k, diff, x, y in ch:
        cells = []
        for f in FIELDS:
            xs, ys = fmt(x, f), fmt(y, f)
            cells.append(ys if xs == ys else "%s->%s" % (xs, ys))
        print("   %-14s %-22s %s%s" % (k, rv(k), " | ".join(cells), "" if diff or y.get("stop_move_p3_bits", 0) else "   (unchanged)"))
    summary.setdefault("stop_moves", {})[nm] = [(k, diff) for k, diff, _, _ in ch]

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
    keys = sorted(set(PALL[off]) & set(PALL[nm]))
    for f, truth in (("is_stm", C.is_stopper), ("michel_found", C.is_michel)):
        cls = collections.defaultdict(list)
        for k in keys:
            x, y = PALL[off][k]["verdict"][f], PALL[nm][k]["verdict"][f]
            if x == y:
                continue
            if k not in R or not C.judged(R[k]):
                c = "not judged"
            else:
                t = truth(R[k])
                c = {(True, 1): "gained TP", (False, 1): "NEW FP", (True, 0): "LOST TP", (False, 0): "removed FP"}[(t, y)]
            cls[c].append("%s[%s %d->%d]" % (k, rv(k), x, y))
        print("   %s %s: %s" % (nm, f, "; ".join("%s %d: %s" % (c, len(L), " ".join(L)) for c, L in sorted(cls.items())) or "no movers"))
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


def base_of(k):
    src = R[k].get("source", "smx1a").split()[0]
    return next((B[n][k] for n in ORDER[src] if k in B[n]), None)


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
        per[k] = (sum(r == 3 for r in rl), len(rl), [RN.get(r, "other") for r in rl])
        for r in rl:
            cnt[RN.get(r, "other")] += 1
    n = sum(cnt.values())
    print("%-8s items %3d tags %3d | %s | role 3 = %.1f%%" % (nm, len(per), n, " ".join("%s %d" % kv for kv in sorted(cnt.items())), 100.0 * cnt["role 3"] / max(n, 1)))
    INM[nm] = per
    summary.setdefault("michel_tags", {})[nm] = dict(items=len(per), tags=n, **cnt)
for nm, _, _ in ARMS[1:]:
    chg = sorted("%s(%s -> %s)" % (k, INM[off].get(k, (0, 0, []))[2], v[2]) for k, v in INM[nm].items() if v[2] != INM[off].get(k, (0, 0, []))[2])
    print("%s: michel tags whose role changed, %d items: %s" % (nm, len(chg), " ".join(chg) or "none"))
    summary.setdefault("michel_tags_delta", {})[nm] = chg
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
    keys = sorted(set(PALL[off]) & set(PALL[nm]))
    tr = collections.Counter(); ch = []
    for k in keys:
        x, y = PALL[off][k]["verdict"], PALL[nm][k]["verdict"]
        tr[(x["michel_conn_type"], y["michel_conn_type"])] += 1
        if (x["michel_conn_type"], x["michel_len"], x["michel_ke_best"]) != (y["michel_conn_type"], y["michel_len"], y["michel_ke_best"]):
            ch.append("%s[%s] conn %d->%d len %.1f->%.1f ke %.1f->%.1f kink %.0f->%.0f seg %s->%s" % (
                k, rv(k), x["michel_conn_type"], y["michel_conn_type"], x["michel_len"], y["michel_len"],
                x["michel_ke_best"], y["michel_ke_best"], x["michel_kink_deg"], y["michel_kink_deg"], x["michel_seg_id"], y["michel_seg_id"]))
    print("%s: transitions %s" % (nm, {"%d->%d" % t: n for t, n in sorted(tr.items()) if t[0] != t[1]} or "none"))
    for c in ch:
        print("   ", c)
    summary.setdefault("conn", {})[nm] = ch

# ---------------------------------------------------------------- E. pins
print("\n=== E. the stop against the scan's pins (cm; pin on the scanner's chain at rr = pin_rr) ===")
PIN = {}
for k in sorted(k for k in R if R[k].get("pin_rr") is not None):
    b = base_of(k)
    if b is None:
        continue
    rr0, _, xyz0 = C.profile(b)
    PIN[k] = xyz0[int(np.argmin(np.abs(rr0 - R[k]["pin_rr"])))]


def resid(nm, k):
    if k not in PALL[nm]:
        return None
    v = PALL[nm][k]["verdict"]
    return float(np.linalg.norm(np.array([v["stop_x"], v["stop_y"], v["stop_z"]], float) - PIN[k]))


print("   %-14s %-22s %6s  %s" % ("item", "record", "pin_rr", "  ".join("%9s" % nm for nm, _, _ in ARMS)))
worse = collections.defaultdict(list); better = collections.defaultdict(list)
for k in sorted(PIN):
    r0 = resid(off, k)
    cells = []
    for nm, _, _ in ARMS:
        r1 = resid(nm, k)
        cells.append("%9s" % ("-" if r1 is None else "%.2f" % r1))
        if nm != off and r0 is not None and r1 is not None:
            if r1 > r0 + 1.0:
                worse[nm].append("%s(%.2f->%.2f)" % (k, r0, r1))
            if r1 < r0 - 1e-6:
                better[nm].append("%s(%.2f->%.2f)" % (k, r0, r1))
    print("   %-14s %-22s %6.2f  %s" % (k, rv(k), R[k]["pin_rr"], "  ".join(cells)))
for nm, _, _ in ARMS:
    rs = [x for x in (resid(nm, k) for k in PIN) if x is not None]
    print("%s: n %d median %.2f cm, within 2 cm %d" % (nm, len(rs), np.median(rs), sum(x <= 2 for x in rs)))
for nm, _, _ in ARMS[1:]:
    print("%s: closer to the pin %s; WORSE by more than 1 cm %s" % (nm, " ".join(better[nm]) or "none", " ".join(worse[nm]) or "none"))
    summary.setdefault("pins", {})[nm] = dict(better=better[nm], worse=worse[nm])

# ---------------------------------------------------------------- F. T2c
print("\n=== F. the moved-stop veto (T2c): changes against %s ===" % off)
for nm, _, _ in ARMS[1:]:
    rows = []
    for k in sorted(set(PALL[off]) & set(PALL[nm])):
        x, y = PALL[off][k]["verdict"], PALL[nm][k]["verdict"]
        if (x.get("n_michel_veto", 0), x.get("n_michel_veto_exempt", 0)) != (y.get("n_michel_veto", 0), y.get("n_michel_veto_exempt", 0)):
            rows.append("%s[%s] veto %d->%d exempt %d->%d ke %.2f kink %.1f" % (k, rv(k), x.get("n_michel_veto", 0), y.get("n_michel_veto", 0),
                                                                               x.get("n_michel_veto_exempt", 0), y.get("n_michel_veto_exempt", 0), y["michel_ke_best"], y["michel_kink_deg"]))
    print("%s: %s" % (nm, "; ".join(rows) or "no change"))
    summary.setdefault("t2c", {})[nm] = rows

# ---------------------------------------------------------------- G. prediction vs arm
if a.pred:
    PR = json.load(open(a.pred))
    print("\n=== G. prediction (d74_sizing.py) vs the arm's stop_move_p3_bits ===")
    for nm, _, var in ARMS[1:]:
        if var not in PR:
            continue
        fired = sorted(k for k, p in PALL[nm].items() if p["verdict"].get("stop_move_p3_bits", 0))
        pred = set(PR[var]); got = set(fired); moved = set(PR.get("moved", []))
        print("%s (%s): predicted %d, fired %d | both %s | predicted only %s | fired only %s (of which already moved in production: %s)"
              % (nm, var, len(pred), len(got), sorted(pred & got) or "none", sorted(pred - got) or "none",
                 sorted(got - pred) or "none", sorted((got - pred) & moved) or "none"))
        summary.setdefault("prediction", {})[nm] = dict(both=sorted(pred & got), pred_only=sorted(pred - got), fired_only=sorted(got - pred))

if a.json:
    json.dump(summary, open(a.json, "w"), indent=1, default=str)
    print("\nwrote", a.json)
