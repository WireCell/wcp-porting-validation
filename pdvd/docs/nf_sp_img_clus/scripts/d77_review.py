#!/usr/bin/env python3
"""doc pdvd/77 -- the PDVD STM / Michel chain against the hand-scan record, on
one bare-production arm: where it stands, every remaining miss by name and
mechanism, what the record itself can and cannot grade, and how much is left.

Read-only over one prep dir and the record.  Sections:
  1  census: is_stm and michel_found on the judged items, purity / efficiency / F1,
     by tranche, by source (smx1a / smx3 / smx4), by run
  2  missed stoppers (FN) by name: reject bits, whether a Michel object exists and
     its KE / length, the anchor and fallback state -- grouped by the deciding bit
  3  false stoppers (FP) by name, the same fields
  4  missed Michels: items the record calls STM_MICHEL with michel_found 0 --
     (a) no candidate on the arm, (b) candidate rejected as a stopper, (c) found
     stopper whose Michel search failed; with the scan's michel tags matched into
     the arm's segments (role given by the chain)
  5  spurious Michels: michel_found 1 on a judged non-STM_MICHEL item; conn / KE / len
  6  the record: confidence and FRAG / MESSY / UNCLEAR counts; judged items where the
     chain and a HIGH / owner-confidence verdict disagree (the reconstruction's
     fault) against LOW / medium ones (a re-judge candidate)
  7  room: the reachable populations, sized here -- FN with shape bits only and a
     sub-threshold Michel; FN on profile_sparse alone; FN on the boundary; FN on
     continuation / PID / geometry; missed Michels by conn type of the nearest tag
  8  the judged items with NO candidate: what the STM tagger recorded for the cluster
     (T_stm_pass status) and whether CheckSTM_Michel's max_candidates cap dropped it
     (the wct_pr log's "N candidates, keeping the first 8" line + the TaggerCheckSTM
     STM=1 lines of the same event)
  9  exact offline re-verdicts of three single-consumer settings, on this arm:
     ks_margin (shape_flat), P1's floors (topology_michel_ke_min / _len_min_cm),
     and a michel_found size gate (ke / len) -- gains and losses by name

Usage: STM_SCAN_RECORD=... d77_review.py --prep DIR --arm LABEL [--json OUT]
"""
import argparse, collections, json, os, sys
import numpy as np

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
os.environ.setdefault("STM_SCAN_RECORD", IMG + "/pdvd/docs/scan/pdvd_stm_michel_smx1a_smx3_smx4_verdicts.json")
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--json", default=None)
a = ap.parse_args()

R = C.load_record()
P, _ = C.load_payloads(a.prep, R)
PALL = {}
for f in sorted(os.listdir(a.prep)):
    if f.startswith("smprep-") and f.endswith(".json"):
        PALL[f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(a.prep, f)))
J = {k: r for k, r in R.items() if C.judged(r)}
print("record %s: %d items, %d judged; arm %s: %d candidates, %d on judged items" % (os.path.basename(C.REC), len(R), len(J), a.arm, len(PALL), sum(1 for k in J if k in PALL)))
out = {}


def v(k):
    return PALL[k]["verdict"] if k in PALL else None


def stm(k):
    x = v(k); return int(x["is_stm"]) if x else 0


def mf(k):
    x = v(k); return int(x["michel_found"]) if x else 0


def rates(keys, pred, truth):
    tp = sum(1 for k in keys if pred(k) and truth(R[k])); fp = sum(1 for k in keys if pred(k) and not truth(R[k]))
    fn = sum(1 for k in keys if not pred(k) and truth(R[k])); tn = len(keys) - tp - fp - fn
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1); f1 = 2 * pu * ef / max(pu + ef, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, purity=round(pu, 3), efficiency=round(ef, 3), f1=round(f1, 3))


def fmt(d):
    return "%3d / %2d / %3d / %3d  pur %.3f eff %.3f F1 %.3f" % (d["tp"], d["fp"], d["fn"], d["tn"], d["purity"], d["efficiency"], d["f1"])


# ---- 1 census
print("\n=== 1. census on the judged items (TP / FP / FN / TN; an item with no candidate counts as 0) ===")
out["census"] = {}
for nm, pred, truth in (("is_stm", stm, C.is_stopper), ("michel_found", mf, C.is_michel)):
    keys = sorted(J)
    print("%-13s all %3d: %s" % (nm, len(keys), fmt(rates(keys, pred, truth))))
    out["census"][nm] = {"all": rates(keys, pred, truth)}
    for lab, grp in (("tranche", lambda r: "tranche %s" % r.get("tranche")), ("source", lambda r: str(r.get("source", "?")).split()[0]),
                     ("run", lambda r: r["key"].split("_")[0]), ("confidence", lambda r: r.get("confidence"))):
        for g in sorted(set(grp(J[k]) for k in keys)):
            ks = [k for k in keys if grp(J[k]) == g]
            print("   %-10s %-10s n %3d: %s" % (lab, g, len(ks), fmt(rates(ks, pred, truth))))
            out["census"][nm][g] = rates(ks, pred, truth)

SHAPE = ("no_bragg", "shape_flat", "plateau_off_mip", "profile_sparse")


def bits(k):
    x = v(k); return list(x["reject_names"]) if x else ["no candidate"]


def desc(k):
    x = v(k)
    if not x:
        return "no candidate on the arm"
    return ("bits %s | michel_found %d conn %d ke %.1f len %.1f | anchor %.2f fb %s | contrast %.2f/%.2f ks %+.3f | stop_dis %.1f n_segs %d" %
            ("+".join(x["reject_names"]) or "none", x["michel_found"], x["michel_conn_type"], x["michel_ke_best"], x["michel_len"],
             x["bragg_anchor_shift_cm"], x.get("bragg_anchor_fallback", "-"), x["contrast"] or 0, x["contrast_expected"] or 0,
             (x["ks_flat"] - x["ks_mu"]), x["stop_dis"], x["n_chain_segs"]))


def rv(k):
    r = R[k]; return "%s/%s/%s" % (r["verdict"], r.get("michel_kind"), r.get("confidence"))


# ---- 2 FN stoppers
print("\n=== 2. missed stoppers (record STM_*, is_stm 0), grouped by the deciding bits ===")
FN = sorted(k for k in J if C.is_stopper(J[k]) and not stm(k))
groups = collections.defaultdict(list)
for k in FN:
    b = bits(k)
    key = "no candidate" if b == ["no candidate"] else ("shape only: " + "+".join(sorted(x for x in b if x in SHAPE)) if all(x in SHAPE for x in b) else "other bits: " + "+".join(sorted(x for x in b if x not in SHAPE)))
    groups[key].append(k)
print("%d missed stoppers; %d carry a Michel object (michel_found 1); %d are STM_MICHEL on the record" % (len(FN), sum(mf(k) for k in FN), sum(C.is_michel(J[k]) for k in FN)))
for g, ks in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print("  -- %s: %d" % (g, len(ks)))
    for k in ks:
        print("     %-14s %-26s %s" % (k, rv(k), desc(k)))
out["fn"] = {g: ks for g, ks in groups.items()}

# ---- 3 FP stoppers
print("\n=== 3. false stoppers (record THRU / FRAG_THRU, is_stm 1) ===")
FP = sorted(k for k in J if not C.is_stopper(J[k]) and stm(k))
for k in FP:
    print("  %-14s %-26s %s | topology_cleared %s" % (k, rv(k), desc(k), v(k).get("topology_cleared_bits")))
out["fp"] = FP

# ---- 4 missed Michels
print("\n=== 4. missed Michels (record STM_MICHEL, michel_found 0) ===")
MM = sorted(k for k in J if C.is_michel(J[k]) and not mf(k))
cls = collections.defaultdict(list)
for k in MM:
    if k not in PALL:
        cls["(a) no candidate on the arm"].append(k)
    elif not stm(k):
        cls["(b) candidate rejected as a stopper"].append(k)
    else:
        cls["(c) found stopper, Michel search failed"].append(k)
for g, ks in sorted(cls.items()):
    print("  -- %s: %d" % (g, len(ks)))
    for k in ks:
        x = v(k)
        tagroles = ""
        if x:
            role = PALL[k]["pf"]["chain_role"]
            mt = [t for t, tg in (J[k].get("tags") or {}).items() if tg == "michel"]
            tagroles = " | scan michel tags -> chain role: " + ", ".join("%s:%s" % (t, role.get(t, "no row")) for t in mt) if mt else " | no michel tag on the record"
        print("     %-14s %-26s %s%s" % (k, rv(k), desc(k), tagroles))
out["missed_michel"] = {g: ks for g, ks in cls.items()}

# ---- 5 spurious Michels
print("\n=== 5. spurious Michels (judged non-STM_MICHEL, michel_found 1) ===")
SM = sorted(k for k in J if not C.is_michel(J[k]) and mf(k))
for k in SM:
    print("  %-14s %-26s %s" % (k, rv(k), desc(k)))
out["spurious_michel"] = SM

# ---- 6 the record
print("\n=== 6. the record ===")
cnt = collections.Counter((r["verdict"], r.get("confidence")) for r in R.values())
print("verdict x confidence:", dict(sorted(cnt.items())))
print("not judged (MESSY / UNCLEAR): %d; FRAG_*: %d; low confidence: %d; owner re-judged: %d" %
      (sum(1 for r in R.values() if not C.judged(r)), sum(1 for r in R.values() if r["verdict"].startswith("FRAG")),
       sum(1 for r in R.values() if r.get("confidence") == "low"), sum(1 for r in R.values() if r.get("confidence") == "owner")))
dis = [k for k in J if stm(k) != int(C.is_stopper(J[k]))]
byc = collections.Counter(J[k].get("confidence") for k in dis)
print("is_stm disagreements %d by confidence: %s" % (len(dis), dict(byc)))
print("  low / medium-confidence disagreements (re-judge candidates before any fix): %s" %
      " ".join("%s[%s]" % (k, rv(k)) for k in sorted(dis) if J[k].get("confidence") in ("low", "medium")))
out["record"] = dict(disagreements=dis, by_confidence=dict(byc))

# ---- 7 room
print("\n=== 7. room: reachable populations, sized on this arm ===")
room = {}
def room_add(name, ks):
    room[name] = sorted(ks); print("  %-70s %3d: %s" % (name, len(ks), " ".join(sorted(ks))))
shape_only = [k for k in FN if k in PALL and bits(k) and all(x in SHAPE for x in bits(k))]
room_add("FN, shape bits only, Michel object present (P1 floors: ke < 10 or len < 3)", [k for k in shape_only if mf(k)])
room_add("FN, shape bits only, NO Michel object, record says STM_MICHEL", [k for k in shape_only if not mf(k) and C.is_michel(J[k])])
room_add("FN, shape bits only, NO Michel object, record says STM_ONLY", [k for k in shape_only if not mf(k) and not C.is_michel(J[k])])
room_add("FN, profile_sparse among the bits", [k for k in FN if "profile_sparse" in bits(k)])
room_add("FN, stop_near_boundary among the bits", [k for k in FN if "stop_near_boundary" in bits(k)])
room_add("FN, continuation among the bits", [k for k in FN if "continuation" in bits(k)])
room_add("FN, not_muon_pid / vertex_hadron / profile_geometry / stop_into_dead / cluster_not_track", [k for k in FN if any(x in bits(k) for x in ("not_muon_pid", "vertex_hadron", "profile_geometry", "stop_into_dead", "cluster_not_track"))])
room_add("FN, no candidate (the tagger's domain)", [k for k in FN if k not in PALL])
room_add("missed Michel, found stopper, michel tag has a chain row (role != 3)", [k for k in cls.get("(c) found stopper, Michel search failed", []) if any(PALL[k]["pf"]["chain_role"].get(t) is not None for t, tg in (J[k].get("tags") or {}).items() if tg == "michel")])
room_add("missed Michel, found stopper, michel tag has NO row", [k for k in cls.get("(c) found stopper, Michel search failed", []) if not any(PALL[k]["pf"]["chain_role"].get(t) is not None for t, tg in (J[k].get("tags") or {}).items() if tg == "michel")])
room_add("missed Michel, stopper rejected (needs the stop first)", cls.get("(b) candidate rejected as a stopper", []))
room_add("missed Michel, no candidate", cls.get("(a) no candidate on the arm", []))
room_add("spurious Michel on a THRU item", [k for k in SM if not C.is_stopper(J[k])])
room_add("spurious Michel on an STM_ONLY item", [k for k in SM if C.is_stopper(J[k])])
out["room"] = room

# ---- 8 no candidate: the tagger's record and the candidate cap
print("\n=== 8. judged items with no candidate: the STM tagger's pass status, and the max_candidates cap ===")
import glob, re
WORK = IMG + "/pdvd/work"
STATUS = {0: "accepted", 1: "TGM", 2: "long leftover past kink", 3: "dQ/dx eval (KS) rejected", 4: "extra-tracks veto",
          5: "proton endpoint", 6: "fitted, no decision", 7: "accept_guards / cathode / second-track / deficit / vertex-kink", 8: "round-2 fit empty"}
capped = {}; stm1 = collections.defaultdict(set)
for d in sorted(glob.glob("%s/*_%s" % (WORK, a.arm))):
    ev = os.path.basename(d)[: -len(a.arm) - 1]
    for f in glob.glob(d + "/wct_pr_*.log"):
        for line in open(f, errors="replace"):
            m = re.search(r"CheckSTM_Michel: (\d+) candidates, keeping the first (\d+)", line)
            if m: capped[ev] = (int(m.group(1)), int(m.group(2)))
            m = re.search(r"TaggerCheckSTM: cluster (\d+) . STM=1", line)
            if m: stm1[ev].add(int(m.group(1)))
cand_ev = collections.defaultdict(set)
for k in PALL: ev, c = k.split("/"); cand_ev[ev].add(int(c))
dropped = sorted("%s/%d" % (ev, c) for ev in capped for c in stm1[ev] if c not in cand_ev[ev])
print("events where the cap fired: %d (%s); tagger-flagged clusters dropped by it: %d, judged %d: %s" %
      (len(capped), " ".join("%s %d->%d" % (ev, n, kk) for ev, (n, kk) in sorted(capped.items())), len(dropped),
       sum(1 for k in dropped if k in J), [(k, rv(k)) for k in dropped if k in J]))
NC = sorted(k for k in J if k not in PALL)
byst = collections.defaultdict(list)
for k in NC:
    ev, c = k.split("/"); t = C.stm_pass(ev, a.arm)
    if t is None: st = "no T_stm_pass"
    else:
        sel = (t["cluster_id"] == int(c))
        st = "not fitted by the tagger" if not sel.any() else "status %s" % "+".join("%d (%s)" % (x, STATUS.get(x, "?")) for x in sorted(set(int(v) for v in t["status"][sel])))
    if k in dropped: st += " | DROPPED BY max_candidates"
    byst[st].append(k)
print("%d judged items without a candidate (%s):" % (len(NC), dict(collections.Counter(J[k]["verdict"] for k in NC))))
for st, ks in sorted(byst.items(), key=lambda kv: -len(kv[1])):
    print("  -- %s: %d" % (st, len(ks)))
    for k in ks: print("     %-14s %s" % (k, rv(k)))
out["no_candidate"] = {st: ks for st, ks in byst.items()}; out["cap_dropped"] = dropped

# ---- 9 exact offline re-verdicts
print("\n=== 9. exact offline re-verdicts on this arm (single-consumer settings) ===")
SFb, NBb, PSb = 1 << 3, 1 << 2, 1 << 9; SH = SFb | NBb | PSb
def score(sel):
    tp = sum(1 for k in J if sel(k) and C.is_stopper(J[k])); fp = sum(1 for k in J if sel(k) and not C.is_stopper(J[k])); fn = sum(1 for k in J if not sel(k) and C.is_stopper(J[k]))
    g = [k for k in J if sel(k) and not stm(k)]; l = [k for k in J if not sel(k) and stm(k)]
    return "TP/FP/FN %d/%d/%d | gained %s | lost %s" % (tp, fp, fn, ["%s[%s]" % (k, J[k]["verdict"][:9]) for k in g] or "-", ["%s[%s]" % (k, J[k]["verdict"][:9]) for k in l] or "-")
print("-- ks_margin (production -0.02): shape_flat re-read from the recorded ks pair")
for m in (0.0, -0.01, -0.02, -0.03, -0.04):
    def sel(k, m=m):
        x = v(k)
        if not x: return False
        b = int(x["reject_bits"] or 0); tcb = int(x.get("topology_cleared_bits") or 0)
        if b & ~SFb: return False
        if tcb & SFb: return True
        return not (x["ks_mu"] + m >= x["ks_flat"])
    print("   m=%+.2f %s" % (m, score(sel)))
print("-- P1 floors (production ke >= 10 MeV, len >= 3 cm; clears no_bragg + shape_flat + sparse on an attached / bridged Michel)")
for K, L in ((10, 3), (10, 2), (8, 3), (5, 3), (3, 3), (5, 2), (3, 2), (0, 0)):
    def sel(k, K=K, L=L):
        x = v(k)
        if not x: return False
        b = int(x["reject_bits"] or 0) | int(x.get("topology_cleared_bits") or 0)
        if x["michel_found"] and x["michel_conn_type"] in (1, 2) and x["michel_ke_best"] >= K and x["michel_len"] >= L: b &= ~SH
        return b == 0
    print("   ke>=%2d len>=%d %s" % (K, L, score(sel)))
print("-- plateau_mip window (production [0.6, 1.6] x mip_dqdx 55000; plateau_off_mip is set at one site): items whose ONLY remaining bit is plateau_off_mip")
PMb = 1 << 10
only_pm = sorted(k for k in J if v(k) and (int(v(k)["reject_bits"] or 0) & ~PMb) == 0 and (int(v(k)["reject_bits"] or 0) & PMb))
for k in only_pm:
    print("   %-14s %-26s plateau_med / mip %.2f  contrast %.2f/%.2f ks %+.3f" % (k, rv(k), (v(k)["plateau_med"] or 0) / 55000.0, v(k)["contrast"] or 0, v(k)["contrast_expected"] or 0, v(k)["ks_flat"] - v(k)["ks_mu"]))
for lo, hi in ((0.6, 1.6), (0.5, 1.6), (0.55, 1.6), (0.6, 2.0), (0.5, 2.0), (0.0, 99.0)):
    def sel(k, lo=lo, hi=hi):
        x = v(k)
        if not x: return False
        b = int(x["reject_bits"] or 0)
        if b & ~PMb: return False
        if not (b & PMb): return True
        pm = (x["plateau_med"] or 0) / 55000.0
        return not (pm < lo or pm > hi)
    print("   window [%.2f, %.2f] %s" % (lo, hi, score(sel)))
print("-- michel_found size gate (an object below ke K or len L would not count): TP / FP of 136 / 12")
M = [(k, v(k)) for k in J if v(k) and v(k)["michel_found"]]
for K, L in ((0, 0), (3, 0), (5, 0), (0, 1.5), (0, 2.5), (3, 1.5), (5, 2.5)):
    tp = sum(1 for k, x in M if C.is_michel(J[k]) and x["michel_ke_best"] >= K and x["michel_len"] >= L); fp = sum(1 for k, x in M if not C.is_michel(J[k]) and x["michel_ke_best"] >= K and x["michel_len"] >= L)
    print("   ke>=%d len>=%.1f  TP %d FP %d" % (K, L, tp, fp))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1, default=str)
    print("wrote", a.json)
