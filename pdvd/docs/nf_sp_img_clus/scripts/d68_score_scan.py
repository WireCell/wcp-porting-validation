#!/usr/bin/env python3
"""doc pdvd/68 -- fold the owner's option scan (tag smx3) into the grading record
and score production and the two options on it.

The merged record is smx1a (doc 55's 569 items) with the owner's smx3 verdict
REPLACING smx1a's on every item the owner re-judged, plus the smx3 items smx1a
never had.  smx1a itself is never written (M13); --write-merged puts the merged
record in a NEW file.

Population: every judged item of the merged record.  An item an arm has no
candidate for counts as is_stm = 0 / michel_found = 0 for that arm -- the only
fair way to compare arms whose candidate SETS differ (dx_norm_length moves it).

Usage: d68_score_scan.py --labels L.json --questions Q.json --prod P --anchor P --dx P [--write-merged OUT]
"""
import argparse, collections, glob, json, os, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
for k in ("labels", "questions", "prod", "anchor", "dx"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--write-merged", default=None)
a = ap.parse_args()

smx1a = C.load_record()
L = json.load(open(a.labels))["labels"]
Q = json.load(open(a.questions))["items"]


def owner_row(k, v):
    # census_score.py splits by tranche: an item smx1a had keeps its tranche and
    # scan_id; an item new to the record goes in tranche 3 (the option scan).
    old = smx1a.get(k) or {}
    return dict(key=k, verdict=v["choice"], michel_kind=v.get("michel_kind"), confidence="owner",
                tranche=old.get("tranche", 3), scan_id=old.get("scan_id", 1000 + int(v.get("scan_id", 0))),
                evidence=v.get("notes") or "", tags=dict(v.get("pf_segments") or {}) or
                dict((smx1a.get(k) or {}).get("tags") or {}), source="smx3 (owner, 2026-09-10)",
                pin=v.get("pin"))


merged = {k: dict(r, source=r.get("source", "smx1a")) for k, r in smx1a.items()}
for k, v in L.items():
    merged[k] = owner_row(k, v)


def load(prep):
    out = {}
    for f in glob.glob(os.path.join(prep, "smprep-*.json")):
        ev, c = os.path.basename(f)[7:-5].rsplit("-c", 1)
        with open(f) as fh:
            out["%s/%s" % (ev, c)] = json.load(fh)["verdict"]
    return out


ARM = {"production": load(a.prod), "anchor 3 cm": load(a.anchor), "dx 4 mm": load(a.dx)}
J = sorted(k for k in merged if C.judged(merged[k]))
STOP = {k: C.is_stopper(merged[k]) for k in J}
MICH = {k: C.is_michel(merged[k]) for k in J}


def val(arm, k, f):
    v = ARM[arm].get(k)
    return int(v[f]) if v is not None else 0


def census(arm, f, truth):
    s = {k: val(arm, k, f) for k in J}
    tp = sum(s[k] and truth[k] for k in J); fp = sum(s[k] and not truth[k] for k in J)
    fn = sum((not s[k]) and truth[k] for k in J)
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1)
    return tp, fp, fn, pu, ef, 2 * pu * ef / max(pu + ef, 1e-9), s


# ---- 1. the owner against smx1a on the re-judged record items -------------
print("=== 1. owner (smx3) vs smx1a on the %d re-judged record items ===" % sum(k in smx1a for k in L))
agree = collections.Counter()
for k in sorted(L, key=lambda k: (Q[k]["group"], k)):
    if k not in smx1a:
        continue
    o, s = L[k]["choice"], smx1a[k]["verdict"]
    same_bin = C.is_stopper(dict(verdict=o)) == C.is_stopper(dict(verdict=s)) and C.judged(dict(verdict=o))
    agree[same_bin] += 1
    print("  g%d %-14s smx1a %-11s (%s)  owner %-11s %s" % (Q[k]["group"], k, s, smx1a[k]["confidence"], o,
                                                          "" if same_bin else "<- CHANGED"))
print("  stopper/not agreement: %d of %d" % (agree[True], sum(agree.values())))

# ---- 2. per group: who was right ----------------------------------------
print("\n=== 2. per item: production vs the option, against the owner ===")
for g, arm in ((1, "anchor 3 cm"), (2, "dx 4 mm"), (3, "dx 4 mm")):
    tally = collections.Counter()
    print("-- group %d (%s)" % (g, arm))
    for k in sorted(k for k in Q if Q[k]["group"] == g):
        o = L[k]["choice"]
        if not C.judged(dict(verdict=o)):
            print("   %-14s owner %-11s  (not judged)" % (k, o)); tally["unjudged"] += 1; continue
        t = C.is_stopper(dict(verdict=o)); p = val("production", k, "is_stm"); x = val(arm, k, "is_stm")
        who = "both right" if p == x == t else "OPTION right" if x == t else "PRODUCTION right" if p == t else "both wrong"
        tally[who] += 1
        mp, mx = val("production", k, "michel_found"), val(arm, k, "michel_found")
        print("   %-14s owner %-11s  prod is_stm %d mf %d | option is_stm %d mf %d  -> %s%s" %
              (k, o, p, mp, x, mx, who, "" if k in ARM["production"] else "  [no production candidate]"))
    print("   tally:", dict(tally))

# ---- 3. the census on the merged record -----------------------------------
print("\n=== 3. census on the merged record (%d judged items; an arm with no candidate reads 0) ===" % len(J))
res = {}
for arm in ARM:
    r1 = census(arm, "is_stm", STOP); r2 = census(arm, "michel_found", MICH); res[arm] = (r1, r2)
    print("  %-11s is_stm TP %3d FP %2d FN %3d pur %.3f eff %.3f F1 %.3f | michel TP %3d FP %2d FN %3d pur %.3f eff %.3f F1 %.3f"
          % ((arm,) + r1[:6] + r2[:6]))
print("\n  by name vs production:")
for arm in ("anchor 3 cm", "dx 4 mm"):
    for nm, idx, truth in (("is_stm", 0, STOP), ("michel_found", 1, MICH)):
        s0, s1 = res["production"][idx][6], res[arm][idx][6]
        gain = [k for k in J if s1[k] and not s0[k] and truth[k]]; lost = [k for k in J if s0[k] and not s1[k] and truth[k]]
        nfp = [k for k in J if s1[k] and not s0[k] and not truth[k]]; rfp = [k for k in J if s0[k] and not s1[k] and not truth[k]]
        print("  %-11s %-12s +%d TP, -%d TP, +%d FP, -%d FP" % (arm, nm, len(gain), len(lost), len(nfp), len(rfp)))
        for lab, lst in (("TP gained", gain), ("TP lost", lost), ("FP new", nfp), ("FP removed", rfp)):
            if lst:
                print("      %-10s %s" % (lab, " ".join(lst)))

# ---- 4. the same census on smx1a alone, for reference -------------------
print("\n=== 4. reference: the same arms on smx1a alone (no owner verdicts) ===")
J0 = sorted(k for k in smx1a if C.judged(smx1a[k]))
for arm in ARM:
    s = {k: val(arm, k, "is_stm") for k in J0}; t = {k: C.is_stopper(smx1a[k]) for k in J0}
    tp = sum(s[k] and t[k] for k in J0); fp = sum(s[k] and not t[k] for k in J0); fn = sum((not s[k]) and t[k] for k in J0)
    print("  %-11s is_stm TP %3d FP %2d FN %3d" % (arm, tp, fp, fn))

if a.write_merged:
    out = [merged[k] for k in sorted(merged)]
    with open(a.write_merged, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote %d merged records -> %s (%d from smx3)" % (len(out), a.write_merged, len(L)))
