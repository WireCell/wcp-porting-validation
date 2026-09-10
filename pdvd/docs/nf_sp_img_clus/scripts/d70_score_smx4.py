#!/usr/bin/env python3
"""doc pdvd/70 sec 9 -- fold the owner's smx4 scan (proposal P1's blind re-judge)
into the grading record and score P1 on it.

P1 (`topology_stop_evidence`) is scored by EXACT offline re-verdict: on a candidate
with michel_found 1, michel_conn_type 1 or 2, michel_ke_best >= KE and michel_len
>= LEN, clear no_bragg + shape_flat; is_stm becomes 1 only if NO other reject bit
remains.  Both bits are single-consumer (doc 67), so this is what the C++ will do.
Variant "+sparse" also clears profile_sparse (an owner's option, sec 9.4).

The merged record is the doc 68 record (smx1a + smx3) with the owner's smx4 verdict
REPLACING the earlier one on all 54 items (four of them were owner smx3 verdicts;
smx4 is the later look).  Nothing is written over an existing file (M13).

Usage: STM_SCAN_RECORD=<pdvd_stm_michel_smx1a_smx3_verdicts.json> \\
       d70_score_smx4.py --labels L.json --key K.tsv --prep PREP [--write-merged OUT]
"""
import argparse, collections, glob, json, os, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
for k in ("labels", "key", "prep"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--smx3-labels", default="/home/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/scan/"
                "pdvd_stm_michel_smx3_labels.json")
ap.add_argument("--write-merged", default=None)
a = ap.parse_args()
if not os.environ.get("STM_SCAN_RECORD", "").endswith("pdvd_stm_michel_smx1a_smx3_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the doc 68 merged record")
if a.write_merged and os.path.exists(a.write_merged):
    sys.exit("REFUSING: %s exists" % a.write_merged)

rec0 = C.load_record()
L = json.load(open(a.labels))["labels"]
L3 = json.load(open(a.smx3_labels))["labels"]
KEY = {}
hdr = None
for line in open(a.key):
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if hdr is None:
        hdr = f
        continue
    KEY[f[1]] = dict(zip(hdr, f))
assert set(L) == set(KEY) and all(L[k].get("choice") for k in L), "not every smx4 item is labelled"

V = {}
for fn in glob.glob(os.path.join(a.prep, "smprep-*.json")):
    ev, c = os.path.basename(fn)[7:-5].rsplit("-c", 1)
    with open(fn) as fh:
        V["%s/%s" % (ev, c)] = json.load(fh)["verdict"]


def owner_row(k, v):
    # pf_segments holds the scanner's OVERRIDES only (stm_michel_viewer.py), so the
    # owner's tags go ON TOP of the record's: replacing would drop every earlier
    # tag the owner did not touch (8 items, 1-9 tags each).
    old = rec0[k]
    tags = dict(old.get("tags") or {})
    tags.update(v.get("pf_segments") or {})
    return dict(key=k, verdict=v["choice"], michel_kind=v.get("michel_kind"), confidence="owner",
                tranche=old.get("tranche"), scan_id=old.get("scan_id"), evidence=v.get("notes") or "",
                tags=tags, source="smx4 (owner, 2026-09-10)", pin=v.get("pin"))


rec1 = {k: dict(r, source=r.get("source", "smx1a")) for k, r in rec0.items()}
for k, v in L.items():
    rec1[k] = owner_row(k, v)

SHAPE = {"no_bragg", "shape_flat"}


def residual(v, sparse=False):
    return set(C.reject_names(v)) - SHAPE - ({"profile_sparse"} if sparse else set())


def p1(v, ke, ln, sparse=False, exact=True):
    if int(v["is_stm"]):
        return 1
    ok = (int(v["michel_found"]) == 1 and v["michel_conn_type"] in (1, 2)
          and (v["michel_ke_best"] or 0) >= ke and (v["michel_len"] or 0) >= ln)
    if exact:
        return int(ok and not residual(v, sparse))
    # doc 70's first sizing (no residual-bit test; only the boundary clause)
    return int(ok and "stop_near_boundary" not in C.reject_names(v))


def bin_(r):
    return "stopper" if C.is_stopper(r) else ("THRU" if C.judged(r) else r["verdict"])


# ---- 1. owner vs the record -------------------------------------------------
print("=== 1. the owner (smx4) against the record it re-judged, by key group ===")
tab = collections.Counter()
changed = collections.defaultdict(list)
for k in KEY:
    g = KEY[k]["group"]
    o = bin_(rec1[k])
    tab[(g, o)] += 1
    if o != bin_(rec0[k]):
        changed[(g, bin_(rec0[k]), o)].append(k)
for g in ("rec_stopper", "rec_thru"):
    n = sum(v for (gg, _), v in tab.items() if gg == g)
    print("  %-11s n=%2d  owner: %s" % (g, n, ", ".join("%s %d" % (o, tab[(g, o)]) for o in ("stopper", "THRU", "MESSY", "UNCLEAR") if tab[(g, o)])))
for (g, was, now), ks in sorted(changed.items()):
    print("  %-11s %-7s -> %-7s %2d: %s" % (g, was, now, len(ks), " ".join(sorted(ks))))
both = [k for k in KEY if C.judged(rec1[k])]
agree = sum(C.is_stopper(rec1[k]) == C.is_stopper(rec0[k]) for k in both)
print("  stopper/not agreement on the %d items the owner judged: %d" % (len(both), agree))

# ---- 2. the four re-judges of smx3 -------------------------------------------
print("\n=== 2. the owner's four repeat looks (smx3 -> smx4) ===")
for k in sorted(k for k in KEY if KEY[k]["rejudge"] == "1"):
    s3, s4 = L3[k], L[k]
    print("  %-14s smx3 %-10s %-8s | smx4 %-10s %-8s | %s" % (k, s3["choice"], s3.get("michel_kind"), s4["choice"],
                                                          s4.get("michel_kind"),
                                                          "SAME" if (s3["choice"], s3.get("michel_kind")) == (s4["choice"], s4.get("michel_kind"))
                                                          else "stopper/not same" if C.is_stopper(dict(verdict=s3["choice"])) == C.is_stopper(dict(verdict=s4["choice"])) else "CHANGED"))

# ---- 3. P1 at the argued point, item by item ---------------------------------
print("\n=== 3. P1 at KE >= 10 MeV, len >= 3 cm on the 54, against the owner ===")
for nm, kw in (("doc 70 first sizing (no residual-bit test)", dict(exact=False)), ("EXACT", dict(exact=True)),
               ("EXACT +sparse", dict(exact=True, sparse=True))):
    fired = [k for k in KEY if p1(V[k], 10, 3, **kw)]
    t = collections.Counter(bin_(rec1[k]) for k in fired)
    r = collections.Counter(KEY[k]["group"] for k in fired)
    print("  %-44s fires on %2d (record: %s)  owner: %s" % (nm, len(fired), dict(r), dict(t)))
fired = [k for k in KEY if p1(V[k], 10, 3)]
first = [k for k in KEY if p1(V[k], 10, 3, exact=False)]
print("  first sizing only (another bit stays set):", " ".join("%s(%s)" % (k, ",".join(sorted(residual(V[k])))) for k in sorted(set(first) - set(fired))))
for o in ("stopper", "THRU", "MESSY", "UNCLEAR"):
    ks = sorted(k for k in fired if bin_(rec1[k]) == o)
    if ks:
        print("  P1 fires, owner %-7s %2d: %s" % (o, len(ks), " ".join("%s[%s]" % (k, rec1[k]["verdict"]) for k in ks)))
nf = sorted(k for k in KEY if not p1(V[k], 10, 3) and C.is_stopper(rec1[k]))
print("  owner stoppers P1 does NOT recover (%d):" % len(nf))
for k in nf:
    v = V[k]
    print("     %-14s %-10s KE %5.1f MeV len %4.1f cm conn %d  bits %s" % (k, rec1[k]["verdict"], v["michel_ke_best"], v["michel_len"],
                                                                        v["michel_conn_type"], ",".join(C.reject_names(v))))
nt = sorted(k for k in KEY if bin_(rec1[k]) == "THRU")
print("  owner THRU on the 54 (%d):" % len(nt), " ".join("%s(KE %.1f, %.1f cm)" % (k, V[k]["michel_ke_best"], V[k]["michel_len"]) for k in nt))

# ---- 4. census ----------------------------------------------------------------
def census(rec, fn, pop):
    J = sorted(k for k in rec if C.judged(rec[k]) and (pop == "union" or k in V))
    s = {k: (fn(V[k]) if k in V else 0) for k in J}
    t = {k: C.is_stopper(rec[k]) for k in J}
    tp = sum(s[k] and t[k] for k in J); fp = sum(s[k] and not t[k] for k in J); fn_ = sum((not s[k]) and t[k] for k in J)
    tn = len(J) - tp - fp - fn_
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn_, 1)
    return tp, fp, fn_, tn, pu, ef, 2 * pu * ef / max(pu + ef, 1e-9)


def mcensus(rec, pop):
    J = sorted(k for k in rec if C.judged(rec[k]) and (pop == "union" or k in V))
    s = {k: (int(V[k]["michel_found"]) if k in V else 0) for k in J}
    t = {k: C.is_michel(rec[k]) for k in J}
    tp = sum(s[k] and t[k] for k in J); fp = sum(s[k] and not t[k] for k in J); fn_ = sum((not s[k]) and t[k] for k in J)
    pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn_, 1)
    return tp, fp, fn_, len(J) - tp - fp - fn_, pu, ef, 2 * pu * ef / max(pu + ef, 1e-9)


ROWS = [("production (d68a3)", lambda v: int(v["is_stm"]))]
for ke, ln in ((10, 3), (5, 3), (10, 2.5), (5, 2.5)):
    ROWS.append(("P1 %g MeV / %g cm" % (ke, ln), (lambda ke, ln: lambda v: p1(v, ke, ln))(ke, ln)))
ROWS.append(("P1 10 MeV / 3 cm +sparse", lambda v: p1(v, 10, 3, sparse=True)))
ROWS.append(("[doc 70 first sizing 10/3]", lambda v: p1(v, 10, 3, exact=False)))
for rn, rec in (("doc 68 record (smx1a+smx3)", rec0), ("NEW record (+smx4)", rec1)):
    for pop in ("payload", "union"):
        print("\n=== 4. is_stm census, %s, %s population ===" % (rn, pop))
        print("  %-28s %4s %4s %4s %4s %7s %7s %7s" % ("", "TP", "FP", "FN", "TN", "purity", "eff", "F1"))
        for nm, fn in ROWS:
            print("  %-28s %4d %4d %4d %4d %7.3f %7.3f %7.3f" % ((nm,) + census(rec, fn, pop)))
        print("  %-28s %4d %4d %4d %4d %7.3f %7.3f %7.3f" % (("michel_found (production)",) + mcensus(rec, pop)))

# ---- 5. the P1 grid on the new record ----------------------------------------
print("\n=== 5. P1 grid (exact), NEW record, payload population: new TP / new FP over production ===")
b = census(rec1, ROWS[0][1], "payload")
lens = (0, 2, 2.5, 3, 4, 5)
print("  KE \\ len " + "".join("%10s" % l for l in lens))
for ke in (0, 3, 5, 8, 10, 12, 15):
    cells = []
    for ln in lens:
        c = census(rec1, (lambda ke, ln: lambda v: p1(v, ke, ln))(ke, ln), "payload")
        cells.append("%+4d/%+d" % (c[0] - b[0], c[1] - b[1]))
    print("  %-9g " % ke + "".join("%10s" % x for x in cells))

# ---- 6. michel_found on the 54, and right-verdict-wrong-reason ---------------
print("\n=== 6. michel_found = 1 on all 54: what the owner says the object is ===")
print("  owner label x michel_kind:", dict(collections.Counter((L[k]["choice"], L[k].get("michel_kind")) for k in KEY)))
wr = sorted(k for k in fired if C.is_stopper(rec1[k]) and not C.is_michel(rec1[k]))
print("  P1 recovers the stopper through an object the owner does not call a Michel (%d): %s" %
      (len(wr), " ".join("%s[%s, kind %s, conn %d]" % (k, rec1[k]["verdict"], L[k].get("michel_kind"), V[k]["michel_conn_type"]) for k in wr)))

# ---- 7. notes and pins --------------------------------------------------------
print("\n=== 7. the owner's notes and placed pins ===")
for k in sorted(KEY, key=lambda k: int(KEY[k]["scan_id"])):
    v = L[k]
    pin = v.get("pin") or {}
    if v.get("notes") or pin.get("placed"):
        print("  #%-2s %-14s %-11s %s%s" % (KEY[k]["scan_id"], k, v["choice"],
                                          ("pin moved %.1f cm%s  " % (pin.get("moved_cm") or 0, " OFF-fit" if pin.get("off_fit") else "")) if pin.get("placed") else "",
                                          ("\"%s\"" % v["notes"].strip()) if v.get("notes") else ""))

if a.write_merged:
    out = [rec1[k] for k in sorted(rec1)]
    with open(a.write_merged, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote %d merged records -> %s (%d from smx4)" % (len(out), a.write_merged, len(L)))
