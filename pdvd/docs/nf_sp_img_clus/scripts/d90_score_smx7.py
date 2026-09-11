#!/usr/bin/env python3
"""doc pdvd/90 -- score the owner's smx7 blind re-judge against its key, fold it into a NEW merged
record (the old one is never rewritten, M13), and re-grade production and the flip arms on it.

Fork by duplication (CLAUDE.md M10) of d88_score_smx6.py; the tag, the record names and the
sections changed (smx7 asked no MECH line, so none is mapped).

smx7 = doc 89's part (b1), the 22 record stoppers production rejects on the dQ/dx shape tests alone
with no Michel object, on a medium / low agent verdict (doc 55 tranche 2), plus 8 record-THRU
controls with the same production reading; shuffled, blind (d90_build_smx7.py).

Merge rules (smx4's owner_row, doc 70, as in smx5 / smx6):
  * verdict / michel_kind / confidence "owner" / evidence = notes / source smx7;
  * tags: the record's, with any pf_segments override on top (none were set in smx7);
  * pin: the smx7 pin, EXCEPT that an unplaced smx7 pin never replaces a placed one
    (stm_michel_viewer's own no-downgrade rule); a scanner pin_rr on the old row is carried
    with pin_rr_source.

Usage: STM_SCAN_RECORD=<smx1a..smx6 record> d90_score_smx7.py --labels L.json --key K.tsv
           --arm LABEL:PREP [--arm ...] [--write-merged OUT]
"""
import argparse, collections, csv, json, os, sys
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--arm", action="append", default=[], help="LABEL:PREP, graded on the old and the new record")
ap.add_argument("--write-merged", default=None)
a = ap.parse_args()
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx1a_smx3_smx4_smx5_smx6_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the smx1a+smx3+smx4+smx5+smx6 record (doc 88 sec 9.2)")
if a.write_merged and os.path.exists(a.write_merged):
    sys.exit("REFUSING: %s exists" % a.write_merged)

rec0 = C.load_record()
L = json.load(open(a.labels))["labels"]
K = {r["key"]: r for r in csv.DictReader([l for l in open(a.key) if not l.startswith("#")], delimiter="\t")}
assert set(L) == set(K), "labels and key differ: %s" % sorted(set(L) ^ set(K))


def load_prep(d):
    P = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("smprep-") and f.endswith(".json"):
            P[f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(d, f)))["verdict"]
    return P


ARMS = [(s.split(":", 1)[0], load_prep(s.split(":", 1)[1])) for s in a.arm]
PROD = ARMS[0][1] if ARMS else {}

print("=== 1. the owner's verdict against the record, by group ===")
cls = lambda v: "THRU" if C.bare(v) == "THRU" else ("STM" if C.bare(v) in ("STM_MICHEL", "STM_ONLY") else v)
for grp in ("b1", "control"):
    ks = sorted((k for k in K if K[k]["group"] == grp), key=lambda t: int(K[t]["scan_id"]))
    t = collections.Counter((C.bare(K[k]["record_verdict"]), L[k]["choice"]) for k in ks)
    print("  -- %s (%d): record -> owner %s" % (grp, len(ks), dict(sorted(t.items()))))
    held = sum(1 for k in ks if cls(K[k]["record_verdict"]) == cls(L[k]["choice"]))
    print("     stopper-vs-THRU call held on %d of %d; overturned: %s" % (held, len(ks), " ".join(
        "%s(%s->%s)" % (k, K[k]["record_verdict"], L[k]["choice"]) for k in ks if cls(K[k]["record_verdict"]) != cls(L[k]["choice"])) or "none"))
    mk = sum(1 for k in ks if C.bare(K[k]["record_verdict"]) == "STM_MICHEL"); mo = sum(1 for k in ks if L[k]["choice"] == "STM_MICHEL")
    print("     Michel at the stop: record %d, owner %d" % (mk, mo))

print("\n=== 2. every item: the owner's call, pin, and production's reading at the stop ===")
print("  %-2s %-14s %-7s %-16s %-6s | %-10s %-13s %-17s | %s" % ("id", "item", "group", "record", "conf", "owner", "michel", "owner pin", "production (p88vprod)"))
for k in sorted(K, key=lambda t: int(K[t]["scan_id"])):
    v, kr, x = L[k], K[k], PROD.get(k, {})
    pin = v.get("pin") or {}
    ps = "rr %.1f (moved %.1f)" % (pin["rr"], pin.get("moved_cm", 0)) if pin.get("placed") else "fit end"
    prod = ("bits %s | anchor %.1f cm, agent pin_rr %s | stop arms %d other %d, body other %d, pieces %d, dots %d (%.1f MeV unfit), stop_dis %.1f" % (
        "+".join(x["reject_names"]), x["bragg_anchor_shift_cm"], kr["record_pin_rr"] or "-", x["n_stop_arms"], x["n_stop_other"],
        x["n_body_other"], x["n_local_pieces"], x["n_dots"], x["dots_ke_unfit"] or 0, x["stop_dis"])) if x else "no payload"
    print("  %-2s %-14s %-7s %-16s %-6s | %-10s %-13s %-17s | %s" % (kr["scan_id"], k, kr["group"], kr["record_verdict"], kr["record_confidence"],
          v["choice"], v.get("michel_kind"), ps, prod))
pins = [(k, L[k]["pin"]) for k in K if (L[k].get("pin") or {}).get("placed")]
print("  owner moved the pin on %d items: %s" % (len(pins), " ".join("%s rr %.1f (anchor %.1f)" % (
    k, p["rr"], PROD[k]["bragg_anchor_shift_cm"] if k in PROD else -1) for k, p in sorted(pins))))


def owner_row(k, v):
    old = rec0[k]
    tags = dict(old.get("tags") or {})
    tags.update(v.get("pf_segments") or {})
    pin = v.get("pin")
    kept = False
    if not (pin or {}).get("placed") and (old.get("pin") or {}).get("placed"):
        pin, kept = old["pin"], True
    row = dict(key=k, verdict=v["choice"], michel_kind=v.get("michel_kind"), confidence="owner",
               tranche=old.get("tranche"), scan_id=old.get("scan_id"), evidence=(v.get("notes") or "").strip(),
               tags=tags, source="smx7 (owner, 2026-09-11)", pin=pin)
    if old.get("pin_rr") is not None:
        row["pin_rr"] = old["pin_rr"]
        row["pin_rr_source"] = old.get("source", "smx1a")
    return row, kept


print("\n=== 3. the fold, field by field ===")
rec1 = dict(rec0)
for k in sorted(L, key=lambda t: int(K[t]["scan_id"])):
    row, kept = owner_row(k, L[k])
    old = rec0[k]
    diff = ["%s %s -> %s" % (f, old.get(f), row.get(f)) for f in ("verdict", "michel_kind", "confidence") if old.get(f) != row.get(f)]
    print("  %-14s %s%s" % (k, "; ".join(diff), "; the old placed pin kept" if kept else ""))
    rec1[k] = row
moved = [k for k in rec0 if (C.is_stopper(rec1[k]), C.is_michel(rec1[k]), C.judged(rec1[k])) !=
         (C.is_stopper(rec0[k]), C.is_michel(rec0[k]), C.judged(rec0[k]))]
print("  stopper / Michel / judged class changed on %d of %d records: %s" % (len(moved), len(rec0), " ".join(moved)))


def census(R, P, keys):
    J = [k for k in keys if C.judged(R[k])]
    def rates(pred, truth):
        tp = sum(1 for k in J if pred(k) and truth(R[k])); fp = sum(1 for k in J if pred(k) and not truth(R[k]))
        fn = sum(1 for k in J if not pred(k) and truth(R[k]))
        pu = tp / max(tp + fp, 1); ef = tp / max(tp + fn, 1)
        return "%3d / %2d / %3d  pur %.3f eff %.3f F1 %.3f" % (tp, fp, fn, pu, ef, 2 * pu * ef / max(pu + ef, 1e-9))
    stm = lambda k: k in P and int(P[k]["is_stm"]) == 1
    mf = lambda k: k in P and int(P[k]["michel_found"]) == 1
    return len(J), rates(stm, C.is_stopper), rates(mf, C.is_michel)


print("\n=== 4. the census before and after the fold (is_stm, michel_found: TP / FP / FN) ===")
for lab, P in ARMS:
    for rn, R in (("smx6 record", rec0), ("smx7 record", rec1)):
        for pop, keys in (("all judged", list(R)), ("with a candidate", [k for k in R if k in P])):
            n, s, m = census(R, P, keys)
            print("  %-9s %-11s %-16s n %3d | is_stm %s | michel_found %s" % (lab, rn, pop, n, s, m))
if a.write_merged:
    with open(a.write_merged, "w") as fh:
        json.dump([rec1[k] for k in sorted(rec1)], fh, indent=1)
    print("\nwrote %d merged records -> %s (%d from smx7)" % (len(rec1), a.write_merged, len(L)))
