#!/usr/bin/env python3
"""doc pdvd/70 sec 10 -- does the P1 C++ do exactly what the offline rule predicted?

Reads two prepped arms of the SAME binary and input: LEG (knob off) and ON
(topology_stop_evidence on, optionally topology_clears_sparse).  For every
candidate the prediction is the doc 70 rule applied to the LEG verdict:

    clear = (no_bragg | shape_flat [| profile_sparse]) & reject_bits
            if michel_found and conn_type in {1, 2} and KE >= ke and len >= len
    is_stm_pred = (reject_bits & ~clear) == 0

and the check is item by item: is_stm, reject_bits and topology_cleared_bits
against the prediction, and EVERY other verdict field bit-identical to LEG
(the rule is verdict-only and runs after everything else, so nothing else may
move).  Candidate SETS must be identical too.

Usage: d70_p1_check.py --leg PREP --on PREP [--sparse] [--ke 10] [--len 3]
"""
import argparse, glob, json, os, sys

BITS = {"no_bragg": 1 << 2, "shape_flat": 1 << 3, "profile_sparse": 1 << 9}
ap = argparse.ArgumentParser()
ap.add_argument("--leg", required=True)
ap.add_argument("--on", required=True)
ap.add_argument("--sparse", action="store_true")
ap.add_argument("--ke", type=float, default=10.0)
ap.add_argument("--len", type=float, default=3.0)
ap.add_argument("--off", action="store_true",
                help="ON is the knob-OFF arm of the new binary: the rule must never fire and the "
                     "topology_cleared_bits branch must be ABSENT on every item (conditional persist)")
a = ap.parse_args()


def load(prep):
    out = {}
    for fn in glob.glob(os.path.join(prep, "smprep-*.json")):
        ev, c = os.path.basename(fn)[7:-5].rsplit("-c", 1)
        with open(fn) as fh:
            out["%s/%s" % (ev, c)] = json.load(fh)["verdict"]
    return out


L, O = load(a.leg), load(a.on)
print("candidates: leg %d, on %d, only-leg %d, only-on %d" % (len(L), len(O), len(set(L) - set(O)), len(set(O) - set(L))))
bad = 0
if set(L) != set(O):
    bad += 1
    print("  *** candidate sets differ:", sorted(set(L) ^ set(O))[:20])

clearable = BITS["no_bragg"] | BITS["shape_flat"] | (BITS["profile_sparse"] if a.sparse else 0)
IGNORE = {"is_stm", "reject_bits", "reject_names", "topology_cleared_bits"}
fired, flipped, field_moves = [], [], []
for k in sorted(set(L) & set(O)):
    v, w = L[k], O[k]
    rb = int(v["reject_bits"])
    ok = (int(v["michel_found"]) == 1 and v["michel_conn_type"] in (1, 2)
          and (v["michel_ke_best"] or 0) >= a.ke and (v["michel_len"] or 0) >= a.len)
    clr = (rb & clearable) if (ok and not a.off) else 0
    pred_bits = rb & ~clr
    pred_stm = int(pred_bits == 0)
    if a.off:
        if "topology_cleared_bits" in w:
            bad += 1
            print("  *** %s: topology_cleared_bits written with the knob off" % k)
        got_clr = 0
    else:
        got_clr = int(w.get("topology_cleared_bits", -1))
    if clr:
        fired.append(k)
    if pred_stm and not int(v["is_stm"]):
        flipped.append(k)
    if int(w["reject_bits"]) != pred_bits or int(w["is_stm"]) != pred_stm or got_clr != clr:
        bad += 1
        print("  *** %s: predicted bits %d is_stm %d cleared %d | C++ bits %s is_stm %s cleared %s  (leg bits %d, KE %.3f, len %.3f, conn %s)"
              % (k, pred_bits, pred_stm, clr, w["reject_bits"], w["is_stm"], got_clr, rb, v["michel_ke_best"], v["michel_len"], v["michel_conn_type"]))
    for f in sorted((set(v) | set(w)) - IGNORE):
        if v.get(f) != w.get(f):
            field_moves.append((k, f, v.get(f), w.get(f)))

print("rule fires (clears >= 1 bit): %d; is_stm 0 -> 1: %d" % (len(fired), len(flipped)))
print("is_stm flips:", " ".join(flipped))
print("fires without a flip (another bit remains):", " ".join(k for k in fired if k not in flipped))
print("mismatches vs prediction: %d" % (bad - (1 if set(L) != set(O) else 0)))
print("other verdict fields moved: %d" % len(field_moves))
for m in field_moves[:20]:
    print("   ", m)
print("RESULT:", "PASS" if bad == 0 and not field_moves else "FAIL")
sys.exit(0 if bad == 0 and not field_moves else 1)
