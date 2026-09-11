#!/usr/bin/env python3
"""doc pdvd/92 -- what is left in PDVD STM + Michel after the wide Bragg-peak read, and whether it is
worth another round.  Read-only.

The owner asked, at the end of this round, whether there is still enough room to improve the STM +
Michel PR or whether it is time to pause.  This answers it as a CEILING in one denominator rather
than as a list of ideas: take the record, apply the rule at its selected operating point, and split
every remaining miss by who can still act on it --
  (a) closed by the owner's own preference (fiducial / continuation / hadron guard: doc 70 sec 3.3,
      doc 77 sec 7.5).  Not a defect; the chain is doing what the owner asked.
  (b) the tagger's, before CheckSTM_Michel ever sees a candidate (doc 89 sec 3).
  (c) chain-side WITH a named lever -- something a knob or a rule could still reach.
  (d) chain-side with NO lever identified: the chain has the candidate, rejects it, and nothing in
      the recorded inputs separates it from a true through-going track.
(c) is the headroom.  (a), (b) and (d) are not, on this record, reachable by more rounds of the same
kind of work.

The same is done for the Michel side (michel_found), which is the other half of the PR.

Usage: STM_SCAN_RECORD=<smx1a..smx7 record> d92_headroom.py --prep DIR --twin JSON [--arm NAME]
"""
import argparse, collections, json, os, sys

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
# Either the smx7 record (doc 90) or the smx8 one this round wrote (d92_score_smx8.py).  The smx8
# record is the honest denominator AFTER the re-judge, but it carries a sampling bias the smx7 one
# does not: only boundary items were re-examined, so its corrections sit exactly where the rule acts.
# Run it on BOTH and quote both, as section 6 of doc 92 does.
_REC = os.environ.get("STM_SCAN_RECORD", "")
if not (_REC.endswith("smx1a_smx3_smx4_smx5_smx6_smx7_verdicts.json")
        or _REC.endswith("smx1a_smx3_smx4_smx5_smx6_smx7_smx8_verdicts.json")):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a..smx7 (doc 90) or smx1a..smx8 (doc 92) record")
print("record: %s" % os.path.basename(_REC))
sys.path.insert(0, IMG + "/pdhd/stm_michel_scan")
import census_lib as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--twin", required=True)
ap.add_argument("--arm", default="p92v13", help="which predicted operating point to credit")
a = ap.parse_args()

R = C.load_record()
J = [k for k in R if C.judged(R[k])]
V = {}
for f in sorted(os.listdir(a.prep)):
    if f.startswith("smprep-") and f.endswith(".json"):
        V[f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(a.prep, f)))["verdict"]
TW = json.load(open(a.twin))
MOV = set(TW["arms"][a.arm]["movers"])

SHAPE = {"no_bragg", "shape_flat", "plateau_off_mip", "profile_sparse"}
PREF = {"stop_near_boundary", "continuation", "vertex_hadron"}   # the owner's closed classes


def stm_after(k):
    return (bool(V[k]["is_stm"]) if k in V else False) or k in MOV


print("=== the record and the two ledgers, after the rule at %s (W %.1f R %.1f D %.1f) ===" % (
    a.arm, TW["arms"][a.arm]["W"], TW["arms"][a.arm]["R"], TW["arms"][a.arm]["D"]))
tp = sum(1 for k in J if stm_after(k) and C.is_stopper(R[k]))
fp = sum(1 for k in J if stm_after(k) and not C.is_stopper(R[k]))
fn = sum(1 for k in J if not stm_after(k) and C.is_stopper(R[k]))
print("  is_stm on the %d judged: %d / %d / %d  (eff %.3f, purity %.3f)" % (len(J), tp, fp, fn, tp / (tp + fn), tp / (tp + fp)))
mtp = sum(1 for k in J if k in V and V[k]["michel_found"] and C.is_michel(R[k]))
mfp = sum(1 for k in J if k in V and V[k]["michel_found"] and not C.is_michel(R[k]))
mfn = sum(1 for k in J if C.is_michel(R[k]) and not (k in V and V[k]["michel_found"]))
print("  michel_found          : %d / %d / %d  (eff %.3f, purity %.3f)" % (mtp, mfp, mfn, mtp / (mtp + mfn), mtp / (mtp + mfp)))

# ---------------------------------------------------------------- the stopper misses
miss = sorted(k for k in J if C.is_stopper(R[k]) and not stm_after(k))
print("\n=== the %d remaining missed stoppers, by who can still act ===" % len(miss))
buckets = collections.defaultdict(list)
detail = {}
for k in miss:
    if k not in V:
        buckets["(b) the tagger's: no candidate reaches CheckSTM_Michel"].append(k); detail[k] = "no candidate"
        continue
    v = V[k]
    rn = set(C.reject_names(v))
    detail[k] = "+".join(sorted(rn))
    if rn & PREF:
        buckets["(a) closed by the owner's fiducial / continuation / hadron preference"].append(k)
    elif rn <= SHAPE:
        if v["michel_found"]:
            buckets["(c) chain-side, lever: a Michel object exists -- P1's floors"].append(k)
        else:
            buckets["(d) chain-side, no lever: shape-only, no Michel object, charge says nothing"].append(k)
    else:
        buckets["(c) chain-side, lever: a named non-shape bit"].append(k)
TH = {k: v[0] for k, v in TW["thresholds"].items()}
KE = TH["topology_michel_ke_min"]; LEN = TH["topology_michel_len_min_cm"]
DECISION = set(TW["decision"])


def extra(k):
    """why the named lever does not already fire, and whether loosening would reach the item"""
    if k not in V:
        return ""
    v = V[k]; out = []
    if v.get("michel_found"):
        why = []
        if int(v.get("michel_conn_type") or 0) not in (1, 2):
            why.append("conn_type %s" % v.get("michel_conn_type"))
        if float(v.get("michel_ke_best") or 0) < KE:
            why.append("KE %.1f < %.1f MeV" % (float(v.get("michel_ke_best") or 0), KE))
        if float(v.get("michel_len") or 0) < LEN:
            why.append("len %.1f < %.1f cm" % (float(v.get("michel_len") or 0), LEN))
        out.append("P1 misses on " + ", ".join(why) if why else
                   "P1 passes: a non-shape bit stands")
    if k in DECISION:
        out.append("a looser W/R/D point would move it, at a judged-THRU cost")
    return ("  | " + "; ".join(out)) if out else ""


for name in sorted(buckets):
    ks = buckets[name]
    conf = collections.Counter(R[k].get("confidence") for k in ks)
    print("  %-62s %2d   %s" % (name, len(ks), dict(conf)))
    for k in ks:
        print("      %-14s %-9s %-28s%s" % (k, R[k].get("confidence"), detail[k], extra(k)))

head = sum(len(v) for kk, v in buckets.items() if kk.startswith("(c)"))
print("\n  headroom (c) = %d of %d misses = %.1f%% of the %d judged stoppers; the ceiling if EVERY (c)"
      % (head, len(miss), 100.0 * head / (tp + fn), tp + fn))
print("  item were recovered at no cost is eff %.3f -> %.3f" % (tp / (tp + fn), (tp + head) / (tp + fn)))

# how much of what is left is a re-judge question rather than an algorithm question
low = [k for k in miss if R[k].get("confidence") in ("medium", "low")]
print("  of the %d misses, %d rest on a single medium/low-confidence agent call (a re-judge question,"
      % (len(miss), len(low)))
print("  not an algorithm question); %d are owner-judged." % (len(miss) - len(low)))

# ---------------------------------------------------------------- the Michel misses
mm = sorted(k for k in J if C.is_michel(R[k]) and not (k in V and V[k]["michel_found"]))
print("\n=== the %d remaining missed Michels, by who can still act ===" % len(mm))
mb = collections.defaultdict(list)
for k in mm:
    if k not in V:
        mb["(b) the tagger's: no candidate"].append(k); continue
    v = V[k]
    if not stm_after(k):
        mb["(e) the stopper call is missed first -- no Michel is looked for"].append(k)
    else:
        mb["(c) the chain calls the stop but finds no Michel (doc 90 sec 8 item 2)"].append(k)
for name in sorted(mb):
    ks = mb[name]
    tag = lambda k: k + ("*" if k in MOV else "")
    print("  %-62s %2d   %s" % (name, len(ks), " ".join(tag(k) for k in ks)))
print("  * = this round's rule is what makes the stop call, so the Michel is the very next step")
print("\nNote: eff/purity here are on the judged record only, and 470 of its %d rows are single-scanner"
      % len(R))
print("agent calls (smx1a).  Purity away from this record is not measured by anything above.")
