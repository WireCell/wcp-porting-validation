#!/usr/bin/env python3
"""doc pr/139 item 1 -- the PRE-REGISTERED prediction for arm work-pr140r1-on-*.

Run and COMMITTED BEFORE the arm's census exists.  That ordering is the whole
point: sec 6 of the doc records a round whose recommendation was chosen after
seeing the movers, and the owner's scan then killed it.  This script fixes the
prediction in the tracked record first, from the owner's 2026-09-01 labels
alone, so the arm can only confirm or refute it.

Operating point: shower_split_skip_shared=1 + shower_split_max_impact=30 (cm).

Reads  docs/pr/pr139-scan-verdicts.tsv   (39 owner-labelled objects)
Writes docs/pr/pr140-prereg.tsv          (per-object predicted fate)

READ-ONLY over em_labels/ (M13).
"""
import csv, os, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(SX, "docs", "pr", "pr139-scan-verdicts.tsv")
OUT = os.path.join(SX, "docs", "pr", "pr140-prereg.tsv")
BOUND = 30.0

rows = list(csv.DictReader(open(SRC), delimiter="\t"))
out = []
for r in rows:
    ev, node = int(r["event"]), int(r["node"])
    confirmed = r["owner_verdict"] != "KEEP"
    fired = r["cxx_fired"] == "1"
    b = float(r["b_cm"])
    refused = r["p11_refuses"] == "1"
    if not fired:
        fate, why = "no-fire", "trigger does not fire today"
    elif refused:
        fate, why = "SUPPRESSED", "skip_shared refuses the peel (shared membership)"
    elif b > BOUND:
        fate, why = "SUPPRESSED", "impact b=%.2f > %.0f cm" % (b, BOUND)
    else:
        fate, why = "fires", "b=%.2f <= %.0f cm, not shared" % (b, BOUND)
    out.append(dict(event=ev, node=node, owner_verdict=r["owner_verdict"],
                    confirmed_cut=int(confirmed), b_cm="%.2f" % b,
                    skip_shared_refuses=int(refused), predicted=fate, reason=why))

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)

conf = [r for r in out if r["confirmed_cut"]]
fires = [r for r in out if r["predicted"] == "fires"]
sup_shared = [r for r in conf if r["skip_shared_refuses"] and r["predicted"] == "SUPPRESSED"]
sup_bound = [r for r in conf if not r["skip_shared_refuses"] and r["predicted"] == "SUPPRESSED"]
right = [r for r in fires if r["confirmed_cut"]]
wrong = [r for r in fires if not r["confirmed_cut"]]

print("PRE-REGISTERED prediction -- skip_shared + max_impact = %.0f cm" % BOUND)
print("  objects labelled            : %d  (%d confirmed cuts, %d KEEP)"
      % (len(out), len(conf), len(out) - len(conf)))
print("  predicted fires             : %d  (%d right, %d wrong)" % (len(fires), len(right), len(wrong)))
print("  trigger efficiency          : %.3f      purity %.3f"
      % (len(right) / len(conf), len(right) / len(fires) if fires else float("nan")))
print()
print("  confirmed cuts SUPPRESSED, TOTAL vs today's config : %d of %d"
      % (len(sup_shared) + len(sup_bound), len(conf)))
print("    by skip_shared (peel refused, cut DEFERRED) : %d  %s"
      % (len(sup_shared), [(r["event"], r["node"]) for r in sup_shared]))
print("    by the b bound  (cut REJECTED)              : %d  %s"
      % (len(sup_bound), [(r["event"], r["node"]) for r in sup_bound]))
print("  false fires left standing   : %s" % [(r["event"], r["b_cm"]) for r in wrong])
print()
# how close is any decision to the bound?  the C++ b and this offline b agree to
# 0.5 cm on 383/390 (pr139_tape_check), so a margin under ~1 cm is not safe.
marg = sorted(((abs(float(r["b_cm"]) - BOUND), r["event"], r["node"], r["b_cm"])
               for r in out if r["predicted"] != "no-fire"))
print("  closest decisions to the %.0f cm bound (offline-vs-C++ b agrees to 0.5 cm):" % BOUND)
for d, ev, nd, b in marg[:3]:
    print("    %7d/%-6d  b=%6s  margin %.2f cm" % (ev, nd, b, d))
