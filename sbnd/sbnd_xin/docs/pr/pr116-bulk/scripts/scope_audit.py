#!/usr/bin/env python3
"""How much charge does the scan's own scope rule drop?

The capture puts a shower in scope on `kine_best >= 20 MeV or pio_id >= 0`, but
correction 17 orders the verdict shower by `kine_charge`.  The two columns
disagree, so a shower can be dropped for kine_best 14.2 while carrying
kine_charge 29.9 (agent 1, evt500083).  Anyone reading "sub-20 MeV skipped" as
negligible would understate the charge left unlooked-at.  This measures it.
"""
import glob, json, os
tot_ev = tot_skip = 0
worst = []
charge_in = charge_out = 0.0
mismatch = []
for p in glob.glob("/home/xqian/tmp/emscan-bulk/shots/*/state.json") + \
         glob.glob("/home/xqian/tmp/emscan-bulk/shots/*/*/state.json"):
    d = json.load(open(p))
    ev = d["event"]
    st = d.get("shower_table") or {}
    if not st.get("node"):
        continue
    tot_ev += 1
    inscope = {int(n) for n in (d["scope_rule"]["in_scope"] or [])}
    ev_out = 0.0
    for i, n in enumerate(st["node"]):
        kb, kc = float(st["kb"][i]), float(st["E"][i])
        if int(n) in inscope:
            charge_in += kc
        else:
            charge_out += kc
            ev_out += kc
            tot_skip += 1
            if kb < 20.0 <= kc:
                mismatch.append((ev, int(n), kb, kc))
    worst.append((ev_out, ev, len(st["node"]) - len(inscope)))
worst.sort(reverse=True)
print("events measured: %d   showers skipped by the scope rule: %d" % (tot_ev, tot_skip))
print("kine_charge in scope   : %9.1f MeV" % charge_in)
print("kine_charge out of scope: %9.1f MeV  (%.1f%% of the total)"
      % (charge_out, 100.0 * charge_out / (charge_in + charge_out)))
print()
print("showers dropped on kine_best<20 that carry kine_charge>=20 -- the column")
print("disagreement agent 1 found on evt500083:  %d shower(s)" % len(mismatch))
for ev, n, kb, kc in sorted(mismatch, key=lambda r: -r[3])[:12]:
    print("   %-11s shower %-7d kine_best %5.1f  kine_charge %6.1f" % (ev, n, kb, kc))
print()
print("events with the most skipped charge:")
for c, ev, n in worst[:10]:
    print("   %-11s %6.1f MeV across %d skipped shower(s)" % (ev, c, n))
