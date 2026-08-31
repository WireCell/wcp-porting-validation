#!/usr/bin/env python3
"""Re-derive a claimed gate inversion using the length reconciliation.

Agent 4 (wave 3) showed that the accept flag taken from `segments.owner` is
UNSAFE: owner_map() is first-owner-wins, so a genuine member of shower A can
carry owner B and read as "rejected by A", which manufactures an inversion.
The safe test is arithmetic: shower_table.length minus the summed path length
of the owner-grouped segments leaves a residual that must be matched by the
segments that really are members but show another owner.

    ./recheck_gate.py evt91917 87053 66021 [more sids...]
"""
import glob, itertools, json, sys

ev, node = sys.argv[1], int(sys.argv[2])
claimed = [int(x) for x in sys.argv[3:]]
p = glob.glob("/home/xqian/tmp/emscan-bulk/shots/%s/**/state.json" % ev,
              recursive=True) + glob.glob(
              "/home/xqian/tmp/emscan-bulk/shots/%s/state.json" % ev)
d = json.load(open(p[0]))
st = d["shower_table"]
i = [k for k, n in enumerate(st["node"]) if int(n) == node][0]
L_table, nseg_table = float(st["length"][i]), int(st["nseg"][i])
sh = [s for s in d["showers"] if int(s["node"]) == node][0]
c = sh["candidates"]
owned = [(c["sid"][k], c["length"][k]) for k in range(len(c["sid"]))
         if str(c["owner"][k]) == str(node)]
L_owned = sum(l for _, l in owned)
resid = L_table - L_owned
print("%s shower %d" % (ev, node))
print("  shower_table: length %.4f cm, nseg %d" % (L_table, nseg_table))
print("  owner-grouped: %d segment(s), summed length %.4f cm" % (len(owned), L_owned))
print("  cmp_div: %s" % sh["cmp_div"].split("\n")[0][:100])
print("  RESIDUAL = %.4f cm" % resid)
if abs(resid) < 1e-3:
    print("  -> owner grouping is complete; nothing hidden. Accept flags are safe here.")
else:
    others = [(c["sid"][k], c["length"][k]) for k in range(len(c["sid"]))
              if str(c["owner"][k]) != str(node)]
    hit = None
    for r in range(1, min(5, len(others)) + 1):
        for comb in itertools.combinations(others, r):
            if abs(sum(l for _, l in comb) - resid) < 1e-3:
                hit = comb
                break
        if hit:
            break
    if hit:
        print("  -> residual matched EXACTLY by %s"
              % ", ".join("%d (%.4f)" % (s, l) for s, l in hit))
        print("     those are MEMBERS wearing another owner -- not rejects.")
    else:
        print("  -> no subset of <=4 non-owned candidates matches the residual.")
for s in claimed:
    k = [j for j in range(len(c["sid"])) if int(c["sid"][j]) == s]
    if not k:
        print("  claimed %d: not in this shower's candidate list" % s)
        continue
    k = k[0]
    print("  claimed %-7d owner=%-8s dist %6.2f angle %6.2f tier %s site %r"
          % (s, c["owner"][k], c["dist"][k], c["angle"][k], c["tier"][k], c["site"][k]))
