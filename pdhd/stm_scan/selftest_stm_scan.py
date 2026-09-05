#!/usr/bin/env python3
"""Headless gate on the STM scan display -- doc sec 13.

Checks the things a "server started, HTTP 200" test cannot: that the render
path actually produces points, that the blind holds structurally, and that
labels round-trip.  Run before handing the app to a scanner.

  python3 selftest_stm_scan.py
"""
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PDHD = os.path.dirname(HERE)
fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


print("1. the blind is structural -- the viewer must never name a leaking layer")
src = open(os.path.join(HERE, "stm_scan_viewer.py")).read()
code = "\n".join(l for l in src.splitlines()
                 if not l.strip().startswith("#"))
code = re.sub(r'""".*?"""', "", code, flags=re.S)      # drop docstrings
for leak in ("stm_fit", "stm_tagged", "steiner_graph", "steiner_terminals",
             "scan_key", "stm_retiler", "stmw"):
    check(leak not in code, "viewer code never references %r" % leak)

print("\n2. the two layers it does read are byte-identical between the arms")
import hashlib
ev = "0"
za = os.path.join(PDHD, "work", "029107_%s_stm0" % ev, "mabc-pr.zip")
zb = os.path.join(PDHD, "work", "029107_%s_stmw" % ev, "mabc-pr.zip")
if os.path.exists(za) and os.path.exists(zb):
    A, B = zipfile.ZipFile(za), zipfile.ZipFile(zb)
    for n in A.namelist():
        if "clustering-global" in n or "channel-deadarea" in n:
            ha = hashlib.sha256(A.read(n)).hexdigest()
            hb = hashlib.sha256(B.read(n)).hexdigest()
            check(ha == hb, "identical across arms: %s" % os.path.basename(n))
else:
    check(False, "both arm zips present for event %s" % ev)

print("\n3. the render path yields points for real items")
sys.argv = [sys.argv[0], "--tag", "_selftest"]
import stm_scan_viewer as V
check(len(V.ITEMS) > 0, "item list loaded (%d items)" % len(V.ITEMS))
tried = ok_t = ok_c = 0
for it in V.ITEMS[:12]:
    ch = V.event_charge(it["event"])
    tried += 1
    if ch is None:
        continue
    X, Y, Z, Q, C = ch
    m = C == it["cluster"]
    ok_t += int(m.sum() > 0)
    ok_c += int((~m).sum() > 0)
check(ok_t == tried, "every sampled item has target points (%d/%d)" % (ok_t, tried))
check(ok_c == tried, "every sampled item has context charge (%d/%d)" % (ok_c, tried))

V.go(0)
for kk, (ctx, tgt) in V.SRC.items():
    check(len(tgt.data["a"]) > 0, "panel %s-%s target source non-empty" % kk)
    check(len(ctx.data["a"]) > 0, "panel %s-%s context source non-empty" % kk)
    check(len(ctx.data["a"]) <= V.CONTEXT_MAX * 1.05,
          "panel %s-%s context decimated (%d <= %d)" % (kk[0], kk[1],
                                                        len(ctx.data["a"]), V.CONTEXT_MAX))

print("\n4. panels default to the full detector volume, not the cluster extent")
for (ha, va), f in V.FIGS.items():
    span = f.x_range.end - f.x_range.start
    full = V.VOL[ha][1] - V.VOL[ha][0]
    check(span >= full, "panel %s-%s x-range spans the detector (%.0f >= %.0f)"
          % (ha, va, span, full))

print("\n5. labels round-trip and are written on every set_label")
V.LABELS.clear()
V.go(0)
V.set_label("THRU")
p = V.LABEL_FILE
check(os.path.isfile(p), "labels.json written without an explicit Save")
d = json.load(open(p))["labels"]
k = V.item_key(V.ITEMS[0])
check(d.get(k, {}).get("label") == "THRU", "label persisted under the (event, cluster) key")
check("/" in k, "key is per-event, not a bare cluster id (%s)" % k)
V.LABELS.clear(); V.save_labels()
try:
    os.remove(p); os.rmdir(os.path.dirname(p))
except OSError:
    pass

print("\n%s  (%d failure%s)" % ("FAILED" if fails else "ALL PASS",
                               len(fails), "" if len(fails) == 1 else "s"))
sys.exit(1 if fails else 0)
