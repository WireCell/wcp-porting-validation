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
             "scan_key", "stm_retiler", "stmw", "pdhd_retile_scan_key"):
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
    cap = V.CONTEXT_MAX * 1.05 + V.DENSE_MAX
    check(len(ctx.data["a"]) <= cap,
          "panel %s-%s context bounded (%d <= %d = thinned + dense cap)"
          % (kk[0], kk[1], len(ctx.data["a"]), cap))

print("\n4. panels default to the full detector volume, not the cluster extent")
for (ha, va), f in V.FIGS.items():
    span = f.x_range.end - f.x_range.start
    full = V.VOL[ha][1] - V.VOL[ha][0]
    check(span >= full, "panel %s-%s x-range spans the detector (%.0f >= %.0f)"
          % (ha, va, span, full))

print("\n5. the dense context is real, and is purely geometric")
import numpy as np
it = next((i for i in V.ITEMS if V.event_charge(i["event"]) is not None), None)
check(it is not None, "an item with charge is available")
X, Y, Z, Q, C = V.event_charge(it["event"])
Pxyz = np.c_[X, Y, Z]
m = C == it["cluster"]
i_far, step, n0 = V.context_index(Pxyz, m, False)
i_den, step2, n1 = V.context_index(Pxyz, m, True)
check(n0 == 0 and n1 > 0, "dense path adds a near set (%d -> %d)" % (n0, n1))
check(len(i_den) > len(i_far),
      "dense context is strictly bigger (%d > %d) -- not inert" % (len(i_den), len(i_far)))
check(set(i_far).issubset(set(i_den)), "dense context is a superset of the thinned one")
check(not m[i_den].any(), "context never includes the target cluster's own points")
# independent reimplementation from coordinates ALONE: no cluster ids, no arm,
# no tagger output can enter a set that this reproduces exactly.
small = None
for i in V.ITEMS:
    ch = V.event_charge(i["event"])
    if ch is None:
        continue
    mm = ch[4] == i["cluster"]
    if 0 < int(mm.sum()) * int((~mm).sum()) < 4e7:
        small = (i, ch, mm)
        break
check(small is not None, "an item small enough for the brute-force check exists")
if small:
    si, sch, sm = small
    sP = np.c_[sch[0], sch[1], sch[2]]
    s_far, _, _ = V.context_index(sP, sm, False)
    s_den, _, _ = V.context_index(sP, sm, True)
    so = np.flatnonzero(~sm)
    dmin = np.sqrt(((sP[so][:, None, :] - sP[sm][None, :, :]) ** 2).sum(-1)).min(1)
    want = set(so[dmin <= V.DENSE_R].tolist()) | set(s_far.tolist())
    check(want == set(s_den.tolist()),
          "dense set == brute-force geometric set on evt %s cl %d (%d vs %d)"
          % (si["event"], si["cluster"], len(want), len(s_den)))
import inspect
csrc = inspect.getsource(V.context_index)
check("cluster" not in csrc.replace("the cluster", "").replace("cluster point", "")
      .replace("clusters the tagger", ""),
      "context_index body references no cluster id")

print("\n6. every label in the alphabet round-trips with its partial flag")
V.LABELS.clear()
p = V.LABEL_FILE
k = V.item_key(V.ITEMS[0])
for choice, spec in V.CHOICES.items():
    V.go(0)
    V.set_label(choice)
    check(os.path.isfile(p), "labels.json written without an explicit Save (%s)" % choice)
    d = json.load(open(p))["labels"]
    rec = d.get(k, {})
    check(rec.get("label") == spec["label"] and rec.get("partial") == spec["partial"]
          and rec.get("choice") == choice,
          "%-9s -> label=%s partial=%s persisted" % (choice, spec["label"], spec["partial"]))
check("/" in k, "key is per-event, not a bare cluster id (%s)" % k)
check(set(V.CHOICES) == {"STM", "THRU", "FRAG_STM", "FRAG_THRU", "MESSY", "UNCLEAR"},
      "the alphabet is the six documented choices")
check(V.CHOICES["FRAG_STM"]["label"] == "STM"
      and V.CHOICES["FRAG_THRU"]["label"] == "THRU",
      "a FRAG choice carries the FULL object's binary verdict")

print("\n7. the scorer refuses a label it does not know")
import subprocess
V.LABELS.clear()
V.LABELS[k] = dict(label="BOGUS", partial=False, choice="BOGUS", notes="",
                   scan_id=1, event=V.ITEMS[0]["event"], cluster=V.ITEMS[0]["cluster"],
                   npts=0, length_cm=0.0)
V.save_labels()
r = subprocess.run([sys.executable, os.path.join(HERE, "score_stm_scan.py"),
                    "--tag", "_selftest"], capture_output=True, text=True)
check(r.returncode != 0 and "unrecognised label" in (r.stdout + r.stderr),
      "score_stm_scan.py exits non-zero on an unknown label (rc=%d)" % r.returncode)
V.LABELS.clear(); V.save_labels()
try:
    os.remove(p); os.rmdir(os.path.dirname(p))
except OSError:
    pass

print("\n%s  (%d failure%s)" % ("FAILED" if fails else "ALL PASS",
                               len(fails), "" if len(fails) == 1 else "s"))
sys.exit(1 if fails else 0)
