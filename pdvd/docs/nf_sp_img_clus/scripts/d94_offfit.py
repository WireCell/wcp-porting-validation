#!/usr/bin/env python3
"""doc pdvd/94 -- is there a discriminator for the 10 candidates where the chain calls the stop and
finds no Michel?  (doc 92 sec 9 item 2 / doc 90 sec 8 item 2 / doc 78 item 8.)

READ-ONLY.  Doc 90 sec 8 item 2 asks to *find a discriminator before any rule*, so this round builds
no knob and runs no arm; the deliverable is a sized discriminator or a documented statement that
none separates.

WHY THIS POPULATION, AND WHY ON THE FLIPPED PREP.  The population is bucket (c) of doc 92 sec 7,
recomputed on prep_p93vprod (the flipped production of doc 93).  Doc 90 sec 8 item 2's pre-flip
"8 unfitted Michels" is a DIFFERENT, now-stale set: three of the ten (039252_9/101, 039349_44/28,
039349_51/29) are stop-called ONLY because doc 93's wide Bragg read fires, so scoping from a pre-flip
prep silently drops them.  Because those three are the agent's own output, every table below also
splits 3 NEW vs 7 pre-existing: a "discriminator" that works mainly on the new three is an artifact
of doc 93, not a Michel-finding rule.

WHAT IS ALREADY RULED OUT (measured in doc 93's round-2 orientation, on this prep + record):
  - n_stop_arms == 0 on 80% of TARGET but 97% of N1 and 8% of TP.  "No arm at the stop" is not a
    discriminator; it is simply what "no Michel found" looks like.
  - pf.seg_rej is EMPTY on all ten: PR never fitted-then-dropped a segment, so no dropped residual
    holds the Michel.
  - the only near_cluster at the stop is the candidate's OWN cluster (its id == the candidate's).
  So if the Michel's charge is anywhere, it is inside the muon's own cluster -- either swallowed by
  the fit (doc 86's "fit-through") or lying off it unfitted.  That is what this script measures.

THE NULL.  N1 = owner-judged stopper, NOT a Michel, stop called, no Michel found: the chain and the
owner AGREE there is nothing there.  A discriminator must fire on TARGET and not on N1.
CAVEAT, stated in the output: only 4 of N1's 93 are owner-judged (the rest single-scanner smx1a),
and only 5 of the 10 targets are.  Doc 90 sec 6 warns against treating un-pinned items as clean
negatives; N1 is a weak null and the output says so rather than hiding it.

Fork by duplication (CLAUDE.md M10) of d90_offfit.py, which stays untouched as doc 90's record.
Changed: groups are computed from the record + prep instead of read from a scan tranche's
labels/key; the per-item printout is kept but gated behind --verbose; the summary gains the
NEW/pre-existing split and a threshold sweep against the null.

Usage: STM_SCAN_RECORD=<smx1a..smx9 record> d94_offfit.py --prep PREP [--verbose] [--json OUT]
"""
import argparse, collections, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan")
import census_lib as C

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--json", default=None)
ap.add_argument("--max-null", type=int, default=0, help="0 = all of N1")
a = ap.parse_args()
if not os.environ.get("STM_SCAN_RECORD", "").endswith("smx7_smx8_smx9_verdicts.json"):
    sys.exit("STM_SCAN_RECORD must name the merged smx1a..smx9 record (doc 93)")
rec = C.load_record()

NEW3 = {"039252_9/101", "039349_44/28", "039349_51/29"}   # created by doc 93's flip

V = {}
for f in sorted(os.listdir(a.prep)):
    if f.startswith("smprep-") and f.endswith(".json"):
        V[f[7:-5].replace("-c", "/")] = json.load(open(os.path.join(a.prep, f)))["verdict"]
J = [k for k in V if k in rec and C.judged(rec[k])]
stop = lambda k: int(V[k].get("is_stm") or 0) == 1
mfnd = lambda k: int(V[k].get("michel_found") or 0) == 1
GROUPS = collections.OrderedDict()
GROUPS["TARGET"] = sorted(k for k in J if C.is_michel(rec[k]) and stop(k) and not mfnd(k))
GROUPS["N1_null"] = sorted(k for k in J if C.is_stopper(rec[k]) and not C.is_michel(rec[k]) and stop(k) and not mfnd(k))
GROUPS["TP_found"] = sorted(k for k in J if C.is_michel(rec[k]) and stop(k) and mfnd(k))
if a.max_null:
    GROUPS["N1_null"] = GROUPS["N1_null"][: a.max_null]
print("populations on %s" % a.prep)
for g, ks in GROUPS.items():
    own = sum(1 for k in ks if rec[k].get("confidence") == "owner")
    print("  %-9s %3d   owner-judged %3d   (%s)" % (g, len(ks), own, "the 10" if g == "TARGET" else "the null" if g == "N1_null" else "the chain succeeds"))
print("  NOTE: the null is weak -- most of N1 is single-scanner smx1a, not owner-judged (doc 90 sec 6).")


def measure(k):
    ev, c = k.split("/")
    p = json.load(open(os.path.join(a.prep, "smprep-%s-c%s.json" % (ev, c))))
    v, m, r = p["verdict"], p["muon"], rec[k]
    M = np.c_[m["x"], m["y"], m["z"]]
    rr = np.asarray(m["rr"], float)
    i = int(np.argmin(rr))
    stop_xyz = M[i]
    im = p["image_near"]
    X = np.c_[im["x"], im["y"], im["z"]]
    cl = np.asarray(im["c"], int); q = np.asarray(im["q"], float)
    dstop = np.linalg.norm(X - stop_xyz, axis=1)
    # distance to the nearest FITTED point: the muon fit rows plus every PR segment of the cluster
    F = np.vstack([M] + [np.c_[s["x"], s["y"], s["z"]] for s in p["pf"]["seg"]])
    dfit = cKDTree(F).query(X)[0]
    own = (cl == int(c)) & (dstop <= 10.0)
    w = (rr - rr[i]) <= 5.0
    ax = stop_xyz - M[w][int(np.argmax(rr[w]))]
    n = np.linalg.norm(ax)
    ax = ax / n if n > 0 else ax
    along = (X - stop_xyz) @ ax
    past = own & (along > 0.5)
    off = own & (dfit > 2.0)
    # the body control: the same reading 30-40 cm upstream, where no Michel can be
    body = M[(rr > rr[i] + 30) & (rr < rr[i] + 40)]
    ctl_off = -1
    if len(body):
        ctl = (cl == int(c)) & (cKDTree(body).query(X)[0] <= 5.0)
        ctl_off = int((ctl & (dfit > 2.0)).sum())
    pin = r.get("pin") or {}
    pin_rr = pin.get("rr") if pin.get("placed") else r.get("pin_rr")
    ndead = 0
    for pl, key in (("u", "pu"), ("v", "pv"), ("w", "pw")):
        ch0, t0 = m[key][i], m["pt"][i]
        dd = p["dead"][pl]
        ndead += bool([1 for x, lo, hi in zip(dd["ch"], dd["t0"], dd["t1"])
                       if abs(x - ch0) <= 15 and lo - 30 <= t0 <= hi + 30])
    return dict(key=k, new=k in NEW3, off=int(off.sum()), q_off=float(q[off].sum() / 1e3),
                past=int(past.sum()), past_off=int((past & (dfit > 2.0)).sum()),
                other10=int(((cl != int(c)) & (dstop <= 10.0)).sum()),
                ctl_off=ctl_off, dead=ndead, arms=int(v.get("n_stop_arms") or 0),
                pin_rr=(None if pin_rr is None else float(pin_rr)),
                conf=rec[k].get("confidence"), muon_cm=p.get("muon_len_cm"))


OUT = collections.OrderedDict()
for g, ks in GROUPS.items():
    OUT[g] = [measure(k) for k in ks]

print("\n" + "=" * 104)
print("THE 10 TARGETS, item by item  (off = own-cluster points > 2 cm off the fit within 10 cm of the fit end)")
print("  %-14s %-4s %-7s %4s %8s %5s %8s %7s %5s %5s %6s" % (
    "item", "new", "conf", "off", "q_off k", "past", "past>2cm", "ctl_off", "arms", "dead", "pin_rr"))
for s in OUT["TARGET"]:
    print("  %-14s %-4s %-7s %4d %8.0f %5d %8d %7d %5d %5d %6s" % (
        s["key"], "NEW" if s["new"] else "", s["conf"], s["off"], s["q_off"], s["past"], s["past_off"],
        s["ctl_off"], s["arms"], s["dead"], "-" if s["pin_rr"] is None else "%.1f" % s["pin_rr"]))

med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")
print("\n" + "=" * 104)
print("GROUP SUMMARY  (frac = the fraction of the group with off >= 1, i.e. any unfitted charge at the stop)")
print("  %-22s %4s %8s %9s %9s %9s %9s" % ("group", "n", "off>=1", "med off", "med q_off", "med past", "med ctl_off"))
for g, ss in OUT.items():
    print("  %-22s %4d %7.0f%% %9.1f %9.0f %9.1f %9.1f" % (
        g, len(ss), 100.0 * sum(1 for s in ss if s["off"] >= 1) / max(len(ss), 1),
        med([s["off"] for s in ss]), med([s["q_off"] for s in ss]),
        med([s["past"] for s in ss]), med([s["ctl_off"] for s in ss if s["ctl_off"] >= 0])))
t_new = [s for s in OUT["TARGET"] if s["new"]]; t_old = [s for s in OUT["TARGET"] if not s["new"]]
for nm, ss in (("TARGET: the 3 NEW (doc 93's)", t_new), ("TARGET: the 7 pre-existing", t_old)):
    print("  %-22s %4d %7.0f%% %9.1f %9.0f %9.1f %9.1f" % (
        nm, len(ss), 100.0 * sum(1 for s in ss if s["off"] >= 1) / max(len(ss), 1),
        med([s["off"] for s in ss]), med([s["q_off"] for s in ss]),
        med([s["past"] for s in ss]), med([s["ctl_off"] for s in ss if s["ctl_off"] >= 0])))

print("\n" + "=" * 104)
print("THRESHOLD SWEEP: how many of each group an 'off >= T unfitted points at the stop' rule would fire on.")
print("  A usable discriminator needs high TARGET and low N1.  TP is shown only as context.")
print("  %-6s %14s %14s %14s %10s" % ("T", "TARGET /%d" % len(OUT["TARGET"]), "N1 null /%d" % len(OUT["N1_null"]),
                                       "TP /%d" % len(OUT["TP_found"]), "new3 /3"))
for T in (1, 3, 5, 10, 20, 30):
    f = lambda ss: sum(1 for s in ss if s["off"] >= T)
    print("  off>=%-2d %14d %14d %14d %10d" % (T, f(OUT["TARGET"]), f(OUT["N1_null"]), f(OUT["TP_found"]), f(t_new)))

if a.json:
    json.dump({g: ss for g, ss in OUT.items()}, open(a.json, "w"), indent=1)
    print("\nwrote %s" % a.json)
