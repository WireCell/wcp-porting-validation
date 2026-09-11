#!/usr/bin/env python3
"""doc pdvd/81 -- the tail and the conn_type split: what decides the flip.

Reads the readout JSON of an ON arm.  (1) the ratio split by michel_conn_type;
(2) every candidate whose q2d energy is above the 52.8 MeV Michel endpoint or
whose ratio to the chain is outside [0.5, 2], by name, with the mechanism
fields; (3) the raw (all-measured) energy beside the headline, via the
per-detector linear conversion scale recovered from a candidate with no
cross-shared cell.
"""
import json, sys, collections
import numpy as np

def combine(sums, n, weights=(0.25, 0.25, 1.0), asym=0.04):
    w = [0.0 if n[i] == 0 else weights[i] for i in range(3)]
    mn = int(np.argmin(sums)); mx = int(np.argmax(sums)); md = 1
    if mn != mx:
        md = [i for i in range(3) if i != mn and i != mx][0]
    else:
        mn, md, mx = 0, 1, 2
    ws = sum(w)
    overall = sum(w[i] * sums[i] for i in range(3)) / ws if ws > 0 else 0.0
    a = 0.0
    if sums[md] + sums[mx] > 0: a = abs(sums[md] - sums[mx]) / (sums[md] + sums[mx])
    if a > asym and (w[md] + w[mn]) > 0:
        overall = (w[md] * sums[md] + w[mn] * sums[mn]) / (w[md] + w[mn])
    return overall

rows = json.load(open(sys.argv[1]))
det = sys.argv[2] if len(sys.argv) > 2 else "pdvd"
M = [r for r in rows if r.get("michel_found") == 1 and r.get("michel_q2d_valid") == 1]
print("== %s: %d Michel-carrying valid candidates" % (det, len(M)))

# the linear charge->energy scale, from a candidate with positive charge
scale = None
for r in M:
    if r["michel_q2d"] > 0 and r["michel_ke_q2d"] > 0:
        scale = r["michel_ke_q2d"] / r["michel_q2d"]; break
print("   conversion scale %.4e MeV/e (linear, the unfitted-piece constant)" % scale)

def tot(r): return r.get("michel_ke_total", r["michel_ke_best"])
for r in M:
    n = [r["michel_q2d_n_u"], r["michel_q2d_n_v"], r["michel_q2d_n_w"]]
    raw = [r.get("michel_q2d_raw_u", 0), r.get("michel_q2d_raw_v", 0), r.get("michel_q2d_raw_w", 0)]
    r["_ke_raw"] = max(combine(raw, n), 0.0) * scale
    r["_nx"] = r.get("michel_q2d_nx_u", 0) + r.get("michel_q2d_nx_v", 0) + r.get("michel_q2d_nx_w", 0)
    r["_n"] = sum(n)
    r["_ratio"] = r["michel_ke_q2d_total"] / tot(r) if tot(r) > 0 else float("nan")

print("\n-- ratio q2d_total / chain total, by michel_conn_type (1 attached, 2 bridged, 3 charge-only)")
for ct in sorted(set(r.get("michel_conn_type", -1) for r in M)):
    g = [r for r in M if r.get("michel_conn_type", -1) == ct]
    v = np.array([r["_ratio"] for r in g]); v = v[np.isfinite(v)]
    z = sum(1 for r in g if r["michel_ke_q2d_total"] <= 0)
    nx = np.array([r["_nx"] / max(r["_n"], 1) for r in g])
    print("   conn %d: n %3d | ratio p10 %.2f p50 %.2f p90 %.2f | zero-energy %d | cross-shared cell frac p50 %.3f" % (
        ct, len(g), np.percentile(v, 10) if len(v) else float('nan'), np.percentile(v, 50) if len(v) else float('nan'),
        np.percentile(v, 90) if len(v) else float('nan'), z, np.median(nx)))

print("\n-- headline vs raw (all-measured): median ke_q2d / ke_q2d_raw")
ok = [r for r in M if r["_ke_raw"] > 0]
rr = np.array([r["michel_ke_q2d"] / r["_ke_raw"] for r in ok])
xs = [r for r in M if r["_nx"] > 0]
print("   n %d, p10 %.3f p50 %.3f p90 %.3f | candidates with any cross-shared cell %d of %d (%.0f%%)" % (
    len(rr), np.percentile(rr, 10), np.percentile(rr, 50), np.percentile(rr, 90), len(xs), len(M), 100.0 * len(xs) / max(len(M), 1)))
if xs:
    rx = np.array([r["michel_ke_q2d"] / r["_ke_raw"] for r in xs if r["_ke_raw"] > 0])
    print("   on those %d alone: p10 %.3f p50 %.3f p90 %.3f (1.0 = the substitution changed nothing)" % (
        len(rx), np.percentile(rx, 10), np.percentile(rx, 50), np.percentile(rx, 90)))

print("\n-- the tail, by name: q2d_total > 52.8 MeV (the Michel endpoint) or ratio outside [0.5, 2]")
print("   %-16s %7s %7s %7s %6s %5s %5s %6s %6s  %s" % ("key", "q2d_tot", "raw", "chain", "len", "conn", "drop", "cells", "xshar", "note"))
tail = [r for r in M if r["michel_ke_q2d_total"] > 52.8 or not (0.5 <= r["_ratio"] <= 2.0)]
tail.sort(key=lambda r: -r["michel_ke_q2d_total"])
for r in tail[:26]:
    note = []
    if tot(r) > 52.8: note.append("chain also over endpoint")
    if r["_nx"] > 0: note.append("cross-shared %d/%d" % (r["_nx"], r["_n"]))
    if r.get("michel_conn_type") == 2: note.append("bridged")
    if r["michel_ke_q2d_total"] <= 0: note.append("ZERO")
    print("   %-16s %7.1f %7.1f %7.1f %6.1f %5d %5d %6d %6d  %s" % (
        r["key"], r["michel_ke_q2d_total"], r["_ke_raw"], tot(r), r.get("michel_len", -1),
        r.get("michel_conn_type", -1), r["michel_q2d_dropped_plane"], r["_n"], r["_nx"], "; ".join(note)))
print("   (%d of %d candidates in the tail)" % (len(tail), len(M)))
