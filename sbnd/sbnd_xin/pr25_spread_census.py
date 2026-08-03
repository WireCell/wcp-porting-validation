#!/usr/bin/env python3
"""doc pr/25 sec 3 round 2: is the per-bucket dir_3 RMS distribution measuring
shower width, or the drift-quantization noise floor?

Parses the WCT_SHOWER_TOPO_DEBUG=1 entry lines emitted by
segment_is_shower_topology (clus/src/PRSegmentFunctions.cxx) and tests the
round-2 fix hypothesis:

  round 1 saw only max_spread (a single-bucket extremum) and n_over0.4 (a count
  at a threshold that sits INSIDE the 0.313 cm drift-slice lattice).  Round 2
  adds the quantiles p75/p90/p95 and the counts at the branch thresholds
  (0.7/0.8/1.0 cm).  On the 21-event L>50cm blast radius those showed a gap at
  p90 0.60 -> 1.17 cm.  That gap has n=3 on the high side and comes from a
  population SELECTED for containing long shower-flagged segments, so it must be
  re-tested against the short (L<50cm) firings, where nobody disputes that real
  showers live.

CONFOUND this script controls for: p90 -> max as the bucket count shrinks, so
short segments score higher on p90 for a purely statistical reason.  The
scale-free statistic `frac_over0.8 = n_over0.8/nbuckets` is therefore reported
alongside, and the p90 comparison is also made in matched nbuckets bands.

Usage:
    python3 pr25_spread_census.py --arm work-pr25s3r2-dbgall
"""
import argparse, glob, os, re
import numpy as np

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"

ENTRY = re.compile(
    r"seg (\S+) L ([\d.]+)cm assoc_npts (\d+) nbuckets (\d+) n_over0\.4cm (\d+) "
    r"n_over0\.7cm (\d+) n_over0\.8cm (\d+) n_over1\.0cm (\d+) "
    r"rms_p50 ([\d.]+)cm rms_p75 ([\d.]+)cm rms_p90 ([\d.]+)cm rms_p95 ([\d.]+)cm "
    r"max_spread ([\d.]+)cm maxcont ([\d.]+)cm lsl ([\d.]+)cm tel ([\d.]+)cm "
    r"lsl/tel ([\d.]+) tel/L ([\d.]+) dir3x ([\d.]+) branch (\d+)")


def load(arm):
    rows, seen = [], set()
    for d in sorted(glob.glob(os.path.join(BASE, arm, "pr_evt*"))):
        ev = int(os.path.basename(d).replace("pr_evt", ""))
        lg = os.path.join(d, "wct_pr_evt%d.log" % ev)
        if not os.path.exists(lg):
            continue
        for ln in open(lg, errors="replace"):
            if "shower_topo dbg" not in ln or "guard" in ln:
                continue
            m = ENTRY.search(ln)
            if not m:
                continue
            g = m.groups()
            r = dict(ev=ev, L=float(g[1]), npts=int(g[2]), nb=int(g[3]),
                     n04=int(g[4]), n07=int(g[5]), n08=int(g[6]), n10=int(g[7]),
                     p50=float(g[8]), p75=float(g[9]), p90=float(g[10]), p95=float(g[11]),
                     mx=float(g[12]), mc=float(g[13]), lsl=float(g[14]), tel=float(g[15]),
                     lsltel=float(g[16]), telL=float(g[17]), dir3x=float(g[18]),
                     branch=int(g[19]))
            # segment->id() reads -1 here (stamped later), so dedupe on
            # (event, rounded L): the same segment is evaluated twice per event
            # by TaggerCheckNeutrino's two clustering_points passes.
            k = (ev, round(r["L"], 1))
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    rows = load(a.arm)
    nev = len(set(r["ev"] for r in rows))
    fired = [r for r in rows if r["branch"] > 0]
    print("# events %d ; distinct segment evaluations %d ; FIRED the disjunction %d"
          % (nev, len(rows), len(fired)))

    lng = [r for r in fired if r["L"] > 50]
    sht = [r for r in fired if r["L"] <= 50]
    print("#   fired with L>50cm: %d   L<=50cm: %d" % (len(lng), len(sht)))

    print("\n=== A. rms_p90 distribution of FIRED segments (the round-2 statistic) ===")
    print("population        |   n  |  p10   p25   med   p75   p90   max   | frac with p90<0.7")
    for nm, s in (("L>50cm  (long)", lng), ("L<=50cm (short)", sht)):
        if not s:
            continue
        v = np.array([r["p90"] for r in s])
        print("%-17s | %4d | %5.2f %5.2f %5.2f %5.2f %5.2f %6.2f | %.3f"
              % (nm, len(s), q(v, 10), q(v, 25), q(v, 50), q(v, 75), q(v, 90), v.max(),
                 float((v < 0.7).mean())))

    print("\n=== B. same, but SCALE-FREE (frac of buckets over the 0.8cm branch cut) ===")
    print("   controls the p90->max confound: p90 rises as nbuckets falls")
    print("population        |   n  |  p10   p25   med   p75   p90   max")
    for nm, s in (("L>50cm  (long)", lng), ("L<=50cm (short)", sht)):
        if not s:
            continue
        v = np.array([r["n08"] / max(r["nb"], 1) for r in s])
        print("%-17s | %4d | %5.3f %5.3f %5.3f %5.3f %5.3f %6.3f"
              % (nm, len(s), q(v, 10), q(v, 25), q(v, 50), q(v, 75), q(v, 90), v.max()))

    print("\n=== C. rms_p90 in MATCHED nbuckets bands (removes the confound) ===")
    print("nbuckets band | pop     |  n  | med p90 | frac p90<0.7 | med frac_over0.8")
    for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 160), (160, 10**6)]:
        for nm, s in (("long ", lng), ("short", sht)):
            b = [r for r in s if lo <= r["nb"] < hi]
            if len(b) < 3:
                continue
            v = np.array([r["p90"] for r in b])
            f = np.array([r["n08"] / max(r["nb"], 1) for r in b])
            print("  %4d-%-6d| %s   | %3d |  %.2f   |    %.3f     |   %.4f"
                  % (lo, hi, nm, len(b), np.median(v), float((v < 0.7).mean()), np.median(f)))

    print("\n=== D. is the FIRED p90 distribution bimodal? (histogram, all L) ===")
    v = np.array([r["p90"] for r in fired])
    edges = [0, .4, .45, .5, .55, .6, .65, .7, .8, .9, 1.0, 1.2, 1.5, 2, 3, 5, 100]
    h, _ = np.histogram(v, bins=edges)
    for i in range(len(h)):
        bar = "#" * min(60, int(60 * h[i] / max(h.max(), 1)))
        print("  %5.2f-%-5.2f | %4d %s" % (edges[i], edges[i + 1], h[i], bar))

    print("\n=== E. q-robustness: does the long-population separation survive p75/p90/p95? ===")
    print("   (the fix substitutes the q-quantile for max_spread in the guard)")
    for qn in ("p75", "p90", "p95"):
        v = sorted(r[qn] for r in lng)
        gaps = [(v[i + 1] - v[i], v[i], v[i + 1]) for i in range(len(v) - 1)]
        big = max(gaps) if gaps else (0, 0, 0)
        n_below = sum(1 for x in v if x < 0.8)
        print("  %s: n=%d  largest gap %.2f cm (%.2f -> %.2f)   n below 0.8cm cut: %d/%d"
              % (qn, len(v), big[0], big[1], big[2], n_below, len(v)))

    print("\n=== F. the target, and the long firings sorted by p90 ===")
    print("evt        L(cm)  nb   n>0.4 n>0.8  p50  p75  p90   p95   max  maxcont lsl/tel dir3x br")
    for r in sorted(lng, key=lambda r: r["p90"]):
        tag = "  <== 321107 TARGET" if r["ev"] == 321107 else ""
        print("%8d %6.1f %4d %5d %5d  %.2f %.2f %.2f %5.2f %5.2f %6.1f  %.3f  %.3f %2d%s"
              % (r["ev"], r["L"], r["nb"], r["n04"], r["n08"], r["p50"], r["p75"], r["p90"],
                 r["p95"], r["mx"], r["mc"], r["lsltel"], r["dir3x"], r["branch"], tag))


if __name__ == "__main__":
    main()
