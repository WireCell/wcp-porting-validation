#!/usr/bin/env python3
"""doc pr/142 -- systematic A/B of two full-sample PR productions.

FORK of scripts/pr83r3_scores_ab.py (CLAUDE.md M10: fork, do not extract a
shared helper -- pr83r3 has live consumers and stays byte-for-byte).  What the
fork changes, and why:

  - pr83r3 shells out to pr_scores_table.py per arm.  This reads the TSVs
    directly, so it also runs against the COMMITTED product tables
    (products/prod0825/*.tsv), whose arms have been retired.
  - pr83r3 reports movers only.  The owner asked for a systematic comparison of
    "1. nueCC, numuCC BDT 2. nusel 3. nu vertex 4. neutrino energy etc.", so
    this adds population summaries, distribution quantiles, the nusel label and
    working-point migration matrices, the rc/failure census, and the runtime +
    peak-RSS distributions.
  - the degenerate-row cut of doc 85 (49 rows that carry NO reconstruction yet
    read nu_evaluated=1) is applied before every distribution.

Population = nu_evaluated == 1 AND not degenerate.  See d85_dists.degenerate():
TaggerCheckNeutrino emits its "selected main cluster" line even when the main it
picked is a ~1.5 cm unmerge shard; KineInfo never fills and NOTHING blanks the
row.  Signature: kine_reco_Enu 0.0 exactly + vertex (0,0,0) exactly.

cosmic_flag is NOT a cosmic verdict (it is !cosmict_flag_9 and reads 1 almost
everywhere) -- cosmict_flag is used here instead.

Usage:
  pr142_campaign_ab.py --a products/empre0901/all.tsv --b products/prod0901/all.tsv
      [--label-a empre0901] [--label-b prod0901]
      [--movers-tsv f] [--summary-tsv f]
      [--numu-thr 0.05] [--nue-thr 0.05] [--enu-thr 50] [--enu-frac 0.10]
      [--vtx-thr 1.0] [--top 40]

Exit 0 always (reporting tool, not a gate).
"""
import argparse
import csv
import math
import os
import sys

# doc 85 sec 7 working points.  NUMU is the uB numu-CC point.  The nue points
# are quoted as a bracket: 7.0 is uB's, 4.30103 was the toolkit clamp ceiling
# (REMOVED 2026-08-30, doc 85 r2 -- so 7.0 is reachable on a post-removal arm
# and was not before), 0.7 is the looser number in the same uB document.
NUMU_SEL, NUE_UB, NUE_CLAMP, NUE_LOOSE = 0.9, 7.0, 4.30103, 0.7
# nue_score == -15 EXACTLY is the "br_filled != 1" sentinel.  Since the clamp
# removal (2026-08-30, toolkit 59f75bb8) the physical range runs to +-16.25562,
# so -15 is INSIDE it: test equality, never `< -14.9`.  Both arms of this round
# are post-removal, so the sentinel behaves identically on each side.
NUE_UNFILLED = -15.0
SAMPLES = ("nuecc48", "ncpi0", "mcp1k", "mcp2k")


def fnum(r, c):
    try:
        return float(r[c])
    except (KeyError, TypeError, ValueError):
        return None


def nue_filled(r):
    v = fnum(r, "nue_score")
    return v is not None and v != NUE_UNFILLED


def degenerate(r):
    """An evaluated row carrying no reconstruction (doc 85; energy+vertex test,
    not the score value -- the sentinel score differs between arms)."""
    e, x, y, z = (fnum(r, "kine_reco_Enu_MeV"), fnum(r, "nu_x_cm"),
                  fnum(r, "nu_y_cm"), fnum(r, "nu_z_cm"))
    return (e == 0.0 and x == 0.0 and y == 0.0 and z == 0.0)


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                # keyed on SAMPLE too: 18255-1-69314 is in both the mcp2k and
                # the nuecc48 selection (identical values), and a (run, subrun,
                # event) key silently drops one of the two rows.
                key = (r.get("sample", ""), int(r["run"]), int(r["subrun"]),
                       int(r["event"]))
            except (KeyError, ValueError):
                continue
            rows[key] = r
    return rows


def q(vals, f):
    if not vals:
        return float("nan")
    v = sorted(vals)
    return v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]


def stats(vals):
    if not vals:
        return dict(n=0, med=float("nan"), mean=float("nan"), p10=float("nan"),
                    p90=float("nan"), lo=float("nan"), hi=float("nan"))
    return dict(n=len(vals), med=q(vals, .5), mean=sum(vals) / len(vals),
                p10=q(vals, .1), p90=q(vals, .9), lo=min(vals), hi=max(vals))


def fmt(s, unit=""):
    if s["n"] == 0:
        return "n=0"
    return (f"n={s['n']} med={s['med']:.3f}{unit} mean={s['mean']:.3f}{unit} "
            f"[p10 {s['p10']:.3f}, p90 {s['p90']:.3f}] range [{s['lo']:.3f}, {s['hi']:.3f}]")


def population(rows):
    """(evaluated-and-clean, evaluated-but-degenerate, not-evaluated)."""
    good, degen, noeval = {}, {}, {}
    for k, r in rows.items():
        if r.get("nu_evaluated") != "1":
            noeval[k] = r
        elif degenerate(r):
            degen[k] = r
        else:
            good[k] = r
    return good, degen, noeval


def label_counts(rows):
    c = {}
    for r in rows.values():
        c[r.get("event_label", "")] = c.get(r.get("event_label", ""), 0) + 1
    return c


def rc_counts(rows):
    c = {}
    for r in rows.values():
        v = r.get("rc", "") or "<blank>"
        c[v] = c.get(v, 0) + 1
    return c


def wp(rows, col, thr):
    return sum(1 for r in rows.values()
               if fnum(r, col) is not None and fnum(r, col) > thr)


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, nargs="+", help="arm-A score TSV(s)")
    ap.add_argument("--b", required=True, nargs="+", help="arm-B score TSV(s)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--movers-tsv", default=None)
    ap.add_argument("--summary-tsv", default=None)
    ap.add_argument("--numu-thr", type=float, default=0.05)
    ap.add_argument("--nue-thr", type=float, default=0.05)
    ap.add_argument("--enu-thr", type=float, default=50.0)
    ap.add_argument("--enu-frac", type=float, default=0.10)
    ap.add_argument("--vtx-thr", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    A, B = {}, {}
    for p in a.a:
        A.update(load(p))
    for p in a.b:
        B.update(load(p))
    LA, LB = a.label_a, a.label_b

    section(f"0. COMPLETENESS -- {LA} vs {LB}")
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    both = sorted(set(A) & set(B))
    print(f"rows: {LA} {len(A)}   {LB} {len(B)}   joined {len(both)}   "
          f"only-{LA} {len(only_a)}   only-{LB} {len(only_b)}")
    for nm, lst in ((LA, only_a), (LB, only_b)):
        if lst:
            print(f"  ONLY IN {nm}: {[k[2] for k in lst][:20]}"
                  f"{' ...' if len(lst) > 20 else ''}")
    for nm, R in ((LA, A), (LB, B)):
        print(f"  {nm} rc census: " +
              "  ".join(f"{k}={v}" for k, v in sorted(rc_counts(R).items())))

    section("1. POPULATION (per sample): nu_evaluated, degenerate, nusel label")
    hdr = f"{'sample':9s} {'arm':10s} {'rows':>5s} {'eval':>5s} {'degen':>6s} {'clean':>6s}   nusel event_label"
    print(hdr)
    summary_rows = []
    for s in SAMPLES + ("ALL",):
        for nm, R in ((LA, A), (LB, B)):
            sub = {k: r for k, r in R.items() if s == "ALL" or r.get("sample") == s}
            if not sub:
                continue
            g, d, ne = population(sub)
            print(f"{s:9s} {nm:10s} {len(sub):5d} {len(g)+len(d):5d} {len(d):6d} {len(g):6d}   "
                  + " ".join(f"{k}={v}" for k, v in sorted(label_counts(sub).items())))
            summary_rows.append([s, nm, len(sub), len(g) + len(d), len(d), len(g)])

    # ---- distributions, on the clean evaluated population of each arm -------
    section("2. DISTRIBUTIONS on the clean evaluated population (own-arm)")
    for col, unit in (("kine_reco_Enu_MeV", " MeV"), ("numu_score", ""),
                      ("nue_score", ""), ("nu_sel_len_cm", " cm")):
        print(f"\n-- {col}")
        for s in SAMPLES + ("ALL",):
            line = f"  {s:9s}"
            for nm, R in ((LA, A), (LB, B)):
                g, _, _ = population({k: r for k, r in R.items()
                                      if s == "ALL" or r.get("sample") == s})
                v = [fnum(r, col) for r in g.values()
                     if fnum(r, col) is not None
                     and (col != "nue_score" or nue_filled(r))]
                line += f"  | {nm}: {fmt(stats(v), unit)}"
            print(line)

    section("3. WORKING-POINT MIGRATION (clean evaluated, both arms joined)")
    print("   doc 85 sec 7: numu_score > 0.9 is uB's numu-CC point; the nue points are a")
    print("   bracket -- 7.0 (uB), 4.30103 (the REMOVED toolkit clamp ceiling), 0.7 (loose).")
    for name, col, thr in (("numu>0.9", "numu_score", NUMU_SEL),
                           ("nue>7.0", "nue_score", NUE_UB),
                           ("nue>4.30103", "nue_score", NUE_CLAMP),
                           ("nue>0.7", "nue_score", NUE_LOOSE)):
        for s in SAMPLES:
            ga = {k: r for k, r in A.items()
                  if r.get("sample") == s and k in both and r.get("nu_evaluated") == "1"
                  and not degenerate(r)}
            gb = {k: r for k, r in B.items()
                  if r.get("sample") == s and k in both and r.get("nu_evaluated") == "1"
                  and not degenerate(r)}
            keys = sorted(set(ga) & set(gb))
            if not keys:
                continue
            pp = pf = fp = ff = 0
            for k in keys:
                va, vb = fnum(ga[k], col), fnum(gb[k], col)
                if va is None or vb is None:
                    continue
                if col == "nue_score" and not (nue_filled(ga[k]) and nue_filled(gb[k])):
                    continue
                if va > thr and vb > thr:
                    pp += 1
                elif va > thr:
                    pf += 1
                elif vb > thr:
                    fp += 1
                else:
                    ff += 1
            net = fp - pf
            print(f"  {name:12s} {s:9s} n={len(keys):5d}  pass both {pp:5d}  "
                  f"{LA}-only {pf:4d}  {LB}-only {fp:4d}  neither {ff:5d}  "
                  f"net {net:+d}")

    section("4. NUSEL event_label MIGRATION (joined events)")
    mig = {}
    for k in both:
        pair = (A[k].get("event_label", ""), B[k].get("event_label", ""))
        mig[pair] = mig.get(pair, 0) + 1
    for (x, y), n in sorted(mig.items(), key=lambda t: -t[1]):
        flag = "" if x == y else "   <-- CHANGED"
        print(f"  {x:20s} -> {y:20s}  {n:6d}{flag}")
    print(f"  nu_evaluated flips: "
          f"{sum(1 for k in both if A[k].get('nu_evaluated') != B[k].get('nu_evaluated'))}")

    # ---- movers ------------------------------------------------------------
    section("5. MOVERS against the pre-registered thresholds")
    print(f"   |dEnu| > {a.enu_thr} MeV or > {a.enu_frac*100:.0f}% | |dnumu| > {a.numu_thr} | "
          f"|dnue| > {a.nue_thr} | vtx > {a.vtx_thr} cm | any label/eval/rc change")
    movers = []
    for k in both:
        o, n = A[k], B[k]
        d = {}
        ea, eb = fnum(o, "kine_reco_Enu_MeV"), fnum(n, "kine_reco_Enu_MeV")
        if ea is not None and eb is not None:
            de = eb - ea
            if abs(de) > a.enu_thr or (ea > 0 and abs(de) / ea > a.enu_frac):
                d["enu"] = (ea, eb)
        va, vb = fnum(o, "numu_score"), fnum(n, "numu_score")
        if va is not None and vb is not None and abs(vb - va) > a.numu_thr:
            d["numu"] = (va, vb)
        fa, fb = nue_filled(o), nue_filled(n)
        if fa != fb:
            d["nue_fill"] = ("filled" if fa else "unfilled",
                             "filled" if fb else "unfilled")
        elif fa and fb:
            va, vb = fnum(o, "nue_score"), fnum(n, "nue_score")
            if va is not None and vb is not None and abs(vb - va) > a.nue_thr:
                d["nue"] = (va, vb)
        va = [fnum(o, c) for c in ("nu_x_cm", "nu_y_cm", "nu_z_cm")]
        vb = [fnum(n, c) for c in ("nu_x_cm", "nu_y_cm", "nu_z_cm")]
        if all(v is not None for v in va + vb):
            dv = math.dist(va, vb)
            if dv > a.vtx_thr:
                d["vtx_cm"] = (0.0, dv)
        if o.get("event_label") != n.get("event_label"):
            d["label"] = (o.get("event_label"), n.get("event_label"))
        if o.get("nu_evaluated") != n.get("nu_evaluated"):
            d["eval"] = (o.get("nu_evaluated"), n.get("nu_evaluated"))
        ra, rb = (o.get("rc") or ""), (n.get("rc") or "")
        # only a real rc change counts.  A blank on one side is not a failure:
        # a group-mode arm writes no per-event .time.meta, so every row of
        # prod0825 / prod0830 carries rc="" and a naive test flags all 3067.
        if ra and rb and ra != rb:
            d["rc"] = (ra, rb)
        if d:
                movers.append((k, A[k].get("sample", ""), d))

    print(f"\n  {len(movers)} movers of {len(both)} joined events "
          f"({100.0*len(movers)/max(1,len(both)):.2f}%)")
    cls = {}
    for _, _, d in movers:
        for c in d:
            cls[c] = cls.get(c, 0) + 1
    print("  by class: " + "  ".join(f"{k}={v}" for k, v in sorted(cls.items())))

    def mag(t):
        d = t[2]
        return max([abs(d[c][1] - d[c][0]) for c in ("enu",) if c in d] +
                   [abs(d[c][1] - d[c][0]) * 100 for c in ("numu", "nue") if c in d] +
                   [d["vtx_cm"][1] * 10 if "vtx_cm" in d else 0] +
                   [1e9 if ("label" in d or "eval" in d or "rc" in d
                            or "nue_fill" in d) else 0])

    print(f"\n  top {a.top} by magnitude (label/eval/rc changes first):")
    for (_s, run, sub, evt), samp, d in sorted(movers, key=mag, reverse=True)[:a.top]:
        parts = []
        for c in ("rc", "eval", "label", "nue_fill"):
            if c in d:
                parts.append(f"{c} {d[c][0]!r}->{d[c][1]!r}")
        if "enu" in d:
            o, n = d["enu"]
            parts.append(f"Enu {o:.1f}->{n:.1f} ({n-o:+.1f} MeV)")
        for c in ("numu", "nue"):
            if c in d:
                parts.append(f"{c} {d[c][0]:.3f}->{d[c][1]:.3f}")
        if "vtx_cm" in d:
            parts.append(f"vtx {d['vtx_cm'][1]:.1f} cm")
        print(f"    {samp:8s} evt {evt:7d} (run {run}): " + "; ".join(parts))

    if a.movers_tsv:
        with open(a.movers_tsv, "w") as f:
            f.write("sample\trun\tsubrun\tevent\tclasses\tenu_a\tenu_b\tdenu\t"
                    "numu_a\tnumu_b\tnue_a\tnue_b\tvtx_move_cm\tlabel_a\tlabel_b\t"
                    "eval_a\teval_b\trc_a\trc_b\tnuefill_a\tnuefill_b\n")
            for (_s, run, sub, evt), samp, d in sorted(movers, key=mag, reverse=True):
                g = lambda c, i: (f"{d[c][i]}" if c in d else "")
                enu = (f"{d['enu'][1]-d['enu'][0]:.3f}" if "enu" in d else "")
                f.write("\t".join([samp, str(run), str(sub), str(evt),
                                   ",".join(sorted(d)), g("enu", 0), g("enu", 1), enu,
                                   g("numu", 0), g("numu", 1), g("nue", 0), g("nue", 1),
                                   (f"{d['vtx_cm'][1]:.3f}" if "vtx_cm" in d else ""),
                                   g("label", 0), g("label", 1), g("eval", 0), g("eval", 1),
                                   g("rc", 0), g("rc", 1),
                                   g("nue_fill", 0), g("nue_fill", 1)]) + "\n")
        print(f"\n# wrote {a.movers_tsv} ({len(movers)} rows)")

    section("6. RUNTIME AND PEAK RSS (per event; blank if the arm ran in group mode)")
    print("   wall_s / core_s come from the job's own \"Timer: Total\" line; maxrss_kb is")
    print("   timecmd.py's RUSAGE_CHILDREN peak, i.e. PER PROCESS and concurrency-")
    print("   insensitive.  Wall under concurrency is contention-dominated (doc pr/11")
    print("   sec 90), so CORE is the number to compare; wall is reported beside it.")
    for s in SAMPLES + ("ALL",):
        for nm, R in ((LA, A), (LB, B)):
            sub = [r for r in R.values() if s == "ALL" or r.get("sample") == s]
            w = [fnum(r, "wall_s") for r in sub if fnum(r, "wall_s") is not None]
            c = [fnum(r, "core_s") for r in sub if fnum(r, "core_s") is not None]
            m = [fnum(r, "maxrss_kb") / 1048576.0 for r in sub
                 if fnum(r, "maxrss_kb") is not None]
            if not w and not m:
                continue
            sw, sc, sm = stats(w), stats(c), stats(m)
            print(f"  {s:9s} {nm:10s} wall n={sw['n']:5d} med={sw['med']:.1f}s "
                  f"p90={sw['p90']:.1f} max={sw['hi']:.1f} sum={sum(w)/3600:.2f}h | "
                  f"core med={sc['med']:.1f}s p90={sc['p90']:.1f} sum={sum(c)/3600:.2f}h | "
                  f"peakRSS med={sm['med']:.2f}GiB p90={sm['p90']:.2f} max={sm['hi']:.2f}")

    if a.summary_tsv:
        with open(a.summary_tsv, "w") as f:
            f.write("sample\tarm\trows\tevaluated\tdegenerate\tclean\n")
            for r in summary_rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"\n# wrote {a.summary_tsv}")


if __name__ == "__main__":
    main()
