#!/usr/bin/env python3
"""pi0 invariant-mass distribution and the EM charge-scale fit (doc pr/126 sec 4).

The reconstructed pi0 mass is `m = sqrt(4 E1 E2 sin^2(theta/2))` with
`E = kine_charge` at every one of the code's three mass sites
(`NeutrinoShowerClustering.cxx:5113`, `:5535`, `:5581`).  `kine_charge` for a
shower-flagged object is

    E = Q / recom / fudge * w_value * 1e-6 MeV,
        recom = kine_shower_recom_factor = 0.58 (SBND, doc pr/10 sec 6)
        fudge = kine_shower_fudge_factor = 0.80 (uBooNE, NEVER re-derived)

so **m is exactly linear in the EM charge scale**.  If the sample's true mass is
134.9768 MeV and we measure m_hat, then

    k = 134.9768 / m_hat              scale correction on the energy
    fudge_new = 0.80 * m_hat / 134.9768

THIS SCRIPT CHANGES NO CODE AND FLIPS NO KNOB.  It reads label files and calib
dumps only (CLAUDE.md M13) and prints a recommendation.

    ./pr126_pi0_mass.py --selftest
    ./pr126_pi0_mass.py --tsv docs/pr/pr126-pi0-mass.tsv
    ./pr126_pi0_mass.py --e2                 # untruncated pair enumeration
    ./pr126_pi0_mass.py --e2 --e2-tsv docs/pr/pr126-pi0-pairs.tsv

TWO ESTIMATORS, BOTH PRE-REGISTERED BEFORE ANY NUMBER WAS READ (CLAUDE.md 5.7).

E1  hand-paired, clean, low N.  Median of the gated hand-pi0 sample on the
    vertex-chord convention, bootstrap 68% CI.  The scanner's PAIRING is the
    truth input; the geometry (axis, start, decay vertex) is the scanner's too
    and is HELD FIXED between arms, so the scan-time -> current-arm difference
    is the ENERGY alone.  Not mass-windowed: the scanner chose the pair from
    topology, so this sample is free of the (100,160)/(65,185) truncation that
    makes the reco-accepted spectrum unfittable.

E2  offline, untruncated, high N.  Every EM shower pair in all 239 events of
    both manifests, with the code's own `pio_kine` direction convention
    (vertex chord, or the 15 cm axis when the start is within 3 cm) and NO mass
    window.  Peak over combinatorial background.

The recommendation stands only if E1 and E2 agree.  If they disagree the doc
reports both and recommends no flip (CLAUDE.md 5.5).
"""
import argparse, csv, json, math, os, random, sys
from collections import Counter, defaultdict

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))
sys.path.insert(0, os.path.join(SX, "scripts"))
import em_geom as G                                          # noqa: E402
import importlib.util                                        # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(SX, "scripts", "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

PI0_MASS = 134.9768          # PDG neutral-pion mass, MeV
FUDGE_NOW = 0.80             # kine_shower_fudge_factor, uBooNE, in force on SBND
RECOM_NOW = 0.58             # kine_shower_recom_factor, SBND (doc pr/10 sec 6)
E_MIN = 15.0                 # the code's OWN tagger threshold, NeutrinoTaggerNuE.cxx:695


# ------------------------------------------------------------------ stats
def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def boot_ci(xs, n=4000, seed=126):
    """68% bootstrap CI on the median.  Fixed seed -> reproducible number."""
    if len(xs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = []
    for _ in range(n):
        meds.append(median([xs[rng.randrange(len(xs))] for _ in range(len(xs))]))
    meds.sort()
    return (meds[int(0.16 * n)], meds[int(0.84 * n)])


def to_fudge(m):
    return FUDGE_NOW * m / PI0_MASS


# ------------------------------------------------------------------ E1
def hand_rows():
    """One row per hand-paired pi0, with masses on three energy hypotheses.

    Geometry is the scanner's and is IDENTICAL across the three, so the only
    thing that moves is E1,E2:
      m_scanreco  E = the scan-time reconstruction's own kine_charge
      m_scanhand  E = the scan-time value corrected by the scanner's IN/OUT
                      marks and orphan additions (label `gammas[i].energy`)
      m_now       E = the CURRENT arm's kine_charge for the same shower
    """
    rows = []
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man_cur = SEL.load_manifest(m_cur)
        buckets = SEL.load_buckets(buck)
        for ev in sorted(labels):
            rec = labels[ev]
            pio = rec.get("pio") or {}
            g = pio.get("gammas") or {}
            if not all(k in g and (g[k].get("energy") or 0) > 0 for k in ("1", "2")):
                continue
            v = pio.get("vertex")
            a1, a2 = g["1"].get("axis"), g["2"].get("axis")
            s1, s2 = g["1"].get("start"), g["2"].get("start")
            th_axis = G.angle_deg(a1, a2) if (a1 and a2) else None
            th_vtx = (G.angle_deg(G.vsub(s1, v), G.vsub(s2, v))
                      if (v and s1 and s2) else None)

            e_hand = (g["1"]["energy"], g["2"]["energy"])
            e_reco = (g["1"].get("energy_as_reconstructed"), g["2"].get("energy_as_reconstructed"))

            # current arm: same shower id (99 of 100 survive), else the largest
            # charge-weighted member overlap with the scanner's target set.
            dump = SEL.load_json(man_cur[ev]["dump"]) if ev in man_cur else None
            e_now, matched = (None, None), (None, None)
            if dump:
                by_id = {int(s["id"]): s for s in (dump.get("showers") or ())}
                actual, seginfo, _grp, _kine, _sh = SEL.digest(
                    dump, SEL.load_prep(p_cur, ev))
                md = (rec.get("em") or {}).get("marks_detail") or {}
                en, mt = [], []
                for k in ("1", "2"):
                    sid = int(g[k]["shower"])
                    if sid in by_id:
                        en.append(by_id[sid].get("kine_charge"))
                        mt.append(sid)
                        continue
                    det = md.get(str(sid)) or {}
                    marked = det.get("marked") or {}
                    target = ((set(int(x) for x in (det.get("members") or ()))
                               | {int(s) for s, m in marked.items() if m.get("kind") == "in"})
                              - {int(s) for s, m in marked.items() if m.get("kind") == "out"})
                    best, bestq = None, 0.0
                    for cand, mem in actual.items():
                        q = sum(seginfo.get(i, {}).get("charge", 0.0) for i in (mem & target))
                        if q > bestq:
                            best, bestq = cand, q
                    en.append(by_id.get(best, {}).get("kine_charge"))
                    mt.append(best)
                e_now, matched = tuple(en), tuple(mt)

            b = buckets.get(ev) or {}
            row = dict(setname=setname, event=ev, sample=rec.get("sample"),
                       origin=rec.get("origin"), vertex_how=pio.get("vertex_how"),
                       bucket=b.get("bucket", ""), bucket_pi0=b.get("pi0", ""),
                       theta_axis=th_axis, theta_vertex=th_vtx,
                       e1_hand=e_hand[0], e2_hand=e_hand[1],
                       e1_reco=e_reco[0], e2_reco=e_reco[1],
                       e1_now=e_now[0], e2_now=e_now[1],
                       g1_shower=g["1"].get("shower"), g2_shower=g["2"].get("shower"),
                       g1_matched=matched[0], g2_matched=matched[1],
                       label_mass_axis=pio.get("mass_axis_convention"),
                       label_mass_vertex=pio.get("mass_vertex_convention"))
            for tagname, (ea, eb) in (("scanhand", e_hand), ("scanreco", e_reco),
                                      ("now", e_now)):
                row["m_axis_" + tagname] = (G.pi0_mass(ea, eb, th_axis)
                                            if (ea and eb and th_axis is not None) else None)
                row["m_vtx_" + tagname] = (G.pi0_mass(ea, eb, th_vtx)
                                           if (ea and eb and th_vtx is not None) else None)
            rows.append(row)
    return rows


def agree(r, tol, which="scanhand"):
    a, v = r.get("m_axis_" + which), r.get("m_vtx_" + which)
    if not a or not v:
        return False
    return abs(a - v) / (0.5 * (a + v)) < tol


# energy-hypothesis name -> the row's energy column suffix
EK = {"scanhand": "hand", "scanreco": "reco", "now": "now"}


def report_e1(rows, which="scanhand"):
    mk = "m_vtx_" + which
    ek = ("e1_" + EK[which], "e2_" + EK[which])

    def sel(pred):
        return [r[mk] for r in rows if r.get(mk) and pred(r)]

    def line(name, xs):
        if not xs:
            print("  %-44s n=  0" % name)
            return
        m = median(xs)
        lo, hi = boot_ci(xs)
        print("  %-44s n=%3d  med=%6.1f  CI68=[%.1f,%.1f]  k=%.4f  fudge=%.3f [%.3f,%.3f]"
              % (name, len(xs), m, lo, hi, PI0_MASS / m, to_fudge(m), to_fudge(lo), to_fudge(hi)))

    have_e = lambda r: (r.get(ek[0]) or 0) > 0 and (r.get(ek[1]) or 0) > 0
    emin = lambda r: min(r[ek[0]], r[ek[1]]) > E_MIN
    nc = lambda r: r["origin"] == "ncpi0"

    print("\n=== E1  energy hypothesis = %s, vertex-chord convention ===" % which)
    line("ungated (all hand pairs)", sel(lambda r: True))
    line("PRIMARY: origin==ncpi0 + min(E)>15 MeV",
         sel(lambda r: nc(r) and have_e(r) and emin(r)))
    line("cross-check: all origins + min(E)>15 MeV",
         sel(lambda r: have_e(r) and emin(r)))
    line("cross-check: NOT ncpi0 + min(E)>15 MeV",
         sel(lambda r: not nc(r) and have_e(r) and emin(r)))
    print("  -- direction-quality scan (systematic, NOT adopted as a gate) --")
    for tol in (0.10, 0.15, 0.20, 0.30, 0.50):
        line("  ncpi0 + minE + |dm|/m < %.2f" % tol,
             sel(lambda r, t=tol: nc(r) and have_e(r) and emin(r) and agree(r, t, which)))
    print("  -- same, shower-axis convention (the other systematic arm) --")
    xs = [r["m_axis_" + which] for r in rows
          if r.get("m_axis_" + which) and nc(r) and have_e(r) and emin(r)]
    line("  ncpi0 + minE, AXIS convention", xs)


# ------------------------------------------------------------------ E2
def e2_pairs(dump, prep=None):
    """Every EM shower pair, the code's own `pio_kine` direction convention,
    NO mass window.

    dir_i = (start_i - main_vertex) when |start_i - vtx| >= 3 cm, else the
    15 cm shower axis anchored at its own start vertex -- exactly
    `NeutrinoShowerClustering.cxx:5145-5153` (and the prototype's
    `NeutrinoID_shower_clustering.h:868-879`).
    """
    mv = dump.get("main_vertex")
    if not mv:
        return []
    v = (mv["x"], mv["y"], mv["z"])
    segs = dump.get("segments") or []
    vtx_by_id = {int(x["id"]): x for x in (dump.get("vertices") or ())}
    cands = []
    for s in (dump.get("showers") or ()):
        e = s.get("kine_charge")
        if not e or e <= E_MIN:
            continue
        if abs(int(s.get("particle_id") or 0)) != 11:
            continue
        sp = G.pt(s.get("start"))
        d = G.vsub(sp, v)
        if G.vmag(d) < 3.0:
            sv = vtx_by_id.get(int(s.get("start_vertex_id", -1)))
            anchor = G.pt(sv["fit"]) if (sv and sv.get("fit")) else sp
            d = G.shower_cal_dir_3vector(G.shower_members(s, segs), anchor, 15.0)
        if G.vmag(d) == 0:
            continue
        cands.append((s, d, e))
    out = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            (s1, d1, e1), (s2, d2, e2) = cands[i], cands[j]
            th = G.angle_deg(d1, d2)
            m = G.pi0_mass(e1, e2, th)
            if m:
                out.append(dict(g1=s1["id"], g2=s2["id"], e1=e1, e2=e2,
                                theta=th, mass=m,
                                paired=int(s1.get("pio_id", -1) >= 0
                                           and s1.get("pio_id") == s2.get("pio_id"))))
    return out


def peak_fit(masses, lo=60.0, hi=300.0, nb=48):
    """Peak position over a linear background, by a coarse scan.

    Deliberately crude and stated as such: with a few hundred pairs a shape fit
    would over-promise.  The estimator is the bin-weighted mode of the
    background-subtracted histogram in a +-40 MeV window, sidebands taken
    OUTSIDE that window from the same histogram.
    """
    w = (hi - lo) / nb
    h = [0] * nb
    for m in masses:
        if lo <= m < hi:
            h[int((m - lo) / w)] += 1
    if sum(h) < 20:
        return None, h, lo, w
    # linear background from the two outer thirds
    idx = list(range(nb))
    side = [i for i in idx if (lo + (i + .5) * w) < 95 or (lo + (i + .5) * w) > 185]
    if len(side) < 4:
        return None, h, lo, w
    sx = sum(side); sy = sum(h[i] for i in side)
    sxx = sum(i * i for i in side); sxy = sum(i * h[i] for i in side)
    n = len(side)
    den = n * sxx - sx * sx
    b1 = (n * sxy - sx * sy) / den if den else 0.0
    b0 = (sy - b1 * sx) / n
    sig = [(i, h[i] - (b0 + b1 * i)) for i in idx if 95 <= lo + (i + .5) * w <= 185]
    tot = sum(max(0.0, s) for _, s in sig)
    if tot <= 0:
        return None, h, lo, w
    cen = sum(max(0.0, s) * (lo + (i + .5) * w) for i, s in sig) / tot
    return cen, h, lo, w


def report_e2(args):
    rows = []
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            d = SEL.load_json(mrow["dump"])
            if not d:
                continue
            for p in e2_pairs(d, SEL.load_prep(p_cur, ev)):
                p.update(setname=setname, event=ev, sample=mrow["sample"])
                rows.append(p)
    masses = [r["mass"] for r in rows]
    print("\n=== E2  untruncated candidate-pair enumeration, current arms ===")
    print("  events scanned : %d      candidate pairs: %d"
          % (len({r['event'] for r in rows}), len(rows)))
    print("  reco-accepted among them: %d" % sum(r["paired"] for r in rows))
    cen, h, lo, w = peak_fit(masses)
    if cen:
        print("  bkg-subtracted peak in (95,185): %6.1f MeV  ->  k=%.4f  fudge=%.3f"
              % (cen, PI0_MASS / cen, to_fudge(cen)))
    else:
        print("  peak fit: NOT ENOUGH PAIRS -- reported as unresolved, not forced")
    print("  histogram (%.0f MeV bins from %.0f):" % (w, lo))
    mx = max(h) or 1
    for i, c in enumerate(h):
        if lo + i * w > 300:
            break
        print("   %6.0f | %-40s %d" % (lo + i * w, "#" * int(40 * c / mx), c))
    acc = [r["mass"] for r in rows if r["paired"]]
    if acc:
        print("  reco-ACCEPTED subset (TRUNCATED by the mass window -- not fittable):"
              " n=%d med=%.1f" % (len(acc), median(acc)))

    # ---- POST-HOC variant, labelled as such (CLAUDE.md 5.7) ----------------
    # E2 as pre-registered fails: the inclusive spectrum is combinatorics that
    # rises monotonically toward low mass, so a linear-sideband peak fit has
    # nothing to find.  The one sub-sample where the pairing is UNAMBIGUOUS --
    # events with exactly two EM showers above threshold, so there is a single
    # candidate pair and no combinatorial background at all -- is reported
    # separately.  It was NOT the pre-registered estimator and is quoted as an
    # exploratory cross-check only.
    per_ev = defaultdict(list)
    for r in rows:
        per_ev[(r["setname"], r["event"])].append(r)
    uniq = [v[0]["mass"] for v in per_ev.values() if len(v) == 1]
    if uniq:
        m = median(uniq)
        lo, hi = boot_ci(uniq)
        print("  POST-HOC (not the pre-registered estimator): events with exactly ONE")
        print("           candidate pair -> n=%d med=%.1f CI68=[%.1f,%.1f] k=%.4f fudge=%.3f"
              % (len(uniq), m, lo, hi, PI0_MASS / m, to_fudge(m)))
    if args.e2_tsv:
        p = args.e2_tsv if os.path.isabs(args.e2_tsv) else os.path.join(SX, args.e2_tsv)
        with open(p, "w", newline="") as fh:
            w2 = csv.DictWriter(fh, delimiter="\t",
                                fieldnames=["setname", "sample", "event", "g1", "g2",
                                            "e1", "e2", "theta", "mass", "paired"])
            w2.writeheader()
            for r in rows:
                w2.writerow(r)
        print("  wrote %s" % p)
    return cen


# ------------------------------------------------------------------ selftest
def selftest():
    ok = True
    rows = hand_rows()
    print("%s  hand rows = %d (expect 50)" % ("OK " if len(rows) == 50 else "FAIL", len(rows)))
    ok &= len(rows) == 50

    # FIDELITY GATE on the em_geom reuse: recomputing the two conventions from
    # the label's own geometry + energies must reproduce the stored masses.
    worst_a = worst_v = 0.0
    na = nv = 0
    for r in rows:
        for got, want, acc in ((r["m_axis_scanhand"], r["label_mass_axis"], "a"),
                               (r["m_vtx_scanhand"], r["label_mass_vertex"], "v")):
            if got is None or want is None:
                continue
            d = abs(got - want) / want
            if acc == "a":
                worst_a = max(worst_a, d); na += 1
            else:
                worst_v = max(worst_v, d); nv += 1
    print("%s  axis-convention round-trip: n=%d worst rel err = %.2e (expect <1e-9)"
          % ("OK " if worst_a < 1e-9 else "FAIL", na, worst_a))
    print("%s  vtx-convention  round-trip: n=%d worst rel err = %.2e (expect <1e-9)"
          % ("OK " if worst_v < 1e-9 else "FAIL", nv, worst_v))
    ok &= worst_a < 1e-9 and worst_v < 1e-9

    # LINEARITY: m(k) == k*m(1) exactly, which is what makes one arm enough to
    # scan every trial scale.
    bad = 0
    for r in rows[:20]:
        if not (r["e1_hand"] and r["e2_hand"] and r["theta_vertex"] is not None):
            continue
        m1 = G.pi0_mass(r["e1_hand"], r["e2_hand"], r["theta_vertex"])
        for k in (0.8, 0.93, 1.07):
            mk = G.pi0_mass(k * r["e1_hand"], k * r["e2_hand"], r["theta_vertex"])
            if abs(mk - k * m1) > 1e-9 * m1:
                bad += 1
    print("%s  m(k) == k*m(1) violations = %d (expect 0)" % ("OK " if not bad else "FAIL", bad))
    ok &= bad == 0

    n_now = sum(1 for r in rows if r["e1_now"] and r["e2_now"])
    print("INFO current-arm energies resolved on %d of 50 rows" % n_now)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    ap.add_argument("--e2", action="store_true")
    ap.add_argument("--e2-tsv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rows = hand_rows()
    print("pi0 mass = 134.9768 MeV;  in force: kine_shower_recom_factor=%.2f "
          "kine_shower_fudge_factor=%.2f" % (RECOM_NOW, FUDGE_NOW))
    for which in ("scanhand", "scanreco", "now"):
        report_e1(rows, which)
    print("\n  origin composition:", Counter(r["origin"] for r in rows))
    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        cols = sorted(rows[0].keys())
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nwrote %s (%d rows)" % (p, len(rows)))
    if a.e2 or a.e2_tsv:
        report_e2(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
