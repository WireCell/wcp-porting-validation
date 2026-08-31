#!/usr/bin/env python3
"""doc pr/136 -- the ABSOLUTE pi0 mass-closure metric, anchored on m_pi0.

WHY THIS EXISTS.  Every energy metric the pi0 campaign shipped is
reco-vs-reco: `pr132_gamma_ledger.py` compares a production `kine_charge`
against a hand label whose energy was itself copied from a scan-time
reconstruction.  A shower missing its downstream tail in BOTH scores OK.
Measured: the ledger's modal ratio is exactly 0.80/fudge -- the EM scale
flip and nothing else (doc pr/136 sec 2).

WHAT REPLACES IT.  The hand labels supply the PAIRING and the opening
angle; PHYSICS supplies the mass.  m_pi0 = 134.9768 MeV is an absolute
anchor that needs neither MC truth nor label energies, so the residual it
measures is a real reconstruction error.  Per hand pair:

    m_prod / m_pi0  =  R * A
      R = E_tot * sin(theta/2) / m_pi0     ENERGY CLOSURE
      A = 2*sqrt(f(1-f)),  f = E1/E_tot    SHARING ASYMMETRY  (A <= 1)

R < 1 is KINEMATICALLY IMPOSSIBLE for a real pi0: since E1*E2 <= E_tot^2/4,
the mass obeys m <= E_tot * sin(theta/2), so no division of the measured
energy reaches m_pi0 at the measured angle.  R and A separate "the pair is
missing total charge" from "one gamma is starved relative to the other" --
a distinction no ratio-against-a-label can make.

THE ANGLE IS EXACT, NOT ESTIMATED.  `theta_vertex_convention` is the angle
between the two vertex->conversion-point rays.  Any point on a gamma's line
of flight gives the same direction from the decay vertex, so a start point
reconstructed too deep does NOT bias theta.  The convention fails only when
the pi0 did not decay at the labelled vertex (a secondary interaction),
which biases theta LOW and therefore R low -- flagged, not corrected.

THE HAND-MARKS RESCUE.  `em.marks_by_shower` / the per-gamma
`energy_marks_detail` record, per SEGMENT, charge the scanner says the
shower should hold ("in") or should not ("out").  Adding that charge to the
production energy asks the decisive question for the whole campaign: IS THE
MISSING CHARGE PRESENT IN THE EVENT?  A pair whose R crosses 1.0 once the
hand-marked segments are added has a lever inside the owner's post-vertex
scope.  A pair that does not, does not.

READ-ONLY.  Reads label JSONs and calib dumps; writes only the --tsv paths.

  scripts/pr136_mass_closure.py \
    --manifest98 em117-134f08698-manifest.tsv \
    --manifest141 em114c-134f086141-manifest.tsv \
    --overlay-tag pi0scan-0829-agent --fudge 0.86 \
    --tsv docs/pr/pr136-mass-closure.tsv
"""
import argparse
import csv
import importlib.util
import math
import os
import statistics as st

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEL = _load("pr126_pi0_select", os.path.join(SD, "pr126_pi0_select.py"))

PI0_MASS = 134.9768          # PDG, MeV -- the anchor
SCAN_FUDGE = 0.80            # every label energy was frozen at this EM scale
WIN = (100.0, 160.0)         # id_pi0_with_vertex acceptance window


def marks(rec):
    """(base, d_in, d_out, d_orphan) in SCAN-TIME MeV for one gamma record.

    `energy` is not usable directly: `energy_includes_marks` is True on 70
    of the 132 gammas and False on 30, and the overlay gammas lack the key
    entirely.  `energy_without_marks` plus the explicit deltas is the only
    representation that means the same thing on every row."""
    base = rec.get("energy_without_marks")
    if base is None:
        base = (rec.get("energy") or 0.0) - (rec.get("energy_marks_delta") or 0.0)
    det = rec.get("energy_marks_detail") or []
    d_in = sum(x.get("energy", 0.0) for x in det if x.get("kind") == "in")
    d_out = sum(x.get("energy", 0.0) for x in det if x.get("kind") == "out")
    return base, d_in, d_out, (rec.get("energy_orphan_delta") or 0.0)


def closure(e1, e2, theta_deg):
    """(m, R, A) for a pair.  R<1 => impossible; A=1 => symmetric sharing."""
    tot = e1 + e2
    if tot <= 0 or theta_deg <= 0:
        return 0.0, 0.0, 0.0
    s = math.sin(math.radians(theta_deg) / 2.0)
    m = math.sqrt(4.0 * e1 * e2) * s
    R = tot * s / PI0_MASS
    f = e1 / tot
    return m, R, 2.0 * math.sqrt(max(f * (1.0 - f), 0.0))


def qs(v):
    v = sorted(v)
    if not v:
        return (float("nan"),) * 5
    g = lambda q: v[min(int(q * len(v)), len(v) - 1)]
    return g(0.10), g(0.25), st.median(v), g(0.75), g(0.90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest98")
    ap.add_argument("--manifest141")
    ap.add_argument("--overlay-tag")
    ap.add_argument("--fudge", type=float, default=0.86,
                    help="kine_shower_fudge_factor in force on the arm")
    ap.add_argument("--tsv")
    a = ap.parse_args()

    if a.manifest98 or a.manifest141:
        sets = []
        for t in SEL.SETS:
            t = list(t)
            if t[0] == "98" and a.manifest98:
                t[4] = a.manifest98
            if t[0] == "141" and a.manifest141:
                t[4] = a.manifest141
            sets.append(tuple(t))
        SEL.SETS = sets
    overlay = SEL.load_labels(a.overlay_tag) if a.overlay_tag else {}

    # scan-time MeV -> this arm's MeV.  The fudge DIVIDES charge
    # (NeutrinoEnergyReco.cxx:188), so a bigger fudge means smaller energies.
    scale = SCAN_FUDGE / a.fudge

    rows = []
    n_pair = n_notheta = n_absent = 0
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            dump = SEL.load_json(mrow["dump"])
            if not dump:
                continue
            by = {int(s["id"]): s for s in (dump.get("showers") or [])}
            for src, rec in (("base", labels.get(ev)), ("overlay", overlay.get(ev))):
                pio = (rec or {}).get("pio")
                if not pio:
                    continue
                g = pio.get("gammas") or {}
                if not all(k in g and (g[k].get("energy") or 0) > 0 for k in ("1", "2")):
                    continue
                n_pair += 1
                th = pio.get("theta_vertex_convention") or 0.0
                if th <= 0:
                    n_notheta += 1
                    continue
                sh1 = by.get(int(g["1"].get("shower") or -1))
                sh2 = by.get(int(g["2"].get("shower") or -1))
                if sh1 is None or sh2 is None:
                    n_absent += 1
                    continue
                e1 = sh1.get("kine_charge") or 0.0
                e2 = sh2.get("kine_charge") or 0.0
                if min(e1, e2) <= 0:
                    n_absent += 1
                    continue

                m_p, R_p, A_p = closure(e1, e2, th)

                # the hand-marks rescue: production energy PLUS the scanner's
                # own per-segment corrections, converted to this arm's scale.
                b1, i1, o1, r1 = marks(g["1"])
                b2, i2, o2, r2 = marks(g["2"])
                d1 = (i1 - o1 + r1) * scale
                d2 = (i2 - o2 + r2) * scale
                m_k, R_k, A_k = closure(max(e1 + d1, 0.0), max(e2 + d2, 0.0), th)

                rows.append(dict(
                    setname=setname, sample=mrow.get("sample", tag), event=ev,
                    labelsrc=src, origin=(rec or {}).get("origin", ""),
                    theta_deg=round(th, 3),
                    e1_prod=round(e1, 1), e2_prod=round(e2, 1),
                    m_prod=round(m_p, 1), R_prod=round(R_p, 3), A_prod=round(A_p, 3),
                    mark_in1=round(i1, 1), mark_out1=round(o1, 1),
                    mark_in2=round(i2, 1), mark_out2=round(o2, 1),
                    d1_arm=round(d1, 1), d2_arm=round(d2, 1),
                    m_marks=round(m_k, 1), R_marks=round(R_k, 3),
                    in_window=int(WIN[0] < m_p < WIN[1]),
                    in_window_marks=int(WIN[0] < m_k < WIN[1]),
                    has_marks=int(bool(d1 or d2)),
                    accepted=int(int(sh1.get("pio_id", -1)) >= 0 and
                                 int(sh1.get("pio_id", -1)) == int(sh2.get("pio_id", -2)))))

    print("pi0 MASS CLOSURE AGAINST m_pi0 = %.4f MeV  (fudge in force %.2f)" % (PI0_MASS, a.fudge))
    print("  hand pi0 seen %d; both gammas matched %d; dropped %d no-theta, %d absent/zero"
          % (n_pair, len(rows), n_notheta, n_absent))
    print("  label marks converted to this arm by x%.4f (= %.2f/%.2f)" % (scale, SCAN_FUDGE, a.fudge))
    if not rows:
        print("  NO ROWS -- check the manifests"); return 1

    print("\nA. THE DECOMPOSITION  m_prod/m_pi0 = R * A")
    print("   %-16s %3s %8s %8s   %s" % ("class", "n", "med R", "med A", "reading"))
    cls = [("m_prod < 100", lambda r: r["m_prod"] < 100, "pure total-energy deficit if A~1"),
           ("100 <= m <= 160", lambda r: 100 <= r["m_prod"] <= 160, "in the acceptance window"),
           ("m_prod > 160", lambda r: r["m_prod"] > 160, "over-clustering / wrong object")]
    for name, sel, note in cls:
        v = [r for r in rows if sel(r)]
        if not v:
            continue
        print("   %-16s %3d %8.2f %8.2f   %s"
              % (name, len(v), st.median([x["R_prod"] for x in v]),
                 st.median([x["A_prod"] for x in v]), note))

    imp = [r for r in rows if r["R_prod"] < 1.0]
    below = [r for r in rows if r["m_prod"] < 135.0]
    print("\nB. WHAT KILLS THE MASS")
    print("   pairs with R < 1 (KINEMATICALLY IMPOSSIBLE): %d of %d" % (len(imp), len(rows)))
    print("   of the %d pairs below 135 MeV: %d killed by ENERGY (R<1), %d by ASYMMETRY (R>=1)"
          % (len(below), sum(1 for r in below if r["R_prod"] < 1.0),
             sum(1 for r in below if r["R_prod"] >= 1.0)))
    print("   outside the (100,160) acceptance window: %d of %d"
          % (sum(1 for r in rows if not r["in_window"]), len(rows)))
    q = qs([r["R_prod"] for r in rows])
    print("   R distribution: q10 %.2f  q25 %.2f  median %.2f  q75 %.2f  q90 %.2f" % q)

    print("\nC. THE DECISIVE TRIAGE -- is the missing charge present as hand-marked segments?")
    print("   %-8s %-8s %7s %8s %9s %9s %s"
          % ("event", "sample", "R_prod", "R_marks", "m_prod", "m_marks", "verdict"))
    healed = []
    for r in sorted(imp, key=lambda x: x["R_prod"]):
        v = ("RESCUED by hand marks" if r["R_marks"] >= 1.0
             else ("marks move it, still impossible" if r["has_marks"] else "no marks -- charge not seen by the scanner"))
        if r["R_marks"] >= 1.0:
            healed.append(r)
        print("   %-8s %-8s %7.2f %8.2f %9.1f %9.1f %s"
              % (r["event"], r["sample"], r["R_prod"], r["R_marks"],
                 r["m_prod"], r["m_marks"], v))
    print("   --> %d of %d impossible pairs are RESCUED by charge the scanner says belongs to the gamma"
          % (len(healed), len(imp)))
    print("       %d gain nothing: their deficit is NOT mis-assigned charge the scanner saw"
          % (len(imp) - len(healed)))
    nom = [r for r in imp if not r["has_marks"]]
    print("       (of those, %d carry no marks at all)" % len(nom))

    print("\nD. WINDOW RECOVERY IF EVERY HAND MARK WERE HONOURED")
    print("   pairs inside (100,160): %d -> %d   (of %d)"
          % (sum(r["in_window"] for r in rows), sum(r["in_window_marks"] for r in rows), len(rows)))

    print("\nE. THE MARKS THEMSELVES (the hand-scan under/over-clustering ledger)")
    mi = [r["mark_in1"] for r in rows if r["mark_in1"]] + [r["mark_in2"] for r in rows if r["mark_in2"]]
    mo = [r["mark_out1"] for r in rows if r["mark_out1"]] + [r["mark_out2"] for r in rows if r["mark_out2"]]
    print("   gammas with marked-IN charge  : %3d   median %6.1f MeV   sum %7.0f MeV"
          % (len(mi), st.median(mi) if mi else 0, sum(mi)))
    print("   gammas with marked-OUT charge : %3d   median %6.1f MeV   sum %7.0f MeV"
          % (len(mo), st.median(mo) if mo else 0, sum(mo)))
    print("   pairs carrying at least one mark: %d of %d"
          % (sum(r["has_marks"] for r in rows), len(rows)))

    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (p, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
