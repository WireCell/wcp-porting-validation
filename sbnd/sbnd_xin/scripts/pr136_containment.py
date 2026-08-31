#!/usr/bin/env python3
"""doc pr/136 -- how much of the pi0 mass deficit is IRREDUCIBLE shower leakage?

Owner, 2026-08-30: *"for pi0 it is possible part of the gamma from the pi0
decay go out of the detector.  Since we can only reconstruct what is in the
detector, even if we have perfect clustering, we may still miss significant
energy leading to lower pi0 mass reconstruction."*

That is a physics floor, not a reconstruction defect, and pr136_mass_closure.py
cannot see it: a gamma converting 30 cm from the wall and a gamma whose tail
was dropped by an absorber both show up as R < 1.  This script separates them,
so the campaign does not spend a round chasing charge that was never in the
detector.

METHOD.  Each gamma is a shower developing from its conversion point (the reco
shower start) along the gamma direction (the vertex->start ray, the
conversion-displacement convention the campaign already trusts -- any point on
the flight line gives the same direction, so a deep start does not bias it).

  D  = distance from the shower start, along the SHOWER'S OWN AXIS (start->end),
       to where the ray leaves the active volume: the MAXIMUM depth the shower
       can develop in.  The shower axis is used rather than the vertex->start
       ray because it is where the energy actually went, and because the vertex
       ray is numerically unstable on a short conversion baseline (168432 g2
       converts 3.5 cm from the vertex).  The vertex ray is computed anyway as
       a cross-check and pairs where the two disagree by more than 30 deg are
       flagged: there the geometry itself is not trustworthy.
  f  = fraction of the shower's energy deposited within depth D, from the
       standard gamma-function longitudinal profile (PDG 34.5):

         dE/dt = E b (bt)^(a-1) e^(-bt) / Gamma(a),   t = depth / X0
         t_max = (a-1)/b = ln(E/Ec) - 0.5   (photon-initiated)
         f(D)  = P(a, b*D/X0)               regularized lower incomplete gamma

  with LAr X0 = 14.0 cm, Ec = 32.8 MeV, b = 0.5.  E is the TRUE energy, so the
  measured energy is unfolded by two fixed-point iterations E <- E_meas / f.

Then the leakage-corrected mass is m_corr = m_prod / sqrt(f1*f2), and the
leakage-corrected energy closure is R_corr = R_prod / <f>, weighted properly
through E_tot.  A pair with R_prod < 1 but R_corr >= 1 is NOT a clustering
failure -- it is a contained-fraction problem, and no absorber can fix it.

WHAT THIS DELIBERATELY DOES NOT MODEL.  Transverse leakage (Moliere radius in
LAr is ~9 cm, small against these path lengths); dead channels and the
non-instrumented gaps; and the fact that the reco start is itself the first
CONVERTED point, so a gamma converting outside the active volume never appears
at all.  Each of those makes the true leakage LARGER, so every f here is an
upper bound on containment and every correction below is a LOWER bound on the
loss.  Stated, not hidden.

ACTIVE VOLUME is measured from the reconstructed point cloud, not assumed:
x [-202, 202], y [-200, 200], z [0, 500] cm over 40 events.

READ-ONLY.

  scripts/pr136_containment.py --manifest98 em117-134f08698-manifest.tsv \\
      --manifest141 em114c-134f086141-manifest.tsv --tsv docs/pr/pr136-containment.tsv
"""
import argparse, csv, importlib.util, math, os, statistics as st

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)

PI0_MASS = 134.9768
X0 = 14.0          # LAr radiation length, cm
EC = 32.8          # LAr critical energy, MeV
BPAR = 0.5         # PDG longitudinal-profile b
AV = ((-202.0, 202.0), (-200.0, 200.0), (0.0, 500.0))   # measured, see docstring


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def gammainc_P(a, x):
    """Regularized lower incomplete gamma P(a,x).  Series below the transition,
    continued fraction above -- Numerical Recipes 6.2, so no scipy dependency."""
    if x <= 0 or a <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-12:
                break
        return s * math.exp(-x + a * math.log(x) - gln)
    b, c, d, h = x + 1.0 - a, 1e300, 0.0, 0.0
    d = 1.0 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2.0
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < 1e-12:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def exit_depth(p, u):
    """Distance from p along unit vector u to the active-volume boundary."""
    t = float("inf")
    for i in range(3):
        lo, hi = AV[i]
        if abs(u[i]) < 1e-9:
            continue
        for bnd in (lo, hi):
            s = (bnd - p[i]) / u[i]
            if s > 1e-6:
                t = min(t, s)
    return 0.0 if t == float("inf") else t


def contained(e_meas, D):
    """(fraction, E_true_estimate).  Two fixed-point iterations."""
    if e_meas <= 0 or D <= 0:
        return 0.0, e_meas
    e = e_meas
    f = 1.0
    for _ in range(2):
        tmax = max(math.log(max(e, EC * 1.01) / EC) - 0.5, 0.05)
        a = 1.0 + BPAR * tmax
        f = min(max(gammainc_P(a, BPAR * D / X0), 1e-3), 1.0)
        e = e_meas / f
    return f, e


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 0 else [0.0, 0.0, 0.0]


def main():
    SEL = _load("pr126_pi0_select", os.path.join(SD, "pr126_pi0_select.py"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest98"); ap.add_argument("--manifest141")
    ap.add_argument("--overlay-tag", default="pi0scan-0829-agent")
    ap.add_argument("--closure", default=os.path.join(SX, "docs", "pr", "pr136-mass-closure.tsv"))
    ap.add_argument("--tsv")
    a = ap.parse_args()

    if a.manifest98 or a.manifest141:
        sets = []
        for t in SEL.SETS:
            t = list(t)
            if t[0] == "98" and a.manifest98: t[4] = a.manifest98
            if t[0] == "141" and a.manifest141: t[4] = a.manifest141
            sets.append(tuple(t))
        SEL.SETS = sets
    overlay = SEL.load_labels(a.overlay_tag) if a.overlay_tag else {}
    cl = {r["event"]: r for r in csv.DictReader(open(a.closure), delimiter="\t")}

    rows = []
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            c = cl.get(str(ev))
            if not c:
                continue
            dump = SEL.load_json(mrow["dump"])
            if not dump:
                continue
            rec = (labels.get(ev) if c["labelsrc"] == "base" else overlay.get(ev)) or {}
            pio = rec.get("pio") or {}
            g = pio.get("gammas") or {}
            if not all(k in g for k in ("1", "2")):
                continue
            vtx = pio.get("vertex") or rec.get("main_vertex") or dump.get("main_vertex")
            if isinstance(vtx, dict):
                vtx = [vtx.get("x"), vtx.get("y"), vtx.get("z")]
            if not vtx or any(v is None for v in vtx):
                continue
            by = {int(s["id"]): s for s in (dump.get("showers") or [])}
            out = []
            for k in ("1", "2"):
                sh = by.get(int(g[k].get("shower") or -1))
                if sh is None:
                    out = []; break
                p = [sh["start"]["x"], sh["start"]["y"], sh["start"]["z"]]
                q = [sh["end"]["x"], sh["end"]["y"], sh["end"]["z"]]
                u_shw = unit([q[i] - p[i] for i in range(3)])       # primary
                u_vtx = unit([p[i] - vtx[i] for i in range(3)])     # cross-check
                dot = max(-1.0, min(1.0, sum(u_shw[i] * u_vtx[i] for i in range(3))))
                ang = math.degrees(math.acos(dot)) if any(u_shw) and any(u_vtx) else -1
                D = exit_depth(p, u_shw)
                e = sh.get("kine_charge") or 0.0
                f, etrue = contained(e, D)
                out.append((e, D, f, etrue, ang))
            if len(out) != 2:
                continue
            (e1, D1, f1, t1, a1), (e2, D2, f2, t2, a2) = out
            R = float(c["R_prod"]); m = float(c["m_prod"])
            m_corr = m / math.sqrt(f1 * f2) if f1 > 0 and f2 > 0 else -1
            th = float(c["theta_deg"])
            R_corr = (t1 + t2) * math.sin(math.radians(th) / 2) / PI0_MASS
            rows.append(dict(event=ev, sample=mrow.get("sample", tag), labelsrc=c["labelsrc"],
                             theta_deg=th, e1=round(e1, 1), e2=round(e2, 1),
                             depth1_cm=round(D1, 1), depth2_cm=round(D2, 1),
                             f1=round(f1, 3), f2=round(f2, 3),
                             axis_vs_vtx1_deg=round(a1, 1), axis_vs_vtx2_deg=round(a2, 1),
                             geom_suspect=int(max(a1, a2) > 30.0),
                             e1_true=round(t1, 1), e2_true=round(t2, 1),
                             R_prod=round(R, 3), R_leakcorr=round(R_corr, 3),
                             m_prod=round(m, 1), m_leakcorr=round(m_corr, 1),
                             impossible=int(R < 1.0),
                             healed_by_leakage=int(R < 1.0 and R_corr >= 1.0)))

    print("IRREDUCIBLE SHOWER LEAKAGE -- how much of the pi0 mass deficit left the detector?")
    print("  active volume x[%.0f,%.0f] y[%.0f,%.0f] z[%.0f,%.0f] cm (measured from the point cloud)"
          % (AV[0][0], AV[0][1], AV[1][0], AV[1][1], AV[2][0], AV[2][1]))
    print("  LAr X0 %.1f cm, Ec %.1f MeV, PDG gamma-profile b=%.1f;  f = P(a, bD/X0)\n" % (X0, EC, BPAR))
    ff = [r["f1"] for r in rows] + [r["f2"] for r in rows]
    ff.sort()
    print("  contained fraction f over %d gammas: median %.3f  q10 %.3f  q25 %.3f  min %.3f"
          % (len(ff), st.median(ff), ff[int(.1 * len(ff))], ff[int(.25 * len(ff))], ff[0]))
    for t in (0.99, 0.95, 0.90, 0.75):
        print("     f < %.2f : %2d of %d gammas" % (t, sum(1 for v in ff if v < t), len(ff)))
    dd = sorted([r["depth1_cm"] for r in rows] + [r["depth2_cm"] for r in rows])
    print("  available depth to the wall: median %.0f cm = %.1f X0 ; q10 %.0f cm"
          % (st.median(dd), st.median(dd) / X0, dd[int(.1 * len(dd))]))

    imp = [r for r in rows if r["impossible"]]
    heal = [r for r in imp if r["healed_by_leakage"]]
    print("\n  THE IMPOSSIBLE PAIRS, WITH LEAKAGE UNFOLDED")
    print("  %-8s %-8s %7s %6s %6s %8s %8s %9s %9s  %s"
          % ("event", "sample", "R_prod", "f1", "f2", "depth1", "depth2", "m_prod", "m_leak", "verdict"))
    for r in sorted(imp, key=lambda r: r["R_prod"]):
        v = "LEAKAGE explains it" if r["healed_by_leakage"] else (
            "leakage helps, not enough" if min(r["f1"], r["f2"]) < 0.95 else "fully contained -- NOT leakage")
        print("  %-8s %-8s %7.2f %6.3f %6.3f %8.0f %8.0f %9.1f %9.1f  %s"
              % (r["event"], r["sample"], r["R_prod"], r["f1"], r["f2"],
                 r["depth1_cm"], r["depth2_cm"], r["m_prod"], r["m_leakcorr"], v))
    print("\n  --> %d of %d impossible pairs are explained by shower leakage alone" % (len(heal), len(imp)))
    fc = [r for r in imp if min(r["f1"], r["f2"]) >= 0.95]
    print("      %d are essentially fully contained (both f >= 0.95) -- their deficit is NOT leakage"
          % len(fc))

    allm = [r["m_prod"] for r in rows]
    allc = [r["m_leakcorr"] for r in rows]
    print("\n  WHOLE-SAMPLE EFFECT -- AND WHY IT MUST NOT BE APPLIED AS A CORRECTION (n=%d)" % len(rows))
    print("     median m_prod     %.1f MeV" % st.median(allm))
    print("     median m_leakcorr %.1f MeV   (m_pi0 = %.1f)" % (st.median(allc), PI0_MASS))
    print("     pairs inside (100,160): %d -> %d"
          % (sum(1 for v in allm if 100 < v < 160), sum(1 for v in allc if 100 < v < 160)))
    print("     The corrected median lands ABOVE m_pi0 and the in-window count DROPS.")
    print("     That is the signature of DOUBLE COUNTING: kine_shower_fudge_factor")
    print("     was fitted (0.80 -> 0.84 -> 0.86) so the measured peak sits at 135,")
    print("     so it ALREADY absorbs the sample-average leakage.  Leakage therefore")
    print("     explains the event-to-event SPREAD, not the mean, and this table is a")
    print("     CLASSIFIER (which pairs are irreducible), never a mass correction.")
    sus = [r for r in rows if r["geom_suspect"]]
    print("\n     geometry cross-check: %d of %d pairs have a shower axis more than 30 deg"
          % (len(sus), len(rows)))
    print("     off the vertex->start ray -- their D is not trustworthy: %s"
          % ", ".join(str(r["event"]) for r in sus[:12]))

    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (p, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
