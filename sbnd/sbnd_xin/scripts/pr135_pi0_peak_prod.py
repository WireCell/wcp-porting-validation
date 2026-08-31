#!/usr/bin/env python3
"""doc pr/135: the pi0 mass-peak fit at the FINAL PRODUCTION point.

Owner (2026-08-31): *"please then do the production and check the mass-peak
fit to see if our scaling factor is good or not."*

Forked in spirit from pr132_pi0_peak_refresh.py (which refit the HAND-SCAN
masses on the enlarged truth set).  What is new here: the energies are the
PRODUCTION reco energies read off the flipped-config arm, so this measures
the scale actually in force.

THE TRAP THIS SCRIPT AVOIDS.  The obvious "fit the accepted pi0 groups'
masses" is CIRCULAR: id_pi0_with_vertex only accepts pairs inside
(100,160) MeV, so the accepted-mass distribution is a window cut-out and
its peak reports the window centre, not the energy scale.  Cell C prints
it anyway, labelled, so the circularity is visible rather than hidden.

THE ESTIMATOR.  The FIXED-PAIRING mass: take the hand pairing (which two
reco showers are the two gammas) and the hand opening angle
theta_vertex_convention, substitute the PRODUCTION kine_charge of the two
matched showers:

    m_prod = sqrt(4 * E1_prod * E2_prod * sin^2(theta_hand/2))

No acceptance window enters (the pair need not have been accepted), and no
reco-angle bias enters (the campaign's measured angle compression would
push the mass down for reasons that have nothing to do with the scale).
What is left moving is exactly the energy scale plus the EM-clustering
charge deficit -- which is why cell B repeats the fit on the gammas the
charge ledger calls OK, separating "the scale is wrong" from "the shower
lost charge".

Peak estimator, bootstrap and to_fudge() are pr126_pi0_peak.py's, reused
unchanged (bounded unbinned truncated-Gaussian ML on [100,185]).

THE DIRECTION OF THE FUDGE.  kine_shower_fudge_factor DIVIDES the charge
(NeutrinoEnergyReco.cxx:188,508: `overall / recom_factor / fudge_factor`),
so a peak ABOVE 135 MeV means the energies are too big and the fudge must
go UP.  implied_fudge = fudge_in_force * peak / 134.9768.

READ-ONLY.

    ./pr135_pi0_peak_prod.py --manifest98 <tsv> --manifest141 <tsv> \
        --overlay-tag pi0scan-0829-agent --fudge 0.84 [--tsv out.tsv]
"""
import argparse, csv, importlib.util, json, math, os
import numpy as np

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEL = _load("pr126_pi0_select", os.path.join(SD, "pr126_pi0_select.py"))
PK = _load("pr126_pi0_peak", os.path.join(SD, "pr126_pi0_peak.py"))

LEDGER_OK = (0.80, 1.25)   # pr132_gamma_ledger.py's OK band


def cell(name, xs, fudge, note=""):
    """Fit one cell.  median_in is the IN-WINDOW median: the peak>=median
    sanity check of pr126_pi0_peak.py is only meaningful against the same
    subset the fit saw (the all-sample median includes masses above the
    window that the fit never used)."""
    x = np.asarray([v for v in xs if v > 0], float)
    nan = float("nan")
    if len(x) < 5:
        return dict(cell=name, n=len(x), n_in=0, peak=nan, peak_lo=nan,
                    peak_hi=nan, median=nan, median_in=nan, implied_fudge=nan,
                    implied_lo=nan, implied_hi=nan, sanity="n/a", note=note)
    xin = x[(x >= PK.WIN[0]) & (x <= PK.WIN[1])]
    pk = PK.peak_fit(x)
    lo, hi = PK.boot(x, PK.peak_fit)
    f = lambda m: fudge * m / PK.PI0_MASS
    med_in = float(np.median(xin)) if len(xin) else nan
    return dict(cell=name, n=len(x), n_in=len(xin), peak=pk, peak_lo=lo,
                peak_hi=hi, median=float(np.median(x)), median_in=med_in,
                implied_fudge=f(pk), implied_lo=f(lo), implied_hi=f(hi),
                sanity=("PASS" if pk >= med_in - 1e-9 else "peak<median"),
                note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest98"); ap.add_argument("--manifest141")
    ap.add_argument("--overlay-tag")
    ap.add_argument("--fudge", type=float, default=0.84,
                    help="kine_shower_fudge_factor in force on the arm")
    ap.add_argument("--tsv")
    a = ap.parse_args()

    if a.manifest98 or a.manifest141:
        newsets = []
        for t in SEL.SETS:
            t = list(t)
            if t[0] == "98" and a.manifest98: t[4] = a.manifest98
            if t[0] == "141" and a.manifest141: t[4] = a.manifest141
            newsets.append(tuple(t))
        SEL.SETS = newsets
    overlay = SEL.load_labels(a.overlay_tag) if a.overlay_tag else {}

    rows, m_all, m_clean, m_accept, m_ncpi0 = [], [], [], [], []
    n_pairs = n_absent = n_lowE = n_notheta = 0
    theta_check_max = 0.0
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            dump = SEL.load_json(mrow["dump"])
            if not dump:
                continue
            by = {int(s["id"]): s for s in (dump.get("showers") or [])}
            for labsrc, rec in (("base", labels.get(ev)), ("overlay", overlay.get(ev))):
                pio = (rec or {}).get("pio")
                if not pio:
                    continue
                g = pio.get("gammas") or {}
                if not all(x in g and (g[x].get("energy") or 0) > 0 for x in ("1", "2")):
                    continue
                n_pairs += 1
                th = pio.get("theta_vertex_convention")
                if not th or th <= 0:
                    n_notheta += 1
                    continue
                # self-check: the stored label mass must come back from
                # (label energies, stored theta) -- guards a unit slip.
                el1 = g["1"]["energy"]; el2 = g["2"]["energy"]
                m_lab_chk = math.sqrt(4*el1*el2*math.sin(math.radians(th)/2)**2)
                if pio.get("mass_vertex_convention"):
                    theta_check_max = max(
                        theta_check_max,
                        abs(m_lab_chk - pio["mass_vertex_convention"]))
                sh1 = by.get(int(g["1"].get("shower") or -1))
                sh2 = by.get(int(g["2"].get("shower") or -1))
                if sh1 is None or sh2 is None:
                    n_absent += 1
                    continue
                e1 = sh1.get("kine_charge") or 0.0
                e2 = sh2.get("kine_charge") or 0.0
                if min(e1, e2) <= PK.E_MIN:
                    n_lowE += 1
                    continue
                m = math.sqrt(4*e1*e2*math.sin(math.radians(th)/2)**2)
                r1 = e1/el1 if el1 > 0 else -1
                r2 = e2/el2 if el2 > 0 else -1
                clean = all(LEDGER_OK[0] <= r <= LEDGER_OK[1] for r in (r1, r2))
                m_all.append(m)
                if clean:
                    m_clean.append(m)
                if (rec or {}).get("origin") == "ncpi0":
                    m_ncpi0.append(m)
                # did production accept exactly this pair?
                acc = (int(sh1.get("pio_id", -1)) >= 0 and
                       int(sh1.get("pio_id", -1)) == int(sh2.get("pio_id", -2)))
                if acc:
                    m_accept.append(m)
                rows.append(dict(setname=setname, sample=mrow.get("sample", tag),
                                 event=ev, labelsrc=labsrc,
                                 theta_deg=round(th, 3),
                                 e1_lab=round(el1, 1), e2_lab=round(el2, 1),
                                 e1_prod=round(e1, 1), e2_prod=round(e2, 1),
                                 ratio1=round(r1, 3), ratio2=round(r2, 3),
                                 m_lab=round(m_lab_chk, 1), m_prod=round(m, 1),
                                 origin=(rec or {}).get("origin", ""),
                                 ledger_clean=int(clean), accepted=int(acc)))

    print("pi0 MASS-PEAK FIT AT THE PRODUCTION POINT")
    print("  fudge in force = %.2f;  pi0 mass = %.4f MeV;  fit window [%.0f,%.0f]"
          % (a.fudge, PK.PI0_MASS, PK.WIN[0], PK.WIN[1]))
    print("  hand pi0 seen %d; both gammas matched %d; dropped: %d no-theta, %d absent, %d min(E)<=%.0f"
          % (n_pairs, len(m_all), n_notheta, n_absent, n_lowE, PK.E_MIN))
    print("  theta self-check (max |label mass - mass from stored theta|) = %.3f MeV"
          % theta_check_max)

    cells = [
        cell("A all matched hand pairs", m_all, a.fudge,
             "fixed pairing, hand angle, PRODUCTION energies -- the scale estimator"),
        cell("B ledger-clean gammas only", m_clean, a.fudge,
             "both gammas 0.80<=E_prod/E_lab<=1.25 -- scale without the clustering deficit"),
        cell("C production-accepted subset", m_accept, a.fudge,
             "NOT a scale estimator: the (100,160) acceptance window truncates it"),
        cell("D ncpi0 origin only", m_ncpi0, a.fudge,
             "the pr/126 PRIMARY gate, like-for-like with its 139.8 MeV at fudge 0.80"),
    ]
    print("\n%-28s %4s %5s %9s %19s %8s %-14s %s" %
          ("cell", "n", "n_in", "med(in)", "PEAK [CI68]", "fudge", "[CI68]", "sanity"))
    for c in cells:
        print("%-28s %4d %5d %9.1f  %6.1f [%5.1f,%5.1f] %8.3f [%.3f,%.3f] %s"
              % (c["cell"], c["n"], c["n_in"], c["median_in"], c["peak"],
                 c["peak_lo"], c["peak_hi"], c["implied_fudge"],
                 c["implied_lo"], c["implied_hi"], c["sanity"]))
    for c in cells:
        print("  %-28s %s" % (c["cell"], c["note"]))

    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("\nwrote %s (%d rows)" % (p, len(rows)))
        pc = p.replace(".tsv", "-cells.tsv")
        with open(pc, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(cells[0].keys()))
            w.writeheader(); w.writerows(cells)
        print("wrote %s (%d cells)" % (pc, len(cells)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
