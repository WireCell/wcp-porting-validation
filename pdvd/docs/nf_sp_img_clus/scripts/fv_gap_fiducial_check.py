#!/usr/bin/env python3
"""Validate the compiled PDVD gap fiducial against the arm it was measured on
(doc pdvd/34).

  usage: fv_gap_fiducial_check.py <compiled-configs.json> <arm-dir> [<arm-dir> ...]

  # build the config the same way the doc's Repro block does:
  echo "local g = import 'pgrapher/experiment/protodunevd/crp_gap_fiducial.jsonnet'; g().configs" \
      > /tmp/h.jsonnet && wcsonnet -o /tmp/gapfid.json /tmp/h.jsonnet
  python3 fv_gap_fiducial_check.py /tmp/gapfid.json work/*_d28dlfp

This is the config's test.  It re-implements CompositeFiducial{logic:'or'} over
BoxFiducial exactly as aux/src/{Box,Composite}Fiducial.cxx do (a closed
axis-aligned containment test, OR'd) and reports, over the arm's imaged points:

  * the share of all points each box flags, and the union's share.  A structural
    exclusion volume that flags a large share of the detector's charge is
    mis-built, not conservative -- expect well under 1 %;
  * whether the flagged band actually brackets the measured deficit, by printing
    the point density just inside and just outside each box edge.  A box whose
    inside density equals its outside density is not sitting on any structure.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fv_gap_measure import load  # same loader, same sentinel cut

MM = 10.0  # compiled config is in wire-cell units (mm); Bee points are cm


def boxes_of(cfg_path):
    cfg = json.load(open(cfg_path))
    boxes, comp = {}, None
    for n in cfg:
        if n["type"] == "BoxFiducial":
            b = n["data"]["bounds"]
            boxes[n["name"]] = (
                np.array([b["tail"]["x"], b["tail"]["y"], b["tail"]["z"]]) / MM,
                np.array([b["head"]["x"], b["head"]["y"], b["head"]["z"]]) / MM)
        elif n["type"] == "CompositeFiducial":
            comp = n
    if comp is None:
        raise SystemExit(f"{cfg_path}: no CompositeFiducial")
    if comp["data"]["logic"] != "or":
        raise SystemExit(f"logic is '{comp['data']['logic']}', this check assumes 'or'")
    order = [f.split(":", 1)[1] for f in comp["data"]["fiducials"]]
    return [(nm, boxes[nm]) for nm in order], comp["name"]


def inside(P, lo, hi):
    return np.all((P >= lo) & (P <= hi), axis=1)


def main():
    cfg_path, arms = sys.argv[1], sys.argv[2:]
    if not arms:
        raise SystemExit(__doc__)
    bx, cname = boxes_of(cfg_path)
    x, y, z, q, _ = load(arms)
    P = np.column_stack([x, y, z])
    n = len(P)

    print(f"\n== {cname}: {len(bx)} boxes, {n:,} imaged points ==")
    print(f"{'box':<22}{'flagged':>12}{'share':>9}{'q/bulk':>8}   half-widths (cm)")
    un = np.zeros(n, bool)
    qbulk = float(np.median(q))
    for nm, (lo, hi) in bx:
        m = inside(P, lo, hi)
        un |= m
        hw = 0.5 * (hi - lo)
        qr = float(np.median(q[m])) / qbulk if m.sum() > 20 else float("nan")
        print(f"{nm:<22}{int(m.sum()):>12,}{100*m.mean():>8.3f}%{qr:>8.2f}   "
              + " ".join(f"{v:8.3f}" for v in hw))
    print(f"{'UNION (or)':<22}{int(un.sum()):>12,}{100*un.mean():>8.3f}%")
    verdict = "OK" if un.mean() < 0.01 else "TOO WIDE -- re-derive a width"
    print(f"  union share {'<' if un.mean()<0.01 else '>='} 1%: {verdict}")

    print("\n== does each box sit on a real deficit? "
          "(density just inside vs just outside its edge) ==")
    print(f"{'box':<22}{'axis':>5}{'edge':>10}{'in':>8}{'out':>8}{'in/out':>8}")
    AX = "xyz"
    for nm, (lo, hi) in bx:
        # the seam axis is the thinnest one
        a = int(np.argmin(hi - lo))
        c = 0.5 * (lo[a] + hi[a])
        w = 0.5 * (hi[a] - lo[a])
        # restrict to the box's own extent in the other two axes
        oth = [i for i in range(3) if i != a]
        m = np.ones(n, bool)
        for i in oth:
            m &= (P[:, i] >= lo[i]) & (P[:, i] <= hi[i])
        d = np.abs(P[m, a] - c)
        n_in = int((d <= w).sum())
        n_out = int(((d > w) & (d <= 3 * w + 2.0)).sum())
        rho_in = n_in / (2 * w) if w > 0 else float("nan")
        rho_out = n_out / (2 * (2 * w + 2.0)) if n_out else float("nan")
        print(f"{nm:<22}{AX[a]:>5}{c:>10.3f}{rho_in:>8.0f}{rho_out:>8.0f}"
              f"{rho_in/rho_out if rho_out else float('nan'):>8.2f}")
    print("  in/out well below 1 = the box is on a real deficit.  ~1 = it is not.")
    print("  A box NARROWER than the ctpc point pitch holds no points whatever it")
    print("  sits on, so its 0.00 is not evidence of a deficit -- read it together")
    print("  with the flagged count above: the CRU boxes are inert by construction,")
    print("  which is what doc 34 sec 3 measured and is the honest thing to ship.")


if __name__ == "__main__":
    main()
