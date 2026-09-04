#!/usr/bin/env python3
"""Measure PDVD's CRP/CRU/cathode gaps from data, to build a fiducial on (doc pdvd/35).

  usage: fv_gap_measure.py [--sbnd <sbnd-arm-dir> ...] <pdvd-arm-dir> [<pdvd-arm-dir> ...]
         fv_gap_measure.py work/*_d28dlfp

Successor to fv_gap_profile.py (doc 33 sec 5), which pooled every instance of a
seam into one profile and could not resolve the z seam at all.  Four changes:

  * the seam positions are PARSED from the AnodePlane 'sensvol' lines the PR log
    prints -- nothing about the geometry is typed in here;
  * every seam is measured PER INSTANCE (per drift volume, per sign), so a dip
    is only called structural when both instances show it;
  * A2, the per-crossing-track gap length, replaces pooled density as the number
    the exclusion box gets built at.  Pooled density mixes tracks that cross the
    seam with tracks that never go near it; the gap a crossing track actually
    loses is the width an exclusion volume is for;
  * every width is quoted against a CONTROL measured the same way at a plane
    with no structure, so the point-spacing floor is visible rather than assumed.

Reads only Bee mabc-pr.zip + wct_pr_*.log of arms already on disk.  Nothing is
written into any work/ dir.
"""
import argparse, glob, json, os, re, zipfile
import numpy as np

SENT = 1e4          # |x| sentinel: clusters with no t0 are written at 1.48e8 cm
CM = 1.0            # Bee coordinates are cm; sensvol lines are mm


# ---------------------------------------------------------------- geometry ---
def read_sensvol(arm_dirs):
    """The 16 (2) per-(anode,face) sensitive boxes, in cm, from any PR log."""
    pat = re.compile(r"sensvol: \[\(([-\d.e+ ]+)\) --> \(([-\d.e+ ]+)\)\]")
    for d in arm_dirs:
        for log in sorted(glob.glob(os.path.join(d, "wct_pr_*.log"))):
            boxes = []
            with open(log, errors="ignore") as fh:
                for line in fh:
                    m = pat.search(line)
                    if m:
                        lo = [float(v) / 10.0 for v in m.group(1).split()]
                        hi = [float(v) / 10.0 for v in m.group(2).split()]
                        boxes.append((lo, hi))
            if boxes:
                return boxes
    raise SystemExit("no 'sensvol' lines found in any arm log")


def axis_gaps(boxes, axis, keep=lambda b: True):
    """Gaps along `axis` (0/1/2) in the union of the kept boxes: [(lo,hi), ...]."""
    iv = sorted({(b[0][axis], b[1][axis]) for b in boxes if keep(b)})
    gaps, cur = [], iv[0]
    for lo, hi in iv[1:]:
        if lo > cur[1] + 1e-9:
            gaps.append((cur[1], lo))
            cur = (lo, hi)
        else:
            cur = (cur[0], max(cur[1], hi))
    return gaps


def describe_geometry(boxes):
    """Print the CRP/CRU tiling the seams come from, and return the seam list."""
    xs = sorted({(b[0][0], b[1][0]) for b in boxes})
    ys = sorted({(b[0][1], b[1][1]) for b in boxes})
    zs = sorted({(b[0][2], b[1][2]) for b in boxes})
    print("== geometry, from the sensvol lines (cm) ==")
    print(f"  {len(boxes)} sensitive boxes; x tiles {len(xs)}, y tiles {len(ys)}, z tiles {len(zs)}")
    for lbl, iv in (("x", xs), ("y", ys), ("z", zs)):
        print(f"  {lbl}: " + ", ".join(f"[{a:.3f},{b:.3f}]" for a, b in iv))

    seams = []
    for lo, hi in axis_gaps(boxes, 0):
        seams.append(dict(name="cathode", axis="x", lo=lo, hi=hi, sel=None))
    for side, keep, tag in ((-1, lambda b: b[1][0] < 0, "bot"),
                            (+1, lambda b: b[0][0] > 0, "top")):
        for lo, hi in axis_gaps(boxes, 1, keep):
            c = 0.5 * (lo + hi)
            nm = "crp_y0" if abs(c) < 1.0 else ("cru_y_pos" if c > 0 else "cru_y_neg")
            seams.append(dict(name=f"{nm}_{tag}", axis="y", lo=lo, hi=hi, sel=side))
        for lo, hi in axis_gaps(boxes, 2, keep):
            seams.append(dict(name=f"cru_z_{tag}", axis="z", lo=lo, hi=hi, sel=side))
    # de-duplicate the y=0 seam across drifts only if both drifts give the same
    # interval; keep them separate otherwise (they are separate structures).
    return seams


# ------------------------------------------------------------------- data ---
def load(arm_dirs, zipname="mabc-pr.zip", member="data/0/0-clustering-global.json"):
    X, Y, Z, Q, C = [], [], [], [], []
    nsent = nfile = 0
    base = 0
    for arm in arm_dirs:
        try:
            d = json.loads(zipfile.ZipFile(os.path.join(arm, zipname)).read(member))
        except Exception:
            continue
        nfile += 1
        x, y, z, q = (np.asarray(d[k], float) for k in ("x", "y", "z", "q"))
        c = np.asarray(d.get("cluster_id", np.zeros(len(x))), float)
        ok = np.abs(x) < SENT
        nsent += int((~ok).sum())
        X.append(x[ok]); Y.append(y[ok]); Z.append(z[ok]); Q.append(q[ok])
        C.append(c[ok] + base)                      # cluster ids are per event
        base += (c.max() + 1) if len(c) else 0
    if not nfile:
        raise SystemExit("no mabc-pr.zip read")
    print(f"== data: {nfile} events, {sum(len(a) for a in X):,} points "
          f"({nsent:,} no-t0 sentinel points dropped) ==")
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(Z),
            np.concatenate(Q), np.concatenate(C))


AX = {"x": 0, "y": 1, "z": 2}


def drift_mask(x, sel):
    if sel is None:
        return np.ones(len(x), bool)
    return (x < 0) if sel < 0 else (x > 0)


# --------------------------------------------------------- A1/A3: profiles ---
# The imaged points sit on the ctpc LATTICE, so a density profile binned finer
# than the point pitch is a comb (alternating 3.4 / 0.0), not a density -- doc
# 33 sec 5 read widths off exactly such a comb.  Everything below is therefore
# either integrated (R(w)) or binned at 0.5 cm, which is coarser than the pitch.

WS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


def ratio(v, centre, half, ws=WS):
    """R(w) = observed points with |d| < w, over the far-field expectation.

    The expectation is the linear density measured in |d| in (half/2, half),
    scaled to the width 2w.  R -> 1 far from any structure by construction.
    """
    d = np.abs(v - centre)
    far_lo, far_hi = half / 2.0, half
    nfar = int(((d > far_lo) & (d < far_hi)).sum())
    lin = nfar / (2.0 * (far_hi - far_lo)) if nfar else float("nan")
    return [float((d < w).sum()) / (2.0 * w * lin) if lin else float("nan") for w in ws], nfar


def qratio(v, q, centre, core):
    """median point charge inside |d| < core, over the far-field median."""
    d = np.abs(v - centre)
    inside, far = q[d < core], q[(d > core) & (d < 10 * core)]
    if len(inside) < 20 or len(far) < 20:
        return float("nan"), len(inside)
    return float(np.median(inside) / np.median(far)), len(inside)


def coarse_profile(v, centre, half=6.0, step=0.5):
    d = v - centre
    m = np.abs(d) < half
    edges = np.arange(-half, half + step / 2, step)
    h, _ = np.histogram(d[m], bins=edges)
    ctr = edges[:-1] + step / 2
    far = h[np.abs(ctr) > half / 2.0]
    return ctr, h / (far.mean() if far.mean() else 1.0)


# ------------------------------------------- A2: per-crossing-track gap len ---
def crossing_gaps(v, other1, other2, cid, centre, reach=5.0, window=6.0,
                  cosmin=0.3, minpts=20):
    """For clusters that straddle `centre` by >= reach on both sides and are not
    running along the seam, the empty interval in `v` that brackets `centre`."""
    m = np.abs(v - centre) < 40.0
    v, o1, o2, cid = v[m], other1[m], other2[m], cid[m]
    order = np.argsort(cid, kind="stable")
    v, o1, o2, cid = v[order], o1[order], o2[order], cid[order]
    bnd = np.flatnonzero(np.diff(cid)) + 1
    out = []
    for a, b in zip(np.r_[0, bnd], np.r_[bnd, len(cid)]):
        if b - a < minpts:
            continue
        vv = v[a:b]
        if vv.min() > centre - reach or vv.max() < centre + reach:
            continue
        # direction: PCA of the 3 coordinates; require a real crossing angle
        P = np.column_stack([vv, o1[a:b], o2[a:b]])
        P = P - P.mean(0)
        w, V = np.linalg.eigh(np.cov(P.T))
        axis = V[:, np.argmax(w)]
        if abs(axis[0]) < cosmin:
            continue
        below = vv[vv < centre]
        above = vv[vv > centre]
        if not len(below) or not len(above):
            continue
        out.append(above.min() - below.max())
    return np.asarray(out)


# --------------------------------------------------------------------- run ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--sbnd", nargs="*", default=[])
    ap.add_argument("--tsv", default=None)
    a = ap.parse_args()

    boxes = read_sensvol(a.arms)
    seams = describe_geometry(boxes)
    x, y, z, q, cid = load(a.arms)
    V = {"x": x, "y": y, "z": z}
    OTHER = {"x": (y, z), "y": (x, z), "z": (x, y)}

    controls = [dict(name="CTRL_y_+84", axis="y", lo=84.0, hi=84.0, sel=None),
                dict(name="CTRL_y_-84", axis="y", lo=-84.0, hi=-84.0, sel=None),
                dict(name="CTRL_z_75",  axis="z", lo=75.0, hi=75.0, sel=None),
                dict(name="CTRL_z_225", axis="z", lo=225.0, hi=225.0, sel=None),
                dict(name="CTRL_x_bot", axis="x", lo=-170.0, hi=-170.0, sel=-1),
                dict(name="CTRL_x_top", axis="x", lo=170.0, hi=170.0, sel=+1)]

    def measure(s):
        c = 0.5 * (s["lo"] + s["hi"])
        geom = s["hi"] - s["lo"]
        half = 15.0 if s["axis"] == "x" and geom > 1.0 else 6.0
        dm = drift_mask(x, s["sel"])
        v, qq = V[s["axis"]][dm], q[dm]
        R, nfar = ratio(v, c, half)
        o1, o2 = (o[dm] for o in OTHER[s["axis"]])
        g = crossing_gaps(v, o1, o2, cid[dm], c)
        qc, nq = qratio(v, qq, c, max(geom / 2.0, 0.5))
        return dict(name=s["name"], axis=s["axis"], centre=c, geom=geom, R=R,
                    gapmed=float(np.median(g)) if len(g) else float("nan"),
                    gapq3=float(np.percentile(g, 75)) if len(g) else float("nan"),
                    n=len(g), qcore=qc, nfar=nfar)

    ctrl = [measure(s) for s in controls]
    null = {ax: float(np.median([c["gapmed"] for c in ctrl if c["axis"] == ax]))
            for ax in "xyz"}
    rows = [measure(s) for s in seams]

    def show(rs, title):
        print(f"\n== {title} ==")
        print(f"{'name':<16}{'centre':>9}{'geom':>7}" +
              "".join(f"{'R'+str(w):>7}" for w in WS) +
              f"{'gapmed':>8}{'gapQ3':>7}{'null':>7}{'excess':>8}{'n':>6}{'q@core':>8}")
        for r in rs:
            ex = r["gapmed"] - null[r["axis"]]
            print(f"{r['name']:<16}{r['centre']:>9.3f}{r['geom']:>7.3f}" +
                  "".join(f"{v:>7.2f}" for v in r["R"]) +
                  f"{r['gapmed']:>8.2f}{r['gapq3']:>7.2f}{null[r['axis']]:>7.2f}"
                  f"{ex:>+8.2f}{r['n']:>6d}{r['qcore']:>8.2f}")

    show(rows, "A1 (R) + A2 (gap) + A3 (q) per seam")
    show(ctrl, "the same, at planes with NO structure (the null)")
    print("""
  R(w)   = points with |d| < w over the far-field expectation (1.0 = no deficit).
  gapmed = median empty interval bracketing the plane, over tracks that cross it
           by >= 5 cm both sides at > 0.3 cos to the normal (A2).
  null   = gapmed at the same-axis control planes: the point-pitch floor of the
           A2 instrument.  excess = gapmed - null is the width a crossing track
           actually loses, and is what an exclusion box should be built at.
  q@core = median point charge inside the seam over the far field (A3): tells a
           gap that loses POINTS from one that only loses CHARGE.""")

    print("\n== A1 density, 0.5 cm bins (coarser than the ctpc pitch) ==")
    for s in seams:
        c = 0.5 * (s["lo"] + s["hi"])
        dm = drift_mask(x, s["sel"])
        ctr, dens = coarse_profile(V[s["axis"]][dm], c)
        print(f"  {s['name']:<16} " + " ".join(f"{d:5.2f}" for d in dens))
    print(f"  {'d(cm)':<16} " + " ".join(f"{t:+5.2f}" for t in ctr))

    print("\n== A5 density inward from the outer walls (2 mm bins, -1 .. +4 cm) ==")
    lim = dict(x_anode_bot=(-339.91, +1), x_anode_top=(339.91, -1),
               y_lo=(-336.39, +1), y_hi=(336.39, -1),
               z_lo=(0.813, +1), z_hi=(298.435, -1))
    for nm, (wall, sgn) in lim.items():
        d = sgn * (V[nm[0]] - wall)
        h, e = np.histogram(d[(d > -1) & (d < 12)], bins=np.arange(-1, 12.2, 0.2))
        norm = h[(e[:-1] > 6)].mean() or 1.0
        print(f"  {nm:<12} " + " ".join(f"{c:5.2f}" for c in (h / norm)[:25]))
    print(f"  {'d(cm)':<12} " + " ".join(f"{t:+5.1f}" for t in np.arange(-1, 4.0, 0.2)))

    if a.sbnd:
        print("\n== SBND, same instrument ==")
        sb = read_sensvol(a.sbnd)
        sx, sy, sz, sq, scid = load(a.sbnd)
        sV, sO = {"x": sx, "y": sy, "z": sz}, {"x": (sy, sz), "y": (sx, sz)}
        srows = []
        for lo, hi in axis_gaps(sb, 0):
            srows.append(("CPA hole", "x", 0.5 * (lo + hi), hi - lo))
        srows.append(("CTRL mid-y", "y", 0.0, 0.0))
        snull = None
        for nm, ax, c, geom in srows:
            R, _ = ratio(sV[ax], c, 6.0)
            g = crossing_gaps(sV[ax], sO[ax][0], sO[ax][1], scid, c)
            gm = float(np.median(g)) if len(g) else float("nan")
            if nm.startswith("CTRL"):
                snull = gm
            qc, _ = qratio(sV[ax], sq, c, max(geom / 2.0, 0.5))
            print(f"  {nm:<12} centre {c:+.3f} geom {geom:.3f}  " +
                  " ".join(f"R{w}={v:.2f}" for w, v in zip(WS, R)) +
                  f"  gapmed {gm:.2f}  n {len(g)}  q@core {qc:.2f}")
        print(f"  (SBND null gapmed = {snull:.2f}; PDVD's x null = {null['x']:.2f})")

    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("name\taxis\tcentre\tgeom\t" + "\t".join("R%g" % w for w in WS) +
                     "\tgapmed\tgapQ3\tnull\texcess\tn\tqcore\n")
            for r in rows + ctrl:
                fh.write(f"{r['name']}\t{r['axis']}\t{r['centre']}\t{r['geom']}\t" +
                         "\t".join("%.4f" % v for v in r["R"]) +
                         f"\t{r['gapmed']:.4f}\t{r['gapq3']:.4f}\t{null[r['axis']]:.4f}"
                         f"\t{r['gapmed']-null[r['axis']]:.4f}\t{r['n']}\t{r['qcore']:.4f}\n")
        print(f"\nwrote {a.tsv}")


if __name__ == "__main__":
    main()
