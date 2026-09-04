#!/usr/bin/env python3
"""Fiducial-volume census over PR arm dirs already on disk (doc pdvd/33).

Reads only products that every PR arm already writes -- the wct_pr_*.log and the
Bee mabc-pr.zip -- so it needs no reco run and touches nothing.

  usage: fv_stage_census.py <arm-dir> [<arm-dir> ...]
         fv_stage_census.py work/*_d28dlfp

It answers four questions:

  (1) why TaggerCheckSTM declined to fit, by DISTINCT cluster.  The log emits a
      specific reason line AND a generic "evaluated but no pass recorded" line
      for the same cluster, so counting lines double-counts.
  (2) how often the holed DetectorVolumes union (what FiducialUtils gets today)
      truncates an outward march that the pdvd_pr_fv box would let run to the
      wall.  check_dead_volume / check_signal_processing loop on
      `while (inside_fiducial_volume(temp_p))` and early-return once cut_value
      (4 / 5) live steps accumulate, so only a truncation inside that window can
      change a verdict -- both counts are reported.
  (3) how many cluster ends have their containment verdict decided by
      tgm_fv_x_margin = 30 cm rather than SBND's 2.5.
  (4) how many fitted and imaged points actually land in the union's holes.

Model, and its limits: each cluster is reduced to its two PCA-extreme points and
the PCA axis as the outward direction.  cluster_fc_check works on steiner extreme
GROUPS (several per cluster with component_extremes on) and takes the outward
direction from a Hough transform, so these are estimates of the population, not a
replay of the tagger.  The primary exit test it models exactly is the direct one,
`!inside_fv(p1)` (Clustering_Util.cxx:146).
"""
import collections, glob, json, re, sys, zipfile
import numpy as np

# pdvd_pr_fv (cfg/pgrapher/experiment/protodunevd/pr.jsonnet), cm
XW, YW, ZLO, ZHI = 339.91, 336.4, 0.05, 299.25
# DetectorVolumes union holes, from the AnodePlane 'sensvol' log lines
CATH, YSLAB = 3.0, 0.61
SENT = 1e4          # clusters with no t0 are written at x = +-1.48e8 cm
XMARGIN, XMARGIN_SBND = 30.0, 2.5
CUT_VALUE = 5       # the larger of the two walks' early-return thresholds


def in_box(P):
    return ((np.abs(P[:, 0]) <= XW) & (np.abs(P[:, 1]) <= YW)
            & (P[:, 2] >= ZLO) & (P[:, 2] <= ZHI))


def in_union(P):
    return in_box(P) & (np.abs(P[:, 0]) >= CATH) & (np.abs(P[:, 1]) >= YSLAB)


def in_fv(P, xm, ym=3.0, zmax=5.0, zmin=3.0):
    """pdvd_pr_fv + pdvd_pr_fv_margins; index 4 insets the downstream z face."""
    return ((np.abs(P[:, 0]) <= XW - xm) & (np.abs(P[:, 1]) <= YW - ym)
            & (P[:, 2] >= ZLO + zmin) & (P[:, 2] <= ZHI - zmax))


def stm_outcomes(logfile):
    cat = collections.defaultdict(set)
    for line in open(logfile):
        m = re.search(r"check_stm_conditions: cluster (\d+) no STM fit: (.*?)(?: \.\./|$)", line)
        if m:
            cat[m.group(2).strip()].add(int(m.group(1)))
        m = re.search(r"TaggerCheckSTM: cluster (\d+) already TGM", line)
        if m:
            cat['already TGM'].add(int(m.group(1)))
        m = re.search(r"TaggerCheckSTM: cluster (\d+) . STM=1", line)
        if m:
            cat['STM=1'].add(int(m.group(1)))
        m = re.search(r"TaggerCheckSTM: cluster (\d+) . STM=", line)
        if m:
            cat['STM evaluated'].add(int(m.group(1)))
    return cat


def main(arms):
    tot = collections.Counter()
    trunc, fcext = [], []
    cats = collections.Counter()
    for arm in sorted(arms):
        logs = glob.glob(arm + '/wct_pr_*.log')
        if not logs:
            continue
        cat = stm_outcomes(logs[0])
        for k, v in cat.items():
            cats[re.sub(r'\d+', 'N', k)[:60]] += len(v)
        try:
            z = zipfile.ZipFile(arm + '/mabc-pr.zip')
            d = json.loads(z.read('data/0/0-clustering-global.json'))
        except Exception:
            tot['no_zip'] += 1
            continue
        tot['events'] += 1
        for pfx, key in (('stm_fit', 'stm'), ('track_fit', 'trk'), ('clustering', 'img')):
            try:
                f = json.loads(z.read('data/0/0-%s-global.json' % pfx))
            except Exception:
                continue
            fx, fy = np.array(f['x']), np.array(f['y'])
            keep = np.abs(fx) < SENT
            fx, fy = fx[keep], fy[keep]
            tot[key] += len(fx)
            tot[key + '_yhole'] += int((np.abs(fy) < YSLAB).sum())
            tot[key + '_xhole'] += int((np.abs(fx) < CATH).sum())

        X, Y, Z = np.array(d['x']), np.array(d['y']), np.array(d['z'])
        ID = np.array(d['real_cluster_id'])
        ok = (np.abs(X) < SENT) & (np.abs(Y) < SENT) & (np.abs(Z) < SENT)
        tot['sentinel_pts'] += int((~ok).sum())
        X, Y, Z, ID = X[ok], Y[ok], Z[ok], ID[ok]
        FC = cat['fully contained (Mid Point A)']

        for cid in np.unique(ID):
            m = ID == cid
            if m.sum() < 10:
                continue
            P = np.stack([X[m], Y[m], Z[m]], 1)
            c = P - P.mean(0)
            axis = np.linalg.svd(c, full_matrices=False)[2][0]
            t = c @ axis
            extent = float(t.max() - t.min())
            if extent < 10:
                continue
            isfc = int(cid) in FC
            tot['clusters'] += 1
            if isfc:
                fcext.append(extent)
            E = np.stack([P[int(np.argmax(t))], P[int(np.argmin(t))]])
            o30 = int((~in_fv(E, XMARGIN)).sum())
            o25 = int((~in_fv(E, XMARGIN_SBND)).sum())
            if o30 == 2:
                tot['two_exit30'] += 1
                if o25 == 1:
                    tot['becomes_single_exit'] += 1
                elif o25 == 2:
                    tot['still_two_exit'] += 1
            for sgn, idx in ((+1, int(np.argmax(t))), (-1, int(np.argmin(t)))):
                p0, dirv = P[idx], sgn * axis
                tot['ends'] += 1
                if in_fv(p0[None, :], XMARGIN_SBND)[0] and not in_fv(p0[None, :], XMARGIN)[0]:
                    tot['margin_decided'] += 1
                steps = np.arange(1, 501)            # 1 cm steps, the walks' step
                R = p0 + steps[:, None] * dirv[None, :]
                bx, un = in_box(R), in_union(R)
                nb = int(np.argmin(bx)) if (~bx).any() else len(steps)
                nu = int(np.argmin(un)) if (~un).any() else len(steps)
                if nu < nb:
                    tot['trunc'] += 1
                    trunc.append(nb - nu)
                    q = R[nu]
                    tot['trunc_cathode' if abs(q[0]) < CATH else 'trunc_yslab'] += 1
                    if nu < CUT_VALUE <= nb:
                        tot['trunc_early'] += 1
                        if isfc:
                            tot['trunc_early_fc'] += 1

    ends = max(tot['ends'], 1)
    print(f"events {tot['events']}  clusters >10 cm {tot['clusters']}  ends {tot['ends']}"
          f"  sentinel-x points dropped {tot['sentinel_pts']}")
    print("\n(1) TaggerCheckSTM outcome, DISTINCT clusters summed over events:")
    for k, n in cats.most_common():
        print(f"    {n:7d}  {k}")
    print(f"\n(2) outward marches truncated by a union hole: {tot['trunc']} of {tot['ends']}"
          f" ({100.0 * tot['trunc'] / ends:.1f} %)"
          f"  [cathode {tot['trunc_cathode']}, y seam {tot['trunc_yslab']}]")
    if trunc:
        a = np.array(trunc)
        print(f"    stopped early by cm: p50={np.median(a):.0f} p90={np.percentile(a, 90):.0f} max={a.max()}")
    print(f"    inside the {CUT_VALUE}-step early-return window (can change a verdict):"
          f" {tot['trunc_early']} ({100.0 * tot['trunc_early'] / ends:.2f} %),"
          f" of them on fully-contained clusters: {tot['trunc_early_fc']}")
    print(f"\n(3) ends inside the FV at a {XMARGIN_SBND} cm x margin, outside at {XMARGIN}:"
          f" {tot['margin_decided']} of {tot['ends']} ({100.0 * tot['margin_decided'] / ends:.1f} %)")
    print(f"    clusters with BOTH ends outside at {XMARGIN} cm: {tot['two_exit30']}"
          f"  -> at {XMARGIN_SBND} cm: {tot['still_two_exit']} still two-ended,"
          f" {tot['becomes_single_exit']} single-exit (stopping-muon candidates)")
    print("\n(4) points inside the union's holes:")
    for key, label in (('stm', 'stm_fit'), ('trk', 'track_fit'), ('img', 'imaged 3-D')):
        n = max(tot[key], 1)
        print(f"    {label:11s} {tot[key]:9d} pts | |y|<{YSLAB}: {tot[key + '_yhole']:6d}"
              f" ({100.0 * tot[key + '_yhole'] / n:.4f} %) | |x|<{CATH}: {tot[key + '_xhole']:6d}"
              f" ({100.0 * tot[key + '_xhole'] / n:.4f} %)")
    if fcext:
        f = np.array(fcext)
        print(f"\n    fully-contained clusters >10 cm: {len(f)}  PCA extent cm p50={np.median(f):.0f}"
              f" p90={np.percentile(f, 90):.0f} max={f.max():.0f}  (>100 cm: {(f > 100).sum()},"
              f" >300 cm: {(f > 300).sum()} -- check components before reading these as tracks)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
