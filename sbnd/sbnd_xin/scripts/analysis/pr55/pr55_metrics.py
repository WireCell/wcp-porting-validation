#!/usr/bin/env python3
"""doc pr/55: measure and case-catalog the three symptoms the owner flagged
against the doc pr/53 round-7 Bee after-set (SBND production, relaxed_strict_img)
on events 142421, 269774, 71372 -- diagnosis only, no fix this session.

The owner's own comparison is "track_fit vs shower_track" (their words), NOT
fitted-vs-raw-image, so this reads all three Bee point layers straight from
each event's own pr_evt<N>/mabc-pr.zip (the PR-stage output, unremapped
cluster_id -- NOT the bee/pr53r7/*.zip upload copy, which has cluster_id
rewritten into img-global's id space, see make_pr_bee.py's docstring):

  IMG = 0-clustering-global.json   the 3D image points, this cluster's own
  ASC = 0-shower_track-global.json the points ASSOCIATED to each PR segment;
        q=0 marks "painted track", q=15000 marks "painted shower"
  FIT = 0-track_fit-global.json    the FITTED trajectory; q = dQ*0.1 - 1000,
        i.e. a proxy for dQ/dx

Both ASC and FIT carry real_cluster_id = cluster*1000 + segment (WCT calls
this field "real_cluster_id" but it is the PR SEGMENT id, not the image
cluster id from IMG -- kept as "seg" throughout, per fitgap_exam.py's
precedent, to avoid confusion).

Four case families, one row per case in the output TSV:

  Family A -- ghost run: a maximal contiguous stretch of one segment's FITTED
    polyline (storage order = trajectory order) farther than d0=1.0cm from the
    nearest IMG point of that fit point's OWN final cluster. Same radius as
    the shipped relaxed_strict_img::m_img_radius, so numbers are comparable to
    round 7. Case ID <evt>-G<k>.

  Family B -- uncovered shower: a segment's ASSOCIATED (ASC) points sit far
    from that segment's own FITTED polyline -- "image with no matching fitted
    trajectory". Case ID <evt>-U<k>, flagged when uncov_frac3 > 0.3 and
    assoc_pts >= 100.

  Family C -- phantom segment: a segment has fit_pts > 0 but assoc_pts == 0 --
    a trajectory drawn with nothing associated under it. Case ID <evt>-P<k>.

  Family D -- honesty label attached to every A/C case: REAL-VOID (no image
    nearby, not dead-excused) / DEAD-BRIDGE (dead-channel excused) /
    IMG-ELSEWHERE (image exists nearby but belongs to a DIFFERENT final
    cluster -- an extension beyond the three labels in the pr/55 plan, added
    because it is a materially different situation from a true void; flagged
    explicitly as an addition in the doc) / PAINT-ONLY (image is there within
    ~1.5cm of the segment's OWN cluster but was not associated into that
    segment's ASC points -- overstates the defect in a raw Bee view).

Also computed per Family-A/C case: strict_verdict -- replay of the SHIPPED
relaxed_strict_img predicate (Graphs::relaxed_img_bad, run_floor=4,
dis_cap_cm=15.0, m_img_radius=1.0cm, clus/src/connect_graph_relaxed_strict.cxx)
plus round-6's S1-S3 base test (oc53_probe.walk_and_score) along the straight
line between the case's two endpoints, using the event's OWN ctpc/dead arrays
(oc53_probe.Loader) -- i.e. "would today's production membership rule have
killed a direct connection here?". This is the number that turns "the fitter
runs on a looser graph than protect_bundle" from a code-reading claim (see
NeutrinoPatternBase.cxx:111, do_rough_path on steiner_graph, a Steiner
reduction of ctpc_ref_pid = closely_pid + the uncapped MST connect_graph_*)
into a measurement on the owner's own cases.

Usage:
  pr55_metrics.py <outdir> [--evt EVT ...] [--arm ARM] [--label LABEL]

Repro (doc pr/55):
  cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  python3 scripts/analysis/pr55/pr55_metrics.py docs/pr --label production
"""
import sys, os, json, zipfile, argparse
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pr53'))
from oc53_probe import Loader, walk_and_score  # noqa: E402

SB = '/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin'

# event -> (PR-stage arm holding pr_evt<N>/mabc-pr.zip, manifest)
EVENT_ARMS = {
    '142421': ('work-pr53r7-on19', 'ncpi0-19'),
    '71372':  ('work-pr53r7-on19', 'ncpi0-19'),
    '269774': ('work-pr53r7-on48', 'nueCC-48'),
}

# owner's exact reported coordinates, for case attribution in the doc
OWNER_POINTS = {
    '142421': [(89.5, -70.0, 243.7, 'shower-no-track+high-dQdx-connector')],
    '269774': [(95.1, 16.6, 137.0, 'pairA-p1'), (84.8, 10.3, 136.4, 'pairA-p2'),
               (45.7, -43.8, 154.0, 'pairB-p1'), (39.5, -59.1, 163.7, 'pairB-p2')],
    '71372':  [(-118.1, -124.0, 271.4, 'p1'), (-175.0, -164.7, 228.1, 'p2'),
               (-164.7, -152.4, 226.0, 'p3')],
}

D0_GHOST = 1.0          # cm, Family A threshold, matches m_img_radius
UNCOV_D = 3.0            # cm, Family B per-point threshold
UNCOV_FRAC = 0.3         # Family B flag threshold
UNCOV_MIN_ASSOC = 100    # Family B flag threshold
PAINT_ONLY_D = 1.5       # cm, Family D discriminator
RUN_FLOOR = 4            # relaxed_img_bad constants (Graphs.h)
DIS_CAP_CM = 15.0
IMG_RADIUS = 1.0


def _z(arm, evt, name):
    with zipfile.ZipFile(os.path.join(SB, arm, 'pr_evt%s' % evt, 'mabc-pr.zip')) as f:
        return json.loads(f.read('data/0/0-%s.json' % name))


def load_event(arm, evt):
    def arr(d):
        return dict(P=np.c_[d['x'], d['y'], d['z']].astype(float),
                    cid=np.array(d['cluster_id']),
                    seg=np.array(d['real_cluster_id']),
                    q=np.array(d['q'], dtype=float))
    return dict(img=arr(_z(arm, evt, 'clustering-global')),
                asc=arr(_z(arm, evt, 'shower_track-global')),
                fit=arr(_z(arm, evt, 'track_fit-global')))


def arclength(P):
    if len(P) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def port_max_ghost_run(ld, p1, p2, img_radius=IMG_RADIUS):
    """Exact port of connect_graph_relaxed_strict.cxx's max_ghost_run lambda:
    1cm interior samples (excludes the final step, which is always p2 itself
    -- S1's endpoint-honest convention), longest contiguous run of samples
    with no 3D image within img_radius AND not dead-channel-excused on any
    plane. Returns (max_run, dead_frac_over_ghosts, nst)."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    dis = np.linalg.norm(p2 - p1)
    nst = int(dis / 1.0) + 1
    run = best = 0
    n_dead = n_ghost = 0
    for ii in range(nst - 1):
        tp = p1 + (p2 - p1) / nst * (ii + 1)
        d = ld.img_closest_dis(tp)
        s = ld.scores(tp)
        dead_excused = (s[3] + s[4] + s[5]) > 0
        ghost = (d > img_radius) and not dead_excused
        if d > img_radius:
            n_ghost += 1
            if dead_excused:
                n_dead += 1
        run = run + 1 if ghost else 0
        best = max(best, run)
    dead_frac = (n_dead / n_ghost) if n_ghost else 0.0
    return best, dead_frac, nst


def strict_verdict(ld, p1, p2):
    """Would today's production relaxed_strict_img predicate kill a direct
    p1-p2 connection? OR of round-6's S1-S3 base test (walk_and_score's own
    'kill') and round-7's S5 image test (relaxed_img_bad)."""
    r = walk_and_score(ld, p1, p2)
    max_run, dead_frac, nst = port_max_ghost_run(ld, p1, p2)
    dis_cm = r['dis']
    s5 = (max_run >= RUN_FLOOR) and (dis_cm < DIS_CAP_CM)
    return dict(kill_base=r['kill'], branch=r['branch'], max_ghost_run=max_run,
                dead_frac=dead_frac, dis_cm=dis_cm, nst=nst, kill_s5=s5,
                kill_any=bool(r['kill'] or s5))


def nearest_cid(tree, cid_arr, p):
    d, i = tree.query(p)
    return int(cid_arr[i]), float(d)


def family_a(ev, ld, d0=D0_GHOST):
    """Ghost runs: per segment, contiguous FIT-order stretches farther than
    d0 from the nearest IMG point of that segment's OWN final cluster."""
    fit = ev['fit']; img = ev['img']
    img_tree_all = cKDTree(img['P'])
    cases = []
    for seg in sorted(set(fit['seg'].tolist())):
        if seg < 0:
            continue
        m = fit['seg'] == seg
        idx = np.where(m)[0]  # storage order == trajectory order (fitgap_exam precedent)
        if len(idx) < 2:
            continue
        P = fit['P'][idx]; q = fit['q'][idx]; cid = int(fit['cid'][idx][0])
        own_img = img['P'][img['cid'] == cid]
        d_own = (cKDTree(own_img).query(P)[0] if len(own_img) else np.full(len(P), 1e9))
        d_glob, i_glob = img_tree_all.query(P)
        cid_glob = img['cid'][i_glob]  # nearest ALL-image cluster id at each fit point
        bad = d_own > d0
        k = 0
        run_no = 0
        while k < len(bad):
            if not bad[k]:
                k += 1; continue
            k2 = k
            while k2 < len(bad) and bad[k2]:
                k2 += 1
            if k2 - k >= 2:
                run_no += 1
                ridx = np.arange(k, k2)
                Pr = P[ridx]; qr = q[ridx]
                dead_frac_scores = []
                for pt in Pr:
                    s = ld.scores(pt)
                    dead_frac_scores.append(1.0 if (s[3] + s[4] + s[5]) > 0 else 0.0)
                dead_frac = float(np.mean(dead_frac_scores))
                q_rest = q[~np.isin(np.arange(len(P)), ridx)]
                q_ratio = (float(np.median(qr)) / float(np.median(q_rest))
                           if len(q_rest) and np.median(q_rest) else float('nan'))
                # cid_a/cid_b: nearest-ANY-cluster id right before/after the run --
                # the blob the trajectory is actually leaving/entering.
                cid_a, _ = nearest_cid(img_tree_all, img['cid'], P[max(k - 1, 0)])
                cid_b, _ = nearest_cid(img_tree_all, img['cid'], P[min(k2, len(P) - 1)])
                # foreign_frac: within the run itself, what fraction of samples' nearest
                # ANY-cluster image point belongs to a DIFFERENT final cluster than this
                # segment's own -- the direct "is the trajectory secretly tracking a
                # foreign blob" signal (bridges_clusters was the endpoint-only, weaker
                # version and stayed False everywhere it was tried).
                foreign = cid_glob[ridx] != cid
                foreign_frac = float(foreign.mean())
                foreign_cids = sorted(set(cid_glob[ridx][foreign].tolist()))
                sv = strict_verdict(ld, P[k], P[min(k2, len(P) - 1)])
                d_glob_run_min = float(d_glob[ridx].min())
                if dead_frac >= 0.5:
                    label = 'DEAD-BRIDGE'
                elif d_glob_run_min <= PAINT_ONLY_D:
                    label = 'IMG-ELSEWHERE'
                else:
                    label = 'REAL-VOID'
                cases.append(dict(
                    family='A', kind='ghost_run', evt=None, cid=cid, seg=seg, run=run_no,
                    n_pts=int(k2 - k), L_cm=round(arclength(Pr), 2), d_max_cm=round(float(d_own[ridx].max()), 2),
                    d_glob_min_cm=round(d_glob_run_min, 2), dead_frac=round(dead_frac, 2),
                    q_ratio=round(q_ratio, 2) if q_ratio == q_ratio else float('nan'),
                    cid_a=cid_a, cid_b=cid_b, bridges_clusters=(cid_a != cid_b),
                    foreign_frac=round(foreign_frac, 2), foreign_cids=str(foreign_cids),
                    label=label, kill_base=sv['kill_base'], kill_s5=sv['kill_s5'],
                    kill_any=sv['kill_any'], max_ghost_run=sv['max_ghost_run'],
                    strict_dis_cm=round(sv['dis_cm'], 2), branch=sv['branch'],
                    p_first=tuple(round(v, 1) for v in P[k]), p_last=tuple(round(v, 1) for v in P[k2 - 1]),
                ))
            k = k2
    return cases


def family_b(ev):
    """Uncovered showers: per segment, ASC points far from the FITTED polyline of
    the segment's OWN FINAL CLUSTER (all segments in that cluster, unioned).

    Correction (same mechanism as family_c's, doc pr/55): comparing ASC(seg) to
    FIT(same seg) alone is misleading for shower members -- MultiAlgBlobClustering
    files a whole shower's associate_points under its START SEGMENT's id, so a
    single-segment fit is compared against points pooled from every member segment,
    which trivially looks "uncovered" (measured on 71372 seg 19065: median coverage
    distance 49.6cm / 99% >3cm against its own 6-point fit alone, vs 3.4cm median /
    52% >3cm against the whole cluster's fit -- still a real, substantial gap, but a
    fair one). Using the cluster union is the correct apples-to-apples comparison a
    Bee viewer effectively makes when scanning a whole shower against its trajectory."""
    asc = ev['asc']; fit = ev['fit']
    cases = []
    for seg in sorted(set(asc['seg'].tolist())):
        if seg < 0:
            continue
        ma = asc['seg'] == seg
        n_assoc = int(ma.sum())
        if n_assoc == 0:
            continue
        A = asc['P'][ma]; qa = asc['q'][ma]
        cid = int(asc['cid'][ma][0])
        mf_own = fit['seg'] == seg
        mf_cid = fit['cid'] == cid
        n_fit_own = int(mf_own.sum())
        n_fit_cid = int(mf_cid.sum())
        if n_fit_cid >= 1:
            F = fit['P'][mf_cid]
            d = cKDTree(F).query(A)[0]
            fit_len = arclength(fit['P'][mf_own]) if n_fit_own >= 2 else 0.0
        else:
            d = np.full(n_assoc, 999.0)
            fit_len = 0.0
        cov_med, cov_p90, cov_max = (float(np.median(d)), float(np.percentile(d, 90)), float(d.max()))
        uncov_frac3 = float((d > UNCOV_D).mean())
        bbox_diag = float(np.linalg.norm(A.max(axis=0) - A.min(axis=0))) if n_assoc >= 2 else 0.0
        spread = (bbox_diag / fit_len) if fit_len > 0 else float('inf')
        n_shower = int((qa > 0).sum()); n_track = int((qa == 0).sum())
        if uncov_frac3 > UNCOV_FRAC and n_assoc >= UNCOV_MIN_ASSOC:
            cases.append(dict(
                family='B', kind='uncovered_shower', evt=None, cid=cid, seg=seg, run=0,
                n_pts=n_assoc, assoc_shower=n_shower, assoc_track=n_track,
                fit_pts=n_fit_cid, fit_pts_own_seg=n_fit_own,
                fit_len_cm=round(fit_len, 2), cov_med_cm=round(cov_med, 2), cov_p90_cm=round(cov_p90, 2),
                cov_max_cm=round(cov_max, 2), uncov_frac3=round(uncov_frac3, 2),
                bbox_diag_cm=round(bbox_diag, 2), spread=round(spread, 2) if spread != float('inf') else float('inf'),
                centroid=tuple(round(v, 1) for v in A.mean(axis=0)),
            ))
    return cases


def family_c(ev, ld):
    """Phantom segments: fit_pts > 0 but assoc_pts == 0 UNDER THIS SEGMENT'S OWN id.

    Correction (confirmed against a C++ sentinel, doc pr/55): MultiAlgBlobClustering's
    shower_track writer files a shower MEMBER segment's associate_points under its
    SHOWER's START SEGMENT id (`seg_to_shower[...]->start_segment()->id()`), not the
    member's own id -- so most "assoc_pts==0 for this seg" cases are a display-layer
    id-reassignment, not a genuine absence of associated points. Runtime-verified: of
    142421's four id-match phantom candidates in cluster 7 (segs 7011/7018/7110/7020),
    three are 100% covered by shower_track points filed under seg 7109 (the shower's
    start segment); only 7020 has a null associate_points dpcloud in production (the
    "pr55 shower_track layer: ... has no associate_points dpcloud" sentinel fires
    exactly once for this event, on seg 7020) and is genuinely uncovered by ANY
    shower_track point. So this function checks coverage by ANY shower_track point
    (regardless of its filed seg id) in addition to the exact-id match, and reports
    which mechanism applies via `attribution`."""
    asc = ev['asc']; fit = ev['fit']; img = ev['img']
    assoc_segs = set(asc['seg'][asc['seg'] >= 0].tolist())
    img_tree_all = cKDTree(img['P'])
    asc_tree_all = cKDTree(asc['P']) if len(asc['P']) else None
    cases = []
    for seg in sorted(set(fit['seg'].tolist())):
        if seg < 0 or seg in assoc_segs:
            continue
        mf = fit['seg'] == seg
        n_fit = int(mf.sum())
        if n_fit == 0:
            continue
        F = fit['P'][mf]; q = fit['q'][mf]; cid = int(fit['cid'][mf][0])
        fit_len = arclength(F)
        own_img = img['P'][img['cid'] == cid]
        d_own = cKDTree(own_img).query(F)[0] if len(own_img) else np.full(len(F), 1e9)
        d_glob, _ = img_tree_all.query(F)
        # coverage by ANY shower_track point (any seg id) -- distinguishes a genuine
        # display gap from the shower-start-segment id reassignment above.
        if asc_tree_all is not None:
            d_any_asc, i_any_asc = asc_tree_all.query(F)
            covered = d_any_asc <= PAINT_ONLY_D
            frac_covered_any = float(covered.mean())
            if covered.any():
                attrib_segs, attrib_n = np.unique(asc['seg'][i_any_asc[covered]], return_counts=True)
                attributed_to = int(attrib_segs[np.argmax(attrib_n)])
            else:
                attributed_to = None
        else:
            frac_covered_any, attributed_to = 0.0, None
        if frac_covered_any >= 0.8:
            attribution = 'MISATTRIBUTED-SHOWER-ID(seg=%s)' % attributed_to
        elif frac_covered_any > 0.0:
            attribution = 'PARTIALLY-COVERED(seg=%s)' % attributed_to
        else:
            attribution = 'NO-ASC-COVERAGE'
        dead_frac_scores = [1.0 if (ld.scores(pt)[3] + ld.scores(pt)[4] + ld.scores(pt)[5]) > 0 else 0.0
                             for pt in F]
        dead_frac = float(np.mean(dead_frac_scores))
        # nearest other segment's fit polyline, to detect duplicates (e.g. 7020 vs 7010)
        best_seg, best_d = None, 1e9
        other_segs = set(fit['seg'][(fit['seg'] >= 0) & (fit['seg'] != seg)].tolist())
        for oseg in other_segs:
            mo = fit['seg'] == oseg
            if mo.sum() < 2:
                continue
            O = fit['P'][mo]
            d_fo = cKDTree(O).query(F)[0]
            d_of = cKDTree(F).query(O)[0]
            dd = max(float(np.median(d_fo)), float(np.median(d_of)))
            if dd < best_d:
                best_d, best_seg = dd, int(oseg)
        d_own_med = float(np.median(d_own))
        # Family D label: MISATTRIBUTED-SHOWER-ID cases are display-layer only (the
        # points exist, just filed under the shower's start segment) and are labeled
        # as such regardless of image proximity -- they are not a reconstruction
        # defect at all. Only NO/PARTIAL-coverage cases get the image-based label.
        if attribution.startswith('MISATTRIBUTED'):
            label = 'MISATTRIBUTED-SHOWER-ID'
        elif d_own_med <= PAINT_ONLY_D:
            label = 'PAINT-ONLY'
        elif dead_frac >= 0.5:
            label = 'DEAD-BRIDGE'
        else:
            label = 'REAL-VOID'
        # "material" = big enough to matter for a Bee hand-scan / a later fix session;
        # the vast majority of phantom segments are short leftover stubs (median
        # fit_len ~1-4cm across all three events) -- flag, don't drop, so the TSV
        # stays a complete census but the doc's headline table isn't 260 rows long.
        material = (fit_len >= 10.0) or (n_fit >= 20)
        cases.append(dict(
            family='C', kind='phantom_segment', evt=None, cid=cid, seg=seg, run=0,
            n_pts=n_fit, fit_len_cm=round(fit_len, 2), d_own_med_cm=round(d_own_med, 2),
            d_own_max_cm=round(float(d_own.max()), 2), d_glob_med_cm=round(float(np.median(d_glob)), 2),
            dead_frac=round(dead_frac, 2), q_med=round(float(np.median(q)), 0),
            nearest_dup_seg=best_seg, nearest_dup_d_cm=round(best_d, 2) if best_seg is not None else float('nan'),
            frac_covered_any_asc=round(frac_covered_any, 2), attribution=attribution,
            label=label, material=material,
            centroid=tuple(round(v, 1) for v in F.mean(axis=0)),
        ))
    return cases


# plan's case-ID convention: Family A -> G(host run), B -> U(ncovered shower),
# C -> P(hantom segment). 'family' stays A/B/C internally (matches the plan's
# family names); ID_LETTER is only the id prefix.
ID_LETTER = {'A': 'G', 'B': 'U', 'C': 'P'}


def assign_ids(evt, cases):
    fam_k = {'A': 0, 'B': 0, 'C': 0}
    for c in sorted(cases, key=lambda r: -(r.get('L_cm') or r.get('n_pts') or 0)):
        fam_k[c['family']] += 1
        c['case_id'] = '%s-%s%d' % (evt, ID_LETTER[c['family']], fam_k[c['family']])
        c['evt'] = evt
    return cases


def attribute_owner_points(evt, cases):
    """Tag each case with the nearest owner-reported coordinate, if within 15cm."""
    pts = OWNER_POINTS.get(evt, [])
    for c in cases:
        cen = c.get('centroid') or c.get('p_first')
        if cen is None:
            c['owner_pt'] = ''
            continue
        cen = np.array(cen, float)
        best, bd = '', 1e9
        for (x, y, z, name) in pts:
            d = np.linalg.norm(cen - np.array([x, y, z]))
            if d < bd:
                bd, best = d, name
        c['owner_pt'] = best if bd < 15.0 else ''
        c['owner_pt_d_cm'] = round(bd, 1) if bd < 1e8 else float('nan')
    return cases


def run_event(evt, arm=None, cen_arm=None):
    arm = arm or EVENT_ARMS[evt][0]
    ev = load_event(arm, evt)
    ld = Loader(os.path.join(SB, arm, 'pr_evt%s' % evt))
    try:
        cases = family_a(ev, ld) + family_b(ev) + family_c(ev, ld)
    finally:
        ld.cleanup()
    cases = assign_ids(evt, cases)
    cases = attribute_owner_points(evt, cases)
    return cases


TSV_COLS = ['case_id', 'evt', 'family', 'kind', 'label', 'material', 'owner_pt', 'owner_pt_d_cm',
            'cid', 'seg', 'run', 'n_pts', 'L_cm', 'fit_len_cm', 'd_max_cm', 'd_glob_min_cm',
            'd_own_med_cm', 'd_own_max_cm', 'd_glob_med_cm', 'cov_med_cm', 'cov_p90_cm', 'cov_max_cm',
            'uncov_frac3', 'bbox_diag_cm', 'spread', 'dead_frac', 'q_ratio', 'q_med',
            'cid_a', 'cid_b', 'bridges_clusters', 'foreign_frac', 'foreign_cids',
            'kill_base', 'kill_s5', 'kill_any',
            'max_ghost_run', 'strict_dis_cm', 'branch', 'nearest_dup_seg', 'nearest_dup_d_cm',
            'frac_covered_any_asc', 'attribution',
            'assoc_shower', 'assoc_track', 'fit_pts', 'fit_pts_own_seg', 'centroid', 'p_first', 'p_last']


def write_tsv(path, all_cases, arm_label):
    with open(path, 'w') as f:
        f.write('# doc pr/55 case catalogue -- arm=%s\n' % arm_label)
        f.write('\t'.join(TSV_COLS) + '\n')
        for c in all_cases:
            row = [str(c.get(k, '')) for k in TSV_COLS]
            f.write('\t'.join(row) + '\n')
    print('wrote', path, '(%d cases)' % len(all_cases))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--evt', nargs='*', default=list(EVENT_ARMS.keys()))
    ap.add_argument('--label', default='production')
    a = ap.parse_args()
    all_cases = []
    for evt in a.evt:
        print('=== %s ===' % evt)
        cases = run_event(evt)
        for fam, nm in (('A', 'ghost runs'), ('B', 'uncovered showers'), ('C', 'phantom segments')):
            fc = [c for c in cases if c['family'] == fam]
            print('  Family %s (%s): %d case(s)' % (fam, nm, len(fc)))
            for c in fc:
                print('    %-14s label=%-13s owner_pt=%-12s' % (c['case_id'], c.get('label', '-'), c['owner_pt']),
                      {k: v for k, v in c.items() if k not in
                       ('family', 'kind', 'evt', 'case_id', 'label', 'owner_pt', 'owner_pt_d_cm')})
        all_cases += cases
    out = os.path.join(SB, a.outdir, 'pr55-cases.tsv') if not os.path.isabs(a.outdir) else \
        os.path.join(a.outdir, 'pr55-cases.tsv')
    write_tsv(out, all_cases, a.label)


if __name__ == '__main__':
    main()
