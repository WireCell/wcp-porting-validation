#!/usr/bin/env python3
"""doc 99 -- census of the per-cluster matched-flash index against the grouping's
canonical "flash" point cloud.

WHAT THIS MEASURES.  QLMatching writes each matched cluster a scalar "flash"
holding the row id of the flash in the PER-APA flash tensor it matched against
(Opflash::get_flash_id()).  Facade::Grouping::flash_at() resolves that scalar as
a ROW INDEX into the grouping-level canonical "flash" point cloud.  SBND runs one
FlashTensorToOpticalPCs per APA and each ASSIGNS lpcs["flash"], so the archive
keeps only the LAST APA's flash list.  Every cluster matched by an earlier APA
therefore carries a scalar that indexes the wrong list.

Each matched cluster is put in one of three classes, using cluster_t0 -- which
QLMatching sets to the time of the flash the cluster ACTUALLY matched, and which
survives in the archive even when that flash's PC row does not -- as ground truth:

  CORRECT : scalar < nflash and flash.time[scalar] == cluster_t0  (bit-equal)
  WRONG   : scalar < nflash and the times differ  => flash_at() silently returns
            a DIFFERENT, real flash
  OOB     : scalar >= nflash => Facade_Mixins.h get_element() runs off the end of
            the array and returns raw memory (the pre-fix undefined read)

Repro (production Q/L arms, all 3067 events):
  python3 scripts/analysis/d99_flash_index_census.py \
      --arm 'work-{s}-d97fv' --samples nuecc48,ncpi0,mcp1k,mcp2k \
      --out /home/xqian/tmp/d99-flash-census.tsv --jobs 8

Repro (PR stage -- this is the one that predicts T_cluster; see --stage):
  python3 scripts/analysis/d99_flash_index_census.py \
      --arm 'work-{s}-d99fixpr' --stage pr --samples ncpi0,nuecc48,mcp1k \
      --out /home/xqian/tmp/d99-census-pr308.tsv \
      --detail /home/xqian/tmp/d99-flash-detail-pr308.tsv --jobs 8
"""
import argparse, io, json, os, re, sys, tarfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# grouping-level named PC arrays this census needs
WANT = {
    ("flash", "time"),
    ("cluster_scalar", "flash"),
    ("cluster_scalar", "cluster_t0"),
    ("cluster_scalar", "ident"),
    ("cluster_scalar", "matched_flash_gid"),
}
RE_DP = re.compile(r"pointtrees/\d+/live/pointclouds/namedpcs/([^/]+)/arrays/([^/]+)$")


def read_arrays(tgz):
    """One sequential pass over the .tar.gz: metadata_N precedes array_N, so we can
    decide from the metadata whether to keep the array that follows."""
    out, keep = {}, {}
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if m.name.endswith("_metadata.json"):
                idx = m.name[: -len("_metadata.json")]
                try:
                    d = json.loads(tf.extractfile(m).read().decode())
                except Exception:
                    continue
                mm = RE_DP.match(d.get("datapath", "") or "")
                if mm and (mm.group(1), mm.group(2)) in WANT:
                    keep[idx] = (mm.group(1), mm.group(2))
            elif m.name.endswith("_array.npy"):
                idx = m.name[: -len("_array.npy")]
                if idx in keep:
                    out[keep[idx]] = np.load(io.BytesIO(tf.extractfile(m).read()))
    return out


def census_event(job):
    """-> (summary row, per-cluster detail rows for the non-CORRECT clusters).

    The detail rows carry the cluster's IDENT, which is what T_cluster writes as
    cluster_id, so a downstream A/B can join on it per cluster instead of
    comparing counts (the PR-stage grouping need not hold the same clusters)."""
    sample, evt, tgz = job
    try:
        a = read_arrays(tgz)
    except Exception as e:
        return (sample, evt, -1, -1, 0, 0, 0, 0, "READ_FAIL:%s" % e), []
    ftime = a.get(("flash", "time"))
    scal = a.get(("cluster_scalar", "flash"))
    t0 = a.get(("cluster_scalar", "cluster_t0"))
    if scal is None or t0 is None:
        return (sample, evt, -1, -1, 0, 0, 0, 0, "NO_CLUSTER_SCALAR"), []
    ident = a.get(("cluster_scalar", "ident"))
    gid = a.get(("cluster_scalar", "matched_flash_gid"))
    nflash = 0 if ftime is None else int(ftime.size)
    n_none = n_ok = n_wrong = n_oob = 0
    detail = []
    for i, (s, t) in enumerate(zip(scal, t0)):
        s = int(s)
        cid = int(ident[i]) if ident is not None and i < ident.size else -1
        g = int(gid[i]) if gid is not None and i < gid.size else -1
        if s < 0:
            n_none += 1
            continue
        if s >= nflash:
            n_oob += 1
            detail.append((sample, evt, cid, s, nflash, g, "OOB", float(t), ""))
        elif float(ftime[s]) == float(t):
            n_ok += 1
        else:
            n_wrong += 1
            detail.append((sample, evt, cid, s, nflash, g, "WRONG", float(t), repr(float(ftime[s]))))
    return (sample, evt, nflash, int(scal.size), n_none, n_ok, n_wrong, n_oob, ""), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm pattern with {s}, e.g. 'work-{s}-d97fv'")
    ap.add_argument("--stage", default="ql", choices=("ql", "pr"),
                    help="ql: ql_evt<ID>/pctree-evt<ID>.tar.gz (the Q/L arm).  "
                         "pr: pr_evt<ID>/pctree-pr-evt<ID>.tar.gz -- use this to predict "
                         "T_cluster, which is written from the PR-stage grouping.  The two "
                         "differ: the PR chain re-clusters, so cluster ids are renumbered and "
                         "there are more of them (evt 59685: 10 clusters in Q/L, 22 in PR).")
    ap.add_argument("--samples", default="nuecc48,ncpi0,mcp1k,mcp2k")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), help="sbnd_xin root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detail", default=None,
                    help="also write one row per non-CORRECT cluster (joinable to "
                         "T_cluster.cluster_id)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for s in args.samples.split(","):
        adir = os.path.join(args.root, args.arm.format(s=s))
        if not os.path.isdir(adir):
            print("missing arm:", adir, file=sys.stderr)
            continue
        dirre = r"pr_evt(\d+)$" if args.stage == "pr" else r"ql_evt(\d+)$"
        fmt = "pctree-pr-evt%s.tar.gz" if args.stage == "pr" else "pctree-evt%s.tar.gz"
        for d in sorted(os.listdir(adir)):
            m = re.match(dirre, d)
            if not m:
                continue
            tgz = os.path.join(adir, d, fmt % m.group(1))
            if os.path.exists(tgz):
                jobs.append((s, int(m.group(1)), tgz))
    print("events: %d" % len(jobs))

    rows, details = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, (r, d) in enumerate(ex.map(census_event, jobs, chunksize=8)):
            rows.append(r); details.extend(d)
            if (i + 1) % 250 == 0:
                print("  %d/%d" % (i + 1, len(jobs)), flush=True)

    rows.sort(key=lambda r: (r[0], r[1]))
    with open(args.out, "w") as fp:
        fp.write("sample\tevent\tnflash\tnclus\tn_noflash\tn_correct\tn_wrong\tn_oob\tnote\n")
        for r in rows:
            fp.write("\t".join(str(x) for x in r) + "\n")

    if args.detail:
        details.sort(key=lambda r: (r[0], r[1], r[2]))
        with open(args.detail, "w") as fp:
            fp.write("sample\tevent\tcluster_id\tscalar\tnflash\tmatched_flash_gid"
                     "\tclass\tcluster_t0\tresolved_time\n")
            for r in details:
                fp.write("\t".join(str(x) for x in r) + "\n")
        print("wrote %s (%d rows)" % (args.detail, len(details)))

    print("\n%-9s %6s %8s %8s %8s %8s %8s" % ("sample", "evts", "matched", "CORRECT", "WRONG", "OOB", "evt_hit"))
    tot = [0] * 5
    for s in args.samples.split(","):
        sub = [r for r in rows if r[0] == s and r[2] >= 0]
        if not sub:
            continue
        ok = sum(r[5] for r in sub); wr = sum(r[6] for r in sub); ob = sum(r[7] for r in sub)
        hit = sum(1 for r in sub if r[6] or r[7])
        print("%-9s %6d %8d %8d %8d %8d %8d" % (s, len(sub), ok + wr + ob, ok, wr, ob, hit))
        tot[0] += len(sub); tot[1] += ok; tot[2] += wr; tot[3] += ob; tot[4] += hit
    print("%-9s %6d %8d %8d %8d %8d %8d" % ("TOTAL", tot[0], tot[1] + tot[2] + tot[3], tot[1], tot[2], tot[3], tot[4]))
    bad = [r for r in rows if r[8]]
    if bad:
        print("\n%d events could not be read:" % len(bad))
        for r in bad[:10]:
            print("  ", r[0], r[1], r[8])
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
