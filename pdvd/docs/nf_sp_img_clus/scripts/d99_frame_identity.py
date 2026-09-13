#!/usr/bin/env python3
"""doc pdvd/99 sec 4.3 -- where do two SP frame arms differ, sample for sample?

    python3 d99_frame_identity.py --base keep --arm p98voff --events 039252_0 ... [--anodes 0-7] [--json J]
    python3 d99_frame_identity.py --base keep --arm p98von --all --anodes 0 1 2 3 --json J    # bottom, 120 events

Every frame_* member of each anode archive (gauss, wiener, ...) is compared after sorting by channel, per plane
(plane from the v7-uvwfit wire file, as d99_frames.py).  Per event x anode x plane x tag: identical or not, the number
of channels with any differing sample, and sum|a-b| / sum|a|.
"""
import argparse, collections, glob, io, json, os, tarfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import d99_frames as F


def load_all(path):
    arr = {}
    with tarfile.open(path, "r:bz2") as tf:
        for m in tf:
            if m.name.endswith(".npy"):
                arr[m.name[:-4]] = np.load(io.BytesIO(tf.extractfile(m).read()))
    out = {}
    for name, a in arr.items():
        if name.startswith("frame_"):
            tag = name[len("frame_"):]
            ch = arr.get("channels_" + tag)
            if ch is not None:
                o = np.argsort(ch)
                out[tag.split("_")[0]] = (ch[o].astype(int), a[o])
    return out


def one(job):
    evt, base, arm, anodes = job
    PM = F.plane_map()
    rows = []
    for an in anodes:
        fb = f"{F.WORK}/{evt}_{base}/protodune-sp-dnnroi-frames-anode{an}.tar.bz2"
        fa = f"{F.WORK}/{evt}_{arm}/protodune-sp-dnnroi-frames-anode{an}.tar.bz2"
        if not (os.path.exists(fb) and os.path.exists(fa)):
            continue
        B, A = load_all(fb), load_all(fa)
        for tag in sorted(set(B) | set(A)):
            if tag not in B or tag not in A:
                rows.append(dict(evt=evt, anode=an, tag=tag, plane="-", missing=True))
                continue
            (cb, xb), (ca, xa) = B[tag], A[tag]
            if not np.array_equal(cb, ca) or xb.shape != xa.shape:
                rows.append(dict(evt=evt, anode=an, tag=tag, plane="-", shape_differs=True))
                continue
            planes = np.array([PM.get((an, int(c)), -1) for c in cb])
            for p in range(3):
                m = planes == p
                d = np.abs(xb[m].astype(float) - xa[m].astype(float))
                den = np.abs(xb[m].astype(float)).sum()
                rows.append(dict(evt=evt, anode=an, tag=tag, plane=F.PL[p], identical=bool(d.max() == 0) if d.size else True,
                                 nchan_diff=int((d.max(axis=1) > 0).sum()) if d.size else 0, nchan=int(m.sum()),
                                 absdiff_frac=float(d.sum() / den) if den > 0 else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--anodes", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    evts = a.events
    if a.all:
        evts = sorted({os.path.basename(d)[:-len("_" + a.arm)] for d in glob.glob(f"{F.WORK}/*_{a.arm}")
                       if glob.glob(d + "/protodune-sp-dnnroi-frames-anode*.tar.bz2")})
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, [(e, a.base, a.arm, a.anodes) for e in evts]):
            rows += r
    print(f"base {a.base}  arm {a.arm}  events {len(evts)}  anodes {a.anodes}")
    bad = [r for r in rows if r.get("missing") or r.get("shape_differs")]
    print(f"  members missing or channel/shape mismatch: {len(bad)}")
    for tag in sorted({r["tag"] for r in rows}):
        for p in F.PL:
            rr = [r for r in rows if r["tag"] == tag and r["plane"] == p]
            if not rr:
                continue
            ni = sum(r["identical"] for r in rr)
            fr = [r["absdiff_frac"] for r in rr if not r["identical"]]
            print(f"  {tag:8s} {p}: identical {ni}/{len(rr)} event-anodes; where not: median sum|diff|/sum {np.median(fr) if fr else 0:.2e}"
                  f"  max {max(fr) if fr else 0:.2e}  channels differing {sum(r['nchan_diff'] for r in rr)}/{sum(r['nchan'] for r in rr)}")
    per_evt = collections.defaultdict(list)
    for r in rows:
        if not r.get("identical", False):
            per_evt[r["evt"]].append(f"{r['anode']}{r['plane']}:{r['tag']}")
    full = [e for e in evts if e not in per_evt]
    print(f"  events with every compared frame identical: {len(full)}/{len(evts)}  {' '.join(full)}")
    for e in evts:
        if e in per_evt:
            print(f"  differs {e}: {' '.join(sorted(set(per_evt[e])))}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=0)


if __name__ == "__main__":
    main()
