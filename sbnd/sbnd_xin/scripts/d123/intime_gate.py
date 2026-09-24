#!/usr/bin/env python3
"""doc sbnd_xin/123 round 0 -- the flash-time acceptance gate.

For every event of a group-layout arm (<root>/g*/opflash_apa{0,1}.tar.gz, one tensor set per
event per TPC), take the flash times (tensor 0, column 0, ns) plus the tensor-set metadata key
frame_apply_at_caf (ns, the per-event re-reference FlashTensorToOpticalPCs adds; required on data,
absent on MC) and ask whether the event has a flash in the beam window +0.3..1.9 us after the
correction (doc 21: 45/48 of the nueCC48 events do with SBND's own flashes).  The same check on
the reco1 flashes (a baseline arm, or reco1flash_apa*.tar.gz written with reco1_reference=true)
is the reference: a new flash source that drops the offset shows 0/48 here.

usage: intime_gate.py <root> [--name opflash|reco1flash] [--lo 0.3] [--hi 1.9] [--mc] [--tsv out]
"""
import argparse, glob, io, json, os, re, sys, tarfile
import numpy as np

RE_SET = re.compile(r"^opflash_tensorset_(\d+)_metadata\.json$")
RE_T0 = re.compile(r"^opflash_tensor_(\d+)_0_array\.npy$")


def read_arm(root, name):
    """{event: {tpc: (times_ns, pe, offset_ns or None)}}"""
    out = {}
    for tpc in (0, 1):
        for path in sorted(glob.glob(os.path.join(root, "g*", "%s_apa%d.tar.gz" % (name, tpc)))):
            md, arr = {}, {}
            with tarfile.open(path) as t:
                for m in t.getmembers():
                    a = RE_SET.match(m.name)
                    if a:
                        md[int(a.group(1))] = json.load(t.extractfile(m))
                        continue
                    b = RE_T0.match(m.name)
                    if b:
                        arr[int(b.group(1))] = np.load(io.BytesIO(t.extractfile(m).read()))
            for ev, d in md.items():
                x = arr.get(ev)
                if x is None or x.ndim != 2 or x.shape[0] == 0:
                    times, pe = np.zeros(0), np.zeros(0)
                else:
                    times, pe = x[:, 0], x[:, 1:].sum(axis=1)
                off = d.get("frame_apply_at_caf")
                out.setdefault(ev, {})[tpc] = (times, pe, off)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--name", default="opflash")
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=1.9)
    ap.add_argument("--mc", action="store_true", help="no frame_apply_at_caf expected")
    ap.add_argument("--tsv")
    a = ap.parse_args()
    arm = read_arm(a.root, a.name)
    if not arm:
        sys.exit("no %s_apa*.tar.gz under %s/g*" % (a.name, a.root))
    rows, n_in, n_nooff = [], 0, 0
    for ev in sorted(arm):
        best = None  # (pe, t_us, tpc)
        nflash = 0
        for tpc, (times, pe, off) in sorted(arm[ev].items()):
            if off is None:
                if not a.mc:
                    n_nooff += 1
                off = 0.0
            t_us = (times + off) / 1000.0
            nflash += len(t_us)
            sel = (t_us >= a.lo) & (t_us <= a.hi)
            for t, p in zip(t_us[sel], pe[sel]):
                if best is None or p > best[0]:
                    best = (p, t, tpc)
        ok = best is not None
        n_in += ok
        rows.append((ev, nflash, ok, best[1] if ok else float("nan"), best[0] if ok else 0.0, best[2] if ok else -1))
    print("%s %s: %d/%d events with a flash in [%.1f, %.1f] us; %d TPC sets without frame_apply_at_caf"
          % (a.root, a.name, n_in, len(rows), a.lo, a.hi, n_nooff))
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("event\tnflash\tintime\tt_us\tpe\ttpc\n")
            for r in rows:
                f.write("%d\t%d\t%d\t%.4f\t%.1f\t%d\n" % r)
    for r in rows:
        if not r[2]:
            print("  MISSING evt %d (%d flashes)" % (r[0], r[1]))


if __name__ == "__main__":
    main()
