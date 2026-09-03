#!/usr/bin/env python3
"""doc pdvd/28: byte-identity gate between two PDVD PR arms (tags), plus the
wall/core/RSS comparison.  Per event: mabc-pr.zip member-content hash
(abtest/hash_archive.py), calib-pr-evt*.json hash minus the
vertex_scoreboard.dual_chain timer (feedback_calib_dump_cmp_timer_field),
tracking-pr.root / tracking-stm.root content hash over every branch of every
TTree via uproot (ROOT files embed timestamps, so never cmp them raw).

Usage: python3 stm/gates/r28_gate.py <baseTag> <postTag> [--out stm/gates/r28_<post>_vs_<base>.txt]
Exit 0 iff every present pair hashes identical and nothing is missing.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.join(os.path.dirname(PDVD), "abtest", "hash_archive.py")


def h_zip(p):
    return subprocess.check_output(["python3", H, p]).split()[0].decode()


def _strip(o):
    if isinstance(o, dict):
        vs = o.get("vertex_scoreboard")
        if isinstance(vs, dict):
            vs.pop("dual_chain", None)
        for v in o.values():
            _strip(v)
    elif isinstance(o, list):
        for v in o:
            _strip(v)


def h_json(p):
    d = json.load(open(p))
    _strip(d)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def h_root(p):
    """Content hash over every branch of every TTree: awkward buffers (form +
    offsets + contents), so nested std::vector branches hash by value."""
    import uproot
    import awkward as ak
    import numpy as np
    h = hashlib.sha256()
    with uproot.open(p) as f:
        for key in sorted(f.keys(cycle=False)):
            obj = f[key]
            if not isinstance(obj, uproot.behaviors.TTree.TTree):
                continue
            h.update(key.encode())
            if key == "T_rec_charge":
                # Row ORDER of this tree is layout-dependent (qlport
                # repeat_check.sh gate 2, doc 90 sec 4/7: two rows of a shared
                # vertex swap between runs/binaries); hash the MULTISET of rows.
                cols = [np.ascontiguousarray(ak.to_numpy(ak.flatten(obj[b].array(library="ak"), axis=None)))
                        for b in sorted(obj.keys())]
                rows = np.concatenate([c.astype(np.float64).reshape(-1, 1) for c in cols], axis=1)
                rb = np.ascontiguousarray(rows).view(np.uint8).reshape(rows.shape[0], -1)
                order = np.lexsort(rb.T[::-1])
                h.update("|".join(sorted(obj.keys())).encode())
                h.update(rb[order].tobytes())
                continue
            for bname in sorted(obj.keys()):
                h.update(bname.encode())
                arr = obj[bname].array(library="ak")
                form, length, container = ak.to_buffers(arr)
                h.update(form.to_json().encode())
                h.update(str(length).encode())
                for k in sorted(container):
                    h.update(k.encode())
                    h.update(np.ascontiguousarray(container[k]).tobytes())
    return h.hexdigest()


def resource(d):
    r = glob.glob(os.path.join(d, "pr_resource_*.txt"))
    wall, peak = -1, -1.0
    if r:
        m = re.search(r"wall_s=(\d+) peak_rss_gb=([0-9.]+)", open(r[0]).read())
        if m:
            wall, peak = int(m.group(1)), float(m.group(2))
    core = -1.0
    for log in glob.glob(os.path.join(d, "wct_pr_*.log")):
        for line in open(log, errors="replace"):
            m = re.search(r"Timer: Total ([0-9.]+) wall-sec, ([0-9.]+) core-sec", line)
            if m:
                core = float(m.group(2))
    return wall, core, peak


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    base, post = sys.argv[1], sys.argv[2]
    out = os.path.join(PDVD, "stm", "gates", "r28_%s_vs_%s.txt" % (post, base))
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    tot = {"same": 0, "diff": 0, "missing": 0}
    lines = []
    rows = []
    for bd in sorted(glob.glob(os.path.join(PDVD, "work", "*_%s" % base))):
        ev = os.path.basename(bd)[: -len(base) - 1]
        pd = os.path.join(PDVD, "work", "%s_%s" % (ev, post))
        if not os.path.isdir(pd) or not glob.glob(os.path.join(pd, "pr_resource_*.txt")):
            continue   # post arm has not produced this event (yet)
        for kind, pat, fn in (("bee", "mabc-pr.zip", h_zip), ("calib", "calib-pr-evt*.json", h_json),
                              ("trkpr", "tracking-pr.root", h_root), ("trkstm", "tracking-stm.root", h_root)):
            a, b = glob.glob(os.path.join(bd, pat)), glob.glob(os.path.join(pd, pat))
            if not a and not b:
                continue
            if not a or not b:
                tot["missing"] += 1
                lines.append("MISSING %s %s" % (ev, kind))
                continue
            ha, hb = fn(a[0]), fn(b[0])
            k = "same" if ha == hb else "diff"
            tot[k] += 1
            lines.append("%s %s %s %s %s" % (k.upper(), ev, kind, ha[:16], hb[:16]))
        wb, cb, pb = resource(bd)
        wp, cp, pp = resource(pd)
        rows.append((ev, wb, wp, cb, cp, pb, pp))
    with open(out, "w") as fh:
        fh.write("# r28 gate %s vs %s: %s\n" % (post, base, tot))
        fh.write("\n".join(lines) + "\n")
        fh.write("# event\twall_base\twall_post\tcore_base\tcore_post\trss_base\trss_post\n")
        for r in rows:
            fh.write("%s\t%d\t%d\t%.1f\t%.1f\t%.2f\t%.2f\n" % r)
    print("R28 gate %s vs %s: %s -> %s" % (post, base, tot, out))
    print("%-12s %7s %7s %8s %8s %7s %7s %6s" % ("event", "wall_b", "wall_p", "core_b", "core_p", "rss_b", "rss_p", "speed"))
    for ev, wb, wp, cb, cp, pb, pp in rows:
        print("%-12s %7d %7d %8.1f %8.1f %7.2f %7.2f %6.2fx" % (ev, wb, wp, cb, cp, pb, pp, cb / cp if cp > 0 else 0))
    sb, sp = sum(r[3] for r in rows), sum(r[4] for r in rows)
    print("%-12s %7d %7d %8.1f %8.1f %7s %7s %6.2fx" % ("SUM", sum(r[1] for r in rows), sum(r[2] for r in rows), sb, sp, "", "", sb / sp if sp else 0))
    return 0 if tot["diff"] == 0 and tot["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
