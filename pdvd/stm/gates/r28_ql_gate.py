#!/usr/bin/env python3
"""doc pdvd/28 round 2: byte-identity gate for two PDVD Q/L-stage arms.

Usage: r28_ql_gate.py <tagA> <tagB> [--out]

For every work/<run6>_<idx>_<tagA>/ with a sibling _<tagB>/ dir, compares the
content hashes (abtest/hash_archive.py: member name + payload, timestamps
ignored) of the post-Q/L point-cloud tree (pctree-evt*.tar.gz, the PR job's
input), every Bee zip (mabc-*.zip) and the calib dump (calib-evt*.json, raw
bytes -- the Q/L dump carries no timer field), and reports the Deghost /
ProtectOverclustering stage times summed over the MultiAlgBlobClustering
instances of each log plus the runner's wall.  --out writes
stm/gates/r28_ql_<tagB>_vs_<tagA>.txt.
"""
import glob, hashlib, os, re, subprocess, sys

PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.join(os.path.dirname(PDVD), "abtest", "hash_archive.py")
TIMING = re.compile(r"MABC timing: (Clustering(?:Deghost|ProtectOverclustering)):\S+ took ([0-9.]+) ms")


def h_archive(p):
    return subprocess.check_output(["python3", H, p]).split()[0].decode()


def h_raw(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def stage_times(d):
    t = {"ClusteringDeghost": 0.0, "ClusteringProtectOverclustering": 0.0}
    for lg in glob.glob(os.path.join(d, "wct_clus_*.log")):
        for line in open(lg, errors="replace"):
            m = TIMING.search(line)
            if m:
                t[m.group(1)] += float(m.group(2)) / 1000.0
    wall = "?"
    for r in glob.glob(os.path.join(d, "clus_resource_*.txt")):
        m = re.search(r"wall_s=(\S+)", open(r).read())
        if m:
            wall = m.group(1)
    return t, wall


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    a_tag, b_tag = args
    out_lines, tot = [], {"same": 0, "diff": 0, "missing": 0}
    table = [f"{'event':<10} {'wall_a':>7} {'wall_b':>7} {'deghost_a':>9} {'deghost_b':>9} {'po_a':>7} {'po_b':>7}"]
    for da in sorted(glob.glob(os.path.join(PDVD, "work", f"*_{a_tag}"))):
        ev = os.path.basename(da)[: -len(a_tag) - 1]
        db = os.path.join(PDVD, "work", f"{ev}_{b_tag}")
        if not os.path.isdir(db):
            continue
        for pat, fn in (("pctree-evt*.tar.gz", h_archive), ("mabc-*.zip", h_archive), ("calib-evt*.json", h_raw)):
            fa = sorted(glob.glob(os.path.join(da, pat)))
            for pa in fa:
                pb = os.path.join(db, os.path.basename(pa))
                if not os.path.exists(pb):
                    tot["missing"] += 1
                    out_lines.append(f"MISSING {ev} {os.path.basename(pa)}")
                    continue
                ha, hb = fn(pa), fn(pb)
                k = "same" if ha == hb else "diff"
                tot[k] += 1
                out_lines.append(f"{k.upper():7} {ev} {os.path.basename(pa):34} {ha[:16]} {hb[:16]}")
        ta, wa = stage_times(da)
        tb, wb = stage_times(db)
        table.append(f"{ev:<10} {wa:>7} {wb:>7} {ta['ClusteringDeghost']:9.1f} {tb['ClusteringDeghost']:9.1f} "
                     f"{ta['ClusteringProtectOverclustering']:7.1f} {tb['ClusteringProtectOverclustering']:7.1f}")
    head = f"# r28_ql_gate {b_tag} vs {a_tag}: {tot}"
    text = "\n".join([head] + out_lines + ["", "# stage seconds (summed over the MABC instances of the event) and runner wall (s)"] + table) + "\n"
    print(text)
    if "--out" in sys.argv:
        p = os.path.join(PDVD, "stm", "gates", f"r28_ql_{b_tag}_vs_{a_tag}.txt")
        open(p, "w").write(text)
        print("wrote", p)


if __name__ == "__main__":
    main()
