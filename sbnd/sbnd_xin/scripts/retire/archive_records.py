#!/usr/bin/env python3
"""Additive archive of the record layer of retiring sbnd_xin work-* arms.

For every arm in the removal set R (plan.json) write, under archive/records/:
  <campaign>/<tag>.tar.gz       -- every REAL file except the heavy classes
                                   (pctree*.tar.gz, mabc*.zip, calib-evt*.json*, *.npz)
  <campaign>/<tag>.links.txt    -- the symlink map (provenance: which arm each
                                   un-redone stage pointed into)
  <campaign>/<tag>.manifest.tsv -- per-class file count + bytes, kept vs dropped
  labels/<tag>/nusel_labels/    -- verbatim copy of hand-scan label dirs (M13)

Reads only; never moves, deletes or rewrites anything under work-*.

State files (inv.json / plan.json) are written to $RETIRE_STATE (default: cwd).
"""
import os, re, json, tarfile, shutil, sys, collections

ROOT = "/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin"
SCR  = os.environ.get("RETIRE_STATE", os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "archive", "records")
os.chdir(ROOT)

plan  = json.load(open(os.path.join(SCR, "plan.json")))
R, grp = plan["R"], plan["group"]
CAMP = {"R-doc63":"doc63-stm-campaign", "R-doc66":"doc66-diffusion",
        "R-doc60":"doc60-trackfit", "R-docs52-57":"docs52-57-arms", "R-dbg":"stmcamp-dbg"}

HEAVY = [("pctree",  re.compile(r'^pctree.*\.tar\.gz$')),
         ("mabc",    re.compile(r'^mabc.*\.zip$')),
         ("calib",   re.compile(r'^calib-evt.*\.json(\.gz)?$')),
         ("npz",     re.compile(r'.*\.npz$')),
         ("clusters",re.compile(r'^clusters-apa.*\.tar\.gz$'))]
def heavy_class(f):
    for name, pat in HEAVY:
        if pat.match(f): return name
    return None

tot_kept = tot_dropped = 0
for tag in sorted(R):
    camp = CAMP[grp[tag]]
    d = os.path.join(OUT, camp); os.makedirs(d, exist_ok=True)
    tgz  = os.path.join(d, tag + ".tar.gz")
    keep, links = [], []
    cls = collections.defaultdict(lambda: [0, 0])   # class -> [n, bytes]
    for cur, sub, files in os.walk(tag):
        for name in list(sub):
            p = os.path.join(cur, name)
            if os.path.islink(p):
                links.append(f"{p}\t->\t{os.readlink(p)}"); sub.remove(name)
        for f in files:
            p = os.path.join(cur, f)
            if os.path.islink(p):
                links.append(f"{p}\t->\t{os.readlink(p)}"); continue
            try: sz = os.path.getsize(p)
            except OSError: continue
            hc = heavy_class(f)
            cls[hc or "record"][0] += 1; cls[hc or "record"][1] += sz
            if hc is None: keep.append(p)
    with tarfile.open(tgz, "w:gz") as tf:
        for p in sorted(keep): tf.add(p)
    with open(os.path.join(d, tag + ".links.txt"), "w") as fh:
        fh.write("\n".join(sorted(links)) + ("\n" if links else ""))
    with open(os.path.join(d, tag + ".manifest.tsv"), "w") as fh:
        fh.write("class\tdisposition\tfiles\tbytes\n")
        for k, (n, b) in sorted(cls.items()):
            fh.write(f"{k}\t{'ARCHIVED' if k=='record' else 'DROPPED'}\t{n}\t{b}\n")
        fh.write(f"symlinks\tRECORDED\t{len(links)}\t0\n")
    # hand-scan labels, verbatim
    lab = os.path.join(tag, "nusel_labels")
    if os.path.isdir(lab) and not os.path.islink(lab):
        dst = os.path.join(OUT, "labels", tag, "nusel_labels")
        if os.path.exists(dst): shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(lab, dst, symlinks=True)
    k = cls["record"][1]; dr = sum(v[1] for kk, v in cls.items() if kk != "record")
    tot_kept += k; tot_dropped += dr
    print(f"{tag:38s} archived {k/2**20:8.1f} MiB -> {os.path.getsize(tgz)/2**20:7.1f} MiB gz   drops {dr/2**20:8.1f} MiB", flush=True)

print(f"\nTOTAL archived {tot_kept/2**20:.1f} MiB (raw)  dropped-if-removed {tot_dropped/2**30:.2f} GiB")
print("archive/records size:", os.popen(f"du -sh {OUT}").read().strip())
