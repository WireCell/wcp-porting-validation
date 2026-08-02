#!/usr/bin/env python3
"""Additive archive of the record layer of the 2026-08-02 retiring arms.

Fork of archive_records.py (2026-07-30 round). Differences:
  - reads plan.json from scripts/retire/state-20260802/ (plan_20260802.py);
  - campaign dirs live under archive/records/pr-era-20260802/<group>/;
  - HEAVY (dropped) additionally includes opflash_apa*.tar.gz -- justified by
    lightcheck_20260802.py, which proved every dropped opflash/sp-frames file
    byte-identical to a surviving copy EXCEPT the paths in
    state-20260802/light_exceptions.txt, which are force-ARCHIVED here;
  - there are no nusel_labels/ dirs in this round's removal set (verified),
    but the label-copy branch is kept as a belt-and-braces guard.

For every arm writes <group>/<tag>.tar.gz + <tag>.links.txt + <tag>.manifest.tsv.
Reads only; never moves, deletes or rewrites anything under work-*.
"""
import os, re, json, tarfile, shutil, collections

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR  = os.path.join(ROOT, "scripts", "retire", "state-20260802")
OUT  = os.path.join(ROOT, "archive", "records", "pr-era-20260802")
os.chdir(ROOT)

plan = json.load(open(os.path.join(SCR, "plan.json")))
R, grp = plan["R"], plan["group"]
exc_path = os.path.join(SCR, "light_exceptions.txt")
FORCE = set()
if os.path.exists(exc_path):
    FORCE = {l.strip() for l in open(exc_path) if l.strip()}

HEAVY = [("pctree",  re.compile(r'^pctree.*\.tar\.gz$')),
         ("mabc",    re.compile(r'^mabc.*\.zip$')),
         ("calib",   re.compile(r'^calib-evt.*\.json(\.gz)?$')),
         ("npz",     re.compile(r'.*\.npz$')),
         ("clusters",re.compile(r'^clusters-apa.*\.tar\.gz$')),
         ("opflash", re.compile(r'^opflash_apa.*\.tar\.gz$'))]
def heavy_class(f):
    for name, pat in HEAVY:
        if pat.match(f): return name
    return None

tot_kept = tot_dropped = nforced = 0
for tag in sorted(R):
    d = os.path.join(OUT, grp[tag]); os.makedirs(d, exist_ok=True)
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
            if hc is not None and p in FORCE:
                hc = None; nforced += 1
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
    lab = os.path.join(tag, "nusel_labels")
    if os.path.isdir(lab) and not os.path.islink(lab):
        dst = os.path.join(ROOT, "archive", "records", "labels", tag, "nusel_labels")
        if os.path.exists(dst): shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(lab, dst, symlinks=True)
        print(f"!! unexpected nusel_labels in removal arm, copied verbatim: {tag}")
    k = cls["record"][1]; dr = sum(v[1] for kk, v in cls.items() if kk != "record")
    tot_kept += k; tot_dropped += dr
    print(f"{tag:38s} archived {k/2**20:8.1f} MiB -> {os.path.getsize(tgz)/2**20:7.1f} MiB gz   drops {dr/2**20:8.1f} MiB", flush=True)

print(f"\nTOTAL archived {tot_kept/2**20:.1f} MiB (raw)  dropped-if-removed {tot_dropped/2**30:.2f} GiB  forced-light-files {nforced}/{len(FORCE)}")
print("archive dir size:", os.popen(f"du -sh {OUT}").read().strip())
