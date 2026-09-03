#!/usr/bin/env python3
"""doc pdvd/27 §6.4: per-event census of a fresh (self-consistent) PR arm against a stale-geometry one.
For every event of stm/events.txt: mains evaluated, STM=1, TGM=true, nu candidates, retile WARN lines,
degenerate-split lines, PR wall time -- from wct_pr_<run>_<evt>.log and pr_resource_*.txt.
usage: r27_fresh_census.py <fresh_tag> <stale_tag> [out.tsv]"""
import sys, os, re, glob
PDVD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fresh, stale = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else f"{PDVD}/stm/gates/r27_census_{fresh}_vs_{stale}.tsv"
def stats(d):
    logs = glob.glob(f"{d}/wct_pr_*.log")
    if not logs: return None
    L = open(logs[0], errors="replace").read()
    r = dict(mains=len(re.findall(r"TaggerCheckSTM: cluster \d+ →", L)),
             stm=len(re.findall(r"TaggerCheckSTM: cluster \d+ → STM=1", L)),
             tgm=len(re.findall(r"TaggerCheckTGM: cluster \d+ → TGM=true", L)),
             nu=len(re.findall(r"candidate main cluster", L)),
             warn=L.count("the retile lands in the wrong face"),
             degen=L.count("degenerate split skipped"))
    res = glob.glob(f"{d}/pr_resource_*.txt"); r["wall_s"] = -1
    if res:
        m = re.search(r"wall_s=(\d+)", open(res[0]).read()); r["wall_s"] = int(m.group(1)) if m else -1
    r["sidecar_wires"] = ""
    tl = glob.glob(f"{d}/pctree-evt*.tlas")
    if tl:
        m = re.search(r"^wires=(.*)$", open(tl[0]).read(), re.M); r["sidecar_wires"] = m.group(1) if m else "(none)"
    return r
rows = []; keys = ["mains", "stm", "tgm", "nu", "warn", "degen", "wall_s"]
tot = {t: {k: 0 for k in keys} for t in (fresh, stale)}; n = {fresh: 0, stale: 0}
for line in open(f"{PDVD}/stm/events.txt"):
    if line.startswith("#") or not line.strip(): continue
    run, idx = line.split()[:2]; r6 = "%06d" % int(run)
    row = [run, idx]
    for tag in (fresh, stale):
        s = stats(f"{PDVD}/work/{r6}_{idx}_{tag}")
        if s is None: row += ["ABSENT"] * (len(keys) + 1); continue
        n[tag] += 1
        for k in keys: tot[tag][k] += s[k]
        row += [str(s[k]) for k in keys] + [s["sidecar_wires"]]
    rows.append(row)
hdr = ["run", "idx"] + [f"{t}_{k}" for t in (fresh, stale) for k in keys + ["wires"]]
with open(out, "w") as f:
    f.write("\t".join(hdr) + "\n")
    for r in rows: f.write("\t".join(r) + "\n")
print(f"wrote {out}")
for tag in (fresh, stale):
    print(f"{tag}: {n[tag]} events; " + ", ".join(f"{k}={tot[tag][k]}" for k in keys))
