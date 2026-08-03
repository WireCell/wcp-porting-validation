#!/usr/bin/env python3
"""mcp1k census: work-mcp1kall-oc19on1k (guard v2 + adopt ON) vs work-mcp1kall-u17on1kb (pr/17 baseline).
Whitespace-split tsv parsing. Beam-label census a la pr/16/18."""
import os, glob, collections

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
A = os.path.join(BASE, "work-mcp1kall-u17on1kb")
B = os.path.join(BASE, "work-mcp1kall-oc19on1k")

def load(arm, evt):
    path = os.path.join(arm, f"nusel_evt{evt}", f"nusel-evt{evt}.tsv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        hdr = f.readline().split()
        for l in f:
            t = l.split()
            if len(t) == len(hdr):
                rows.append(dict(zip(hdr, t)))
    return rows

evA = {os.path.basename(d)[9:] for d in glob.glob(os.path.join(A, "nusel_evt*"))}
evB = {os.path.basename(d)[9:] for d in glob.glob(os.path.join(B, "nusel_evt*"))}
print(f"events A={len(evA)} B={len(evB)} common={len(evA & evB)} onlyA={len(evA-evB)} onlyB={len(evB-evA)}")

KEY = ("nu-candidate", "tgm", "stm", "lm", "fc")
def keyof(lab):
    lab = lab.lower()
    for k in KEY:
        if k in lab:
            return k
    return None

n_ident = n_diff = n_missing = 0
verdict_changes = []   # events where the multiset of verdict-class labels changed
extra_only = 0
drifts = []
labclass_flow = collections.Counter()
for evt in sorted(evA & evB, key=int):
    ra, rb = load(A, evt), load(B, evt)
    if ra is None or rb is None:
        n_missing += 1
        continue
    def norm(rs):
        return sorted(tuple(v for k, v in r.items() if k not in ("main_id", "flash_gid")) for r in rs)
    if norm(ra) == norm(rb):
        n_ident += 1
        continue
    n_diff += 1
    ca = collections.Counter(keyof(r["label"]) for r in ra if keyof(r["label"]))
    cb = collections.Counter(keyof(r["label"]) for r in rb if keyof(r["label"]))
    if ca != cb:
        verdict_changes.append((evt, dict(ca), dict(cb)))
        for k in set(ca) | set(cb):
            d = cb.get(k, 0) - ca.get(k, 0)
            if d:
                labclass_flow[(k, "+" if d > 0 else "-")] += abs(d)
    else:
        extra_only += 1
    # nu-candidate npts drift
    na = {r["label"]: r for r in ra if "nu" in r["label"]}
    nb = {r["label"]: r for r in rb if "nu" in r["label"]}
    for lab in set(na) & set(nb):
        d = int(nb[lab]["npts_bundle"]) - int(na[lab]["npts_bundle"])
        if d:
            drifts.append((int(evt), lab, int(na[lab]["npts_bundle"]), int(nb[lab]["npts_bundle"]), d))

print(f"identical tables: {n_ident}   differing: {n_diff}   missing tsv: {n_missing}")
print(f"\nVERDICT-CLASS multiset changes: {len(verdict_changes)}")
for v in verdict_changes:
    print("  ", v)
print("\nlabel-class flow:", dict(labclass_flow))
print(f"\ndiffering-with-verdicts-unchanged (extra not-tagged rows / npts only): {extra_only}")
print(f"\nnu-npts drifts: {len(drifts)}  range: "
      f"{min((d[4] for d in drifts), default=0)}..{max((d[4] for d in drifts), default=0)}")
for v in sorted(drifts, key=lambda x: abs(x[4]), reverse=True)[:20]:
    print("  evt%-7d %-14s %6d -> %6d  (%+d)" % v)
