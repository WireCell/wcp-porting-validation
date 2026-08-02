#!/usr/bin/env python3
"""Census: work-nuecc48-oc19on2 (guard v2 with 40cm dis_floor + adopt) vs work-nuecc48-u17on (baseline).
Whitespace-split parsing (tables are space-separated, NOT tab)."""
import os, glob, collections

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
A = os.path.join(BASE, "work-nuecc48-u17on")     # baseline ON arm (pr/17 validated)
B = os.path.join(BASE, "work-nuecc48-oc19on2")   # v2 guard+adopt arm

def load(arm, evt):
    path = os.path.join(arm, f"nusel_evt{evt}", f"nusel-evt{evt}.tsv")
    rows = []
    with open(path) as f:
        hdr = f.readline().split()
        for l in f:
            t = l.split()
            if len(t) != len(hdr):
                continue
            rows.append(dict(zip(hdr, t)))
    return rows

evts = sorted(int(d.split("evt")[1]) for d in glob.glob(os.path.join(B, "nusel_evt*")))
print(f"events: {len(evts)}")

n_ident = n_diff = 0
verdict_changes = []
beam_label_changes = []
npts_drifts = []
extra_rows = []
for evt in evts:
    ra, rb = load(A, evt), load(B, evt)
    la = collections.Counter(r["label"] for r in ra)
    lb = collections.Counter(r["label"] for r in rb)
    # beam-window nu-candidate rows (labels containing 'nu' or in_beam)
    def beamrows(rs):
        return [r for r in rs if r["in_beam"] == "1" or "nu" in r["label"]]
    ba, bb = beamrows(ra), beamrows(rb)
    # raw table comparison ignoring main_id (cluster ids drift)
    def norm(rs):
        return sorted(tuple(v for k, v in r.items() if k not in ("main_id", "flash_gid")) for r in rs)
    if norm(ra) == norm(rb):
        n_ident += 1
        continue
    n_diff += 1
    if la != lb:
        # label multiset changed: check whether any nu-candidate/tgm/stm class changed
        da = la - lb
        db = lb - la
        keyclasses = ("nu-candidate", "tgm", "stm", "lm", "fc")
        if any(any(kc in lab for kc in keyclasses) for lab in list(da) + list(db)):
            verdict_changes.append((evt, dict(da), dict(db)))
        else:
            extra_rows.append((evt, dict(da), dict(db)))
    # nu-candidate npts drift
    na = {r["label"]: r for r in ba}
    nb = {r["label"]: r for r in bb}
    for lab in set(na) & set(nb):
        if "nu" in lab:
            d = int(nb[lab]["npts_bundle"]) - int(na[lab]["npts_bundle"])
            if d:
                npts_drifts.append((evt, lab, int(na[lab]["npts_bundle"]), int(nb[lab]["npts_bundle"]), d))
    # beam-row label change
    if sorted(r["label"] for r in ba) != sorted(r["label"] for r in bb):
        beam_label_changes.append((evt, sorted(r["label"] for r in ba), sorted(r["label"] for r in bb)))

print(f"identical tables: {n_ident}/{len(evts)}   differing: {n_diff}")
print(f"\nVERDICT-CLASS label changes: {len(verdict_changes)}")
for v in verdict_changes: print("  ", v)
print(f"\nbeam-row label changes: {len(beam_label_changes)}")
for v in beam_label_changes: print("  ", v)
print(f"\nnon-verdict label multiset changes (extra/missing rows): {len(extra_rows)}")
for v in extra_rows: print("  ", v)
print(f"\nnu-npts drifts: {len(npts_drifts)}")
for v in sorted(npts_drifts, key=lambda x: x[4]): print("  evt%-7d %-14s %6d -> %6d  (%+d)" % v)
