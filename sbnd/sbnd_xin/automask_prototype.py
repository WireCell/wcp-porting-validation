#!/usr/bin/env python3
"""Prototype + validation of the QLMatching per-event dynamic PMT auto-mask.

Rule (per event, per TPC): mask PMT i if
  (1) the TPC has >= MIN_FLASH flashes, AND
  (2) i never fires:  max_f pe_i[f] < PE_LOW, AND
  (3) i should have fired: in >= N_CONTRAST flashes the median PE of its K nearest
      *live* same-TPC PMTs exceeds PE_BRIGHT (neighbours bright, i still dark).
Only currently-active type-1 PMTs (not in the static ch_mask) are candidates/neighbours.

Validated on: lan-reco2 (150 evt, ch69 dead) and the original 10 data events (ch69 healthy)
and a few MC events. Goal: flag ch69 in lan-reco2, NEVER in the original, no healthy FPs.
"""
import json, glob, os, tarfile, tempfile, math, statistics
import numpy as np

SX = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
CALIB0 = SX + "/work/ql_evt58667/calib-evt58667.json"
CH_MASK = {39,64,66,67,71,85,86,87,92,115,138,141,170,197,217,218,221,222,223,226,245,248,249,302}

# params
PE_LOW, K, PE_BRIGHT, N_CONTRAST, MIN_FLASH = 5.0, 4, 50.0, 2, 3

opd = {o["ch"]: o for o in json.load(open(CALIB0))["opdets"]}
# precompute, per TPC, the candidate PMTs and their K nearest live (active, non-static-mask) PMTs
def tpc_pmts(apa):
    return [ch for ch,o in opd.items() if o["apa"]==apa and o["type"]==1 and ch not in CH_MASK]
NEAR = {}
for apa in (0,1):
    pm = tpc_pmts(apa)
    for ch in pm:
        o=opd[ch]
        d=sorted((math.hypot(opd[c]["y"]-o["y"], opd[c]["z"]-o["z"]), c) for c in pm if c!=ch)
        NEAR[ch]=[c for _,c in d[:K]]

def automask(pe_by_flash, apa):
    """pe_by_flash: list of dict{ch:pe} for each flash in this TPC. Returns set masked."""
    nf = len(pe_by_flash)
    masked=set()
    if nf < MIN_FLASH: return masked
    for ch in tpc_pmts(apa):
        pes=[f[ch] for f in pe_by_flash]
        if max(pes) >= PE_LOW:        # it fires somewhere -> healthy
            continue
        contrast=0
        for f in pe_by_flash:
            nb=[f[c] for c in NEAR[ch]]
            if nb and statistics.median(nb) > PE_BRIGHT:
                contrast+=1
        if contrast >= N_CONTRAST:
            masked.add(ch)
    return masked

# ---- data loaders: per event -> {apa: [ {ch:pe} per flash ]} ----
def from_calib(path):
    d=json.load(open(path)); out={0:[],1:[]}
    for f in d["flashes"]:
        out[f["apa"]].append({ch:f["pe"][ch] for ch in opd})
    return out

def from_opflash_dir(d):
    out={0:[],1:[]}
    for apa in (0,1):
        arc=os.path.join(d,"opflash_apa%d.tar.gz"%apa)
        if not os.path.isfile(arc): continue
        with tempfile.TemporaryDirectory() as tmp:
            tarfile.open(arc).extractall(tmp)
            for fp in sorted(glob.glob(os.path.join(tmp,"opflash_tensor_*_0_array.npy"))):
                a=np.load(fp)
                eid=os.path.basename(fp).split("_")[2]
                for r in range(a.shape[0]):
                    out[apa].append((eid,{ch:float(a[r,ch+1]) for ch in opd}))
    # group by event id
    ev={}
    for apa in (0,1):
        for eid,ped in out[apa]:
            ev.setdefault(eid,{0:[],1:[]})[apa].append(ped)
    return ev

print("params: PE_LOW=%g K=%d PE_BRIGHT=%g N_CONTRAST=%d MIN_FLASH=%d\n"%(PE_LOW,K,PE_BRIGHT,N_CONTRAST,MIN_FLASH))

# ---- lan-reco2: all 150 events ----
print("=== lan-reco2 (150 events) ===")
from collections import Counter
masked_counter=Counter(); nev=0; ev_with_mask=0
for sd in ("1","2","3"):
    ev=from_opflash_dir(os.path.join(SX,"input_files/input-3files-lan-reco2",sd))
    for eid,perapa in ev.items():
        nev+=1
        m=automask(perapa[0],0)|automask(perapa[1],1)
        if m: ev_with_mask+=1
        for ch in m: masked_counter[ch]+=1
print("events=%d, events with >=1 auto-mask=%d"%(nev,ev_with_mask))
print("channels auto-masked (ch: #events):", dict(sorted(masked_counter.items(), key=lambda kv:-kv[1])))

# ---- original 10 data events (ch69 healthy here -> must NOT be masked) ----
print("\n=== original data events 686..2050 ===")
orig=(686,1258,1302,1346,1698,1720,1808,1852,2028,2050)
cnt=Counter()
for fp in sorted(glob.glob(SX+"/work/ql_evt*/calib-evt*.json")):
    eid=int(''.join(c for c in os.path.basename(fp) if c.isdigit()))
    if eid not in orig: continue
    pa=from_calib(fp)
    m=automask(pa[0],0)|automask(pa[1],1)
    for ch in m: cnt[ch]+=1
    if m: print("  evt%d -> masked %s"%(eid,sorted(m)))
print("original-data auto-mask tally:", dict(cnt), " (ch69 should be ABSENT)")

# ---- MC events ----
print("\n=== mc events ===")
mc=(2,9,11,12,14,18,31,35,41,42)
cntm=Counter()
for fp in sorted(glob.glob(SX+"/work/ql_evt*/calib-evt*.json")):
    eid=int(''.join(c for c in os.path.basename(fp) if c.isdigit()))
    if eid not in mc: continue
    pa=from_calib(fp)
    m=automask(pa[0],0)|automask(pa[1],1)
    for ch in m: cntm[ch]+=1
    if m: print("  evt%d -> masked %s"%(eid,sorted(m)))
print("mc auto-mask tally:", dict(cntm))
