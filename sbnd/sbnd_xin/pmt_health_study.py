#!/usr/bin/env python3
"""Per-PMT PE health across the 150-event lan-reco2 data sample, read straight from
the opflash tensors (channel `ch` -> array column ch+1; col0 = flash time)."""
import numpy as np, json, glob, os, tarfile, tempfile, statistics

BASE = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/input_files/input-3files-lan-reco2"
SUBDIRS = ["1", "2", "3"]
CALIB = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work/ql_evt58667/calib-evt58667.json"
CH_MASK = {39,64,66,67,71,85,86,87,92,115,138,141,170,197,217,218,221,222,223,226,245,248,249,302}

# channel -> (apa, type) from the calib opdet table (same detector)
opd = {o["ch"]: o for o in json.load(open(CALIB))["opdets"]}

# accumulate per-channel PE across all flashes in the sample (only for the channel's own apa)
pe_vals = {ch: [] for ch in opd}
nflash = {0: 0, 1: 0}
nevt = 0
for sd in SUBDIRS:
    for apa in (0, 1):
        arc = os.path.join(BASE, sd, "opflash_apa%d.tar.gz" % apa)
        if not os.path.isfile(arc):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(arc) as tf:
                tf.extractall(tmp)
            for f in glob.glob(os.path.join(tmp, "opflash_tensor_*_0_array.npy")):
                a = np.load(f)                       # (nflash, 313)
                if a.ndim != 2 or a.shape[1] < 313:
                    continue
                nflash[apa] += a.shape[0]
                for ch, o in opd.items():
                    if o["apa"] != apa:
                        continue
                    pe_vals[ch].extend(a[:, ch + 1].tolist())

print("flashes scanned: apa0=%d apa1=%d" % (nflash[0], nflash[1]))

def stats(ch):
    v = pe_vals[ch]
    if not v:
        return None
    v_sorted = sorted(v)
    n = len(v)
    p = lambda q: v_sorted[min(n - 1, int(q * n))]
    return dict(n=n, mx=max(v), med=statistics.median(v), p90=p(0.90), p99=p(0.99),
                f5=sum(x > 5 for x in v) / n, f20=sum(x > 20 for x in v) / n)

# ---- ch69 focus ----
s = stats(69)
print("\n=== ch69 (apa%d type%d) across the 150-event sample ===" % (opd[69]["apa"], opd[69]["type"]))
print("   n_flash=%d  max=%.2f  median=%.2f  p90=%.2f  p99=%.2f  frac>5PE=%.3f  frac>20PE=%.3f"
      % (s["n"], s["mx"], s["med"], s["p90"], s["p99"], s["f5"], s["f20"]))

# ---- all PMTs: rank by max PE; flag chronically-low UNMASKED PMTs ----
print("\n=== type-1 PMTs sorted by max PE over the whole sample ===")
print("(masked = already in global ch_mask)")
rows = []
for ch, o in opd.items():
    if o["type"] != 1:
        continue
    s = stats(ch)
    if s is None:
        continue
    rows.append((s["mx"], ch, o["apa"], ch in CH_MASK, s))
rows.sort()
print("\n  ch  apa  masked   max     p99    p90   med   frac>5PE  frac>20PE")
for mx, ch, apa, masked, s in rows:
    flag = ""
    if not masked and s["mx"] < 50:
        flag = "  <== LOW & UNMASKED"
    if mx < 100 or masked or ch == 69:      # print the low tail + all masked for context
        print("  %3d  %d   %-6s  %7.1f %7.1f %6.1f %5.1f   %.3f     %.3f%s"
              % (ch, apa, "Y" if masked else "", s["mx"], s["p99"], s["p90"], s["med"],
                 s["f5"], s["f20"], flag))

# ---- summary: candidate run-bad PMTs (unmasked, chronically low) ----
print("\n=== CANDIDATE bad-in-this-run PMTs (unmasked type-1, max PE < 50 over 150 evt) ===")
cand = [(mx, ch, apa, s) for mx, ch, apa, masked, s in rows if not masked and mx < 50]
for mx, ch, apa, s in sorted(cand):
    print("   ch%3d apa%d  max=%.1f p99=%.1f med=%.2f frac>5PE=%.3f" % (ch, apa, mx, s["p99"], s["med"], s["f5"]))
print("   total candidates:", len(cand))
# healthy reference: median max PE of clearly-healthy PMTs
healthy_max = sorted(mx for mx, ch, apa, masked, s in rows if not masked and mx >= 50)
if healthy_max:
    print("\n   for scale: among healthy unmasked PMTs, max PE ranges %.0f .. %.0f (median %.0f)"
          % (healthy_max[0], healthy_max[-1], statistics.median(healthy_max)))
