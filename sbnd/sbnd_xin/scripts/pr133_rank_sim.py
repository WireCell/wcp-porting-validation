#!/usr/bin/env python3
"""doc pr/133 iteration 2 -- the P1 pair-selection ranking simulator.

Fork-of-method from pr132_angle_census.py (tape-first, no knob spent): reads
the WCT_PI0_PAIR_DEBUG path-1 tape (P1 begin / P1 pair / P1 accept lines)
from the --arm-prefix arms and re-runs the greedy selection OFFLINE under
alternative ranking policies, scoring each policy by how many HAND label
pairs (base emscan + --overlay-tag) end up in the accepted set, and by which
events flip either way vs the taped legacy selection.

Motivation: SBND 18255-166870 (doc pr/133 sec 4.4) -- with K20 the true pair
(m=114.6, in-window) exists and LOSES the ranking to a crumb-partner pair
(m=123.8): the legacy key is |m-125| MINUS a 6 MeV bonus for ct2+ct2 pairs.

Policies:
  legacy      |m-125| - 6*(ct1==2 and ct2==2)         (the C++ selection loop)
  nobonus     |m-125|
  mainfirst   (vtx != main_vtx, legacy)               (main-vertex pairs rank first)
  mainnb      (vtx != main_vtx, |m-125|)
K3=28 veto (production) is applied in all policies.
"""
import argparse, os, re, sys, importlib.util
from collections import Counter

_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(os.path.dirname(__file__), "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)
SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OFFSET = 10.0; K3 = 28.0
def delta(m): return m - 135.0 + OFFSET
def inwin(m): return -25.0 < delta(m) < 35.0

PAT_BEGIN = re.compile(r"PI0_PAIR P1 begin main_vtx=(-?\d+)")
PAT_PAIR = re.compile(r"PI0_PAIR P1 pair sh1=(\d+) sh2=(\d+) ct1=(\d+) ct2=(\d+) vtx=(-?\d+) E1=([\d.\-]+) E2=([\d.\-]+) m=([\d.\-]+)")
PAT_ACC = re.compile(r"PI0_PAIR P1 accept sh1=(\d+) sh2=(\d+) vtx=(-?\d+) m=([\d.\-]+)")

def read_tape(log):
    main_vtx, pairs, accepted = None, {}, []
    for line in open(log, errors="replace"):
        mb = PAT_BEGIN.search(line)
        if mb: main_vtx = int(mb.group(1)); continue
        mp = PAT_PAIR.search(line)
        if mp:
            s1, s2 = int(mp.group(1)), int(mp.group(2))
            pairs.setdefault((s1, s2), []).append(dict(
                ct1=int(mp.group(3)), ct2=int(mp.group(4)), vtx=int(mp.group(5)),
                E1=float(mp.group(6)), E2=float(mp.group(7)), m=float(mp.group(8))))
            continue
        ma = PAT_ACC.search(line)
        if ma:
            accepted.append((int(ma.group(1)), int(ma.group(2))))
    return main_vtx, pairs, accepted

def simulate(main_vtx, pairs, policy):
    remaining = dict(pairs)
    out = []
    while remaining:
        best = None
        for (s1, s2), recs in remaining.items():
            for r in recs:
                if not inwin(r["m"]): continue
                # K3=28 production veto: ct1 member attached at MAIN vertex
                # with a small partner.
                if (r["ct1"] == 1 and r["vtx"] == main_vtx and r["E2"] < K3): continue
                if (r["ct2"] == 1 and r["vtx"] == main_vtx and r["E1"] < K3): continue
                bonus = 6.0 if (r["ct1"] == 2 and r["ct2"] == 2) else 0.0
                base = abs(delta(r["m"])) - (bonus if policy in ("legacy", "mainfirst") else 0.0)
                key = ((0 if r["vtx"] == main_vtx else 1, base) if policy in ("mainfirst", "mainnb")
                       else (0, base))
                if best is None or key < best[0]:
                    best = (key, (s1, s2))
        if best is None: break
        acc = best[1]
        out.append(acc)
        remaining = {k: v for k, v in remaining.items()
                     if k[0] not in acc and k[1] not in acc and k[0] != acc[0] and k[0] != acc[1] and k[1] != acc[0] and k[1] != acc[1]}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest141"); ap.add_argument("--manifest98")
    ap.add_argument("--overlay-tag"); ap.add_argument("--arm-prefix", default="work-pr133-k2021p")
    a = ap.parse_args()
    if a.manifest98 or a.manifest141:
        newsets = []
        for t in SEL.SETS:
            t = list(t)
            if t[0] == "98" and a.manifest98: t[4] = a.manifest98
            if t[0] == "141" and a.manifest141: t[4] = a.manifest141
            newsets.append(tuple(t))
        SEL.SETS = newsets
    overlay = SEL.load_labels(a.overlay_tag) if a.overlay_tag else {}

    POL = ["legacy", "nobonus", "mainfirst", "mainnb"]
    score = Counter(); flips = {p: [] for p in POL}
    n_pairs = 0; calib_ok = 0; calib_bad = []
    seen = set()
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            sample = mrow.get("sample") or mrow.get("det") or ""
            log = os.path.join(SX, f"{a.arm_prefix}-{sample}", f"pr_evt{ev}", "stdout.log")
            if not os.path.exists(log): continue
            key_ev = (sample, ev)
            first_time = key_ev not in seen
            seen.add(key_ev)
            main_vtx, pairs, taped_acc = read_tape(log)
            if first_time and pairs:
                sim_leg = set(simulate(main_vtx, pairs, "legacy"))
                if sim_leg == set(taped_acc): calib_ok += 1
                else: calib_bad.append((sample, ev, sorted(sim_leg), sorted(taped_acc)))
            for labsrc, rec in (("base", labels.get(ev)), ("overlay", overlay.get(ev))):
                g = ((rec or {}).get("pio") or {}).get("gammas")
                if not g or not all(x in g and (g[x].get("energy") or 0) > 0 for x in ("1", "2")):
                    continue
                try:
                    i1 = int(g["1"].get("shower") or -1); i2 = int(g["2"].get("shower") or -1)
                except (TypeError, ValueError):
                    continue
                lab = tuple(sorted((i1, i2)))
                n_pairs += 1
                got = {}
                for p in POL:
                    acc = simulate(main_vtx, pairs, p)
                    got[p] = any(tuple(sorted(x)) == lab for x in acc)
                    if got[p]: score[p] += 1
                for p in POL:
                    if got[p] != got["legacy"]:
                        flips[p].append((sample, ev, "GAIN" if got[p] else "LOSS"))
                break  # one label source per event (base wins, overlay fallback)
            else:
                continue
    print(f"=== calibration: simulated legacy == taped accepts on {calib_ok} events; mismatches {len(calib_bad)} ===")
    for b in calib_bad[:10]: print("  CALIB", b)
    print(f"=== label pairs considered: {n_pairs} ===")
    for p in POL:
        print(f"  {p:10s} label-pair accepted: {score[p]}")
    for p in POL:
        if p == "legacy": continue
        for f in flips[p]:
            print(f"  {p:10s} {f[2]} {f[0]} {f[1]}")

if __name__ == "__main__":
    sys.exit(main())
