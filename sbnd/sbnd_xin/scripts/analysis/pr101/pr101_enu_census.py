#!/usr/bin/env python3
"""doc pr/101 -- Enu accounting census between two PR arms.

Reads pr_evt<ID>/calib-pr-evt<ID>.json (kine block, showers, tagger scores)
from two arms and the pr/101 census log lines from the B arm
(kine_long_muon / kine_hadronic / kine_track_ctx / kine_mass_census /
kine_mainvtx_guard / pi0 window reject / pi0 incoming stamp), and prints:

  * per-event TSV: Enu, add_energy, pi0 mass + paired flag, numu/nue score,
    A vs B, with flip flags at the selection thresholds;
  * per-sample summary: dEnu quantiles, pi0 pairs gained/lost, add_energy
    rule-residual (a +938 residual = a proton-typed shower took the proton
    mass), selection flips;
  * log censuses from arm B: long-muon range-vs-dQdx, hadronic showers
    (dqdx/charge), track contexts (won/ke).

Usage:
  pr101_enu_census.py <armA> <armB> [--out prefix] [--numu-sel 0.9] [--nue-sel 7.0]

Exit 0 always (reporting tool, not a gate).
"""
import argparse, glob, json, os, re, statistics, sys

MASS = {13: 105.658, 211: 139.570, 321: 493.677}
BIND = 8.6

def load_evt(arm, evt):
    p = os.path.join(arm, f"pr_evt{evt}", f"calib-pr-evt{evt}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def evts_of(arm):
    out = []
    for d in glob.glob(os.path.join(arm, "pr_evt*")):
        e = d.rsplit("pr_evt", 1)[1]
        if e.isdigit():
            out.append(int(e))
    return sorted(out)

def rule_add(types):
    """Paper rule: mass for mu/pi/K, binding for nucleons, nothing for e."""
    tot = 0.0
    for t in types:
        a = abs(int(t))
        if a in (2212, 2112):
            tot += BIND
        elif a in MASS:
            tot += MASS[a]
    return tot

def summarize(d):
    k = d["kine"]
    paired = any(s.get("pio_id", -1) >= 0 for s in d.get("showers", []))
    tg = d.get("tagger", {})
    return dict(
        enu=k["kine_reco_Enu"], add=k["kine_reco_add_energy"],
        pio_mass=k.get("kine_pio_mass", 0.0), pio_flag=k.get("kine_pio_flag", 0),
        paired=paired, types=k["kine_particle_type"], es=k["kine_energy_particle"],
        info=k["kine_energy_info"],
        numu=tg.get("numu_score", float("nan")), nue=tg.get("nue_score", float("nan")),
        n2212_shower=sum(1 for s in d.get("showers", []) if abs(int(s.get("particle_id", 0))) == 2212),
    )

def grep_logs(arm, evt, pat):
    # The same DEBUG line lands in both wct_pr_evt<ID>.log and stdout.log
    # (routing quirk, pr/99 r3) -- read the first file that has any hit.
    for name in (f"wct_pr_evt{evt}.log", "stdout.log"):
        p = os.path.join(arm, f"pr_evt{evt}", name)
        if not os.path.exists(p):
            continue
        out = []
        with open(p, errors="replace") as f:
            for line in f:
                if pat in line:
                    out.append(line.rstrip("\n"))
        if out:
            return out
    return []

def kv(line):
    return {m.group(1): m.group(2) for m in re.finditer(r"(\w+)=([^\s]+)", line)}

def q(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
    return v[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA"); ap.add_argument("armB")
    ap.add_argument("--out", default=None)
    ap.add_argument("--numu-sel", type=float, default=0.9)
    ap.add_argument("--nue-sel", type=float, default=7.0)
    a = ap.parse_args()

    evts = sorted(set(evts_of(a.armA)) & set(evts_of(a.armB)))
    rows, denu, gained, lost, flips = [], [], [], [], []
    lm, had, tctx, mass_c, guard, win_rej, stamp = [], [], [], [], [], [], []
    for e in evts:
        A, B = load_evt(a.armA, e), load_evt(a.armB, e)
        if A is None or B is None:
            continue
        sa, sb = summarize(A), summarize(B)
        d = sb["enu"] - sa["enu"]
        denu.append(d)
        if sb["paired"] and not sa["paired"]: gained.append(e)
        if sa["paired"] and not sb["paired"]: lost.append(e)
        fl = []
        if (sa["numu"] > a.numu_sel) != (sb["numu"] > a.numu_sel): fl.append("numu")
        if (sa["nue"] > a.nue_sel) != (sb["nue"] > a.nue_sel): fl.append("nue")
        if fl: flips.append((e, fl))
        resA = sa["add"] - rule_add(sa["types"]); resB = sb["add"] - rule_add(sb["types"])
        rows.append((e, sa["enu"], sb["enu"], d, sa["add"], sb["add"], resA, resB,
                     sa["pio_mass"], int(sa["paired"]), sb["pio_mass"], int(sb["paired"]),
                     sa["numu"], sb["numu"], sa["nue"], sb["nue"], ",".join(fl),
                     len(sa["types"]), len(sb["types"])))
        lm += [(e, kv(l)) for l in grep_logs(a.armB, e, "kine_long_muon:")]
        had += [(e, kv(l)) for l in grep_logs(a.armB, e, "kine_hadronic:")]
        tctx += [(e, kv(l)) for l in grep_logs(a.armB, e, "kine_track_ctx: seg")]
        mass_c += [(e, kv(l)) for l in grep_logs(a.armB, e, "kine_mass_census:")]
        guard += [(e, kv(l)) for l in grep_logs(a.armB, e, "kine_mainvtx_guard:")]
        win_rej += [(e, l) for l in grep_logs(a.armB, e, "pi0 window reject:")]
        stamp += [(e, l) for l in grep_logs(a.armB, e, "pi0 incoming stamp:")]

    hdr = ["event", "EnuA", "EnuB", "dEnu", "addA", "addB", "resA", "resB", "pioA", "pairA",
           "pioB", "pairB", "numuA", "numuB", "nueA", "nueB", "flip", "npartA", "npartB"]
    out = sys.stdout if not a.out else open(a.out + ".tsv", "w")
    print("\t".join(hdr), file=out)
    for r in rows:
        print("\t".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r), file=out)
    if a.out: out.close()

    print(f"# pr101_enu_census {a.armA} -> {a.armB}: {len(rows)} events")
    if denu:
        print(f"dEnu MeV: mean={statistics.mean(denu):.1f} median={statistics.median(denu):.1f} "
              f"q10={q(denu,.1):.1f} q90={q(denu,.9):.1f} min={min(denu):.1f} max={max(denu):.1f} "
              f"n_moved(|d|>1)={sum(1 for x in denu if abs(x)>1)}")
    nA = sum(r[9] for r in rows); nB = sum(r[11] for r in rows)
    print(f"pi0 pairs (pio_id>=0): A={nA} B={nB} gained={gained} lost={lost}")
    big = [r for r in rows if abs(r[6]) > 100 or abs(r[7]) > 100]
    print(f"add_energy minus naive per-type rule sum, |res|>100 MeV (continuation refunds make this negative; a +938 would be the proton-shower mass class): "
          f"A={sum(1 for r in rows if abs(r[6])>100)} B={sum(1 for r in rows if abs(r[7])>100)} "
          f"events={[r[0] for r in big]}")
    print(f"selection flips (numu>{a.numu_sel} / nue>{a.nue_sel}): {flips}")
    if lm:
        used = [x.get("used") for _, x in lm]
        ratios = [float(x["ratio"]) for _, x in lm if "ratio" in x]
        print(f"long muons (B): n={len(lm)} range={used.count('range')} dqdx={used.count('dqdx')} "
              f"ratio median={statistics.median(ratios):.2f} q10={q(ratios,.1):.2f} q90={q(ratios,.9):.2f}")
        for e, x in lm:
            print(f"  LM evt {e} id={x.get('id')} L={x.get('L_cm')} range={x.get('range')} dqdx={x.get('dqdx')} "
                  f"ratio={x.get('ratio')} end_degree={x.get('end_degree')} used={x.get('used')}")
    if had:
        print(f"hadronic showers (B): n={len(had)} written={sum(1 for _,x in had if x.get('used')=='dqdx')}")
        for e, x in had:
            try:
                r = float(x["dqdx"]) / float(x["charge"]) if float(x["charge"]) > 0 else float("nan")
            except Exception:
                r = float("nan")
            print(f"  HAD evt {e} id={x.get('id')} pdg={x.get('pdg')} conn={x.get('conn')} nseg={x.get('nseg')} "
                  f"charge={x.get('charge')} dqdx={x.get('dqdx')} dqdx/charge={r:.2f} used={x.get('used')}")
    if tctx:
        rat = []
        for _, x in tctx:
            try:
                ke = float(x["ke_mev"]); won = float(x["won_mev"])
                if ke > 5: rat.append(won / ke)
            except Exception:
                pass
        print(f"track contexts (B): n={len(tctx)} won/ke (ke>5 MeV): median={statistics.median(rat) if rat else float('nan'):.2f} "
              f"q10={q(rat,.1):.2f} q90={q(rat,.9):.2f}")
    if mass_c:
        n938 = sum(1 for _, x in mass_c if int(x.get("n_2212_showers_graph", 0)) > 0)
        nlo = sum(int(x.get("n_leftover_nonEM", 0)) for _, x in mass_c)
        ng = sum(int(x.get("n_mainvtx_guard_skip", 0)) for _, x in mass_c)
        print(f"mass census (B): events with graph-reachable 2212 showers={n938} leftover nonEM showers={nlo} mainvtx guard skips={ng}")
        for e, x in mass_c:
            if int(x.get("n_2212_showers_graph", 0)) > 0 or int(x.get("n_mainvtx_guard_skip", 0)) > 0:
                print(f"  MASS evt {e} add_legacy={x.get('add_legacy')} add_rules={x.get('add_rules')} "
                      f"n2212={x.get('n_2212_showers_graph')} guard_skip={x.get('n_mainvtx_guard_skip')}")
    for e, x in guard:
        print(f"  GUARD evt {e} {x}")
    print(f"pi0 window rejects (B): {len(win_rej)}; incoming stamps (B): {len(stamp)}")

if __name__ == "__main__":
    main()
