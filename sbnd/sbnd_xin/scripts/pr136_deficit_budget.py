#!/usr/bin/env python3
"""doc pr/136 -- the SCOPE-BOUNDARY measurement: is the missing charge in the event at all?

The mass-closure metric (pr136_mass_closure.py) says 19 of 56 hand pi0 pairs
are kinematically impossible at the production energies, and that hand marks
rescue only 4.  For the other 15 the campaign has to answer a prior question
before proposing any clustering change:

    the pair needs dE = E_tot * (1/R - 1) more energy.  Does that charge EXIST
    in the reconstruction anywhere the owner's post-vertex scope can reach?

Three budgets are computed per event, all in MeV, all from the calib dump:

  ORPHAN   charge in segments held by NO shower (`shower_id` < 0).  Reachable:
           absorbing an orphan is exactly what the late passes do.
  OTHER    charge in OTHER showers of the event.  Reachable: this is pr/130's
           SPLIT + STOLEN pool -- the charge is reconstructed, just attributed
           to the wrong object.
  PAIR     what the two labelled gammas already hold.

UNITS, AND THE ERROR BAR ON THEM.  Segment charge is raw dQ; `kine_charge` is
MeV after recombination, the W value and the EM fudge.  Rather than re-deriving
that chain (which would import every constant the campaign has been flipping),
each event calibrates ITSELF: k = sum(kine_charge of the two gamma showers) /
sum(their segments' dQ), then every other dQ in that event is priced at k.

k is NOT a clean conversion: `kine_charge` is a 2D integration within 0.6 cm of
the shower's own cloud with plane weights and a possible max-plane drop, so it
is not a sum over member dQ and the direction of the error is not known a
priori.  The two gammas of one pair therefore give two INDEPENDENT estimates of
k; their spread (median 1.34x, worst 2.62x on this sample) is the honest error
bar, and every budget is reported as the range [q*k_lo, q*k_hi].

Verdicts follow from that range, not from a point estimate:
  EXCLUDED       the deficit exceeds even the OPTIMISTIC budget -> no
                 re-attribution of reconstructed charge can fix this pair,
                 whatever absorber is changed.  The number this script exists
                 to produce.
  REACHABLE      even the PESSIMISTIC budget covers it.
  INDETERMINATE  the spread straddles the answer -- report it as unknown
                 rather than resolving it with a constant we cannot defend.

Separately, the OTHER budget is an upper bound in a second way: much of it is
track-like charge that carries a different recombination factor and that an EM
absorber must never take.  Stated, not hidden.

READ-ONLY.

  scripts/pr136_deficit_budget.py --closure docs/pr/pr136-mass-closure.tsv \
      --manifest98 em117-134f08698-manifest.tsv \
      --manifest141 em114c-134f086141-manifest.tsv [--tsv out.tsv]
"""
import argparse
import csv
import importlib.util
import os

SD = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(SD)
SEL = None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seg_dq(seg):
    return sum(p.get("dQ", 0.0) for p in (seg.get("points") or []) if p.get("dQ", 0.0) > 0)


def main():
    global SEL
    SEL = _load("pr126_pi0_select", os.path.join(SD, "pr126_pi0_select.py"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--closure", default=os.path.join(SX, "docs", "pr", "pr136-mass-closure.tsv"))
    ap.add_argument("--manifest98"); ap.add_argument("--manifest141")
    ap.add_argument("--overlay-tag", default="pi0scan-0829-agent")
    ap.add_argument("--tsv")
    a = ap.parse_args()

    if a.manifest98 or a.manifest141:
        sets = []
        for t in SEL.SETS:
            t = list(t)
            if t[0] == "98" and a.manifest98: t[4] = a.manifest98
            if t[0] == "141" and a.manifest141: t[4] = a.manifest141
            sets.append(tuple(t))
        SEL.SETS = sets
    overlay = SEL.load_labels(a.overlay_tag) if a.overlay_tag else {}

    cl = {r["event"]: r for r in csv.DictReader(open(a.closure), delimiter="\t")}

    out = []
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            c = cl.get(str(ev))
            if not c or float(c["R_prod"]) >= 1.0:
                continue                       # only the impossible pairs
            dump = SEL.load_json(mrow["dump"])
            if not dump:
                continue
            # take the SAME label source the closure row used: an event can
            # carry a base record with no pio block AND an overlay pairing,
            # and "base or overlay" would silently pick the empty one.
            rec = (labels.get(ev) if c["labelsrc"] == "base" else overlay.get(ev)) or {}
            g = (rec.get("pio") or {}).get("gammas") or {}
            if not all(k in g for k in ("1", "2")):
                continue
            ids = {int(g["1"].get("shower") or -1), int(g["2"].get("shower") or -1)}
            by = {int(s["id"]): s for s in (dump.get("showers") or [])}
            if not ids <= set(by):
                continue

            per = {i: 0.0 for i in ids}
            q_other = q_orph = 0.0
            for seg in (dump.get("segments") or []):
                q = seg_dq(seg)
                if q <= 0:
                    continue
                sid = int(seg.get("shower_id", -1))
                if sid in per:
                    per[sid] += q
                elif sid < 0:
                    q_orph += q
                else:
                    q_other += q
            q_pair = sum(per.values())
            e_pair = sum((by[i].get("kine_charge") or 0.0) for i in ids)
            if q_pair <= 0 or e_pair <= 0:
                continue
            k = e_pair / q_pair                       # MeV per dQ, calibrated in-event

            # HOW WELL DOES k TRAVEL?  kine_charge is a 2D integration within
            # 0.6 cm of the shower's own cloud, NOT a sum over member dQ, so k
            # is a ratio of two differently-computed quantities.  The two
            # gammas of one pair give two independent estimates; their spread
            # is the honest error bar on every budget below, and a verdict
            # that flips between k_lo and k_hi is INDETERMINATE, not excluded.
            ks = [((by[i].get("kine_charge") or 0.0) / per[i]) for i in ids if per[i] > 0]
            k_lo, k_hi = (min(ks), max(ks)) if len(ks) == 2 else (k, k)

            R = float(c["R_prod"])
            e_tot = float(c["e1_prod"]) + float(c["e2_prod"])
            need = e_tot * (1.0 / R - 1.0)
            q_avail = q_orph + q_other
            orph, other = q_orph * k, q_other * k
            budget = orph + other
            b_lo, b_hi = q_avail * k_lo, q_avail * k_hi
            rescued = int(float(c["R_marks"]) >= 1.0)
            if rescued:
                verdict = "RESCUED"
            elif b_hi < need:
                verdict = "EXCLUDED"          # not even the optimistic k reaches it
            elif b_lo >= need:
                verdict = "REACHABLE"         # even the pessimistic k reaches it
            else:
                verdict = "INDETERMINATE"     # the k spread straddles the answer
            out.append(dict(event=ev, sample=mrow.get("sample", tag), setname=setname,
                            R_prod=round(R, 3), m_prod=c["m_prod"],
                            e_pair=round(e_pair, 1), need_mev=round(need, 1),
                            orphan_mev=round(orph, 1), other_shower_mev=round(other, 1),
                            budget_mev=round(budget, 1),
                            budget_lo=round(b_lo, 1), budget_hi=round(b_hi, 1),
                            k_ratio=round(k_hi / k_lo, 2) if k_lo > 0 else -1,
                            need_over_budget=round(need / budget, 3) if budget > 0 else -1,
                            verdict=verdict, rescued_by_marks=rescued))

    out.sort(key=lambda r: r["R_prod"])
    print("SCOPE BOUNDARY -- can the impossible pairs be fixed by RE-ATTRIBUTING reconstructed charge?")
    print("  budget = ORPHAN (segments in no shower) + OTHER (segments in other showers of the event)")
    print("  priced with an in-event dQ->MeV constant from the two labelled gamma showers (upper bound)\n")
    print("  %-8s %-8s %6s %8s %9s %9s %10s %7s %6s %s"
          % ("event", "sample", "R", "E_pair", "need", "budget", "bud_range", "nd/bud", "k_hi/lo", "verdict"))
    for r in out:
        print("  %-8s %-8s %6.2f %8.1f %9.1f %9.1f %4.0f-%-5.0f %7.2f %6.2f %s"
              % (r["event"], r["sample"], r["R_prod"], r["e_pair"], r["need_mev"],
                 r["budget_mev"], r["budget_lo"], r["budget_hi"],
                 r["need_over_budget"], r["k_ratio"], r["verdict"]))
    nm = [r for r in out if not r["rescued_by_marks"]]
    kr = sorted(r["k_ratio"] for r in out if r["k_ratio"] > 0)
    print("\n  k SPREAD (the two gammas' independent MeV-per-dQ estimates):"
          "  median %.2fx, worst %.2fx" % (kr[len(kr) // 2], kr[-1]))
    print("  of the %d impossible pairs the hand marks do NOT rescue:" % len(nm))
    for v in ("REACHABLE", "INDETERMINATE", "EXCLUDED"):
        rs = [r for r in nm if r["verdict"] == v]
        print("     %-14s %2d   %s" % (v, len(rs), ", ".join(str(r["event"]) for r in rs)))
    hard = [r for r in nm if r["verdict"] == "EXCLUDED"]
    if hard:
        print("\n  EXCLUDED means the deficit exceeds the OPTIMISTIC budget: %s"
              % ", ".join("%s needs %.1fx" % (r["event"], r["need_over_budget"]) for r in hard))
    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print("\nwrote %s (%d rows)" % (p, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
