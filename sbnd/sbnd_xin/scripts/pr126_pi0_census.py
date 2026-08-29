#!/usr/bin/env python3
"""pi0 identification census against the hand scan (doc pr/126 sec 4f + sec 5).

Three questions, one pass over the current arms:

  A. How often does the reconstruction find the pi0 the scanner found?
  B. When it does not, WHICH gate stopped it?
  C. What do `pio_id` (the accepted pairing) and `kine_pio_*` (a separate
     highest-total-energy scan) actually do over all 239 events?

READ-ONLY (CLAUDE.md M13).  Changes nothing.

    ./pr126_pi0_census.py
    ./pr126_pi0_census.py --tsv docs/pr/pr126-pi0-census.tsv
"""
import argparse, csv, os, re, sys, importlib.util
from collections import Counter, defaultdict

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))
_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(SX, "scripts", "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

# The finder's own acceptance windows, from the source:
#   id_pi0_with_vertex     NeutrinoShowerClustering.cxx:5190  -25 < m-135+10 < 35  -> (100,160)
#   id_pi0_without_vertex  NeutrinoShowerClustering.cxx:5660  |m-135+10| < 60      -> ( 65,185)
WIN1 = (100.0, 160.0)
WIN2 = (65.0, 185.0)


def classify(ev, rec, dump):
    """Why the reconstruction did not reproduce the scanner's pair."""
    g = rec["pio"]["gammas"]
    by = {int(s["id"]): s for s in (dump.get("showers") or ())}
    s1, s2 = by.get(int(g["1"]["shower"])), by.get(int(g["2"]["shower"]))
    reasons = []
    for name, s in (("g1", s1), ("g2", s2)):
        if s is None:
            reasons.append(name + ":absent-on-arm")
            continue
        pdg = abs(int(s.get("particle_id") or 0))
        if pdg != 11:
            reasons.append("%s:pdg=%s" % (name, s.get("particle_id")))
        if (s.get("total_length") or 0) < 3.0:
            reasons.append("%s:len<3cm" % name)
    m = rec["pio"].get("mass_vertex_convention")
    if m and not (WIN1[0] <= m <= WIN1[1]):
        reasons.append("mass %.0f outside with-vertex window (100,160)" % m)
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    a = ap.parse_args()

    rows = []
    match = Counter()
    reasons = Counter()
    # ---- global census over every event of both manifests -----------------
    tot = Counter()
    kine_vs_group = Counter()
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        labels = SEL.load_labels(tag)
        man = SEL.load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            dump = SEL.load_json(mrow["dump"])
            if not dump:
                continue
            tot["events"] += 1
            groups = defaultdict(list)
            for s in (dump.get("showers") or ()):
                pid = int(s.get("pio_id", -1))
                if pid >= 0:
                    groups[pid].append(s)
            tot["events_with_group"] += 1 if groups else 0
            tot["groups"] += len(groups)
            for shs in groups.values():
                tot["group_size_%d" % len(shs)] += 1
            k = dump.get("kine") or {}
            kf = k.get("kine_pio_flag")
            tot["kine_flag_%s" % (kf or 0)] += 1
            if kf and groups:
                e1, e2 = k.get("kine_pio_energy_1"), k.get("kine_pio_energy_2")
                want = sorted([round(e1 or 0, 2), round(e2 or 0, 2)])
                hit = any(sorted(round(s.get("kine_charge") or 0, 2) for s in shs) == want
                          for shs in groups.values())
                kine_vs_group["names an ACCEPTED pair" if hit
                              else "names a pair NO group accepted"] += 1
            elif kf:
                kine_vs_group["filled although NO pi0 was accepted"] += 1

            # ---- hand-pi0 events only ----
            rec = labels.get(ev)
            if not rec or not (rec.get("pio") or {}).get("gammas"):
                continue
            g = rec["pio"]["gammas"]
            if not all(x in g and (g[x].get("energy") or 0) > 0 for x in ("1", "2")):
                continue
            hand = {int(g["1"]["shower"]), int(g["2"]["shower"])}
            cls = "no-group"
            if groups:
                cls = "none"
                for shs in groups.values():
                    ids = {int(s["id"]) for s in shs}
                    if ids == hand:
                        cls = "exact"
                        break
                    if ids & hand:
                        cls = "partial"
            match[cls] += 1
            why = [] if cls == "exact" else classify(ev, rec, dump)
            for w in why:
                reasons[w.split(":")[-1] if ":" in w else w] += 1
            rows.append(dict(setname=setname, event=ev, sample=mrow["sample"],
                             origin=rec.get("origin"), match=cls,
                             mass_vertex=rec["pio"].get("mass_vertex_convention"),
                             mass_axis=rec["pio"].get("mass_axis_convention"),
                             why="; ".join(why)))

    n = sum(match.values())
    print("=== A. pi0 pairing vs the hand scan (current arms, %d hand pi0) ===" % n)
    for k in ("exact", "partial", "none", "no-group"):
        print("  %-9s %3d   %5.1f %%" % (k, match[k], 100.0 * match[k] / n))
    print("  -> the reconstruction reproduces the scanner's pair EXACTLY on %.0f %%,"
          % (100.0 * match["exact"] / n))
    print("     and shares at least one gamma on %.0f %%."
          % (100.0 * (match["exact"] + match["partial"]) / n))

    print("\n=== B. what stopped the other %d ===" % (n - match["exact"]))
    for k, v in reasons.most_common():
        print("  %-52s %d" % (k, v))

    print("\n=== C. pio_id and kine_pio_* over all %d events ===" % tot["events"])
    print("  events with >=1 accepted pi0 group : %d  (%.0f %%)"
          % (tot["events_with_group"], 100.0 * tot["events_with_group"] / tot["events"]))
    print("  accepted groups                    : %d  (all of size %d)"
          % (tot["groups"], 2 if tot.get("group_size_2") == tot["groups"] else -1))
    print("  kine_pio_flag = 1 / 2 / 0          : %d / %d / %d"
          % (tot["kine_flag_1"], tot["kine_flag_2"], tot["kine_flag_0"]))
    print("     (kine_pio_flag is the BDT-FEATURE selection, not the accepted")
    print("      pairing -- which finder accepted a group is section D below)")
    for k, v in kine_vs_group.most_common():
        print("  kine_pio_*  %-42s %d" % (k, v))

    # ---- D. which FINDER accepted each group: the audit counters ---------
    # NeutrinoShowerClustering increments g_pr33_audit.f3_pi0_with_vertex /
    # f3_pi0_without_vertex at each acceptance and TaggerCheckNeutrino prints
    # them as `pi0=with:N,without:M`.  That is the only place the accepted
    # group's TYPE is visible: PrDisplayDump writes map_pio_id_mass[..].first
    # (the mass) and drops .second (1 = with vertex, 2 = without).
    pat = re.compile(r"pi0=with:(\d+),without:(\d+)")
    aw = awo = nlog = 0
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SEL.SETS:
        for ev, mrow in SEL.load_manifest(m_cur).items():
            d = os.path.dirname(os.path.join(SX, mrow["dump"]))
            if not os.path.isdir(d):
                continue
            got = None
            for fn in os.listdir(d):
                if not fn.endswith(".log"):
                    continue
                for line in open(os.path.join(d, fn), errors="replace"):
                    m = pat.search(line)
                    if m:
                        got = (int(m.group(1)), int(m.group(2)))
            if got:
                nlog += 1
                aw += got[0]
                awo += got[1]
    print("\n=== D. which finder accepted each group (audit counters, %d events) ==="
          % nlog)
    print("  id_pi0_WITH_vertex    accepted : %d" % aw)
    print("  id_pi0_WITHOUT_vertex accepted : %d" % awo)
    if awo == 0:
        print("  -> the without-vertex path is DORMANT on SBND: it accepts nothing.")

    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t",
                               fieldnames=["setname", "sample", "event", "origin",
                                           "match", "mass_vertex", "mass_axis", "why"])
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["match"], r["event"])):
                w.writerow(r)
        print("\nwrote %s (%d rows)" % (p, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
