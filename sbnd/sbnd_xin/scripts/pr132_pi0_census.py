#!/usr/bin/env python3
"""pi0 identification census, doc pr/132 (fork of pr126_pi0_census.py).

Forked rather than edited (house convention: pr126's script keeps producing
pr126's numbers).  What the fork adds, all needed by the pr/132 round:

  * --fudge / --offset: the hand labels store SCAN-TIME masses (EM charge
    scale = fudge 0.80).  On an arm running at a different
    kine_shower_fudge_factor F every reco energy -- and therefore every mass
    the finder tests -- scales by 0.80/F, and a pi0_mass_offset knob moves the
    windows.  The section-B blocker check must use the arm's own geometry:
      hand mass is scaled by 0.80/F, and the with-vertex window is
      (135-offset-25, 135-offset+35).
  * --overlay-tag: a second label dir (the pr/132 pairing pass,
    em_labels/pi0scan-*) whose pio blocks extend the denominator for events
    the base scans left unpaired.  Base labels always win on conflict.
  * section E: nueCC-fake counter -- accepted groups where one member is
    conn_type==1 AT THE MAIN VERTEX and the partner is tiny (the topology the
    owner wants suppressed; measures knob K3's target and its effect).
  * section F: rescan coverage -- how many of the 109 scanned-but-unpaired
    candidates (docs/pr/pr126-pi0-rescan.tsv) carry an accepted group.

READ-ONLY (CLAUDE.md M13).  Changes nothing.

    ./pr132_pi0_census.py --manifest141 <tsv> --manifest98 <tsv> \
        [--fudge 0.84] [--offset 10] [--overlay-tag pi0scan-0829-agent] \
        [--fake-mev 30] [--tsv out.tsv]
"""
import argparse, csv, os, re, sys, importlib.util
from collections import Counter, defaultdict

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))
_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(SX, "scripts", "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)


def windows(offset):
    """The finders' acceptance windows for a given pi0_mass_offset (MeV).

    id_pi0_with_vertex     -25 < m-135+offset < 35
    id_pi0_without_vertex  |m-135+offset| < 60
    """
    return ((135.0 - offset - 25.0, 135.0 - offset + 35.0),
            (135.0 - offset - 60.0, 135.0 - offset + 60.0))


def classify(rec, dump, scale, win1):
    """Why the reconstruction did not reproduce the scanner's pair.

    `scale` maps the label's scan-time mass onto the arm's energy scale
    (0.80/fudge); the window is the arm's own (offset knob).
    """
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
    if m:
        ms = m * scale
        if not (win1[0] <= ms <= win1[1]):
            reasons.append("mass %.0f (scaled %.0f) outside with-vertex window (%.0f,%.0f)"
                           % (m, ms, win1[0], win1[1]))
    return reasons


def main_vertex_id(dump):
    for v in (dump.get("vertices") or ()):
        if v.get("is_main"):
            return int(v["id"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    ap.add_argument("--manifest141", help="override the 141-set manifest")
    ap.add_argument("--manifest98", help="override the 98-set manifest")
    ap.add_argument("--fudge", type=float, default=0.80,
                    help="the arm's kine_shower_fudge_factor (default 0.80 = uBooNE legacy)")
    ap.add_argument("--offset", type=float, default=10.0,
                    help="the arm's pi0_mass_offset in MeV (default 10 = legacy)")
    ap.add_argument("--overlay-tag",
                    help="extra label dir (em_labels/<tag>) whose pio blocks extend the denominator")
    ap.add_argument("--fake-mev", type=float, default=30.0,
                    help="section E partner-energy bound (default 30 MeV)")
    a = ap.parse_args()

    scale = 0.80 / a.fudge
    win1, win2 = windows(a.offset)

    if a.manifest98 or a.manifest141:
        newsets = []
        for t in SEL.SETS:
            t = list(t)
            if t[0] == "98" and a.manifest98:
                t[4] = a.manifest98
            if t[0] == "141" and a.manifest141:
                t[4] = a.manifest141
            newsets.append(tuple(t))
        SEL.SETS = newsets

    overlay = {}
    if a.overlay_tag:
        overlay = SEL.load_labels(a.overlay_tag)

    def hand_pair(rec):
        if not rec:
            return None
        g = (rec.get("pio") or {}).get("gammas")
        if not g or not all(x in g and (g[x].get("energy") or 0) > 0 for x in ("1", "2")):
            return None
        return rec

    rows = []
    match = Counter()
    reasons = Counter()
    tot = Counter()
    kine_vs_group = Counter()
    fake = Counter()
    fake_rows = []
    groups_by_event = {}
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
            groups_by_event[(setname, ev)] = len(groups)
            tot["events_with_group"] += 1 if groups else 0
            tot["groups"] += len(groups)
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

            # ---- section E: nueCC-fake topology among accepted groups -----
            mvid = main_vertex_id(dump)
            for pid, shs in groups.items():
                if len(shs) != 2 or mvid is None:
                    continue
                for A, B in ((shs[0], shs[1]), (shs[1], shs[0])):
                    if (int(A.get("start_connection_type") or 0) == 1
                            and int(A.get("start_vertex_id") or -2) == mvid
                            and (B.get("kine_charge") or 0) < a.fake_mev):
                        fake["groups"] += 1
                        fake_rows.append((setname, mrow["sample"], ev,
                                          int(A["id"]), A.get("kine_charge"),
                                          int(B["id"]), B.get("kine_charge")))
                        break

            # ---- hand-pi0 events (base labels first, overlay extends) -----
            rec = hand_pair(labels.get(ev))
            src = "base"
            if rec is None and overlay:
                rec = hand_pair(overlay.get(ev))
                src = "overlay"
            if rec is None:
                continue
            g = rec["pio"]["gammas"]
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
            why = [] if cls == "exact" else classify(rec, dump, scale, win1)
            for w in why:
                reasons[w.split(":")[-1] if ":" in w else w] += 1
            rows.append(dict(setname=setname, event=ev, sample=mrow["sample"],
                             origin=rec.get("origin"), labelsrc=src, match=cls,
                             mass_vertex=rec["pio"].get("mass_vertex_convention"),
                             mass_axis=rec["pio"].get("mass_axis_convention"),
                             why="; ".join(why)))

    n = sum(match.values())
    print("=== A. pi0 pairing vs the hand scan (%d hand pi0; fudge=%.2f offset=%.0f) ==="
          % (n, a.fudge, a.offset))
    for kk in ("exact", "partial", "none", "no-group"):
        print("  %-9s %3d   %5.1f %%" % (kk, match[kk], 100.0 * match[kk] / n))
    print("  -> exact %.0f %%, sharing a gamma %.0f %%."
          % (100.0 * match["exact"] / n,
             100.0 * (match["exact"] + match["partial"]) / n))

    print("\n=== B. what stopped the other %d (windows scaled to the arm) ===" % (n - match["exact"]))
    for kk, v in reasons.most_common():
        print("  %-64s %d" % (kk, v))

    print("\n=== C. pio_id and kine_pio_* over all %d events ===" % tot["events"])
    print("  events with >=1 accepted pi0 group : %d  (%.0f %%)"
          % (tot["events_with_group"], 100.0 * tot["events_with_group"] / tot["events"]))
    print("  accepted groups                    : %d" % tot["groups"])
    print("  kine_pio_flag = 1 / 2 / 0          : %d / %d / %d"
          % (tot["kine_flag_1"], tot["kine_flag_2"], tot["kine_flag_0"]))
    for kk, v in kine_vs_group.most_common():
        print("  kine_pio_*  %-42s %d" % (kk, v))

    # ---- D. audit counters from the logs ---------------------------------
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
    print("\n=== D. which finder accepted each group (audit counters, %d events) ===" % nlog)
    print("  id_pi0_WITH_vertex    accepted : %d" % aw)
    print("  id_pi0_WITHOUT_vertex accepted : %d" % awo)

    print("\n=== E. nueCC-fake topology among accepted groups (partner < %.0f MeV) ===" % a.fake_mev)
    print("  attached-at-main + tiny partner groups : %d" % fake["groups"])
    for r in fake_rows:
        print("    %s-set %s evt %s: attached sh %d (%.1f MeV) + partner sh %d (%.1f MeV)"
              % (r[0], r[1], r[2], r[3], r[4] or 0, r[5], r[6] or 0))

    # ---- F. rescan coverage ----------------------------------------------
    rescan = os.path.join(SX, "docs", "pr", "pr126-pi0-rescan.tsv")
    cov = tot_rescan = 0
    if os.path.exists(rescan):
        with open(rescan) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                tot_rescan += 1
                if groups_by_event.get((r["setname"], int(r["event"])), 0) > 0:
                    cov += 1
    print("\n=== F. rescan coverage: accepted group on the 109 unpaired candidates ===")
    print("  %d of %d rescan events carry >=1 accepted pi0 group" % (cov, tot_rescan))

    if a.tsv:
        p = a.tsv if os.path.isabs(a.tsv) else os.path.join(SX, a.tsv)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t",
                               fieldnames=["setname", "sample", "event", "origin", "labelsrc",
                                           "match", "mass_vertex", "mass_axis", "why"])
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["match"], r["event"])):
                w.writerow(r)
        print("\nwrote %s (%d rows)" % (p, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
