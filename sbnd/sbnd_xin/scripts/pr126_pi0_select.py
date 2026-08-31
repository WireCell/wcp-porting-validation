#!/usr/bin/env python3
"""Select and prepare the hand-scanned pi0 sample (doc pr/126 sec 3).

The two hand scans -- 98 events (`emscan-0827`, owner) and 141 events
(`emscan-0828-agent5`) -- carry a `pio` block whenever the scanner stored a
gamma pairing.  That block is the only pi0 ground truth we have: it names the
two gamma showers, their marks- and orphan-corrected energies, the decay vertex
and how it was obtained, and the two mass conventions.  This script turns it
into a table and a re-runnable manifest.

    ./pr126_pi0_select.py --selftest
    ./pr126_pi0_select.py --tsv docs/pr/pr126-pi0-events.tsv \
                          --manifest em_display/pr126-pi0-manifest.tsv

READ-ONLY over `em_labels/`, the manifests and the calib dumps (CLAUDE.md M13).

WHAT "hand pi0" MEANS HERE.  `label["pio"]["gammas"]["1"|"2"]` present with a
positive energy on both.  That is a pairing the scanner actively stored, not a
bucket keyword: `docs/pr/pr116-bulk/buckets-141.tsv` was written before the
scan finished and reports 18 where 24 are live on disk.  Counts from the
labels, never from the bucket TSV.

THE FOUR pi0 NUMBERS, kept apart (pr/114 sec 6.2).
  hand pair          -- `pio.gammas`, the scanner's choice; NOT mass-windowed
  reco groups        -- showers sharing `pio_id >= 0` in the dump; the winner
                        loop's accepted pairs, mass-windowed
  `kine_pio_*`       -- a SEPARATE highest-total-energy scan over all candidate
                        pairs; can name a pair no reconstruction accepted
  bucket `pi0` col   -- pr/115's inferred classification, advisory only
"""
import argparse, csv, json, os, sys
from collections import defaultdict

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))

LABEL_ROOT = os.path.join(SX, "em_labels")

# (set name, scan tag, scan-time manifest, scan-time prep, current manifest, current prep)
SETS = [
    ("98",  "emscan-0827",
     "em114-manifest.tsv",  "emprep",
     "em117-124onA98-manifest.tsv",  "emprep-124onA98",
     "pr115-handscan-buckets.tsv"),
    ("141", "emscan-0828-agent5",
     "em114c-manifest.tsv", "emprep-c",
     "em114c-124onA141-manifest.tsv", "emprep-124onA141",
     os.path.join("pr116-bulk", "buckets-141.tsv")),
]

# The arms above are the PROBE-ARMED knob-on arms, content-equivalent to the
# post-flip production baseline (work-pr124r1-flipA*) on every event.  pr/120
# and pr/124 both measured that scoring against probes-off sidecars makes
# cross-run numbers wobble, so the probe-armed arm is the one to read.


# ---------------------------------------------------------------- loading
def load_labels(tag):
    out = {}
    d = os.path.join(LABEL_ROOT, tag)
    for fn in sorted(os.listdir(d)):
        if fn.startswith("labels-evt") and fn.endswith(".json"):
            with open(os.path.join(d, fn)) as fh:
                rec = json.load(fh)
            out[int(rec["eventNo"])] = rec
    return out


def load_manifest(name):
    out = {}
    with open(os.path.join(SX, "em_display", name)) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[int(r["event"])] = r
    # 2026-08-31 (doc pr/135 sec 11.2): a manifest whose dump ARM was released
    # by a retire round must fail LOUDLY here.  load_json() below returns None
    # for a missing dump and every caller `continue`s on None, so a released
    # arm would otherwise turn into a silent zero-row census -- a wrong number
    # that looks like a clean run.  Checked once per manifest, on the arm
    # directory, not per row.
    arms = {(r.get("dump") or "").split("/")[0]
            for r in out.values() if (r.get("dump") or "").count("/")}
    gone = sorted(a for a in arms if a and not os.path.isdir(os.path.join(SX, a)))
    if gone:
        raise FileNotFoundError(
            "manifest %s points at released arm(s) %s -- its dumps were retired "
            "(see scripts/retire/state-*/removed.tsv and the record layer under "
            "archive/records/).  Re-point the manifest at a surviving arm or use "
            "the current-production manifest instead." % (name, gone))
    return out


def load_buckets(name):
    p = os.path.join(SX, "docs", "pr", name)
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[int(r["event"])] = r
    return out


def load_json(path):
    if not os.path.isabs(path):
        path = os.path.join(SX, path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def load_prep(prep_dir, ev):
    return load_json(os.path.join(SX, "em_display", prep_dir, "emprep-evt%d.json" % ev))


# ---------------------------------------------------------------- digest
def seg_charge(seg):
    return sum(p["dQ"] for p in (seg.get("points") or ()) if (p.get("dQ") or 0) > 0)


def digest(dump, prep):
    """-> (members: shower id -> set(seg id), seginfo, pio_groups, kine).

    Membership comes from the probe sidecar when present -- the dump's
    `segments[].shower_id` is single-valued, so a segment held by two showers
    is credited to one and the join invents misses (em115_score.py docstring).
    """
    seginfo, actual = {}, defaultdict(set)
    for s in dump.get("segments") or ():
        sid = int(s["id"])
        seginfo[sid] = {"charge": seg_charge(s), "length": float(s.get("length") or 0.0),
                        "pdg": s.get("particle_id")}
        own = int(s.get("shower_id", -1))
        if own >= 0:
            actual[own].add(sid)
    if prep:
        actual = defaultdict(set)
        for node, e in (prep.get("showers") or {}).items():
            for m in e.get("members") or ():
                sid = int(m["seg"])
                actual[int(node)].add(sid)
                info = seginfo.setdefault(sid, {"charge": 0.0, "length": 0.0, "pdg": m.get("pdg")})
                if m.get("dQ"):
                    info["charge"] = float(m["dQ"])
    groups = defaultdict(list)
    showers = {}
    for s in dump.get("showers") or ():
        showers[int(s["id"])] = s
        pid = int(s.get("pio_id", -1))
        if pid >= 0:
            groups[pid].append(s)
    return actual, seginfo, dict(groups), (dump.get("kine") or {}), showers


def completeness(rec, shw, actual, seginfo):
    """Charge-weighted completeness / purity of ONE scanned shower.

    Mirrors em115_score.score_shower.  An UNMARKED shower has no row there --
    which reads as `the scanner found nothing to correct`, i.e. the
    reconstruction's membership IS the target.  We return 1.0/1.0 with
    marked=False so the caller can tell the two cases apart instead of
    dropping the event.
    """
    md = (rec.get("em") or {}).get("marks_detail") or {}
    det = md.get(str(shw)) or md.get(shw)
    have = actual.get(shw, set())
    q = lambda s: sum(seginfo.get(i, {}).get("charge", 0.0) for i in s)
    if not det or not (det.get("marked") or {}):
        return dict(marked=False, q_comp=1.0, q_pur=1.0, n_target=len(have), n_actual=len(have))
    marked = det["marked"]
    members = set(int(x) for x in (det.get("members") or ()))
    ins = set(int(s) for s, m in marked.items() if m.get("kind") == "in")
    outs = set(int(s) for s, m in marked.items() if m.get("kind") == "out")
    target = (members | ins) - outs
    inter = have & target
    qt, qh, qi = q(target), q(have), q(inter)
    return dict(marked=True,
                q_comp=(qi / qt) if qt > 0 else float("nan"),
                q_pur=(qi / qh) if qh > 0 else float("nan"),
                n_target=len(target), n_actual=len(have))


# ---------------------------------------------------------------- rows
FIELDS = ["setname", "event", "sample", "origin", "run", "subrun",
          "g1_shower", "g2_shower", "g1_E", "g2_E",
          "g1_E_noMarks", "g2_E_noMarks", "g1_marks_delta", "g2_marks_delta",
          "g1_orphan_delta", "g2_orphan_delta",
          "vertex_how", "mass_axis", "mass_vertex", "theta_axis", "theta_vertex",
          "backproject_geom",
          "n_reco_groups", "reco_pair_same", "reco_pio_mass", "reco_pio_type",
          "kine_pio_flag", "kine_pio_mass", "kine_pio_E1", "kine_pio_E2",
          "g1_marked", "g1_qcomp", "g1_qpur", "g2_marked", "g2_qcomp", "g2_qpur",
          "em_verdict", "confidence", "bucket", "bucket_pi0", "scan_note", "note"]


def build_rows(current=True):
    rows = []
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SETS:
        labels = load_labels(tag)
        man_scan = load_manifest(m_scan)
        man_cur = load_manifest(m_cur)
        buckets = load_buckets(buck)
        man, prep_dir = (man_cur, p_cur) if current else (man_scan, p_scan)
        for ev in sorted(labels):
            rec = labels[ev]
            pio = rec.get("pio")
            if not pio or not (pio.get("gammas") or {}):
                continue
            g = pio["gammas"]
            if not all(k in g and (g[k].get("energy") or 0) > 0 for k in ("1", "2")):
                continue
            mrow = man.get(ev)
            dump = load_json(mrow["dump"]) if mrow else None
            row = {f: "" for f in FIELDS}
            row.update(setname=setname, event=ev,
                       sample=rec.get("sample"), origin=rec.get("origin"),
                       run=rec.get("runNo"), subrun=rec.get("subRunNo"),
                       vertex_how=pio.get("vertex_how"),
                       backproject_geom=pio.get("backproject_geometry"),
                       mass_axis=pio.get("mass_axis_convention"),
                       mass_vertex=pio.get("mass_vertex_convention"),
                       theta_axis=pio.get("theta_axis_convention"),
                       theta_vertex=pio.get("theta_vertex_convention"),
                       em_verdict=(rec.get("em") or {}).get("verdict"),
                       confidence=rec.get("confidence"),
                       note=(rec.get("note") or "").replace("\t", " ").replace("\n", " ")[:160])
            b = buckets.get(ev) or {}
            row["bucket"] = b.get("bucket", "")
            row["bucket_pi0"] = b.get("pi0", "")
            sm = man_scan.get(ev) or {}
            row["scan_note"] = (sm.get("scan_note") or "").replace("\t", " ")
            for i, k in ((1, "1"), (2, "2")):
                gg = g[k]
                row["g%d_shower" % i] = gg.get("shower")
                row["g%d_E" % i] = gg.get("energy")
                row["g%d_E_noMarks" % i] = gg.get("energy_without_marks")
                row["g%d_marks_delta" % i] = gg.get("energy_marks_delta")
                row["g%d_orphan_delta" % i] = gg.get("energy_orphan_delta")
            if dump:
                actual, seginfo, groups, kine, showers = digest(dump, load_prep(prep_dir, ev))
                row["n_reco_groups"] = len(groups)
                hand = {int(g["1"]["shower"]), int(g["2"]["shower"])}
                same = ""
                for pid, shs in groups.items():
                    ids = {int(s["id"]) for s in shs}
                    if ids == hand:
                        same = "exact"
                        row["reco_pio_mass"] = shs[0].get("pio_mass")
                        break
                    if ids & hand:
                        same = same or "partial"
                        row["reco_pio_mass"] = shs[0].get("pio_mass")
                row["reco_pair_same"] = same or ("none" if groups else "no-group")
                row["kine_pio_flag"] = kine.get("kine_pio_flag")
                row["kine_pio_mass"] = kine.get("kine_pio_mass")
                row["kine_pio_E1"] = kine.get("kine_pio_energy_1")
                row["kine_pio_E2"] = kine.get("kine_pio_energy_2")
                for i, k in ((1, "1"), (2, "2")):
                    c = completeness(rec, int(g[k]["shower"]), actual, seginfo)
                    row["g%d_marked" % i] = int(c["marked"])
                    row["g%d_qcomp" % i] = round(c["q_comp"], 4)
                    row["g%d_qpur" % i] = round(c["q_pur"], 4)
            rows.append(row)
    return rows


def write_tsv(rows, path):
    if not os.path.isabs(path):
        path = os.path.join(SX, path)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s  (%d rows)" % (path, len(rows)))


def write_manifest(rows, path):
    """A `sample run subrun event dump`-shaped manifest for the pi0 subset, so
    the 50 events can be re-run or re-scanned as their own arm."""
    if not os.path.isabs(path):
        path = os.path.join(SX, path)
    man = {}
    for (_, _, _, _, m_cur, _, _) in SETS:
        man.update(load_manifest(m_cur))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sample", "run", "subrun", "event", "dump"])
        for r in rows:
            m = man.get(r["event"])
            if not m:
                continue
            w.writerow([m["sample"], m["run"], m["subrun"], m["event"], m["dump"]])
    print("wrote %s" % path)


def rescan_candidates():
    """Scanned events that carry NO stored pi0 pairing but where one was
    physically possible: >= 2 EM showers over the code's own 15 MeV threshold
    and over the 3 cm length cut.

    This is the answer to "why is the calibration sample only n=19": the two
    scans were EM-shower-CLUSTERING scans, so a pi0 pairing was stored only
    when the scanner chose to pair one -- 50 of 238 events.  The rest is not
    absent data, it is unasked data.
    """
    paired, origin = set(), {}
    rows = []
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SETS:
        labels = load_labels(tag)
        for ev, rec in labels.items():
            origin[ev] = rec.get("origin")
            g = (rec.get("pio") or {}).get("gammas") or {}
            if all(k in g and (g[k].get("energy") or 0) > 0 for k in ("1", "2")):
                paired.add(ev)
    for (setname, tag, m_scan, p_scan, m_cur, p_cur, buck) in SETS:
        man = load_manifest(m_cur)
        for ev, mrow in sorted(man.items()):
            if ev not in origin or ev in paired:
                continue
            dump = load_json(mrow["dump"])
            if not dump:
                continue
            em = [s for s in (dump.get("showers") or ())
                  if abs(int(s.get("particle_id") or 0)) == 11
                  and (s.get("kine_charge") or 0) > 15.0
                  and (s.get("total_length") or 0) >= 3.0]
            if len(em) < 2:
                continue
            e = sorted((s.get("kine_charge") or 0) for s in em)
            rows.append(dict(setname=setname, sample=mrow["sample"], run=mrow["run"],
                             subrun=mrow["subrun"], event=ev, origin=origin.get(ev),
                             n_em=len(em), e_max=round(e[-1], 1), e_2nd=round(e[-2], 1),
                             dump=mrow["dump"]))
    rows.sort(key=lambda r: -r["e_2nd"])
    return rows


def selftest():
    ok = True
    per_tag = {}
    for (setname, tag, *_rest) in SETS:
        labels = load_labels(tag)
        n = sum(1 for r in labels.values()
                if (r.get("pio") or {}).get("gammas")
                and all(k in r["pio"]["gammas"] and (r["pio"]["gammas"][k].get("energy") or 0) > 0
                        for k in ("1", "2")))
        per_tag[setname] = n
    exp = {"98": 26, "141": 24}
    for k, v in exp.items():
        s = "OK " if per_tag.get(k) == v else "FAIL"
        ok &= per_tag.get(k) == v
        print("%s  %s-set hand pi0 = %s (expect %d)" % (s, k, per_tag.get(k), v))

    rows = build_rows(current=True)
    print("%s  total rows = %d (expect 50)" % ("OK " if len(rows) == 50 else "FAIL", len(rows)))
    ok &= len(rows) == 50

    # every row's hand masses must match the label file byte for byte
    bad = 0
    for (setname, tag, *_r) in SETS:
        labels = load_labels(tag)
        for r in rows:
            if r["setname"] != setname:
                continue
            p = labels[r["event"]]["pio"]
            for col, key in (("mass_axis", "mass_axis_convention"),
                             ("mass_vertex", "mass_vertex_convention")):
                if r[col] != p.get(key):
                    bad += 1
    print("%s  hand-mass round-trip mismatches = %d (expect 0)" % ("OK " if not bad else "FAIL", bad))
    ok &= bad == 0

    # the bucket TSV is a stale snapshot -- assert that, so a future reader does
    # not silently trust it (doc pr/126 sec 3).
    b141 = load_buckets(os.path.join("pr116-bulk", "buckets-141.tsv"))
    n_b = sum(1 for v in b141.values() if v.get("has_pio") == "True")
    print("INFO buckets-141.tsv has_pio=True is %d vs %d live labels -- stale, as documented"
          % (n_b, per_tag["141"]))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    ap.add_argument("--manifest")
    ap.add_argument("--scan-time", action="store_true",
                    help="read the prod0825 scan-time arms instead of the current onA arms")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rescan", metavar="OUT.tsv", nargs="?", const="-",
                    help="emit the scanned-but-unpaired events where a pi0 pairing "
                         "was possible (doc pr/126 sec 4i)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.rescan:
        rows = rescan_candidates()
        nc = sum(1 for r in rows if r["origin"] == "ncpi0")
        print("scanned-but-unpaired with >=2 EM showers >15 MeV: %d  (ncpi0: %d)" % (len(rows), nc))
        if a.rescan != "-":
            p = a.rescan if os.path.isabs(a.rescan) else os.path.join(SX, a.rescan)
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print("wrote %s" % p)
        else:
            for r in rows[:30]:
                print("  ", r["event"], r["origin"], "nEM=%d" % r["n_em"], r["e_max"], r["e_2nd"])
        return 0
    rows = build_rows(current=not a.scan_time)
    if a.tsv:
        write_tsv(rows, a.tsv)
    if a.manifest:
        write_manifest(rows, a.manifest)
    if not a.tsv and not a.manifest:
        print("\t".join(FIELDS))
        for r in rows:
            print("\t".join(str(r[f]) for f in FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
