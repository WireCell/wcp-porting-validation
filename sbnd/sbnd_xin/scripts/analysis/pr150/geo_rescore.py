#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 -- POSITION-ANCHORED re-scorer for the two SBND hand scans.

*** THIS IS A NEW DENOMINATOR.  Nothing printed here is the doc pr/135 pi0    ***
*** number or the doc pr/148 PID number, and it must never be quoted as one.  ***
*** Those two docs score through per-event RECONSTRUCTION IDS; this script    ***
*** scores through POSITION.  The two answer different questions and the      ***
*** counts are not interchangeable in either direction.                       ***

WHY IT EXISTS
-------------
Both committed scorers resolve a hand label through an id that a clustering
change renumbers:

  pi0 census (scripts/pr132_pi0_census.py)   label pio.gammas[k].shower
                                             -> dump showers[].id  (= the start
                                             segment's pf_node_id)
  PID scan   (scripts/pr148_score_scan.py)   verdict row shower_id
                                             -> dump showers[].shower_id
                                             (a compact per-event index)

Neither id is a property of the physical object.  Measured with
scripts/analysis/pr150/pi0_id_drift.py: between pr150s0 and pr150csp3bw the
median Jaccard of the showers[].id SET is 0.00, so 0 of 132 hand gammas and
0 of 72 PID verdict rows resolve on pr150csp3bw.  That is a scorer property,
not physics -- and the committed scorers report it in the same column as a
genuine reconstruction failure ("absent-on-arm").

This script matches a hand-labelled object to a reco object by WHERE IT IS.

WHAT THE LABEL ACTUALLY STORES (checked, not assumed; 132 gammas)
-----------------------------------------------------------------
  reco_start           132/132   the label arm's own showers[].start -- the
                                 em_display viewer's reco_start(node) returns
                                 G.pt(sh["start"]) verbatim
                                 (em_display_viewer.py:1295-1299)
  axis                 132/132   axis_source: probe 90, shower_init_dir 32,
                                 python@start_override 8, manual@override 2
  energy / shower      132/132   scan-time energy (fudge 0.80) and the label
                                 arm's showers[].id
  em_start_correction    9/132   the SCANNER's corrected start; 8 of the 9 move
                                 the start (max 67.5 cm from reco_start)
  start                132/132   = em_start_correction when set, else reco_start
                                 (shower_start(), em_display_viewer.py:1301-1327)

So the task's fallback "recover the start from the label's own arm dump when
em_start_correction is null" is MOOT and is not implemented: `reco_start` in
the label already IS that dump's shower start, and in any case NOT ONE label
arm survives on disk (work-{mcp1k,mcp2k,ncpi0,nuecc48}-prod0825 and
work-pr131-denom{98,141}-* are all released).  The default anchor is therefore
`reco_start`; `--anchor corrected` is offered as a sensitivity line over the 9.

CALIBRATED, NOT ASSUMED (see section 3 / 4 of every pi0 report)
----------------------------------------------------------------
The gammas whose label `shower` id still resolves on an arm are a truth set:
there the id join IS the right answer.  Measured over those 39 pairs on
d102mpr (`--sweep` reproduces the table):

  dist(label reco_start, arm showers[].start)  median 0.020 cm
                                               <=1 cm 35/39, <=5 cm 37/39 (95 %)
                                               max 11.89 cm
  angle(label axis, reco shower_init_dir)      median 12.8 deg
                                               <=30 deg 32/39 (82 %)
                                               <=90 deg 38/39 (97 %)

=> R default 5.0 cm: the 95th percentile of a KNOWN-CORRECT distance.  Position
   is an extremely sharp discriminator here (median 200 microns).
=> THETA default 90 deg, NOT the 30 deg the task proposed: 30 deg throws away
   18 % of known-correct pairs.  It has to, because 100 of the 132 label axes
   are NOT the reco's own `shower_init_dir` -- 90 are `probe` axes and 8 are
   scanner overrides (one is anti-parallel, 172 deg).  A 30 deg gate would
   depress every arm's match rate for a reason that has nothing to do with the
   arm, which is the exact artefact this script exists to remove.  Run with
   `--theta 30` to see that cost; the sweep prints it either way.
=> the reco axis is `shower_init_dir` (em_display/em_geom.py:195, itself a
   mirror of PRShower.cxx:1552-1640), which beats normalize(end-start) on the
   truth set at every threshold (32 vs 27 at 30 deg, 38 vs 36 at 90 deg).
   `--axis endstart` switches it.

MATCHING RULE
-------------
Per event, over the arm's showers[]:
  candidate  iff  |start_reco - anchor| <= R  and  angle(axis) <= THETA
  ranked by  (distance, angle, |ln(E_reco / E_hand_scaled)|, shower_id)
  assigned   greedily, ONE-TO-ONE: a reco shower serves at most one hand gamma.
The one-to-one rule stops one reco shower from satisfying both hand gammas and
so reading as "both gammas matched".  Be clear about how a MERGE actually shows
up, because the two are easy to confuse:
  * the "arm MERGED the pair" bucket needs BOTH hand starts within R of the SAME
    reco start.  That is rare -- a merged shower keeps ONE gamma's start -- and
    the bucket is 0 on every arm measured so far.  It is kept because it is the
    only correct place to put the case if it occurs.
  * the common signature is "exactly one gamma matched", with the OTHER gamma's
    NEAREST reco shower (at any distance) being the very shower the matched
    gamma took.  That is counted separately as `one_gamma_merge_beyond_R`
    (2 on pr150csp3bw).
  * ncpi0 evt142421 on d102mpr is the worked example: both hand gammas sit on
    one 190 cm / 1148 MeV shower, but g1's start is 8.63 cm away, so at R=5 it
    is not a candidate at all and the event lands in the second bucket, not the
    first.  Widening to R=12 does NOT move it there either -- g1 then matches a
    DIFFERENT, 1.5 MeV / 0.8 cm stub 11.4 cm away, which is a bad match and is
    why R stays at 5.
E_hand_scaled = label energy * 0.80 / F, F = the arm's own
kine_shower_fudge_factor read from .wct-cfg-evt*.json (never hardcoded).

Sort keys are fully explicit and every iteration is over a sorted sequence, so
two runs on the same inputs produce byte-identical output.

READ-ONLY on the repo.  Writes only under --out
(default /home/xqian/tmp/pr150/scorers2/): geo_<mode>_<arm>.txt and the
per-object TSV geo_<mode>_<arm>.tsv.  Nothing is written to em_labels/,
vertex_labels/ or docs/pr/ (CLAUDE.md M13).

Repro (exact CLI used for the committed numbers):

  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  ./scripts/analysis/pr150/geo_rescore.py --mode pi0 \
      --arm d102mpr --arm pr150s0 --arm pr150cs --arm pr150p3bw \
      --arm pr150csp3bw --arm pr150tfull --sweep
  ./scripts/analysis/pr150/geo_rescore.py --mode pid \
      --arm d102mpr --arm pr150s0 --arm pr150cs --arm pr150p3bw \
      --arm pr150csp3bw --arm pr150tfull
"""
import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # .../sbnd_xin
sys.path.insert(0, os.path.join(SX, "em_display"))
import em_geom as G                                              # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(SX, "scripts", "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

SAMPLES = ("nuecc48", "ncpi0", "mcp1k", "mcp2k")

# pi0: the two committed denominator manifests and their base label dirs.
# NOTE: read DIRECTLY, never through SEL.load_manifest() -- that helper aborts
# the whole run when a manifest's dump arm directory is missing (a guard for
# retired arms, doc pr/135 sec 11.2), and on a Stage-1 pr150 cell the mcp2k arm
# genuinely does not exist yet.  Here an absent dump is a COUNTED outcome.
PI0_SETS = [("98",  "em117-132denom98-manifest.tsv",  "emscan-0827"),
            ("141", "em114c-132denom141-manifest.tsv", "emscan-0828-agent5")]
PI0_OVERLAY = "pi0scan-0829-agent"

# pid: the three committed pr/148 scan sets.  scan3 is the RESCAN of the same
# 36 objects scan0+scan2 labelled (doc pr/148 sec 15), so the sets are never
# summed: pass 1 = scan0 + scan2, pass 2 = scan3.
PID_SCANS = [("scan0", "pr148-pidscan.KEY.tsv",  "pr148-pidscan-verdicts.tsv"),
             ("scan2", "pr148-pidscan2.KEY.tsv", "pr148-pidscan2-verdicts.tsv"),
             ("scan3", "pr148-pidscan3.KEY.tsv", "pr148-pidscan3-verdicts.tsv")]
PID_REF_ARM = "d102mpr"      # the arm that still resolves 72/72 by id
LEN_TOL = 0.02               # pr148 identity bar, reused verbatim

R_DEFAULT = 5.0
THETA_DEFAULT = 90.0


# ------------------------------------------------------------------ helpers
def rd_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader((l for l in fh if not l.startswith("#")),
                                   delimiter="\t"))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def dump_path(tag, sample, ev):
    return os.path.join(SX, "work-%s-%s" % (sample, tag),
                        "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)


def vdist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def vangle(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return None
    c = sum(a[i] * b[i] for i in range(3)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pct(vals, p):
    if not vals:
        return float("nan")
    v = sorted(vals)
    return v[min(len(v) - 1, int(p / 100.0 * len(v)))]


def stat_line(vals, unit=""):
    if not vals:
        return "n=0"
    return ("n=%d med %.3f%s p90 %.2f%s p95 %.2f%s max %.2f%s"
            % (len(vals), pct(vals, 50), unit, pct(vals, 90), unit,
               pct(vals, 95), unit, max(vals), unit))


MIN_RATE_N = 5               # below this a "precision" is a count, not a rate


def rate(good, tot):
    """A precision NEVER prints without its denominator, and a denominator
    below MIN_RATE_N prints with the refusal attached.  The pid tables live on
    denominators of 1-5 on a Stage-1 cell, where three decimals would invite
    exactly the wrong reading."""
    if not tot:
        return "n/a (0 re-typed)"
    t = "%d/%d = %.3f" % (good, tot, 1.0 * good / tot)
    return t if tot >= MIN_RATE_N else t + " [n<%d: a count, NOT a rate]" % MIN_RATE_N


def inventory(tag):
    """Per-sample calib-dump count on an arm, RIGHT NOW.

    Stage-1 pr150 cells are still being produced while this runs, so two runs
    minutes apart legitimately differ.  Stamping the inventory is what makes a
    number here re-checkable later (CLAUDE.md: report gates by label)."""
    out = []
    for s in SAMPLES:
        d = os.path.join(SX, "work-%s-%s" % (s, tag))
        if os.path.isdir(d):
            n = len(glob.glob(os.path.join(d, "pr_evt*", "calib-pr-evt*.json")))
            out.append("%s=%d" % (s, n))
        else:
            out.append("%s=-" % s)
    return " ".join(out)


def fudge_of(tag):
    """The arm's kine_shower_fudge_factor / pi0_mass_offset, from its OWN
    compiled per-event config -- the way pi0_score.sh reads it.  Absent key =>
    the C++ default (TaggerCheckNeutrino.h: 0.80 / 10)."""
    for s in SAMPLES:
        for f in sorted(glob.glob(os.path.join(
                SX, "work-%s-%s" % (s, tag), "pr_evt*", ".wct-cfg-evt*.json")))[:1]:
            got = []

            def walk(o):
                if isinstance(o, dict):
                    if o.get("type") == "TaggerCheckNeutrino":
                        got.append(o.get("data", {}))
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(json.load(open(f)))
            if got:
                return (float(got[0].get("kine_shower_fudge_factor") or 0.80),
                        float(got[0].get("pi0_mass_offset") or 10.0))
    return (0.80, 10.0)


class ArmEvent(object):
    """One arm's calib dump for one event, with the reco geometry precomputed."""

    def __init__(self, dump):
        self.dump = dump
        self.showers = sorted((dump.get("showers") or ()), key=lambda s: int(s["id"]))
        self.segments = dump.get("segments") or []
        self.vby = {int(v["id"]): v for v in (dump.get("vertices") or ())}
        self._axis = {}

    def start(self, sh):
        p = sh.get("start") or {}
        return (p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0))

    def axis(self, sh, how):
        k = (int(sh["id"]), how)
        if k not in self._axis:
            if how == "endstart":
                a, b = self.start(sh), sh.get("end") or {}
                v = (b.get("x", 0.0) - a[0], b.get("y", 0.0) - a[1],
                     b.get("z", 0.0) - a[2])
            else:
                v, _br = G.shower_init_dir(sh, self.segments, self.vby)
            self._axis[k] = tuple(v)
        return self._axis[k]


def match_objects(arm, wants, r, theta, how):
    """Greedy ONE-TO-ONE position match of `wants` onto `arm.showers`.

    wants: list of dicts {key, anchor(3), axis(3) or None, energy or None}.
    Returns {key: {shower, dist, angle, rank}} plus a per-key miss reason.
    """
    cands = []
    for w in wants:
        for sh in arm.showers:
            d = vdist(w["anchor"], arm.start(sh))
            if d > r:
                continue
            a = vangle(w["axis"], arm.axis(sh, how)) if w.get("axis") else 0.0
            if a is None:
                a = 180.0
            if a > theta:
                continue
            eh, er = w.get("energy"), sh.get("kine_charge")
            if eh and er and eh > 0 and er > 0:
                ed = abs(math.log(er / eh))
            else:
                ed = 99.0
            cands.append((round(d, 6), round(a, 4), round(ed, 6),
                          int(sh["id"]), w["key"], sh, d, a))
    cands.sort(key=lambda t: (t[0], t[1], t[2], t[3], str(t[4])))
    out, used_k, used_s = {}, set(), set()
    for t in cands:
        if t[4] in used_k or t[3] in used_s:
            continue
        used_k.add(t[4])
        used_s.add(t[3])
        out[t[4]] = {"shower": t[5], "dist": t[6], "angle": t[7]}
    miss = {}
    for w in wants:
        if w["key"] in out:
            continue
        near = min((vdist(w["anchor"], arm.start(sh)) for sh in arm.showers),
                   default=None)
        if near is None:
            miss[w["key"]] = ("no shower on the arm", None)
        elif near > r:
            miss[w["key"]] = ("no shower within R", near)
        elif any(t[4] == w["key"] for t in cands):
            miss[w["key"]] = ("lost the one-to-one assignment (MERGED)", near)
        else:
            miss[w["key"]] = ("within R but angle > THETA", near)
    return out, miss


# =============================================================== pi0 mode
def pi0_hand_gammas():
    """-> sorted list of hand-pi0 records over the two committed manifests.

    Base labels win, the pr/132 pairing overlay extends -- the precedence
    pr132_pi0_census.py uses.  The manifests ARE the denominator."""
    ov = SEL.load_labels(PI0_OVERLAY)
    recs = []
    for setname, man, ltag in PI0_SETS:
        base = SEL.load_labels(ltag)
        for r in rd_tsv(os.path.join(SX, "em_display", man)):
            ev, smp = int(r["event"]), r["sample"]
            rec, src = base.get(ev), "base"
            g = _hand_pair(rec)
            if g is None:
                rec, src = ov.get(ev), "overlay"
                g = _hand_pair(rec)
            if g is None:
                continue
            recs.append({"setname": setname, "sample": smp, "event": ev,
                         "src": src, "origin": rec.get("origin"),
                         "label_arm": (rec.get("arm") or "").split("/")[0],
                         "gammas": g, "pio": rec["pio"]})
    recs.sort(key=lambda r: (r["sample"], r["event"]))
    return recs


def _hand_pair(rec):
    g = ((rec or {}).get("pio") or {}).get("gammas")
    if g and all(k in g and (g[k].get("energy") or 0) > 0 for k in ("1", "2")):
        return g
    return None


def anchor_of(gg, mode):
    if mode == "corrected":
        return gg.get("em_start_correction") or gg.get("start") or gg.get("reco_start")
    return gg.get("reco_start") or gg.get("start")


def pi0_run(tag, r, theta, how, anchor, sweep, out_dir, restrict=None, quiet=False):
    F, OFF = fudge_of(tag)
    scale = 0.80 / F                     # hand (scan-time) energy -> arm scale
    recs = pi0_hand_gammas()
    L = []
    P = L.append

    P("=== geo_rescore  mode=pi0  arm=%s ===" % tag)
    P("    *** NEW, POSITION-ANCHORED denominator.  NOT the doc pr/135 pi0")
    P("    *** numbers and not comparable to them.  See the script header.")
    P("    R=%.1f cm   THETA=%.0f deg   reco axis=%s   anchor=%s"
      % (r, theta, how, anchor))
    P("    arm scale: kine_shower_fudge_factor=%.2f (hand energies x %.3f)"
      "   pi0_mass_offset=%.0f MeV" % (F, scale, OFF))
    P("    run %s   arm inventory AT RUN TIME: %s"
      % (time.strftime("%Y-%m-%d %H:%M:%S"), inventory(tag)))
    P("    (a Stage-1 cell is still being produced; two runs minutes apart")
    P("     legitimately differ.  Quote a number WITH this inventory line.)")

    # ---------------------------------------------------------- 0. reach
    per = defaultdict(Counter)
    work = []
    for rec in recs:
        if restrict is not None and (rec["sample"], rec["event"]) not in restrict:
            continue
        c = per[rec["sample"]]
        c["hand_pi0"] += 1
        d = load_json(dump_path(tag, rec["sample"], rec["event"]))
        if d is None:
            c["absent"] += 1
            continue
        c["reachable"] += 1
        work.append((rec, ArmEvent(d)))
    P("")
    P("--- 0. reach on this arm (denominator = the two committed manifests) ---")
    P("    %-9s %9s %10s %9s" % ("sample", "hand pi0", "reachable", "no dump"))
    tot = Counter()
    for s in sorted(per):
        c = per[s]
        P("    %-9s %9d %10d %9d" % (s, c["hand_pi0"], c["reachable"], c["absent"]))
        for k in c:
            tot[k] += c[k]
    P("    %-9s %9d %10d %9d" % ("TOTAL", tot["hand_pi0"], tot["reachable"],
                                 tot["absent"]))
    P("    gammas reachable on this arm : %d  (of %d in the label sets)"
      % (2 * tot["reachable"], 2 * tot["hand_pi0"]))
    P("    An unreachable pi0 is NOT a reconstruction failure -- the event has")
    P("    no calib dump on this arm yet.  Every rate below is over the")
    P("    REACHABLE count; the cross-arm table restricts to a common event set.")

    # ---------------------------------------------- 1./2. match and score
    rows = []
    gm = Counter()
    dists, angles, eratio = [], [], []
    miss_reason = Counter()
    pi0c = Counter()
    masses = []
    calib = Counter()
    calib_bad = []
    idres_unmatched = Counter()
    for rec, arm in work:
        wants = []
        for k in ("1", "2"):
            gg = rec["gammas"][k]
            a = anchor_of(gg, anchor)
            if not a:
                continue
            wants.append({"key": k, "anchor": a, "axis": gg.get("axis"),
                          "energy": (gg.get("energy") or 0) * scale})
        got, miss = match_objects(arm, wants, r, theta, how)
        by_id = {int(s["id"]): s for s in arm.showers}
        for k in ("1", "2"):
            gg = rec["gammas"][k]
            gm["gammas"] += 1
            m = got.get(k)
            idsh = by_id.get(int(gg["shower"]))
            if m:
                gm["matched"] += 1
                dists.append(m["dist"])
                angles.append(m["angle"])
                eh = (gg.get("energy") or 0) * scale
                er = m["shower"].get("kine_charge") or 0
                if eh > 0 and er > 0:
                    eratio.append(er / eh)
            else:
                gm["unmatched"] += 1
                miss_reason[miss.get(k, ("?", None))[0]] += 1
            # ---- calibration: geometry vs the id join, where both exist ----
            if idsh is not None:
                calib["id_resolves"] += 1
                if m is None:
                    calib["geo_missed"] += 1
                elif int(m["shower"]["id"]) == int(idsh["id"]):
                    calib["agree"] += 1
                else:
                    calib["disagree"] += 1
                    calib_bad.append((rec["sample"], rec["event"], k,
                                      int(idsh["id"]), int(m["shower"]["id"]),
                                      m["dist"]))
            else:
                calib["id_absent"] += 1
                idres_unmatched["geo_matched" if m else "geo_missed"] += 1
            rows.append(dict(
                sample=rec["sample"], event=rec["event"], gamma=k,
                label_shower=int(gg["shower"]), label_arm=rec["label_arm"],
                anchor_used=anchor,
                e_hand=round(gg.get("energy") or 0, 2),
                matched=1 if m else 0,
                reco_id=int(m["shower"]["id"]) if m else "",
                reco_shower_id=int(m["shower"].get("shower_id", -1)) if m else "",
                dist_cm=round(m["dist"], 4) if m else "",
                angle_deg=round(m["angle"], 2) if m else "",
                e_reco=round(m["shower"].get("kine_charge") or 0, 2) if m else "",
                pio_id=int(m["shower"].get("pio_id", -1)) if m else "",
                pio_mass=round(m["shower"].get("pio_mass") or -1, 2) if m else "",
                id_join=("agree" if (idsh is not None and m
                                     and int(m["shower"]["id"]) == int(idsh["id"]))
                         else "disagree" if (idsh is not None and m)
                         else "id-absent-on-arm" if idsh is None
                         else "geo-missed"),
                miss=miss.get(k, ("", None))[0] if not m else ""))
        # ---- pi0 level ----
        m1, m2 = got.get("1"), got.get("2")
        merged = (m1 is None or m2 is None) and any(
            miss.get(k, ("", None))[0].startswith("lost the one-to-one")
            for k in ("1", "2"))
        if m1 and m2:
            pi0c["both"] += 1
            p1 = int(m1["shower"].get("pio_id", -1))
            p2 = int(m2["shower"].get("pio_id", -1))
            if p1 >= 0 and p1 == p2:
                pi0c["recovered"] += 1
                mm = m1["shower"].get("pio_mass")
                if mm and mm > 0:
                    masses.append(mm)
            elif p1 < 0 and p2 < 0:
                pi0c["neither_grouped"] += 1
            elif p1 < 0 or p2 < 0:
                pi0c["one_grouped"] += 1
            else:
                pi0c["different_groups"] += 1
        elif merged:
            pi0c["merged_into_one_shower"] += 1
        elif m1 or m2:
            pi0c["one_gamma"] += 1
            # A merge usually does NOT present as the one-to-one collision: the
            # merged shower keeps ONE gamma's start, so the other gamma simply
            # falls outside R.  Detect it explicitly -- is the unmatched
            # gamma's NEAREST reco shower (at any distance) the very shower the
            # matched gamma took?
            mk = "1" if m1 else "2"
            uk = "2" if m1 else "1"
            ua = anchor_of(rec["gammas"][uk], anchor)
            if ua and arm.showers:
                near = min(arm.showers,
                           key=lambda s: (vdist(ua, arm.start(s)), int(s["id"])))
                if int(near["id"]) == int(got[mk]["shower"]["id"]):
                    pi0c["one_gamma_merge_beyond_R"] += 1
        else:
            pi0c["neither_gamma"] += 1

    ng = gm["gammas"]
    P("")
    P("--- 1. gamma matching (position-anchored) ---")
    P("    gammas reachable : %d" % ng)
    P("    MATCHED          : %d   (%.1f %%)"
      % (gm["matched"], 100.0 * gm["matched"] / max(1, ng)))
    P("    unmatched        : %d" % gm["unmatched"])
    for k, v in sorted(miss_reason.items(), key=lambda kv: (-kv[1], kv[0])):
        P("        %-46s %d" % (k, v))
    P("    matched distance (cm) : %s" % stat_line(dists))
    P("      histogram  <=0.01 %d | <=0.1 %d | <=1 %d | <=2 %d | <=5 %d"
      % tuple(sum(1 for x in dists if x <= b) for b in (0.01, 0.1, 1, 2, 5)))
    P("    matched angle (deg)   : %s" % stat_line(angles))
    P("    energy sanity E_reco / E_hand(scaled) : %s" % stat_line(eratio))
    P("      outside [0.25, 4]: %d of %d  -- a large-R run pulls a big hand gamma"
      % (sum(1 for x in eratio if x < 0.25 or x > 4), len(eratio)))
    P("      onto a nearby stub (ncpi0 evt142421 at R=12: a 706 MeV gamma matched")
    P("      a 1.5 MeV / 0.8 cm shower 11.4 cm away).  This line is the guard that")
    P("      says whether R is still safe; at the default R it should stay small.")

    nb = pi0c["both"]
    P("")
    P("--- 2. pi0 level (hand pi0 reachable on this arm: %d) ---" % tot["reachable"])
    P("    DEFINITION, and it is NOT the committed census's: pr132_pi0_census.py")
    P("    calls a pi0 'exact' only when the reco group's shower-id SET EQUALS the")
    P("    hand pair.  'RECOVERED' below is looser -- both gammas matched to")
    P("    distinct showers that share a pio_id >= 0 -- so a reco group carrying a")
    P("    THIRD member still counts.  Part of any gap to the committed number is")
    P("    that looser bar, not only renumbering.  Do not quote one as the other.")
    P("    BOTH gammas matched, to DISTINCT showers : %d   (%.1f %%)"
      % (nb, 100.0 * nb / max(1, tot["reachable"])))
    P("        same reco pio_id  = RECOVERED        : %d   (%.1f %% of both)"
      % (pi0c["recovered"], 100.0 * pi0c["recovered"] / max(1, nb)))
    P("        different reco pio_id                : %d" % pi0c["different_groups"])
    P("        one gamma in a group, one not        : %d" % pi0c["one_grouped"])
    P("        neither gamma in any group           : %d" % pi0c["neither_grouped"])
    P("    both gammas' best match is the SAME shower")
    P("      -> the arm MERGED the pair               : %d" % pi0c["merged_into_one_shower"])
    P("    exactly one gamma matched                  : %d" % pi0c["one_gamma"])
    P("        of those, the OTHER gamma's nearest reco shower is the very one")
    P("        the matched gamma took -> a merge just beyond R : %d"
      % pi0c["one_gamma_merge_beyond_R"])
    P("    neither gamma matched                      : %d" % pi0c["neither_gamma"])
    if masses:
        P("    reco pi0 mass of the RECOVERED pairs (MeV): %s"
          % ", ".join("%.1f" % m for m in sorted(masses)))
        P("       median %.1f MeV  (arm offset %.0f MeV; the with-vertex window is"
          " (%.0f, %.0f))" % (pct(masses, 50), OFF, 135 - OFF - 25, 135 - OFF + 35))
    else:
        P("    reco pi0 mass of the RECOVERED pairs: none to report")

    # ------------------------------------------------- 3. calibration
    P("")
    P("--- 3. CALIBRATION: geometry vs the id join, on the gammas the id join resolves ---")
    P("    gammas whose label showers[].id EXISTS on this arm : %d of %d"
      % (calib["id_resolves"], ng))
    P("        geometry picked the SAME shower  : %d" % calib["agree"])
    P("        geometry picked a DIFFERENT one  : %d" % calib["disagree"])
    P("        geometry matched nothing         : %d" % calib["geo_missed"])
    den = calib["agree"] + calib["disagree"]
    if den:
        P("        MATCHING PRECISION (same / matched) : %d/%d = %.3f"
          % (calib["agree"], den, 1.0 * calib["agree"] / den))
    else:
        P("        MATCHING PRECISION: not measurable on this arm -- the id join")
        P("        resolves nothing here, which is the whole point of this script.")
    for b in calib_bad:
        P("          disagreement %s evt %s gamma %s: id -> %d, geometry -> %d (%.2f cm)"
          % b)
    P("    gammas the id join CANNOT resolve                 : %d" % calib["id_absent"])
    P("        of those, geometry DID match a reco shower    : %d   <- the committed"
      % idres_unmatched["geo_matched"])
    P("          census counts every one of these as 'absent-on-arm', i.e. as a")
    P("          reconstruction failure; position says the shower is there.")
    P("          Are they the RIGHT showers?  Section 1's energy-sanity line is the")
    P("          answer to that challenge: %d of %d matched gammas have"
      % (sum(1 for x in eratio if x < 0.25 or x > 4), len(eratio)))
    P("          E_reco/E_hand outside [0.25, 4], and the calibration above is")
    P("          %s."
      % (("%d/%d = %.3f against the id join" % (calib["agree"], den,
                                                1.0 * calib["agree"] / den))
         if den else "not measurable here (the id join is dead on this arm)"))
    P("        of those, geometry found nothing either       : %d   <- genuinely"
      " not reconstructed" % idres_unmatched["geo_missed"])

    # ------------------------------------------------- 4. sweep
    if sweep:
        P("")
        P("--- 4. R / THETA sweep (same arm, same labels; stability of section 1) ---")
        P("    %6s %7s %9s %9s" % ("R cm", "THETA", "matched", "of"))
        for rr, tt in ((2.0, theta), (5.0, theta), (12.0, theta), (25.0, theta),
                       (r, 30.0), (r, 60.0), (r, 180.0)):
            n = 0
            for rec, arm in work:
                wants = []
                for k in ("1", "2"):
                    gg = rec["gammas"][k]
                    a = anchor_of(gg, anchor)
                    if a:
                        wants.append({"key": k, "anchor": a, "axis": gg.get("axis"),
                                      "energy": (gg.get("energy") or 0) * scale})
                g2, _ = match_objects(arm, wants, rr, tt, how)
                n += len(g2)
            P("    %6.1f %7.0f %9d %9d" % (rr, tt, n, ng))

    # ------------------------------------------------- 5. anchor sensitivity
    P("")
    P("--- 5. anchor sensitivity: the 9 gammas carrying an em_start_correction ---")
    nc = ncm = nrm = 0
    for rec, arm in work:
        for k in ("1", "2"):
            gg = rec["gammas"][k]
            if not gg.get("em_start_correction"):
                continue
            nc += 1
            for mode, acc in (("reco", "r"), ("corrected", "c")):
                a = anchor_of(gg, mode)
                g2, _ = match_objects(
                    arm, [{"key": k, "anchor": a, "axis": gg.get("axis"),
                           "energy": (gg.get("energy") or 0) * scale}], r, theta, how)
                if acc == "r":
                    nrm += 1 if g2 else 0
                else:
                    ncm += 1 if g2 else 0
    P("    reachable here: %d   matched from reco_start: %d   from the corrected"
      " start: %d" % (nc, nrm, ncm))
    P("    (reported for transparency only -- section 1-3 use --anchor %s)" % anchor)

    if not quiet:
        txt = os.path.join(out_dir, "geo_pi0_%s.txt" % tag)
        with open(txt, "w") as fh:
            fh.write("\n".join(L) + "\n")
        tsv = os.path.join(out_dir, "geo_pi0_%s.tsv" % tag)
        if rows:
            with open(tsv, "w", newline="") as fh:
                w = csv.DictWriter(fh, delimiter="\t", fieldnames=sorted(rows[0]))
                w.writeheader()
                for rr in sorted(rows,
                                 key=lambda x: (x["sample"], x["event"], x["gamma"])):
                    w.writerow(rr)
        print("\n".join(L))
        print("\nwrote %s\nwrote %s" % (txt, tsv))
    return {"tag": tag, "reach_pi0": tot["reachable"], "gammas": ng,
            "matched": gm["matched"], "both": nb, "recovered": pi0c["recovered"],
            "merged": pi0c["merged_into_one_shower"],
            "precision": (1.0 * calib["agree"] / den) if den else None,
            "calib_n": den, "id_absent": calib["id_absent"],
            "id_absent_geo_ok": idres_unmatched["geo_matched"],
            "events": {(rec["sample"], rec["event"]) for rec, _ in work}}


# =============================================================== pid mode
def pid_labels():
    """-> [(scanname, pass, verdict row)] over the three committed scan sets."""
    out = []
    for name, keyf, labf in PID_SCANS:
        pas = "pass2" if name == "scan3" else "pass1"
        for r in rd_tsv(os.path.join(SX, "docs", "pr", labf)):
            if r.get("verdict"):
                out.append((name, pas, r))
    out.sort(key=lambda t: (t[0], t[2]["sample"], int(t[2]["event"]),
                            int(t[2]["shower_id"])))
    return out


def a5_census_for(tag, out_dir, quiet=False):
    """The arm's A5 census, keyed (sample, event, shower_id).

    Prefers a census already on disk (the previous round wrote them under
    /home/xqian/tmp/pr150/scorers/ named by CELL, not by tag); otherwise runs
    the UNFORKED scripts/pr148_a5_census.py into out_dir.  A shower that is not
    in the census was never EVALUATED by A5 (it only looks at conn-1 |pdg|=11
    showers) -- that is a third bucket, never 'kept'."""
    cell = tag[5:] if tag.startswith("pr150") else tag
    for p in (os.path.join(out_dir, "a5census-%s.tsv" % tag),
              "/home/xqian/tmp/pr150/scorers/a5census-%s.tsv" % cell,
              "/home/xqian/tmp/pr150/scorers/a5census-%s.tsv" % tag):
        if os.path.exists(p):
            src = p
            break
    else:
        src = os.path.join(out_dir, "a5census-%s.tsv" % tag)
        arms = []
        for s in SAMPLES:
            d = os.path.join(SX, "work-%s-%s" % (s, tag))
            if os.path.isdir(d):
                arms += ["--arm", "work-%s-%s" % (s, tag)]
        if not arms:
            return {}, "no arm directory"
        cmd = [sys.executable, os.path.join(SX, "scripts", "pr148_a5_census.py")] \
            + arms + ["--tsv", src]
        p = subprocess.run(cmd, cwd=SX, capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(src):
            return {}, "pr148_a5_census.py rc=%d" % p.returncode
    by = {}
    for r in rd_tsv(src):
        by[(r["sample"], int(r["event"]), int(r["shower_id"]))] = r
    return by, src


def pid_reference():
    """Geometry for every labelled object, from the reference arm (d102mpr).

    Identity is ASSERTED, not assumed: the reference shower's start segment id
    must equal the verdict row's `obj` and its total_length must be within 2 %
    of the recorded one (the pr/148 identity bar).  Anything that fails is
    dropped WITH A COUNT."""
    ref, bad, cache = {}, [], {}
    for name, pas, r in pid_labels():
        smp, ev, sid = r["sample"], int(r["event"]), int(r["shower_id"])
        key = (smp, ev, sid)
        if key in ref or key in dict(bad):
            continue
        if key not in cache:
            d = load_json(dump_path(PID_REF_ARM, smp, ev))
            cache[key] = ArmEvent(d) if d else None
        arm = cache[key]
        if arm is None:
            bad.append((key, "no dump on %s" % PID_REF_ARM))
            continue
        sh = None
        for s in arm.showers:
            if int(s.get("shower_id", -1)) == sid:
                sh = s
                break
        if sh is None:
            bad.append((key, "shower_id absent on %s" % PID_REF_ARM))
            continue
        L0 = float(r["total_len_cm"])
        L1 = float(sh.get("total_length") or 0)
        ok_obj = int(sh["id"]) == int(r["obj"])
        ok_len = L0 > 0 and abs(L1 - L0) / L0 <= LEN_TOL
        if not (ok_obj and ok_len):
            bad.append((key, "identity FAILED on %s (obj %s len %.1f/%.1f)"
                        % (PID_REF_ARM, ok_obj, L1, L0)))
            continue
        ref[key] = {"start": arm.start(sh), "axis_init": arm.axis(sh, "init"),
                    "axis_endstart": arm.axis(sh, "endstart"),
                    "len": L1, "charge": sh.get("kine_charge"),
                    "obj": int(sh["id"])}
    return ref, bad


def pid_run(tag, r, theta, how, out_dir, restrict=None, quiet=False):
    ref, bad = pid_reference()
    if restrict is not None:
        ref = {k: v for k, v in ref.items() if (k[0], k[1]) in restrict}
    cen, cen_src = a5_census_for(tag, out_dir)
    labs = pid_labels()
    L = []
    P = L.append
    P("=== geo_rescore  mode=pid  arm=%s ===" % tag)
    P("    *** NEW, POSITION-ANCHORED denominator.  NOT the doc pr/148 precision")
    P("    *** table and not comparable to it.  See the script header.")
    P("    R=%.1f cm   THETA=%.0f deg   reco axis=%s" % (r, theta, how))
    P("    A5 census: %s" % cen_src)
    P("    run %s   arm inventory AT RUN TIME: %s"
      % (time.strftime("%Y-%m-%d %H:%M:%S"), inventory(tag)))
    P("    (a Stage-1 cell is still being produced; two runs minutes apart")
    P("     legitimately differ.  Quote a number WITH this inventory line.)")
    P("")
    P("--- 0. reference geometry (arm %s, which still resolves the ids) ---"
      % PID_REF_ARM)
    P("    distinct labelled objects : %d" % len({(x[2]["sample"], int(x[2]["event"]),
                                                   int(x[2]["shower_id"])) for x in labs}))
    P("    identity-CONFIRMED on %s (start_seg == obj AND len within %.0f %%) : %d"
      % (PID_REF_ARM, 100 * LEN_TOL, len(ref)))
    for k, why in bad:
        P("        dropped %s evt %d shower_id %d : %s" % (k[0], k[1], k[2], why))

    # ------------------------------------------------- match on this arm
    byev = defaultdict(list)
    for key in sorted(ref):
        byev[(key[0], key[1])].append(key)
    got, miss, armsh = {}, {}, {}
    n_noevt = 0
    present = set()
    for (smp, ev) in sorted(byev):
        d = load_json(dump_path(tag, smp, ev))
        if d is None:
            for key in byev[(smp, ev)]:
                miss[key] = ("event has no dump on this arm", None)
                n_noevt += 1
            continue
        present.add((smp, ev))
        arm = ArmEvent(d)
        wants = [{"key": key, "anchor": ref[key]["start"],
                  "axis": ref[key]["axis_init" if how == "init" else "axis_endstart"],
                  "energy": ref[key]["charge"]} for key in byev[(smp, ev)]]
        g, m = match_objects(arm, wants, r, theta, how)
        got.update(g)
        miss.update(m)
        for key in byev[(smp, ev)]:
            armsh[key] = arm

    dists = [got[k]["dist"] for k in sorted(got)]
    P("")
    P("--- 1. position match onto arm %s ---" % tag)
    P("    objects with reference geometry : %d" % len(ref))
    P("    event not on this arm           : %d" % n_noevt)
    P("    MATCHED by position             : %d" % len(got))
    mr = Counter(miss[k][0] for k in miss if k not in got)
    for k, v in sorted(mr.items(), key=lambda kv: (-kv[1], kv[0])):
        P("        %-46s %d" % (k, v))
    P("    matched distance (cm) : %s" % stat_line(dists))

    # --------------------------------------- A5 verdict, three buckets
    bucket, idcalib, idcalib_rows = {}, Counter(), []
    for key in sorted(ref):
        m = got.get(key)
        if m is None:
            bucket[key] = "unmatched"
            continue
        asid = int(m["shower"].get("shower_id", -1))
        row = cen.get((key[0], key[1], asid))
        if row is None:
            bucket[key] = "not_evaluated"
        else:
            bucket[key] = "retyped" if row["verdict"] == "1" else "kept"
        # calibration vs the id join on this arm
        arm = armsh.get(key)
        idsh = None
        if arm is not None:
            for s in arm.showers:
                if int(s.get("shower_id", -1)) == key[2]:
                    idsh = s
                    break
        if idsh is None:
            idcalib["id_absent"] += 1
        elif int(idsh["id"]) == int(m["shower"]["id"]):
            idcalib["agree"] += 1
        else:
            idcalib["disagree"] += 1
            # Adjudicate: a disagreement does NOT mean geometry is wrong.  On a
            # renumbered arm the scan-time shower_id can still EXIST and point
            # at a different physical object.  The pr/148 identity bar is size:
            # whichever of the two is closer to the scan-time total_length is
            # the one that is the same object.
            L0 = ref[key]["len"]
            dg = abs((m["shower"].get("total_length") or 0) - L0)
            di = abs((idsh.get("total_length") or 0) - L0)
            idcalib["adj_geo" if dg < di else
                    "adj_id" if di < dg else "adj_tie"] += 1
            idcalib_rows.append((key[0], key[1], key[2], int(idsh["id"]),
                                 int(m["shower"]["id"]), L0,
                                 idsh.get("total_length") or 0,
                                 m["shower"].get("total_length") or 0))
    P("")
    P("--- 2. A5 verdict on the matched objects (THREE buckets) ---")
    bc = Counter(bucket.values())
    P("    re-typed by A5      : %d" % bc["retyped"])
    P("    kept by A5          : %d" % bc["kept"])
    P("    NOT EVALUATED by A5 : %d   <- A5 only looks at conn-1 |pdg|=11 showers;"
      % bc["not_evaluated"])
    P("                             folding these into 'kept' (as a two-bucket")
    P("                             scorer does) fakes the precision denominator")
    P("                             on any arm that moves conn type or pdg.")
    P("    unmatched           : %d" % bc["unmatched"])

    P("")
    P("--- 3. verdict agreement, position-anchored ---")
    tab = {"pass1": defaultdict(Counter), "pass2": defaultdict(Counter)}
    seen = set()
    for name, pas, rrow in labs:
        key = (rrow["sample"], int(rrow["event"]), int(rrow["shower_id"]))
        if key not in ref:
            continue
        if (pas, key) in seen:
            continue
        seen.add((pas, key))
        tab[pas][rrow["verdict"]][bucket[key]] += 1
    for pas, what in (("pass1", "pass 1 = scan0 + scan2, 36 distinct objects"),
                      ("pass2", "pass 2 = scan3, the SAME 36 objects rescanned")):
        t = tab[pas]
        P("")
        P("  %s" % what)
        P("    %-10s %10s %7s %14s %10s %6s"
          % ("hand", "A5 retyped", "kept", "not evaluated", "unmatched", "total"))
        tt = Counter()
        for v in ("HADRONIC", "MIXED", "EM"):
            c = t.get(v, Counter())
            n = sum(c.values())
            P("    %-10s %10d %7d %14d %10d %6d"
              % (v, c["retyped"], c["kept"], c["not_evaluated"], c["unmatched"], n))
            for kk in c:
                tt[kk] += c[kk]
        P("    %-10s %10d %7d %14d %10d %6d"
          % ("TOTAL", tt["retyped"], tt["kept"], tt["not_evaluated"],
             tt["unmatched"], sum(tt.values())))
        good = t.get("HADRONIC", Counter())["retyped"]
        if tt["retyped"]:
            P("    A5 precision on labelled re-types (HADRONIC / re-typed) : %s"
              % rate(good, tt["retyped"]))
            P("      [not-evaluated %d excluded from the denominator]"
              % tt["not_evaluated"])
        else:
            P("    A5 precision: no position-matched re-type on this arm -- NOT 0,")
            P("    simply not measurable here.")

    P("")
    P("--- 4. CALIBRATION: geometry vs the id join on this arm ---")
    n = idcalib["agree"] + idcalib["disagree"]
    P("    matched objects whose scan-time shower_id still exists here : %d" % n)
    P("        geometry picked the SAME shower   : %d" % idcalib["agree"])
    P("        geometry picked a DIFFERENT one   : %d" % idcalib["disagree"])
    if n:
        P("        MATCHING PRECISION : %d/%d = %.3f"
          % (idcalib["agree"], n, 1.0 * idcalib["agree"] / n))
    else:
        P("        MATCHING PRECISION: not measurable -- the scan-time shower_id")
        P("        resolves nothing here, which is why this script exists.")
    if tag == PID_REF_ARM:
        P("      (this IS the reference arm, so agreement here is a self-consistency")
        P("       check of the matcher's mechanics, not an independent measurement.)")
    P("    matched objects whose shower_id is GONE here : %d" % idcalib["id_absent"])
    if idcalib["disagree"]:
        P("    ADJUDICATING the %d disagreement(s).  A disagreement does NOT mean"
          % idcalib["disagree"])
        P("    geometry is wrong: on a renumbered arm the scan-time shower_id can")
        P("    still EXIST and name a different physical object.  The pr/148")
        P("    identity bar is SIZE, so whichever candidate's total_length is")
        P("    closer to the scan-time length is the same object.")
        P("        geometry closer to the scan-time length : %d" % idcalib["adj_geo"])
        P("        the id   closer to the scan-time length : %d" % idcalib["adj_id"])
        P("        tie                                     : %d" % idcalib["adj_tie"])
        for rr in idcalib_rows:
            P("          %s evt %d shower_id %d: id -> seg %d (%.1f cm), geometry ->"
              " seg %d (%.1f cm); scan-time %.1f cm"
              % (rr[0], rr[1], rr[2], rr[3], rr[6], rr[4], rr[7], rr[5]))

    rows = []
    for key in sorted(ref):
        m = got.get(key)
        rows.append(dict(sample=key[0], event=key[1], scan_shower_id=key[2],
                         ref_obj=ref[key]["obj"],
                         ref_len_cm=round(ref[key]["len"], 2),
                         matched=1 if m else 0,
                         arm_id=int(m["shower"]["id"]) if m else "",
                         arm_shower_id=int(m["shower"].get("shower_id", -1)) if m else "",
                         dist_cm=round(m["dist"], 4) if m else "",
                         angle_deg=round(m["angle"], 2) if m else "",
                         arm_len_cm=round(m["shower"].get("total_length") or 0, 2) if m else "",
                         bucket=bucket[key],
                         miss=miss[key][0] if not m else ""))
    if not quiet:
        txt = os.path.join(out_dir, "geo_pid_%s.txt" % tag)
        with open(txt, "w") as fh:
            fh.write("\n".join(L) + "\n")
        tsv = os.path.join(out_dir, "geo_pid_%s.tsv" % tag)
        if rows:
            with open(tsv, "w", newline="") as fh:
                w = csv.DictWriter(fh, delimiter="\t", fieldnames=sorted(rows[0]))
                w.writeheader()
                for rr in rows:
                    w.writerow(rr)
        print("\n".join(L))
        print("\nwrote %s\nwrote %s" % (txt, tsv))
    p1 = tab["pass1"]
    tt = Counter()
    for v in p1:
        for kk in p1[v]:
            tt[kk] += p1[v][kk]
    return {"tag": tag, "objects": len(ref), "matched": len(got),
            "retyped": tt["retyped"], "kept": tt["kept"],
            "not_eval": tt["not_evaluated"],
            "prec": (1.0 * p1.get("HADRONIC", Counter())["retyped"] / tt["retyped"])
            if tt["retyped"] else None,
            "good": p1.get("HADRONIC", Counter())["retyped"],
            "calib": (idcalib["agree"], idcalib["agree"] + idcalib["disagree"]),
            "events": present}


# =============================================================== main
def main():
    ap = argparse.ArgumentParser(
        description="position-anchored re-scorer for the SBND pi0 and PID hand scans")
    ap.add_argument("--mode", required=True, choices=("pi0", "pid"))
    ap.add_argument("--arm", action="append", required=True,
                    help="arm tag, e.g. d102mpr / pr150s0 / pr150csp3bw (repeatable)")
    ap.add_argument("--r", type=float, default=R_DEFAULT,
                    help="match radius in cm (default %.1f = the 95th percentile of a "
                         "KNOWN-CORRECT start distance, measured)" % R_DEFAULT)
    ap.add_argument("--theta", type=float, default=THETA_DEFAULT,
                    help="axis-angle gate in deg (default %.0f, measured; 30 rejects "
                         "18 %% of known-correct pairs -- see the header)" % THETA_DEFAULT)
    ap.add_argument("--axis", default="init", choices=("init", "endstart"),
                    help="reco axis definition (default init = em_geom.shower_init_dir)")
    ap.add_argument("--anchor", default="reco", choices=("reco", "corrected"),
                    help="pi0 mode: which label start point anchors the match")
    ap.add_argument("--sweep", action="store_true", help="pi0 mode: R/THETA sweep")
    ap.add_argument("--out", default="/home/xqian/tmp/pr150/scorers2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    res = []
    for tag in a.arm:
        if not any(os.path.isdir(os.path.join(SX, "work-%s-%s" % (s, tag)))
                   for s in SAMPLES):
            print("=== arm %s: NO work-<sample>-%s directory -- skipped ===\n"
                  % (tag, tag))
            continue
        print("")
        if a.mode == "pi0":
            res.append(pi0_run(tag, a.r, a.theta, a.axis, a.anchor, a.sweep, a.out))
        else:
            res.append(pid_run(tag, a.r, a.theta, a.axis, a.out))

    if len(res) < 2:
        return 0
    common = set.intersection(*[r["events"] for r in res]) if res else set()
    S = []
    S.append("")
    S.append("=================== cross-arm summary, mode=%s ===================" % a.mode)
    S.append("*** POSITION-ANCHORED, a NEW denominator.  Not doc pr/135 / pr/148. ***")
    if a.mode == "pi0":
        S.append("%-13s %8s %8s %8s %8s %8s %8s %11s"
                 % ("arm", "pi0", "gammas", "matched", "both", "recov", "merged",
                    "precision"))
        for r in res:
            S.append("%-13s %8d %8d %8d %8d %8d %8d %11s"
                     % (r["tag"], r["reach_pi0"], r["gammas"], r["matched"],
                        r["both"], r["recovered"], r["merged"],
                        ("%.3f (%d)" % (r["precision"], r["calib_n"]))
                        if r["precision"] is not None else "n/a (0)"))
        S.append("")
        S.append("  'precision' = geometry match agrees with the id join, on the")
        S.append("  gammas the id join still resolves (n in brackets).  n=0 means the")
        S.append("  id join is dead on that arm -- the reason this script exists.")
        S.append("")
        S.append("  id-anchored loss recovered by position:")
        for r in res:
            S.append("    %-13s id join cannot resolve %3d gammas; position finds a"
                     " reco shower for %3d of them"
                     % (r["tag"], r["id_absent"], r["id_absent_geo_ok"]))
    else:
        S.append("%-13s %8s %8s %8s %8s %13s  %s"
                 % ("arm", "objects", "matched", "retyped", "kept", "not-evaluated",
                    "precision (HADRONIC / re-typed)"))
        for r in res:
            S.append("%-13s %8d %8d %8d %8d %13d  %s"
                     % (r["tag"], r["objects"], r["matched"], r["retyped"],
                        r["kept"], r["not_eval"], rate(r["good"], r["retyped"])))
        S.append("")
        S.append("  A precision here rests on the RE-TYPED count, which on a Stage-1")
        S.append("  cell is 1-2 objects.  Read the fraction, never the decimal.")
        S.append("")
        S.append("  geometry-vs-id calibration (agree / id-resolvable):")
        for r in res:
            S.append("    %-13s %d / %d" % (r["tag"], r["calib"][0], r["calib"][1]))
    S.append("")
    S.append("--- CELL-TO-CELL COMPARABLE: every arm restricted to the %d event(s)"
             " ALL of them reach ---" % len(common))
    S.append("  The raw table above sits on DIFFERENT denominators (a Stage-1 cell")
    S.append("  has fewer dumps), so only this restricted table compares cells.")
    S.append("  The pi0 and pid common sets are different sizes on purpose -- they")
    S.append("  are the intersections over two DIFFERENT object sets (the 66 hand")
    S.append("  pi0 events and the 36 PID-scan events), not the same set counted twice.")
    rest = []
    for tag in [r["tag"] for r in res]:
        if a.mode == "pi0":
            rest.append(pi0_run(tag, a.r, a.theta, a.axis, a.anchor, False, a.out,
                                restrict=common, quiet=True))
        else:
            rest.append(pid_run(tag, a.r, a.theta, a.axis, a.out,
                                restrict=common, quiet=True))
    if a.mode == "pi0":
        S.append("  %-13s %8s %8s %8s %8s %8s %8s"
                 % ("arm", "pi0", "gammas", "matched", "both", "recov", "merged"))
        for r in rest:
            S.append("  %-13s %8d %8d %8d %8d %8d %8d"
                     % (r["tag"], r["reach_pi0"], r["gammas"], r["matched"],
                        r["both"], r["recovered"], r["merged"]))
    else:
        S.append("  %-13s %8s %8s %8s %8s %13s  %s"
                 % ("arm", "objects", "matched", "retyped", "kept", "not-evaluated",
                    "precision (HADRONIC / re-typed)"))
        for r in rest:
            S.append("  %-13s %8d %8d %8d %8d %13d  %s"
                     % (r["tag"], r["objects"], r["matched"], r["retyped"],
                        r["kept"], r["not_eval"], rate(r["good"], r["retyped"])))
    txt = "\n".join(S)
    print(txt)
    p = os.path.join(a.out, "geo_%s_summary.txt" % a.mode)
    with open(p, "w") as fh:
        fh.write(txt + "\n")
        fh.write("\ncommon events:\n")
        for e in sorted(common):
            fh.write("  %s %s\n" % e)
    print("\nwrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
