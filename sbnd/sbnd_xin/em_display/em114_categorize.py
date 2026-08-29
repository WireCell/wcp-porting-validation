#!/usr/bin/env python3
"""Categorise the em_display hand scan into work buckets (doc pr/115).

Reads the labels written by `em_display_viewer.py` plus the sample manifest and
files each event into a bucket, with the evidence that put it there.
READ-ONLY over `em_labels/` (CLAUDE.md M13): nothing here writes into a tag.

    ./em114_categorize.py                                   # table to stdout
    ./em114_categorize.py --tsv ../docs/pr/pr115-scan.tsv    # the row table
    ./em114_categorize.py --tag emscan-0827                  # another tag

Why a script and not a hand-written table: `em.verdict` -- the display's own
radio button -- is set on 1 of 97 records, so every bucket below is INFERRED
from the free-text note plus the marks.  The inference rule is therefore the
reviewable artifact, and it has to be re-runnable and overrulable line by line.
"""
import argparse, csv, glob, json, math, os, re
from collections import Counter

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(SX, "em_display", "em114-manifest.tsv")
LABEL_ROOT = os.path.join(SX, "em_labels")

# ---------------------------------------------------------------- vocabulary
# Phrases the scanner actually used, read off the 65 notes on disk -- not a
# guessed vocabulary.  Order inside a pattern does not matter; order BETWEEN
# the tests in `classify` does.
RE_VERTEX = re.compile(
    r"(incorrect|wrong|not clear|unclear|not likely correct|not ideal|"
    r"give up)[^.]{0,45}vertex"
    r"|vertex[^.]{0,45}(wrong|incorrect|not likely correct|not ideal|"
    r"not clear|unclear)", re.I)
RE_GIVEUP = re.compile(r"give up|difficult to scan|not easily label", re.I)
RE_OVER = re.compile(r"overcluster|over-cluster|separate|separat|merged", re.I)
RE_UNDER = re.compile(r"not clustered together|not tagged|include all the pieces"
                      r"|missing|should be (a |an )?(em shower|part of)", re.I)
RE_PID = re.compile(r"ided as|id'?ed as|should be an em shower|pid", re.I)
RE_GOOD = re.compile(r"^\s*(very good|pretty good|good|ok)\s*\.?\s*$", re.I)

# Buckets: the four the owner asked for, the two he named as extra, and the two
# the data forces -- an event nobody scanned, and an event scanned with neither
# a comment nor a correction, which is NOT the same as "good".
B_UNDER = "1 under-clustered"
B_OVER = "2 over-clustered"
B_BOTH = "1+2 both"
B_GOOD = "good (no major change)"
B_VTXBAD = "vertex-bad (not actionable)"
B_BUSY = "undecidable / too busy"
B_SILENT = "scanned, no clustering correction"
B_UNSCANNED = "not scanned"
BUCKET_ORDER = [B_UNDER, B_OVER, B_BOTH, B_GOOD, B_VTXBAD, B_BUSY,
                B_SILENT, B_UNSCANNED]

# pi0 axis (the owner's groups 3 and 4).  ORTHOGONAL to the bucket above: an
# event can be a no-vertex pi0 AND under-clustered -- evt76346 is both.
P_KNOWN = "3 pi0, vertex usable"
P_NOVTX = "4 pi0, vertex not usable"
P_NONE = "-"

# A hand-placed pi0 vertex this far from the reco's main vertex is the scanner
# REJECTING the main vertex, not confirming it.  15 cm is well outside vertex
# resolution and well inside the smallest rejection actually seen (47.9 cm);
# the one confirmation seen sits at 3.5 cm.
MANUAL_REJECT_CM = 15.0


def load_manifest(path=None):
    with open(path or MANIFEST) as fh:
        return {"evt" + r["event"]: r for r in csv.DictReader(fh, delimiter="\t")}


def load_labels(tag):
    out = {}
    for p in sorted(glob.glob(os.path.join(LABEL_ROOT, tag, "labels-evt*.json"))):
        with open(p) as fh:
            out[os.path.basename(p)[len("labels-"):-len(".json")]] = json.load(fh)
    return out


def summarise(lbl, rec, man):
    """Everything the rule is allowed to look at, flattened.

    The mark breakdown uses `marks_detail[shw].marked[seg].owner`, which is the
    shower that owned the segment at scan time.  Without it a mark count is
    ambiguous in two ways that change the answer:

    * on evt142421 the scanner selected a ONE-segment stub and used it as a
      scratch pad to split somebody else's shower into two gammas, so the 33
      "IN" marks there are an over-clustering statement about shower 108104,
      not an under-clustering statement about the stub;
    * an IN mark on a segment the reco left as its OWN one-segment shower is a
      different repair from an IN mark on a segment belonging to a real
      neighbouring shower.  The first is an orphan stub that was never
      absorbed; the second is one shower split in two.  Both read as "IN".

    So each IN mark is filed as merge / orphan / unowned / noop, and each OUT
    as member / other.  Only merge+orphan+unowned are corrections.
    """
    em = (rec or {}).get("em") or {}
    pio = (rec or {}).get("pio") or {}
    marks = em.get("marks_by_shower") or {}
    detail = em.get("marks_detail") or {}
    per, agg = [], Counter()
    for nd in sorted(marks, key=lambda s: int(s)):
        nd_i = int(nd)
        mem = set((detail.get(nd) or {}).get("members") or [])
        md = (detail.get(nd) or {}).get("marked") or {}
        kinds = {int(k): v for k, v in marks[nd].items()}
        cnt, from_ = Counter(), Counter()
        owners = set()
        for sid, kind in sorted(kinds.items()):
            own = (md.get(str(sid)) or {}).get("owner")
            if own is not None and own != nd_i:
                owners.add(own)
            if kind == "in":
                # already inside the shower -> the scanner used IN as a
                # highlighter, and `marks_energy` skips it.
                cnt["in_noop" if sid in mem else
                    "in_unowned" if own is None else
                    "in_orphan" if own == sid else
                    "in_merge"] += 1
                if own is not None and own not in (sid, nd_i):
                    from_[own] += 1
            elif kind == "out":
                # OUT on a non-member is a no-op for the energy, but when the
                # selected shower is a stub it is how the scanner said "these
                # form the OTHER gamma".  Kept separately, never dropped.
                cnt["out_mem" if sid in mem else "out_other"] += 1
        per.append(dict(shower=nd_i, n_members=len(mem),
                        seed_out=kinds.get(nd_i) == "out",
                        other_owners=sorted(owners),
                        # the single shower most of the IN marks came from --
                        # 19 of evt168596's 20 came from shower 14058, which is
                        # "these two showers are one", not "20 loose pieces".
                        from_top=(from_.most_common(1)[0] if from_ else None),
                        **cnt))
        agg.update(cnt)
    mv = (rec or {}).get("main_vertex") or {}
    v = pio.get("vertex")
    dvtx = (math.dist(v, [mv["x"], mv["y"], mv["z"]])
            if v and mv.get("x") is not None else None)
    note = ((rec or {}).get("note") or "").strip()
    return dict(
        label=lbl, event=int(lbl[3:]),
        sample=(rec or {}).get("sample") or man.get("sample"),
        origin=(rec or {}).get("origin") or man.get("origin"),
        scanned=rec is not None, note=note,
        manifest_note=(man.get("scan_note") or "").strip(),
        flags=(rec or {}).get("event_flags") or [],
        verdict=em.get("verdict"), confidence=(rec or {}).get("confidence"),
        saved=(rec or {}).get("saved_utc"), sel_shower=em.get("shower"),
        per_shower=per,
        in_merge=agg["in_merge"], in_orphan=agg["in_orphan"],
        in_unowned=agg["in_unowned"], in_noop=agg["in_noop"],
        out_mem=agg["out_mem"], out_other=agg["out_other"],
        n_in=agg["in_merge"] + agg["in_orphan"] + agg["in_unowned"]
        + agg["in_noop"],
        n_out=agg["out_mem"] + agg["out_other"],
        seed_out=[p["shower"] for p in per if p["seed_out"]],
        n_showers_touched=len(per),
        n_starts=len(em.get("start_override_by_shower") or {}),
        n_dirpts=len(em.get("dir_point_by_shower") or {}),
        has_pio=bool(pio), n_cands=len(pio.get("candidates") or []),
        vertex_how=pio.get("vertex_how"), vertex_dist=dvtx,
        mass_axis=pio.get("mass_axis_convention"),
        mass_vertex=pio.get("mass_vertex_convention"),
        reco_pio_mass=(pio.get("reco_kine") or {}).get("kine_pio_mass"),
        n_reco_groups=len(pio.get("reco_groups") or {}),
        man_pio_groups=man.get("n_pio_groups"), man_n_em=man.get("n_em"),
        man_n_shower=man.get("n_shower"), man_n_seg=man.get("n_seg"),
        man_pio_mass=man.get("kine_pio_mass"),
    )


def _pio_axis(s, why):
    """The pi0 axis, lifted verbatim out of `classify` so the --use-verdict
    early return can reach it.  Same statements, same order, same `why` text."""
    vtx_note = bool(RE_VERTEX.search(s["note"].lower()))
    pio = P_NONE
    if s["has_pio"]:
        if "no_vertex_ncpi0" in s["flags"]:
            pio = P_NOVTX
            why.append("scanner set the no_vertex_ncpi0 flag")
        elif s["vertex_how"] == "backproject":
            pio = P_NOVTX
            why.append("pi0 vertex from back-projecting the two gamma rays")
        elif (s["vertex_how"] == "manual" and s["vertex_dist"] is not None
              and s["vertex_dist"] > MANUAL_REJECT_CM):
            pio = P_NOVTX
            why.append("pi0 vertex placed by hand %.0f cm off the main vertex"
                       % s["vertex_dist"])
        elif vtx_note:
            pio = P_NOVTX
            why.append("note rejects the vertex")
        else:
            pio = P_KNOWN
            why.append("pi0 anchored on the reco main vertex"
                       + ("" if s["vertex_how"] != "manual" else
                          ", re-placed by hand only %.1f cm away -- a "
                          "confirmation" % (s["vertex_dist"] or 0.0)))
    return pio


# The display's own verdict radio -> bucket.  Used ONLY under --use-verdict
# (doc pr/116): the pr/115 scan set it on 1 of 97 records, so the default path
# must stay the note+mark inference below or that table changes under us.  The
# two PID verdicts are deliberately absent -- they are statements about particle
# id, not about clustering, so they fall through to the inference.
VERDICT_BUCKET = {
    "correct": B_GOOD,
    "over-clustered": B_OVER,
    "under-clustered": B_UNDER,
    "both": B_BOTH,
    "vertex-bad (undecidable)": B_VTXBAD,
}


def classify(s, use_verdict=False):
    """Return (bucket, pi0 axis, [reason, ...]).

    PRECEDENCE, deliberate and in this order:

    0. never scanned beats everything -- absence of a correction is not
       evidence of correctness.
    1. abandoned scan: a vertex complaint AND no marks AND no pi0.  The
       conjunction matters: evt54332's note says the vertex is wrong and the
       scanner corrected the clustering anyway, so a vertex complaint on its
       own must not send an event to the not-actionable pile.
    2. THE NOTE GOVERNS DIRECTION, not the marks.  evt463565 carries 26 IN
       marks and a note saying the two gammas are merged and that including
       the pieces was the achievable thing, not the right thing; evt84229
       carries 8 IN marks, 7 of them on segments the shower already had, and a
       note saying it swallowed a second gamma.  Reading the marks alone files
       both under under-clustering, which is backwards.
    3. only then do the marks decide, and only the marks that are not no-ops.
    """
    why = []
    if not s["scanned"]:
        return B_UNSCANNED, P_NONE, ["no label file in this scan tag"]

    note, low = s["note"], s["note"].lower()
    if use_verdict and s.get("verdict") in VERDICT_BUCKET:
        # The pi0 axis is still computed from the record, never from the radio:
        # there is no pi0 verdict (pr/114 README), the gamma slots are it.
        b = VERDICT_BUCKET[s["verdict"]]
        why.append("scanner set the verdict radio to %r%s"
                   % (s["verdict"],
                      "" if not s.get("confidence")
                      else " (%s)" % s["confidence"]))
        return b, _pio_axis(s, why), why
    over_note = bool(RE_OVER.search(low))
    under_note = bool(RE_UNDER.search(low))
    vtx_note = bool(RE_VERTEX.search(low))
    corrected = bool(s["n_in"] or s["n_out"] or s["has_pio"])
    real_in = s["in_merge"] + s["in_orphan"] + s["in_unowned"]
    real_out = s["out_mem"] + s["out_other"]

    # ---- pi0 axis, computed independently of the bucket ------------------
    pio = _pio_axis(s, why)

    # ---- bucket ----------------------------------------------------------
    # A NAMED DIRECTION OUTRANKS a vertex complaint or a give-up.  evt281567
    # says "nu vertex is wrong ... , 95128 has an overclustering issue" and
    # evt176502 says "Significant EM shower overclustering ... not easily
    # label": both are diagnoses of over-clustering, and burying them in the
    # not-actionable pile loses the only thing the scanner did record.  The
    # `vertex-suspect` / `hard` / `no-marks` tags carry the rest.
    if not (over_note or under_note):
        if vtx_note and not corrected:
            why.append("vertex complaint, no correction attempted: %r" % note)
            return B_VTXBAD, pio, why
        if RE_GIVEUP.search(low) and not corrected:
            why.append("recorded as unscannable: %r" % note)
            return B_BUSY, pio, why

    if over_note and under_note:
        why.append("note names both directions: %r" % note)
        return B_BOTH, pio, why
    if over_note:
        why.append("note names over-clustering: %r" % note)
        if real_in and not real_out:
            why.append("NOTE OVERRIDES MARKS: %d IN mark(s) and no OUT, but "
                       "the note says the fix is separation" % real_in)
        return B_OVER, pio, why
    if under_note:
        why.append("note names missing pieces: %r" % note)
        return B_UNDER, pio, why

    if real_in and real_out:
        why.append("%d IN and %d OUT marks that are not no-ops" % (real_in, real_out))
        return B_BOTH, pio, why
    if real_out:
        why.append("%d OUT mark(s), none IN" % real_out)
        return B_OVER, pio, why
    if real_in:
        why.append("%d IN mark(s), none OUT" % real_in)
        return B_UNDER, pio, why
    if s["in_noop"]:
        why.append("%d IN mark(s), every one on a segment the shower already "
                   "had -- the scanner was highlighting, not correcting"
                   % s["in_noop"])
        return B_SILENT, pio, why

    if RE_GOOD.match(note):
        why.append("scanner wrote %r and made no correction" % note)
        return B_GOOD, pio, why
    if note:
        why.append("note carries no clustering direction: %r" % note)
        return B_SILENT, pio, why
    why.append("no note and no marks"
               + ("; a pi0 was built" if s["has_pio"] else ""))
    return B_SILENT, pio, why


def tags(s):
    """Secondary labels.  These do not move an event between buckets; they say
    what kind of work it is once it is in one."""
    t = []
    if s["per_shower"]:
        # `stub` = the reco grew almost nothing and the scanner attached the
        # object to a seed; `tail` = a real shower that lost pieces.  Different
        # failures: never grown vs the acceptance gate stopping too early.
        worst, ratio = None, -1.0
        for p in s["per_shower"]:
            gained = (p.get("in_merge", 0) + p.get("in_orphan", 0)
                      + p.get("in_unowned", 0))
            if not gained:
                continue
            r = gained / float(max(p["n_members"], 1))
            if r > ratio:
                ratio, worst = r, p
        if worst is not None:
            t.append("stub" if worst["n_members"] <= 3 else "tail")
    if s["seed_out"]:
        t.append("seed-out")          # the shower's own seed segment rejected
    if s["in_merge"]:
        t.append("merge")             # pieces taken from a real neighbour
    if s["in_orphan"]:
        t.append("absorb-orphan")     # pieces the reco left as 1-segment showers
    if s["out_other"]:
        t.append("split-by-proxy")    # OUT used on a non-member to name a gamma
    if s["in_noop"] and not (s["in_merge"] + s["in_orphan"] + s["in_unowned"]):
        t.append("highlight-only")
    if s["n_showers_touched"] > 1:
        t.append("multi-shower")
    if RE_PID.search(s["note"]) or RE_PID.search(s["manifest_note"]):
        t.append("pid")
    if RE_VERTEX.search(s["note"]):
        t.append("vertex-suspect")
    if RE_GIVEUP.search(s["note"]):
        t.append("hard")
    if not (s["n_in"] or s["n_out"]):
        t.append("no-marks")
    # The pr/113 survey note and the hand-scan note disagree: the survey saw a
    # PID or topology problem, the scan wrote "good".  Most likely both are
    # right -- this display asks about SEGMENT MEMBERSHIP, so "good" is scoped
    # to clustering and is silent on PID.  Flagged, not reconciled.
    if s["manifest_note"] and RE_GOOD.match(s["note"]):
        t.append("note-conflict")
    if s["n_starts"] or s["n_dirpts"]:
        t.append("start-moved")
    return t


def ratio(s):
    best = 0.0
    for p in s["per_shower"]:
        gained = (p.get("in_merge", 0) + p.get("in_orphan", 0)
                  + p.get("in_unowned", 0))
        if gained:
            best = max(best, gained / float(max(p["n_members"], 1)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="emscan-0827")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--manifest", default=None,
                    help="sample manifest TSV (default: em114-manifest.tsv, "
                         "the pr/115 sample)")
    ap.add_argument("--use-verdict", action="store_true",
                    help="let the display's verdict radio decide the bucket "
                         "when it is set, instead of inferring from the note "
                         "and marks.  OFF by default: the pr/115 scan set the "
                         "radio on 1 of 97 records, so turning this on would "
                         "change that table.")
    a = ap.parse_args()

    man, labs = load_manifest(a.manifest), load_labels(a.tag)
    rows = []
    for lbl in sorted(man, key=lambda k: int(k[3:])):
        s = summarise(lbl, labs.get(lbl), man[lbl])
        s["bucket"], s["pi0"], s["why"] = classify(s, a.use_verdict)
        s["tags"] = tags(s)
        s["gain_ratio"] = ratio(s)
        rows.append(s)

    print("scan tag %s -- %d of %d manifest events carry a label"
          % (a.tag, sum(r["scanned"] for r in rows), len(rows)))
    print()
    for b in BUCKET_ORDER:
        sel = [r for r in rows if r["bucket"] == b]
        if not sel:
            continue
        print("== %s -- %d event(s)" % (b, len(sel)))
        for r in sorted(sel, key=lambda r: -r["gain_ratio"]):
            print("   evt%-8s %-8s %-24s in %d/%d/%d/%d out %d/%d  %-34s %s"
                  % (r["event"], r["sample"], r["pi0"],
                     r["in_merge"], r["in_orphan"], r["in_unowned"],
                     r["in_noop"], r["out_mem"], r["out_other"],
                     ",".join(r["tags"]), r["why"][-1][:60]))
        print()
    print("   (in merge/orphan/unowned/noop = taken from a real neighbour / "
          "from a 1-segment\n    shower / from nobody / already a member.  "
          "out member/other.)")
    print()
    print("== pi0 axis")
    for p in (P_KNOWN, P_NOVTX):
        sel = [r for r in rows if r["pi0"] == p]
        print("   %-24s %2d : %s" % (p, len(sel),
                                     " ".join("evt%d" % r["event"] for r in sel)))
    print()
    print("== cross-tab  (rows = bucket, cols = %s | %s | %s)"
          % (P_KNOWN, P_NOVTX, P_NONE))
    ct = Counter((r["bucket"], r["pi0"]) for r in rows)
    for b in BUCKET_ORDER:
        print("   %-34s %3d %3d %3d" % (b, ct.get((b, P_KNOWN), 0),
                                        ct.get((b, P_NOVTX), 0),
                                        ct.get((b, P_NONE), 0)))
    print()
    print("== tags")
    for k, n in Counter(t for r in rows for t in r["tags"]).most_common():
        print("   %-16s %2d : %s" % (k, n, " ".join(
            "evt%d" % r["event"] for r in rows if k in r["tags"])))

    if a.tsv:
        cols = ["event", "sample", "origin", "bucket", "pi0", "tags",
                "gain_ratio", "in_merge", "in_orphan", "in_unowned",
                "in_noop", "out_mem", "out_other", "seed_out", "n_showers_touched", "sel_shower",
                "has_pio", "n_cands", "vertex_how", "vertex_dist", "mass_axis",
                "mass_vertex", "reco_pio_mass", "flags", "confidence",
                "verdict", "note", "manifest_note", "why"]
        with open(a.tsv, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(cols)
            for r in rows:
                out = []
                for c in cols:
                    v = r[c]
                    if v is None:
                        out.append("")
                    elif c == "gain_ratio":
                        out.append("%.3f" % v)
                    elif c in ("vertex_dist", "mass_axis", "mass_vertex",
                               "reco_pio_mass"):
                        out.append("%.1f" % v)
                    elif isinstance(v, list):
                        out.append(" | ".join(map(str, v)))
                    else:
                        out.append(str(v).replace("\t", " ").replace("\n", " "))
                w.writerow(out)
        print("\nwrote %s" % a.tsv)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print("wrote %s" % a.json)


if __name__ == "__main__":
    main()
