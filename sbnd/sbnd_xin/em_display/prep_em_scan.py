#!/usr/bin/env python3
"""doc pr/114 -- prepare the EM / pi0 hand-scan sample.

Three jobs, each independently runnable:

  (1) MANIFEST  (default)   the scan event list with per-event summary stats and
                            a Bee deep link, as one .tsv the viewer reads.
  (2) PROBES    --parse-probes
                            turn the stage-2 `stdout.log` files into one small
                            emprep-evt<ID>.json per event: NON-LOSSY shower
                            membership plus, per segment, WHICH pass absorbed it.
  (3) BEE       --bee-index (on by default) / --bee-build
                            resolve per-event Bee links offline, and optionally
                            assemble a fresh upload zip.  NEVER uploads: that is
                            outward-facing and owner-gated (CLAUDE.md sec 5.6,
                            bee-review SKILL sec 5).

The display reads the PROD0825 dumps, not the stage-2 arm's -- they are equal on
every physics field (doc pr/114 sec 3), and prod0825 is the canonical product.
The stage-2 arm contributes probe text only, joined on the stable ids.

Usage:
  prep_em_scan.py --out em_display/em114-manifest.tsv
  prep_em_scan.py --out ... --parse-probes work-em114-ncpi0 work-em114-nuecc48 ...
  prep_em_scan.py --out ... --bee-build bee/em114
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import em_geom as G  # noqa: E402

PROD = "prod0825"
BEE_BASE = "https://www.phy.bnl.gov/twister/bee"


# ---------------------------------------------------------------------------
# the scan sample
# ---------------------------------------------------------------------------


def read_index(path):
    """(sample, run, subrun, event) rows of a pr113-*.index.txt."""
    out = []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.split()
            if len(f) > 4:
                out.append((f[0], f[2], f[3], f[4]))
    return out


def read_owner_adds(path):
    """(sample, run, subrun, event, origin, note) rows of pr114-owner-adds.index.txt.

    TAB-separated, unlike the whitespace-split pr113 lists, because the note is
    free text with spaces in it and is the last field."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 5:
                out.append((f[0], f[1], f[2], f[3], f[4],
                            f[5].strip() if len(f) > 5 else ""))
    return out


def scan_sample(docdir):
    """The pr/114 sample -> (rows, notes).

    The base is the whole corrected NCpi0 list + the curated nueCC48 arm.  Both
    are doc pr/113 ROUND 2 -- round 1's NCpi0 list was half this size because of
    the falsy-zero pio_id bug (doc pr/113 sec 10).

    Round 6 adds pr114-owner-adds.index.txt on top.  It does two jobs at once and
    the ORDER of the two matters: an event already in the pr/113 lists keeps the
    row it already had (`setdefault`, so its origin and its place in the sample
    do not move) and gains only a note; an event that is new is appended to the
    sample.  Doing it the other way round -- letting the adds file overwrite --
    would silently rewrite the origin of eight rows that are mid-scan.
    """
    sel = {}
    for r in read_index(os.path.join(docdir, "pr113-ncpi0.index.txt")):
        sel[(r[0], r[3])] = r + ("ncpi0",)
    for r in read_index(os.path.join(docdir, "pr113-nuecc.index.txt")):
        if r[0] == "nuecc48":
            sel.setdefault((r[0], r[3]), r + ("nuecc",))
    notes = {}
    for samp, run, subrun, evt, origin, note in read_owner_adds(
            os.path.join(docdir, "pr114-owner-adds.index.txt")):
        sel.setdefault((samp, evt), (samp, run, subrun, evt, origin))
        if note:
            notes[evt] = note
    rows = [sel[k] for k in sorted(sel, key=lambda k: (k[0], int(k[1])))]
    return rows, notes


def scan_sample_from_index(path):
    """The sample as ONE explicit index file -> (rows, notes).

    Round 7.  `scan_sample()` hard-codes the pr/114 sample (NCpi0 + the nueCC48
    arm + the owner adds) and that is the reproduction record of the scan now on
    port 5017 -- it is not touched.  This is the opt-in path for a DIFFERENT
    sample, e.g. the coverage extension in
    docs/pr/pr114c-beam-em100-unscanned.index.txt.

    Same 5-tuple shape as scan_sample() so bee_build() and the manifest writer
    take either without knowing which.  Column 2 carries the bucket label and
    lands in the manifest's `origin`, which is what the display shows and what a
    later filter (numucc_em / nuecc / other_em) reads.
    """
    rows, seen = [], set()
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.split()
            if len(f) <= 4:
                continue
            samp, origin, run, subrun, evt = f[0], f[1], f[2], f[3], f[4]
            if (samp, evt) in seen:
                continue
            seen.add((samp, evt))
            rows.append((samp, run, subrun, evt, origin))
    rows.sort(key=lambda r: (r[0], int(r[3])))
    return rows, {}


# ---------------------------------------------------------------------------
# Bee: offline link resolution
# ---------------------------------------------------------------------------


def bee_index(beedir, prefer=("em114", "em114b")):
    """event -> (url, round_tag).  Fully offline.

    A Bee set's UUID is minted SERVER-side (wire-cell-bee3/events/views.py:226,
    returned as the bare body at :256), so a new link cannot be produced here.
    But a per-event deep link into an ALREADY-uploaded set can: the `<n>` in
    /event/<n>/ is used directly as the on-disk directory name (views.py:175),
    and that is the same `bee_idx` make_pr_bee.py wrote into the sibling
    .index.txt (:193, :239).  So url + idx are both on disk.

    **`prefer` wins every collision, and that matters more than it looks.** The
    same event appears in many sets, and those sets are DIFFERENT
    RECONSTRUCTIONS -- `prod0813` is an older epoch entirely.  Sorted-glob order
    put `em114` (an 'e') before `prod0813` (a 'p'), so a plain last-wins map sent
    78 of 94 events to a Bee set whose clustering does not match the calib dump
    the display draws beside it.  Clicking through would have shown a different
    answer to the one being scanned -- silently, since both render fine.

    So: build the map from every round first, then let `prefer` overwrite.  A
    Bee link is only trustworthy next to a scan if it is the SAME epoch.

    `prefer` is a SEQUENCE for round 6, and it has to be.  The four events the
    owner added late (169626 174752 347129 394532) are not in the em114 set at
    all, but they ARE in prod0813 and prod0819 -- and 'em114b' sorts before
    'prod0813', so a single-string prefer would have handed exactly those four
    rows a two-epoch-old reconstruction.  That is the same failure the paragraph
    above describes, re-armed by adding a round whose name still starts with 'e'.
    """
    if isinstance(prefer, str):
        prefer = (prefer,)
    out = {}
    urlfiles = sorted(glob.glob(os.path.join(beedir, "*", "*.url")))
    # preferred rounds last, in the order given => the last one wins
    def _rank(u):
        rnd = os.path.basename(os.path.dirname(u))
        return prefer.index(rnd) + 1 if rnd in prefer else 0
    urlfiles = sorted(urlfiles, key=_rank)
    for urlfile in urlfiles:
        with open(urlfile) as fh:
            url = fh.read().strip()
        m = re.search(r"/set/([0-9a-f-]{36})/", url)
        if not m:
            continue
        uuid = m.group(1)
        rnd = os.path.basename(os.path.dirname(urlfile))
        tag = os.path.basename(urlfile)[:-4]
        # prefer the per-arm sidecar, fall back to the round-level sheet: the
        # per-arm one is a 179-byte stub in some rounds (e.g. pr112k).
        for cand in (os.path.join(beedir, rnd, tag + ".index.txt"),
                     os.path.join(beedir, rnd, rnd + ".index.txt")):
            if not os.path.exists(cand):
                continue
            with open(cand) as fh:
                for ln in fh:
                    if ln.startswith("#"):
                        continue
                    f = ln.split()
                    if len(f) >= 2 and f[0].isdigit() and f[1].isdigit():
                        out[f[1]] = ("%s/set/%s/event/%s/" % (BEE_BASE, uuid, f[0]),
                                     "%s/%s" % (rnd, tag))
            break

    # --- second pass: preferred rounds that exist LOCALLY but were never
    # uploaded -----------------------------------------------------------------
    # `bee_round` and `bee_url` answer two different questions, and only the loop
    # above can answer the second.  The 3-D charge cloud is read out of
    # bee/<round>.zip via bee_round (em3d.bee_zip_path); the external link needs a
    # server-minted UUID, which only exists after an upload.  A round built here
    # and not yet uploaded has an index and a zip but no .url, so the loop above
    # skips it entirely and the event keeps whatever OLDER uploaded set happened
    # to contain it -- for the four round-6 events that is prod0813, two epochs
    # back, and it would have rendered silently under a prod0825 skeleton.
    #
    # So: give the local set the round (the cloud is correct) and leave the url
    # empty (there is no honest link to give).  Only PREFERRED rounds get this --
    # a stale local zip from some old sweep must not capture an event.
    for rnd in prefer:
        for idxf in sorted(glob.glob(os.path.join(beedir, rnd, "*.index.txt"))):
            tag = os.path.basename(idxf)[:-len(".index.txt")]
            if not os.path.exists(os.path.join(beedir, rnd, tag + ".zip")):
                continue
            if os.path.exists(os.path.join(beedir, rnd, tag + ".url")):
                continue                      # already resolved, with a real link
            with open(idxf) as fh:
                for ln in fh:
                    if ln.startswith("#"):
                        continue
                    f = ln.split()
                    if len(f) >= 2 and f[0].isdigit() and f[1].isdigit():
                        out[f[1]] = ("", "%s/%s" % (rnd, tag))
    return out


def bee_build(sample, outdir, dry_run=False, only=None):
    """Assemble a fresh Bee set for the scan sample.  LOCAL ONLY -- the upload is
    the owner's step.  One zip per arm, because make_pr_bee.py pairs a ql_root
    with a pr_root and those are per-sample.

    The set is named after `outdir`, so `--bee-build bee/em114b` writes
    em114b-<arm>.zip and looks for its PR products in work-em114b-<arm> first.
    That is what lets round 6 add four events WITHOUT rebuilding (or uploading
    again) the 94-event em114 set the live display is reading right now.

    `only` restricts the build to an explicit event set."""
    os.makedirs(outdir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(outdir))
    by_arm = {}
    for s in sample:
        if only is not None and s[3] not in only:
            continue
        by_arm.setdefault(s[0], []).append(s[3])
    NFIXED = 8      # python, script, -q, qlroot, -p, prroot, -o, zip
    cmds = []
    for arm, evts in sorted(by_arm.items()):
        zip_out = os.path.join(outdir, "%s-%s.zip" % (tag, arm))
        # PR side: the stage-2 arm, NOT prod0825.  `make_pr_bee.py:78` decides
        # whether an event was evaluated by grepping the per-event log for
        # "TaggerCheckNeutrino: selected main cluster", and ONE prod0825 log is
        # truncated -- ncpi0 evt399860's is 22 KB against a 207 KB median and is
        # missing the line, so the builder refuses an event whose dump is
        # perfectly good (main vertex, 9 showers, 47 segments).  The em114 arms
        # were written per-event and carry the line for 94 of 94.  Safe to
        # substitute because the two arms are equal on every physics field
        # (doc pr/114 sec 3), which is what selftest_repro.py checks.
        #
        # The set's OWN arm comes first (work-<tag>-<arm>), so a round that adds
        # events picks up the probe run that was made for them; work-em114-* is
        # the fallback for the original 94, and prod0825 the last resort.  A dir
        # that exists but lacks THIS event is not a match -- work-em114-mcp1k
        # exists and holds the original ten, so an existence test alone would
        # have sent round 6's four events to a root that does not contain them.
        pr_root = None
        for cand in (os.path.join(SX, "work-%s-%s" % (tag, arm)),
                     os.path.join(SX, "work-em114-%s" % arm),
                     os.path.join(SX, "work-%s-%s" % (arm, PROD))):
            if all(os.path.isdir(os.path.join(cand, "pr_evt%s" % e)) for e in evts):
                pr_root = cand
                break
        if pr_root is None:
            pr_root = os.path.join(SX, "work-%s-%s" % (arm, PROD))
        cmd = [sys.executable, os.path.join(SX, "scripts", "bee", "make_pr_bee.py"),
               "-q", os.path.join(SX, "work-%s-grp0825" % arm),
               "-p", pr_root,
               "-o", zip_out] + sorted(evts, key=int)
        cmds.append((arm, zip_out, cmd))

    def shown(p):
        """A path relative to SX only when that is actually shorter -- an outdir
        outside the tree otherwise renders as ../../../../../.. and is unusable
        as a copy-pasteable command."""
        r = os.path.relpath(p, SX)
        return r if not r.startswith("../..") else p

    ok = []
    for arm, zip_out, cmd in cmds:
        print("[bee] %s -> %s (%d events)" % (arm, zip_out, len(cmd) - NFIXED))
        if dry_run:
            print("      " + " ".join(cmd[:NFIXED]) + " <%d events>" % (len(cmd) - NFIXED))
            continue
        rc = subprocess.call(cmd)
        print("      rc=%d%s" % (rc, "" if rc == 0 else "   <-- FAILED"))
        if rc == 0 and os.path.exists(zip_out):
            ok.append((arm, zip_out))
    if dry_run:
        return cmds
    print()
    if not ok:
        print("nothing built.")
        return cmds
    print("NOT uploaded -- that is outward-facing and owner-gated.  To publish:")
    for arm, zip_out in ok:
        print("  ./upload-to-bee.sh %s   # then save the printed URL as %s"
              % (shown(zip_out), shown(zip_out)[:-4] + ".url"))
    print("Then re-run this script (no flags) and every link fills in.")
    return cmds


# ---------------------------------------------------------------------------
# probe parsing
# ---------------------------------------------------------------------------

_TRIPLE = r"\(([-\d.]+),([-\d.]+),([-\d.]+)\)"
RE_CONTENT_HDR = re.compile(
    r"SHOWER_CONTENT shower_id=(-?\d+) node_id=(-?\d+) conn=(-?\d+) pdg=(-?\d+) "
    r"nseg=(\d+) ncls=(\d+) kine_best=([-\d.]+) kine_charge=([-\d.]+) "
    r"kine_range=([-\d.]+) len=([-\d.]+) sum_len=([-\d.]+) start_vtx=(-?\d+) "
    r"start=" + _TRIPLE + r" end=" + _TRIPLE +
    r" dir15=" + _TRIPLE + r" dir100=" + _TRIPLE + r" idx=(-?\d+)")
RE_CONTENT_MEM = re.compile(
    r"SHOWER_CONTENT\s+shower_id=(-?\d+) seg=(-?\d+) cluster=(-?\d+) len=([-\d.]+) "
    r"pdg=(-?\d+) dQ=([-\d.]+) dQdx=([-\d.]+) Efrac=([-\d.]+) E_est=([-\d.]+) "
    r"flags=(\S+) dirsign=(-?\d+)")
RE_SITE = re.compile(r"SHOWER_ABSORB site=(\S+) shower_start_seg=(-?\d+) guard=(-?\d+)")
RE_DIRECT = re.compile(
    r"SHOWER_ABSORB DIRECT site=(\S+) shower_start_seg=(-?\d+) seg=(-?\d+) pdg=(-?\d+)")
RE_SPLICE = re.compile(
    r"SHOWER_ABSORB SPLICE site=(\S+) into_start_seg=(-?\d+) from_start_seg=(-?\d+) "
    r"from_nseg=(-?\d+)")
RE_WALK = re.compile(
    r"SHOWER_ABSORB walk_begin shower_start_seg=(-?\d+) start_vtx=(-?\d+) "
    r"cached_type=(-?\d+) guard_arg=(-?\d+) apply_guard=(-?\d+)")
RE_ADD = re.compile(
    r"SHOWER_ABSORB ADD shower_start_seg=(-?\d+) seg=(-?\d+) pdg=(-?\d+) "
    r"len_cm=([-\d.]+) straight=(-?\d+)")
RE_EXCL = re.compile(
    r"SHOWER_ABSORB EXCLUDE shower_start_seg=(-?\d+) seg=(-?\d+) pdg=(-?\d+)")


def parse_probe_log(path):
    """One event's stdout.log -> the sidecar dict.

    Ordering matters and is used: a `site=` line is emitted immediately BEFORE
    the walk it labels (NeutrinoShowerClustering.cxx:69-74), so the parser keeps
    the last site seen and binds it to the next walk_begin for that shower.  A
    DIRECT line carries its own site and needs no context.

    NOTE the join key: everything is keyed on `node_id` / `seg`, never on
    `shower_id`.  `shower_id` is a per-PROCESS sequential counter -- prod0825 ran
    16 events per process and got 20..33 for the very showers this run numbers
    0..13 (doc pr/114 sec 3).  `node_id` is cluster_id*1000 + segment index and
    is stable, and it is what the dump's showers[].id holds.
    """
    showers, absorb, walks, splices = {}, {}, [], []
    seq_to_node = {}
    cur_site = None
    n_merge = 0

    with open(path, errors="replace") as fh:
        for ln in fh:
            if "SHOWER_MERGE" in ln:
                n_merge += 1
                continue
            if "SHOWER_CONTENT" in ln:
                m = RE_CONTENT_HDR.search(ln)
                if m:
                    g = m.groups()
                    node = int(g[1])
                    seq_to_node[int(g[0])] = node
                    showers[node] = dict(
                        seq_shower_id=int(g[0]), conn=int(g[2]), pdg=int(g[3]),
                        nseg=int(g[4]), ncls=int(g[5]),
                        kine_best=float(g[6]), kine_charge=float(g[7]),
                        kine_range=float(g[8]), length=float(g[9]),
                        sum_len=float(g[10]), start_vtx=int(g[11]),
                        start=[float(g[12]), float(g[13]), float(g[14])],
                        end=[float(g[15]), float(g[16]), float(g[17])],
                        dir15=[float(g[18]), float(g[19]), float(g[20])],
                        dir100=[float(g[21]), float(g[22]), float(g[23])],
                        members=[])
                    continue
                m = RE_CONTENT_MEM.search(ln)
                if m:
                    g = m.groups()
                    node = seq_to_node.get(int(g[0]))
                    if node is None or node not in showers:
                        continue
                    showers[node]["members"].append(dict(
                        seg=int(g[1]), cluster=int(g[2]), length=float(g[3]),
                        pdg=int(g[4]), dQ=float(g[5]), dQdx=float(g[6]),
                        Efrac=float(g[7]), E_est=float(g[8]), flags=g[9],
                        dirsign=int(g[10])))
                continue
            if "SHOWER_ABSORB" not in ln:
                continue
            m = RE_SITE.search(ln)
            if m:
                cur_site = (m.group(1), int(m.group(2)))
                continue
            m = RE_DIRECT.search(ln)
            if m:
                absorb.setdefault(int(m.group(3)), []).append(
                    dict(how="direct", site=m.group(1), shower=int(m.group(2)),
                         pdg=int(m.group(4))))
                continue
            m = RE_SPLICE.search(ln)
            if m:
                splices.append(dict(site=m.group(1), into=int(m.group(2)),
                                    frm=int(m.group(3)), from_nseg=int(m.group(4))))
                continue
            m = RE_WALK.search(ln)
            if m:
                sh = int(m.group(1))
                site = cur_site[0] if (cur_site and cur_site[1] == sh) else None
                walks.append(dict(shower=sh, site=site, start_vtx=int(m.group(2)),
                                  cached_type=int(m.group(3)),
                                  apply_guard=int(m.group(5))))
                cur_site = None
                continue
            m = RE_ADD.search(ln)
            if m:
                sh = int(m.group(1))
                site = walks[-1]["site"] if walks and walks[-1]["shower"] == sh else None
                absorb.setdefault(int(m.group(2)), []).append(
                    dict(how="walk_add", site=site, shower=sh, pdg=int(m.group(3)),
                         length=float(m.group(4)), straight=int(m.group(5))))
                continue
            m = RE_EXCL.search(ln)
            if m:
                sh = int(m.group(1))
                site = walks[-1]["site"] if walks and walks[-1]["shower"] == sh else None
                absorb.setdefault(int(m.group(2)), []).append(
                    dict(how="walk_exclude", site=site, shower=sh,
                         pdg=int(m.group(3))))
    return dict(showers=showers, absorb=absorb, walks=walks, splices=splices,
                n_merge_lines=n_merge)


def parse_probes(arm_roots, outdir):
    os.makedirs(outdir, exist_ok=True)
    n = 0
    for root in arm_roots:
        for log in sorted(glob.glob(os.path.join(root, "pr_evt*", "stdout.log"))):
            evt = re.search(r"pr_evt(\d+)", log).group(1)
            d = parse_probe_log(log)
            if not d["showers"]:
                print("  [warn] %s: no SHOWER_CONTENT lines -- probes were off?" % log)
                continue
            d["event"] = int(evt)
            d["source_log"] = os.path.realpath(log)
            out = os.path.join(outdir, "emprep-evt%s.json" % evt)
            tmp = out + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(d, fh, sort_keys=True)
            os.replace(tmp, out)
            n += 1
    print("wrote %d sidecars to %s" % (n, outdir))
    return n


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def event_stats(dump_path):
    with open(dump_path) as fh:
        d = json.load(fh)
    segs = d.get("segments") or []
    shs = d.get("showers") or []
    em = [s for s in shs if s.get("particle_id") == 11]
    lossy = 0
    for sh in shs:
        j, n = G.join_completeness(sh, segs)
        if j != n:
            lossy += 1
    grp = G.pi0_groups(shs)
    kine = d.get("kine") or {}
    em_e = sorted((s.get("kine_best") or 0.0) for s in em)
    return dict(
        n_seg=len(segs), n_shower=len(shs), n_em=len(em),
        em_max=round(em_e[-1], 1) if em_e else 0.0,
        n_pio_groups=len(grp),
        n_pio_showers=sum(len(v) for v in grp.values()),
        lossy_showers=lossy,
        pio_flag=int(kine.get("kine_pio_flag") or 0),
        kine_pio_mass=round(float(kine.get("kine_pio_mass") or -1), 1),
        has_main=int(d.get("main_vertex") is not None))


# `scan_note` is APPENDED, never inserted: the viewer reads the manifest by
# column name, but a human reading a diff of this .tsv reads by position, and an
# inserted column makes all 94 pre-existing rows look changed when none of them
# are.  Empty for every event the owner did not comment on.
COLS = ["sample", "run", "subrun", "event", "origin", "n_seg", "n_shower", "n_em",
        "em_max", "n_pio_groups", "n_pio_showers", "lossy_showers", "pio_flag",
        "kine_pio_mass", "has_main", "has_probe", "bee_round", "bee_url", "dump",
        "scan_note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "em114-manifest.tsv"))
    ap.add_argument("--docdir", default=os.path.join(SX, "docs", "pr"))
    ap.add_argument("--beedir", default=os.path.join(SX, "bee"))
    ap.add_argument("--prepdir", default=os.path.join(HERE, "emprep"))
    ap.add_argument("--parse-probes", nargs="*", default=None,
                    metavar="ARMROOT", help="stage-2 arms to parse (default: work-em114-*)")
    ap.add_argument("--bee-build", metavar="DIR", default=None,
                    help="assemble a fresh Bee set into DIR (local only, never uploads)")
    ap.add_argument("--bee-dry-run", action="store_true")
    ap.add_argument("--bee-events", default=None,
                    help="comma-separated events to restrict --bee-build to")
    ap.add_argument("--no-bee-index", action="store_true")
    ap.add_argument("--bee-prefer", default="em114,em114b", metavar="R1,R2",
                    help="rounds that win a Bee collision, LAST wins "
                         "(round 7).  Default is the pr/114 pair, so the "
                         "98-event manifest regenerates byte-identically.  A "
                         "new round MUST be appended here or its events keep "
                         "an older epoch's cloud -- see bee_index.__doc__.")
    ap.add_argument("--sample-index", default=None, metavar="INDEX",
                    help="take the sample from ONE index file instead of "
                         "the built-in pr/114 sample (round 7).  Absent => "
                         "identical behaviour to before this option existed.")
    args = ap.parse_args()

    if args.sample_index:
        sample, notes = scan_sample_from_index(args.sample_index)
        print("scan sample: %d events from %s"
              % (len(sample), args.sample_index))
    else:
        sample, notes = scan_sample(args.docdir)
        print("scan sample: %d events (%d carry an owner note)"
              % (len(sample), len(notes)))

    if args.parse_probes is not None:
        roots = args.parse_probes or sorted(glob.glob(os.path.join(SX, "work-em114-*")))
        roots = [r for r in roots if os.path.isdir(r)]
        print("parsing probes from: %s" % ", ".join(os.path.basename(r) for r in roots))
        parse_probes(roots, args.prepdir)

    if args.bee_build:
        only = (set(e.strip() for e in args.bee_events.split(",") if e.strip())
                if args.bee_events else None)
        bee_build(sample, args.bee_build, dry_run=args.bee_dry_run, only=only)

    prefer = tuple(r.strip() for r in args.bee_prefer.split(",") if r.strip())
    links = {} if args.no_bee_index else bee_index(args.beedir, prefer=prefer)
    print("bee index: %d events resolvable offline" % len(links))

    rows, missing = [], []
    for samp, run, subrun, evt, origin in sample:
        dump = os.path.join(SX, "work-%s-%s" % (samp, PROD),
                            "pr_evt%s" % evt, "calib-pr-evt%s.json" % evt)
        if not os.path.exists(dump):
            missing.append("%s-%s" % (run, evt))
            continue
        st = event_stats(dump)
        url, rnd = links.get(evt, ("", ""))
        rows.append(dict(sample=samp, run=run, subrun=subrun, event=evt,
                         origin=origin, bee_url=url, bee_round=rnd,
                         dump=os.path.relpath(dump, SX),
                         scan_note=notes.get(evt, ""),
                         has_probe=int(os.path.exists(os.path.join(
                             args.prepdir, "emprep-evt%s.json" % evt))),
                         **st))
    with open(args.out, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in COLS) + "\n")

    npio = sum(1 for r in rows if r["n_pio_groups"] > 0)
    nloss = sum(1 for r in rows if r["lossy_showers"] > 0)
    nbee = sum(1 for r in rows if r["bee_url"])
    nrnd = sum(1 for r in rows if r["bee_round"])
    nprobe = sum(1 for r in rows if r["has_probe"])
    nnote = sum(1 for r in rows if r["scan_note"])
    print("wrote %d rows to %s" % (len(rows), args.out))
    print("  events with a reconstructed pi0 group : %d" % npio)
    print("  events with a lossy shower join       : %d" % nloss)
    print("  events with a probe sidecar           : %d" % nprobe)
    print("  events with a local Bee cloud (round) : %d" % nrnd)
    print("  events with an offline Bee link (url) : %d" % nbee)
    print("  events with an owner note             : %d" % nnote)
    if missing:
        # NAMED, not counted.  A bare "1 dumps missing" is the kind of line that
        # gets read as noise; the run-event pair is a to-do that survives.
        print("  NOT IN THE DISPLAY -- no calib dump under work-<sample>-%s:" % PROD)
        for m in missing:
            print("      %s" % m)


if __name__ == "__main__":
    main()
