"""Shared loading for the neutrino-vertex hand-scan rule work (doc pr/80).

Deliberately its own module rather than an import from `dl_vtx_training/`: that
directory is under active development in a parallel line of work, and the rule
round must not be able to change its numbers by picking up an edit there.  The
*conventions* are copied from it on purpose so the numbers stay comparable:

  truth   = the label's rank-1 pick          (scn_vtx/io.py:69-78)
  correct = Euclidean distance <= 1.0 cm     (taxonomy.py:82, --tol default)

A 3 cm column is reported alongside, as doc pr/78 sec 3 does.  Any other
definition of "correct" makes these numbers incomparable to pr/78 and pr/79 and
must not be introduced quietly.
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                      # sbnd_xin/
LABELS_ROOT = os.path.join(BASE, "vertex_labels")

# The four production scan tags.  `uitest75` and `vtxscan1` are single-event UI
# tests of the same evt10550 that vtxscan-prod0813 already holds, so they add no
# events and are excluded rather than silently deduplicated.
TAGS = ["vtxscan-prod0813", "vtxscan-prod0813-ncpi0",
        "vtxscan-prod0813-mcp1k", "vtxscan-prod0813-mc"]

# doc pr/82 sec 4.1: the same scans carried onto the current-production arms
# (work-*-harv3), plus `vtxscan-harv3-delta` for the events the carry-forward
# declined and a human re-answered.
#
# These are NOT in TAGS, on purpose.  They cover the SAME EVENTS as the
# prod0813 tags above, so a default holding both would hand every unfiltered
# consumer -- baselines.py, selfscan.py score, build_dataset.py -- roughly 922
# labels with duplicate event keys and quietly wrong denominators.  There is no
# error message for that; the numbers just come out wrong.  Callers that want
# the current epoch ask for it: load_labels(tags=vtx_io.TAGS_HARV3).
TAGS_HARV3 = ["vtxscan-harv3-nuecc48", "vtxscan-harv3-ncpi0",
              "vtxscan-harv3-mcp1k", "vtxscan-harv3-delta"]

# doc pr/88: the owner's review of the mcp2k scan (`work-mcp2k-harv3`, the
# 2000-event MCP2025C data sample of doc pr/82 sec 2).  174 events, hand-scanned
# across six port-5017 instalments.
#
# Kept out of TAGS for the SAME reason as TAGS_HARV3, but note the reason is
# different in kind: these are DISJOINT events, not a re-labelling, so pooling
# them would not duplicate an event key.  What it would do is silently change
# every denominator in every pr/78-82 number -- "358/473", "372/473", the
# lockbox fractions -- by enlarging the set those were measured over.  Ask for
# the pool you mean:
#     load_labels(tags=vtx_io.TAGS_HARV3)                    # the old 473
#     load_labels(tags=vtx_io.TAGS_MCP2K)                    # the new 174
#     load_labels(tags=vtx_io.TAGS_HARV3 + vtx_io.TAGS_MCP2K)  # training pool
#
# The 174 are the REVIEWED core.  The other 339 mcp2k labels are auto-accepted
# scanner picks and live in the scan run dirs, not here -- they are admitted to
# training by the doc pr/88 sec 7 gate (39/40 = 97.5%), not by a human.
TAGS_MCP2K = ["vtxscan-mcp2k"]

# doc pr/88 sec 8.6: the 299 gated auto-accept picks, materialised as labels
# so `build_dataset.py` can read them at all.  A SEPARATE tag on purpose --
# these are AI-scanner picks admitted by the sec 7 gate (39/40 = 97.5%), not
# events a human looked at, and every record carries `label_source:
# "ai-scanner"` so the distinction survives being pooled into a --tags line.
# The 40 calibration events are NOT here: the owner labelled them, the scanner
# was wrong on one, and the owner's pick wins.
TAGS_MCP2K_AUTO = ["vtxscan-mcp2k-auto"]

# doc pr/100 / pr/103: the labels carried onto the work-vtx100-base-* arms
# (2026-08-20 epoch, post pr/98-99 flips).  Same events as TAGS_HARV3 (+ the
# mcp2k core/auto/ragree scans), re-anchored -- kept out of TAGS for the same
# duplicate-key reason as TAGS_HARV3.  Ask for it: load_labels(tags=vtx_io.TAGS_VTX100).
TAGS_VTX100 = ["vtxscan-vtx100-nuecc48", "vtxscan-vtx100-ncpi0",
               "vtxscan-vtx100-mcp1k", "vtxscan-vtx100-delta",
               "vtxscan-vtx100-mcp2k"]

TOL = 1.0            # cm, the pr/78/79 headline tolerance
TOL_LOOSE = 3.0      # cm, the column printed beside it


def dist(a, b):
    """3-D distance between two (x, y, z) triples, or None if either is absent."""
    if a is None or b is None or None in a or None in b:
        return None
    return math.dist(a, b)


def xyz(d):
    """(x, y, z) out of any dict that carries those keys, else None."""
    if not d:
        return None
    if d.get("x") is None or d.get("y") is None or d.get("z") is None:
        return None
    return (d["x"], d["y"], d["z"])


def load_labels(tags=None):
    """Every hand-scan label, as a list of dicts sorted by (tag, event).

    Each record adds four derived keys to the on-disk document:
      key        (tag, runNo, subRunNo, eventNo) -- the join key.  NOT eventNo:
                 vtxscan-prod0813-mc mixes two arms and three run/subrun combos
                 in one tag, so eventNo alone is one collision from wrong.
      truth      the rank-1 pick's (x, y, z)
      truth_vid  that pick's vertex_id (None for a manual pick)
      b1         distance from truth to the label's own recorded main_vertex,
                 i.e. the arm the scan was taken on (min_accept = 4.0).
    """
    out = []
    for tag in (tags or TAGS):
        for path in sorted(glob.glob(os.path.join(LABELS_ROOT, tag,
                                                  "labels-evt*.json"))):
            with open(path) as fh:
                doc = json.load(fh)
            picks = sorted(doc.get("picks") or [],
                           key=lambda p: p.get("rank", 99))
            if not picks:
                # The writer refuses to save a pick-less label, so this cannot
                # happen today; treat it as data corruption rather than skip it
                # silently, since a dropped event changes every denominator.
                raise ValueError("label has no picks: %s" % path)
            doc["path"] = path
            doc["tag"] = tag
            doc["key"] = (tag, doc.get("runNo"), doc.get("subRunNo"),
                          doc.get("eventNo"))
            doc["truth"] = xyz(picks[0])
            doc["truth_vid"] = picks[0].get("vertex_id")
            doc["b1"] = dist(doc["truth"], xyz(doc.get("main_vertex")))
            out.append(doc)
    return out


def correct(d, tol=TOL):
    """Distance -> verdict.  `None` (not measurable) is NOT correct."""
    return d is not None and d <= tol


def load_dump(label):
    """The calib dump the label was taken on, from its recorded `source`."""
    with open(label["source"]) as fh:
        return json.load(fh)


# --------------------------------------------------------------- dump helpers
#
# Two schema traps that cost real time if forgotten, so they live behind these
# helpers rather than being open-coded at every call site:
#   * vertices[] has NO flat x/y/z -- position is vertices[i]["fit"]["x"...].
#   * segments[].particle_id is a PDG code EXCEPT for the sentinels 1 (shower,
#     no PID) and 4 (track, no PID).

PID_SENTINELS = (1, 4)


def vertex_xyz(v):
    return xyz(v.get("fit"))


def vertices_by_id(dump):
    return {v["id"]: v for v in dump.get("vertices", [])}


def segments_of_vertex(dump):
    """vertex id -> list of segments attached to it.

    `vertices[].degree` says how many there should be; a mismatch means the
    graph and the dump disagree and is worth reporting, not papering over.
    """
    out = {}
    for s in dump.get("segments", []):
        for key in ("start_vertex_id", "end_vertex_id"):
            vid = s.get(key)
            if vid is not None and vid >= 0:
                out.setdefault(vid, []).append(s)
    return out


def is_pdg(seg):
    """True when particle_id is a real PDG code rather than a sentinel."""
    return seg.get("particle_id") not in PID_SENTINELS


def main_cluster_id(dump):
    """The neutrino-candidate cluster, or None when there is no PR graph."""
    mv = dump.get("main_vertex")
    if mv and mv.get("cluster_id") is not None:
        return mv["cluster_id"]
    for s in dump.get("segments", []):
        if s.get("is_main_cluster"):
            return s["cluster_id"]
    return None


def cluster_length(dump, cid):
    """Total fitted track length (cm) of one cluster."""
    return sum(s.get("length", 0.0) for s in dump.get("segments", [])
               if s.get("cluster_id") == cid)
