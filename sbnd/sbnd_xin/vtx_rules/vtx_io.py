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
