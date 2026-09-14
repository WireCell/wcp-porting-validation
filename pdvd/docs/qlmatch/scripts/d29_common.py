"""doc qlmatch/29 -- shared helpers for the Q/L history scripts (read-only: nothing here writes).

The scorer is IMPORTED, not forked: pdvd/ql_display/ql_agree_score.py's load_truth / build_uid_map / score_event /
cluster_len_cm are the definitions every number here must agree with (d29_attribution.py reproduces the recorded totals
before printing anything else).  Run 039252, events 298567 + 14*idx, idx 0..17, as the scorer hard-codes.
"""
import collections
import json
import os
import sys

import numpy as np

PDVD = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd"
sys.path.insert(0, os.path.join(PDVD, "ql_display"))
import ql_agree_score as S  # noqa: E402

TOL, MIN_LEN, MIN_NPTS = 0.5, 25.0, 100
GOLD = os.path.join(PDVD, "work/ql_labels/wfresc/labels-evt298567.json")
DECISIONS = os.path.join(PDVD, "ql_display/decisions-cathxa")
IDX = range(S.NEVT)


def truth():
    return S.load_truth(GOLD, DECISIONS)


def calib_path(tag, idx):
    return os.path.join(PDVD, "work", f"{S.RUN}_{idx}_{tag}", f"calib-evt{S.evt_of_idx(idx)}.json")


def load(tag, idx):
    with open(calib_path(tag, idx)) as fh:
        return json.load(fh)


def uid_map(ref, tag, idx):
    """ref cluster uid -> tag cluster uid, the scorer's own geometric map (identity when ref == tag)."""
    if ref == tag:
        return {c["uid"]: c["uid"] for c in load(tag, idx)["clusters"]}
    return S.build_uid_map(calib_path(ref, idx), calib_path(tag, idx))


def is_long(c):
    return S.cluster_len_cm(c) >= MIN_LEN or c["npoints"] >= MIN_NPTS


def centroid(c):
    return float(np.median(c["y"])) if c.get("y") else float("nan"), float(np.median(c["z"])) if c.get("z") else float("nan")


def region(c):
    y, z = centroid(c)
    return f"{'top' if c['apa'] == 4 else 'bottom'} y{'+' if y >= 0 else '-'} z{'hi' if z >= 150 else 'lo'}"


class Arm:
    """One tag's calib dump for one event, indexed the way the scorer reads it."""

    def __init__(self, tag, idx, ref="keep"):
        self.tag, self.idx = tag, idx
        self.d = load(tag, idx)
        self.map = uid_map(ref, tag, idx)
        self.clusters = {c["uid"]: c for c in self.d["clusters"]}
        self.flash_time = {f["gid"]: f["time"] for f in self.d["flashes"]}
        self.flash_times = [f["time"] for f in self.d["flashes"]]
        self.bundles = collections.defaultdict(list)
        self.autos = collections.defaultdict(list)
        for b in self.d["bundles"]:
            t = self.flash_time[b["flash_gid"]]
            self.bundles[b["main_cluster"]].append((t, b))
            if b.get("auto_selected"):
                self.autos[b["main_cluster"]].append((t, b))

    def mapped_truth(self, entries):
        return [dict(e, uid=self.map[e["uid"]]) for e in entries if e["uid"] in self.map]

    def score(self, entries):
        """The scorer's score_event on this event with the given (unmapped, ref-space) truth entries."""
        return S.score_event(calib_path(self.tag, self.idx), self.mapped_truth(entries), TOL, MIN_LEN, MIN_NPTS,
                             S.OBJECTIVE_TIERS)

    def positive_status(self, e):
        """An objective truth positive, scorer logic: unmapped / cluster-missing / short / covered / missed."""
        if e["uid"] not in self.map:
            return "unmapped", None
        u = self.map[e["uid"]]
        if u not in self.clusters:
            return "cluster-missing", u
        if not is_long(self.clusters[u]):
            return "short", u
        if any(abs(t - e["time"]) <= TOL for t, _ in self.autos.get(u, ())):
            return "covered", u
        return "missed", u

    def negative_status(self, e):
        """An objective truth negative: 'phantom' when a long auto sits on the mapped cluster at the truth flash."""
        if e["uid"] not in self.map:
            return "unmapped", None
        u = self.map[e["uid"]]
        c = self.clusters.get(u)
        if c is None or not is_long(c):
            return "not-long", u
        if any(abs(t - e["time"]) <= TOL for t, _ in self.autos.get(u, ())):
            return "phantom", u
        return "rejected", u


def key(evt, e):
    return (evt, e["uid"], round(e["time"], 2), e["positive"])
