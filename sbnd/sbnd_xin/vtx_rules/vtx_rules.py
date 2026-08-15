"""The neutrino-vertex hand-scan decision procedure (doc pr/80).

An ORDERED procedure with explicit tie-breaks and an abstain branch -- not a
scalar score over features.  That distinction is the point: doc pr/79 already
fitted six scalar formulations over the vertex_scoreboard columns and every one
was net-negative end-to-end.  None of them used dQ/dx, Bragg direction, shower
starts or vertex topology, which is what the owner actually scans on and what
this module is built from.

The eight owner heuristics, and where each lives:

  R1  dQ/dx rises toward a stopping end -> the vertex is at the OTHER end.
      => `outgoing_purity`, the strongest single discriminator measured
         (86.5% of truth vertices have every attached decisive track pointing
         away from them, vs 31.9% of the other main-cluster vertices).
  R2  the vertex is generally upstream (low z).      => `pick_lowest_z`, a
      TIE-BREAK only: measured median dz on production's misses is -5.1 cm, so
      z decides near-ties, it does not select.
  R3  muon -> Bragg rise -> Michel at the stopping end.  => `michel_veto`.
  R4  hadronic showers.  NOT IMPLEMENTED -- the dump carries no hadronic-shower
      tag, and `segments[].particle_id` on shower segments is e/gamma-oriented.
      Declared out of scope rather than faked.
  R5  a long muon connects to the neutrino vertex.   => `longest_track_bonus`.
  R6  NC: a short stub with several showers pointing at it. => `shower_convergence`.
  R7  NC with no obvious vertex: the start of the biggest EM shower. => `biggest_shower_start`.
  R8  "only dots" -- the answer does not matter.     => the DOTS branch, which
      abstains.  Scored as neither right nor wrong (see eval_rules.py).

Every threshold is in CFG, was fitted on the DEV half only, and the staged dev
numbers are recorded in doc pr/80 so drift is visible rather than hidden.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtx_geom as G                                             # noqa: E402
import vtx_io                                                    # noqa: E402

CFG = dict(
    dots_total_len=8.0,      # cm: main cluster shorter than this is "just dots"
    dots_max_seg=3.0,        # cm: ...or no single segment longer than this
    track_branch_len=10.0,   # cm: a track this long puts us in the track branch
    min_track_for_bragg=5.0,  # cm: shorter tracks get no Bragg vote
    bragg_ratio=1.3,         # end-window dQ/dx ratio that counts as decisive
    michel_len=8.0,          # cm: a stub this short at a Bragg end is a Michel
    long_track_frac=0.6,     # fraction of the longest track's length that still
                             # counts as "a long track" for R5
    tie_z=5.0,               # cm: candidates within this in z are a real tie
    converge_cos=0.85,       # shower direction alignment for R6
    converge_dis=8.0,        # cm: how close a shower start must be for R6
)


def _cand_vertices(dump, cid):
    """Every PR-graph vertex of the chosen cluster, sorted by id.

    Sorted by id, not by dict order: iterating a pointer- or insertion-ordered
    container is how a result stops being reproducible (CLAUDE.md determinism).
    """
    out = [v for v in dump.get("vertices", []) if v.get("cluster_id") == cid
           and vtx_io.vertex_xyz(v) is not None]
    out.sort(key=lambda v: v["id"])
    return out


def outgoing_purity(seg_of, vid, cfg=CFG):
    """R1.  Of the attached tracks with a decisive Bragg end, what fraction
    have that end AWAY from this vertex?

    Returns (purity, n_decisive).  purity is None when no attached track gives
    a decisive answer -- "no opinion", which must never collapse to 0.0.
    """
    good = dec = 0
    for s in seg_of.get(vid, []):
        if not G.is_track(s) or s.get("length", 0) < cfg["min_track_for_bragg"]:
            continue
        be = G.bragg_end(s, ratio=cfg["bragg_ratio"])
        if be is None:
            continue
        dec += 1
        if be != G.end_name_of_vertex(s, vid):
            good += 1
    return ((good / dec) if dec else None), dec


def michel_veto(dump, seg_of, vid, cfg=CFG):
    """R3.  True when this vertex looks like a muon's stopping end wearing a
    Michel: a long track whose Bragg end is HERE, plus a short stub also here.

    That is the classic decay topology, and the neutrino vertex is at the far
    end of the muon, never at the Michel.
    """
    att = seg_of.get(vid, [])
    long_bragg = any(
        G.is_track(s) and s.get("length", 0) >= cfg["track_branch_len"]
        and G.bragg_end(s, ratio=cfg["bragg_ratio"]) == G.end_name_of_vertex(s, vid)
        for s in att)
    stub = any(s.get("length", 0) < cfg["michel_len"] for s in att)
    return bool(long_bragg and stub and len(att) >= 2)


def _longest_track(dump, cid):
    best = None
    for s in dump.get("segments", []):
        if s.get("cluster_id") != cid or not G.is_track(s):
            continue
        if best is None or s.get("length", 0) > best.get("length", 0):
            best = s
    return best


def _long_track_vertices(dump, cid, cfg=CFG):
    """R5.  Vertex ids sitting on a track at least `long_track_frac` of the
    longest one -- "the long muon connects to the neutrino vertex", without
    betting everything on a single longest segment."""
    lt = _longest_track(dump, cid)
    if lt is None:
        return set(), 0.0
    cut = max(cfg["track_branch_len"], cfg["long_track_frac"] * lt["length"])
    out = set()
    for s in dump.get("segments", []):
        if s.get("cluster_id") != cid or not G.is_track(s):
            continue
        if s.get("length", 0) >= cut:
            out.update(v for v in G.endpoint_vertices(s) if v is not None and v >= 0)
    return out, lt.get("length", 0.0)


def shower_starts(dump, cid):
    """R6/R7.  [(vertex id, shower energy MeV, shower dict)] for this cluster,
    biggest first.  `start_vertex_id` joins to vertices[].id."""
    out = []
    for sh in dump.get("showers", []):
        vid = sh.get("start_vertex_id")
        if vid is None or vid < 0:
            continue
        out.append((vid, sh.get("kine_best") or 0.0, sh))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def _pick_lowest_z(cands):
    """R2, the tie-break.  Ties on z resolve by vertex id so the answer is
    reproducible rather than dependent on enumeration order."""
    return min(cands, key=lambda v: (vtx_io.vertex_xyz(v)[2], v["id"]))


def _result(v, rule, branch, conf, notes, margin=None, runner_up=None):
    p = vtx_io.vertex_xyz(v)
    return dict(decision="answer", x=p[0], y=p[1], z=p[2],
                vertex_id=v["id"], cluster_id=v["cluster_id"],
                branch=branch, rule=rule, confidence=conf,
                margin=margin, runner_up=runner_up, notes=notes)


def _abstain(rule, branch, notes, cand=None):
    out = dict(decision="abstain", x=None, y=None, z=None,
               vertex_id=None, cluster_id=None, branch=branch, rule=rule,
               confidence="unclear", margin=None, runner_up=None, notes=notes)
    if cand is not None:                 # best guess, recorded but not claimed
        p = vtx_io.vertex_xyz(cand)
        out.update(guess_x=p[0], guess_y=p[1], guess_z=p[2],
                   guess_vertex_id=cand["id"])
    return out


def crosscheck(res, dump, cfg=CFG):
    """R9 -- the step a human scanner performs without naming it.

    The display draws the reconstructed neutrino vertex as a star, so the
    scanner is never choosing from a blank picture: they are deciding whether
    to accept the star or move it.  This reproduces that, and it is the only
    confidence signal in the whole procedure that has ground truth on both
    sides, so it is the one that can actually be validated.

    Measured on the dev half: when the independent rules land on the same
    vertex as the reconstruction the answer is right ~89% of the time; when
    they disagree it is right ~35% of the time.  So a disagreement is not a
    reason to prefer either answer -- it is a reason to say "unclear" and put
    the event in front of a human, which is exactly what the owner asked the
    procedure to record.
    """
    if res["decision"] != "answer":
        return res
    reco = vtx_io.xyz(dump.get("main_vertex"))
    if reco is None:
        res["notes"].append("R9: no reconstructed vertex to cross-check against")
        res["agrees_with_reco"] = None
        return res
    d = vtx_io.dist((res["x"], res["y"], res["z"]), reco)
    res["agrees_with_reco"] = bool(d is not None and d <= vtx_io.TOL)
    res["reco_dis"] = d
    if res["agrees_with_reco"]:
        res["notes"].append("R9: agrees with the reconstructed vertex")
        # Deliberately does NOT promote everything to "certain": the tier the
        # rules themselves reached still carries information, and flattening it
        # here would make the precision-vs-coverage curve a single point.
        return res
    res["notes"].append("R9: DISAGREES with the reconstructed vertex by %.1f cm "
                        "-- flagged for a human" % d)
    out = _abstain(res["rule"] + "+R9-disagree", res["branch"], res["notes"])
    out.update(guess_x=res["x"], guess_y=res["y"], guess_z=res["z"],
               guess_vertex_id=res["vertex_id"], agrees_with_reco=False,
               reco_dis=d, margin=res.get("margin"),
               runner_up=res.get("runner_up"))
    return out


def decide(dump, cfg=CFG, crosscheck_reco=True):
    """dump -> decision dict.  See the module docstring for the field meanings.

    `crosscheck_reco=False` gives the rules' INDEPENDENT answer, with no
    knowledge of what the reconstruction chose.  That is the honest measure of
    the rules themselves and is what the doc quotes as "independent"; the
    default True adds the R9 cross-check, which is the mode meant for use.
    """
    res = _decide_independent(dump, cfg)
    return crosscheck(res, dump, cfg) if crosscheck_reco else res


def _decide_independent(dump, cfg=CFG):
    notes = []
    cid = vtx_io.main_cluster_id(dump)
    if cid is None:
        return _abstain("R0-no-main-cluster", "none",
                        ["no main cluster in this dump"])

    cands = _cand_vertices(dump, cid)
    if not cands:
        return _abstain("R0-no-candidates", "none",
                        ["main cluster %s has no positioned vertices" % cid])

    seg_of = vtx_io.segments_of_vertex(dump)
    total_len = vtx_io.cluster_length(dump, cid)
    seg_lens = [s.get("length", 0) for s in dump.get("segments", [])
                if s.get("cluster_id") == cid]
    max_seg = max(seg_lens, default=0.0)
    notes.append("cluster %s: %d vertices, total %.1f cm, longest segment %.1f cm"
                 % (cid, len(cands), total_len, max_seg))

    # ---- R8: "just dots" -------------------------------------------------
    if total_len < cfg["dots_total_len"] or max_seg < cfg["dots_max_seg"]:
        notes.append("R8 dots: nothing long enough to read a direction from")
        return _abstain("R8-just-dots", "dots", notes, _pick_lowest_z(cands))

    long_vids, lt_len = _long_track_vertices(dump, cid, cfg)

    # ---- branch ----------------------------------------------------------
    if lt_len >= cfg["track_branch_len"]:
        return _track_branch(dump, cid, cands, seg_of, long_vids, lt_len,
                             notes, cfg)
    return _shower_branch(dump, cid, cands, seg_of, notes, cfg)


def _track_branch(dump, cid, cands, seg_of, long_vids, lt_len, notes, cfg):
    """Ordered: eliminate stopping ends and Michel ends, then prefer a proved
    outgoing vertex, then take the most upstream survivor.

    The ordering was chosen on the DEV half by comparing five candidate orders
    built from the owner's rules -- (outgoing, low z) won at 40/56, against
    (outgoing, on-long-track, low z) 37/56, (on-long, outgoing, low z) 39/56,
    lowest-z alone 36/56, and (outgoing, most attached length, low z) 32/56.

    R5 "the long muon connects to the vertex" is therefore NOT a filter here,
    which is the one place the data contradicted the heuristic as literally
    stated: on the dev misses the truth vertex sat OFF any long track in 10 of
    15 cases, so filtering on it threw the answer away.  It survives as
    supporting evidence in the notes, not as a cut.
    """
    notes.append("track branch: longest track %.1f cm" % lt_len)

    scored = []
    for v in cands:
        pur, dec = outgoing_purity(seg_of, v["id"], cfg)
        scored.append((v, pur, dec, michel_veto(dump, seg_of, v["id"], cfg)))

    # R1, as an ELIMINATION not a score: a vertex with a stopping end on it is
    # not the neutrino vertex.  Vertices with no opinion survive this step --
    # "unmeasured" is not evidence against.
    keep = [t for t in scored if t[1] is None or t[1] > 0.0]
    if not keep:
        notes.append("R1: every candidate has a Bragg end on it -- no survivor")
        return _abstain("R1-all-stopping", "track", notes,
                        _pick_lowest_z(cands))
    if len(keep) < len(scored):
        notes.append("R1: dropped %d of %d candidates carrying a stopping end"
                     % (len(scored) - len(keep), len(scored)))

    # R3: a Michel-bearing end is disqualified, but only if that leaves someone.
    nomichel = [t for t in keep if not t[3]]
    if nomichel and len(nomichel) < len(keep):
        notes.append("R3 Michel: dropped %d muon-decay end(s)"
                     % (len(keep) - len(nomichel)))
        keep = nomichel

    # R1 again, now as a PREFERENCE: a vertex that proved fully outgoing beats
    # one that merely had no opinion.  This is what makes R1 the primary rule
    # rather than only a veto.
    proved = [t for t in keep if t[1] == 1.0 and t[2] > 0]
    rule = "R1-outgoing"
    if proved:
        if len(proved) < len(keep):
            notes.append("R1: %d candidate(s) proved fully outgoing" % len(proved))
        keep = proved
    else:
        rule = "R2-upstream"
        notes.append("R1: no candidate proved outgoing; upstream prior decides")

    onlong = sum(1 for t in keep if t[0]["id"] in long_vids)
    notes.append("R5: %d of %d survivors sit on a long track (evidence, not a cut)"
                 % (onlong, len(keep)))
    return _finish(keep, "track", rule, notes, cfg)


def _shower_branch(dump, cid, cands, seg_of, notes, cfg):
    notes.append("shower branch: no track above %.0f cm" % cfg["track_branch_len"])
    by_id = {v["id"]: v for v in cands}
    starts = shower_starts(dump, cid)
    starts = [(vid, e, sh) for vid, e, sh in starts if vid in by_id]

    if not starts:
        notes.append("no shower start lands on a vertex of this cluster")
        return _abstain("R7-no-shower-start", "shower", notes,
                        _pick_lowest_z(cands))

    # R6: several showers starting at the same vertex is the NC stub topology,
    # and is a much stronger statement than "the biggest shower starts here".
    counts = {}
    for vid, _e, _sh in starts:
        counts[vid] = counts.get(vid, 0) + 1
    conv = sorted((vid for vid, n in counts.items() if n >= 2),
                  key=lambda vid: (-counts[vid], vid))
    if conv:
        notes.append("R6: %d shower(s) start at vertex %s"
                     % (counts[conv[0]], conv[0]))
        keep = [(by_id[vid], None, 0, False) for vid in conv]
        return _finish(keep, "shower", "R6-shower-convergence", notes, cfg)

    # R7: the start of the biggest EM shower.
    vid, energy, _sh = starts[0]
    notes.append("R7: biggest shower %.0f MeV starts at vertex %s" % (energy, vid))
    margin = None
    if len(starts) > 1:
        margin = energy - starts[1][1]
    conf = "certain" if (margin is None or margin > 0.5 * max(energy, 1.0)) \
        else "likely"
    return _result(by_id[vid], "R7-biggest-shower-start", "shower", conf,
                   notes, margin=margin)


def _finish(keep, branch, rule, notes, cfg):
    """Common tail: R2 tie-break on z, then confidence."""
    if len(keep) == 1:
        return _result(keep[0][0], rule, branch, "certain", notes)

    vs = [t[0] for t in keep]
    winner = _pick_lowest_z(vs)
    zs = sorted(vtx_io.vertex_xyz(v)[2] for v in vs)
    margin = zs[1] - zs[0]
    notes.append("R2 z tie-break over %d candidates, gap %.1f cm"
                 % (len(vs), margin))

    runner = min((v for v in vs if v["id"] != winner["id"]),
                 key=lambda v: (vtx_io.vertex_xyz(v)[2], v["id"]))
    ru = dict(vertex_id=runner["id"], z=vtx_io.vertex_xyz(runner)[2])

    if margin < cfg["tie_z"]:
        # Genuinely ambiguous on the deciding rule: this is the abstain the
        # owner asked for, not a coin flip dressed up as an answer.
        notes.append("R2: gap below %.0f cm -- ambiguous" % cfg["tie_z"])
        out = _abstain(rule + "+R2-ambiguous", branch, notes, winner)
        out["margin"] = margin
        out["runner_up"] = ru
        return out
    return _result(winner, rule + "+R2-z", branch, "likely", notes,
                   margin=margin, runner_up=ru)
