"""doc pr/94 Phase 5 -- pick THE neutrino row out of a multi-row T_tagger.

Before pr/94, `T_tagger` held exactly one entry per event and every consumer
hard-indexed `array()[0]`.  With `nu_per_bundle` on it holds one entry per
in-beam-window flash bundle, and `[0]` silently means "whichever bundle happened
to be enumerated first" -- a real reporting bug, not just a re-baseline.

`primary_index()` reproduces the LEGACY meaning of "the candidate": the longest
selected main activity in the event.  That is exactly what the pre-pr/94
single-winner selector picked (longest untagged in-window main), so a
single-bundle event reports identically to before, and a two-bundle event
reports the same activity the old code would have chosen -- while the other
bundle's row stays available to anyone who wants it.

Files written before pr/94, and files written with the knob off, carry no
identity branches (or carry nu_index == -1); both fall back to row 0.

Usage:

    from pr94_rows import primary_index
    i = primary_index(f['T_tagger'])
    numu = f['T_tagger']['numu_score'].array()[i]
"""


def primary_index(t_tagger):
    """Return the row index of the event's primary neutrino candidate."""
    try:
        keys = set(t_tagger.keys())
    except Exception:                                     # noqa: BLE001
        return 0
    if "nu_index" not in keys:
        return 0                       # pre-pr/94 or knob-off file
    try:
        nu_index = t_tagger["nu_index"].array()
        if len(nu_index) <= 1 or int(nu_index[0]) < 0:
            return 0
        if not {"act_is_selected", "act_length_cm"} <= keys:
            return 0
        sel = t_tagger["act_is_selected"].array()
        length = t_tagger["act_length_cm"].array()
        best_i, best_l = 0, None
        for i in range(len(nu_index)):
            row_l = None
            for s, ln in zip(sel[i], length[i]):
                if int(s) == 1:
                    row_l = float(ln)
                    break
            if row_l is None:
                continue
            if best_l is None or row_l > best_l:
                best_i, best_l = i, row_l
        return best_i
    except Exception:                                     # noqa: BLE001
        return 0


def n_rows(t_tagger):
    """Number of per-bundle rows (1 for a legacy single-candidate file)."""
    try:
        return len(t_tagger["nu_index"].array())
    except Exception:                                     # noqa: BLE001
        return 1
