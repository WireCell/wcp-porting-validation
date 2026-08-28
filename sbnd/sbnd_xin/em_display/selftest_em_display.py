"""doc pr/114 -- headless self-test for the EM / pi0 hand-scan display.

Drives the viewer's own callbacks with no browser: the three regression cases
named in doc pr/114 sec 8 (the 0-of-5 lossy shower, the 43-of-50 one, and the
two-pi0 event whose kine block names a THIRD pairing), plus a mark/impact pass,
a hand-built pi0, the snap, a label round trip and the M13 tag guard.

Run:  python em_display/selftest_em_display.py        (expects 0 failures)
"""
import os, sys, json, shutil, collections

SX = "/nfs/data/1/xqian/toolkit-dev/toolkit/sbnd_xin"
TAG = "selftest114"
shutil.rmtree(os.path.join(SX, "em_labels", TAG), ignore_errors=True)

sys.argv = ["em_display_viewer.py", "--scan-tag", TAG]
sys.path.insert(0, os.path.join(SX, "em_display"))
os.chdir(SX)
import em_display_viewer as V  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print("%-58s %s %s" % (name, "PASS" if cond else "**FAIL**", detail))
    if not cond:
        fails.append(name)


# 94 at round 5, 98 from round 6.  Tied to the manifest rather than to a literal
# so that ADDING events is a data change, with a floor so that LOSING them is
# still a failure.
MANIFEST_EVENTS = sorted(V.MANIFEST.items())


def _has_showers(evt, want):
    """Does this event's probe sidecar hold every one of these shower ids?"""
    p = os.path.join(SX, "em_display", "emprep", "emprep-evt%s.json" % evt)
    if not os.path.exists(p):
        return False
    with open(p) as fh:
        sh = set(int(k) for k in (json.load(fh).get("showers") or {}))
    return set(want) <= sh


check("events loaded from the manifest",
      len(V.LABELS) == len(V.MANIFEST) and len(V.LABELS) >= 98,
      "%d labels, %d manifest rows" % (len(V.LABELS), len(V.MANIFEST)))

# ---- regression case 1: the 0-of-5 lossy shower must not render as empty ----
V.event_select.value = "evt463565"
V.on_event(None, None, "evt463565")
row = None
for i, n in enumerate(V.shower_src.data["node"]):
    if n == 109073:
        row = i
check("evt463565 shower 109073 present in the table", row is not None)
if row is not None:
    joined = V.shower_src.data["joined"][row]
    nseg = V.shower_src.data["nseg"][row]
    note = V.shower_src.data["flag"][row]
    check("  ... membership repaired (not 0)", str(joined) == "5" and nseg == 5,
          "joined=%s nseg=%s" % (joined, nseg))
    check("  ... and the note says the join was lossy", "lossy" in note, note[:60])
check("evt463565 members_of returns 5", len(V.members_of(109073)) == 5,
      str(len(V.members_of(109073))))

# ---- regression case 2: the 43-of-50 shower still draws ----
V.on_event(None, None, "evt84229")
V.state["sel_shower"] = 69134
V.fill_cand_table()
V.draw_arrows()
ax, br, src = V.shower_axis(69134)
check("evt84229 shower 69134 has 50 members", len(V.members_of(69134)) == 50,
      str(len(V.members_of(69134))))
check("  ... axis is unit-length and from the probe",
      abs(V.G.vmag(ax) - 1.0) < 1e-6 and src == "probe", "%s %s" % (br, src))
check("  ... an axis arrow was drawn in >=2 panels",
      sum(1 for k in ("xy", "yz", "xz") if V.arrow_src[k].data["xs"]) >= 2)
check("  ... candidate table is populated", len(V.cand_src.data["sid"]) > 0,
      "%d rows" % len(V.cand_src.data["sid"]))
nsite = sum(1 for x in V.cand_src.data["site"] if x)
check("  ... absorb-site column is filled for the members that were absorbed",
      nsite > 0, "%d of %d rows carry a site" % (nsite, len(V.cand_src.data["site"])))
tiers = [t for t in V.cand_src.data["tier"] if t != "-"]
check("  ... some candidate falls inside a pass-1 tier", len(tiers) > 0,
      "%d in-tier" % len(tiers))

# ---- marking + impact ----
sid = V.cand_src.data["sid"][0]
V.cand_src.selected.indices = [0]
V.mark("out")()
check("marking a segment records it, against the selected shower",
      V.marks_for(V.state["sel_shower"]).get(sid) == "out")
check("  ... impact readout mentions the shower", "kine_charge" in V.impact.text)
check("  ... marked polyline pushed to the out layer",
      bool(V.out_src["xy"].data["xs"]))

# ---- regression case 3: the two-pi0 event, groups kept SEPARATE from kine ----
V.on_event(None, None, "evt21073")
V.mode_group.active = 1
V.on_mode(None, None, 1)
grp = V.G.pi0_groups(V.cur_showers())
check("evt21073 has exactly 2 accepted pi0 groups", len(grp) == 2, str(sorted(grp)))
masses = sorted(round(shl[0].get("pio_mass"), 1) for shl in grp.values())
check("  ... group masses are 111.2 and 127.2", masses == [111.2, 127.2], str(masses))
V.refresh_kine()
txt = V.kine_div.text
check("  ... kine block is labelled a BDT feature, not the pairing",
      "BDT feature" in txt and "NOT the pairing" in txt)
check("  ... the kine block's own 207.25 is shown separately", "207.2" in txt or "207.3" in txt)

# ---- build a pi0 by hand and get a mass ----
V.state["sel_shower"] = 60081
V.on_gamma(1)()
V.state["sel_shower"] = 31023
V.on_gamma(2)()
V.refresh_kine()
e1 = V.shower_energy(60081)
e2 = V.shower_energy(31023)
d1 = V.shower_axis(60081)[0]
d2 = V.shower_axis(31023)[0]
th = V.G.angle_deg(d1, d2)
m = V.G.pi0_mass(e1, e2, th)
check("hand-built gamma pair yields a mass", m is not None and m > 0,
      "E1=%.1f E2=%.1f theta=%.1f m=%.1f" % (e1, e2, th, m))
check("  ... both gamma markers drawn", len(V.gstart_src.data["x"]) == 2)

# back-projection path
V.vtx_mode_group.active = 1
v, how, detail = V.pio_vertex()
check("back-projection returns a verdict", detail.get("verdict") is not None,
      "%s gap=%s" % (detail.get("verdict"), detail.get("gap")))

# ---- back-projection: all three branches must be mirrored --------------
# 49 of the 78 accepted pairs on disk take the one-short branch, where the C++
# re-rays the short gamma and keeps the closest point on the LONG gamma's ray
# instead of the midpoint.  A midpoint-only mirror was wrong by a median 2.7 cm
# (p90 24.9, max 43.4) on exactly those pairs.
import glob as _glob
_br = collections.Counter()
for _p in sorted(_glob.glob(os.path.join(SX, "work-*-prod0825", "pr_evt*",
                                         "calib-pr-evt*.json"))):
    _d = json.load(open(_p))
    _g = V.G.pi0_groups(_d.get("showers") or [])
    if not _g:
        continue
    _anchor = V.G.pt(_d.get("main_vertex"))
    if _anchor is None:
        continue
    for _pid, _shl in _g.items():
        if len(_shl) == 2:
            _br[V.G.pi0_backproject(_shl[0], _shl[1], _d["segments"],
                                    _anchor)["branch"]] += 1
check("back-projection exercises both live branches",
      _br.get("both_long", 0) > 0 and _br.get("one_short", 0) > 0, str(dict(_br)))
check("  ... and the one-short branch is the majority (49/78)",
      _br.get("one_short", 0) == 49 and sum(_br.values()) == 78, str(dict(_br)))

# ---- snap ----
V.gstart_slot.active = 0
V.man_x.value, V.man_y.value, V.man_z.value = "-32.0", "26.0", "367.0"
V.on_snap()
check("snap moved gamma1's start onto a fitted point",
      V.state["gstart"][1] is not None, str(V.state["gstart"][1]))

# ---- save + reload round trip ----
V.em_verdict.active = 1
V.conf_group.active = 0
V.note_in.value = "selftest"
V.on_save()
p = V.label_path("evt21073")
check("label file written", os.path.exists(p), p)
rec = json.load(open(p))
check("  ... record carries the pi0 block", rec.get("pio") is not None)
check("  ... and the reco's own groups alongside the human's",
      len((rec["pio"] or {}).get("reco_groups") or {}) == 2)
check("  ... and the kine_pio_* block, separately",
      "kine_pio_mass" in ((rec["pio"] or {}).get("reco_kine") or {}))
check("  ... provenance recorded", bool(rec.get("source")) and bool(rec.get("arm"))
      and bool(rec.get("probe_sidecar")))
check("  ... and NO pi0 verdict is written (retired in 5d)",
      "verdict" not in (rec.get("pio") or {}), str(sorted((rec.get("pio") or {}))))
V.load("evt21073")
check("reload restores both gamma slots",
      V.state["gamma"][1] == 60081 and V.state["gamma"][2] == 31023,
      str(V.state["gamma"]))

# ---- M13: an implicit tag must refuse to write into a populated dir ----
V.SCAN_TAG_EXPLICIT = False
check("implicit tag refuses to write into a populated tag", not V.write_allowed())
V.SCAN_TAG_EXPLICIT = True
check("explicit tag is consent", V.write_allowed())


# ===========================================================================
# round 3 -- the 3-D view
# ===========================================================================
import math as _m                                                # noqa: E402
import zipfile as _zip                                           # noqa: E402
import em3d as D3                                                # noqa: E402

print()

# ---- camera geometry -------------------------------------------------------
_ortho_ok = _inv_ok = _fit_ok = True
for _i in range(13):
    for _j in range(9):
        _az = -_m.pi + 2 * _m.pi * _i / 12.0
        _el = -1.4 + 2.8 * _j / 8.0
        r, u, f = D3.camera_basis(_az, _el)
        for _a in (r, u, f):
            if abs(sum(v * v for v in _a) - 1.0) > 1e-12:
                _ortho_ok = False
        for _a, _b in ((r, u), (u, f), (f, r)):
            if abs(sum(x * y for x, y in zip(_a, _b))) > 1e-12:
                _ortho_ok = False
        for _p in ((17.0, -3.5, 220.0), (-201.0, 199.0, 0.9), (0.0, 0.0, 0.0)):
            _c = (5.0, -2.0, 250.0)
            (_u, _v, _d), = D3.project([_p], _az, _el, _c)
            _r2 = sum((a - b) ** 2 for a, b in zip(_p, _c))
            if abs(_u * _u + _v * _v + _d * _d - _r2) > 1e-8:
                _inv_ok = False
check("camera basis is orthonormal over a (az, el) grid", _ortho_ok)
check("  ... so u^2 + v^2 + d^2 == |p - centre|^2 exactly", _inv_ok)

# The framing guarantee the whole zoom story rests on: |(u,v)| <= R for EVERY
# camera, so rotating cannot push the event out of frame or shrink it to a dot.
_pts = [(x, y, z) for x in (-180.0, 12.0, 190.0) for y in (-150.0, 0.0, 170.0)
        for z in (5.0, 250.0, 495.0)]
_c, _R = D3.bounding_sphere(_pts)
for _i in range(24):
    for _j in range(7):
        for (_u, _v, _d) in D3.project(_pts, -_m.pi + _m.pi * _i / 12.0,
                                       -1.5 + 3.0 * _j / 6.0, _c):
            if _u * _u + _v * _v > _R * _R + 1e-6:
                _fit_ok = False
check("bounding sphere frames every camera (no rotation can escape it)", _fit_ok,
      "R=%.1f" % _R)
check("  ... and it has a floor so a tiny event is not magnified to noise",
      D3.bounding_sphere([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)])[1] >= 30.0)

# Three presets must reproduce the 2-D panels exactly, or "step back to a view
# you trust" is a lie.
_r, _u, _f = D3.camera_basis(_m.radians(-90.0), 0.0)
check("preset x-z reproduces the X-Z panel (right=+x, up=+z)",
      max(abs(_r[0] - 1), abs(_r[1]), abs(_r[2])) < 1e-9
      and max(abs(_u[0]), abs(_u[1]), abs(_u[2] - 1)) < 1e-9)
_r, _u, _f = D3.camera_basis(0.0, 0.0)
check("preset z-y reproduces the Y-Z panel (right=+y, up=+z)",
      max(abs(_r[0]), abs(_r[1] - 1), abs(_r[2])) < 1e-9
      and max(abs(_u[0]), abs(_u[1]), abs(_u[2] - 1)) < 1e-9)

# ---- the frame the charge cloud is in -- doc pr/114 sec 11 -----------------
# THE blocker of this round, pinned so it cannot rot: the calib dump and the Bee
# zip's own PR layers are the same numbers, which is what licenses drawing
# clustering-global (the corrected frame, doc pr/13) under the skeleton.
_res = []
for _lbl in ("evt21073", "evt84229", "evt463565"):
    _row = V.MANIFEST[_lbl[3:]]
    _idx = D3.bee_event_index(SX, _row, _lbl[3:])
    _zp = D3.bee_zip_path(SX, _row)
    with _zip.ZipFile(_zp) as _z:
        _tf = json.loads(_z.read("data/%d/%d-track_fit-global.json" % (_idx, _idx)))
    _T = list(zip(_tf["x"], _tf["y"], _tf["z"]))
    _d = json.load(open(V.EVENTS[_lbl]))
    _P = [(p["x"], p["y"], p["z"]) for s in _d["segments"]
          for p in (s.get("points") or [])]
    _nn = []
    for _p in _P[::7]:
        _nn.append(min((_p[0] - t[0]) ** 2 + (_p[1] - t[1]) ** 2
                       + (_p[2] - t[2]) ** 2 for t in _T) ** 0.5)
    _nn.sort()
    _res.append((_lbl, _nn[len(_nn) // 2], _nn[-1]))
check("dump fit points ARE the Bee track_fit-global layer (same frame)",
      all(m < 0.001 for _, m, _x in _res),
      "; ".join("%s med %.5f max %.5f cm" % r for r in _res))

# ---- round 6: the same check, but over EVERY row -- the epoch guard ---------
# The three events above pin the FRAME.  This pins the BINDING: bee_round names
# a zip and bee_event_index names a directory inside it, and nothing in the zip
# records which event a directory holds.  So a row pointing at the wrong set, or
# at the right set but the wrong index, renders a different event's charge cloud
# under this event's skeleton -- silently, because both halves are valid data.
#
# Round 6 armed exactly that trap: the four added events are absent from em114
# but present in prod0813 (uploaded, so it has a .url) and prod0819.  'em114b'
# sorts before 'prod0813', so the old single-string `prefer` would have bound
# them to a two-epoch-old reconstruction.  A spatial test catches it where an
# id comparison cannot -- there is no id in the zip to compare.
_worst, _nrows, _skipped = [], 0, []
for _e, _row in sorted(V.MANIFEST.items(), key=lambda kv: int(kv[0])):
    _zp = D3.bee_zip_path(SX, _row)
    _idx = D3.bee_event_index(SX, _row, _e)
    if not _zp or _idx is None or not os.path.exists(_zp):
        _skipped.append(_e)
        continue
    _lbl = "evt%s" % _e
    if _lbl not in V.EVENTS:
        _skipped.append(_e)
        continue
    try:
        with _zip.ZipFile(_zp) as _z:
            _tf = json.loads(_z.read("data/%d/%d-track_fit-global.json" % (_idx, _idx)))
    except (KeyError, OSError):
        _worst.append((_e, float("inf")))
        continue
    _T = list(zip(_tf["x"], _tf["y"], _tf["z"]))
    _d = json.load(open(V.EVENTS[_lbl]))
    _P = [(p["x"], p["y"], p["z"]) for s in _d["segments"]
          for p in (s.get("points") or [])]
    if not _P or not _T:
        _skipped.append(_e)
        continue
    _step = max(1, len(_P) // 40)
    _nn = sorted(min((_p[0] - t[0]) ** 2 + (_p[1] - t[1]) ** 2
                     + (_p[2] - t[2]) ** 2 for t in _T) ** 0.5
                 for _p in _P[::_step])
    _nrows += 1
    _worst.append((_e, _nn[len(_nn) // 2]))
_bad = [w for w in _worst if not (w[1] < 0.01)]
check("EVERY manifest row's Bee cloud is THIS event (round/idx binding)",
      _nrows > 90 and not _bad,
      "%d rows checked, worst median %.5f cm%s"
      % (_nrows, max((w[1] for w in _worst if w[1] != float("inf")), default=-1),
         "" if not _bad else "  BAD: %s" % _bad[:4]))
check("  ... and the four round-6 additions are among them",
      all(_e in {w[0] for w in _worst}
          for _e in ("169626", "174752", "347129", "394532")),
      "skipped: %s" % (_skipped[:6] or "none"))
check("  ... bound to em114b (prod0825), NOT to prod0813/prod0819",
      all(V.MANIFEST[_e]["bee_round"] == "em114b/em114b-mcp1k"
          for _e in ("169626", "174752", "347129", "394532")),
      ", ".join(V.MANIFEST[_e]["bee_round"]
                for _e in ("169626", "174752", "347129", "394532")))

# ---- the cloud loader ------------------------------------------------------
_row = V.MANIFEST["21073"]
_cl = D3.load_bee_cloud(SX, _row, "21073", max_pts=5000)
check("cloud loader honours the point budget exactly",
      _cl is not None and _cl["kept"] == 5000 and _cl["total"] > 5000,
      "kept %s of %s" % (_cl["kept"], _cl["total"]))
check("  ... and returns equal-length columns",
      len({len(_cl[k]) for k in ("x", "y", "z", "q", "cid20")}) == 1)
check("  ... a budget above the total keeps everything",
      D3.load_bee_cloud(SX, _row, "21073", max_pts=10 ** 7)["kept"] == _cl["total"])
check("  ... every cluster survives decimation (proportional, not truncating)",
      len({int(v) for v in D3.load_bee_cloud(SX, _row, "21073",
                                             max_pts=2000)["cid20"]})
      == len({int(v) for v in D3.load_bee_cloud(SX, _row, "21073",
                                                max_pts=10 ** 7)["cid20"]}))
check("  ... a missing zip degrades to None, it does not raise",
      D3.load_bee_cloud(SX, {"bee_round": "nope/nope", "bee_url": "x/event/0/"},
                        "21073") is None)
check("  ... img-global is available but is the RAW frame",
      D3.CLOUD_LAYERS == ["clustering-global", "img-global"])

# ---- the layer contract ----------------------------------------------------
# A 3-D renderer not registered under a layer key would silently ignore the
# checkboxes; one registered under a key nobody lists would be unreachable.
check("every RENDER key is a listed layer",
      set(V.RENDER) == set(V.LAYER_KEYS),
      "render=%s layers=%s" % (sorted(V.RENDER), sorted(V.LAYER_KEYS)))
_f3d_r = set(V.f3d.renderers)
_reg = {r for rs in V.RENDER.values() for r in rs}
_unreg = [r for r in _f3d_r if r not in _reg]
check("every 3-D renderer is layer-controlled except the pick surface",
      _unreg == [V.r_pick], "%d unregistered" % len(_unreg))
V.layer_group.active = [i for i in range(len(V.LAYER_KEYS))]
V.apply_layers(None, None, None)
check("  ... turning every layer on shows exactly one cloud colour mode",
      V.r_cloud_c.visible != V.r_cloud_q.visible)
V.layer_group.active = [i for i, k in enumerate(V.LAYER_KEYS) if k != "cloud"]
V.apply_layers(None, None, None)
check("  ... and the cloud checkbox hides BOTH colour modes",
      not V.r_cloud_c.visible and not V.r_cloud_q.visible)
V.layer_group.active = [0, 1, 2, 3, 4, 6, 7, 8]
V.apply_layers(None, None, None)

# ---- the CustomJS, linted (there is no JS engine in this tree) --------------
_JS_KEYWORDS = set("""break case catch class const continue debugger default
delete do else export extends finally for function if import in instanceof let
new return super switch this throw try typeof var void while with yield of
true false null undefined Math Number Array Float64Array Float32Array Object
JSON String Boolean console window document isNaN parseFloat parseInt NaN
Infinity cb_obj cb_data""".split())


def js_free_names(code):
    """Identifiers a CustomJS body reads without declaring: they must all be
    supplied through `args` or the browser throws a ReferenceError that nothing
    here would ever see.

    Deliberately crude, and it has to handle multi-declarator statements --
    `const rx = -_sa, ry = _ca, rz = 0.0;` declares three names, and a regex that
    only takes the first reports the other two as missing args.  That false
    positive is what this comment exists to stop somebody re-introducing."""
    import re
    body = re.sub(r"//[^\n]*", "", code)
    body = re.sub(r'"[^"]*"|\'[^\']*\'', '""', body)
    decl = set()
    for m in re.finditer(r"\b(?:const|let|var)\b([^;]*)", body):
        seg = m.group(1)
        decl |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*=(?!=)", seg))
        decl |= {n.strip() for n in seg.split(",")
                 if re.fullmatch(r"\s*[A-Za-z_$][\w$]*\s*", n)}
    body = re.sub(r"\.\s*[A-Za-z_$][\w$]*", "", body)
    names = set(re.findall(r"[A-Za-z_$][\w$]*", body))
    return {n for n in names if n not in decl and n not in _JS_KEYWORDS}


_js_all = [("panstart", V.js_panstart), ("rotate", V.js_rotate),
           ("panend", V.js_panend), ("apply", V.js_apply)]
_js_all += [("preset[%d]" % i, cb) for i, cb in enumerate(V.preset_js)]
_missing = []
for _nm, _cb in _js_all:
    for _free in sorted(js_free_names(_cb.code)):
        if _free not in _cb.args:
            _missing.append("%s:%s" % (_nm, _free))
check("every CustomJS free name is supplied through args",
      not _missing, ", ".join(_missing) or "%d handlers" % len(_js_all))
# Test the test: a linter that never fires proves nothing.  It must flag a name
# that is genuinely undefined and must NOT flag a second declarator or a
# property, which is exactly where the first version of it was wrong.
check("  ... and the linter itself catches a genuinely missing name",
      "nosuchthing" in js_free_names("const a = 1, b = 2;\n"
                                     "cam.data.az[0] = a + b + nosuchthing;")
      and "b" not in js_free_names("const a = 1, b = 2; a = b;")
      and "az" not in js_free_names("cam.data.az[0] = 1;"))
_bal_ok = True
for _nm, _cb in _js_all:
    _s = _cb.code
    for _o, _c in "{}", "()", "[]":
        if _s.count(_o) != _s.count(_c):
            _bal_ok = False
check("  ... and their brackets balance", _bal_ok)

# A shape trap worth pinning, stated as it actually is: Bokeh serialises a Python
# dict as {"type":"map", ...}, and bokehjs's `_decode_map` (bokeh.js) returns a
# plain JS object ONLY when every key is a string -- one non-string key and the
# handler receives a real Map instead, so `cfg.foo` is undefined, sizes go NaN,
# and the layer vanishes with no server-side error anywhere to notice.
_dicty = []
for _nm, _cb in _js_all:
    for _k, _val in _cb.args.items():
        for _cand in ([_val] + list(_val) if isinstance(_val, (list, tuple))
                      else [_val]):
            if isinstance(_cand, dict) and not all(isinstance(kk, str)
                                                   for kk in _cand):
                _dicty.append("%s:%s" % (_nm, _k))
check("no CustomJS arg is a non-string-keyed dict (that arrives as a JS Map)",
      not _dicty, ", ".join(sorted(set(_dicty))) or "checked %d handlers"
      % len(_js_all))
# ... and the one table that both mirrors read really is one table.
check("  ... base size/alpha have a single source of truth",
      [V._PT_CFG[s][0] for s in V._PT_SRC] == V._PT_SIZE
      and [V._PT_CFG[s][1] for s in V._PT_SRC] == V._PT_ALPHA
      and V.js_apply.args["ptsize"] is V._PT_SIZE)
# bokehjs's Toolbar._active_change writes the live gesture state to
# `gestures[et].active` and never to `active_drag`, which stays at whatever it
# was configured to.  A guard on active_drag would therefore never fire and
# rotation would fight box-select on every drag -- with nothing on the server to
# show for it.  There is no JS engine here to prove the fix, so pin the shape.
_rot_code = __import__("re").sub(r"//[^\n]*", "", V.js_rotate.code)
check("rotate guards on the LIVE gesture state, not the config property",
      "gestures" in _rot_code and "pan.active" in _rot_code
      and "active_drag" not in _rot_code,
      "(comments stripped; the code names active_drag only to warn against it)")
check("  ... and Python still starts with no drag tool auto-activated",
      V.f3d.toolbar.active_drag is None)
check("the viewer splices em3d's JS, it does not keep a second copy",
      D3.JS_PROJECT in V.js_rotate.code and D3.JS_PROJECT in V.js_apply.code
      and "const rx =" not in open(os.path.join(SX, "em_display",
                                                "em_display_viewer.py")).read())

# ---- 3-D selection resolves to SEGMENTS ------------------------------------
V.on_event(None, None, "evt84229")
V.tap_action.value = V.TAP_SELECT
_sids = V.pick_src.data["sid"]
_idx = [i for i, s in enumerate(_sids) if s == _sids[0]]
V.pick_src.selected.indices = _idx + [len(_sids) - 1]
check("a 3-D box over many points resolves to a handful of segments",
      len(V.selected_cand_ids()) <= 2 and len(_idx) > 2,
      "%d points -> %d segment(s)" % (len(_idx) + 1, len(V.selected_cand_ids())))
check("  ... and the cyan halo shows exactly what would be marked",
      len(V.sel3_src.data["xs3"]) == len(V.selected_cand_ids())
      and len(V.sel_src["xy"].data["xs"]) == len(V.selected_cand_ids()),
      "%d polylines" % len(V.sel3_src.data["xs3"]))
V.state["sel_shower"] = 69134
V.fill_cand_table()
V.mark("in")()
check("  ... and marking works off that 3-D selection",
      V.marks_for(V.state["sel_shower"]).get(_sids[0]) == "in")
V.pick_src.selected.indices = []

# tap in "fill x/y/z" mode must land on a REAL fitted point (a ray needs an
# anchor; this is the 3-D answer to the two-panel tap).
V.tap_action.value = V.TAP_XYZ
V.pick_src.selected.indices = [5]
_want = (V.pick_src.data["x"][5], V.pick_src.data["y"][5], V.pick_src.data["z"][5])
check("tap in fill mode writes a real fitted point into x/y/z",
      abs(float(V.man_x.value) - _want[0]) < 0.06
      and abs(float(V.man_y.value) - _want[1]) < 0.06
      and abs(float(V.man_z.value) - _want[2]) < 0.06,
      "%s %s %s" % (V.man_x.value, V.man_y.value, V.man_z.value))

# ---- round 4: a tap that IS the mark ---------------------------------------
V.state["marks"] = {}
_target = _sids[7]
_hit = [i for i, s in enumerate(_sids) if s == _target][:3]
for want in ("in", "out", None):
    V.tap_action.value = V.TAP_TOGGLE
    V.pick_src.selected.indices = list(_hit)
    got = V.marks_for(V.state["sel_shower"]).get(_target)
    check("toggle tap: %s -> %s" % (want and "next" or "third", want or "cleared"),
          got == want, "seg %s -> %r" % (_target, got))
    # THE re-arm check: Bokeh does not re-fire selected.indices for the same
    # index, so a toggle that did not clear its own selection would be dead
    # after one click.  The loop above only works if _clear_pick ran.
    check("  ... and the selection re-armed for the next tap on the SAME segment",
          list(V.pick_src.selected.indices) == [])
V.tap_action.value = V.TAP_IN
V.pick_src.selected.indices = list(_hit)
check("tap in 'mark IN' mode marks on the click itself",
      V.marks_for(V.state["sel_shower"]).get(_target) == "in")
# Found by the REAL browser, not by this file: a selection left standing from
# the previous action makes the first gesture of the new one a no-op, because
# Bokeh fires selected.indices only on a change and the index list is identical.
V.pick_src.selected.indices = list(_hit)
V.tap_action.value = V.TAP_OUT
check("  ... and switching the action drops the standing selection, so the "
      "next gesture fires",
      list(V.pick_src.selected.indices) == []
      and not V.sel3_src.data["xs3"])
V.tap_action.value = V.TAP_OUT
V.pick_src.selected.indices = list(_hit)
check("  ... and 'mark OUT' overrides it",
      V.marks_for(V.state["sel_shower"]).get(_target) == "out")

# ---- round 4: orbit centre --------------------------------------------------
V.tap_action.value = V.TAP_CENTRE
_span0 = V.f3d.x_range.end - V.f3d.x_range.start
_R0 = V.state["cam_R"]
V.pick_src.selected.indices = [11]
_p = (V.pick_src.data["x"][11], V.pick_src.data["y"][11], V.pick_src.data["z"][11])
check("tap in 'orbit around it' re-centres the camera on that point",
      max(abs(V.state["cam_c"][i] - _p[i]) for i in range(3)) < 0.06,
      "centre %s" % (tuple(round(v, 1) for v in V.state["cam_c"]),))
check("  ... keeping the zoom (span unchanged) and centred on zero",
      abs((V.f3d.x_range.end - V.f3d.x_range.start) - _span0) < 1e-6
      and abs(V.f3d.x_range.start + V.f3d.x_range.end) < 1e-6,
      "span %.1f" % (V.f3d.x_range.end - V.f3d.x_range.start))
check("  ... and NOT rewriting cam_R, which the depth cue normalises by",
      V.state["cam_R"] == _R0)
check("  ... the browser is told the new centre", V.cam_src.data["cx"][0] == _p[0])
# The ON-SCREEN reprojection is the browser's job: pushing cam_src.data fires
# em3d.JS_APPLY, which rewrites u/v in place for every source without shipping
# 25 000 points back.  So do NOT assert on the server's stale u/v columns here --
# assert the invariant the server actually owns, that its projection of the
# clicked point is now the origin, because that is what every later Python-side
# fill will use.
_pu, _pv, _ = V._proj([_p])[0]
check("  ... and the server's own projection now puts it at the origin",
      abs(_pu) < 1e-9 and abs(_pv) < 1e-9, "u=%.2e v=%.2e" % (_pu, _pv))
V.refit_camera()

# ---- round 4: a vertex is clickable, and can never be marked ----------------
V.tap_action.value = V.TAP_PIO
_vx = (V.vtx3_src.data["x"][2], V.vtx3_src.data["y"][2], V.vtx3_src.data["z"][2])
V.vtx3_src.selected.indices = [2]
check("tapping a reconstructed vertex sets the pi0 vertex",
      V.vtx_mode_group.active == 2
      and max(abs(V.state["vtx_manual"][i] - _vx[i]) for i in range(3)) < 0.06,
      "manual=%s" % (tuple(round(v, 1) for v in (V.state["vtx_manual"] or ())),))
check("  ... and the pi0 vertex marker moved there",
      V.piovtx_src.data["x"] and abs(V.piovtx_src.data["x"][0] - _vx[0]) < 0.06)
check("  ... x/y/z boxes agree with the state (no stale-read reentrancy)",
      abs(float(V.man_x.value) - _vx[0]) < 0.06)
_marks_before = json.dumps({str(k): v for k, v in V.state["marks"].items()},
                           sort_keys=True)
V.tap_action.value = V.TAP_IN
V.vtx3_src.selected.indices = [3]
check("a vertex tap can NEVER reach state['marks']",
      json.dumps({str(k): v for k, v in V.state["marks"].items()},
                 sort_keys=True) == _marks_before, str(V.state["marks"]))
check("  ... and box-select cannot pick vertices up at all",
      [r.data_source for r in V._box3.renderers] == [V.pick_src]
      and V.vtx3_src in [r.data_source for r in V._tap3.renderers],
      "box sees %d source(s), tap sees %d"
      % (len(V._box3.renderers), len(V._tap3.renderers)))
V.tap_action.value = V.TAP_SELECT
V.vtx_mode_group.active = 0

# ---- round 4: the reco/hand distinction is DRAWN, not just stored -----------
V.on_event(None, None, "evt84229")
V.state["sel_shower"] = 69134
V.push_polys(V.mem_src, V.members_of(69134), V.mem3_src)
_mem = V.members_of(69134)
V.state["marks"] = {V.state["sel_shower"]: {
    _mem[0]: "out",
    (_sids[0] if _sids[0] not in _mem else _mem[1]): "in"}}
V.refresh_marks()
check("reco membership and your marks are separate, simultaneous layers",
      len(V.mem3_src.data["xs3"]) == len(_mem)
      and len(V.out3_src.data["xs3"]) == 1 and len(V.in3_src.data["xs3"]) == 1,
      "member %d / in %d / out %d" % (len(V.mem3_src.data["xs3"]),
                                      len(V.in3_src.data["xs3"]),
                                      len(V.out3_src.data["xs3"])))
_order = [r.data_source for r in V.f3d.renderers if r in V.RENDER.get("mark", [])
          + V.RENDER.get("member", []) + V.RENDER.get("select", [])]
check("  ... and the widest halo is UNDERNEATH, so a mark cannot erase the "
      "reco band",
      _order.index(V.sel3_src) < _order.index(V.in3_src) < _order.index(V.mem3_src),
      " -> ".join({id(V.sel3_src): "select", id(V.in3_src): "in",
                   id(V.out3_src): "out", id(V.mem3_src): "member",
                   id(V.g1mem3_src): "g1", id(V.g2mem3_src): "g2"}[id(s)]
                  for s in _order))
_srcs = [r.data_source for r in V.f3d.renderers]
_iseg = _srcs.index(V.seg3_src)
_dash = [i for i, r in enumerate(V.f3d.renderers)
         if getattr(r.glyph, "line_dash", None) == "dashed"]
check("  ... the dashed repeat of your mark is drawn ON TOP of the segment",
      len(_dash) == 2 and min(_dash) > _iseg
      and {_srcs[i] for i in _dash} == {V.in3_src, V.out3_src},
      "segment at %d, dashes at %s" % (_iseg, _dash))
check("dim is OFF by default (hiding markable segments went backwards once)",
      V.dim_toggle.active is False and set(V.seg3_src.data["a"]) == {0.95})
# refresh_dim runs on EVERY mark, and since round 4 every tap is a mark.
# Assigning .data there re-serialises every polyline in all four segment sources
# (~7 400 coordinates) and ships them down the ssh tunnel to change one list of
# floats.  Static guard, in the spirit of the JS lint: it must patch, not assign.
import inspect as _insp                                             # noqa: E402
_dimsrc = _insp.getsource(V.refresh_dim)
_dimcode = __import__("re").sub(r"#[^\n]*", "", _dimsrc.split('"""')[-1])
check("  ... and marking patches the alpha column, it does not re-push geometry",
      ".patch(" in _dimcode and "m.data =" not in _dimcode
      and "continue" in _dimcode,
      "patch=%s assign=%s early-return=%s"
      % (".patch(" in _dimcode, "m.data =" in _dimcode, "continue" in _dimcode))
V.dim_toggle.active = True
V.refresh_dim()
_a = dict(zip(V.seg3_src.data["sid"], V.seg3_src.data["a"]))
check("  ... and with it on, members stay bright and the rest fade",
      _a[_mem[0]] > 0.9 and min(_a.values()) < 0.2,
      "members %.2f, faintest %.2f" % (_a[_mem[0]], min(_a.values())))
V.dim_toggle.active = False
V.refresh_dim()

# ---- round 4: a reopened label draws its own marks --------------------------
V.state["marks"] = {V.state["sel_shower"]: {_mem[0]: "out"}}
V.state["sel_shower"] = 69134
V.em_verdict.active = 1
V.on_save()
V.on_event(None, None, "evt21073")
V.on_event(None, None, "evt84229")
check("re-opening a labelled event DRAWS the marks it restores",
      V.marks_for(V.state["sel_shower"]).get(_mem[0]) == "out"
      and len(V.out3_src.data["xs3"]) == 1
      and len(V.mem3_src.data["xs3"]) == len(_mem),
      "marks=%s out-halos=%d member-halos=%d"
      % (V.state["marks"], len(V.out3_src.data["xs3"]),
         len(V.mem3_src.data["xs3"])))
check("  ... and the shower table row is selected again",
      list(V.shower_src.selected.indices) != [])

# ---- round 4: the neutrino-candidate cloud filter --------------------------
V.on_event(None, None, "evt84229")
_cl_on = V.state["cloud"]
check("the cloud defaults to the neutrino candidate, not the whole readout",
      V.cloud_scope.active == 0 and _cl_on["filtered"]
      and _cl_on["candidate"] < _cl_on["total"],
      "%d of %d points, %d of %d clusters (ids %s)"
      % (_cl_on["candidate"], _cl_on["total"], _cl_on["ncluster_kept"],
         _cl_on["ncluster"], _cl_on["kept_ids"]))
check("  ... the readout names all three numbers, not two",
      "of %s clusters" % _cl_on["ncluster"] in V.cloud_div.text
      and "{:,}".format(_cl_on["total"]) in V.cloud_div.text)
V.cloud_scope.active = 1
V.on_cloud_opt(None, None, 1)
_cl_off = V.state["cloud"]
check("  ... 'all clusters' really does put the cosmics back",
      not _cl_off["filtered"] and _cl_off["candidate"] == _cl_off["total"]
      and _cl_off["total"] == _cl_on["total"],
      "%d points" % _cl_off["candidate"])
V.cloud_scope.active = 0
V.on_cloud_opt(None, None, 0)
# The filter must run BEFORE decimation, or the budget eats the candidate twice.
V.cloud_max.value = "10000"
V.on_cloud_opt(None, None, None)
_cl_small = V.state["cloud"]
check("  ... and the budget is spent on the CANDIDATE, not on the whole cloud",
      _cl_small["kept"] == min(10000, _cl_small["candidate"]),
      "kept %d of candidate %d (total %d)"
      % (_cl_small["kept"], _cl_small["candidate"], _cl_small["total"]))
V.cloud_max.value = "25000"
V.on_cloud_opt(None, None, None)
V.tap_action.value = V.TAP_SELECT

# ---- round 4: framing on the shower rather than on the whole TPC -----------
# EM mode explicitly: focus_points() follows the MODE (the selected shower in EM,
# the two assigned gammas in pi0), and an earlier check left the app in pi0.
V.mode_group.active = 0
V.on_event(None, None, "evt64591")
V.fit_mode.active = 0
V.refit_camera()
_R_all = V.state["cam_R"]
V.fit_mode.active = 2
V.on_shower_select(None, None, [0])          # the biggest shower, row 0
_R_shw = V.state["cam_R"]
check("'frame the shower' fills the panel with the thing being judged",
      _R_shw < 0.5 * _R_all,
      "R %.0f cm over all reco -> %.0f cm over the shower" % (_R_all, _R_shw))
check("  ... and it is NOT the default: a table click must not move the camera "
      "under the scanner",
      V.fit_mode.labels[0] == "frame the reco")
V.state["sel_shower"] = None
V.refit_camera()
check("  ... with nothing selected it falls back to the reco, not to nothing",
      abs(V.state["cam_R"] - _R_all) < 1e-9, "R=%.0f" % V.state["cam_R"])
V.fit_mode.active = 0

# ---- round 4: the filter, over the WHOLE sample ----------------------------
# Two things that a single-event check cannot say, and that the doc quotes:
#   (a) the numpy grid hash and scipy's cKDTree return the SAME kept set.  The
#       grid is the fallback when scipy is absent, and a fallback that quietly
#       disagreed would change what is on screen depending on the box.
#   (b) the filter does not eat the charge of the thing being scanned -- the
#       largest shower's fitted points stay covered by a KEPT cluster.
# Slow (~40 s: it parses all 94 clouds) but this is the claim the round rests on.
import time as _time                                                # noqa: E402
try:
    from scipy.spatial import cKDTree as _KD                        # noqa: E402
except ImportError:
    _KD = None
_zips, _same, _cov, _covall, _frac = {}, 0, [], [], []
_t0 = _time.time()
_rows = [l.rstrip("\n").split("\t")
         for l in open(os.path.join(SX, "em_display", "em114-manifest.tsv"))]
_MAN = [dict(zip(_rows[0], r)) for r in _rows[1:] if len(r) == len(_rows[0])]
for _r in _MAN:
    _d = json.load(open(os.path.join(SX, _r["dump"])))
    _zp = D3.bee_zip_path(SX, _r)
    _zi = D3.bee_event_index(SX, _r, _r["event"])
    import zipfile as _zf                                           # noqa: E402
    _z = _zips.setdefault(_zp, _zf.ZipFile(_zp))
    _c = json.loads(_z.read("data/%d/%d-clustering-global.json" % (_zi, _zi)))
    _cid = _c.get("real_cluster_id") or _c["cluster_id"]
    _pts = [(p["x"], p["y"], p["z"]) for s in (_d.get("segments") or [])
            for p in (s.get("points") or []) if p.get("x") is not None]
    _ts = _d.get("track_shower") or {}
    _pts += [(_ts["x"][i], _ts["y"][i], _ts["z"][i])
             for i in range(len(_ts.get("x") or []))]
    _keep, _ = D3.candidate_clusters(_c["x"], _c["y"], _c["z"], _cid, _pts)
    _kg, _ = D3.candidate_clusters(_c["x"], _c["y"], _c["z"], _cid, _pts,
                                   force_grid=True)
    _same += (_keep == _kg)
    _frac.append(sum(1 for q in _cid if q in _keep) / max(1, len(_cid)))
    # Coverage of the largest shower's own fitted points.  NOTE the join key:
    # segments carry the shower's `id`, not its `shower_id` field (`shower_id` on
    # a segment is 78025 while the shower's own `shower_id` is 4).  Joining on the
    # wrong one silently yields an empty member list and a vacuous check.
    _sh = max((_d.get("showers") or []), key=lambda s: s.get("kine_charge") or 0,
              default=None)
    if _sh is not None and _KD is not None:
        _seg = [s for s in (_d.get("segments") or [])
                if s.get("shower_id") == _sh.get("id")]
        _q = [(p["x"], p["y"], p["z"]) for s in _seg for p in (s.get("points") or [])
              if p.get("x") is not None]
        import numpy as _np                                         # noqa: E402
        _tree = _KD(_np.column_stack([_c["x"], _c["y"], _c["z"]]))
        _cida = _np.asarray(_cid)
        if _q:
            _dd, _ii = _tree.query(_np.asarray(_q), distance_upper_bound=2.0)
            _ok = _np.isfinite(_dd)
            if _ok.any():
                _cov.append(float(_np.isin(_cida[_ii[_ok]], list(_keep)).mean()))
        # and of EVERY reco point, which is the stronger statement
        _dd, _ii = _tree.query(_np.asarray(_pts), distance_upper_bound=2.0)
        _ok = _np.isfinite(_dd)
        _covall.append(float(_np.isin(_cida[_ii[_ok]], list(_keep)).mean())
                       if _ok.any() else 1.0)
check("scipy and the numpy fallback agree on the kept clusters, 94/94",
      _same == len(_MAN), "%d/%d agree (%.0f s)" % (_same, len(_MAN),
                                                    _time.time() - _t0))
_frac.sort()
check("  ... and the candidate is a minority of the cloud on every event",
      max(_frac) < 0.75,
      "kept median %.3f p90 %.3f max %.3f" % (_frac[len(_frac) // 2],
                                              _frac[int(0.9 * len(_frac))],
                                              max(_frac)))
check("  ... while the largest shower's own charge stays covered",
      bool(_cov) and min(_cov) > 0.95,
      ("largest shower: min %.4f median %.4f over %d events"
       % (min(_cov), sorted(_cov)[len(_cov) // 2], len(_cov))) if _cov
      else "NOT MEASURED (no scipy) -- the claim in the doc is unchecked here")
check("  ... and so does every reco point of the candidate, not just that shower",
      bool(_covall) and min(_covall) > 0.98,
      ("all reco: min %.4f median %.4f"
       % (min(_covall), sorted(_covall)[len(_covall) // 2])) if _covall
      else "NOT MEASURED (no scipy)")
for _z in _zips.values():
    _z.close()

# A stale selection would make the next "mark IN" hit a segment of the PREVIOUS
# event: replacing a CDS's .data does not clear its .selected.
V.pick_src.selected.indices = [3, 4, 5]
V.on_event(None, None, "evt21073")
check("switching events clears the 3-D selection",
      list(V.pick_src.selected.indices) == [] and V.selected_cand_ids() == [],
      str(list(V.pick_src.selected.indices)))
V.on_event(None, None, "evt84229")

# ---- camera round trip -----------------------------------------------------
V.on_camtxt(None, None, "1.2345,-0.5000")
check("the browser's panend report sets the server-side camera",
      abs(V.state["cam"][0] - 1.2345) < 1e-9
      and abs(V.state["cam"][1] + 0.5) < 1e-9)
check("  ... and the readout shows it", "camera az" in V.cam_div.text)
V.on_camtxt(None, None, "not-a-camera")
check("  ... a malformed report is ignored, not crashed on",
      abs(V.state["cam"][0] - 1.2345) < 1e-9)
V.on_save()
_rec = json.load(open(V.label_path("evt84229")))
check("the saved record carries the view the judgement was made from",
      (_rec.get("camera") or {}).get("cloud") == "clustering-global"
      and _rec["camera"]["az_deg"] is not None, str(_rec.get("camera")))

# ---------------------------------------------------------------------------
# round 5: marks belong to a shower, views are brushed together, showers have
# colours, and the acceptance plot is scaled to what is being compared
# ---------------------------------------------------------------------------
print()
import re  # noqa: E402
V.on_event(None, None, "evt64591")
_rows = list(V.shower_src.data["node"])


def pick(node):
    V.on_shower_select(None, None, [_rows.index(node)])


# --- a mark belongs to a shower ---------------------------------------------
V.view_tabs.active = 1
pick(83044)
check("a table click brings the 3-D view up", V.view_tabs.active == 0)
# Read from the source, not from the live widget: earlier checks in this file
# drive fit_mode themselves, so the running value says nothing about the DEFAULT.
_fmblk = (open(os.path.join(SX, "em_display", "em_display_viewer.py")).read()
          .split("fit_mode = RadioButtonGroup(")[1].split(")")[0])
check("  ... and 'frame the shower' is the default that makes it useful",
      "active=2" in _fmblk.replace(" ", ""), _fmblk.strip()[:70])
V.fit_mode.active = 2
_R0 = V.state["cam_R"]
pick(78025)
check("  ... so picking a small shower actually re-frames onto it",
      V.state["cam_R"] < _R0, "R %.1f -> %.1f" % (_R0, V.state["cam_R"]))
pick(83044)
V.apply_marks([60008], "in")
pick(78025)
V.apply_marks([59007], "out")
check("two showers hold their own marks in one event",
      V.marks_for(83044) == {60008: "in"} and V.marks_for(78025) == {59007: "out"},
      "%s / %s" % (V.marks_for(83044), V.marks_for(78025)))
check("  ... and the halos show only the shower being scanned",
      len(V.out3_src.data["xs3"]) == 1 and len(V.in3_src.data["xs3"]) == 0,
      "in=%d out=%d" % (len(V.in3_src.data["xs3"]),
                        len(V.out3_src.data["xs3"])))
check("  ... while the mark list names every shower that has one",
      "83044" in V.marks_div.text and "78025" in V.marks_div.text)
_before = json.dumps({str(k): v for k, v in V.state["marks"].items()},
                     sort_keys=True)
V.state["sel_shower"] = None
V.apply_marks([24003], "in")
check("a mark with NO shower selected is refused, not filed somewhere",
      "pick a shower" in V.save_note.text
      and json.dumps({str(k): v for k, v in V.state["marks"].items()},
                     sort_keys=True) == _before)

# --- the record ---------------------------------------------------------
pick(83044)
V.em_verdict.active = 0
V.on_save()
_r = json.load(open(V.label_path("evt64591")))["em"]
check("the record keys marks by shower", _r["marks_by_shower"] ==
      {"78025": {"59007": "out"}, "83044": {"60008": "in"}},
      str(_r["marks_by_shower"]))
check("  ... and writes NO flat map that could disagree with it",
      "marks" not in _r, str(sorted(_r)))
_d = _r["marks_detail"]["83044"]
check("  ... and carries the gate numbers a later fit needs",
      _d["marked"]["60008"]["tier"] == 2
      and abs(_d["marked"]["60008"]["dist"] - 84.4) < 0.1
      and abs(_d["marked"]["60008"]["angle"] - 10.8) < 0.1
      and _d["member_span"]["n"] == 17,
      json.dumps(_d["marked"]["60008"]))

# --- a round-4 file still reads, and says so --------------------------------
_legacy = dict(json.load(open(V.label_path("evt64591"))))
_legacy["em"] = dict(shower=78025, marks={"60008": "in"}, verdict="correct",
                     reco=_r["reco"])
with open(V.label_path("evt64591"), "w") as _fh:
    json.dump(_legacy, _fh)
V.on_event(None, None, "evt21073")
V.on_event(None, None, "evt64591")
check("a round-4 flat-mark label still loads",
      V.marks_for(78025) == {60008: "in"}, str(V.state["marks"]))
check("  ... attributed to the shower the file named, and SAID so",
      V.state["legacy_marks"] == (78025, 1)
      and "predates per-shower marks" in V.banner.text)

# --- exclusion --------------------------------------------------------------
pick(83044)
_n0 = len(V.cand_src.data["sid"])
_opt = [o for o in V.excl_choice.options if o.startswith("78025")][0]
V.on_excl(None, None, [_opt])
_ex = V.excluded_segments()
_a = V.seg3_src.data["a"]
_s = V.seg3_src.data["sid"]
check("excluding a shower dims exactly its segments in 3-D",
      _ex and {_a[i] for i, q in enumerate(_s) if q in _ex} == {0.05}
      and {_a[i] for i, q in enumerate(_s) if q not in _ex} == {0.95},
      str(sorted(_ex)))
check("  ... and drops them from the candidate table",
      len(V.cand_src.data["sid"]) == _n0 - len(_ex),
      "%d -> %d, %d excluded" % (_n0, len(V.cand_src.data["sid"]), len(_ex)))
V.on_excl(None, None, [])

# --- linked brushing --------------------------------------------------------
pick(83044)
_i = list(V.cand_src.data["sid"]).index(60008)
V.cand_src.selected.indices = [_i]
check("a candidate-table click lights the acceptance plot and the 3-D view",
      [V.cand_pt_src.data["sid"][j] for j in V.cand_pt_src.selected.indices]
      == [60008]
      and {V.pick_src.data["sid"][j]
           for j in V.pick_src.selected.indices} == {60008})
check("  ... without doubling the selection halo",
      V.selected_cand_ids() == [60008]
      and len(V.sel_src["xy"].data["xs"]) == 1,
      "%s / %d" % (V.selected_cand_ids(), len(V.sel_src["xy"].data["xs"])))
V.cand_src.selected.indices = []
_j = list(V.cand_pt_src.data["sid"]).index(83050)
V.cand_pt_src.selected.indices = [_j]
check("  ... and the acceptance plot drives them back the other way",
      [V.cand_src.data["sid"][k] for k in V.cand_src.selected.indices] == [83050])
V.cand_pt_src.selected.indices = []

# --- one colour per shower --------------------------------------------------
_own = V.owner_map()
_bad = {}
for _k, _sid in enumerate(V.seg_src["xy"].data["sid"]):
    _bad.setdefault(_own.get(_sid), set()).add(V.seg_src["xy"].data["c"][_k])
check("every segment of a shower is drawn in ONE colour",
      all(len(v) == 1 for k, v in _bad.items() if k is not None),
      str({k: v for k, v in _bad.items() if k is not None and len(v) > 1}))
check("  ... the two pi0 gammas are different HUES, not two shades of one",
      V.shower_color(83044) != V.shower_color(78025)
      and V.shower_color(83044)[:4] != V.shower_color(78025)[:4],
      "%s vs %s" % (V.shower_color(83044), V.shower_color(78025)))
check("  ... segments no shower claims stay neutral",
      V.shower_color(None) == V.NO_SHOWER_COLOR)
check("  ... and the shower table carries the colour key",
      len(V.shower_src.data["color"]) == len(V.shower_src.data["node"])
      and V.shower_src.data["color"][0] == V.shower_color(
          V.shower_src.data["node"][0]))
V.seg_color_mode.active = 1
V.on_seg_color_mode(None, None, 1)
check("  ... and the per-segment mode still works",
      len({V.seg_src["xy"].data["c"][k]
           for k, q in enumerate(V.seg_src["xy"].data["sid"])
           if _own.get(q) == 83044}) > 1)
V.seg_color_mode.active = 0
V.on_seg_color_mode(None, None, 0)

# --- the acceptance plot is about the members now ---------------------------
pick(83044)
_mem = set(V.members_of(83044))
_plotted = [q for q in V.cand_pt_src.data["sid"] if q in _mem]
check("the shower's own seed segment is ON the acceptance plot",
      len(_plotted) == len(_mem) and 83044 in _plotted,
      "%d of %d members plotted" % (len(_plotted), len(_mem)))
check("  ... at angle 0, which is what a zero-length start vector means",
      abs(V.cand_pt_src.data["y"][list(V.cand_pt_src.data["sid"]).index(83044)])
      < 1e-9)
check("  ... members are drawn as squares, the rest as circles",
      {V.cand_pt_src.data["mk"][k]
       for k, q in enumerate(V.cand_pt_src.data["sid"]) if q in _mem} == {"square"}
      and "circle" in set(V.cand_pt_src.data["mk"]))
V.apply_marks([60008], "in")
_xh, _yh = V.acc.x_range.end, V.acc.y_range.end
check("the plot is scaled to the comparison, not to the 220x90 gate box",
      _xh < 220 and _yh < 90 and _xh > 84.4,
      "x 0..%.1f  y 0..%.1f" % (_xh, _yh))
check("  ... and anything cropped out is counted, not silently dropped",
      V.state["acc_hidden"] > 0
      and "outside the zoomed range" in V.cmp_div.text,
      "hidden=%s" % V.state["acc_hidden"])
check("  ... the readout compares the mark with the members in words",
      "already in shower 83044" in V.cmp_div.text
      and "angle <b>inside</b> the member spread" in V.cmp_div.text
      and "1.6&times;" in V.cmp_div.text
      and "tier <b>2</b>" in V.cmp_div.text,
      re.sub("<[^>]+>", "", V.cmp_div.text)[:200])
V.acc_zoom.active = False
V.fill_cand_table()
check("  ... and zoom off restores the full gate box",
      (V.acc.x_range.end, V.acc.y_range.end) == (220, 90))
V.acc_zoom.active = True

# --- a mark you cannot see is a trap -----------------------------------------
V.fit_mode.active = 2
pick(83044)
V.state["marks"][83044] = {}          # start from the members alone
V.refit_camera()
_R_mem = V.state["cam_R"]
_i = list(V.cand_src.data["sid"]).index(60008)
V.cand_src.selected.indices = [_i]
V.mark("in")()
check("marking off-frame says so instead of hiding the halo",
      "outside the current view" in V.save_note.text,
      re.sub("<[^>]+>", "", V.save_note.text)[:90])
V.refit_camera()
check("  ... and refit then reaches it, because the frame counts your marks",
      V.state["cam_R"] > _R_mem + 5,
      "R %.1f -> %.1f" % (_R_mem, V.state["cam_R"]))
check("  ... which is what focus_points now returns",
      any(abs(p[0] + 172.4) < 1.0 for p in V.focus_points()),
      "%d points" % len(V.focus_points()))

# --- a marked segment is repainted into its new shower ----------------------
V.on_event(None, None, "evt64591")
V.seg_color_mode.active = 0
# Start from no marks: an earlier block in this file left a label on disk, and
# load_label restores it -- 60008 would already be painted into a shower.
V.state["marks"] = {}
V.refresh_marks()
_rowsC = list(V.shower_src.data["node"])


def _col(sid):
    d = V.seg3_src.data
    return d["c"][list(d["sid"]).index(sid)]


V.on_shower_select(None, None, [_rowsC.index(83044)])
# 60008 is a one-segment shower of its OWN in the reco, so it starts in its own
# colour -- neutral grey is for segments no shower claims at all, like the
# 186 cm track 17002 (shower_id -1).
check("a segment starts in the colour of the shower the RECO gave it",
      _col(60008) == V.shower_color(60008) != V.shower_color(83044),
      "%s vs its own %s" % (_col(60008), V.shower_color(60008)))
check("  ... and a segment no shower claims is neutral",
      _col(17002) == V.NO_SHOWER_COLOR, _col(17002))
V.apply_marks([60008], "in")
check("  ... marking it IN repaints it in that shower's colour",
      _col(60008) == V.shower_color(83044) == _col(83050),
      "%s vs shower %s" % (_col(60008), V.shower_color(83044)))
V.apply_marks([83050], "out")
check("  ... and marking a member OUT drops it back to neutral",
      _col(83050) == V.NO_SHOWER_COLOR, _col(83050))
V.apply_marks([60008], None)
V.apply_marks([83050], None)
check("  ... unmarking restores the reconstruction's own colouring",
      _col(60008) == V.shower_color(60008)
      and _col(83050) == V.shower_color(83044),
      "%s / %s" % (_col(60008), _col(83050)))
_csrc = _insp.getsource(V.refresh_colors)
check("  ... and the repaint patches the colour column, not the geometry",
      ".patch(" in _csrc and "m.data =" not in _csrc and "continue" in _csrc)

# --- IN against two showers is a contradiction, and must be visible ----------
V.on_event(None, None, "evt64591")
_rows3 = list(V.shower_src.data["node"])
V.on_shower_select(None, None, [_rows3.index(78025)])
V.apply_marks([60008], "in")
V.on_shower_select(None, None, [_rows3.index(83044)])
V.apply_marks([60008], "in")
check("a segment marked IN against two showers is detected",
      V.mark_conflicts() == {60008: [78025, 83044]}
      or V.mark_conflicts() == {60008: [83044, 78025]},
      str(V.mark_conflicts()))
check("  ... and called out with the numbers that decide it",
      "marked IN against 2 showers" in V.marks_div.text
      and "tier <b>2</b>" in V.marks_div.text
      and "14.25" in V.marks_div.text and "412.84" in V.marks_div.text,
      re.sub("<[^>]+>", "", V.marks_div.text)[-190:])
V.on_save()
check("  ... and saving a contradictory record warns at the save itself",
      "marked IN against showers 78025 and 83044" in V.save_note.text,
      re.sub("<[^>]+>", "", V.save_note.text)[-110:])
V.on_shower_select(None, None, [_rows3.index(78025)])
V.apply_marks([60008], None)
check("  ... and clearing one side ends the conflict",
      V.mark_conflicts() == {} and "marked IN against" not in V.marks_div.text)
V.on_save()
check("  ... after which the save is clean",
      "marked IN against" not in V.save_note.text
      and json.load(open(V.label_path("evt64591")))["em"]["marks_by_shower"]
      == {"83044": {"60008": "in"}},
      str(json.load(open(V.label_path("evt64591")))["em"]["marks_by_shower"]))

# --- the comparison line survives a shower with NO usable axis ---------------
# Every other round-5 check runs on events whose showers have the probe's dir15.
# selftest_repro reports exactly two that do not; evt285567's falls all the way
# back to a ZERO vector, so nothing plots and refresh_cmp has only the
# "not on the plot" branch to take.  That branch indexes cand_pt_src, which does
# not contain the segment -- so this is where it would throw.
V.on_event(None, None, "evt285567")
_rows2 = list(V.shower_src.data["node"])
V.on_shower_select(None, None, [_rows2.index(15047)])
_ax, _br, _src = V.shower_axis(15047)
check("the axis-less shower really has no axis (else this proves nothing)",
      _src == "python" and V.G.vmag(_ax) == 0.0, "%s/%s" % (_br, _src))
_other = [s for s in V.cand_src.data["sid"]
          if s not in set(V.members_of(15047))][0]
V.apply_marks([_other], "in")
check("  ... and marking against it degrades instead of throwing",
      len(V.cand_pt_src.data["sid"]) == 0
      and "no member plotted" in V.cmp_div.text
      and "not on the plot" in V.cmp_div.text,
      re.sub("<[^>]+>", "", V.cmp_div.text)[:80])

# ---------------------------------------------------------------------------
# round 5c: which recombination an energy was converted with, and what a PID
# correction does to the pi0 mass
# ---------------------------------------------------------------------------
print()
V.on_event(None, None, "evt166870")
check("shower_is_em mirrors get_flag_shower off the dump",
      V.shower_is_em(85045) is False       # pdg 13, neither shower flag
      and V.shower_is_em(87058) is True    # kShowerTopology on the start segment
      and V.shower_is_em(10013) is True,   # no flag, but |pdg| == 11
      "%s %s %s" % (V.shower_is_em(85045), V.shower_is_em(87058),
                    V.shower_is_em(10013)))
_lbl, _used, _alt = V.kine_hypothesis(85045)
check("  ... a muon-PID'd object's kine_charge used the TRACK factors",
      _lbl == "track" and _used == V.KINE_TRACK, "%s %s" % (_lbl, _used))
_e85 = V.shower_by_node(85045)["kine_charge"]
check("  ... and the shower hypothesis is 1.66x, not a re-measurement",
      abs(_alt - _e85 * (0.7 * 0.95) / (0.5 * 0.8)) < 1e-9
      and abs(_alt / _e85 - 1.6625) < 1e-9,
      "%.4f -> %.4f (x%.4f)" % (_e85, _alt, _alt / _e85))
check("  ... a proton gets its own recombination, fudge stays at the track one",
      V.kine_hypothesis(10074)[1] == V.KINE_PROTON == (0.35, 0.95))
V.mode_group.active = 1
V.state["gamma"] = {1: 87058, 2: 85045}
V.vtx_mode_group.active = 2
V.state["vtx_manual"] = None
V.refresh_kine()
check("the panel names the hypothesis each gamma's energy was converted with",
      "converted as <b>shower</b>" in V.kine_div.text
      and "converted as <b>track</b>" in V.kine_div.text)
check("  ... and only the TRACK-flagged gamma is promoted (both would cancel)",
      "149.7 MeV</b> (E 173.8 + 64.2)" in V.kine_div.text,
      re.sub("<[^>]+>", "", V.kine_div.text.replace("<br>", " | "))[-260:])
check("a manual pi0 vertex with no point set is called out, not left blank",
      "no point set" in V.kine_div.text)
V.state["gamma"] = {1: None, 2: None}
V.mode_group.active = 0

check("the new EM verdict is APPENDED, so old labels keep their index",
      V.EM_VERDICTS[:6] == ["correct", "over-clustered", "under-clustered",
                            "both", "vertex-bad (undecidable)",
                            "not an EM shower"]
      and V.EM_VERDICTS[6] == "is an EM shower (reco PID wrong)",
      str(V.EM_VERDICTS))
_rowsP = list(V.shower_src.data["node"])
V.on_shower_select(None, None, [_rowsP.index(85045)])
V.em_verdict.active = 6
V.state["gamma"] = {1: 87058, 2: 85045}
V.on_save()
_rp = json.load(open(V.label_path("evt166870")))
check("  ... and the record says what the reco called it at the time",
      _rp["em"]["verdict"] == "is an EM shower (reco PID wrong)"
      and _rp["em"]["reco"]["particle_id"] == 13
      and _rp["em"]["reco"]["flag_shower"] is False
      and _rp["em"]["reco"]["kine_hypothesis"] == "track"
      and abs(_rp["em"]["reco"]["kine_charge_other_hypothesis"] - 64.16) < 0.02,
      json.dumps({k: _rp["em"]["reco"][k] for k in
                  ("particle_id", "flag_shower", "kine_hypothesis",
                   "kine_charge_other_hypothesis")}))
check("  ... and each gamma slot carries its own hypothesis too",
      _rp["pio"]["gammas"]["2"]["kine_hypothesis"] == "track"
      and _rp["pio"]["gammas"]["1"]["kine_hypothesis"] == "shower",
      str({s: g.get("kine_hypothesis")
           for s, g in _rp["pio"]["gammas"].items()}))

# ---------------------------------------------------------------------------
# round 5d: the pi0 verdict is retired, but a pre-5d one is not destroyed
# ---------------------------------------------------------------------------
print()
check("the pi0 verdict control is gone", not hasattr(V, "pio_verdict"))
check("  ... and the old vocabulary is kept only for READING",
      V.PIO_VERDICTS_LEGACY[0] == "pi0 correct" and len(V.PIO_VERDICTS_LEGACY) == 6)
_lp = V.label_path("evt21073")
_old = json.load(open(_lp))
_old["pio"]["verdict"] = "wrong pairing"          # as a pre-5d build wrote it
with open(_lp, "w") as _fh:
    json.dump(_old, _fh)
V.on_event(None, None, "evt463565")
V.on_event(None, None, "evt21073")
check("a pre-5d pi0 verdict is read back into state",
      V.state["pio_verdict_legacy"] == "wrong pairing",
      str(V.state["pio_verdict_legacy"]))
V.on_save()
check("  ... and re-saving PRESERVES it rather than deleting a past judgement",
      json.load(open(_lp))["pio"]["verdict"] == "wrong pairing")
V.on_event(None, None, "evt84229")
check("  ... while an event that never had one still writes none",
      V.state["pio_verdict_legacy"] is None)

# ---- round 6: the owner's hint, and the four added events -------------------
check("the scan sample grew to 98 rows", len(V.MANIFEST) == 98, str(len(V.MANIFEST)))
check("  ... the four owner adds are loadable events",
      all("evt%s" % _e in V.EVENTS
          for _e in ("169626", "174752", "347129", "394532")))
check("  ... each with a probe sidecar, like the original 94",
      all(V.MANIFEST[_e]["has_probe"] == "1"
          for _e in ("169626", "174752", "347129", "394532")))
# Data-driven, not a magic number: the count is whatever the durable input says,
# so adding an event to pr114-owner-adds.index.txt is a data change -- while a
# note that silently fails to reach the manifest is still a failure.
_adds = os.path.join(V.SX, "docs", "pr", "pr114-owner-adds.index.txt")
_want = {}
with open(_adds) as _fh:
    for _l in _fh:
        if _l.startswith("#") or not _l.strip():
            continue
        _f = _l.rstrip("\n").split("\t")
        if len(_f) >= 6 and _f[5].strip():
            _want[_f[3]] = _f[5]
check("every owner note in the index reaches the manifest",
      all(V.MANIFEST.get(_e, {}).get("scan_note") == _n
          for _e, _n in _want.items()),
      "%d notes; missing/mismatched: %s"
      % (len(_want), [_e for _e, _n in _want.items()
                      if V.MANIFEST.get(_e, {}).get("scan_note") != _n] or "none"))
check("  ... and no manifest row invents a note that is not in the index",
      {_e for _e, _r in V.MANIFEST.items() if _r.get("scan_note")} == set(_want),
      str(sorted({_e for _e, _r in V.MANIFEST.items()
                  if _r.get("scan_note")} - set(_want))))

# ---- round 7: the 259774 -> 269774 typo ------------------------------------
# The owner wrote "18255-259774: multiple pi0"; 259774 was never reconstructed
# anywhere and 269774 is the unique 1-edit neighbour.  Pin BOTH halves so the
# resolution cannot quietly rot back.
check("the mis-typed 259774 is not a live row in the index",
      "259774" not in _want,
      str(sorted(_want)))
check("  ... and the note landed on 269774, which really is multiple-pi0",
      V.MANIFEST.get("269774", {}).get("scan_note") == "multiple pi0"
      and int(V.MANIFEST["269774"]["n_pio_groups"]) >= 2,
      "note=%r groups=%s" % (V.MANIFEST.get("269774", {}).get("scan_note"),
                             V.MANIFEST.get("269774", {}).get("n_pio_groups")))

# evt84229 is loaded: its hint must be ON SCREEN and OUT of the record.
check("the owner's hint is shown for an event that has one",
      "pi0 two gamma merged" in V.scan_note_div.text
      and "what you asked to look at here" in V.scan_note_div.text)
check("  ... and it is NOT loaded into the editable note box",
      "pi0 two gamma merged" not in (V.note_in.value or ""),
      "note_in=%r" % (V.note_in.value,))
V.on_save()
_saved = json.load(open(V.label_path("evt84229")))
check("  ... so a save cannot record the question as if it were the answer",
      "pi0 two gamma merged" not in json.dumps(_saved.get("note")),
      "note=%r" % (_saved.get("note"),))
V.on_event(None, None, "evt463565")
check("  ... and an event without a hint shows no banner at all",
      V.scan_note_div.text == "", repr(V.scan_note_div.text[:40]))

# 169626's Bee set is built but not uploaded: the banner must not read as
# "no 3-D for this event", which is what the pre-round-6 wording said.
V.on_event(None, None, "evt169626")
check("a built-but-unuploaded Bee set says so, and says the cloud is fine",
      "not uploaded" in V.banner.text and "3-D" in V.banner.text
      and "no Bee set" not in V.banner.text)

# ---- round 7: "have I already scanned this one?" ---------------------------
# The counter said 3/98; it could not say whether THIS event was one of the 3.
# evt84229 was saved a few checks above, so it is a real saved event here.
V.on_event(None, None, "evt84229")
check("an already-scanned event says so at the top",
      "already scanned this event" in V.scan_status.text
      and "not scanned yet" not in V.scan_status.text,
      repr(V.scan_status.text[:80]))
check("  ... and names the tag the result is in",
      V.SCAN_TAG in V.scan_status.text)

V.on_event(None, None, "evt463565")
check("an unscanned event says THAT, rather than going blank",
      "not scanned yet" in V.scan_status.text
      and "already scanned" not in V.scan_status.text,
      repr(V.scan_status.text[:80]))

# The two-tab case.  state["saved"] is a load-time snapshot; if the chip were
# driven from it, a second tab would keep saying "not scanned yet" after this
# one wrote the file.  Blank the snapshot with the file still on disk: the
# answer must not change, because the answer comes from the filesystem.
V.on_event(None, None, "evt84229")
_snap = V.state["saved"]
V.state["saved"] = None
V.refresh_scan_status()
check("  ... and the answer is read from disk, not from a stale in-memory copy",
      "already scanned this event" in V.scan_status.text,
      repr(V.scan_status.text[:80]))
check("  ... dropping the timestamp it can no longer honestly quote",
      "saved 20" not in V.scan_status.text)
V.state["saved"] = _snap
V.refresh_scan_status()

# Disk state and edit state are different questions and are rendered by
# different widgets; the chip must not start duplicating [unsaved].
V.state["dirty"] = True
V.refresh_info()
check("the chip reports disk state only -- [unsaved] stays refresh_info's job",
      "unsaved" not in V.scan_status.text and "unsaved" in V.info.text,
      "chip=%r" % (V.scan_status.text[:60],))
V.state["dirty"] = False
V.refresh_info()

# "at the top" is the whole point of the request, so pin the POSITION, not just
# the text: directly under the header row and above the Bee banner.  `info`,
# which already carried the n/98 counter, lives in the right column instead --
# which is exactly why the counter could not answer this question.
_kids = [getattr(_w, "name", None) for _w in V.layout.children]
# A disk read is only worth having if something wakes it.  refresh_scan_status
# otherwise fires only from refresh_info -- load, save, touch -- so a save made
# in another tab would not show until the scanner navigated away and back.
_periodic = [cb for cb in V.curdoc().session_callbacks
             if getattr(getattr(cb, "callback", None), "__name__", "")
             == "refresh_scan_status"]
check("a periodic callback wakes the chip while sitting on one event",
      len(_periodic) == 1, "%d registered" % len(_periodic))
check("  ... at a period that is cheap (one stat) but not sluggish",
      2000 <= getattr(_periodic[0], "period", 0) <= 15000 if _periodic else False,
      str(getattr(_periodic[0], "period", None) if _periodic else None))

# The real two-tab case, end to end: another writer creates the label while this
# session sits on the event, and the periodic tick must flip the chip.  Done in a
# THROWAWAY tag so the suite stays re-runnable and no real scan record is
# touched (M13); the file it creates is the only thing it removes.
_realtag = V.SCAN_TAG
V.SCAN_TAG = "selftest114-flip"
try:
    V.on_event(None, None, "evt463565")
    _before = V.scan_status.text
    _p = V.label_path("evt463565")
    os.makedirs(os.path.dirname(_p), exist_ok=True)
    with open(_p, "w") as _fh:
        json.dump({"saved_utc": "written-by-another-tab"}, _fh)
    V.refresh_scan_status()          # what the periodic callback calls
    check("a save made by ANOTHER tab flips the chip without navigating",
          "not scanned yet" in _before
          and "already scanned this event" in V.scan_status.text,
          "before=%r after=%r" % (_before[:40], V.scan_status.text[:40]))
finally:
    try:
        os.remove(_p)
        os.rmdir(os.path.dirname(_p))
    except OSError:
        pass
    V.SCAN_TAG = _realtag
    V.on_event(None, None, "evt84229")

check("the chip sits at the top, directly under the header row",
      _kids[1] == "scan_status", str(_kids))
check("  ... above the Bee banner, not below it",
      _kids.index("scan_status") < _kids.index("banner"), str(_kids))

# ---- round 8: the scanner's own start and direction -------------------------
# evt169626 is the event the request came from.
V.on_event(None, None, "evt169626")
_node = V.shower_src.data["node"][0]
V.on_shower_select(None, None, [0])
_rs = V.reco_start(_node)
_ax0, _br0, _src0 = V.shower_axis(_node)
check("with no override the axis is still the probe's dir15",
      _src0 == "probe" and _br0 == "dir15", "%s / %s" % (_br0, _src0))
check("  ... and the start is the reconstruction's",
      V.shower_start(_node) == _rs, str(_rs))

_new = (_rs[0] + 20.0, _rs[1], _rs[2])
V.set_em_start(_new, "")
check("a new start reaches shower_start",
      V.shower_start(_node) == _new, str(V.shower_start(_node)))

# THE invariant.  If the start moved and the axis did not, seg_vs_shower would
# take the angle between a direction anchored at the OLD start and a
# displacement measured from the NEW one -- not a physical quantity, and one
# that looks perfectly plausible on screen and in the saved record.
_ax1, _br1, _src1 = V.shower_axis(_node)
check("  ... and THE AXIS MOVES WITH IT -- never a stale probe value",
      _src1 == "python@start_override" and _ax1 != _ax0,
      "%s / %s" % (_br1, _src1))
check("  ... recomputed with the same formula, at the new point",
      all(abs(a - b) < 1e-12 for a, b in zip(
          _ax1, V.G.shower_cal_dir_3vector(
              [sg for sg in V.cur_segments()
               if sg.get("id") in set(V.members_of(_node))], _new, 15.0))),
      str([round(v, 4) for v in _ax1]))

# and the gate's own inputs follow, which is the point of the whole feature
_sid = V.cur_segments()[0].get("id")
V.state["em_start"].pop(_node); V.state["_axis_cache"] = {}
_m0 = V.seg_vs_shower(_node, _sid)
V.state["em_start"][_node] = _new; V.state["_axis_cache"] = {}
_m1 = V.seg_vs_shower(_node, _sid)
check("  ... so the pass-1 gate's distance/angle are measured from it",
      _m0["dist"] != _m1["dist"] or _m0["angle"] != _m1["angle"],
      "before=%s after=%s" % (_m0["dist"], _m1["dist"]))

# direction by clicking a second point
V.set_em_dir((_new[0], _new[1] + 30.0, _new[2]), "")
_ax2, _br2, _src2 = V.shower_axis(_node)
check("aiming through a second point gives exactly that direction",
      _br2 == "two_point" and _src2 == "manual@override"
      and all(abs(a - b) < 1e-12
              for a, b in zip(_ax2, V.G.vnorm((0.0, 30.0, 0.0)))),
      "%s %s" % ([round(v, 4) for v in _ax2], _src2))

# The pi0 path SEES the correction (round 8b -- round 8 had it ignore one, and
# that is exactly what the owner reported: "I am also confused which one was
# used to do the calculation of the pi0 mass").
check("the pi0 gamma starts follow an EM start correction",
      V.shower_start(_node, 1) == _new and V.shower_start(_node, 2) == _new,
      str(V.shower_start(_node, 1)))
check("  ... unless that gamma slot has its own start, which is more specific",
      (V.state["gstart"].__setitem__(1, (9.0, 9.0, 9.0)),
       V.shower_start(_node, 1) == (9.0, 9.0, 9.0),
       V.state["gstart"].__setitem__(1, None))[1])

# the record: the reco block must stay the RECONSTRUCTION's answer
V.event_flag_group.active = [0]
V.on_save()
_r8 = json.load(open(V.label_path("evt169626")))
_em8 = _r8["em"]
check("the record keeps the reco's axis and the one you used APART",
      _em8["reco"]["axis_source"] == "probe"
      and _em8["axis_used_source"] == "manual@override",
      "reco=%s used=%s" % (_em8["reco"]["axis_source"],
                           _em8["axis_used_source"]))
check("  ... and both start points, so the move is checkable later",
      _em8["reco_start"] == list(_rs) and _em8["start_used"] == list(_new),
      "reco=%s used=%s" % (_em8["reco_start"], _em8["start_used"]))
check("  ... keyed BY SHOWER, since mark_metrics runs for every marked one",
      _em8["start_override_by_shower"] == {str(_node): list(_new)},
      str(_em8["start_override_by_shower"]))

# round 8: the event-level topology flag
check("the no-vertex NCpi0 flag is saved at the ROOT, beside em and pio",
      _r8.get("event_flags") == ["no_vertex_ncpi0"],
      str(_r8.get("event_flags")))

# everything must come back on reload -- including for showers not selected
V.on_event(None, None, "evt84229")
check("leaving the event drops the overrides from state",
      not V.state["em_start"] and not V.state["em_dir"],
      str(V.state["em_start"]))
V.on_event(None, None, "evt169626")
check("  ... and re-opening the event restores them from the record",
      V.state["em_start"].get(_node) == _new
      and V.state["em_dir"].get(_node) == (_new[0], _new[1] + 30.0, _new[2]),
      str(V.state["em_start"]))
check("  ... and the topology flag comes back too",
      V.event_flag_group.active == [0], str(V.event_flag_group.active))
os.remove(V.label_path("evt169626"))

check("clearing puts the reconstruction's own start and axis back",
      (V.clear_em_start(), V.clear_em_dir(),
       V.shower_start(_node) == _rs
       and V.shower_axis(_node)[2] == "probe")[-1],
      str(V.shower_axis(_node)[2]))

_ei = V.LAYER_KEYS.index("emstart") if "emstart" in V.LAYER_KEYS else None
check("the start / direction markers register under a nameable layer key",
      _ei is not None and bool(V.RENDER.get("emstart")),
      "%s renderers" % len(V.RENDER.get("emstart") or []))
V.layer_group.active = [_ei]
V.apply_layers(None, None, None)
_on = all(r.visible for r in V.RENDER["emstart"])
V.layer_group.active = []
V.apply_layers(None, None, None)
_off = not any(r.visible for r in V.RENDER["emstart"])
check("  ... and its checkbox actually drives them, both ways",
      _on and _off, "on=%s off=%s" % (_on, _off))
V.layer_group.active = [i for i in range(len(V.LAYER_KEYS)) if i != 6]
V.apply_layers(None, None, None)

# ---- round 8b: the correction must reach the pi0 arithmetic too -------------
# Reported from the live scan: "I already adjusted the start vertex ... I am
# also confused which one was used to do the calculation of the pi0 mass."
V.on_event(None, None, "evt169626")
V.on_shower_select(None, None, [0])
_n = V.state["sel_shower"]
_rs2 = V.reco_start(_n)
_new2 = (_rs2[0] + 20.0, _rs2[1], _rs2[2])
V.set_em_start(_new2, "")
check("an EM start correction reaches the pi0 gamma path as well",
      V.shower_start(_n, 1) == _new2 and V.shower_start(_n, 2) == _new2,
      str(V.shower_start(_n, 1)))
check("  ... so both mass conventions are built on the SAME geometry",
      # the axis convention takes no slot and always used the correction; the
      # vertex convention goes through shower_start(node, slot).  When these
      # disagreed, one mass came from the scanner's start and the other from
      # the reconstruction's, in the same saved record.
      V.shower_start(_n, 1) == V.shower_start(_n),
      "slot=%s bare=%s" % (V.shower_start(_n, 1), V.shower_start(_n)))
check("  ... and the record names which start each gamma used",
      V.start_source(_n, 1) == "em_start_correction", V.start_source(_n, 1))

# precedence: a start set for THIS gamma slot is more specific, and still wins
V.state["gstart"][1] = (1.0, 2.0, 3.0)
check("a pi0 slot override still beats the EM correction",
      V.shower_start(_n, 1) == (1.0, 2.0, 3.0)
      and V.start_source(_n, 1) == "gamma_slot_override",
      str(V.shower_start(_n, 1)))
V.state["gstart"][1] = None
check("  ... and removing it falls back to the correction, not the reco",
      V.shower_start(_n, 1) == _new2, str(V.shower_start(_n, 1)))

# the reported symptom: a mode round trip must not appear to revert anything
V.mode_group.active = 1
V.on_mode(None, None, 1)
_mid = dict(V.state["em_start"])
V.mode_group.active = 0
V.on_mode(None, None, 0)
check("a pi0-tab round trip leaves the corrected start standing",
      _mid.get(_n) == _new2 and V.state["em_start"].get(_n) == _new2
      and V.shower_start(_n) == _new2, str(V.state["em_start"].get(_n)))
check("  ... and the readout is redrawn, not left stale",
      "yours" in V.emstart_div.text and len(V.emstart_src.data["x"]) == 2,
      "%d marker(s)" % len(V.emstart_src.data["x"]))
V.clear_em_start()

# ---- round 9: which recombination pair a gamma's charge is converted with ---
# evt166870, the owner's own case: "the energy of the EM shower should use the
# charge inferred one instead of the kinetic energy", with the note on that
# record reading "85045 should be an EM shower, part of pi0".
V.on_event(None, None, "evt166870")
_g1, _g2 = 87058, 85045          # 85045 is pdg 13, flag_shower False
V.state["gamma"][1] = _g1
V.state["gamma"][2] = _g2
V.mode_group.active = 1
check("the reco converted 85045's charge as a TRACK, not a shower",
      V.shower_is_em(_g2) is False and V.kine_hypothesis(_g2)[0] == "track",
      str(V.kine_hypothesis(_g2)[0]))
check("  ... so the default keeps the reco's number, byte-for-byte",
      V.g2_ehyp.value == V.EHYP_RECO
      and abs(V.gamma_energy(2) - V.shower_energy(_g2)) < 1e-12,
      "%.1f" % V.gamma_energy(2))

_m_reco = V.G.pi0_mass(V.gamma_energy(1), V.gamma_energy(2),
                       V.G.angle_deg(V.shower_axis(_g1)[0], V.shower_axis(_g2)[0]))
V.g2_ehyp.value = V.EHYP_EM
_e_em = V.gamma_energy(2)
check("switching to the EM hypothesis re-converts the SAME charge",
      abs(_e_em - V.kine_hypothesis(_g2)[2]) < 1e-12
      and abs(_e_em - 64.2) < 0.1,
      "%.1f MeV, was %.1f" % (_e_em, V.shower_energy(_g2)))
_m_em = V.G.pi0_mass(V.gamma_energy(1), _e_em,
                     V.G.angle_deg(V.shower_axis(_g1)[0], V.shower_axis(_g2)[0]))
check("  ... and the pi0 mass follows: 116.1 -> 149.7 MeV",
      abs(_m_reco - 116.1) < 0.2 and abs(_m_em - 149.7) < 0.2,
      "%.1f -> %.1f" % (_m_reco, _m_em))

# a gamma the reco ALREADY called a shower must not be re-converted twice
V.g1_ehyp.value = V.EHYP_EM
check("a gamma already charge-inferred as a shower is left alone",
      abs(V.gamma_energy(1) - V.shower_energy(_g1)) < 1e-12,
      "%.1f" % V.gamma_energy(1))
V.g1_ehyp.value = V.EHYP_RECO

# the record must carry both numbers and which one was used
V.on_save()
_r9 = json.load(open(V.label_path("evt166870")))
_gm = _r9["pio"]["gammas"]["2"]
check("the record says which pair was used, and keeps the reco's number too",
      _gm["energy_hypothesis"] == "as_em_shower"
      and abs(_gm["energy"] - 64.2) < 0.1
      and abs(_gm["energy_as_reconstructed"] - 38.6) < 0.1,
      "%s E=%.1f reco=%.1f" % (_gm["energy_hypothesis"], _gm["energy"],
                               _gm["energy_as_reconstructed"]))

# THE regression that matters: a record saved before this control existed has no
# energy_hypothesis key, and re-opening it must show the mass it was saved with.
_p9 = V.label_path("evt166870")
_old = json.load(open(_p9))
for _sl in ("1", "2"):
    _old["pio"]["gammas"][_sl].pop("energy_hypothesis", None)
with open(_p9, "w") as _fh:
    json.dump(_old, _fh)
V.on_event(None, None, "evt84229")
V.on_event(None, None, "evt166870")
check("a pre-round-9 record re-opens on the reco's energy, not the EM one",
      V.g1_ehyp.value == V.EHYP_RECO and V.g2_ehyp.value == V.EHYP_RECO,
      "%s / %s" % (V.g1_ehyp.value, V.g2_ehyp.value))
os.remove(_p9)

# ---------------------------------------------------------------------------
# round 10: a WHOLE shower into another one, in one gesture
#
# evt172942 is the case that asked for it.  The event number is pinned here
# rather than in prose because it was reached by fingerprint, not by being
# quoted: shower 4002 and shower 71022 occur together in exactly ONE of the 98
# scan events, and that event is 172942 (the owner wrote 179242 -- one adjacent
# transposition).  If a future manifest change breaks that uniqueness, this
# check is where it shows up.
# ---------------------------------------------------------------------------
_both = [e for e, _row in MANIFEST_EVENTS
         if _has_showers(e, (4002, 71022))]
check("shower pair (4002, 71022) identifies exactly one event",
      _both == ["172942"], str(_both))

# The round-10 bug the browser found: a stale highlight left over from the
# previous event, with state["sel_shower"] None behind it, made clicking that
# very row a no-op.  Python cannot see the no-op (it assigns indices directly),
# but it can pin the invariant the fix installs.
V.shower_src.selected.indices = [1]
V.on_event(None, None, "evt172942")
check("an event switch clears the shower table's stale highlight",
      list(V.shower_src.selected.indices) == [],
      str(list(V.shower_src.selected.indices)))
_rows = list(V.shower_src.data["node"])
check("evt172942 holds both showers", 4002 in _rows and 71022 in _rows,
      str(_rows))
_i = _rows.index(4002)
V.shower_src.selected.indices = [_i]
V.on_shower_select(None, None, [_i])
check("  ... 4002 is the shower being scanned", V.state["sel_shower"] == 4002)

# the menu: everything but the shower being scanned, biggest first
_opts = list(V.bulk_shower.options)
check("the bulk menu lists the other showers, energy-ordered",
      [V._excl_node(o) for o in _opts] == [71022, 47004, 51008, 48005],
      str([V._excl_node(o) for o in _opts]))
check("  ... and never the shower being scanned",
      all(V._excl_node(o) != 4002 for o in _opts))
V.state["sel_shower"] = 71022
V.refresh_bulk_options()
check("  ... which follows the table: scanning 71022 puts 4002 back in it",
      V._excl_node(V.bulk_shower.options[0]) == 4002
      and all(V._excl_node(o) != 71022 for o in V.bulk_shower.options),
      str([V._excl_node(o) for o in V.bulk_shower.options]))
V.state["sel_shower"] = 4002
V.refresh_bulk_options()

_want = sorted(V.members_of(71022))
check("shower 71022 has 10 segments over 8 clusters", len(_want) == 10,
      str(_want))
V.bulk_shower.value = [o for o in V.bulk_shower.options
                       if V._excl_node(o) == 71022][0]
V.on_bulk(False)()
# Resolved through the SAME function the mark buttons call, not by counting
# table rows: the three views can hold different subsets and only this union is
# what a mark would act on.
check("select-all picks up every one of its segments",
      sorted(V.selected_cand_ids()) == _want,
      "%d of %d" % (len(V.selected_cand_ids()), len(_want)))
check("  ... and the note reports the count that will be marked",
      "10 of 10" in V.save_note.text, V.save_note.text[:90])

# Bokeh fires selected.indices only on CHANGE -- pressing it twice for the same
# shower must still leave the selection standing.
V.on_bulk(False)()
check("  ... pressing it again for the same shower keeps the 10 selected",
      sorted(V.selected_cand_ids()) == _want,
      "%d" % len(V.selected_cand_ids()))

V.mark("in")()
check("one mark IN files all 10 against shower 4002",
      sorted(V.marks_for(4002)) == _want
      and set(V.marks_for(4002).values()) == {"in"},
      "%d marks" % len(V.marks_for(4002)))
check("  ... and membership of 71022 is not itself a conflict",
      V.mark_conflicts() == {}, str(V.mark_conflicts()))
_imp = V.impact.text
check("  ... impact counts 10 non-members IN",
      "10 non-member(s) IN" in _imp, _imp[:140])

# THE one that would write a silently incomplete record: a dimmed shower is
# dropped from the candidate table but kept in pick_src, so a selection built
# from table rows would be empty and the mark would reach nothing.
V.mark(None)()
_o71 = [o for o in V.excl_choice.options if V._excl_node(o) == 71022][0]
V.excl_choice.value = [_o71]
V.on_excl(None, None, [_o71])
check("dimming 71022 away removes its rows from the candidate table",
      not [s for s in V.cand_src.data["sid"] if s in set(_want)],
      "%d rows left" % len(V.cand_src.data["sid"]))
V.on_bulk(False)()
check("  ... but select-all still reaches all 10 of them",
      sorted(V.selected_cand_ids()) == _want,
      "%d of %d" % (len(V.selected_cand_ids()), len(_want)))
check("  ... and says so rather than letting the empty table read as a loss",
      "not listed in the candidate table" in V.save_note.text,
      V.save_note.text[-160:])
V.mark("in")()
check("  ... the mark lands on all 10 with the shower dimmed",
      len(V.marks_for(4002)) == 10, "%d" % len(V.marks_for(4002)))
V.excl_choice.value = []
V.on_excl(None, None, [])

# accumulate, then replace
V.mark(None)()
V.on_bulk(False)()
_n1 = len(V.selected_cand_ids())
V.bulk_shower.value = [o for o in V.bulk_shower.options
                       if V._excl_node(o) == 47004][0]
V.on_bulk(True)()
check("add-to-selection accumulates a second shower", 
      len(V.selected_cand_ids()) == _n1 + 1,
      "%d -> %d" % (_n1, len(V.selected_cand_ids())))
V.on_bulk(False)()
check("  ... while select-all replaces rather than accumulating",
      len(V.selected_cand_ids()) == 1, "%d" % len(V.selected_cand_ids()))

# refuses, rather than filing a mark with nowhere to belong
_before = {n: dict(m) for n, m in V.state["marks"].items()}
V.state["sel_shower"] = None
V.on_bulk(False)()
check("with no shower scanned it refuses and files nothing",
      "pick the shower you are scanning" in V.save_note.text
      and {n: dict(m) for n, m in V.state["marks"].items()} == _before,
      V.save_note.text[:80])
V.state["sel_shower"] = 4002
V.refresh_bulk_options()

# the record round trip: 10 marks, each with the pass-1 numbers behind it
V.bulk_shower.value = [o for o in V.bulk_shower.options
                       if V._excl_node(o) == 71022][0]
V.on_bulk(False)()
V.mark("in")()
V.em_verdict.active = 0
V.on_save()
_r10 = json.load(open(V.label_path("evt172942")))
_mb = _r10["em"]["marks_by_shower"].get("4002", {})
check("the saved record carries all 10 marks against 4002",
      sorted(int(k) for k in _mb) == _want
      and set(_mb.values()) == {"in"}, "%d entries" % len(_mb))
_md = _r10["em"]["marks_detail"].get("4002", {})
_mk = _md.get("marked") or {}
# The other half of the round-10 load() change: clearing must not cost the
# RESTORE.  A labelled event has to re-open with its shower row highlighted, or
# the fix would have traded a stale highlight for a missing one on every event
# that already has an answer.
V.on_event(None, None, "evt84229")
V.on_event(None, None, "evt172942")
check("a labelled event still re-opens with its shower row highlighted",
      V.state["sel_shower"] == 4002
      and list(V.shower_src.selected.indices)
          == [list(V.shower_src.data["node"]).index(4002)],
      "sel=%s indices=%s" % (V.state["sel_shower"],
                             list(V.shower_src.selected.indices)))
check("  ... with the pass-1 numbers measured for every one of them",
      sorted(int(k) for k in _mk) == _want
      and all({"dist", "angle", "tier"} <= set(v) for v in _mk.values()),
      "%d measured, keys %s" % (len(_mk),
                                sorted(list(_mk.values())[0]) if _mk else "-"))

# ---------------------------------------------------------------------------
# round 11: more than one pi0 pairing per event
#
# evt281485 -- 20 showers, 19 EM, and the reconstruction groups exactly one pi0.
# The scanner reads more than one, which needs more than one pair of slots.
# ---------------------------------------------------------------------------
V.on_event(None, None, "evt281485")
V.mode_group.active = 1
V.on_mode(None, None, 1)
_grp = V.G.pi0_groups(V.cur_showers())
check("evt281485: the reco finds exactly one pi0 group",
      len(_grp) == 1, str({k: [s.get("id") for s in v] for k, v in _grp.items()}))
check("  ... and nothing is stored yet",
      V.state["pio_cands"] == [] and not V.pio_cand_src.data["n"])


def _pair(a, b, how=0):
    V.state["gamma"][1] = a
    V.state["gamma"][2] = b
    V.state["gstart"] = {1: None, 2: None}
    V.vtx_mode_group.active = how
    V.refresh_kine()


# two pi0 made of four DISTINCT showers
_pair(15036, 87078)
V.on_pio_add()
_pair(84070, 91112)
V.on_pio_add()
check("two pairings store as two candidates", len(V.state["pio_cands"]) == 2,
      "%d" % len(V.state["pio_cands"]))
check("  ... each with its own mass, both conventions",
      [round(c["mass_axis_convention"], 1) for c in V.state["pio_cands"]]
      == [219.7, 81.2]
      and [round(c["mass_vertex_convention"], 1) for c in V.state["pio_cands"]]
      == [208.4, 93.7],
      str([(round(c["mass_axis_convention"], 1),
            round(c["mass_vertex_convention"], 1)) for c in V.state["pio_cands"]]))
check("  ... and four distinct showers read as two separate pi0",
      "No shower is used twice" in V.pio_cand_div.text,
      V.pio_cand_div.text[-120:])

# an ALTERNATIVE pairing that reuses a gamma: legal, but it is not a third pi0
_pair(15036, 88090)
V.on_pio_add()
check("an alternative pairing that reuses a gamma is stored too",
      len(V.state["pio_cands"]) == 3)
check("  ... and is called what it is, not counted as a third pi0",
      "alternative pairings" in V.pio_cand_div.text
      and "15036" in V.pio_cand_div.text
      and "candidates 1 and 3" in V.pio_cand_div.text,
      V.pio_cand_div.text[-200:])

V.on_pio_add()
check("storing the identical pairing twice is refused, by name",
      len(V.state["pio_cands"]) == 3
      and "already stored as candidate <b>3</b>" in V.save_note.text,
      V.save_note.text[:100])

# the same two gammas under a DIFFERENT vertex convention is a different mass,
# and must be storable -- the dedup key is not the shower pair
_pair(84070, 91112, how=1)
V.on_pio_add()
check("the same gammas under another vertex convention do store",
      len(V.state["pio_cands"]) == 4
      and V.state["pio_cands"][3]["vertex_how"] == "backproject",
      "%d, how=%s" % (len(V.state["pio_cands"]),
                      V.state["pio_cands"][3].get("vertex_how")))

# clearing the SLOTS must not clear the record
V.on_gamma_clear()
check("`clear gammas` clears the slots and keeps the stored pairings",
      V.state["gamma"] == {1: None, 2: None}
      and len(V.state["pio_cands"]) == 4)

# ... and a save with the slots empty must still write them
V.on_save()
_r11 = json.load(open(V.label_path("evt281485")))
check("a save with empty slots still writes the stored pairings",
      _r11["pio"] is not None
      and len(_r11["pio"]["candidates"]) == 4,
      "%s" % (len(_r11["pio"]["candidates"]) if _r11.get("pio") else "no pio"))
check("  ... each candidate self-contained: gammas, starts, energies, vertex",
      all({"gammas", "vertex", "vertex_how", "mass_axis_convention",
           "mass_vertex_convention", "stored_utc"} <= set(c)
          and {"1", "2"} == set(c["gammas"])
          and all({"shower", "start", "axis", "energy", "energy_hypothesis",
                   "energy_as_reconstructed", "start_source"} <= set(g)
                  for g in c["gammas"].values())
          for c in _r11["pio"]["candidates"]))

V.on_event(None, None, "evt64591")
V.on_event(None, None, "evt281485")
check("re-opening the event restores all four",
      len(V.state["pio_cands"]) == 4
      and len(V.pio_cand_src.data["n"]) == 4,
      "%d in state, %d rows" % (len(V.state["pio_cands"]),
                                len(V.pio_cand_src.data["n"])))

# load one back into the slots
V.pio_cand_src.selected.indices = [1]
V.on_pio_load()
check("loading a stored pairing puts it back in the slots",
      V.state["gamma"] == {1: 84070, 2: 91112},
      str(V.state["gamma"]))
# The provenance trap: pinning gstart unconditionally would make every loaded
# candidate report `gamma_slot_override` for a start that came from the reco.
check("  ... without inventing a slot override the scanner never made",
      V.state["gstart"] == {1: None, 2: None}
      and V.start_source(84070, 1) == "reco",
      "%s / %s" % (V.state["gstart"], V.start_source(84070, 1)))

# THE one that matters: a stored mass must not be re-priced by a later edit.
_before = round(V.state["pio_cands"][1]["mass_axis_convention"], 1)
V.state["em_start"][84070] = (-170.0, 175.0, 285.0)
V.state["_axis_cache"] = {}
V.pio_cand_src.selected.indices = [1]
V.on_pio_load()
check("an EM-mode start correction does NOT re-price a stored mass",
      round(V.state["pio_cands"][1]["mass_axis_convention"], 1) == _before,
      "%s, was %s" % (round(V.state["pio_cands"][1]["mass_axis_convention"], 1),
                      _before))
check("  ... and the drift is reported, not absorbed",
      "not</b> what was stored" in V.save_note.text
      and "axis-convention mass" in V.save_note.text,
      V.save_note.text[-200:])
V.state["em_start"].pop(84070, None)
V.state["_axis_cache"] = {}

# remove
V.pio_cand_src.selected.indices = [0]
V.on_pio_del()
check("removing a pairing drops exactly that one",
      len(V.state["pio_cands"]) == 3
      and round(V.state["pio_cands"][0]["mass_axis_convention"], 1) == 81.2,
      "%d left, first now %s" % (len(V.state["pio_cands"]),
                                 round(V.state["pio_cands"][0]["mass_axis_convention"], 1)))
check("  ... and says the numbering moved", "renumbered" in V.save_note.text,
      V.save_note.text[:90])

# the forgotten-pairing warning, rather than a silent auto-add
_pair(89095, 90104)
V.refresh_pio_cands()
check("a pairing in the slots that is not stored is called out",
      "not one of them" in V.pio_cand_div.text, V.pio_cand_div.text[-180:])
check("  ... and is NOT auto-added to the list",
      len(V.state["pio_cands"]) == 3, "%d" % len(V.state["pio_cands"]))

# a record written before the list existed: absent must mean empty
_p11 = V.label_path("evt281485")
V.on_save()
_old11 = json.load(open(_p11))
_top = _old11["pio"]["mass_axis_convention"]
_old11["pio"].pop("candidates")
with open(_p11, "w") as _fh:
    json.dump(_old11, _fh)
V.on_event(None, None, "evt64591")
V.on_event(None, None, "evt281485")
check("a pre-round-11 record re-opens with no candidates and its own mass",
      V.state["pio_cands"] == []
      and V.state["gamma"] == {1: 89095, 2: 90104}
      and abs(V.pio_pairing()["mass_axis_convention"] - _top) < 1e-9,
      "%d candidates, mass %s vs %s"
      % (len(V.state["pio_cands"]),
         V.pio_pairing()["mass_axis_convention"], _top))
os.remove(_p11)

# ---------------------------------------------------------------------------
# round 12: the marks reach the pi0 energy
#
# evt409634 -- shower 27015 (10 segments, 105.05 MeV) merged into shower 69032
# (2 segments, 39.06 MeV).  Round 10 made that one gesture; the energy did not
# follow it, which is the gap this round closes.
# ---------------------------------------------------------------------------
_ev12 = [e for e, _row in MANIFEST_EVENTS if _has_showers(e, (69032, 27015))]
check("shower 69032 + shower 27015 identify exactly one event",
      _ev12 == ["409634"], str(_ev12))

V.on_event(None, None, "evt409634")
V.mode_group.active = 1
V.on_mode(None, None, 1)
# The load-bearing fact: E_est is an EXACT decomposition of kine_charge, so
# adding a marked segment's share is arithmetic and not an estimate.
_pr12 = V.state["prep"]["showers"]["69032"]
check("the probe's per-segment E_est sums to the shower's kine_charge",
      abs(sum(m["E_est"] for m in _pr12["members"])
          - V.shower_energy(69032)) < 5e-3,
      "%.4f vs %.4f" % (sum(m["E_est"] for m in _pr12["members"]),
                        V.shower_energy(69032)))

V.state["sel_shower"] = 69032
V.fill_cand_table()
V.refresh_bulk_options()
V.bulk_shower.value = [o for o in V.bulk_shower.options
                       if V._excl_node(o) == 27015][0]
V.on_bulk(False)()
V.mark("in")()
_mk12 = V.marks_energy(69032)
check("merging shower 27015 in is worth its whole kine_charge",
      abs(_mk12["delta"] - V.shower_energy(27015)) < 5e-3
      and len(_mk12["rows"]) == 10 and not _mk12["unknown"],
      "%+.3f vs %.3f, %d rows" % (_mk12["delta"], V.shower_energy(27015),
                                  len(_mk12["rows"])))

V.state["gamma"][1] = 69032
V.state["gamma"][2] = 21002
V.vtx_mode_group.active = 0
V.refresh_kine()
check("the default is still the reconstruction's membership",
      V.emark_mode.value == V.EMARK_RECO
      and abs(V.gamma_energy(1) - 39.06) < 0.01,
      "%s, E1 %.2f" % (V.emark_mode.value, V.gamma_energy(1)))
check("  ... and the panel says what is not being counted, and what it costs",
      "not counted below" in V.kine_div.text
      and "144.1" in V.kine_div.text and "78.8" in V.kine_div.text,
      "-")
_m_off = V.pio_pairing()
V.emark_mode.value = V.EMARK_MARKS
V.refresh_kine()
check("switching membership on moves the energy, not the angle",
      abs(V.gamma_energy(1) - 144.11) < 0.02
      and abs(V.pio_pairing()["theta_axis_convention"]
              - _m_off["theta_axis_convention"]) < 1e-9,
      "E1 %.2f, theta %.3f" % (V.gamma_energy(1),
                               V.pio_pairing()["theta_axis_convention"]))
check("  ... and the pi0 mass follows: 41.0 -> 78.8 (axis), 83.4 -> 160.3 (vertex)",
      abs(_m_off["mass_axis_convention"] - 41.0) < 0.1
      and abs(V.pio_pairing()["mass_axis_convention"] - 78.8) < 0.1
      and abs(_m_off["mass_vertex_convention"] - 83.4) < 0.1
      and abs(V.pio_pairing()["mass_vertex_convention"] - 160.3) < 0.1,
      "%.1f -> %.1f, %.1f -> %.1f"
      % (_m_off["mass_axis_convention"],
         V.pio_pairing()["mass_axis_convention"],
         _m_off["mass_vertex_convention"],
         V.pio_pairing()["mass_vertex_convention"]))

# Double counting.  Round 10's whole-shower select can sweep up members --
# `show members too` is ON by default -- so a member marked IN must be worth 0.
_e_before = V.gamma_energy(1)
V.marks_for(69032)[69033] = "in"          # 69033 IS a member of 69032
check("a MEMBER marked IN adds nothing (it is already in kine_charge)",
      abs(V.gamma_energy(1) - _e_before) < 1e-9,
      "%.4f vs %.4f" % (V.gamma_energy(1), _e_before))
V.marks_for(69032)[69033] = "out"
check("  ... and marking that same member OUT takes its E_est off",
      abs(V.gamma_energy(1) - (_e_before - 9.56)) < 0.01,
      "%.2f, expected %.2f" % (V.gamma_energy(1), _e_before - 9.56))
V.marks_for(69032).pop(69033)
# A segment that is neither a member of 69032 nor already marked -- picked from
# the event rather than hardcoded, since every segment of 27015 is now marked IN.
_free = [sg.get("id") for sg in V.cur_segments()
         if sg.get("id") not in set(V.members_of(69032))
         and sg.get("id") not in V.marks_for(69032)
         and sg.get("id") is not None]
V.marks_for(69032)[_free[0]] = "out"      # marked OUT but never a member
check("  ... and a NON-member marked OUT takes nothing off",
      abs(V.gamma_energy(1) - _e_before) < 1e-9,
      "seg %s: %.4f vs %.4f" % (_free[0], V.gamma_energy(1), _e_before))
V.marks_for(69032).pop(_free[0])

# A segment no shower owns has no E_est, and no dQ estimate is invented for it.
_far = [sg.get("id") for sg in V.cur_segments()
        if sg.get("id") not in V._est_map() and sg.get("id") is not None
        and sg.get("id") not in V.marks_for(69032)]
if _far:
    V.marks_for(69032)[_far[0]] = "in"
    _u = V.marks_energy(69032)
    check("a segment owned by no shower is named, not counted, not estimated",
          _far[0] in _u["unknown"] and abs(V.gamma_energy(1) - _e_before) < 1e-9,
          "unknown=%s" % _u["unknown"])
    V.refresh_kine()
    check("  ... and the panel says so in red",
          "owned by no shower" in V.kine_div.text, "-")
    V.marks_for(69032).pop(_far[0])
else:
    check("evt409634 has no unowned segment to test with (skipped)", True,
          "all segments belong to a shower here")

# the record
V.on_save()
_r12 = json.load(open(V.label_path("evt409634")))
_g12 = _r12["pio"]["gammas"]["1"]
check("the record carries the membership switch, the delta and the arithmetic",
      _g12["energy_includes_marks"] is True
      and abs(_g12["energy_marks_delta"] - 105.05) < 0.01
      and len(_g12["energy_marks_detail"]) == 10
      and abs(_g12["energy_without_marks"] - 39.06) < 0.01
      and abs(_g12["energy"] - 144.11) < 0.02,
      "delta %+.2f, %d rows, without %.2f, with %.2f"
      % (_g12["energy_marks_delta"], len(_g12["energy_marks_detail"]),
         _g12["energy_without_marks"], _g12["energy"]))
check("  ... and each contribution names the shower it came from",
      all(r["owner"] == 27015 and r["kind"] == "in"
          for r in _g12["energy_marks_detail"]),
      str(sorted({r["owner"] for r in _g12["energy_marks_detail"]})))

# THE regression: six records saved before this control existed carry marks on a
# gamma's shower, and re-opening one must show the mass it was saved with.
_p12 = V.label_path("evt409634")
_o12 = json.load(open(_p12))
for _sl in ("1", "2"):
    _o12["pio"]["gammas"][_sl].pop("energy_includes_marks", None)
with open(_p12, "w") as _fh:
    json.dump(_o12, _fh)
V.on_event(None, None, "evt84229")
V.on_event(None, None, "evt409634")
check("a pre-round-12 record re-opens on the reco's membership",
      V.emark_mode.value == V.EMARK_RECO
      and abs(V.gamma_energy(1) - 39.06) < 0.01,
      "%s, E1 %.2f" % (V.emark_mode.value, V.gamma_energy(1)))
os.remove(_p12)

# cross-pair: a marked segment's E_est was converted with ITS OWN shower's pair
V.on_event(None, None, "evt166870")
V.mode_group.active = 1
check("evt166870 shower 85045 is track-flagged, 87058 is not",
      V.shower_is_em(85045) is False and V.shower_is_em(87058) is True)
V.marks_for(87058)[85045] = "in"
_plain = V.marks_energy(87058)["delta"]
_asem = V.marks_energy(87058, as_em=True)["delta"]
check("a marked TRACK-flagged segment is re-converted under the EM hypothesis",
      abs(_asem / _plain - (0.7 * 0.95) / (0.5 * 0.8)) < 1e-9,
      "%.3f -> %.3f, ratio %.4f" % (_plain, _asem, _asem / _plain))
V.marks_for(87058).pop(85045)

# ---------------------------------------------------------------------------
# round 13: the pi0 vertex point itself is in the record, and survives a re-open
# ---------------------------------------------------------------------------
V.on_event(None, None, "evt64591")
V.mode_group.active = 1
_n13 = list(V.shower_src.data["node"])
V.state["gamma"][1], V.state["gamma"][2] = _n13[0], _n13[1]
V.vtx_mode_group.active = 2
V.set_pio_vertex((-96.6, -27.6, 175.7))
V.on_save()
_r13 = json.load(open(V.label_path("evt64591")))
check("a MANUAL pi0 vertex is saved as a point, not just as a mode",
      _r13["pio"]["vertex_how"] == "manual"
      and [round(x, 1) for x in _r13["pio"]["vertex"]] == [-96.6, -27.6, 175.7],
      "%s %s" % (_r13["pio"]["vertex_how"], _r13["pio"]["vertex"]))

# THE bug: the boxes were left empty on a re-open, so the vertex lived only in
# state["vtx_manual"] -- and one keystroke in any box wiped it to null.
V.on_event(None, None, "evt84229")
V.on_event(None, None, "evt64591")
check("re-opening it fills the x/y/z boxes, not just the state",
      (V.man_x.value, V.man_y.value, V.man_z.value)
      == ("-96.6", "-27.6", "175.7")
      and V.vtx_mode_group.active == 2,
      "%s / %s / %s" % (V.man_x.value, V.man_y.value, V.man_z.value))
V.man_x.value = "-96.0"
V.on_manual(None, None, "-96.0")
check("  ... so nudging ONE coordinate keeps the vertex instead of wiping it",
      V.state["vtx_manual"] is not None
      and [round(x, 1) for x in V.state["vtx_manual"]] == [-96.0, -27.6, 175.7],
      str(V.state["vtx_manual"]))

# and the back-projected NCpi0 vertex, with the code's own working
V.vtx_mode_group.active = 1
V.on_vtx_mode(None, None, 1)
_bp = V.pio_pairing()
check("a BACK-PROJECTED vertex is saved as a point too, with its branch",
      _bp["vertex_how"] == "backproject"
      and (_bp["vertex"] is None or len(_bp["vertex"]) == 3)
      and _bp["backproject"] is not None
      and {"verdict", "branch", "gap", "theta", "mass", "angle1", "angle2",
           "dis1", "dis2", "len1", "len2"} <= set(_bp["backproject"]),
      "verdict=%s branch=%s" % (_bp["backproject"].get("verdict"),
                                _bp["backproject"].get("branch")))
os.remove(V.label_path("evt64591"))

# ---------------------------------------------------------------------------
# round 14: the back-projection uses the scanner's corrected start and axis
# ---------------------------------------------------------------------------
# The event the owner reported, found by fingerprint rather than by number: the
# three showers they corrected co-occur in exactly one manifest event.
_ev14 = [e for e, _row in MANIFEST_EVENTS if _has_showers(e, (14058, 14059, 50052))]
check("evt76346 is the only event holding showers 14058 + 14059 + 50052",
      _ev14 == ["76346"], str(_ev14))

# THE guarantee.  A gamma the scanner never touched must inject no ray, because
# that -- not "the arithmetic agrees" -- is what makes the mirror byte-identical.
# gamma_scan_ray resolves membership through the probe sidecar and gamma_ray
# through the dump join, and em_geom.join_completeness names evt347129 as a case
# where those differ.
V.on_event(None, None, "evt76346")
V.mode_group.active = 1
check("with nothing corrected, no gamma injects a ray",
      all(V.gamma_scan_ray(n, sl)[0] is None
          for sl in (1, 2) for n in list(V.shower_src.data["node"])),
      "%d showers" % len(V.shower_src.data["node"]))
_pairs = 0
for _e, _row in MANIFEST_EVENTS:
    V.on_event(None, None, _e)
    _shs = (V.state["data"] or {}).get("showers") or []
    _anc = V.G.pt((V.state["data"] or {}).get("main_vertex"))
    if _anc is None or len(_shs) < 2:
        continue
    for _a, _b in ((_shs[0], _shs[1]), (_shs[-1], _shs[-2])):
        _x = V.G.pi0_backproject(_a, _b, V.cur_segments(), _anc)
        _y = V.G.pi0_backproject(_a, _b, V.cur_segments(), _anc,
                                 ray1=None, ray2=None)
        if _x != _y:
            _pairs = -1
            break
        _pairs += 1
check("ray1=ray2=None is the pre-round-14 mirror, on every manifest event",
      _pairs > 150, "%d pairs compared" % _pairs)

# The owner's own corrections, replayed: shower 14059 moved 30.3 cm and flipped
# 155.5 deg, shower 14058 moved 55.6 cm.  Values read from their saved label.
V.on_event(None, None, "evt76346")
V.mode_group.active = 1
V.state["em_start"][14059] = (-109.93760681152344, -35.379791259765625,
                              345.8183898925781)
V.state["em_dir"][14059] = (-103.09359741210938, -1.9308925867080688,
                            371.24139404296875)
V.state["em_start"][14058] = (-92.0087661743164, -58.29239273071289,
                              338.35675048828125)
V.state["em_dir"][14058] = (-36.24102020263672, -54.461673736572266,
                            359.2472839355469)
V.state["_axis_cache"] = {}
V.state["gamma"][1], V.state["gamma"][2] = 14059, 14058
V.vtx_mode_group.active = 1

V.bp_geom.value = V.BPG_RECO
_v_reco, _how, _d_reco = V.pio_vertex()
check("today's back-projection DEGENERATES on this pair",
      _how == "backproject" and _d_reco["verdict"] == "degenerate"
      and _d_reco["geometry"] == "reco",
      "verdict=%s vertex=%s" % (_d_reco["verdict"],
                                [round(x, 1) for x in (_v_reco or [])]))
check("  ... because both reco rays start at the SAME point, the main vertex",
      _v_reco is not None
      and V.G.vmag(V.G.vsub(_v_reco,
                            V.G.pt(V.state["data"]["main_vertex"]))) < 1e-6,
      str([round(x, 2) for x in _v_reco]))

V.bp_geom.value = V.BPG_SCAN
_v_scan, _how2, _d_scan = V.pio_vertex()
check("the scanner's start and axis give a real vertex instead",
      _d_scan["verdict"] == "ok" and _d_scan["branch"] == "one_short"
      and _v_scan is not None
      and [round(x, 1) for x in _v_scan] == [-114.7, -58.8, 328.0],
      "verdict=%s branch=%s vertex=%s"
      % (_d_scan["verdict"], _d_scan["branch"],
         [round(x, 2) for x in _v_scan]))
check("  ... 60.0 cm from where the reco's own rays put it",
      abs(V.G.vmag(V.G.vsub(_v_scan, _v_reco)) - 59.98) < 0.05,
      "%.2f cm" % V.G.vmag(V.G.vsub(_v_scan, _v_reco)))
check("  ... and the closest-approach gap closes to 2.0 cm",
      _d_scan["gap"] is not None and abs(_d_scan["gap"] - 2.03) < 0.05,
      "%.2f cm" % _d_scan["gap"])
check("  ... the 0.4 cm stub is NOT re-rayed: the scanner stated its direction",
      _d_scan["short_rerayed"] is False and _d_scan["ray2_given"] is True)
check("  ... both rays name their provenance",
      "manual_axis" in _d_scan["ray1_source"]
      and "manual_axis" in _d_scan["ray2_source"],
      "%s | %s" % (_d_scan["ray1_source"], _d_scan["ray2_source"]))
check("  ... and the panel carries the OTHER reading to compare against",
      _d_scan["alt"] is not None and _d_scan["alt"]["geometry"] == "reco"
      and _d_scan["alt"]["verdict"] == "degenerate")

# A start correction alone (no aimed axis) re-evaluates dir15 at the new point,
# over PROBE membership -- shower_axis's own `python@start_override` branch.
V.state["em_dir"].pop(14059)
V.state["_axis_cache"] = {}
_r14, _s14 = V.gamma_scan_ray(14059, 1)
check("a moved start alone re-derives dir15 at the new point",
      _r14 is not None and "dir15@probe_members" in _s14, _s14)
check("  ... from the same point shower_start reports",
      V.G.vmag(V.G.vsub(_r14[0], V.shower_start(14059, 1))) < 1e-9)

# And when the scanner moves a start clear of the shower's own points, the 15 cm
# window is empty and there is no direction to be had.  That gamma injects no
# ray and the mirror builds its own -- said in the provenance rather than
# quietly substituting some other vector.  14058 is 6.9 cm long and its start
# was moved 55.6 cm, so this is the owner's own event, not a constructed case.
V.state["em_dir"].pop(14058)
V.state["_axis_cache"] = {}
_r14c, _s14c = V.gamma_scan_ray(14058, 2)
check("a start moved clear of the shower yields no ray, and says why",
      _r14c is None and "dir15 undefined" in _s14c, _s14c)

# An aimed axis alone keeps the MIRROR's own origin -- not the shower's `start`,
# which is a different point.
V.state["em_start"].pop(14058)
V.state["em_dir"].pop(14058, None)
V.state["em_dir"][14058] = (-36.24102020263672, -54.461673736572266,
                            359.2472839355469)
V.state["_axis_cache"] = {}
_r14b, _s14b = V.gamma_scan_ray(14058, 2)
_ray0 = V.G.gamma_ray(V.shower_by_node(14058), V.cur_segments(),
                      V.G.pt(V.state["data"]["main_vertex"]))[1]
check("an aimed axis alone keeps the mirror's own start point",
      _r14b is not None and _s14b.startswith("reco_ray_point")
      and V.G.vmag(V.G.vsub(_r14b[0], _ray0)) < 1e-9, _s14b)

# The record, and the legacy rule that protects what is already on disk.
V.state["em_start"][14058] = (-92.0087661743164, -58.29239273071289,
                              338.35675048828125)
V.state["em_dir"][14059] = (-103.09359741210938, -1.9308925867080688,
                            371.24139404296875)
V.state["_axis_cache"] = {}
V.on_save()
_r14rec = json.load(open(V.label_path("evt76346")))
_pio14 = _r14rec["pio"]
check("the record says which geometry built its vertex",
      _pio14["backproject_geometry"] == "handscan"
      and _pio14["backproject"]["geometry"] == "handscan",
      _pio14["backproject_geometry"])
check("  ... and carries the vertex it was NOT using, so both are recoverable",
      _pio14["backproject"]["alt"]["geometry"] == "reco"
      and _pio14["backproject"]["alt"]["verdict"] == "degenerate")

_p14 = V.label_path("evt76346")
_o14 = json.load(open(_p14))
_o14["pio"].pop("backproject_geometry")
_o14["pio"]["backproject"].pop("geometry", None)
with open(_p14, "w") as _fh:
    json.dump(_o14, _fh)
V.on_event(None, None, "evt84229")
V.on_event(None, None, "evt76346")
check("a pre-round-14 back-projected record re-opens on the reco's rays",
      V.bp_geom.value == V.BPG_RECO, V.bp_geom.value)
os.remove(_p14)

# A fresh event defaults the other way -- and cannot move anything, because a
# gamma nobody corrected injects no ray.
V.on_event(None, None, "evt84229")
check("a fresh event defaults to the scanner's geometry",
      V.bp_geom.value == V.BPG_SCAN, V.bp_geom.value)

print()
print("FAILURES: %d" % len(fails))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
