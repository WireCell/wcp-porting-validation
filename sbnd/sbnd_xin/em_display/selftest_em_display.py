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
check("12 rows carry an owner note",
      sum(1 for r in V.MANIFEST.values() if r.get("scan_note")) == 12,
      str(sum(1 for r in V.MANIFEST.values() if r.get("scan_note"))))

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

print()
print("FAILURES: %d" % len(fails))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
