#!/usr/bin/env python3
"""Review-only variant of em_display_viewer.py: the note box is readable.

Why this file exists.  The scan notes this campaign wrote are 2000-5000
characters (doc pr/116 Part 2); `note_in` in the production viewer is a
single-line `TextInput` 520 px wide, so about eighty characters of a note are
visible at a time and the rest can only be reached by arrowing along the line.
That is fine for typing a one-line note, which is what it was built for, and
useless for reading one back.

Why it is a LOADER and not a copy.  em_display_viewer.py is 273 kB and is
served live on 5018 (the owner's own scan) and 5019 (all 141 events); editing it
would change what those sessions render on their next tab, with no restart and
no warning.  CLAUDE.md M10 says fork by duplication rather than extract a shared
helper -- but a 273 kB duplicate would silently drift from the original the
first time anyone fixes the real one.  So this file duplicates *nothing*: it
reads the production source, applies three asserted textual substitutions, and
execs the result.  The production file stays byte-identical, the delta is the
twenty lines below, and any upstream fix reaches this viewer for free.

If a substitution target ever stops matching, this raises at startup rather than
serving a viewer that silently lacks the change.

    ./em_display/serve_em116_confirm.sh 5027 --scan-tag ... --manifest ... --prepdir ...
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "em_display_viewer.py")

with open(SRC) as fh:
    src = fh.read()

# (target, replacement, why)
PATCHES = [
    # TextAreaInput is not imported by the production file.
    ("from bokeh.models import (Button, CheckboxButtonGroup, CheckboxGroup,",
     "from bokeh.models import (Button, CheckboxButtonGroup, CheckboxGroup,\n"
     "                          TextAreaInput,",
     "import the multi-line input"),

    # The note box itself.  `max_length` matters: bokeh's TextAreaInput defaults
    # to 500 characters, which would TRUNCATE every note this campaign wrote --
    # a silent data loss on save, not a display problem.  20000 is ~4x the
    # longest note on disk (5165 chars, evt499577).
    ('note_in = TextInput(title="note (optional)", value="", width=520)',
     'note_in = TextAreaInput(title="note (optional)", value="", width=RW,\n'
     '                        rows=18, max_length=20000)',
     "single-line 520 px input -> 18-row full-width textarea"),

    # Give it its own full-width row instead of sharing one with the radio
    # group and the save button, which would otherwise be pinned to its top.
    ("    row(conf_group, note_in, save_btn),",
     "    row(conf_group, save_btn),\n    note_in,",
     "note gets its own row"),
]

for target, repl, why in PATCHES:
    n = src.count(target)
    if n != 1:
        raise SystemExit(
            "em116_confirm_viewer: expected exactly one occurrence of %r in "
            "%s, found %d.  The production viewer changed shape; fix this "
            "loader rather than serving a viewer that quietly lacks '%s'."
            % (target[:60] + "...", SRC, n, why))
    src = src.replace(target, repl, 1)

# `__file__` must point at the PRODUCTION file: the app derives HERE from it and
# resolves em3d/em_geom, the label root and the prep dir off that.
exec(compile(src, SRC, "exec"),
     {"__file__": SRC, "__name__": "__main__", "__builtins__": __builtins__})
